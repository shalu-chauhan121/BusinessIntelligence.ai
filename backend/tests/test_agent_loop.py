"""
`agent.loop.answer` -- the orchestrator composing A1's tool-use transport
with A2's tool registry, over one A4 `AgentContext`, with A6's seeded prompt.

Nothing here re-implements turn iteration or `stop_reason` handling; a
scripted `LLMClient` (`tests/fakes.py`) drives every scenario so this stays
headless. Live behaviour against the real API is a separate, manual smoke
test (see the batch plan), not part of this suite.
"""
from __future__ import annotations

import unittest

from app.agent.context import AgentContext
from app.agent.loop import AgentAnswer, answer
from app.llm.client import LLMClient

from .base import EngineTestCase
from .fakes import ScriptedAnthropic, scripted_message, text_block, tool_use_block


def _client_with(responses) -> LLMClient:
    client = LLMClient()
    client._client = ScriptedAnthropic(responses)
    return client


class LoopTestCase(EngineTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.ctx = AgentContext.build(cls.uid, cls.dataset)
        cls.kpi = cls.ctx.api.list_kpi_keys()[0]
        cls.tf = {"type": "year", "year": 2024}


class TestATier1QuestionTerminatesInOneToolCall(LoopTestCase):
    def test_a_single_query_kpi_call_then_end_turn_makes_no_needless_extra_turn(self):
        client = _client_with([
            scripted_message([tool_use_block("t1", "query_kpi",
                                            {"kpi_keys": self.kpi, "time_filter": self.tf})], "tool_use"),
            scripted_message([text_block("Here is the figure.")], "end_turn"),
        ])
        result = answer("What was it in 2024?", self.ctx, llm=client)
        self.assertIsInstance(result, AgentAnswer)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.engine["turns"], 2)
        self.assertEqual(len(result.evidence), 1)
        self.assertEqual(result.evidence[0]["tool"], "query_kpi")


class TestAMultiFactorQuestionMakesSeveralCallsAndConverges(LoopTestCase):
    def test_several_tool_calls_across_turns_all_appear_in_order_in_the_trace(self):
        client = _client_with([
            scripted_message([tool_use_block("t1", "scan_kpis", {"time_filter": self.tf})], "tool_use"),
            scripted_message([tool_use_block("t2", "rank_drivers",
                                            {"kpi_key": self.kpi, "period_a": self.tf,
                                             "period_b": {"type": "year", "year": 2025}})], "tool_use"),
            scripted_message([tool_use_block("t3", "check_temporal_precedence",
                                            {"kpi_key": self.kpi, "cause_kpi": self.kpi})], "tool_use"),
            scripted_message([text_block("Several factors are involved.")], "end_turn"),
        ])
        result = answer("What factors are affecting profit?", self.ctx, llm=client)
        self.assertEqual(result.status, "ok")
        self.assertEqual([e["tool"] for e in result.evidence],
                         ["scan_kpis", "rank_drivers", "check_temporal_precedence"])
        self.assertEqual([e["step"] for e in result.evidence], [1, 2, 3])


class TestEveryTraceEntryNamesARealRegisteredTool(LoopTestCase):
    def test_every_tool_name_in_the_trace_is_in_the_registry(self):
        client = _client_with([
            scripted_message([tool_use_block("t1", "query_kpi",
                                            {"kpi_keys": self.kpi, "time_filter": self.tf})], "tool_use"),
            scripted_message([text_block("done")], "end_turn"),
        ])
        result = answer("q", self.ctx, llm=client)
        registered = set(self.ctx.registry.tool_names())
        for entry in result.evidence:
            self.assertIn(entry["tool"], registered)


class TestAMidLoopToolFailureStillProducesAnAnswer(LoopTestCase):
    def test_a_hallucinated_kpi_key_recorded_as_an_error_does_not_end_the_run(self):
        client = _client_with([
            scripted_message([tool_use_block("t1", "query_kpi",
                                            {"kpi_keys": "not_a_real_kpi", "time_filter": self.tf})], "tool_use"),
            scripted_message([tool_use_block("t2", "query_kpi",
                                            {"kpi_keys": self.kpi, "time_filter": self.tf})], "tool_use"),
            scripted_message([text_block("Recovered after the correction.")], "end_turn"),
        ])
        result = answer("q", self.ctx, llm=client)
        self.assertEqual(result.status, "ok")
        self.assertTrue(result.evidence[0]["is_error"])
        self.assertFalse(result.evidence[1]["is_error"])
        self.assertEqual(result.answer, "Recovered after the correction.")


class TestKpisUsedIsDerivedFromTheTraceNeverFromProse(LoopTestCase):
    def test_kpis_used_reflects_actual_tool_arguments_not_the_final_answer_text(self):
        client = _client_with([
            scripted_message([tool_use_block("t1", "query_kpi",
                                            {"kpi_keys": self.kpi, "time_filter": self.tf})], "tool_use"),
            # The model's prose names a KPI it never actually queried --
            # kpis_used must not pick that up.
            scripted_message([text_block("Also relevant: totally_unqueried_kpi_xyz.")], "end_turn"),
        ])
        result = answer("q", self.ctx, llm=client)
        self.assertEqual(result.kpis_used, [self.kpi])
        self.assertNotIn("totally_unqueried_kpi_xyz", result.kpis_used)

    def test_periods_used_formats_a_quarter_and_a_year_filter(self):
        client = _client_with([
            scripted_message([tool_use_block("t1", "compare_periods",
                                            {"kpi_key": self.kpi,
                                             "period_a": {"type": "quarter", "year": 2024, "quarter": 2},
                                             "period_b": {"type": "year", "year": 2025}})], "tool_use"),
            scripted_message([text_block("done")], "end_turn"),
        ])
        result = answer("q", self.ctx, llm=client)
        self.assertIn("2024-Q2", result.periods_used)
        self.assertIn("2025", result.periods_used)


class TestNoUsableLlmReturnsATypedStatusAndTouchesNoDataset(LoopTestCase):
    def test_a_disabled_client_returns_llm_required_without_calling_any_tool(self):
        client = LLMClient()  # constructed with no API key -> enabled is False
        self.assertFalse(client.enabled)
        result = answer("q", self.ctx, llm=client)
        self.assertEqual(result.status, "llm_required")
        self.assertEqual(result.answer, "")
        self.assertEqual(result.evidence, [])
        self.assertEqual(self.ctx.budget.turns_used, 0)


class TestNonConvergenceStillReturnsAPartialAnswer(LoopTestCase):
    def test_max_turns_exhaustion_returns_the_trace_accumulated_so_far(self):
        client = _client_with([
            scripted_message([tool_use_block(f"t{i}", "query_kpi",
                                            {"kpi_keys": self.kpi, "time_filter": self.tf})], "tool_use")
            for i in range(5)
        ])
        result = answer("q", self.ctx, llm=client, max_turns=2)
        self.assertEqual(result.status, "max_turns_exhausted")
        self.assertEqual(len(result.evidence), 2)

    def test_budget_is_recorded_after_the_run(self):
        client = _client_with([
            scripted_message([tool_use_block("t1", "query_kpi",
                                            {"kpi_keys": self.kpi, "time_filter": self.tf})], "tool_use"),
            scripted_message([text_block("done")], "end_turn"),
        ])
        ctx = AgentContext.build(self.uid, self.dataset)
        self.assertEqual(ctx.budget.turns_used, 0)
        answer("q", ctx, llm=client)
        self.assertEqual(ctx.budget.turns_used, 2)


if __name__ == "__main__":
    unittest.main()
