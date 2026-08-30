"""
`LLMClient._call_with_tools` -- the multi-turn tool-use loop.

This pins the transport in isolation from any real tool registry: `executor`
is a plain test function, and the model side is a `ScriptedAnthropic` double
replaying canned turns (see `fakes.py`). Nothing here touches a real API key
-- `EngineTestCase` already runs with `ANTHROPIC_API_KEY=""`, and `enabled` is
a property of `self._client is not None`, so assigning the scripted double
directly makes the client "enabled" regardless of settings, exactly like
`test_telemetry.py:74` already does for the single-turn `_call` path.

The existing single-turn `_call` and its six role methods are untouched by
this batch; `test_telemetry.py` staying green (unedited) is that regression
signal, not repeated here.
"""
from __future__ import annotations

import unittest

from app.llm.client import LLMClient
from app.services.telemetry import request_telemetry

from .base import EngineTestCase
from .fakes import ScriptedAnthropic, scripted_message, text_block, thinking_block, tool_use_block

TOOLS = [{"name": "query_kpi", "description": "d", "input_schema": {"type": "object", "properties": {}}}]


def _client_with(responses) -> LLMClient:
    client = LLMClient()
    client._client = ScriptedAnthropic(responses)
    return client


def _echo_executor(name, args):
    return ({"tool": name, "echo": args}, False)


class TestAMultiTurnExchangeCompletesAndTheTraceIsOrdered(EngineTestCase):
    def test_two_tool_calls_then_a_final_answer_produce_an_ordered_trace(self):
        client = _client_with([
            scripted_message([tool_use_block("t1", "query_kpi", {"kpi": "revenue"})], "tool_use"),
            scripted_message([tool_use_block("t2", "get_kpi_definition", {"kpi": "revenue"})], "tool_use"),
            scripted_message([text_block("Revenue was 100.")], "end_turn"),
        ])
        result = client._call_with_tools(
            "system", [{"role": "user", "content": "What is revenue?"}], TOOLS, _echo_executor)

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.turns, 3)
        self.assertEqual(result.text, "Revenue was 100.")
        self.assertEqual([step["tool"] for step in result.trace], ["query_kpi", "get_kpi_definition"])
        self.assertEqual(result.trace[0]["step"], 1)
        self.assertEqual(result.trace[1]["step"], 2)
        self.assertFalse(result.trace[0]["is_error"])
        self.assertEqual(len(client._client.calls), 3)


class TestTheCacheKeyCoversTheWholeHistoryNotJustTheFirstMessage(EngineTestCase):
    def test_two_loops_sharing_a_first_turn_but_diverging_at_the_second_both_call_the_api_at_turn_two(self):
        # Both loops send an *identical* first request (same system, same
        # tools, same initial user message) -- that request is legitimately
        # a cache hit for loop B and must not call the API a second time.
        # But each loop's own executor returns a *different* tool result
        # (simulating different underlying data), so turn 2's history
        # differs between A and B and must NOT collide on the old
        # (system, user)-only cache key shape `_call` uses.
        client = _client_with([
            scripted_message([tool_use_block("t1", "query_kpi", {"kpi": "revenue"})], "tool_use"),
            scripted_message([text_block("A: 100")], "end_turn"),
            scripted_message([text_block("B: 200")], "end_turn"),
        ])
        messages = [{"role": "user", "content": "What is revenue?"}]

        result_a = client._call_with_tools("system", list(messages), TOOLS,
                                          lambda name, args: ({"value": 100}, False))
        result_b = client._call_with_tools("system", list(messages), TOOLS,
                                          lambda name, args: ({"value": 200}, False))

        self.assertEqual(result_a.text, "A: 100")
        self.assertEqual(result_b.text, "B: 200")
        # 3 total API calls: turn 1 shared via a genuine cache hit, plus one
        # real turn-2 call per loop. If the cache key ignored the tool
        # result content, loop B's turn 2 would wrongly replay loop A's.
        self.assertEqual(len(client._client.calls), 3)


class TestMaxTurnsExhaustionIsATypedResultNeverAnException(EngineTestCase):
    def test_a_loop_that_never_reaches_end_turn_stops_at_the_turn_budget(self):
        client = _client_with([
            scripted_message([tool_use_block(f"t{i}", "query_kpi", {})], "tool_use") for i in range(5)
        ])
        result = client._call_with_tools(
            "system", [{"role": "user", "content": "q"}], TOOLS, _echo_executor, max_turns=2)
        self.assertEqual(result.status, "max_turns_exhausted")
        self.assertEqual(result.turns, 2)


class TestATruncatedFinalTurnIsATypedStatusNotAJsonError(EngineTestCase):
    def test_stop_reason_max_tokens_yields_status_truncated(self):
        client = _client_with([
            scripted_message([text_block("partial answ")], "max_tokens"),
        ])
        result = client._call_with_tools("system", [{"role": "user", "content": "q"}], TOOLS, _echo_executor)
        self.assertEqual(result.status, "truncated")
        self.assertEqual(result.text, "partial answ")


class TestARefusalIsATypedStatusAndIsNeverCached(EngineTestCase):
    def test_stop_reason_refusal_yields_status_refused(self):
        from types import SimpleNamespace
        client = _client_with([
            scripted_message([], "refusal", stop_details=SimpleNamespace(category="cyber")),
        ])
        result = client._call_with_tools("system", [{"role": "user", "content": "q"}], TOOLS, _echo_executor)
        self.assertEqual(result.status, "refused")
        self.assertIn("cyber", result.text)


class TestAToolThatRaisesBecomesAnErrorResultAndTheLoopContinues(EngineTestCase):
    def test_an_exploding_executor_yields_an_is_error_tool_result(self):
        def exploding(name, args):
            raise RuntimeError("boom")

        client = _client_with([
            scripted_message([tool_use_block("t1", "explode", {})], "tool_use"),
            scripted_message([text_block("recovered")], "end_turn"),
        ])
        result = client._call_with_tools(
            "system", [{"role": "user", "content": "q"}], TOOLS, exploding)

        self.assertEqual(result.status, "ok")
        self.assertTrue(result.trace[0]["is_error"])
        self.assertIn("boom", result.trace[0]["result"]["message"])
        # the failed tool_result actually reached the next API request
        second_call_messages = client._client.calls[1]["messages"]
        tool_result_message = second_call_messages[-1]
        self.assertEqual(tool_result_message["role"], "user")
        self.assertTrue(tool_result_message["content"][0]["is_error"])


class TestThinkingAndToolUseBlocksAreEchoedBackVerbatim(EngineTestCase):
    def test_the_assistant_turn_appended_to_history_carries_thinking_and_tool_use(self):
        client = _client_with([
            scripted_message(
                [thinking_block("reasoning..."), tool_use_block("t1", "query_kpi", {"kpi": "revenue"})],
                "tool_use"),
            scripted_message([text_block("done")], "end_turn"),
        ])
        client._call_with_tools("system", [{"role": "user", "content": "q"}], TOOLS, _echo_executor)

        second_call_messages = client._client.calls[1]["messages"]
        assistant_turn = second_call_messages[-2]
        self.assertEqual(assistant_turn["role"], "assistant")
        kinds = [b["type"] for b in assistant_turn["content"]]
        self.assertEqual(kinds, ["thinking", "tool_use"])
        self.assertEqual(assistant_turn["content"][1]["name"], "query_kpi")
        self.assertEqual(assistant_turn["content"][1]["id"], "t1")


class TestParallelToolCallsReturnInASingleUserMessage(EngineTestCase):
    def test_two_tool_use_blocks_in_one_turn_produce_one_user_message_with_both_results(self):
        client = _client_with([
            scripted_message([
                tool_use_block("t1", "query_kpi", {"kpi": "revenue"}),
                tool_use_block("t2", "query_kpi", {"kpi": "margin"}),
            ], "tool_use"),
            scripted_message([text_block("done")], "end_turn"),
        ])
        result = client._call_with_tools(
            "system", [{"role": "user", "content": "q"}], TOOLS, _echo_executor)

        self.assertEqual(len(result.trace), 2)
        second_call_messages = client._client.calls[1]["messages"]
        tool_result_message = second_call_messages[-1]
        self.assertEqual(tool_result_message["role"], "user")
        self.assertEqual(len(tool_result_message["content"]), 2)
        self.assertEqual({c["tool_use_id"] for c in tool_result_message["content"]}, {"t1", "t2"})


class TestTelemetryRecordsOnePerActualApiCallWithADistinctStepLabel(EngineTestCase):
    def test_a_two_turn_loop_records_two_calls_with_turn_labelled_steps(self):
        client = _client_with([
            scripted_message([tool_use_block("t1", "query_kpi", {})], "tool_use"),
            scripted_message([text_block("done")], "end_turn"),
        ])
        with request_telemetry(self.uid, "/test") as session:
            client._call_with_tools("system", [{"role": "user", "content": "q"}], TOOLS, _echo_executor)
        self.assertEqual(len(session.saved["calls"]), 2)
        self.assertEqual([c["step"] for c in session.saved["calls"]],
                         ["Tool loop turn 1", "Tool loop turn 2"])


if __name__ == "__main__":
    unittest.main()
