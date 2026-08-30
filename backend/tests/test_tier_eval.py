"""
The tier eval harness (`scripts/validate_agent.py`) -- tested headless.

**This suite asserts the instrument, not the reading.** `test_rca_ground_truth.py`
can call `validate_rca.run_pipeline()` and assert the real scorecard, because
that pipeline is deterministic and needs no API key. The agent loop is neither,
so the equivalent here is impossible and its absence is deliberate, not an
oversight. What is asserted instead:

1. the manifest is sound -- every tool and KPI a case names really exists;
2. the Tier-1/2 expectations are genuinely independent of the tools they check;
3. the scorer **fails a wrong run** -- an eval that always passes is worse than
   no eval, so this is the part that earns the rest.

Category 3 runs the *real* `loop.answer` against the *real* registry with a
`ScriptedAnthropic` model, so every tool result is genuinely computed from the
sample data; only the model's choices are scripted. `validate_agent.main()` --
a thin live driver -- is the one untested part, exactly as `validate_rca.main()`
is.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
for path in (str(BACKEND), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import pandas as pd                                             # noqa: E402

from app.agent.context import AgentContext                      # noqa: E402
from app.agent.loop import answer                               # noqa: E402
from app.agent.registry import TOOL_SPECS, _SPECS_BY_NAME       # noqa: E402
from app.llm.client import LLMClient                            # noqa: E402
from scripts import validate_agent as va                        # noqa: E402

from .base import EngineTestCase, assert_json_safe              # noqa: E402
from .fakes import ScriptedAnthropic, scripted_message, text_block, tool_use_block  # noqa: E402


def _case(case_id: str) -> va.Case:
    return next(c for c in va.CASES if c.id == case_id)


class HarnessTestCase(EngineTestCase):
    """One context and one raw frame, shared -- neither is mutated by scoring."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.raw = va.raw_frame()
        probe = AgentContext.build(cls.uid, cls.dataset)
        cls.available = frozenset(probe.api.available_keys(probe.df))

    def run_scripted(self, question, responses):
        """The real loop and the real registry; only the model is scripted.

        A **fresh context per run**, deliberately: `answer` defaults its turn
        budget to `ctx.budget.remaining_turns` and calls `budget.record` after
        every run, so a context shared across cases would silently starve the
        later ones. `validate_agent.main()` builds one per case for the same
        reason."""
        ctx = AgentContext.build(self.uid, self.dataset)
        client = LLMClient()
        client._client = ScriptedAnthropic(responses)
        return answer(question, ctx, llm=client)

    def score(self, case_id, responses, question=None):
        case = _case(case_id)
        ans = self.run_scripted(question or case.question, responses)
        obs = va.Observed.of(ans)
        return case, obs, va.score(case, obs, self.raw, self.available)

    @staticmethod
    def hard_failures(checks):
        return [c["check"] for c in checks if not c["pass"] and not c["soft"]]


# ---------------------------------------------------------------------------
# 1. the manifest
# ---------------------------------------------------------------------------
class TestTierEvalManifestIsSound(HarnessTestCase):
    """The instrument has to be sound before it can measure anything."""

    def test_there_are_seventeen_cases_spanning_every_tier_from_one_to_five(self):
        self.assertEqual(len(va.CASES), 17)
        self.assertEqual(sorted({c.tier for c in va.CASES}), [1, 2, 3, 4, 5])

    def test_every_case_id_is_unique(self):
        ids = [c.id for c in va.CASES]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_tool_named_by_a_case_is_a_registered_tool(self):
        named = set()
        for case in va.CASES:
            named |= case.required_all | case.forbidden
            for alternatives in case.required_any:
                named |= alternatives
        unknown = sorted(named - set(_SPECS_BY_NAME))
        self.assertEqual(unknown, [], f"cases name tools that do not exist: {unknown}")

    def test_every_kpi_named_by_a_case_exists_in_the_sample_dataset(self):
        named = set()
        for case in va.CASES:
            named |= case.expects_kpis
        self.assertEqual(sorted(named - self.available), [])

    def test_only_tier_one_and_two_cases_declare_an_expected_value(self):
        """Tiers 3-5 are scored on what was called, never on a number -- the
        board is explicit that they are not asserted on the answer."""
        for case in va.CASES:
            if case.expects_value is not None:
                self.assertIn(case.tier, (1, 2), f"{case.id} is tier {case.tier}")

    def test_turn_budgets_are_non_decreasing_by_tier(self):
        bars = [va.MAX_TURNS_BY_TIER[t] for t in sorted(va.MAX_TURNS_BY_TIER)]
        self.assertEqual(bars, sorted(bars))

    def test_the_tier_one_turn_budget_allows_one_call_plus_the_answer(self):
        """`loop.answer` counts the final end_turn as a turn, so a bar of 1
        would fail every possible run."""
        self.assertEqual(va.MAX_TURNS_BY_TIER[1], 2)

    def test_the_forbidden_set_for_the_definition_question_is_derived_from_the_registry(self):
        expected = {s.name for s in TOOL_SPECS if s.group != "orient"}
        self.assertEqual(set(_case("t1_definition_only").forbidden), expected)
        self.assertNotIn("get_kpi_definition", expected)

    def test_every_case_asserts_something_beyond_merely_terminating(self):
        for case in va.CASES:
            with self.subTest(case=case.id):
                self.assertTrue(
                    case.required_all or case.required_any or case.expects_value
                    or case.expects_kpis or case.expects_error_code,
                    f"{case.id} would pass on any run at all")


# ---------------------------------------------------------------------------
# 2. the scorer cannot see the prose
# ---------------------------------------------------------------------------
class TestTheScorerCannotSeeTheAnswerProse(HarnessTestCase):
    def test_observed_has_no_field_that_can_hold_the_answer_text(self):
        """The structural enforcement, in place of a comment asking nobody to
        write a wording assertion: there is nowhere to write one against."""
        fields = set(va.Observed.__dataclass_fields__)
        self.assertNotIn("answer", fields)
        self.assertNotIn("text", fields)
        self.assertIn("answer_chars", fields)

    def test_two_runs_with_identical_traces_and_opposite_prose_score_identically(self):
        def script(prose):
            return [
                scripted_message([tool_use_block(
                    "t1", "query_kpi",
                    {"kpi_keys": "revenue",
                     "time_filter": {"type": "years", "values": [2023, 2025]}})], "tool_use"),
                scripted_message([text_block(prose)], "end_turn"),
            ]

        _, _, right = self.score("t2_odd_years", script("Revenue in odd years was 31,631,471."))
        _, _, wrong = self.score("t2_odd_years", script("I have absolutely no idea."),
                                 question="What was revenue in all odd-numbered years, again?")
        self.assertEqual([c["pass"] for c in right], [c["pass"] for c in wrong])


# ---------------------------------------------------------------------------
# 3. the expectations are independent
# ---------------------------------------------------------------------------
class TestIndependentExpectationsAreNotComputedByTheToolUnderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = va.raw_frame()

    def test_the_raw_frame_is_the_csv_not_the_prepared_frame(self):
        """`metrics.prepare` writes `_year`/`_quarter`/`_month`/`_period`. Their
        absence is what proves this expectation shares no code with the tools."""
        for column in ("_year", "_quarter", "_month", "_period"):
            self.assertNotIn(column, self.raw.columns)

    def test_the_odd_year_expectation_sums_only_2023_and_2025(self):
        manual = 0.0
        for year in (2023, 2025):
            manual += float(self.raw.loc[self.raw.date.dt.year == year, "revenue"].sum())
        self.assertAlmostEqual(va._odd_year_revenue(self.raw), manual, places=2)

    def test_the_odd_year_expectation_differs_from_the_all_years_total(self):
        """A scorer that matched the whole-frame total by accident would pass
        the flagship case without the model having reasoned about 'odd' at all."""
        self.assertNotAlmostEqual(va._odd_year_revenue(self.raw),
                                  va._all_revenue(self.raw), places=2)

    def test_the_quarter_expectation_matches_an_independently_reassembled_monthly_sum(self):
        months = self.raw[self.raw.date.dt.year == 2026]
        manual = float(months.loc[months.date.dt.month.isin([4, 5, 6]), "revenue"].sum())
        self.assertAlmostEqual(va._q2_2026_revenue(self.raw), manual, places=2)

    def test_the_quarter_expectation_agrees_with_the_planted_ground_truth(self):
        """`ground_truth.json` records this same figure from the generator's own
        simulation -- two unrelated derivations of one number."""
        import json

        truth = json.loads((ROOT / "sample_data" / "ground_truth.json").read_text())
        self.assertAlmostEqual(va._q2_2026_revenue(self.raw),
                               truth["observed"]["current_value"], places=2)

    def test_the_ratio_expectation_is_a_sum_over_sum_not_a_mean_of_member_ratios(self):
        year = self.raw[self.raw.date.dt.year == 2026]
        mean_of_ratios = float((100.0 * year.fulfilled_orders / year.orders).mean())
        value = va._fulfilment_rate_2026(self.raw)
        self.assertAlmostEqual(
            value, 100.0 * float(year.fulfilled_orders.sum()) / float(year.orders.sum()),
            places=6)
        # The two must actually differ, or the check discriminates nothing.
        self.assertNotAlmostEqual(value, mean_of_ratios, places=2)

    def test_every_tier_one_and_two_expectation_is_finite_and_positive(self):
        for case in va.CASES:
            if case.expects_value is None:
                continue
            with self.subTest(case=case.id):
                got = case.expects_value(self.raw)
                for value in (got if isinstance(got, list) else [got]):
                    self.assertTrue(pd.notna(value) and value > 0)


# ---------------------------------------------------------------------------
# 4. the scorer detects a wrong run
# ---------------------------------------------------------------------------
class TestTheScorerDetectsAWrongRun(HarnessTestCase):
    """Every tool result below is really computed from the sample data; only
    the model's tool choices are scripted."""

    ODD = {"type": "years", "values": [2023, 2025]}
    EVEN = {"type": "years", "values": [2022, 2024]}

    def odd_year_script(self, time_filter):
        return [
            scripted_message([tool_use_block("t1", "query_kpi",
                                             {"kpi_keys": "revenue",
                                              "time_filter": time_filter})], "tool_use"),
            scripted_message([text_block("Here is the figure.")], "end_turn"),
        ]

    def test_a_correct_odd_year_run_passes_every_check(self):
        _, _, checks = self.score("t2_odd_years", self.odd_year_script(self.ODD))
        self.assertEqual(self.hard_failures(checks), [])

    def test_a_run_that_queried_the_even_years_fails_the_value_check(self):
        _, _, checks = self.score(
            "t2_odd_years", self.odd_year_script(self.EVEN),
            question="What was revenue in all odd-numbered years, even ones aside?")
        self.assertIn("retrieved_the_right_number", self.hard_failures(checks))

    def test_a_run_that_never_called_a_required_tool_fails_the_tool_check(self):
        _, _, checks = self.score("t2_odd_years", [
            scripted_message([tool_use_block("t1", "list_kpis", {})], "tool_use"),
            scripted_message([text_block("I looked at the catalogue only.")], "end_turn"),
        ], question="Revenue in the odd-numbered years, without querying?")
        failures = self.hard_failures(checks)
        self.assertIn("required_tools", failures)

    def test_a_definition_question_that_touched_data_fails_the_forbidden_tool_check(self):
        _, _, checks = self.score("t1_definition_only", [
            scripted_message([tool_use_block("t1", "get_kpi_definition",
                                             {"key": "fulfillment_rate"})], "tool_use"),
            scripted_message([tool_use_block("t2", "query_kpi",
                                             {"kpi_keys": "fulfillment_rate",
                                              "time_filter": {"type": "year", "year": 2025}})],
                             "tool_use"),
            scripted_message([text_block("It means this, and here is its value.")], "end_turn"),
        ])
        self.assertIn("forbidden_tools", self.hard_failures(checks))

    def test_a_definition_question_answered_from_the_catalogue_alone_passes(self):
        _, _, checks = self.score("t1_definition_only", [
            scripted_message([tool_use_block("t1", "get_kpi_definition",
                                             {"key": "fulfillment_rate"})], "tool_use"),
            scripted_message([text_block("It is fulfilled orders over orders.")], "end_turn"),
        ], question="What does the fulfilment rate mean here, in plain terms?")
        self.assertEqual(self.hard_failures(checks), [])

    def test_a_run_over_its_tier_turn_budget_fails_the_budget_check(self):
        script = []
        for i in range(1, 5):
            script.append(scripted_message([tool_use_block(
                f"t{i}", "query_kpi",
                {"kpi_keys": "revenue", "time_filter": self.ODD})], "tool_use"))
        script.append(scripted_message([text_block("Took my time.")], "end_turn"))
        _, obs, checks = self.score("t2_odd_years", script,
                                    question="Revenue in odd years, taking the long way?")
        self.assertGreater(obs.turns, va.MAX_TURNS_BY_TIER[2])
        self.assertIn("turn_budget", self.hard_failures(checks))

    def test_a_run_that_satisfied_one_alternative_of_a_required_any_set_passes(self):
        _, _, checks = self.score("t3_normal_range", [
            scripted_message([tool_use_block("t1", "get_normal_range",
                                             {"kpi_key": "fulfillment_rate"})], "tool_use"),
            scripted_message([text_block("That is inside the usual band.")], "end_turn"),
        ])
        self.assertNotIn("required_any_1", self.hard_failures(checks))

    def test_a_hallucinated_kpi_key_from_an_errored_call_is_not_counted_as_used(self):
        _, obs, _ = self.score("t2_odd_years", [
            scripted_message([tool_use_block("t1", "query_kpi",
                                             {"kpi_keys": "imaginary_kpi",
                                              "time_filter": self.ODD})], "tool_use"),
            scripted_message([tool_use_block("t2", "query_kpi",
                                             {"kpi_keys": "revenue",
                                              "time_filter": self.ODD})], "tool_use"),
            scripted_message([text_block("Recovered.")], "end_turn"),
        ], question="Revenue in the odd years, after a false start?")
        self.assertNotIn("imaginary_kpi", obs.kpis_used_ok)
        self.assertIn("revenue", obs.kpis_used_ok)

    def test_the_harness_and_the_loop_agree_about_which_kpis_were_used(self):
        """`kpis_used_ok` is re-derived here rather than imported, so a
        regression in `loop._derive_kpis_used` surfaces as a disagreement
        instead of being inherited silently."""
        _, obs, checks = self.score("t2_odd_years", self.odd_year_script(self.ODD))
        self.assertEqual(set(obs.kpis_used), set(obs.kpis_used_ok))
        self.assertNotIn("loop_derivation_agrees", self.hard_failures(checks))

    def test_an_out_of_extent_period_scores_as_an_empty_period_error_not_a_wrong_number(self):
        _, obs, checks = self.score("t1_out_of_extent", [
            scripted_message([tool_use_block(
                "t1", "query_kpi",
                {"kpi_keys": "revenue",
                 "time_filter": {"type": "quarter", "year": 2026, "quarter": 4}})], "tool_use"),
            scripted_message([text_block("The data does not reach Q4 2026.")], "end_turn"),
        ])
        self.assertIn("empty_period", obs.error_codes)
        self.assertEqual(self.hard_failures(checks), [])

    def test_inventing_a_figure_for_a_period_the_data_lacks_still_fails(self):
        """Answering an out-of-extent question without the tool ever reporting
        the gap is the failure this case exists to catch."""
        _, _, checks = self.score("t1_out_of_extent", [
            scripted_message([tool_use_block("t1", "query_kpi",
                                             {"kpi_keys": "revenue",
                                              "time_filter": {"type": "year", "year": 2026}})],
                             "tool_use"),
            scripted_message([text_block("Revenue in Q4 2026 was 3.2m.")], "end_turn"),
        ], question="What was total revenue in Q4 2026, approximately?")
        self.assertIn("reported_the_missing_period", self.hard_failures(checks))


# ---------------------------------------------------------------------------
# 5. the tool-usage report
# ---------------------------------------------------------------------------
class TestToolUsageReport(unittest.TestCase):
    def setUp(self):
        self.runs = [
            {"tools": ["query_kpi", "list_kpis"], "tools_ok": ["query_kpi", "list_kpis"]},
            {"tools": ["query_kpi", "scan_kpis"], "tools_ok": ["scan_kpis"]},
        ]

    def test_the_report_accounts_for_all_fifty_six_tools_across_the_six_groups(self):
        cover = va.coverage(self.runs)
        self.assertEqual(cover["tools_total"], len(TOOL_SPECS))
        self.assertEqual(cover["tools_fired"] + cover["tools_never_fired"], len(TOOL_SPECS))
        self.assertEqual(sum(g["total"] for g in cover["by_group"].values()), len(TOOL_SPECS))

    def test_the_group_totals_match_the_registry(self):
        cover = va.coverage(self.runs)
        for group, block in cover["by_group"].items():
            expected = sum(1 for s in TOOL_SPECS if s.group == group)
            self.assertEqual(block["total"], expected, group)
            self.assertEqual(block["fired"] + len(block["never_names"]), expected)

    def test_a_tool_that_never_fires_is_reported_but_does_not_fail_the_run(self):
        """Seventeen questions cannot exercise 56 tools. `never_fired` is the
        evidence the X5 and progressive-disclosure decisions rest on, and
        gating on it would only invite gaming."""
        cover = va.coverage(self.runs)
        self.assertIn("test_confounders", cover["never_fired"])
        self.assertNotIn("pass", cover)

    def test_an_errored_call_still_counts_as_the_tool_having_fired(self):
        cover = va.coverage(self.runs)
        self.assertEqual(cover["tool_calls"]["query_kpi"], 2)
        self.assertEqual(cover["tool_errors"]["query_kpi"], 1)

    def test_the_report_is_json_safe(self):
        assert_json_safe(va.coverage(self.runs))


# ---------------------------------------------------------------------------
# 6. the live driver refuses to fake a run
# ---------------------------------------------------------------------------
class TestValidateAgentRefusesToFakeALiveRun(unittest.TestCase):
    def test_main_exits_without_writing_a_report_when_no_api_key_is_configured(self):
        """The suite runs with ANTHROPIC_API_KEY="" -- a scorecard that looked
        live but was not would be worse than no scorecard."""
        import os

        before = os.environ.get("ANTHROPIC_API_KEY")
        os.environ["ANTHROPIC_API_KEY"] = ""
        try:
            existed = va.REPORT_PATH.exists()
            stamp = va.REPORT_PATH.stat().st_mtime if existed else None
            self.assertEqual(va.main(), 2)
            self.assertEqual(va.REPORT_PATH.exists(), existed)
            if existed:
                self.assertEqual(va.REPORT_PATH.stat().st_mtime, stamp)
        finally:
            if before is not None:
                os.environ["ANTHROPIC_API_KEY"] = before


if __name__ == "__main__":
    unittest.main()
