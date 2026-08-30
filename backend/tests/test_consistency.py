"""
`find_counterexamples` / `test_consistency_across_dimension` /
`test_holdout_segments` -- `agent/consistency.py`.

The property that matters most for `find_counterexamples` is polarity, and it
is not a corner case: `analysis.counterexamples` is hardcoded to find members
whose KPI *fell*, so on a lower-is-better KPI (`readmission_rate`,
`return_rate`, ...) it returns the wrong members entirely --
`TestCounterexamplesArePolarityAware` pins the measured inversion on the
hospital fixture, where the legacy function's answer and this tool's answer
are disjoint sets.

For `test_consistency_across_dimension`, what matters is that a member-level
sign-agreement rate and a pooled Pearson r are genuinely different
statistics, not two views of the same number: on the retail fixture's own
`region` / `revenue ~ fulfillment_rate` pair the rate is 1.0 (every region
moved the same way on both KPIs) while the pooled r is -0.953 (noise from a
one-point cause spread against a thirty-point KPI spread). Both are reported;
`TestRateAndPooledRCanDisagree` is the test that would fail if a future
"simplification" collapsed them into one number.

For `test_holdout_segments`, what matters is that the headline statistic is
the scale-free difference-in-differences, not the level gap
`compare_cohorts` reports by default -- measured on the retail product
split, the level gap is a six-figure group-size artefact while the real
answer is -25.30 percentage points.
"""
from __future__ import annotations

import unittest

import pandas as pd

from app.agent.consistency import ConsistencyEngine
from app.agent.contract_api import ContractAPI
from app.agent.errors import InvalidArgumentError
from app.engines import metrics
from app.engines.analysis import counterexamples, member_change_table

from .base import EngineTestCase, FIXTURES, assert_json_safe, assert_no_prose_leak, contracted

# `mechanism_check`'s prose field (`conclusion`) joins the usual banned set,
# same rationale as `test_temporal.py`.
BANNED = {"interpretation", "detail", "narrative", "note", "conclusion"}


def _assert_no_banned_keys(payload, path: str = "$") -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            assert key not in BANNED, f"{path}.{key} is a banned prose field"
            _assert_no_banned_keys(value, f"{path}.{key}")
    elif isinstance(payload, list):
        for i, item in enumerate(payload):
            _assert_no_banned_keys(item, f"{path}[{i}]")


def _member_frame(members, kpi_values_a, kpi_values_b, cause_values_a, cause_values_b,
                  kpi_column="revenue", cause_column="units_sold", dimension="region"):
    """A minimal contract-less two-period frame with one row per member per
    period, so a dimension-level consistency/holdout call has an exact,
    hand-computable answer."""
    rows = []
    for m, kv, cv in zip(members, kpi_values_a, cause_values_a):
        rows.append({"date": "2026-05-15", dimension: m, kpi_column: kv, cause_column: cv})
    for m, kv, cv in zip(members, kpi_values_b, cause_values_b):
        rows.append({"date": "2026-02-15", dimension: m, kpi_column: kv, cause_column: cv})
    raw = pd.DataFrame(rows)
    schema = metrics.detect_schema(raw)
    df = metrics.prepare(raw, schema)
    return df, schema


class ConsistencyTestCase(EngineTestCase):
    """The shared retail fixture, with a compiled contract."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.rdf, cls.rschema = contracted("business_metrics_sample.csv")
        cls.api = ContractAPI(cls.rschema)
        cls.ce = ConsistencyEngine(cls.api)
        cls.period_a = {"type": "quarter", "year": 2026, "quarter": 2}
        cls.period_b = {"type": "quarter", "year": 2026, "quarter": 1}


# ---------------------------------------------------------------------------
# find_counterexamples() -- polarity
# ---------------------------------------------------------------------------
class TestCounterexamplesArePolarityAware(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.schema = contracted("hospital_kpi_smoke_sample.csv")
        cls.api = ContractAPI(cls.schema)
        cls.ce = ConsistencyEngine(cls.api)
        cls.period_a = {"type": "quarter", "year": 2026, "quarter": 2}
        cls.period_b = {"type": "quarter", "year": 2026, "quarter": 1}

    def test_a_lower_is_better_kpi_finds_what_the_legacy_helper_misses(self):
        """`readmission_rate` is worse when it rises. Every department in
        this fixture got worse; the legacy function, hardcoded to look for a
        KPI *fall*, finds none of them."""
        self.assertFalse(self.api.polarity("readmission_rate"))

        result = self.ce.find_counterexamples(
            self.df, "readmission_rate", "avg_length_of_stay", "department",
            self.period_a, self.period_b, cause_direction="down")
        self.assertEqual(result.status, "ok")
        self.assertGreater(result.n_counterexamples, 0)
        found = {c.member for c in result.counterexamples}

        cur = self.df[(self.df._year == 2026) & (self.df._quarter == 2)]
        base = self.df[(self.df._year == 2026) & (self.df._quarter == 1)]
        table = member_change_table(cur, base, "department",
                                    ["readmission_rate", "avg_length_of_stay"],
                                    self.schema.contract_resolver)
        legacy = {row["member"] for row in counterexamples(
            table, "readmission_rate", "avg_length_of_stay",
            kpi_drop_pct=-5.0, cause_move_pct=5.0, cause_direction="down")}

        self.assertTrue(found, "the fixed tool must find at least one counterexample")
        self.assertEqual(legacy, set(), "the legacy helper, unfixed, finds none on this fixture")

    def test_the_denominator_is_unfavourable_members_not_every_member(self):
        result = self.ce.find_counterexamples(
            self.df, "readmission_rate", "avg_length_of_stay", "department",
            self.period_a, self.period_b, cause_direction="down")
        self.assertEqual(result.n_kpi_unfavourable, result.n_counterexamples)
        self.assertAlmostEqual(result.counterexample_rate,
                               result.n_counterexamples / result.n_kpi_unfavourable, places=6)
        # Every department happens to be unfavourable on this fixture, so
        # the denominator and the member count coincide here -- the real
        # property is the arithmetic relationship, checked directly, not
        # that the two counts must differ.
        self.assertLessEqual(result.n_kpi_unfavourable, result.n_members)

    def test_a_favourable_member_is_excluded_from_the_denominator(self):
        """A synthetic case, since no dimension on either sample fixture
        happens to have a member that *improved*: two members fall (bad, for
        a higher-is-better KPI), one rises (good) -- the denominator must
        count only the two that got worse."""
        df, schema = _member_frame(
            ["A", "B", "C"], kpi_values_a=[70, 80, 130], kpi_values_b=[100, 100, 100],
            cause_values_a=[50, 50, 50], cause_values_b=[50, 50, 50])
        result = ConsistencyEngine(ContractAPI(schema)).find_counterexamples(
            df, "revenue", "units_sold", "region",
            {"type": "quarter", "year": 2026, "quarter": 2},
            {"type": "quarter", "year": 2026, "quarter": 1}, cause_direction="any")
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.n_kpi_unfavourable, 2)
        self.assertLess(result.n_kpi_unfavourable, result.n_members)


class TestCounterexamplesOnHigherIsBetter(ConsistencyTestCase):
    def test_a_higher_is_better_kpi_still_finds_members_that_fell(self):
        self.assertTrue(self.api.polarity("revenue"))
        result = self.ce.find_counterexamples(
            self.rdf, "revenue", "fulfillment_rate", "product",
            self.period_a, self.period_b, cause_direction="down")
        self.assertEqual(result.status, "ok")
        self.assertGreater(result.n_counterexamples, 0)

    def test_no_counterexample_when_the_cause_moved_everywhere(self):
        result = self.ce.find_counterexamples(
            self.rdf, "revenue", "fulfillment_rate", "region",
            self.period_a, self.period_b, cause_direction="down")
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.n_counterexamples, 0)


class TestCounterexampleStatusesAndArguments(ConsistencyTestCase):
    def test_the_default_threshold_is_min_material_change_pct(self):
        from app.config import get_settings
        result = self.ce.find_counterexamples(
            self.rdf, "revenue", "fulfillment_rate", "product",
            self.period_a, self.period_b)
        self.assertEqual(result.kpi_move_pct, get_settings().min_material_change_pct)
        self.assertEqual(result.cause_move_pct, get_settings().min_material_change_pct)

    def test_an_invented_cause_direction_is_rejected(self):
        with self.assertRaises(InvalidArgumentError):
            self.ce.find_counterexamples(
                self.rdf, "revenue", "fulfillment_rate", "product",
                self.period_a, self.period_b, cause_direction="sideways")

    def test_an_impossibly_high_move_threshold_finds_no_unfavourable_members(self):
        result = self.ce.find_counterexamples(
            self.rdf, "revenue", "fulfillment_rate", "product",
            self.period_a, self.period_b, kpi_move_pct=1e6)
        self.assertEqual(result.status, "no_unfavourable_members")
        self.assertEqual(result.counterexamples, ())


# ---------------------------------------------------------------------------
# test_consistency_across_dimension()
# ---------------------------------------------------------------------------
class TestDimensionConsistency(ConsistencyTestCase):
    def test_cause_flat_members_are_their_own_bucket_not_a_disagreement(self):
        """`fulfillment_rate` is exactly 0.00 for two of three products --
        the cause said nothing there, which is `find_counterexamples`'
        business, not a sign disagreement."""
        result = self.ce.test_consistency_across_dimension(
            self.rdf, "revenue", "fulfillment_rate", "product", self.period_a, self.period_b)
        self.assertEqual(result.status, "ok")
        flat = [m for m in result.members if m.bucket == "cause_flat"]
        self.assertGreaterEqual(len(flat), 2)

    def test_the_rate_and_the_pooled_r_can_disagree_and_both_are_reported(self):
        result = self.ce.test_consistency_across_dimension(
            self.rdf, "revenue", "fulfillment_rate", "region", self.period_a, self.period_b)
        self.assertEqual(result.status, "ok")
        self.assertAlmostEqual(result.consistency_rate, 1.0, places=6)
        self.assertLess(result.pooled.r, 0)   # the pooled r reads the opposite way
        self.assertEqual(result.direction, "positive")

    def test_a_consistently_inverse_relationship_reads_as_opposite_not_a_disagreement(self):
        """Real fixture, threshold lowered to the actual magnitude of a
        ratio KPI's own percent change: every product's margin moves
        opposite its cost -- the mechanically correct relationship, held
        perfectly, not "inconsistency"."""
        result = self.ce.test_consistency_across_dimension(
            self.rdf, "gross_margin_pct", "cost_of_goods", "product",
            self.period_a, self.period_b, move_pct=0.01)
        self.assertEqual(result.status, "ok")
        self.assertTrue(all(m.bucket == "opposite_sign" for m in result.members))
        self.assertAlmostEqual(result.consistency_rate, 0.0, places=6)
        self.assertAlmostEqual(result.co_movement_strength, 1.0, places=6)
        self.assertEqual(result.direction, "negative")

    def test_the_default_min_n_is_three_not_correlates_six(self):
        """Retail `region` has 4 members -- `insufficient_n` at
        `correlate.DEFAULT_MIN_N=6` but `ok` at this module's own floor."""
        result = self.ce.test_consistency_across_dimension(
            self.rdf, "revenue", "fulfillment_rate", "region", self.period_a, self.period_b)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.pooled.status, "ok")

    def test_a_two_member_dimension_is_insufficient_members(self):
        result = self.ce.test_consistency_across_dimension(
            self.rdf, "revenue", "fulfillment_rate", "channel", self.period_a, self.period_b)
        self.assertEqual(result.status, "insufficient_members")
        self.assertIsNone(result.consistency_rate)
        self.assertIsNone(result.pooled)

    def test_the_pooled_r_matches_an_independent_correlate_kpis_call(self):
        from app.agent.correlate import CorrelationEngine
        result = self.ce.test_consistency_across_dimension(
            self.rdf, "revenue", "fulfillment_rate", "region", self.period_a, self.period_b)
        direct = CorrelationEngine(self.api).correlate_kpis(
            self.rdf, "revenue", "fulfillment_rate", mode="cross_sectional",
            period_a=self.period_a, period_b=self.period_b, dimension="region", min_n=3)
        self.assertEqual(result.pooled.to_payload(), direct.to_payload())


# ---------------------------------------------------------------------------
# test_holdout_segments()
# ---------------------------------------------------------------------------
class TestHoldoutSegments(ConsistencyTestCase):
    def test_the_affected_group_falls_harder_than_the_holdout(self):
        result = self.ce.test_holdout_segments(
            self.rdf, "revenue", "fulfillment_rate", "product", self.period_a, self.period_b,
            cause_direction="down")
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.n_affected, 1)
        self.assertEqual(result.n_unaffected, 2)
        self.assertAlmostEqual(result.did_pct, -25.29, places=1)
        self.assertEqual(result.verdict, "cause_explains_the_gap")

    def test_the_holdout_statistic_is_the_difference_in_differences_not_the_level_gap(self):
        """The level gap `compare_cohorts` reports by default is a
        group-size artefact on this fixture (one member vs two) -- pinning
        the divergence here stops a future "simplification" back to
        `comparison.gap.gap_abs`."""
        result = self.ce.test_holdout_segments(
            self.rdf, "revenue", "fulfillment_rate", "product", self.period_a, self.period_b,
            cause_direction="down")
        self.assertLess(abs(result.did_pct), 100)
        self.assertGreater(abs(result.comparison.gap.gap_abs), 1_000_000)

    def test_it_delegates_to_compare_cohorts(self):
        from app.agent.segments import SegmentEngine
        result = self.ce.test_holdout_segments(
            self.rdf, "revenue", "fulfillment_rate", "product", self.period_a, self.period_b,
            cause_direction="down")
        direct = SegmentEngine(self.api).compare_cohorts(
            self.rdf, "revenue", {"product": ["Product A"]},
            {"product": ["Product B", "Product C"]}, self.period_a, self.period_b)
        self.assertEqual(result.comparison.to_payload(), direct.to_payload())

    def test_every_member_affected_is_a_status_not_an_error(self):
        result = self.ce.test_holdout_segments(
            self.rdf, "revenue", "fulfillment_rate", "product", self.period_a, self.period_b,
            cause_move_pct=0.0, cause_direction="any")
        self.assertEqual(result.status, "no_unaffected_members")
        self.assertIsNone(result.did_pct)
        self.assertIsNone(result.comparison)

    def test_no_member_affected_is_a_status_not_an_error(self):
        result = self.ce.test_holdout_segments(
            self.rdf, "revenue", "fulfillment_rate", "product", self.period_a, self.period_b,
            cause_move_pct=1e6, cause_direction="up")
        self.assertEqual(result.status, "no_affected_members")

    def test_a_small_did_pct_is_kpi_moved_regardless(self):
        members = ["A", "B", "C", "D"]
        # every member's kpi falls by roughly the same amount, whether or
        # not the cause moved -- the cause does not differentiate the groups.
        df, schema = _member_frame(
            members, kpi_values_a=[90, 88, 91, 89], kpi_values_b=[100, 100, 100, 100],
            cause_values_a=[50, 20, 51, 19], cause_values_b=[50, 50, 50, 50])
        api = ContractAPI(schema)
        result = ConsistencyEngine(api).test_holdout_segments(
            df, "revenue", "units_sold", "region",
            {"type": "quarter", "year": 2026, "quarter": 2},
            {"type": "quarter", "year": 2026, "quarter": 1}, cause_move_pct=10.0, cause_direction="any")
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.verdict, "kpi_moved_regardless")


# ---------------------------------------------------------------------------
# integrity / prose guard / determinism / portability
# ---------------------------------------------------------------------------
class TestNumericIntegrityAndProseGuard(ConsistencyTestCase):
    def test_the_payload_survives_json_dumps_with_no_nan_or_inf(self):
        counter = self.ce.find_counterexamples(
            self.rdf, "revenue", "fulfillment_rate", "product", self.period_a, self.period_b)
        consistency = self.ce.test_consistency_across_dimension(
            self.rdf, "revenue", "fulfillment_rate", "region", self.period_a, self.period_b)
        holdout = self.ce.test_holdout_segments(
            self.rdf, "revenue", "fulfillment_rate", "product", self.period_a, self.period_b,
            cause_direction="down")
        assert_json_safe(counter.to_payload())
        assert_json_safe(consistency.to_payload())
        assert_json_safe(holdout.to_payload())

    def test_no_string_field_outside_the_allowlist_looks_like_prose(self):
        counter = self.ce.find_counterexamples(
            self.rdf, "revenue", "fulfillment_rate", "product", self.period_a, self.period_b)
        consistency = self.ce.test_consistency_across_dimension(
            self.rdf, "revenue", "fulfillment_rate", "region", self.period_a, self.period_b)
        holdout = self.ce.test_holdout_segments(
            self.rdf, "revenue", "fulfillment_rate", "product", self.period_a, self.period_b,
            cause_direction="down")
        assert_no_prose_leak(counter.to_payload())
        assert_no_prose_leak(consistency.to_payload())
        assert_no_prose_leak(holdout.to_payload())
        _assert_no_banned_keys(counter.to_payload())
        _assert_no_banned_keys(consistency.to_payload())
        _assert_no_banned_keys(holdout.to_payload())


class TestDeterminism(ConsistencyTestCase):
    def test_identical_input_gives_identical_output_across_runs(self):
        first = self.ce.find_counterexamples(
            self.rdf, "revenue", "fulfillment_rate", "product",
            self.period_a, self.period_b).to_payload()
        second = self.ce.find_counterexamples(
            self.rdf, "revenue", "fulfillment_rate", "product",
            self.period_a, self.period_b).to_payload()
        self.assertEqual(first, second)

    def test_members_are_reported_in_sorted_order(self):
        result = self.ce.test_consistency_across_dimension(
            self.rdf, "revenue", "fulfillment_rate", "region", self.period_a, self.period_b)
        names = [m.member for m in result.members]
        self.assertEqual(names, sorted(names))


class TestPortabilityAcrossDatasets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.schema = contracted("hospital_sample.csv", FIXTURES)
        cls.ce = ConsistencyEngine(ContractAPI(cls.schema))
        assert "recovery_rate" in cls.schema.available_kpis
        cls.period_a = {"type": "quarter", "year": 2025, "quarter": 2}
        cls.period_b = {"type": "quarter", "year": 2025, "quarter": 1}

    def test_all_three_tools_run_on_a_contract_only_ratio_kpi(self):
        counter = self.ce.find_counterexamples(
            self.df, "recovery_rate", "admissions", "department", self.period_a, self.period_b)
        self.assertIn(counter.status, ("ok", "no_members", "insufficient_members",
                                       "no_unfavourable_members"))

        consistency = self.ce.test_consistency_across_dimension(
            self.df, "recovery_rate", "admissions", "department", self.period_a, self.period_b)
        self.assertIn(consistency.status, ("ok", "insufficient_members"))

        holdout = self.ce.test_holdout_segments(
            self.df, "recovery_rate", "admissions", "department", self.period_a, self.period_b)
        self.assertIn(holdout.status, ("ok", "no_affected_members", "no_unaffected_members"))


if __name__ == "__main__":
    unittest.main()
