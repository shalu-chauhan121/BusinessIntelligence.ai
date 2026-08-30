"""
`rank_entities` / `get_distribution` / `cross_tabulate` -- `agent/breakdown.py`.

All three delegate every number to `QueryEngine.query`, so the tests that
matter most are not about arithmetic `test_query_kpi.py` already covers --
they are about the bookkeeping layered on top: rank is dense and handles
ties, share is undefined (not fabricated) for a non-additive KPI, percentiles
match `numpy.percentile` exactly, and a cross-tab's row/column/grand totals
reconcile to `query_kpi` because each comes from its own slice rather than
from summing cells -- the one thing `drivers.cell_delta_grid` gets wrong and
this module deliberately does not reuse.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from app.agent.breakdown import BreakdownEngine
from app.agent.contract_api import ContractAPI
from app.agent.errors import InvalidArgumentError, UnknownKpiError
from app.agent.query import QueryEngine
from app.engines import metrics
from app.engines.drivers import robust_sigma

from .base import (EngineTestCase, FIXTURES, assert_json_safe,
                   assert_no_prose_leak, contracted)


class BreakdownTestCase(EngineTestCase):
    """The shared retail fixture, with a compiled contract so ratio KPIs are
    breakdownable."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.rdf, cls.rschema = contracted("business_metrics_sample.csv")
        cls.api = ContractAPI(cls.rschema)
        cls.be = BreakdownEngine(cls.api)
        cls.qe = QueryEngine(cls.api)


# ---------------------------------------------------------------------------
# rank_entities()
# ---------------------------------------------------------------------------
class TestRankEntitiesTopAndBottom(BreakdownTestCase):
    def test_top_n_and_bottom_n_are_exact_complements(self):
        full = self.be.rank_entities(self.rdf, "revenue", "region", {"type": "all"}, limit=None)
        all_members = {e.member for e in full.entries}
        n = len(full.entries)
        half = n // 2

        top = self.be.rank_entities(self.rdf, "revenue", "region", {"type": "all"},
                                    order="desc", limit=half)
        bottom = self.be.rank_entities(self.rdf, "revenue", "region", {"type": "all"},
                                       order="asc", limit=n - half)
        top_members = {e.member for e in top.entries}
        bottom_members = {e.member for e in bottom.entries}

        self.assertEqual(top_members | bottom_members, all_members)
        self.assertEqual(top_members & bottom_members, set())

    def test_best_worst_invert_for_a_higher_is_better_false_kpi(self):
        worst = self.be.rank_entities(self.rdf, "marketing_spend", "region", {"type": "all"},
                                      order="worst", limit=1)
        best = self.be.rank_entities(self.rdf, "marketing_spend", "region", {"type": "all"},
                                     order="best", limit=1)
        self.assertEqual(worst.resolved_order, "desc")   # worst spend = highest spend
        self.assertEqual(best.resolved_order, "asc")      # best spend = lowest spend
        self.assertNotEqual(worst.entries[0].member, best.entries[0].member)

    def test_others_rolls_up_the_truncated_tail_rather_than_dropping_it(self):
        full = self.be.rank_entities(self.rdf, "revenue", "region", {"type": "all"}, limit=None)
        top2 = self.be.rank_entities(self.rdf, "revenue", "region", {"type": "all"}, limit=2)
        self.assertIsNotNone(top2.others)
        self.assertEqual(top2.others.count, len(full.entries) - 2)
        tail_sum = sum(e.value for e in full.entries[2:])
        self.assertAlmostEqual(top2.others.value, tail_sum, places=4)


class TestShareOfTotal(BreakdownTestCase):
    def test_share_of_total_sums_to_100_for_an_additive_kpi(self):
        full = self.be.rank_entities(self.rdf, "revenue", "region", {"type": "all"}, limit=None)
        self.assertTrue(full.additive)
        self.assertAlmostEqual(sum(e.share_of_total_pct for e in full.entries), 100.0, places=4)

    def test_share_is_none_for_a_ratio_kpi_not_a_fabricated_number(self):
        full = self.be.rank_entities(self.rdf, "gross_margin_pct", "region", {"type": "all"}, limit=None)
        self.assertFalse(full.additive)
        for entry in full.entries:
            self.assertIsNone(entry.share_of_total_pct)
            self.assertIsNone(entry.cumulative_share_pct)


class TestRankIsDenseAndHandlesTies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "region": ["North", "South", "East"],
            "revenue": [100.0, 100.0, 50.0],
        })
        cls.schema = metrics.detect_schema(raw)
        cls.df = metrics.prepare(raw, cls.schema)
        cls.be = BreakdownEngine(ContractAPI(cls.schema))

    def test_tied_values_share_a_rank_and_the_next_rank_has_no_gap(self):
        result = self.be.rank_entities(self.df, "revenue", "region", {"type": "all"},
                                       order="desc", limit=None)
        self.assertEqual([e.rank for e in result.entries], [1, 1, 2])
        self.assertFalse(result.entries[0].tied_with_previous)
        self.assertTrue(result.entries[1].tied_with_previous)
        self.assertFalse(result.entries[2].tied_with_previous)


# ---------------------------------------------------------------------------
# get_distribution()
# ---------------------------------------------------------------------------
class TestDistributionMatchesIndependentComputation(BreakdownTestCase):
    def _member_values(self, kpi_key, dimension):
        result = self.qe.query(self.rdf, [kpi_key], {"type": "all"}, group_by=[dimension])
        return [c.values[kpi_key] for c in result.cells]

    def test_percentiles_match_numpy_percentile_exactly(self):
        result = self.be.get_distribution(self.rdf, "revenue", "product", {"type": "all"})
        values = self._member_values("revenue", "product")
        expected = np.percentile(values, [10, 25, 50, 75, 90])
        for pct, exp in zip((10, 25, 50, 75, 90), expected):
            self.assertAlmostEqual(result.percentiles[f"p{pct}"], float(exp), places=6)

    def test_median_and_robust_sigma_match_drivers_robust_sigma(self):
        result = self.be.get_distribution(self.rdf, "revenue", "product", {"type": "all"})
        values = self._member_values("revenue", "product")
        med, sigma = robust_sigma(values)
        self.assertAlmostEqual(result.median, med, places=6)
        self.assertAlmostEqual(result.robust_sigma, sigma, places=6)

    def test_histogram_bucket_counts_sum_to_the_member_count(self):
        result = self.be.get_distribution(self.rdf, "revenue", "region", {"type": "all"})
        self.assertEqual(sum(b.count for b in result.histogram), result.count)

    def test_iqr_and_spread_match_a_hand_computation(self):
        result = self.be.get_distribution(self.rdf, "revenue", "product", {"type": "all"})
        values = self._member_values("revenue", "product")
        p25, p75 = np.percentile(values, [25, 75])
        self.assertAlmostEqual(result.iqr, float(p75 - p25), places=6)
        self.assertAlmostEqual(result.spread, max(values) - min(values), places=6)


class TestSingleMemberDimensionIsHandledNotCrashed(unittest.TestCase):
    """`school_kpi_smoke_sample.csv` has `school` at cardinality 1 -- a
    genuine edge case, not a synthetic one."""

    @classmethod
    def setUpClass(cls):
        cls.df, cls.schema = contracted("school_kpi_smoke_sample.csv")
        assert cls.schema.dimensions[0] == "school"
        cls.be = BreakdownEngine(ContractAPI(cls.schema))

    def test_a_single_member_distribution_is_insufficient_not_a_crash(self):
        result = self.be.get_distribution(self.df, "students_enrolled", "school", {"type": "all"})
        self.assertEqual(result.count, 1)
        self.assertEqual(result.status, "insufficient")

    def test_a_single_member_ranking_is_a_ranking_of_one(self):
        result = self.be.rank_entities(self.df, "students_enrolled", "school", {"type": "all"})
        self.assertEqual(len(result.entries), 1)
        self.assertEqual(result.entries[0].rank, 1)


# ---------------------------------------------------------------------------
# cross_tabulate()
# ---------------------------------------------------------------------------
class TestCrossTabReconciliation(BreakdownTestCase):
    def test_totals_reconcile_to_the_ungrouped_total_for_an_additive_kpi(self):
        ct = self.be.cross_tabulate(self.rdf, "revenue", "region", "product", {"type": "all"})
        ungrouped = self.qe.query(self.rdf, "revenue", {"type": "all"}).cells[0].values["revenue"]
        cell_sum = sum(c.value for c in ct.cells if c.value == c.value)
        row_sum = sum(v for v in ct.row_totals.values() if v == v)
        col_sum = sum(v for v in ct.col_totals.values() if v == v)

        self.assertAlmostEqual(ct.grand_total, ungrouped, places=6)
        self.assertAlmostEqual(cell_sum, ungrouped, places=4)
        self.assertAlmostEqual(row_sum, ungrouped, places=4)
        self.assertAlmostEqual(col_sum, ungrouped, places=4)

    def test_grand_total_matches_query_kpi_but_cell_sum_does_not_for_a_ratio_kpi(self):
        ct = self.be.cross_tabulate(self.rdf, "gross_margin_pct", "region", "product", {"type": "all"})
        ungrouped = self.qe.query(
            self.rdf, "gross_margin_pct", {"type": "all"}).cells[0].values["gross_margin_pct"]
        cell_sum = sum(c.value for c in ct.cells if c.value == c.value)

        self.assertAlmostEqual(ct.grand_total, ungrouped, places=6)
        self.assertGreater(abs(cell_sum - ungrouped), 1.0)
        self.assertFalse(ct.additive)


class TestEmptyCombinationIsNoneNotZero(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "region": ["North", "South"],
            "product": ["A", "B"],
            "revenue": [0.0, 500.0],
        })
        cls.schema = metrics.detect_schema(raw)
        cls.df = metrics.prepare(raw, cls.schema)
        cls.be = BreakdownEngine(ContractAPI(cls.schema))

    def test_a_combination_with_no_rows_is_value_none_not_zero(self):
        ct = self.be.cross_tabulate(self.df, "revenue", "region", "product", {"type": "all"})
        empty_cell = next(c for c in ct.cells if c.a == "North" and c.b == "B")
        self.assertIsNone(empty_cell.value)
        self.assertEqual(empty_cell.rows, 0)

    def test_a_real_zero_is_distinguishable_from_the_empty_combination(self):
        ct = self.be.cross_tabulate(self.df, "revenue", "region", "product", {"type": "all"})
        real_zero = next(c for c in ct.cells if c.a == "North" and c.b == "A")
        self.assertEqual(real_zero.value, 0.0)
        self.assertEqual(real_zero.rows, 1)


class TestHighCardinalityIsCappedAndReported(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        n_a, n_b = 30, 20   # 600 combinations, over the 500-cell cap
        rows = n_a * n_b
        raw = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=rows, freq="D").astype(str),
            "dim_a": [f"a{i}" for i in range(n_a) for _ in range(n_b)],
            "dim_b": [f"b{j}" for _ in range(n_a) for j in range(n_b)],
            "revenue": rng.uniform(1, 100, size=rows),
        })
        cls.schema = metrics.detect_schema(raw)
        cls.df = metrics.prepare(raw, cls.schema)
        cls.be = BreakdownEngine(ContractAPI(cls.schema))

    def test_over_500_combinations_is_truncated_but_reports_the_true_total(self):
        ct = self.be.cross_tabulate(self.df, "revenue", "dim_a", "dim_b", {"type": "all"})
        self.assertTrue(ct.truncated)
        self.assertEqual(ct.total_groups, 600)
        self.assertEqual(len(ct.cells), len(ct.members_a) * len(ct.members_b))

    def test_limit_a_and_limit_b_pick_members_by_magnitude_and_report_truncated(self):
        ct = self.be.cross_tabulate(self.df, "revenue", "dim_a", "dim_b", {"type": "all"},
                                    limit_a=5, limit_b=5)
        self.assertEqual(len(ct.members_a), 5)
        self.assertEqual(len(ct.members_b), 5)
        self.assertEqual(len(ct.cells), 25)
        self.assertTrue(ct.truncated)


# ---------------------------------------------------------------------------
# airlock
# ---------------------------------------------------------------------------
class TestAirlock(BreakdownTestCase):
    def test_a_time_grain_column_as_the_ranking_dimension_is_rejected(self):
        with self.assertRaises(InvalidArgumentError) as ctx:
            self.be.rank_entities(self.rdf, "revenue", "_period", {"type": "all"})
        self.assertIn("get_timeseries", ctx.exception.to_payload()["reason"])

    def test_crossing_a_dimension_with_itself_is_rejected(self):
        with self.assertRaises(InvalidArgumentError):
            self.be.cross_tabulate(self.rdf, "revenue", "region", "region", {"type": "all"})

    def test_an_unknown_kpi_is_rejected(self):
        with self.assertRaises(UnknownKpiError):
            self.be.rank_entities(self.rdf, "not_a_real_kpi", "region", {"type": "all"})


# ---------------------------------------------------------------------------
# integrity / prose guard / determinism / portability
# ---------------------------------------------------------------------------
class TestNumericIntegrityAndProseGuard(BreakdownTestCase):
    def test_the_payload_survives_json_dumps_with_no_nan_or_inf(self):
        rank = self.be.rank_entities(self.rdf, "gross_margin_pct", "region", {"type": "all"})
        dist = self.be.get_distribution(self.rdf, "revenue", "product", {"type": "all"})
        ct = self.be.cross_tabulate(self.rdf, "gross_margin_pct", "region", "product", {"type": "all"})
        assert_json_safe(rank.to_payload())
        assert_json_safe(dist.to_payload())
        assert_json_safe(ct.to_payload())

    def test_no_string_field_outside_the_allowlist_looks_like_prose(self):
        rank = self.be.rank_entities(self.rdf, "revenue", "region", {"type": "all"})
        dist = self.be.get_distribution(self.rdf, "revenue", "product", {"type": "all"})
        ct = self.be.cross_tabulate(self.rdf, "revenue", "region", "product", {"type": "all"})
        assert_no_prose_leak(rank.to_payload())
        assert_no_prose_leak(dist.to_payload())
        assert_no_prose_leak(ct.to_payload())


class TestDeterminism(BreakdownTestCase):
    def test_identical_input_gives_identical_output_across_runs(self):
        first = self.be.cross_tabulate(self.rdf, "revenue", "region", "product", {"type": "all"}).to_payload()
        second = self.be.cross_tabulate(self.rdf, "revenue", "region", "product", {"type": "all"}).to_payload()
        self.assertEqual(first, second)


class TestPortabilityAcrossDatasets(unittest.TestCase):
    """The hospital fixture has a single dimension (`department`) -- a
    cross-tab there has to cross it against a time grain rather than assume a
    second dimension exists."""

    @classmethod
    def setUpClass(cls):
        cls.df, cls.schema = contracted("hospital_sample.csv", FIXTURES)
        assert cls.schema.dimensions == ["department"]
        cls.be = BreakdownEngine(ContractAPI(cls.schema))

    def test_cross_tabulate_crosses_the_single_dimension_against_a_time_grain(self):
        ct = self.be.cross_tabulate(self.df, "admissions", "department", "_period", {"type": "all"})
        self.assertTrue(ct.members_a)
        self.assertTrue(ct.members_b)

    def test_rank_and_distribution_work_on_a_contract_only_ratio_kpi(self):
        rank = self.be.rank_entities(self.df, "recovery_rate", "department", {"type": "all"})
        self.assertTrue(rank.entries)
        dist = self.be.get_distribution(self.df, "recovery_rate", "department", {"type": "all"})
        self.assertIn(dist.status, ("ok", "insufficient"))


if __name__ == "__main__":
    unittest.main()
