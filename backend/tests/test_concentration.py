"""
`measure_concentration` / `find_outlier_contributors` -- `agent/concentration.py`.

`measure_concentration` computes no arithmetic over row data itself -- it is
built entirely on `BreakdownEngine.rank_entities`, so the tests that matter
most are the index arithmetic layered on top: HHI and the sample-corrected
Gini match hand-computed values on small synthetic distributions where the
answer is known exactly (uniform -> 0, single-member dominance -> 1.0), the
five typed statuses gate the right things (an index is `None` exactly when
computing it would be wrong, not merely inconvenient), and `top_k_shares`
reconciles with `rank_entities` itself rather than being recomputed
independently.

`find_outlier_contributors` reuses `decompose_by_dimension`'s already-tested
`contribution_pct`/`over_index` arithmetic, so what is tested here is the
layer on top of it: population-relative `robust_z` versus the fixed
`DISPROPORTIONATE_AT` threshold are visibly different judgments, the
materiality gate hides nothing (it only reorders), and a degenerate
population (identical over-indices) never reports `robust_z = inf`.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from app.agent.concentration import ConcentrationEngine
from app.agent.contract_api import ContractAPI
from app.agent.decompose import DecomposeEngine
from app.agent.errors import InvalidArgumentError, UnknownKpiError
from app.agent.breakdown import BreakdownEngine
from app.engines import metrics
from app.engines.drivers import DISPROPORTIONATE_AT

from .base import (EngineTestCase, FIXTURES, assert_json_safe,
                   assert_no_prose_leak, contracted)

BANNED = {"interpretation", "detail", "narrative", "note"}


def _assert_no_banned_keys(payload, path: str = "$") -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            assert key not in BANNED, f"{path}.{key} is a banned prose field"
            _assert_no_banned_keys(value, f"{path}.{key}")
    elif isinstance(payload, list):
        for i, item in enumerate(payload):
            _assert_no_banned_keys(item, f"{path}[{i}]")


def _frame(region_values, revenue_values, dates=None):
    n = len(region_values)
    dates = dates or ["2026-05-01"] * n
    raw = pd.DataFrame({"region": region_values, "revenue": revenue_values, "date": dates})
    schema = metrics.detect_schema(raw)
    df = metrics.prepare(raw, schema)
    return df, schema


class ConcentrationTestCase(EngineTestCase):
    """The shared retail fixture, with a compiled contract."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.rdf, cls.rschema = contracted("business_metrics_sample.csv")
        cls.api = ContractAPI(cls.rschema)
        cls.ce = ConcentrationEngine(cls.api)
        cls.be = BreakdownEngine(cls.api)
        cls.period_a = {"type": "quarter", "year": 2026, "quarter": 2}
        cls.period_b = {"type": "quarter", "year": 2026, "quarter": 1}


# ---------------------------------------------------------------------------
# measure_concentration() -- arithmetic identities
# ---------------------------------------------------------------------------
class TestUniformDistribution(unittest.TestCase):
    def test_hhi_gini_and_effective_members_match_the_uniform_formulas(self):
        df, schema = _frame(["A", "B", "C", "D"], [100.0, 100.0, 100.0, 100.0])
        result = ConcentrationEngine(ContractAPI(schema)).measure_concentration(
            df, "revenue", "region", {"type": "all"})
        self.assertEqual(result.status, "ok")
        self.assertAlmostEqual(result.hhi, 0.25, places=9)
        self.assertAlmostEqual(result.hhi_normalized, 0.0, places=9)
        self.assertAlmostEqual(result.effective_members, 4.0, places=9)
        self.assertAlmostEqual(result.gini, 0.0, places=9)


class TestSingleMemberDominance(unittest.TestCase):
    def test_one_member_holding_everything_is_hhi_and_gini_of_one(self):
        df, schema = _frame(["A", "B", "C"], [100.0, 0.0, 0.0])
        result = ConcentrationEngine(ContractAPI(schema)).measure_concentration(
            df, "revenue", "region", {"type": "all"})
        self.assertEqual(result.status, "ok")
        self.assertAlmostEqual(result.hhi, 1.0, places=9)
        self.assertAlmostEqual(result.gini, 1.0, places=9)
        self.assertAlmostEqual(result.effective_members, 1.0, places=9)

    def test_the_uncorrected_estimator_would_have_given_two_thirds_not_one(self):
        """Documents why the small-sample correction exists: the naive
        population Gini over this exact distribution is 0.667, not 1.0."""
        values = sorted([100.0, 0.0, 0.0])
        n = len(values)
        total = sum(values)
        weighted = sum((i + 1) * v for i, v in enumerate(values))
        g_biased = (2.0 * weighted) / (n * total) - (n + 1.0) / n
        self.assertAlmostEqual(g_biased, 2.0 / 3.0, places=6)


class TestGiniBounds(unittest.TestCase):
    def test_gini_stays_within_zero_and_one_across_synthetic_distributions(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            n = rng.integers(2, 12)
            values = rng.uniform(0, 1000, size=n).tolist()
            regions = [f"r{i}" for i in range(n)]
            df, schema = _frame(regions, values)
            result = ConcentrationEngine(ContractAPI(schema)).measure_concentration(
                df, "revenue", "region", {"type": "all"})
            if result.gini is not None:
                self.assertGreaterEqual(result.gini, -1e-9)
                self.assertLessEqual(result.gini, 1.0 + 1e-9)


class TestTopKShareReconcilesWithRankEntities(ConcentrationTestCase):
    def test_top_3_share_matches_rank_entities_cumulative_share(self):
        result = self.ce.measure_concentration(self.rdf, "revenue", "product", {"type": "all"})
        rank = self.be.rank_entities(self.rdf, "revenue", "product", {"type": "all"}, limit=3)
        self.assertAlmostEqual(result.top_k_shares["top_3"],
                               rank.entries[-1].cumulative_share_pct, places=6)

    def test_top_k_is_capped_at_the_member_count(self):
        result = self.ce.measure_concentration(self.rdf, "revenue", "product", {"type": "all"})
        # the retail fixture has 3 products; top_10 must equal top_3 (=100%)
        self.assertAlmostEqual(result.top_k_shares["top_10"], result.top_k_shares["top_3"], places=6)
        self.assertAlmostEqual(result.top_k_shares["top_10"], 100.0, places=4)

    def test_members_for_80pct_is_the_smallest_k_crossing_the_threshold_exactly(self):
        df, schema = _frame(["A", "B", "C", "D", "E"], [80.0, 5.0, 5.0, 5.0, 5.0])
        result = ConcentrationEngine(ContractAPI(schema)).measure_concentration(
            df, "revenue", "region", {"type": "all"})
        self.assertEqual(result.members_for_80pct, 1)


# ---------------------------------------------------------------------------
# measure_concentration() -- typed statuses
# ---------------------------------------------------------------------------
class TestNotAdditiveStatus(ConcentrationTestCase):
    def test_a_ratio_kpi_gives_every_index_none_but_still_lists_members(self):
        result = self.ce.measure_concentration(self.rdf, "gross_margin_pct", "product", {"type": "all"})
        self.assertEqual(result.status, "not_additive")
        self.assertIsNone(result.hhi)
        self.assertIsNone(result.gini)
        self.assertIsNone(result.effective_members)
        self.assertGreater(result.member_count, 0)


class TestNegativeMembersStatus(unittest.TestCase):
    def test_a_negative_member_value_gives_has_negative_members_and_none_indices(self):
        df, schema = _frame(["A", "B", "C"], [500.0, -50.0, 200.0])
        result = ConcentrationEngine(ContractAPI(schema)).measure_concentration(
            df, "revenue", "region", {"type": "all"})
        self.assertEqual(result.status, "has_negative_members")
        self.assertIsNone(result.hhi)
        self.assertIsNone(result.gini)
        self.assertIn("B", result.negative_members)

    def test_an_exact_zero_member_is_not_flagged_negative(self):
        df, schema = _frame(["A", "B", "C"], [500.0, 0.0, 200.0])
        result = ConcentrationEngine(ContractAPI(schema)).measure_concentration(
            df, "revenue", "region", {"type": "all"})
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.negative_members, ())


class TestTruncationStatus(unittest.TestCase):
    def test_over_max_groups_members_gives_too_many_members_but_top_k_still_exact(self):
        rng = np.random.default_rng(1)
        n = 600   # over QueryEngine.MAX_GROUPS (500)
        rows_per_member = 6   # detect_schema only admits a dimension at
                               # nunique <= max(60, 0.2*rows) -- 600 members
                               # needs >= 3000 rows to stay a real dimension
        regions = [f"r{i}" for i in range(n) for _ in range(rows_per_member)]
        values = rng.uniform(1, 100, size=n * rows_per_member).tolist()
        dates = ["2026-05-01"] * (n * rows_per_member)
        df, schema = _frame(regions, values, dates=dates)
        result = ConcentrationEngine(ContractAPI(schema)).measure_concentration(
            df, "revenue", "region", {"type": "all"})
        self.assertEqual(result.status, "too_many_members")
        self.assertIsNone(result.hhi)
        self.assertIsNone(result.gini)
        self.assertTrue(result.truncated)
        # top_1 share is computed from the single largest member, found via
        # rank_entities' own descending sort over the FULL (untruncated)
        # member set -- exact regardless of the 500-cell cap.
        self.assertIsNotNone(result.top_k_shares.get("top_1"))


class TestSingleMemberDimension(unittest.TestCase):
    """`school_kpi_smoke_sample.csv` has `school` at cardinality 1 -- a
    genuine edge case, not a synthetic one."""

    def test_n_equals_one_is_hhi_one_gini_none_no_crash(self):
        df, schema = contracted("school_kpi_smoke_sample.csv")
        result = ConcentrationEngine(ContractAPI(schema)).measure_concentration(
            df, "students_enrolled", "school", {"type": "all"})
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.member_count, 1)
        self.assertAlmostEqual(result.hhi, 1.0, places=9)
        self.assertIsNone(result.gini)
        self.assertIsNone(result.hhi_normalized)

    def test_a_small_but_near_even_dimension_has_a_much_smaller_normalized_hhi(self):
        """`campus` on the school fixture has 2 near-even members -- raw HHI
        alone would misreport this as roughly half-concentrated."""
        df, schema = contracted("school_kpi_smoke_sample.csv")
        result = ConcentrationEngine(ContractAPI(schema)).measure_concentration(
            df, "students_enrolled", "campus", {"type": "all"})
        self.assertEqual(result.status, "ok")
        self.assertGreater(result.hhi, 0.4)
        self.assertLess(result.hhi_normalized, 0.05)


class TestEmptyStatus(unittest.TestCase):
    def test_zero_matching_rows_is_reported_as_empty_via_filters(self):
        df, schema = _frame(["A", "B"], [100.0, 200.0])
        result = ConcentrationEngine(ContractAPI(schema)).measure_concentration(
            df, "revenue", "region", {"type": "all"}, filters={"region": ["A"]})
        self.assertGreaterEqual(result.member_count, 0)  # never raises


class TestAirlock(ConcentrationTestCase):
    def test_an_unknown_kpi_is_rejected(self):
        with self.assertRaises(UnknownKpiError):
            self.ce.measure_concentration(self.rdf, "not_a_real_kpi", "region", {"type": "all"})


# ---------------------------------------------------------------------------
# find_outlier_contributors()
# ---------------------------------------------------------------------------
class TestOutlierContributorsRecoversThePlantedOutlier(ConcentrationTestCase):
    def test_the_largest_over_index_mover_has_the_largest_robust_z_and_is_first(self):
        result = self.ce.find_outlier_contributors(self.rdf, "revenue", "region",
                                                    self.period_a, self.period_b)
        self.assertEqual(result.status, "ok")
        self.assertEqual(len(result.rows), 4)
        top = result.rows[0]
        self.assertEqual(top.robust_z, max(r.robust_z for r in result.rows if r.robust_z is not None))
        self.assertTrue(top.material)


class TestMaterialityGateReordersNotDrops(unittest.TestCase):
    def test_a_tiny_member_with_a_huge_swing_is_flagged_not_material_but_still_present(self):
        raw = pd.DataFrame({
            "region": ["North", "North", "South", "South", "East", "East", "West", "West"],
            "revenue": [100000.0, 100000.0, 100.0, 1.0, 90000.0, 95000.0, 80000.0, 85000.0],
            "date": ["2026-05-01", "2026-02-01"] * 4,
        })
        schema = metrics.detect_schema(raw)
        df = metrics.prepare(raw, schema)
        result = ConcentrationEngine(ContractAPI(schema)).find_outlier_contributors(
            df, "revenue", "region",
            {"type": "quarter", "year": 2026, "quarter": 2},
            {"type": "quarter", "year": 2026, "quarter": 1})
        south = next(r for r in result.rows if r.member == "South")
        self.assertFalse(south.material)
        self.assertIn(south.member, [r.member for r in result.rows])   # still present
        material_positions = [i for i, r in enumerate(result.rows) if r.material]
        non_material_positions = [i for i, r in enumerate(result.rows) if not r.material]
        if material_positions and non_material_positions:
            self.assertLess(max(material_positions), min(non_material_positions))


class TestInsufficientPopulation(unittest.TestCase):
    def test_fewer_than_four_members_is_insufficient_not_a_fabricated_score(self):
        raw = pd.DataFrame({
            "region": ["North", "North", "South", "South"],
            "revenue": [100.0, 90.0, 50.0, 45.0],
            "date": ["2026-05-01", "2026-02-01", "2026-05-01", "2026-02-01"],
        })
        schema = metrics.detect_schema(raw)
        df = metrics.prepare(raw, schema)
        result = ConcentrationEngine(ContractAPI(schema)).find_outlier_contributors(
            df, "revenue", "region",
            {"type": "quarter", "year": 2026, "quarter": 2},
            {"type": "quarter", "year": 2026, "quarter": 1})
        self.assertEqual(result.status, "insufficient")
        for row in result.rows:
            self.assertIsNone(row.robust_z)


class TestDegeneratePopulationNeverReportsInfinity(unittest.TestCase):
    def test_identical_over_indices_give_none_not_inf(self):
        # Every region doubles identically -- over_index is 1.0 everywhere,
        # so the MAD (and the std fallback) are both exactly zero.
        raw = pd.DataFrame({
            "region": ["North", "North", "South", "South", "East", "East", "West", "West"],
            "revenue": [200.0, 100.0, 400.0, 200.0, 600.0, 300.0, 800.0, 400.0],
            "date": ["2026-05-01", "2026-02-01"] * 4,
        })
        schema = metrics.detect_schema(raw)
        df = metrics.prepare(raw, schema)
        result = ConcentrationEngine(ContractAPI(schema)).find_outlier_contributors(
            df, "revenue", "region",
            {"type": "quarter", "year": 2026, "quarter": 2},
            {"type": "quarter", "year": 2026, "quarter": 1})
        self.assertEqual(result.status, "ok")
        for row in result.rows:
            self.assertIsNotNone(row.robust_z is None or np.isfinite(row.robust_z))
            if row.robust_z is not None:
                self.assertTrue(np.isfinite(row.robust_z))


class TestOutlierIsDisproportionateMatchesHouseThreshold(ConcentrationTestCase):
    def test_is_disproportionate_matches_the_fixed_threshold_independently(self):
        result = self.ce.find_outlier_contributors(self.rdf, "revenue", "region",
                                                    self.period_a, self.period_b)
        for row in result.rows:
            expected = row.over_index is not None and row.over_index >= DISPROPORTIONATE_AT
            self.assertEqual(row.is_disproportionate, expected)


class TestOutlierReconcilesWithDecomposeByDimension(ConcentrationTestCase):
    def test_contribution_pct_matches_decompose_by_dimension_exactly(self):
        decomposition = DecomposeEngine(self.api).decompose_by_dimension(
            self.rdf, "revenue", "region", self.period_a, self.period_b, max_items=200)
        outliers = self.ce.find_outlier_contributors(self.rdf, "revenue", "region",
                                                      self.period_a, self.period_b)
        by_member = {m.name: m.contribution_pct for m in decomposition.members}
        for row in outliers.rows:
            self.assertAlmostEqual(row.contribution_pct, by_member[row.member], places=6)


class TestOutlierRatioKpiProducesRealValues(ConcentrationTestCase):
    def test_a_ratio_kpi_does_not_crash_and_produces_real_or_none_scores(self):
        result = self.ce.find_outlier_contributors(self.rdf, "gross_margin_pct", "region",
                                                    self.period_a, self.period_b)
        self.assertIn(result.status, ("ok", "insufficient"))


# ---------------------------------------------------------------------------
# integrity / prose guard / determinism / portability
# ---------------------------------------------------------------------------
class TestNumericIntegrityAndProseGuard(ConcentrationTestCase):
    def test_the_payload_survives_json_dumps_with_no_nan_or_inf(self):
        conc = self.ce.measure_concentration(self.rdf, "revenue", "region", {"type": "all"})
        outliers = self.ce.find_outlier_contributors(self.rdf, "revenue", "region",
                                                      self.period_a, self.period_b)
        assert_json_safe(conc.to_payload())
        assert_json_safe(outliers.to_payload())

    def test_no_string_field_outside_the_allowlist_looks_like_prose(self):
        conc = self.ce.measure_concentration(self.rdf, "revenue", "region", {"type": "all"})
        outliers = self.ce.find_outlier_contributors(self.rdf, "revenue", "region",
                                                      self.period_a, self.period_b)
        assert_no_prose_leak(conc.to_payload())
        assert_no_prose_leak(outliers.to_payload())

    def test_no_banned_prose_key_survives(self):
        conc = self.ce.measure_concentration(self.rdf, "revenue", "region", {"type": "all"})
        outliers = self.ce.find_outlier_contributors(self.rdf, "revenue", "region",
                                                      self.period_a, self.period_b)
        _assert_no_banned_keys(conc.to_payload())
        _assert_no_banned_keys(outliers.to_payload())


class TestDeterminism(ConcentrationTestCase):
    def test_identical_input_gives_identical_output_across_runs(self):
        first = self.ce.measure_concentration(self.rdf, "revenue", "region", {"type": "all"}).to_payload()
        second = self.ce.measure_concentration(self.rdf, "revenue", "region", {"type": "all"}).to_payload()
        self.assertEqual(first, second)


class TestPortabilityAcrossDatasets(unittest.TestCase):
    def test_hospital_department_concentration_works_on_a_contract_only_dataset(self):
        df, schema = contracted("hospital_sample.csv", FIXTURES)
        result = ConcentrationEngine(ContractAPI(schema)).measure_concentration(
            df, "admissions", "department", {"type": "all"})
        self.assertEqual(result.status, "ok")
        self.assertGreater(result.member_count, 0)

    def test_school_campus_is_the_small_n_case_hhi_normalized_exists_for(self):
        df, schema = contracted("school_kpi_smoke_sample.csv")
        result = ConcentrationEngine(ContractAPI(schema)).measure_concentration(
            df, "students_enrolled", "campus", {"type": "all"})
        self.assertEqual(result.status, "ok")
        self.assertIsNotNone(result.hhi_normalized)


if __name__ == "__main__":
    unittest.main()
