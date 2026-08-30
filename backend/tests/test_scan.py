"""
`scan_kpis` / `rank_kpis_by_movement` / `scan_anomalies` / `scan_dimension_outliers`
-- `agent/scan.py`.

The tests that matter most here are not about arithmetic -- every number a
scan produces is either `ComparisonEngine.significance` or `drivers.robust_sigma`,
both already covered by their own suites. They are about the two things this
module adds on top: **batching is equivalent to, and dramatically cheaper
than, a naive per-KPI loop** (`TestBatchingIsEquivalentToPerKpiCalls`,
`TestBatchingCost`), and **`rank_kpis_by_movement` ranks by how bad a move is,
not how big it is** (`TestRankKpisByMovementIsPolarityAware`) -- the board's
own stated test for the tool.

`scan_dimension_outliers` is also checked against the retail sample's planted
scenario (`sample_data/ground_truth.json`): Product A on `units_sold` and
North on `units_sold` by region are both known, real outliers, and recovering
them is a stronger guarantee than any synthetic fixture.
"""
from __future__ import annotations

import unittest
from unittest import mock

import pandas as pd

from app.agent.compare import ComparisonEngine
from app.agent.contract_api import ContractAPI
from app.agent.errors import InvalidArgumentError
from app.agent.quality import QualityEngine
from app.agent.query import QueryEngine
from app.agent.scan import MAX_ANOMALY_FOLLOWUPS, ScanEngine
from app.engines import metrics

from .base import (EngineTestCase, FIXTURES, assert_json_safe,
                   assert_no_prose_leak, contracted)


class ScanTestCase(EngineTestCase):
    """The shared retail fixture, with a compiled contract."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.rdf, cls.rschema = contracted("business_metrics_sample.csv")
        cls.api = ContractAPI(cls.rschema)
        cls.se = ScanEngine(cls.api)


def _counting_patch(cls, method_name: str):
    """Patches `cls.method_name` with a wrapper that counts calls while still
    executing the real implementation -- used to assert a scan's cost is
    bounded without timing, which would be flaky."""
    original = getattr(cls, method_name)
    calls = {"n": 0}

    def wrapper(self, *args, **kwargs):
        calls["n"] += 1
        return original(self, *args, **kwargs)

    return mock.patch.object(cls, method_name, wrapper), calls


# ---------------------------------------------------------------------------
# scan_kpis()
# ---------------------------------------------------------------------------
class TestScanKpisCoverage(ScanTestCase):
    def test_scan_kpis_covers_exactly_available_keys_no_more_no_less(self):
        period = {"type": "quarter", "year": 2026, "quarter": 2}
        scan = self.se.scan_kpis(self.rdf, period)
        self.assertEqual({r.kpi for r in scan.rows}, set(self.api.available_keys(self.rdf)))
        self.assertEqual(scan.considered, len(self.api.available_keys(self.rdf)))
        self.assertFalse(scan.truncated)


class TestBatchingIsEquivalentToPerKpiCalls(ScanTestCase):
    def test_every_batched_row_matches_an_independent_significance_call(self):
        period = {"type": "quarter", "year": 2025, "quarter": 4}
        scan = self.se.scan_kpis(self.rdf, period)
        ce = ComparisonEngine(self.api)
        for row in scan.rows:
            sig = ce.significance(self.rdf, row.kpi, period)
            self.assertEqual(row.change_pct, sig.change_pct, row.kpi)
            self.assertEqual(row.robust_z, sig.robust_z, row.kpi)
            self.assertEqual(row.verdict, sig.verdict, row.kpi)
            self.assertEqual(row.history_status, sig.history_status, row.kpi)
            self.assertEqual(row.power, sig.power, row.kpi)


class TestBatchingCost(ScanTestCase):
    def test_scan_kpis_issues_one_grouped_query_regardless_of_kpi_count(self):
        patch, calls = _counting_patch(QueryEngine, "query")
        with patch:
            self.se.scan_kpis(self.rdf, {"type": "quarter", "year": 2026, "quarter": 2})
        self.assertEqual(calls["n"], 1)

    def test_rank_kpis_by_movement_issues_one_grouped_query(self):
        patch, calls = _counting_patch(QueryEngine, "query")
        with patch:
            self.se.rank_kpis_by_movement(self.rdf, {"type": "quarter", "year": 2026, "quarter": 2})
        self.assertEqual(calls["n"], 1)

    def test_normal_range_is_never_called_inside_a_scan(self):
        """The 76 ms/KPI trap -- locked so a later refactor cannot quietly
        reintroduce a `normal_range` call into a scan loop."""
        patch, calls = _counting_patch(ComparisonEngine, "normal_range")
        with patch:
            period = {"type": "quarter", "year": 2026, "quarter": 2}
            self.se.scan_kpis(self.rdf, period)
            self.se.rank_kpis_by_movement(self.rdf, period)
            self.se.scan_anomalies(self.rdf, period)
        self.assertEqual(calls["n"], 0)


# ---------------------------------------------------------------------------
# rank_kpis_by_movement()
# ---------------------------------------------------------------------------
class TestRankKpisByMovementIsPolarityAware(unittest.TestCase):
    """A synthetic fixture where a lower-is-better KPI (`marketing_spend`)
    rises 10% and a higher-is-better KPI (`revenue`) rises 30% -- the bigger
    move is good news, the smaller one is bad news."""

    @classmethod
    def setUpClass(cls):
        raw = pd.DataFrame({
            "date": ["2024-01-15", "2024-01-16", "2024-04-15", "2024-04-16"],
            "revenue": [1000.0, 1000.0, 1300.0, 1300.0],          # +30%, favourable
            "marketing_spend": [500.0, 500.0, 550.0, 550.0],      # +10%, unfavourable
        })
        cls.schema = metrics.detect_schema(raw)
        cls.df = metrics.prepare(raw, cls.schema)
        cls.se = ScanEngine(ContractAPI(cls.schema))
        cls.period = {"type": "quarter", "year": 2024, "quarter": 2}

    def test_the_smaller_but_unfavourable_move_ranks_first_under_unfavourability(self):
        result = self.se.rank_kpis_by_movement(self.df, self.period, order_by="unfavourability")
        self.assertEqual(result.entries[0].row.kpi, "marketing_spend")
        self.assertEqual(result.entries[1].row.kpi, "revenue")

    def test_the_bigger_favourable_move_ranks_first_under_magnitude(self):
        result = self.se.rank_kpis_by_movement(self.df, self.period, order_by="magnitude")
        self.assertEqual(result.entries[0].row.kpi, "revenue")
        self.assertEqual(result.entries[1].row.kpi, "marketing_spend")

    def test_an_invalid_order_by_is_rejected(self):
        with self.assertRaises(InvalidArgumentError):
            self.se.rank_kpis_by_movement(self.df, self.period, order_by="nonsense")


# ---------------------------------------------------------------------------
# scan_anomalies()
# ---------------------------------------------------------------------------
class TestEmptyScanIsHonest(unittest.TestCase):
    def test_no_material_movement_returns_an_empty_scan_not_noise(self):
        raw = pd.DataFrame({
            "date": ["2024-01-15", "2024-01-16", "2024-04-15", "2024-04-16"],
            "revenue": [1000.0, 1000.0, 1000.0, 1000.0],
        })
        schema = metrics.detect_schema(raw)
        df = metrics.prepare(raw, schema)
        se = ScanEngine(ContractAPI(schema))
        anomalies = se.scan_anomalies(df, {"type": "quarter", "year": 2024, "quarter": 2})
        self.assertEqual(anomalies.status, "ok")
        self.assertEqual(anomalies.rows, ())


class TestScanAnomaliesComposesTheGateCorrectly(ScanTestCase):
    """`scan_anomalies` must flag exactly what `QualityEngine.filter_material`
    -- called independently on the same scan -- would retain, up to the
    follow-up cap. This is the same "no adapter" composition claim O5 was
    built on, extended to a batched caller."""

    def test_flagged_kpis_are_a_subset_of_what_filter_material_retains(self):
        period = {"type": "quarter", "year": 2026, "quarter": 2}
        scan = self.se.scan_kpis(self.rdf, period)
        candidates = [{"kpi": r.kpi, "change_pct": r.change_pct, "robust_z": r.robust_z}
                     for r in scan.rows]
        gate = QualityEngine(self.api).filter_material(candidates)
        expected_material = {c.kpi for c in gate.candidates if c.is_material}

        anomalies = self.se.scan_anomalies(self.rdf, period)
        flagged = {r.kpi for r in anomalies.rows}
        self.assertTrue(flagged <= expected_material)
        if len(expected_material) <= MAX_ANOMALY_FOLLOWUPS:
            self.assertEqual(flagged, expected_material)
        else:
            self.assertTrue(anomalies.followups_truncated)
            self.assertEqual(len(flagged), MAX_ANOMALY_FOLLOWUPS)


# ---------------------------------------------------------------------------
# scan_dimension_outliers()
# ---------------------------------------------------------------------------
class TestDimensionOutliersRecoverThePlantedLoci(ScanTestCase):
    def test_product_a_is_the_top_outlier_on_units_sold_by_product(self):
        result = self.se.scan_dimension_outliers(self.rdf, "units_sold", "product", {"type": "all"})
        self.assertEqual(result.entries[0].member, "Product A")
        self.assertLess(result.entries[0].robust_z, -2.0)

    def test_north_is_the_top_outlier_on_units_sold_by_region(self):
        result = self.se.scan_dimension_outliers(self.rdf, "units_sold", "region", {"type": "all"})
        self.assertEqual(result.entries[0].member, "North")
        self.assertLess(result.entries[0].robust_z, -2.0)


class TestDimensionOutliersInsufficientHistoryIsHonest(unittest.TestCase):
    def test_a_member_with_too_few_periods_is_insufficient_not_a_fabricated_z(self):
        raw = pd.DataFrame({
            "date": ["2024-01-15", "2024-04-15"],
            "region": ["North", "North"],
            "revenue": [100.0, 110.0],
        })
        schema = metrics.detect_schema(raw)
        df = metrics.prepare(raw, schema)
        se = ScanEngine(ContractAPI(schema))
        result = se.scan_dimension_outliers(df, "revenue", "region", {"type": "all"})
        self.assertEqual(result.entries[0].status, "insufficient")
        self.assertIsNone(result.entries[0].robust_z)


class TestAirlock(ScanTestCase):
    def test_a_time_grain_column_is_rejected_as_an_outlier_scan_dimension(self):
        with self.assertRaises(InvalidArgumentError):
            self.se.scan_dimension_outliers(self.rdf, "revenue", "_period", {"type": "all"})


# ---------------------------------------------------------------------------
# integrity / prose guard / determinism / portability
# ---------------------------------------------------------------------------
class TestNumericIntegrityAndProseGuard(ScanTestCase):
    def test_the_payload_survives_json_dumps_with_no_nan_or_inf(self):
        period = {"type": "quarter", "year": 2026, "quarter": 2}
        scan = self.se.scan_kpis(self.rdf, period)
        ranked = self.se.rank_kpis_by_movement(self.rdf, period)
        anomalies = self.se.scan_anomalies(self.rdf, period)
        outliers = self.se.scan_dimension_outliers(self.rdf, "units_sold", "product", {"type": "all"})
        assert_json_safe(scan.to_payload())
        assert_json_safe(ranked.to_payload())
        assert_json_safe(anomalies.to_payload())
        assert_json_safe(outliers.to_payload())

    def test_no_string_field_outside_the_allowlist_looks_like_prose(self):
        period = {"type": "quarter", "year": 2026, "quarter": 2}
        scan = self.se.scan_kpis(self.rdf, period)
        outliers = self.se.scan_dimension_outliers(self.rdf, "units_sold", "product", {"type": "all"})
        assert_no_prose_leak(scan.to_payload())
        assert_no_prose_leak(outliers.to_payload())


class TestDeterminism(ScanTestCase):
    def test_identical_input_gives_identical_output_across_runs(self):
        period = {"type": "quarter", "year": 2026, "quarter": 2}
        first = self.se.scan_kpis(self.rdf, period).to_payload()
        second = self.se.scan_kpis(self.rdf, period).to_payload()
        self.assertEqual(first, second)


class TestPortabilityAcrossDatasets(unittest.TestCase):
    def test_scan_kpis_works_on_the_hospital_fixture(self):
        df, schema = contracted("hospital_sample.csv", FIXTURES)
        se = ScanEngine(ContractAPI(schema))
        scan = se.scan_kpis(df, {"type": "quarter", "year": 2025, "quarter": 2})
        self.assertEqual(scan.status, "ok")
        self.assertTrue(scan.rows)

    def test_scan_dimension_outliers_on_a_single_member_dimension_does_not_crash(self):
        """`school_kpi_smoke_sample.csv` has `school` at cardinality 1 --
        exactly one entry, scored on that member's own 14-quarter history."""
        df, schema = contracted("school_kpi_smoke_sample.csv")
        assert schema.dimensions[0] == "school"
        se = ScanEngine(ContractAPI(schema))
        result = se.scan_dimension_outliers(df, "students_enrolled", "school", {"type": "all"})
        self.assertEqual(len(result.entries), 1)
        self.assertEqual(result.entries[0].member, "Northstar Public School")


if __name__ == "__main__":
    unittest.main()
