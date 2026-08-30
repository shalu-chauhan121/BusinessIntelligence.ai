"""
Arbitrary A/B comparison, significance judgment, and a no-comparison control
band -- `agent/compare.py`.

`observe.observe` (`engines/observe.py:394`) can only compare a period to
`tf.previous()` or `tf.year_ago()` -- that single line is the entire
comparison vocabulary the product has ever had. `ComparisonEngine.compare` has
no such restriction; the tests that matter most for it exercise a genuinely
non-adjacent pair, the case with no code path today.

`ComparisonEngine.significance` is a narrower wrap on purpose --
`observe.assess_significance` only has a history distribution for the
previous-period and year-over-year transitions -- so its most important test
is the regression suite: every retained field, for every quarter and both
modes, must match `observe.assess_significance` exactly. That is the guard
that stripping the three prose fields (`history_note`, `statistical_power`,
`dispersion_note`) into `history_status` / `power` / `sigma_floored` changed
nothing about the numbers underneath.
"""
from __future__ import annotations

import unittest

from app.agent.compare import ComparisonEngine
from app.agent.contract_api import ContractAPI
from app.agent.errors import EmptyPeriodError, InvalidArgumentError
from app.agent.series import SeriesEngine
from app.engines import observe
from app.engines.drivers import robust_sigma
from app.engines.metrics import compute as legacy_compute
from app.engines.metrics import pct_change

from .base import (EngineTestCase, FIXTURES, assert_json_safe,
                   assert_no_prose_leak, contracted)


class ComparisonTestCase(EngineTestCase):
    """The shared retail fixture, with a compiled contract so ratio KPIs are
    comparable."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.rdf, cls.rschema = contracted("business_metrics_sample.csv")
        cls.api = ContractAPI(cls.rschema)
        cls.ce = ComparisonEngine(cls.api)

    def _assert_matches_legacy(self, result, raw) -> None:
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.method, raw["method"])
        self.assertEqual(result.history_status, raw["history_status"])
        self.assertEqual(result.comparison, raw["comparison"])
        self.assertEqual(result.current_period, raw["current_period"])
        self.assertEqual(result.baseline_period, raw["baseline_period"])
        for field in ("change_pct", "robust_z", "median_historical_change_pct",
                      "robust_sigma_pct", "median_all_history_pct",
                      "sigma_all_history_pct", "z_threshold",
                      "material_threshold_pct", "expected_value"):
            self.assertEqual(getattr(result, field), raw[field], field)
        self.assertEqual(result.history_points, raw["history_points"])
        self.assertEqual(result.same_quarter_points, raw["same_quarter_points"])
        self.assertEqual(result.is_material, raw["is_material"])
        self.assertEqual(result.is_statistically_unusual, raw["is_statistically_unusual"])
        self.assertEqual(result.is_anomaly, raw["is_anomaly"])
        self.assertEqual(result.verdict, raw["verdict"])
        expected_band = tuple(raw["normal_range"]) if raw["normal_range"] else None
        self.assertEqual(result.normal_range, expected_band)
        self.assertEqual(list(result.historical_changes), raw["historical_changes"])
        self.assertEqual(result.sigma_floored, raw["dispersion_note"] is not None)


# ---------------------------------------------------------------------------
# compare()
# ---------------------------------------------------------------------------
class TestCompareHandlesArbitraryPairs(ComparisonTestCase):
    def test_a_non_adjacent_pair_matches_a_hand_built_two_slice_computation(self):
        """'Q4 2026 vs Q1 2024' -- no code path in `observe.observe` today."""
        result = self.ce.compare(
            self.rdf, "revenue",
            {"type": "quarter", "year": 2026, "quarter": 1},
            {"type": "quarter", "year": 2023, "quarter": 3})

        cur_mask = (self.rdf["_year"] == 2026) & (self.rdf["_quarter"] == 1)
        base_mask = (self.rdf["_year"] == 2023) & (self.rdf["_quarter"] == 3)
        expected_cur = legacy_compute(self.rdf[cur_mask], "revenue", self.rschema.contract_resolver)
        expected_base = legacy_compute(self.rdf[base_mask], "revenue", self.rschema.contract_resolver)

        self.assertAlmostEqual(result.current.value, expected_cur, places=6)
        self.assertAlmostEqual(result.baseline.value, expected_base, places=6)
        self.assertAlmostEqual(result.change_abs, expected_cur - expected_base, places=6)
        self.assertAlmostEqual(result.change_pct, pct_change(expected_cur, expected_base), places=6)

    def test_is_unfavourable_respects_kpi_polarity(self):
        """`marketing_spend` is `higher_is_better=False` -- a rise must be
        reported unfavourable, a fall favourable."""
        result = self.ce.compare(
            self.rdf, "marketing_spend",
            {"type": "quarter", "year": 2026, "quarter": 1},
            {"type": "quarter", "year": 2023, "quarter": 1})
        self.assertIsNotNone(result.direction)
        self.assertNotEqual(result.direction, "flat")
        self.assertEqual(result.is_unfavourable, result.direction == "up")

    def test_a_filtered_comparison_equals_the_same_comparison_on_a_hand_filtered_frame(self):
        result = self.ce.compare(
            self.rdf, "revenue",
            {"type": "quarter", "year": 2026, "quarter": 1},
            {"type": "quarter", "year": 2023, "quarter": 3},
            filters={"region": "North"})

        north = self.rdf[self.rdf["region"] == "North"]
        cur_mask = (north["_year"] == 2026) & (north["_quarter"] == 1)
        base_mask = (north["_year"] == 2023) & (north["_quarter"] == 3)
        expected_cur = legacy_compute(north[cur_mask], "revenue", self.rschema.contract_resolver)
        expected_base = legacy_compute(north[base_mask], "revenue", self.rschema.contract_resolver)

        self.assertAlmostEqual(result.current.value, expected_cur, places=6)
        self.assertAlmostEqual(result.baseline.value, expected_base, places=6)

    def test_a_period_the_dataset_does_not_hold_raises_empty_period_error(self):
        with self.assertRaises(EmptyPeriodError):
            self.ce.compare(
                self.rdf, "revenue",
                {"type": "quarter", "year": 2026, "quarter": 1},
                {"type": "year", "year": 2099})


# ---------------------------------------------------------------------------
# significance() -- regression against observe.assess_significance
# ---------------------------------------------------------------------------
class TestSignificanceRegression(ComparisonTestCase):
    def test_every_quarter_and_mode_matches_legacy_assess_significance_exactly(self):
        legacy_series = observe.quarterly_series(self.rdf, "revenue", self.rschema.contract_resolver)

        for row in legacy_series:
            tf = observe.Timeframe(row["year"], row["quarter"])
            period_a = {"type": "quarter", "year": tf.year, "quarter": tf.quarter}

            with self.subTest(period=tf.label, mode="previous_period"):
                raw = observe.assess_significance(legacy_series, tf, "previous_period")
                result = self.ce.significance(self.rdf, "revenue", period_a)
                self._assert_matches_legacy(result, raw)

            year_ago = tf.year_ago()
            with self.subTest(period=tf.label, mode="year_over_year"):
                raw = observe.assess_significance(legacy_series, tf, "year_over_year")
                period_b = {"type": "quarter", "year": year_ago.year, "quarter": year_ago.quarter}
                result = self.ce.significance(self.rdf, "revenue", period_a, period_b)
                self._assert_matches_legacy(result, raw)


class TestSignificanceUnsupportedBaseline(ComparisonTestCase):
    def test_a_non_adjacent_non_year_ago_pair_is_unsupported(self):
        result = self.ce.significance(
            self.rdf, "revenue",
            {"type": "quarter", "year": 2026, "quarter": 2},
            {"type": "quarter", "year": 2025, "quarter": 3})
        self.assertEqual(result.status, "unsupported_baseline")
        self.assertIsNone(result.robust_z)
        self.assertEqual(result.supported_baselines, ("previous_period", "year_over_year"))
        payload = result.to_payload()
        self.assertIn("requested_baseline", payload)
        self.assertEqual(payload["supported_baselines"], ["previous_period", "year_over_year"])


class TestSignificanceRejectsANonQuarterPeriodA(ComparisonTestCase):
    def test_a_full_year_period_a_is_rejected(self):
        """A full year never appears in the quarterly series `assess_significance`
        consumes -- silently answering `newly_launched` would be wrong, not honest."""
        with self.assertRaises(InvalidArgumentError):
            self.ce.significance(self.rdf, "revenue", {"type": "year", "year": 2024})


class TestThinHistoryIsHonestNotFabricated(ComparisonTestCase):
    """G8: a statistical tool must return a typed status, never a fabricated
    number, when there is not enough history."""

    def test_sparse_history_reports_a_typed_verdict_with_no_z_score(self):
        result = self.ce.significance(self.rdf, "revenue", {"type": "quarter", "year": 2023, "quarter": 2})
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.history_status, "sparse_history")
        self.assertEqual(result.verdict, "sparse_history")
        self.assertIsNone(result.robust_z)


class TestPowerReproducesAllFourBranches(ComparisonTestCase):
    def test_none_when_newly_launched(self):
        result = self.ce.significance(self.rdf, "revenue", {"type": "quarter", "year": 2023, "quarter": 1})
        self.assertEqual(result.history_status, "newly_launched")
        self.assertEqual(result.power, "none")

    def test_weak_when_sparse(self):
        result = self.ce.significance(self.rdf, "revenue", {"type": "quarter", "year": 2023, "quarter": 2})
        self.assertEqual(result.history_status, "sparse_history")
        self.assertEqual(result.power, "weak")

    def test_moderate_when_sufficient_but_not_seasonal(self):
        result = self.ce.significance(self.rdf, "revenue", {"type": "quarter", "year": 2024, "quarter": 2})
        self.assertEqual(result.method, "robust_z")
        self.assertEqual(result.power, "moderate")

    def test_good_and_floored_on_the_seasonal_quarter(self):
        result = self.ce.significance(self.rdf, "revenue", {"type": "quarter", "year": 2026, "quarter": 2})
        self.assertEqual(result.method, "seasonal_robust_z")
        self.assertEqual(result.power, "good")
        self.assertTrue(result.sigma_floored)


class TestSignificanceProseGuard(ComparisonTestCase):
    BANNED = {"history_note", "statistical_power", "dispersion_note", "rule"}

    def _assert_no_banned_keys(self, payload, path: str = "$") -> None:
        if isinstance(payload, dict):
            for key, value in payload.items():
                self.assertNotIn(key, self.BANNED, f"{path}.{key} is a banned prose field")
                self._assert_no_banned_keys(value, f"{path}.{key}")
        elif isinstance(payload, list):
            for i, item in enumerate(payload):
                self._assert_no_banned_keys(item, f"{path}[{i}]")

    def test_no_banned_prose_key_survives_in_a_significance_payload(self):
        result = self.ce.significance(self.rdf, "revenue", {"type": "quarter", "year": 2026, "quarter": 2})
        self._assert_no_banned_keys(result.to_payload())

    def test_no_string_field_outside_the_allowlist_looks_like_prose(self):
        result = self.ce.significance(self.rdf, "revenue", {"type": "quarter", "year": 2026, "quarter": 2})
        assert_no_prose_leak(result.to_payload())

    def test_the_payload_survives_json_dumps_with_no_nan_or_inf(self):
        result = self.ce.significance(self.rdf, "revenue", {"type": "quarter", "year": 2023, "quarter": 1})
        assert_json_safe(result.to_payload())


# ---------------------------------------------------------------------------
# normal_range()
# ---------------------------------------------------------------------------
class TestNormalRange(ComparisonTestCase):
    def test_median_and_sigma_match_robust_sigma_over_the_same_series(self):
        result = self.ce.normal_range(self.rdf, "revenue", grain="week")
        series = SeriesEngine(self.api).series(self.rdf, "revenue", grain="week")
        values = [p.value for p in series.points if p.value == p.value]
        med, sigma = robust_sigma(values)
        self.assertAlmostEqual(result.level.median, med, places=6)
        self.assertAlmostEqual(result.level.sigma, sigma, places=6)

    def test_the_band_is_symmetric_about_the_median(self):
        result = self.ce.normal_range(self.rdf, "revenue", grain="week")
        self.assertAlmostEqual(result.level.median - result.level.lower,
                               result.level.upper - result.level.median, places=6)

    def test_both_level_and_change_bands_are_present_on_a_healthy_series(self):
        result = self.ce.normal_range(self.rdf, "revenue", grain="week")
        self.assertEqual(result.status, "ok")
        self.assertIsNotNone(result.level)
        self.assertIsNotNone(result.change)

    def test_fewer_points_than_min_points_is_reported_as_insufficient(self):
        result = self.ce.normal_range(self.rdf, "revenue", grain="year", min_points=100)
        self.assertEqual(result.status, "insufficient_history")
        self.assertIsNone(result.level)
        self.assertIsNone(result.change)

    def test_the_grain_argument_is_honoured(self):
        weekly = self.ce.normal_range(self.rdf, "revenue", grain="week")
        monthly = self.ce.normal_range(self.rdf, "revenue", grain="month")
        self.assertEqual(weekly.grain, "week")
        self.assertEqual(monthly.grain, "month")
        self.assertNotEqual(weekly.points, monthly.points)

    def test_an_unknown_grain_is_rejected(self):
        with self.assertRaises(InvalidArgumentError):
            self.ce.normal_range(self.rdf, "revenue", grain="fortnight")


# ---------------------------------------------------------------------------
# portability
# ---------------------------------------------------------------------------
class TestPortabilityAcrossDatasets(unittest.TestCase):
    """Catches hardcoded retail vocabulary the way the hospital fixture caught
    it for the resolver-propagation bug (G3)."""

    @classmethod
    def setUpClass(cls):
        cls.df, cls.schema = contracted("hospital_sample.csv", FIXTURES)
        cls.ce = ComparisonEngine(ContractAPI(cls.schema))
        assert "recovery_rate" in cls.schema.available_kpis

    def test_compare_and_significance_work_on_a_contract_only_ratio_kpi(self):
        comparison = self.ce.compare(
            self.df, "recovery_rate",
            {"type": "quarter", "year": 2025, "quarter": 2},
            {"type": "quarter", "year": 2025, "quarter": 1})
        self.assertIsNotNone(comparison.current.value)

        sig = self.ce.significance(self.df, "recovery_rate", {"type": "quarter", "year": 2025, "quarter": 2})
        self.assertEqual(sig.status, "ok")


if __name__ == "__main__":
    unittest.main()
