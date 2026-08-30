"""
`detect_trend` / `detect_changepoint` -- `agent/trend.py`.

`detect_trend` is genuinely new maths (no trend fitting exists anywhere in
`backend/app`), fit with Theil-Sen rather than least squares specifically so
one outlier cannot flip the verdict -- `TestOutlierRobustness` proves that
choice by showing `numpy.polyfit` *does* flip on the same series, not just by
exercising the estimator that doesn't.

`detect_changepoint` wraps `analysis.detect_onset`, which collapses three
different "nothing to report" reasons into the same bare `None`. The tests
that matter most here construct a fixture for each one individually so the
wrapper's typed `status` can be checked against the right answer, not just
against "not ok".

The retail sample has a planted three-factor scenario
(`sample_data/ground_truth.json`) with exact onset dates and loci --
`supply_disruption_product_a` (2026-05-04, product Product A) and
`north_demand_erosion` (2026-04-06, region North). `TestChangepointAgainstGroundTruth`
asserts `detect_changepoint` recovers those dates from the real data, which is
a far stronger guarantee than a synthetic fixture alone.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from app.agent.contract_api import ContractAPI
from app.agent.errors import InvalidArgumentError
from app.agent.series import SeriesEngine
from app.agent.trend import TrendEngine
from app.engines import metrics

from .base import (EngineTestCase, FIXTURES, assert_json_safe,
                   assert_no_prose_leak, contracted)

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
TRUTH_PATH = ROOT / "sample_data" / "ground_truth.json"


def _quarterly_frame(values, column: str = "revenue"):
    """A minimal contract-less frame with one row per quarter, in order,
    starting 2020-Q1 -- consecutive quarters, so ordinal spacing is 1 and a
    hand-computed pairwise-slope check can use plain list position as `x`."""
    dates = []
    month = {1: 2, 2: 5, 3: 8, 4: 11}
    for i in range(len(values)):
        q, yr = i % 4 + 1, 2020 + i // 4
        dates.append(f"{yr}-{month[q]:02d}-15")
    raw = pd.DataFrame({"date": dates, column: values})
    schema = metrics.detect_schema(raw)
    df = metrics.prepare(raw, schema)
    return df, schema


def _weekly_frame(values, column: str = "revenue", start: str = "2024-01-01"):
    dates = pd.date_range(start, periods=len(values), freq="7D").astype(str)
    raw = pd.DataFrame({"date": dates, column: values})
    schema = metrics.detect_schema(raw)
    df = metrics.prepare(raw, schema)
    return df, schema


def _planted_factor(factor_id: str):
    truth = json.loads(TRUTH_PATH.read_text())
    return next(f for f in truth["planted_factors"] if f["id"] == factor_id)


class TrendTestCase(EngineTestCase):
    """The shared retail fixture, with a compiled contract."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.rdf, cls.rschema = contracted("business_metrics_sample.csv")
        cls.api = ContractAPI(cls.rschema)
        cls.te = TrendEngine(cls.api)


# ---------------------------------------------------------------------------
# detect_trend() -- classification
# ---------------------------------------------------------------------------
class TestSyntheticSeriesClassifyCorrectly(unittest.TestCase):
    def test_a_monotonic_rising_series_is_rising_and_improving(self):
        df, schema = _quarterly_frame([100 + 10 * i for i in range(10)])
        result = TrendEngine(ContractAPI(schema)).detect_trend(df, "revenue", grain="quarter")
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.direction, "rising")
        self.assertGreater(result.slope_per_period, 0)
        self.assertTrue(result.is_improving)   # revenue: higher_is_better

    def test_a_monotonic_falling_series_is_falling_and_not_improving(self):
        df, schema = _quarterly_frame([200 - 10 * i for i in range(10)])
        result = TrendEngine(ContractAPI(schema)).detect_trend(df, "revenue", grain="quarter")
        self.assertEqual(result.direction, "falling")
        self.assertLess(result.slope_per_period, 0)
        self.assertFalse(result.is_improving)

    def test_a_flat_series_is_flat_with_no_improvement_verdict(self):
        df, schema = _quarterly_frame([100.0] * 10)
        result = TrendEngine(ContractAPI(schema)).detect_trend(df, "revenue", grain="quarter")
        self.assertEqual(result.direction, "flat")
        self.assertEqual(result.slope_per_period, 0.0)
        self.assertIsNone(result.is_improving)


class TestOutlierRobustness(unittest.TestCase):
    def test_one_injected_outlier_does_not_flip_the_verdict(self):
        """A clearly falling series, with one point near the end spiked far
        above everything else."""
        values = [100 - 5 * i for i in range(12)]
        values[-1] = 2000.0
        df, schema = _quarterly_frame(values)

        result = TrendEngine(ContractAPI(schema)).detect_trend(df, "revenue", grain="quarter")
        self.assertEqual(result.direction, "falling")
        self.assertLess(result.slope_per_period, 0)

        # Least squares, on the identical data, DOES flip -- proving the
        # estimator choice, not just exercising it.
        ols_slope, _ = np.polyfit(np.arange(len(values), dtype=float), values, 1)
        self.assertGreater(ols_slope, 0)


class TestTheilSenMatchesHandComputation(unittest.TestCase):
    def test_slope_is_the_median_of_every_pairwise_slope(self):
        values = [10, 12, 11, 15, 20, 18, 24]
        df, schema = _quarterly_frame(values)
        result = TrendEngine(ContractAPI(schema)).detect_trend(
            df, "revenue", grain="quarter", min_points=3)

        xs = list(range(len(values)))   # consecutive quarters -> ordinal step 1, matches list position
        pairwise = [(values[j] - values[i]) / (xs[j] - xs[i])
                   for i in range(len(values)) for j in range(i + 1, len(values))]
        self.assertAlmostEqual(result.slope_per_period, float(np.median(pairwise)), places=6)


class TestGapStretchesTheXAxis(unittest.TestCase):
    """A perfectly linear series `v(t) = 100 + 10t` evaluated at any subset of
    its true ordinals still has slope 10 under ordinal-based Theil-Sen. A
    naive implementation that used list *position* instead of the calendar
    ordinal would compute a steeper slope once a middle point is missing,
    because it would treat two calendar-quarters-apart points as one step
    apart. This is the test that would fail if `period_ordinal` were dropped
    in favour of `enumerate(points)`."""

    def test_removing_a_middle_quarter_does_not_change_the_slope(self):
        full = [(2020, 1, 100), (2020, 2, 110), (2020, 3, 120),
               (2020, 4, 130), (2021, 1, 140), (2021, 2, 150)]
        gapped = [row for row in full if row[:2] != (2020, 4)]

        def _frame(rows):
            month = {1: 2, 2: 5, 3: 8, 4: 11}
            dates = [f"{yr}-{month[q]:02d}-15" for yr, q, _ in rows]
            raw = pd.DataFrame({"date": dates, "revenue": [v for *_, v in rows]})
            schema = metrics.detect_schema(raw)
            return metrics.prepare(raw, schema), schema

        df_full, schema_full = _frame(full)
        df_gap, schema_gap = _frame(gapped)

        result_full = TrendEngine(ContractAPI(schema_full)).detect_trend(
            df_full, "revenue", grain="quarter", min_points=3)
        result_gap = TrendEngine(ContractAPI(schema_gap)).detect_trend(
            df_gap, "revenue", grain="quarter", min_points=3)

        self.assertAlmostEqual(result_full.slope_per_period, 10.0, places=6)
        self.assertAlmostEqual(result_gap.slope_per_period, 10.0, places=6)


class TestInsufficientHistoryIsHonest(unittest.TestCase):
    def test_fewer_than_min_points_returns_insufficient_with_no_slope(self):
        df, schema = _quarterly_frame([100, 110, 120])
        result = TrendEngine(ContractAPI(schema)).detect_trend(
            df, "revenue", grain="quarter", min_points=5)
        self.assertEqual(result.status, "insufficient")
        self.assertIsNone(result.slope_per_period)
        self.assertIsNone(result.direction)


class TestPolarityAwareImprovement(unittest.TestCase):
    """`marketing_spend` is a seed KPI with `higher_is_better=False`
    (`engines/metrics.py`) -- a raw single-column sum, so a synthetic frame
    can drive it directly."""

    def test_a_falling_lower_is_better_kpi_is_improving(self):
        df, schema = _quarterly_frame([200 - 10 * i for i in range(8)], column="marketing_spend")
        result = TrendEngine(ContractAPI(schema)).detect_trend(df, "marketing_spend", grain="quarter")
        self.assertEqual(result.direction, "falling")
        self.assertTrue(result.is_improving)

    def test_a_rising_lower_is_better_kpi_is_not_improving(self):
        df, schema = _quarterly_frame([100 + 10 * i for i in range(8)], column="marketing_spend")
        result = TrendEngine(ContractAPI(schema)).detect_trend(df, "marketing_spend", grain="quarter")
        self.assertEqual(result.direction, "rising")
        self.assertFalse(result.is_improving)


# ---------------------------------------------------------------------------
# detect_changepoint() -- the four `detect_onset` outcomes, told apart
# ---------------------------------------------------------------------------
class TestChangepointFindsAnInjectedStep(unittest.TestCase):
    def test_a_synthetic_step_is_found_at_the_right_week(self):
        values = [100.0] * 20 + [150.0] * 10
        df, schema = _weekly_frame(values)
        api = ContractAPI(schema)
        expected_period = SeriesEngine(api).series(df, "revenue", grain="week").points[20].period

        result = TrendEngine(api).detect_changepoint(
            df, "revenue", grain="week", direction="up",
            baseline_periods=8, persistence=2, k_sigma=1.0)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.direction, "up")
        self.assertEqual(result.period, expected_period)


class TestChangepointDistinguishesEveryNonAnswer(unittest.TestCase):
    def test_a_short_series_is_insufficient_history(self):
        df, schema = _weekly_frame([100.0, 105.0, 102.0, 110.0, 108.0])
        result = TrendEngine(ContractAPI(schema)).detect_changepoint(
            df, "revenue", grain="week", baseline_periods=8, persistence=2)
        self.assertEqual(result.status, "insufficient_history")

    def test_a_constant_zero_baseline_is_degenerate_dispersion(self):
        """The genuine trap in the wrapped code: `detect_onset` returns the
        same bare `None` for this as it does for 'ran the whole series and
        found nothing' -- the two must not be reported as the same status."""
        values = [0.0] * 12 + [50.0, 60.0, 70.0]
        df, schema = _weekly_frame(values)
        result = TrendEngine(ContractAPI(schema)).detect_changepoint(
            df, "revenue", grain="week", direction="up", baseline_periods=8, persistence=2)
        self.assertEqual(result.status, "degenerate_dispersion")

    def test_a_series_with_no_persistent_breach_is_no_changepoint(self):
        values = [100.0, 101.0] * 10
        df, schema = _weekly_frame(values)
        result = TrendEngine(ContractAPI(schema)).detect_changepoint(
            df, "revenue", grain="week", direction="up", baseline_periods=8, persistence=2)
        self.assertEqual(result.status, "no_changepoint")

    def test_an_invalid_direction_is_rejected_not_silently_treated_as_up(self):
        df, schema = _weekly_frame([100.0] * 15)
        with self.assertRaises(InvalidArgumentError):
            TrendEngine(ContractAPI(schema)).detect_changepoint(
                df, "revenue", grain="week", direction="sideways")


class TestChangepointAgainstGroundTruth(TrendTestCase):
    def test_product_a_units_sold_break_matches_the_planted_onset_exactly(self):
        factor = _planted_factor("supply_disruption_product_a")
        result = self.te.detect_changepoint(
            self.rdf, "units_sold", grain="week", filters={"product": "Product A"},
            direction="down")
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.period, factor["onset_date"])

    def test_north_units_sold_break_is_within_three_weeks_of_the_planted_onset(self):
        factor = _planted_factor("north_demand_erosion")
        result = self.te.detect_changepoint(
            self.rdf, "units_sold", grain="week", filters={"region": "North"},
            direction="down")
        self.assertEqual(result.status, "ok")
        found = pd.Timestamp(result.period)
        planted = pd.Timestamp(factor["onset_date"])
        self.assertLessEqual(abs((found - planted).days), 21)


# ---------------------------------------------------------------------------
# integrity / prose guard / determinism / portability
# ---------------------------------------------------------------------------
class TestNumericIntegrityAndProseGuard(TrendTestCase):
    def test_the_payload_survives_json_dumps_with_no_nan_or_inf(self):
        trend = self.te.detect_trend(self.rdf, "revenue", grain="quarter")
        cp = self.te.detect_changepoint(self.rdf, "revenue", grain="week")
        assert_json_safe(trend.to_payload())
        assert_json_safe(cp.to_payload())

    def test_no_string_field_outside_the_allowlist_looks_like_prose(self):
        trend = self.te.detect_trend(self.rdf, "revenue", grain="quarter")
        cp = self.te.detect_changepoint(self.rdf, "revenue", grain="week")
        assert_no_prose_leak(trend.to_payload())
        assert_no_prose_leak(cp.to_payload())


class TestDeterminism(TrendTestCase):
    def test_identical_input_gives_identical_output_across_runs(self):
        first = self.te.detect_trend(self.rdf, "revenue", grain="quarter").to_payload()
        second = self.te.detect_trend(self.rdf, "revenue", grain="quarter").to_payload()
        self.assertEqual(first, second)


class TestPortabilityAcrossDatasets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.schema = contracted("hospital_sample.csv", FIXTURES)
        cls.te = TrendEngine(ContractAPI(cls.schema))
        assert "recovery_rate" in cls.schema.available_kpis

    def test_trend_and_changepoint_work_on_a_contract_only_ratio_kpi(self):
        trend = self.te.detect_trend(self.df, "recovery_rate", grain="quarter")
        self.assertIn(trend.status, ("ok", "insufficient"))
        cp = self.te.detect_changepoint(self.df, "recovery_rate", grain="week")
        self.assertIn(cp.status, ("ok", "insufficient_history", "degenerate_dispersion", "no_changepoint"))


if __name__ == "__main__":
    unittest.main()
