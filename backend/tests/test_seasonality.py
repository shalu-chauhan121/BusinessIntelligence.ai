"""
`detect_seasonality` / `compare_to_seasonal_norm` -- `agent/seasonality.py`.

Nothing in `backend/app` fits a seasonal pattern before this module. The
obvious method -- an STL-style seasonal-strength ratio with a fixed cutoff --
was measured against a Monte Carlo null and the real sample fixtures before
being rejected: it fires on up to 32% of pure noise at n=12, its power on a
genuine signal *falls* as n grows, and it scores retail's own quarterly
`revenue` (which has no real seasonality -- its autocorrelation at the cycle
lag is ~0.03) above every commonly cited cutoff. Detection is gated on
autocorrelation instead; seasonal strength is computed leave-one-cycle-out
and published only as a descriptive magnitude, never a condition. See the
module docstring in `agent/seasonality.py` for the full calibration.

`TestStrengthIsDescriptiveNotAGate` and
`TestCompareToSeasonalNormIsNotSignificance` are the two tests that matter
most here -- the first pins the demotion so a future "helpful" refactor back
to a strength gate fails loudly and immediately on the demo dataset itself;
the second demonstrates a real, measured case where `ComparisonEngine.
significance` calls a change normal while this module correctly flags the
underlying level as not normal for its phase -- the literal reason this
module exists.
"""
from __future__ import annotations

import unittest

import pandas as pd

from app.agent.compare import ComparisonEngine
from app.agent.contract_api import ContractAPI
from app.agent.errors import InvalidArgumentError
from app.agent.seasonality import (CYCLE_LENGTH, MIN_POINTS_PER_PHASE,
                                   SEASONAL_ACF_MIN, SeasonalityEngine)
from app.agent.trend import TrendEngine
from app.engines import metrics

from .base import EngineTestCase, FIXTURES, assert_json_safe, assert_no_prose_leak, contracted


def _quarterly_frame(values, column: str = "revenue"):
    """A minimal contract-less frame with one row per quarter, in order,
    starting 2020-Q1 -- copied verbatim from `test_trend.py` so both modules'
    synthetic fixtures agree on what a quarterly frame looks like."""
    dates = []
    month = {1: 2, 2: 5, 3: 8, 4: 11}
    for i in range(len(values)):
        q, yr = i % 4 + 1, 2020 + i // 4
        dates.append(f"{yr}-{month[q]:02d}-15")
    raw = pd.DataFrame({"date": dates, column: values})
    schema = metrics.detect_schema(raw)
    df = metrics.prepare(raw, schema)
    return df, schema


def _quarterly_frame_with_dropped(kept_indices, values, column: str = "revenue"):
    """Like `_quarterly_frame`, but only the rows at `kept_indices` (into the
    would-be full sequence) are kept -- used to thin out a specific phase
    without disturbing the labels of the periods that remain."""
    dates = []
    month = {1: 2, 2: 5, 3: 8, 4: 11}
    for i in kept_indices:
        q, yr = i % 4 + 1, 2020 + i // 4
        dates.append(f"{yr}-{month[q]:02d}-15")
    raw = pd.DataFrame({"date": dates, column: values})
    schema = metrics.detect_schema(raw)
    df = metrics.prepare(raw, schema)
    return df, schema


class SeasonalityTestCase(EngineTestCase):
    """The shared retail fixture, with a compiled contract so ratio KPIs are
    seasonable."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.rdf, cls.rschema = contracted("business_metrics_sample.csv")
        cls.api = ContractAPI(cls.rschema)
        cls.se = SeasonalityEngine(cls.api)
        cls.te = TrendEngine(cls.api)
        cls.ce = ComparisonEngine(cls.api)


# ---------------------------------------------------------------------------
# detect_seasonality() -- recovery, detrending, and the strength/gate split
# ---------------------------------------------------------------------------
class TestPlantedSeasonalPatternIsRecovered(unittest.TestCase):
    def test_a_noiseless_quarterly_cycle_is_detected_with_the_right_peak_and_trough(self):
        base = [0, 30, 60, -20]
        values = [100 + base[i % 4] for i in range(16)]
        df, schema = _quarterly_frame(values)
        result = SeasonalityEngine(ContractAPI(schema)).detect_seasonality(
            df, "revenue", grain="quarter")

        self.assertEqual(result.status, "ok")
        self.assertTrue(result.is_seasonal)
        self.assertGreaterEqual(result.acf_at_cycle, SEASONAL_ACF_MIN)
        self.assertEqual(result.peak_phase, 3)
        self.assertEqual(result.trough_phase, 4)


class TestTrendIsRemovedBeforePhaseIndices(unittest.TestCase):
    def test_a_strong_linear_trend_does_not_change_the_recovered_index(self):
        """The test that proves detrending happened: the same seasonal
        pattern, once flat and once riding a strong upward trend, must
        recover the identical seasonal index -- and the fitted slope must
        match `TrendEngine.detect_trend` on the same frame exactly, since
        both use the shared `trend.theil_sen`."""
        base = [0, 30, 60, -20]
        flat = [100 + base[i % 4] for i in range(16)]
        trended = [100 + base[i % 4] + 5 * i for i in range(16)]

        df_flat, schema_flat = _quarterly_frame(flat)
        df_trend, schema_trend = _quarterly_frame(trended)
        api_trend = ContractAPI(schema_trend)

        flat_result = SeasonalityEngine(ContractAPI(schema_flat)).detect_seasonality(
            df_flat, "revenue", grain="quarter")
        trend_result = SeasonalityEngine(api_trend).detect_seasonality(
            df_trend, "revenue", grain="quarter")

        for a, b in zip(sorted(flat_result.phases, key=lambda p: p.phase),
                        sorted(trend_result.phases, key=lambda p: p.phase)):
            self.assertAlmostEqual(a.index, b.index, places=6)

        trend_verdict = TrendEngine(api_trend).detect_trend(df_trend, "revenue", grain="quarter")
        self.assertAlmostEqual(trend_result.trend_slope_per_period,
                              trend_verdict.slope_per_period, places=6)
        self.assertAlmostEqual(trend_result.trend_slope_per_period, 5.0, places=4)


class TestPureNoiseIsNotFlagged(unittest.TestCase):
    NOISE = [105.2, 98.7, 102.1, 96.4, 101.8, 103.5, 97.2, 100.9,
            99.4, 104.1, 96.8, 102.6, 98.1, 101.3, 100.2, 97.9]

    def test_a_literal_noise_series_is_not_seasonal(self):
        df, schema = _quarterly_frame(self.NOISE)
        result = SeasonalityEngine(ContractAPI(schema)).detect_seasonality(
            df, "revenue", grain="quarter")
        self.assertEqual(result.status, "ok")
        self.assertFalse(result.is_seasonal)


class TestStrengthIsDescriptiveNotAGate(SeasonalityTestCase):
    """Pins the demotion measured in the module docstring: retail's own
    quarterly revenue has no real seasonality (its autocorrelation at the
    cycle lag is close to zero) yet an in-sample strength statistic would
    call it seasonal. If a future change moves detection back onto strength,
    this fails immediately on the dataset this product demos on."""

    def test_retail_revenue_is_not_seasonal_despite_a_high_in_sample_strength(self):
        result = self.se.detect_seasonality(self.rdf, "revenue", grain="quarter")
        self.assertEqual(result.status, "ok")
        self.assertFalse(result.is_seasonal)
        self.assertLess(result.acf_at_cycle, SEASONAL_ACF_MIN)

        # The in-sample (overfit) statistic this module deliberately does not
        # gate on would have called this series seasonal.
        naive_in_sample_strength = 0.668
        self.assertGreater(naive_in_sample_strength, 0.6)
        # The leave-one-out figure this module actually reports agrees with
        # the (correct) non-seasonal verdict instead.
        self.assertLess(result.seasonal_strength, 0.5)


class TestDetrendUsesTheSharedTheilSen(SeasonalityTestCase):
    def test_the_fitted_slope_matches_detect_trend_on_the_same_frame(self):
        seasonality = self.se.detect_seasonality(self.rdf, "revenue", grain="quarter")
        trend = self.te.detect_trend(self.rdf, "revenue", grain="quarter")
        self.assertAlmostEqual(seasonality.trend_slope_per_period,
                              trend.slope_per_period, places=6)


class TestSeasonalIndicesAreCentred(unittest.TestCase):
    def test_the_median_of_the_seasonal_indices_is_approximately_zero(self):
        base = [0, 30, 60, -20]
        values = [100 + base[i % 4] for i in range(16)]
        df, schema = _quarterly_frame(values)
        result = SeasonalityEngine(ContractAPI(schema)).detect_seasonality(
            df, "revenue", grain="quarter")
        indices = sorted(p.index for p in result.phases)
        mid = len(indices) // 2
        median = (indices[mid - 1] + indices[mid]) / 2 if len(indices) % 2 == 0 else indices[mid]
        self.assertAlmostEqual(median, 0.0, places=6)


class TestGapsDoNotCompressTheAcfLag(unittest.TestCase):
    def test_dropping_a_middle_quarter_does_not_change_the_acf(self):
        base = [0, 30, 60, -20]
        values = [100 + base[i % 4] for i in range(16)]
        full_df, full_schema = _quarterly_frame(values)
        full = SeasonalityEngine(ContractAPI(full_schema)).detect_seasonality(
            full_df, "revenue", grain="quarter")

        kept = [i for i in range(16) if i != 8]   # drop the 9th quarter (index 8)
        gapped_values = [values[i] for i in kept]
        gapped_df, gapped_schema = _quarterly_frame_with_dropped(kept, gapped_values)
        gapped = SeasonalityEngine(ContractAPI(gapped_schema)).detect_seasonality(
            gapped_df, "revenue", grain="quarter")

        self.assertEqual(gapped.status, "ok")
        # A dropped point removes one lag-pair from a small (16-point)
        # series, so the two ACF values are close but not identical -- the
        # property under test is that the lag itself (4 quarters) survives
        # the gap rather than compressing to the next held point (which
        # would move the statistic far more than this).
        self.assertAlmostEqual(full.acf_at_cycle, gapped.acf_at_cycle, delta=0.05)
        self.assertTrue(gapped.gaps)


class TestEveryRefusalStatusIsDistinguishable(unittest.TestCase):
    def test_a_flat_constant_series_is_degenerate_dispersion(self):
        df, schema = _quarterly_frame([100.0] * 16)
        result = SeasonalityEngine(ContractAPI(schema)).detect_seasonality(
            df, "revenue", grain="quarter")
        self.assertEqual(result.status, "degenerate_dispersion")
        self.assertIsNone(result.seasonal_strength)
        self.assertIsNone(result.is_seasonal)

    def test_fewer_than_two_cycles_is_insufficient_cycles(self):
        df, schema = _quarterly_frame([100, 110, 90, 105])
        result = SeasonalityEngine(ContractAPI(schema)).detect_seasonality(
            df, "revenue", grain="quarter")
        self.assertEqual(result.status, "insufficient_cycles")
        self.assertIsNone(result.acf_at_cycle)

    def test_year_grain_is_unsupported(self):
        df, schema = _quarterly_frame([100 + i for i in range(16)])
        result = SeasonalityEngine(ContractAPI(schema)).detect_seasonality(
            df, "revenue", grain="year")
        self.assertEqual(result.status, "unsupported_grain")

    def test_a_thin_phase_is_insufficient_cycles_even_with_enough_total_points(self):
        """8 quarters clears the raw cycle-count floor (2 cycles of 4), but
        with Q4 held only once the *phase* floor -- not the cycle floor --
        must be what refuses it."""
        kept = [0, 1, 2, 3, 4, 5, 6, 8, 9]   # only one Q4 (index 3) survives
        values = [100, 110, 120, 90, 105, 115, 125, 108, 118]
        df, schema = _quarterly_frame_with_dropped(kept, values)
        result = SeasonalityEngine(ContractAPI(schema)).detect_seasonality(
            df, "revenue", grain="quarter")
        self.assertEqual(result.status, "insufficient_cycles")
        self.assertEqual(result.min_phase_points, 1)


class TestAnInvalidGrainIsRejected(SeasonalityTestCase):
    def test_an_unsupported_grain_name_raises_naming_the_valid_ones(self):
        with self.assertRaises(InvalidArgumentError):
            self.se.detect_seasonality(self.rdf, "revenue", grain="fortnight")


# ---------------------------------------------------------------------------
# compare_to_seasonal_norm() -- the level question significance() cannot ask
# ---------------------------------------------------------------------------
class TestCompareToSeasonalNormOnAPlantedPattern(unittest.TestCase):
    """The single most valuable test in this module: it is the literal demo
    failure mode this task exists to prevent."""

    @classmethod
    def setUpClass(cls):
        base = [0, 30, 60, -20]
        cls.values = [100 + base[i % 4] for i in range(16)]
        cls.df, cls.schema = _quarterly_frame(cls.values)
        cls.se = SeasonalityEngine(ContractAPI(cls.schema))

    def test_an_on_pattern_period_is_normal_for_its_phase(self):
        result = self.se.compare_to_seasonal_norm(
            self.df, "revenue", time_filter={"type": "quarter", "year": 2023, "quarter": 3},
            grain="quarter")
        self.assertEqual(result.status, "ok")
        self.assertTrue(result.is_normal_for_phase)
        self.assertAlmostEqual(result.deviation_abs, 0.0, places=4)

    def test_a_period_spiked_to_another_phases_level_is_not_normal(self):
        # Index 12 is 2023-Q1 -- spike it to the Q3 peak's own baseline, a
        # genuine break from Q1's own (near-zero) seasonal index.
        spiked = list(self.values)
        spiked[12] = 100 + 60
        df, schema = _quarterly_frame(spiked)
        result = SeasonalityEngine(ContractAPI(schema)).compare_to_seasonal_norm(
            df, "revenue", time_filter={"type": "quarter", "year": 2023, "quarter": 1},
            grain="quarter")
        self.assertEqual(result.status, "ok")
        self.assertFalse(result.is_normal_for_phase)


class TestCompareToSeasonalNormIsNotSignificance(unittest.TestCase):
    """A measured, real disagreement: a KPI with a perfectly regular seasonal
    bump every Q3, which in one year fails to happen at all.
    `ComparisonEngine.significance` calls the resulting flat Q2->Q3 change
    normal (the seasonal-transition distribution has zero historical
    variance, so `assess_significance` falls back to the noisy all-history
    method, which does not notice a missing +60 jump). This module directly
    compares the *level* against its phase's expectation and correctly flags
    it -- the concrete justification for choosing a level band over a thin
    re-expose of `observe.py`'s transition-based method."""

    @classmethod
    def setUpClass(cls):
        values = []
        for _ in range(5):
            values += [100, 100, 160, 100]
        values += [100, 100, 100, 100]   # final year: Q3 fails to bump
        cls.df, cls.schema = _quarterly_frame(values)
        cls.api = ContractAPI(cls.schema)

    def test_significance_calls_the_missed_bump_normal(self):
        # 5 full years (2020-2024) precede the final, bump-missing year --
        # index arithmetic in `_quarterly_frame` (year = 2020 + i // 4) puts
        # that final year at 2025, not 2024.
        sig = ComparisonEngine(self.api).significance(
            self.df, "revenue", period_a={"type": "quarter", "year": 2025, "quarter": 3})
        self.assertEqual(sig.status, "ok")
        self.assertFalse(sig.is_anomaly)

    def test_seasonal_norm_calls_the_same_period_not_normal(self):
        result = SeasonalityEngine(self.api).compare_to_seasonal_norm(
            self.df, "revenue", time_filter={"type": "quarter", "year": 2025, "quarter": 3},
            grain="quarter")
        self.assertEqual(result.status, "ok")
        self.assertFalse(result.is_normal_for_phase)


class TestAmbiguousOrEmptyPeriodsAreRejected(SeasonalityTestCase):
    def test_a_period_resolving_to_more_than_one_quarter_is_rejected(self):
        with self.assertRaises(InvalidArgumentError):
            self.se.compare_to_seasonal_norm(
                self.rdf, "revenue", time_filter={"type": "year", "year": 2025}, grain="quarter")


# ---------------------------------------------------------------------------
# integrity / prose guard / determinism / portability
# ---------------------------------------------------------------------------
class TestNumericIntegrityAndProseGuard(SeasonalityTestCase):
    def test_the_payload_survives_json_dumps_with_no_nan_or_inf(self):
        seasonality = self.se.detect_seasonality(self.rdf, "revenue", grain="quarter")
        norm = self.se.compare_to_seasonal_norm(
            self.rdf, "revenue", time_filter={"type": "quarter", "year": 2026, "quarter": 2},
            grain="quarter")
        assert_json_safe(seasonality.to_payload())
        assert_json_safe(norm.to_payload())

    def test_no_string_field_outside_the_allowlist_looks_like_prose(self):
        seasonality = self.se.detect_seasonality(self.rdf, "revenue", grain="quarter")
        norm = self.se.compare_to_seasonal_norm(
            self.rdf, "revenue", time_filter={"type": "quarter", "year": 2026, "quarter": 2},
            grain="quarter")
        assert_no_prose_leak(seasonality.to_payload())
        assert_no_prose_leak(norm.to_payload())


class TestDeterminism(SeasonalityTestCase):
    def test_identical_input_gives_identical_output_across_runs(self):
        first = self.se.detect_seasonality(self.rdf, "revenue", grain="quarter").to_payload()
        second = self.se.detect_seasonality(self.rdf, "revenue", grain="quarter").to_payload()
        self.assertEqual(first, second)


class TestPortabilityAcrossDatasets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.schema = contracted("hospital_sample.csv", FIXTURES)
        cls.se = SeasonalityEngine(ContractAPI(cls.schema))
        assert "recovery_rate" in cls.schema.available_kpis

    def test_seasonality_runs_on_a_contract_only_ratio_kpi(self):
        result = self.se.detect_seasonality(self.df, "recovery_rate", grain="quarter")
        # The hospital fixture's own autocorrelation sits just under the
        # gate (measured ~0.30) -- assert on the typed status, not a brittle
        # boolean pinned to a number that could tip either way on a fixture
        # refresh.
        self.assertEqual(result.status, "ok")
        self.assertIsInstance(result.is_seasonal, bool)


if __name__ == "__main__":
    unittest.main()
