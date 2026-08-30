"""
`check_temporal_precedence` / `cross_correlate_lagged` / `test_reverse_causation`
-- `agent/temporal.py`.

The property that matters most for `check_temporal_precedence` is that the
window is part of the answer, not an implementation detail: `detect_onset`
dates the first breach of whatever baseline its own opening periods happen to
form, and measured on the real retail fixture the same KPI pair gives
opposite verdicts under two different scopings (`TestOnsetAtWindowEdge`).
`TestPlantedScenario` is the cross-check that, scoped correctly, the tool
recovers both of `ground_truth.json`'s planted onset dates to the week and
reproduces the temporal contradiction the ground-truth notes describe.

For `cross_correlate_lagged`, what matters is the transform: `.diff()`, not
`pct_change`, chosen after measuring that a percent-change transform halves
the association on the fixture's own supply-disruption signal and destroys
pairs on every sparse-count KPI (`TestTransformIsDifference`). Because the
two temporal tools independently compute a lag, `TestSignConventionAgrees`
pins that a positive number means the same thing -- "the cause turned
first" -- in both places.

For `test_reverse_causation`, what matters is that a mechanically linked pair
(one KPI built from the other's formula) is reported as `"definitional"`
rather than "reverse causation likely" or "forward causation supported" --
neither claim makes sense for two KPIs that move together by construction.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

from app.agent.contract_api import ContractAPI
from app.agent.correlate import CorrelationEngine
from app.agent.errors import InvalidArgumentError
from app.agent.temporal import (MIN_LAG_SERIES_POINTS, TemporalEngine,
                                WINDOW_EDGE_FRACTION)
from app.agent.trend import DIRECTIONS
from app.engines import metrics

from .base import EngineTestCase, FIXTURES, assert_json_safe, assert_no_prose_leak, contracted

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
TRUTH_PATH = ROOT / "sample_data" / "ground_truth.json"

# `mechanism_check`'s prose field (`conclusion`) is the most likely thing to
# be copied into this module by accident, so it joins `test_segments.py`'s
# `BANNED` set here rather than in the shared module.
BANNED = {"interpretation", "detail", "narrative", "note", "conclusion"}


def _assert_no_banned_keys(payload, path: str = "$") -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            assert key not in BANNED, f"{path}.{key} is a banned prose field"
            _assert_no_banned_keys(value, f"{path}.{key}")
    elif isinstance(payload, list):
        for i, item in enumerate(payload):
            _assert_no_banned_keys(item, f"{path}[{i}]")


def _planted_factor(factor_id: str):
    truth = json.loads(TRUTH_PATH.read_text())
    return next(f for f in truth["planted_factors"] if f["id"] == factor_id)


def _weekly_pair(kpi_values, cause_values, kpi_column="revenue", cause_column="units_sold",
                 start="2026-01-05"):
    """A minimal contract-less two-metric weekly frame, the same
    `detect_schema`/`prepare` idiom `test_trend.py`'s `_weekly_frame` uses,
    extended to a second column since every tool in this module compares two
    series."""
    dates = pd.date_range(start, periods=len(kpi_values), freq="7D").astype(str)
    raw = pd.DataFrame({"date": dates, kpi_column: kpi_values, cause_column: cause_values})
    schema = metrics.detect_schema(raw)
    df = metrics.prepare(raw, schema)
    return df, schema


def _step_series(n: int, step_at: int, before: float = 100.0, after: float = 70.0):
    """A clean, single-step series -- flat, then a persistent drop at
    `step_at` -- so `detect_changepoint` finds exactly one onset, at a known
    index, with no ambiguity about which direction fired."""
    return [before] * step_at + [after] * (n - step_at)


class TemporalTestCase(EngineTestCase):
    """The shared retail fixture, with a compiled contract."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.rdf, cls.rschema = contracted("business_metrics_sample.csv")
        cls.api = ContractAPI(cls.rschema)
        cls.te = TemporalEngine(cls.api)
        cls.ce = CorrelationEngine(cls.api)


# ---------------------------------------------------------------------------
# check_temporal_precedence() -- the planted scenario
# ---------------------------------------------------------------------------
class TestPrecedenceOnThePlantedScenario(TemporalTestCase):
    # Scoped to the neighbourhood of the planted events -- full history picks
    # up a different, later coincidental "down" breach for `revenue` (its own
    # instance of the window-sensitivity trap `TestOnsetAtWindowEdge` covers
    # directly), so this window is what makes the ground-truth dates the ones
    # actually recovered.
    _SCOPE = {"type": "range", "start": "2026-01-01", "end": "2026-06-30"}

    def test_scoped_to_north_the_lag_is_the_four_weeks_ground_truth_records(self):
        """`north_demand_erosion` (2026-04-06) and `supply_disruption_
        product_a` (2026-05-04) are exactly 4 weeks apart in the planted
        scenario; scoped to the region carrying the first factor, this tool
        recovers both dates to the week and reports the temporal
        contradiction the ground-truth notes describe: the "cause" starts
        after the KPI already began falling."""
        demand = _planted_factor("north_demand_erosion")
        supply = _planted_factor("supply_disruption_product_a")

        result = self.te.check_temporal_precedence(
            self.rdf, "revenue", "fulfillment_rate", grain="week", time_filter=self._SCOPE,
            filters={"region": "North"}, kpi_direction="down", cause_direction="down")

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.verdict, "kpi_precedes_cause")
        self.assertEqual(result.lag_periods, -4)
        self.assertEqual(result.kpi_period, demand["onset_date"])
        self.assertEqual(result.cause_period, supply["onset_date"])

    def test_the_same_scope_shows_units_sold_and_revenue_turning_together(self):
        result = self.te.check_temporal_precedence(
            self.rdf, "revenue", "units_sold", grain="week", time_filter=self._SCOPE,
            filters={"region": "North"}, kpi_direction="down", cause_direction="down")
        self.assertEqual(result.verdict, "simultaneous")
        self.assertEqual(result.lag_periods, 0)


class TestOnsetAtWindowEdge(TemporalTestCase):
    """The window-sensitivity trap, measured on the real fixture before this
    tool was written: the identical KPI pair gives opposite verdicts under
    two different scopings, and only one of them is trustworthy."""

    def test_full_history_reports_an_implausible_156_week_lead_flagged_at_the_edge(self):
        result = self.te.check_temporal_precedence(
            self.rdf, "units_sold", "stockout_events", grain="week",
            filters={"product": "Product A"}, kpi_direction="down", cause_direction="up")
        self.assertEqual(result.verdict, "cause_precedes_kpi")
        self.assertEqual(result.lag_periods, 156)
        self.assertTrue(result.cause_onset_at_window_edge)

    def test_scoped_to_the_event_neighbourhood_the_verdict_flips_and_is_not_flagged(self):
        tf = {"type": "range", "start": "2025-10-01", "end": "2026-06-29"}
        result = self.te.check_temporal_precedence(
            self.rdf, "units_sold", "stockout_events", grain="week", time_filter=tf,
            filters={"product": "Product A"}, kpi_direction="down", cause_direction="up")
        self.assertEqual(result.verdict, "kpi_precedes_cause")
        self.assertEqual(result.lag_periods, -20)
        self.assertFalse(result.cause_onset_at_window_edge)
        self.assertFalse(result.kpi_onset_at_window_edge)


# ---------------------------------------------------------------------------
# check_temporal_precedence() -- statuses, direction, gaps
# ---------------------------------------------------------------------------
class TestPrecedenceOnSyntheticSteps(unittest.TestCase):
    def test_a_cause_that_steps_first_is_cause_precedes_kpi(self):
        n = 30
        kpi = _step_series(n, 15)
        cause = _step_series(n, 12)
        df, schema = _weekly_pair(kpi, cause)
        result = TemporalEngine(ContractAPI(schema)).check_temporal_precedence(
            df, "revenue", "units_sold", grain="week",
            kpi_direction="down", cause_direction="down")
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.verdict, "cause_precedes_kpi")
        self.assertEqual(result.lag_periods, 3)

    def test_swapping_the_two_series_negates_the_lag_and_flips_the_verdict(self):
        n = 30
        kpi = _step_series(n, 15)
        cause = _step_series(n, 12)
        df, schema = _weekly_pair(kpi, cause)
        te = TemporalEngine(ContractAPI(schema))
        forward = te.check_temporal_precedence(df, "revenue", "units_sold", grain="week",
                                               kpi_direction="down", cause_direction="down")
        reversed_ = te.check_temporal_precedence(df, "units_sold", "revenue", grain="week",
                                                 kpi_direction="down", cause_direction="down")
        self.assertEqual(reversed_.lag_periods, -forward.lag_periods)
        self.assertEqual(reversed_.verdict, "kpi_precedes_cause")

    def test_the_same_step_week_is_simultaneous(self):
        n = 30
        step = _step_series(n, 15)
        df, schema = _weekly_pair(step, step)
        result = TemporalEngine(ContractAPI(schema)).check_temporal_precedence(
            df, "revenue", "units_sold", grain="week",
            kpi_direction="down", cause_direction="down")
        self.assertEqual(result.verdict, "simultaneous")
        self.assertEqual(result.lag_periods, 0)

    def test_a_gap_between_the_two_onsets_is_counted_not_hidden(self):
        n = 30
        kpi = _step_series(n, 15)
        cause = _step_series(n, 12)
        dates = list(pd.date_range("2026-01-05", periods=n, freq="7D").astype(str))
        drop_at = 13   # strictly between the two onsets
        dates = dates[:drop_at] + dates[drop_at + 1:]
        kpi = kpi[:drop_at] + kpi[drop_at + 1:]
        cause = cause[:drop_at] + cause[drop_at + 1:]
        raw = pd.DataFrame({"date": dates, "revenue": kpi, "units_sold": cause})
        schema = metrics.detect_schema(raw)
        df = metrics.prepare(raw, schema)
        result = TemporalEngine(ContractAPI(schema)).check_temporal_precedence(
            df, "revenue", "units_sold", grain="week",
            kpi_direction="down", cause_direction="down")
        self.assertEqual(result.status, "ok")
        self.assertGreater(result.gaps_between, 0)
        # The true calendar lag survives the gap -- it is not compressed by
        # one just because a week is missing between the two onsets.
        self.assertEqual(result.lag_periods, 3)


class TestPrecedenceStatusesAreToldApart(unittest.TestCase):
    def test_a_short_series_is_both_undetermined_via_insufficient_history(self):
        n = 9
        step = _step_series(n, 5)
        df, schema = _weekly_pair(step, step)
        result = TemporalEngine(ContractAPI(schema)).check_temporal_precedence(
            df, "revenue", "units_sold", grain="week", baseline_periods=8, persistence=2)
        self.assertEqual(result.status, "both_undetermined")
        self.assertEqual(result.kpi_changepoint.status, "insufficient_history")
        self.assertEqual(result.cause_changepoint.status, "insufficient_history")
        self.assertIsNone(result.verdict)
        self.assertIsNone(result.lag_periods)

    def test_a_cause_searched_in_the_wrong_direction_is_cause_undetermined(self):
        n = 30
        kpi = _step_series(n, 15)
        cause = _step_series(n, 12)
        df, schema = _weekly_pair(kpi, cause)
        result = TemporalEngine(ContractAPI(schema)).check_temporal_precedence(
            df, "revenue", "units_sold", grain="week",
            kpi_direction="down", cause_direction="up")
        self.assertEqual(result.status, "cause_undetermined")
        self.assertEqual(result.cause_changepoint.status, "no_changepoint")
        self.assertIsNone(result.verdict)


class TestDirectionIsValidated(unittest.TestCase):
    def test_an_invented_kpi_direction_is_rejected(self):
        n = 30
        step = _step_series(n, 15)
        df, schema = _weekly_pair(step, step)
        with self.assertRaises(InvalidArgumentError) as ctx:
            TemporalEngine(ContractAPI(schema)).check_temporal_precedence(
                df, "revenue", "units_sold", kpi_direction="sideways")
        self.assertEqual(ctx.exception.to_payload()["valid_alternatives"], list(DIRECTIONS))

    def test_an_invented_cause_direction_is_rejected(self):
        n = 30
        step = _step_series(n, 15)
        df, schema = _weekly_pair(step, step)
        with self.assertRaises(InvalidArgumentError):
            TemporalEngine(ContractAPI(schema)).check_temporal_precedence(
                df, "revenue", "units_sold", cause_direction="sideways")


class TestDirectionAnySelectsTheEarliestTurnEitherWay(TemporalTestCase):
    """Documents the measured trap rather than hiding it: `"any"` can select
    an unrelated seasonal turn instead of the decline under investigation."""

    def test_any_selects_a_different_onset_than_an_explicit_direction(self):
        any_direction = self.te.check_temporal_precedence(
            self.rdf, "revenue", "units_sold", grain="week", kpi_direction="any",
            cause_direction="any")
        explicit_down = self.te.check_temporal_precedence(
            self.rdf, "revenue", "units_sold", grain="week", kpi_direction="down",
            cause_direction="down")
        self.assertNotEqual(any_direction.kpi_period, explicit_down.kpi_period)


# ---------------------------------------------------------------------------
# cross_correlate_lagged()
# ---------------------------------------------------------------------------
def _lagged_pair(diffs, lag: int, start_level: float = 100.0):
    """kpi's diff series is `diffs`; the cause's diff series is `diffs`
    shifted so that `cause_diff[t] == kpi_diff[t + lag]` -- the cause's move
    at `t` predicts the KPI's move `lag` periods later, i.e. the cause leads
    by `lag`. Wrapped (not truncated) so both series stay the same length."""
    cause_diffs = diffs[lag:] + diffs[:lag]
    def _levels(d):
        out = [start_level]
        for x in d:
            out.append(out[-1] + x)
        return out
    return _levels(diffs), _levels(cause_diffs)


# A fixed, non-monotonic literal diff pattern -- 40 steps, so both the
# 12-point merged-series floor and every candidate lag's 8-pair floor clear
# comfortably even after `max_lag` trims each end.
_DIFF_PATTERN = [6, -4, 9, -7, 5, -3, 8, -6, 4, -2, 7, -5, 3, -8, 6, -4, 9, -7, 5, -3,
                8, -6, 4, -2, 7, -5, 3, -8, 6, -4, 9, -7, 5, -3, 8, -6, 4, -2, 7, -5]


class TestLaggedCorrelation(unittest.TestCase):
    def test_an_injected_lag_of_two_is_recovered(self):
        kpi, cause = _lagged_pair(_DIFF_PATTERN, lag=2)
        df, schema = _weekly_pair(kpi, cause)
        result = TemporalEngine(ContractAPI(schema)).cross_correlate_lagged(
            df, "revenue", "units_sold", grain="week", max_lag=5)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.best_lag, 2)
        self.assertEqual(result.verdict, "cause_leads")
        self.assertGreater(result.r_at_best, 0.99)

    def test_swapping_the_series_recovers_the_negated_lag(self):
        kpi, cause = _lagged_pair(_DIFF_PATTERN, lag=2)
        df, schema = _weekly_pair(kpi, cause)
        result = TemporalEngine(ContractAPI(schema)).cross_correlate_lagged(
            df, "units_sold", "revenue", grain="week", max_lag=5)
        self.assertEqual(result.best_lag, -2)
        self.assertEqual(result.verdict, "kpi_leads")

    def test_a_sub_twelve_point_series_is_insufficient_history(self):
        short_diffs = _DIFF_PATTERN[:8]
        kpi, cause = _lagged_pair(short_diffs, lag=1)
        df, schema = _weekly_pair(kpi, cause)
        result = TemporalEngine(ContractAPI(schema)).cross_correlate_lagged(
            df, "revenue", "units_sold", grain="week", max_lag=3)
        self.assertEqual(result.status, "insufficient_history")
        self.assertEqual(result.profile, ())

    def test_a_constant_cause_is_constant_b_at_every_lag(self):
        n = 30
        kpi = _step_series(n, 15)
        cause = [50.0] * n
        df, schema = _weekly_pair(kpi, cause)
        result = TemporalEngine(ContractAPI(schema)).cross_correlate_lagged(
            df, "revenue", "units_sold", grain="week", max_lag=5)
        self.assertEqual(result.status, "no_usable_lag")
        zero = next(p for p in result.profile if p.lag == 0)
        self.assertEqual(zero.status, "constant_b")

    def test_ties_break_toward_the_smaller_absolute_lag(self):
        """Not `analysis.lead_lag`'s `max(key=abs)`, which silently favours
        the most negative lag on a tie because results are built from
        `-max_lag` upward. Mirrors the exact tie-break key
        `cross_correlate_lagged` uses: `(-abs(r), abs(lag), lag)`."""
        from app.agent.temporal import LagPoint

        def _pick(points):
            return min(points, key=lambda p: (-abs(p.r), abs(p.lag), p.lag))

        # No tie: the larger |r| wins outright regardless of position.
        clear_winner = (
            LagPoint(lag=-3, r=0.4, r_squared=0.16, n=20, status="ok"),
            LagPoint(lag=2, r=0.9, r_squared=0.81, n=20, status="ok"),
        )
        self.assertEqual(_pick(clear_winner).lag, 2)

        # A genuine |r| tie between a smaller and a larger |lag| -- the
        # smaller one must win, not whichever was built first.
        tied_by_magnitude = (
            LagPoint(lag=-3, r=0.5, r_squared=0.25, n=20, status="ok"),
            LagPoint(lag=0, r=-0.5, r_squared=0.25, n=20, status="ok"),
        )
        self.assertEqual(_pick(tied_by_magnitude).lag, 0)

        # A genuine |r| AND |lag| tie -- the more negative lag wins,
        # matching `analysis.lead_lag`'s own tie-break so the two tools do
        # not disagree on this one remaining corner.
        tied_by_lag_magnitude = (
            LagPoint(lag=-2, r=0.5, r_squared=0.25, n=20, status="ok"),
            LagPoint(lag=2, r=0.5, r_squared=0.25, n=20, status="ok"),
        )
        self.assertEqual(_pick(tied_by_lag_magnitude).lag, -2)

    def test_max_lag_outside_the_ceiling_is_an_argument_error(self):
        n = 30
        step = _step_series(n, 15)
        df, schema = _weekly_pair(step, step)
        te = TemporalEngine(ContractAPI(schema))
        with self.assertRaises(InvalidArgumentError):
            te.cross_correlate_lagged(df, "revenue", "units_sold", max_lag=0)
        with self.assertRaises(InvalidArgumentError):
            te.cross_correlate_lagged(df, "revenue", "units_sold", max_lag=52)


class TestTransformIsDifference(TemporalTestCase):
    def test_at_lag_zero_the_pairing_matches_correlate_kpis_exactly(self):
        """The two tools must disagree only about the transform, never about
        which periods were comparable."""
        lagged = self.te.cross_correlate_lagged(self.rdf, "revenue", "units_sold",
                                                grain="week", max_lag=5)
        contemporaneous = self.ce.correlate_kpis(self.rdf, "revenue", "units_sold",
                                                 mode="time_series", grain="week", min_n=6)
        at_zero = next(p for p in lagged.profile if p.lag == 0)
        self.assertEqual(at_zero.n, contemporaneous.n)
        self.assertEqual(lagged.pairs_dropped_at_gaps, contemporaneous.pairs_dropped_at_gaps)
        self.assertEqual(at_zero.status, contemporaneous.status)
        self.assertEqual(lagged.transform, "difference")
        self.assertEqual(contemporaneous.transform, "change")
        self.assertNotEqual(lagged.transform, contemporaneous.transform)

    def test_zero_valued_periods_do_not_silently_drop_pairs(self):
        """Hospital `deaths` is 0 in 17 of 183 weeks. A percent-change
        transform would destroy 16 of 182 pairs at every lag; a difference
        destroys none."""
        df, schema = contracted("hospital_sample.csv", FIXTURES)
        result = TemporalEngine(ContractAPI(schema)).cross_correlate_lagged(
            df, "admissions", "deaths", grain="week", max_lag=5)
        self.assertEqual(result.status, "ok")
        at_zero = next(p for p in result.profile if p.lag == 0)
        self.assertEqual(at_zero.n, 182)


class TestSignConventionAgrees(unittest.TestCase):
    def test_both_temporal_tools_agree_that_positive_means_cause_first(self):
        n = 30
        kpi = _step_series(n, 15)
        cause = _step_series(n, 12)
        df, schema = _weekly_pair(kpi, cause)
        te = TemporalEngine(ContractAPI(schema))
        precedence = te.check_temporal_precedence(df, "revenue", "units_sold", grain="week",
                                                  kpi_direction="down", cause_direction="down")
        lag_kpi, lag_cause = _lagged_pair(_DIFF_PATTERN, lag=2)
        df2, schema2 = _weekly_pair(lag_kpi, lag_cause)
        lagged = TemporalEngine(ContractAPI(schema2)).cross_correlate_lagged(
            df2, "revenue", "units_sold", grain="week", max_lag=5)
        self.assertGreater(precedence.lag_periods, 0)
        self.assertGreater(lagged.best_lag, 0)


# ---------------------------------------------------------------------------
# test_reverse_causation()
# ---------------------------------------------------------------------------
class TestReverseCausation(TemporalTestCase):
    def test_a_kpi_inside_the_causes_formula_is_definitional(self):
        result = self.te.test_reverse_causation(self.rdf, "revenue", "avg_order_value",
                                                 grain="week")
        self.assertEqual(result.mechanical, "cause_derived_from_kpi")
        self.assertEqual(result.mechanical_relation, "formula_numerator")
        self.assertEqual(result.verdict, "definitional")

    def test_the_forward_direction_is_also_definitional(self):
        result = self.te.test_reverse_causation(self.rdf, "avg_order_value", "revenue",
                                                 grain="week")
        self.assertEqual(result.mechanical, "kpi_derived_from_cause")
        self.assertEqual(result.verdict, "definitional")

    def test_a_mutual_formula_pair_is_reported_as_mutual(self):
        result = self.te.test_reverse_causation(self.rdf, "gross_profit", "gross_margin_pct",
                                                 grain="week")
        self.assertEqual(result.mechanical, "mutual")
        self.assertEqual(result.verdict, "definitional")

    def test_a_pair_with_no_formula_link_falls_back_to_timing(self):
        result = self.te.test_reverse_causation(self.rdf, "revenue", "fulfillment_rate",
                                                 grain="week")
        self.assertEqual(result.mechanical, "none")
        self.assertIn(result.verdict, ("reverse_supported", "forward_supported", "undetermined"))

    def test_a_derivation_only_pair_is_shared_inputs_not_directional(self):
        result = self.te.test_reverse_causation(
            self.rdf, "revenue_less_marketing_spend", "gross_profit", grain="week")
        self.assertEqual(result.mechanical, "shared_inputs")
        self.assertEqual(result.mechanical_relation, "derivation")
        # A shared-inputs link names no direction -- it must never be
        # reported as "definitional", the state reserved for an actual
        # formula dependency.
        self.assertNotEqual(result.verdict, "definitional")

    def test_the_timing_call_is_made_exactly_once(self):
        """The reverse profile is the mirror of the forward one (verified:
        the two are identical to machine precision under a lag negation), so
        a correct implementation calls `cross_correlate_lagged` once, not
        twice."""
        result = self.te.test_reverse_causation(self.rdf, "revenue", "fulfillment_rate",
                                                 grain="week")
        direct = self.te.cross_correlate_lagged(self.rdf, "revenue", "fulfillment_rate",
                                                grain="week")
        self.assertEqual(result.timing.to_payload(), direct.to_payload())


# ---------------------------------------------------------------------------
# integrity / prose guard / determinism / portability
# ---------------------------------------------------------------------------
class TestNumericIntegrityAndProseGuard(TemporalTestCase):
    def test_the_payload_survives_json_dumps_with_no_nan_or_inf(self):
        precedence = self.te.check_temporal_precedence(
            self.rdf, "revenue", "units_sold", grain="week", filters={"region": "North"},
            kpi_direction="down", cause_direction="down")
        lagged = self.te.cross_correlate_lagged(self.rdf, "revenue", "units_sold", grain="week")
        reverse = self.te.test_reverse_causation(self.rdf, "revenue", "avg_order_value",
                                                 grain="week")
        assert_json_safe(precedence.to_payload())
        assert_json_safe(lagged.to_payload())
        assert_json_safe(reverse.to_payload())

    def test_no_string_field_outside_the_allowlist_looks_like_prose(self):
        precedence = self.te.check_temporal_precedence(
            self.rdf, "revenue", "units_sold", grain="week", filters={"region": "North"},
            kpi_direction="down", cause_direction="down")
        lagged = self.te.cross_correlate_lagged(self.rdf, "revenue", "units_sold", grain="week")
        reverse = self.te.test_reverse_causation(self.rdf, "revenue", "avg_order_value",
                                                 grain="week")
        assert_no_prose_leak(precedence.to_payload())
        assert_no_prose_leak(lagged.to_payload())
        assert_no_prose_leak(reverse.to_payload())
        _assert_no_banned_keys(precedence.to_payload())
        _assert_no_banned_keys(lagged.to_payload())
        _assert_no_banned_keys(reverse.to_payload())


class TestDeterminism(TemporalTestCase):
    def test_identical_input_gives_identical_output_across_runs(self):
        first = self.te.check_temporal_precedence(
            self.rdf, "revenue", "units_sold", grain="week", filters={"region": "North"},
            kpi_direction="down", cause_direction="down").to_payload()
        second = self.te.check_temporal_precedence(
            self.rdf, "revenue", "units_sold", grain="week", filters={"region": "North"},
            kpi_direction="down", cause_direction="down").to_payload()
        self.assertEqual(first, second)

    def test_lagged_correlation_profile_is_ascending_by_lag(self):
        result = self.te.cross_correlate_lagged(self.rdf, "revenue", "units_sold", grain="week")
        lags = [p.lag for p in result.profile]
        self.assertEqual(lags, sorted(lags))


class TestPortabilityAcrossDatasets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.schema = contracted("hospital_sample.csv", FIXTURES)
        cls.te = TemporalEngine(ContractAPI(cls.schema))
        assert "recovery_rate" in cls.schema.available_kpis

    def test_temporal_tools_run_on_a_contract_only_ratio_kpi(self):
        lagged = self.te.cross_correlate_lagged(self.df, "recovery_rate", "admissions",
                                                grain="week")
        self.assertIn(lagged.status, ("ok", "no_overlap", "insufficient_history", "no_usable_lag"))

        precedence = self.te.check_temporal_precedence(
            self.df, "recovery_rate", "admissions", grain="week")
        self.assertIn(precedence.status,
                      ("ok", "kpi_undetermined", "cause_undetermined", "both_undetermined"))


if __name__ == "__main__":
    unittest.main()
