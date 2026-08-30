"""
`detect_seasonality` / `compare_to_seasonal_norm` -- does a KPI move with the
calendar, and is a given period's *value* normal for its place in that cycle.

Nothing in this codebase fits a seasonal pattern before this module: no ACF,
periodogram, Fourier term, or seasonal index exists anywhere in `backend/app`.
The one seasonality-aware code path, `observe.assess_significance`
(`engines/observe.py:189-193`), only ever asks about a *transition* -- it
distributes the current quarter's change against the history of that same
quarter-over-quarter change, when at least three such transitions exist. That
answers "is this change normal", never "is this level normal". A KPI that
dips by the same amount every Q3 has a perfectly normal Q2->Q3 *change* every
year, and also a Q3 *level* that could be 30% below every other Q3 ever
recorded -- `assess_significance` cannot see the second question at all.
Without it, the agent has no way to tell a leader "that dip is just Q3" and
will explain a seasonal pattern as a problem -- the single highest-credibility
risk in this rewrite.

`compare_to_seasonal_norm` below is therefore not a thin re-expose of
`observe.py:189-193`: `ComparisonEngine.significance` (`agent/compare.py`)
already publishes that entire result (`method="seasonal_robust_z"`,
`same_quarter_points`, `expected_value`, `normal_range`, `robust_z`, `power`),
and a second tool name for the same fields would be exactly the kind of
tool-selection hazard `concentration.py`'s own header warns about. This
module answers the level question `significance()` cannot, and both tools
sit on the same shared trend/season fit so a caller can compare their
verdicts directly (see `TestCompareToSeasonalNormIsNotSignificance`).

**Why detection is gated on autocorrelation, not on a seasonal-strength
ratio.** The obvious method -- an STL-style `1 - sigma(residual after
season) / sigma(detrended)` with a fixed cutoff -- was measured against a
Monte Carlo null and against the real sample fixtures before being rejected.
Two failures, both fatal for a leader-facing demo:

1. It is not a safe gate. At n=12 quarters (three cycles -- a perfectly
   ordinary ask) a `strength >= 0.60` cutoff fires on 32% of pure Gaussian
   noise, and its power on a genuine strong seasonal signal *falls* from 0.69
   to 0.57 as n goes from 14 to 24, because more observations per phase
   shrink the ratio. A threshold whose power decreases with sample size is
   not defensible at any cutoff.
2. It would misfire on the demo dataset itself. Neither sample fixture
   contains real seasonality (every autocorrelation at the cycle lag is
   approximately zero), yet retail's headline KPI, quarterly `revenue`,
   scores an in-sample strength of 0.668 -- above every commonly cited STL
   cutoff. A fixed strength gate would announce, on the dataset this product
   demos on, that revenue is seasonal when it is not: the exact failure this
   module exists to prevent, inverted.

Detection is gated on the autocorrelation of the detrended series at the
cycle lag instead (calibrated to <=7% false-positive across n in
{12, 14, 24, 42, 182}, 0.84-0.99 power on genuine seasonality).
`seasonal_strength` is still published, but computed leave-one-cycle-out --
each point's seasonal index comes from every *other* observation of its
phase -- rather than in-sample, so the number it reports agrees with the ACF
verdict instead of contradicting it (in-sample 0.668 vs leave-one-out 0.335
on retail `revenue`, against a genuinely seasonal series' 0.970 vs 0.947).
It is reported as a descriptive magnitude and is never itself a condition.

A seeded permutation test was also measured: correct size (0.05-0.09) but
only 0.60 power at n=14 against the ACF gate's 0.97, at 99x the per-call
cost, with determinism resting on NumPy's RNG stream staying stable across
versions -- strictly dominated, and skipped. A calibrated p-value is the
natural `scipy`-backed task at X3, not here (`requirements.txt` carries numpy
and pandas only; `agent/trend.py` already declined `scipy` for the same
reason).

Detrending reuses `trend.theil_sen` -- the identical fit `detect_trend` uses
-- rather than a second hand-rolled copy free to drift from it, the same
reason `series.period_ordinal` is a shared function rather than duplicated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..config import get_settings
from ..engines.drivers import robust_sigma
from ..engines.metrics import pct_change, safe
from .compare import Band
from .contract_api import ContractAPI
from .errors import InvalidArgumentError
from .series import SeriesEngine, label_to_period, period_ordinal
from .timefilter import GRAIN_COLUMNS, TimeFilter
from .timefilter import TimeSelection
from .timefilter import parse as parse_time_filter

from .trend import theil_sen

# The cycle length of each grain a phase can be derived from. `year` has no
# sub-annual phase, so it is refused rather than silently treated as a
# one-phase cycle.
CYCLE_LENGTH = {"quarter": 4, "month": 12, "week": 52}

SEASONAL_ACF_MIN = 0.30
MIN_CYCLES = 2
MIN_POINTS_PER_PHASE = 3


def _phase(grain: str, label: str) -> int:
    """The position within its cycle a period label occupies -- derived from
    the label itself (via `series.label_to_period`), never from a point's
    position in the series, so a gap in the middle never shifts every later
    phase."""
    period = label_to_period(grain, label)
    if grain == "quarter":
        return int(period.quarter)
    if grain == "month":
        return int(period.month)
    return int(period.start_time.isocalendar()[1])   # week -> ISO week number


@dataclass(frozen=True)
class PhaseEffect:
    phase: int
    index: Optional[float]
    observations: int

    def to_payload(self) -> Dict[str, Any]:
        return {"phase": self.phase, "index": safe(self.index),
               "observations": self.observations}


@dataclass(frozen=True)
class Seasonality:
    kpi: str
    label: str
    unit: str
    grain: str
    status: str                       # "ok" | "insufficient_cycles" | "unsupported_grain" | "degenerate_dispersion"
    method: str = "phase_median_detrended"
    points: int = 0
    cycles: Optional[float] = None
    cycle_length: Optional[int] = None
    min_phase_points: Optional[int] = None
    seasonal_strength: Optional[float] = None    # descriptive only -- never a detection gate
    acf_at_cycle: Optional[float] = None
    is_seasonal: Optional[bool] = None
    phases: Tuple[PhaseEffect, ...] = ()
    peak_phase: Optional[int] = None
    trough_phase: Optional[int] = None
    trend_slope_per_period: Optional[float] = None
    residual_sigma: Optional[float] = None
    gaps: Tuple[str, ...] = ()
    selection: Optional[TimeSelection] = None
    # `()` cannot stand in for an empty mapping (`.items()` below raises
    # `AttributeError` on a default-constructed instance); `{}` cannot be the
    # literal default (a dataclass forbids a mutable default), hence
    # `field(default_factory=dict)`.
    filters: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "kpi": self.kpi, "label": self.label, "unit": self.unit, "grain": self.grain,
            "status": self.status, "method": self.method, "points": self.points,
            "cycles": safe(self.cycles), "cycle_length": self.cycle_length,
            "min_phase_points": self.min_phase_points,
            "seasonal_strength": safe(self.seasonal_strength),
            "acf_at_cycle": safe(self.acf_at_cycle),
            "is_seasonal": self.is_seasonal,
            "phases": [p.to_payload() for p in self.phases],
            "peak_phase": self.peak_phase, "trough_phase": self.trough_phase,
            "trend_slope_per_period": safe(self.trend_slope_per_period),
            "residual_sigma": safe(self.residual_sigma),
            "gaps": list(self.gaps),
            "selection": self.selection.to_payload() if self.selection else None,
            "filters": {k: list(v) for k, v in self.filters.items()},
        }


@dataclass(frozen=True)
class SeasonalNorm:
    kpi: str
    label: str
    unit: str
    grain: str
    status: str                       # "ok" | "unsupported_grain" | "insufficient_cycles" | "degenerate_dispersion" | "phase_not_covered"
    period: Optional[str] = None
    phase: Optional[int] = None
    phase_points: Optional[int] = None
    actual: Optional[float] = None
    trend_level: Optional[float] = None
    seasonal_index: Optional[float] = None
    expected: Optional[float] = None
    deviation_abs: Optional[float] = None
    deviation_pct: Optional[float] = None
    robust_z: Optional[float] = None
    band: Optional[Band] = None
    is_normal_for_phase: Optional[bool] = None
    is_seasonal: Optional[bool] = None
    seasonal_strength: Optional[float] = None
    selection: Optional[TimeSelection] = None
    filters: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "kpi": self.kpi, "label": self.label, "unit": self.unit, "grain": self.grain,
            "status": self.status, "period": self.period,
            "phase": self.phase, "phase_points": self.phase_points,
            "actual": safe(self.actual), "trend_level": safe(self.trend_level),
            "seasonal_index": safe(self.seasonal_index), "expected": safe(self.expected),
            "deviation_abs": safe(self.deviation_abs), "deviation_pct": safe(self.deviation_pct),
            "robust_z": safe(self.robust_z),
            "band": self.band.to_payload() if self.band else None,
            "is_normal_for_phase": self.is_normal_for_phase,
            "is_seasonal": self.is_seasonal, "seasonal_strength": safe(self.seasonal_strength),
            "selection": self.selection.to_payload() if self.selection else None,
            "filters": {k: list(v) for k, v in self.filters.items()},
        }


@dataclass
class _Fit:
    """Internal only -- never returned from a tool. The trend + seasonal
    index shared by both public methods, plus enough of the raw arithmetic
    for each to apply its own phase-coverage judgment (see the module
    docstring on why that judgment differs between the two)."""
    status: str
    points: int = 0
    gaps: Tuple[str, ...] = ()
    selection: Optional[TimeSelection] = None
    cycle_length: Optional[int] = None
    cycles: Optional[float] = None
    slope: Optional[float] = None
    intercept: Optional[float] = None
    trend_residual_sigma: Optional[float] = None   # dispersion after removing trend only
    residual_sigma: Optional[float] = None         # dispersion after removing trend AND season
    phase_index: Optional[Dict[int, float]] = None
    phase_counts: Optional[Dict[int, int]] = None
    seasonal_strength: Optional[float] = None
    acf_at_cycle: Optional[float] = None


class SeasonalityEngine:
    """`detect_seasonality` / `compare_to_seasonal_norm`, bound to one
    dataset's `ContractAPI`."""

    def __init__(self, api: ContractAPI):
        self._api = api
        self._series = SeriesEngine(api)

    # -- the shared fit -------------------------------------------------------
    def _fit(self, df: pd.DataFrame, kpi_key: str, grain: str,
            filters: Optional[Mapping[str, Union[str, Sequence[str]]]]) -> _Fit:
        series = self._series.series(df, kpi_key, grain=grain, time_filter={"type": "all"},
                                     filters=filters)
        points = [p for p in series.points if p.value == p.value]   # drop NaN
        n = len(points)

        if grain not in CYCLE_LENGTH:
            return _Fit(status="unsupported_grain", points=n, gaps=series.gaps,
                       selection=series.selection)

        cycle_length = CYCLE_LENGTH[grain]
        if n < MIN_CYCLES * cycle_length:
            return _Fit(status="insufficient_cycles", points=n, gaps=series.gaps,
                       selection=series.selection, cycle_length=cycle_length,
                       cycles=n / cycle_length)

        xs = np.array([period_ordinal(grain, p.period) for p in points], dtype=float)
        ys = np.array([p.value for p in points], dtype=float)
        phases = [_phase(grain, p.period) for p in points]

        slope, intercept = theil_sen(xs, ys)
        resid = ys - (intercept + slope * xs)   # trend-only residual

        _, trend_sigma = robust_sigma(resid)
        if not (trend_sigma == trend_sigma) or trend_sigma <= 0:
            return _Fit(status="degenerate_dispersion", points=n, gaps=series.gaps,
                       selection=series.selection, cycle_length=cycle_length,
                       cycles=n / cycle_length, slope=slope, intercept=intercept)

        by_phase: Dict[int, List[float]] = {}
        for phase, r in zip(phases, resid):
            by_phase.setdefault(phase, []).append(float(r))
        phase_counts = {phase: len(vals) for phase, vals in by_phase.items()}

        raw_index = {phase: float(np.median(vals)) for phase, vals in by_phase.items()}
        centre = float(np.median(list(raw_index.values())))
        phase_index = {phase: v - centre for phase, v in raw_index.items()}

        # Seasonal strength, leave-one-cycle-out: each point's index comes
        # from the *other* observations of its phase, so a phase median
        # fitted on nothing but noise cannot inflate the reported strength --
        # see the module docstring for the measured in-sample failure this
        # avoids.
        # Group point indices (not values) by phase once, so "every other
        # observation of this point's phase" is a plain list exclusion --
        # no positional re-derivation per point.
        positions_by_phase: Dict[int, List[int]] = {}
        for i, phase in enumerate(phases):
            positions_by_phase.setdefault(phase, []).append(i)

        oos_resid = []
        for i, (phase, r) in enumerate(zip(phases, resid)):
            peers = [resid[j] for j in positions_by_phase[phase] if j != i]
            oos_index = (float(np.median(peers)) - centre) if peers else 0.0
            oos_resid.append(r - oos_index)
        _, oos_sigma = robust_sigma(oos_resid)
        strength = max(0.0, 1.0 - oos_sigma / trend_sigma) if oos_sigma == oos_sigma else 0.0

        # The dispersion actually left once BOTH trend and season are
        # removed -- in-sample, not leave-one-out, since `compare_to_
        # seasonal_norm` evaluates against the fitted model directly rather
        # than cross-validating a single held-out point. This, not the
        # trend-only `trend_sigma` above, is the right denominator for a
        # z-score against the seasonal expectation: `trend_sigma` still
        # contains the seasonal swing itself, which would make a real
        # deviation look small merely because the KPI is normally volatile
        # by season.
        season_adjusted = [r - phase_index[phase] for phase, r in zip(phases, resid)]
        _, season_sigma = robust_sigma(season_adjusted)
        if not (season_sigma == season_sigma):
            season_sigma = 0.0

        # Autocorrelation at the cycle lag, paired by calendar ordinal so a
        # gap in the series shortens the count of valid pairs instead of
        # silently compressing the lag.
        d = resid - resid.mean()
        d_by_ordinal = {int(x): v for x, v in zip(xs, d)}
        num, den = 0.0, float(np.sum(d * d))
        for x, dv in d_by_ordinal.items():
            prev = d_by_ordinal.get(x - cycle_length)
            if prev is not None:
                num += dv * prev
        acf = (num / den) if den > 0 else 0.0

        return _Fit(status="ok", points=n, gaps=series.gaps, selection=series.selection,
                   cycle_length=cycle_length, cycles=n / cycle_length,
                   slope=slope, intercept=intercept,
                   trend_residual_sigma=trend_sigma, residual_sigma=season_sigma,
                   phase_index=phase_index, phase_counts=phase_counts,
                   seasonal_strength=strength, acf_at_cycle=float(acf))

    # -- detect_seasonality() -------------------------------------------------
    def detect_seasonality(self, df: pd.DataFrame, kpi_key: str, grain: str = "quarter",
                           time_filter: Optional[Union[Mapping[str, Any], TimeFilter]] = None,
                           filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None
                           ) -> Seasonality:
        label, unit = self._api.label(kpi_key), self._api.unit(kpi_key)
        # The cycle itself is always measured over the dataset's full time
        # extent -- `time_filter` is accepted for signature symmetry with the
        # rest of this layer but is unused: there is no single-period scope
        # that a seasonal pattern could be measured within.
        scope_filters = dict(filters or {})
        fit = self._fit(df, kpi_key, grain, scope_filters)

        if fit.status != "ok":
            return Seasonality(kpi=kpi_key, label=label, unit=unit, grain=grain,
                              status=fit.status, points=fit.points, cycles=fit.cycles,
                              cycle_length=fit.cycle_length, gaps=fit.gaps,
                              selection=fit.selection, filters=scope_filters)

        min_phase_points = min(fit.phase_counts.values())
        if min_phase_points < MIN_POINTS_PER_PHASE:
            return Seasonality(kpi=kpi_key, label=label, unit=unit, grain=grain,
                              status="insufficient_cycles", points=fit.points,
                              cycles=fit.cycles, cycle_length=fit.cycle_length,
                              min_phase_points=min_phase_points, gaps=fit.gaps,
                              selection=fit.selection, filters=scope_filters)

        phases = tuple(
            PhaseEffect(phase=phase, index=fit.phase_index[phase],
                       observations=fit.phase_counts[phase])
            for phase in sorted(fit.phase_index))
        peak_phase = max(fit.phase_index, key=lambda p: fit.phase_index[p])
        trough_phase = min(fit.phase_index, key=lambda p: fit.phase_index[p])
        is_seasonal = fit.acf_at_cycle >= SEASONAL_ACF_MIN

        return Seasonality(
            kpi=kpi_key, label=label, unit=unit, grain=grain, status="ok",
            points=fit.points, cycles=fit.cycles, cycle_length=fit.cycle_length,
            min_phase_points=min_phase_points,
            seasonal_strength=fit.seasonal_strength, acf_at_cycle=fit.acf_at_cycle,
            is_seasonal=is_seasonal, phases=phases,
            peak_phase=peak_phase, trough_phase=trough_phase,
            trend_slope_per_period=fit.slope, residual_sigma=fit.residual_sigma,
            gaps=fit.gaps, selection=fit.selection, filters=scope_filters)

    # -- compare_to_seasonal_norm() --------------------------------------------
    def compare_to_seasonal_norm(self, df: pd.DataFrame, kpi_key: str,
                                 time_filter: Union[Mapping[str, Any], TimeFilter],
                                 grain: str = "quarter",
                                 filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
                                 k: Optional[float] = None) -> SeasonalNorm:
        """
        `time_filter` names the ONE period under test -- required, unlike
        every scope-over-a-range `time_filter` elsewhere in this layer,
        because "is this period normal" has no meaning without a period.
        `grain` must agree with what `time_filter` resolves to; it is the
        second, not the first, positional argument here (rather than
        `series.py`'s `kpi_key, grain, time_filter` order) because the
        period being asked about is this tool's entire reason to exist.
        """
        label, unit = self._api.label(kpi_key), self._api.unit(kpi_key)
        scope_filters = dict(filters or {})
        if k is None:
            k = get_settings().anomaly_z_threshold

        if grain not in CYCLE_LENGTH:
            raise InvalidArgumentError("grain", grain,
                                       "must be one of the supported grains.",
                                       list(CYCLE_LENGTH))

        tf = time_filter if isinstance(time_filter, TimeFilter) else parse_time_filter(time_filter)
        column = GRAIN_COLUMNS[grain]
        target_frame = df[tf.mask(df)]
        if len(target_frame) == 0:
            raise InvalidArgumentError("time_filter", tf.to_payload(),
                                       "matched no rows in this dataset.")
        target_labels = sorted(target_frame[column].dropna().unique().tolist())
        if len(target_labels) != 1:
            raise InvalidArgumentError(
                "time_filter", tf.to_payload(),
                f"must resolve to exactly one {grain} for a seasonal comparison, "
                f"matched {len(target_labels)}.")
        target_label = target_labels[0]

        fit = self._fit(df, kpi_key, grain, scope_filters)
        if fit.status != "ok":
            return SeasonalNorm(kpi=kpi_key, label=label, unit=unit, grain=grain,
                               status=fit.status, period=target_label,
                               selection=fit.selection, filters=scope_filters)

        target_phase = _phase(grain, target_label)
        phase_points = fit.phase_counts.get(target_phase, 0)
        if phase_points < MIN_POINTS_PER_PHASE:
            return SeasonalNorm(kpi=kpi_key, label=label, unit=unit, grain=grain,
                               status="phase_not_covered", period=target_label,
                               phase=target_phase, phase_points=phase_points,
                               selection=fit.selection, filters=scope_filters)

        actual_series = self._series.series(df, kpi_key, grain=grain,
                                            time_filter={"type": "all"}, filters=scope_filters)
        actual = next((p.value for p in actual_series.points if p.period == target_label), None)

        seasonal_index = fit.phase_index[target_phase]
        trend_level = fit.intercept + fit.slope * period_ordinal(grain, target_label)
        expected = trend_level + seasonal_index
        deviation_abs = (actual - expected) if (actual is not None and actual == actual) else None
        deviation_pct = pct_change(actual, expected) if deviation_abs is not None else None

        sigma = fit.residual_sigma
        if sigma > 0:
            robust_z = (deviation_abs / sigma) if deviation_abs is not None else None
            is_normal = (abs(robust_z) < k) if robust_z is not None else None
        else:
            # A perfectly seasonal, noiseless fit: any real deviation at all
            # is unambiguous, and there is no meaningful sigma to divide by --
            # report that honestly (`robust_z=None`) rather than a fabricated
            # infinity, and judge normality directly against the deviation.
            robust_z = None
            is_normal = (abs(deviation_abs) < 1e-9) if deviation_abs is not None else None
        band = Band(median=expected, sigma=sigma, lower=expected - k * sigma, upper=expected + k * sigma)
        is_seasonal = fit.acf_at_cycle >= SEASONAL_ACF_MIN

        return SeasonalNorm(
            kpi=kpi_key, label=label, unit=unit, grain=grain, status="ok",
            period=target_label, phase=target_phase, phase_points=phase_points,
            actual=actual, trend_level=trend_level, seasonal_index=seasonal_index,
            expected=expected, deviation_abs=deviation_abs, deviation_pct=deviation_pct,
            robust_z=robust_z, band=band, is_normal_for_phase=is_normal,
            is_seasonal=is_seasonal, seasonal_strength=fit.seasonal_strength,
            selection=fit.selection, filters=scope_filters)
