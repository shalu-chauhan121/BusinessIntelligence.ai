"""
`detect_trend` / `detect_changepoint` -- is a KPI drifting, and when did it
break.

`detect_trend` is genuinely new maths: no `polyfit`, `linregress`,
`theilslopes`, `curve_fit`, `OLS`, or any other trend-fitting call exists
anywhere in `backend/app`. Without it, "is churn getting better or worse?" --
a Tier-3 question -- has no code path, and every Tier-5 scan needs it to tell
a sustained drift from a one-quarter blip.

`requirements.txt` carries numpy and pandas only, no `scipy`. The estimator
is Theil-Sen -- the median of every pairwise slope `(y_j - y_i) / (x_j - x_i)`
-- fit by hand in a few lines rather than adding a dependency six batches
before the master plan calls for one (`scipy` is sequenced at X3, where
p-values and bootstrap CIs genuinely need it). Theil-Sen is not a stopgap
choice: a single outlier moves at most one of the `n` points a pairwise slope
touches, so the *median* slope barely moves, where a least-squares fit
(`numpy.polyfit`) can flip sign on the same series. It also matches the
median/MAD estimator already used everywhere else in this codebase
(`drivers.robust_sigma`, `engines/drivers.py:175`), so a caller reading both a
trend and a significance verdict is reading the same kind of statistic twice,
not two unrelated ones.

The x-axis is each period's **calendar ordinal** (`series.period_ordinal`),
not its position in the list of points. `SeriesEngine.series` reports a gap
rather than filling it (`agent/series.py`), so a series missing its middle
quarter has only 13 points for a 14-quarter span -- indexing those points
0..12 would silently compress the gap into a single step and inflate the
fitted slope. The ordinal keeps the true spacing.

`detect_changepoint` wraps `analysis.detect_onset` (`engines/analysis.py:37`),
which is already prose-free -- its success dict is entirely numbers and
enums. It has one real defect worth closing: **it returns a bare `None` for
three structurally different reasons** -- too few points, a genuinely
degenerate baseline (median and MAD and std all zero), and a real "nothing
broke" answer -- and a caller cannot tell which happened. This module
re-derives the same pre-conditions `detect_onset` checks internally so the
three collapse into distinct typed statuses instead of one uninformative
`None`. It also validates `direction`, which `detect_onset` does not (only
`"down"` is ever tested there; anything else, including a typo, is silently
treated as `"up"`), and adds `direction="any"` so a general "when did this
break" question does not have to guess the sign first.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..config import get_settings
from ..engines.analysis import detect_onset
from ..engines.drivers import robust_sigma
from ..engines.metrics import safe
from .contract_api import ContractAPI
from .errors import InvalidArgumentError
from .series import Series, SeriesEngine, period_ordinal
from .timefilter import TimeFilter

DIRECTIONS = ("down", "up", "any")


def theil_sen(xs: np.ndarray, ys: np.ndarray) -> Tuple[float, float]:
    """(slope, intercept) -- the median of every pairwise slope
    `(y_j - y_i) / (x_j - x_i)`, then the median residual as the intercept.

    One outlier touches at most n-1 of the n*(n-1)/2 pairs, so the median
    barely moves -- the whole reason this estimator was chosen over least
    squares (see the module docstring). Lifted out of `detect_trend` so
    `agent/seasonality.py` detrends with the identical fit rather than a
    second hand-rolled copy free to drift from this one -- the same reason
    `series.period_ordinal` is a shared function rather than duplicated.
    """
    n = len(xs)
    pairwise = [(ys[j] - ys[i]) / (xs[j] - xs[i])
               for i in range(n) for j in range(i + 1, n) if xs[j] != xs[i]]
    slope = float(np.median(pairwise)) if pairwise else 0.0
    intercept = float(np.median(ys - slope * xs))
    return slope, intercept


# ---------------------------------------------------------------------------
# detect_trend()
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Trend:
    kpi: str
    label: str
    unit: str
    grain: str
    status: str                       # "ok" | "insufficient"
    points: int
    method: str                       # "theil_sen"
    slope_per_period: Optional[float]
    slope_pct_per_period: Optional[float]   # slope over the series' own median level -- scale-free
    total_change_pct: Optional[float]       # first point to last point, across the whole window
    residual_sigma: Optional[float]         # robust sigma of (actual - fitted) -- how well the line fits
    persistence: Optional[float]            # share of consecutive steps agreeing with the slope's sign
    direction: Optional[str]                # "rising" | "falling" | "flat"
    is_improving: Optional[bool]            # polarity-aware: a falling churn rate is improving

    def to_payload(self) -> Dict[str, Any]:
        return {
            "kpi": self.kpi, "label": self.label, "unit": self.unit, "grain": self.grain,
            "status": self.status, "points": self.points, "method": self.method,
            "slope_per_period": safe(self.slope_per_period),
            "slope_pct_per_period": safe(self.slope_pct_per_period),
            "total_change_pct": safe(self.total_change_pct),
            "residual_sigma": safe(self.residual_sigma),
            "persistence": safe(self.persistence),
            "direction": self.direction, "is_improving": self.is_improving,
        }


class TrendEngine:
    """`detect_trend` / `detect_changepoint`, bound to one dataset's
    `ContractAPI`."""

    def __init__(self, api: ContractAPI):
        self._api = api
        self._series = SeriesEngine(api)

    def detect_trend(self, df: pd.DataFrame, kpi_key: str, grain: str = "quarter",
                     time_filter: Optional[Union[Mapping[str, Any], TimeFilter]] = None,
                     filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
                     min_points: int = 5) -> Trend:
        series = self._series.series(df, kpi_key, grain=grain, time_filter=time_filter,
                                     filters=filters)
        label, unit = self._api.label(kpi_key), self._api.unit(kpi_key)
        points = [p for p in series.points if p.value == p.value]   # drop NaN (a zero-denominator ratio)
        n = len(points)

        if n < min_points:
            return Trend(kpi=kpi_key, label=label, unit=unit, grain=grain,
                        status="insufficient", points=n, method="theil_sen",
                        slope_per_period=None, slope_pct_per_period=None,
                        total_change_pct=None, residual_sigma=None, persistence=None,
                        direction=None, is_improving=None)

        xs = np.array([period_ordinal(grain, p.period) for p in points], dtype=float)
        ys = np.array([p.value for p in points], dtype=float)

        # Theil-Sen fit, shared with `agent/seasonality.py` so both modules
        # detrend the identical way.
        slope, intercept = theil_sen(xs, ys)

        residuals = ys - (intercept + slope * xs)
        _, residual_sigma = robust_sigma(residuals)

        median_level = float(np.median(ys))
        slope_pct = (slope / abs(median_level) * 100.0) if median_level else None

        total_change_pct: Optional[float] = None
        if ys[0] != 0:
            total_change_pct = float((ys[-1] - ys[0]) / abs(ys[0]) * 100.0)

        diffs = np.diff(ys)
        sign = 1 if slope > 0 else (-1 if slope < 0 else 0)
        persistence = float(np.mean(np.sign(diffs) == sign)) if (sign != 0 and len(diffs)) else None

        s = get_settings()
        if slope == 0:
            direction = "flat"
        elif total_change_pct is not None and abs(total_change_pct) < s.min_material_change_pct:
            direction = "flat"
        else:
            direction = "rising" if slope > 0 else "falling"

        is_improving: Optional[bool] = None
        if direction != "flat":
            higher_better = self._api.polarity(kpi_key)
            is_improving = (direction == "rising") if higher_better else (direction == "falling")

        return Trend(
            kpi=kpi_key, label=label, unit=unit, grain=grain, status="ok", points=n,
            method="theil_sen", slope_per_period=slope, slope_pct_per_period=slope_pct,
            total_change_pct=total_change_pct, residual_sigma=residual_sigma,
            persistence=persistence, direction=direction, is_improving=is_improving)

    # -- detect_changepoint() ---------------------------------------------------
    def detect_changepoint(self, df: pd.DataFrame, kpi_key: str, grain: str = "week",
                           time_filter: Optional[Union[Mapping[str, Any], TimeFilter]] = None,
                           filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
                           direction: str = "any", baseline_periods: int = 8,
                           persistence: int = 2, k_sigma: float = 1.0) -> "Changepoint":
        if direction not in DIRECTIONS:
            raise InvalidArgumentError("direction", direction,
                                       "must be one of the supported directions.", list(DIRECTIONS))

        series = self._series.series(df, kpi_key, grain=grain, time_filter=time_filter,
                                     filters=filters)
        label, unit = self._api.label(kpi_key), self._api.unit(kpi_key)
        return changepoint_from_series(series, kpi_key, label, unit, grain, direction,
                                        baseline_periods, persistence, k_sigma)


def changepoint_from_series(series: "Series", kpi_key: str, label: str, unit: str, grain: str,
                             direction: str, baseline_periods: int, persistence: int,
                             k_sigma: float) -> "Changepoint":
    """
    Everything `detect_changepoint` does after building its own series,
    lifted out so a caller that already holds a `Series` -- `TemporalEngine`,
    batching two KPIs through one `SeriesEngine.series_multi` call -- can
    classify it without a second, redundant series build. `direction` is
    assumed already validated by the caller (`detect_changepoint`'s own
    `DIRECTIONS` check, or `TemporalEngine`'s identical one); this function
    does not re-validate it, the same division of labour `theil_sen` uses
    with the fit it was extracted from.
    """
    def _result(status: str, **kw: Any) -> "Changepoint":
        fields = dict(direction=None, period=None, baseline_median=None,
                     baseline_sigma=None, threshold=None, value_at_onset=None,
                     persistence_periods=persistence, baseline_periods=baseline_periods,
                     gaps=series.gaps)
        fields.update(kw)
        return Changepoint(kpi=kpi_key, label=label, unit=unit, grain=grain,
                           status=status, **fields)

    n = len(series.points)
    if n < baseline_periods + persistence + 1:
        return _result("insufficient_history")

    values = np.array([p.value for p in series.points], dtype=float)
    base = values[:baseline_periods]
    base = base[~np.isnan(base)]
    if len(base) < 3:
        return _result("insufficient_history")

    # The same dispersion estimate `detect_onset` computes internally
    # (`analysis.py:53-55`) -- re-derived here only so a genuinely
    # degenerate baseline (median, MAD and std all zero) can be told apart
    # from "ran the whole series and found no breach", which
    # `detect_onset` itself returns as the identical bare `None`.
    med = float(np.median(base))
    mad = float(np.median(np.abs(base - med)))
    sigma = 1.4826 * mad or float(np.std(base, ddof=1)) or abs(med) * 0.05
    if not (sigma == sigma) or sigma <= 0:
        return _result("degenerate_dispersion")

    weeks_df = pd.DataFrame({"week": [p.period for p in series.points],
                             "value": [p.value for p in series.points]})
    candidates = ("down", "up") if direction == "any" else (direction,)

    best_direction: Optional[str] = None
    best_onset: Optional[Dict[str, Any]] = None
    for d in candidates:
        onset = detect_onset(weeks_df, baseline_weeks=baseline_periods, direction=d,
                             persistence=persistence, k_sigma=k_sigma)
        if onset is not None and (best_onset is None or onset["index"] < best_onset["index"]):
            best_direction, best_onset = d, onset

    if best_onset is None:
        return _result("no_changepoint")

    return _result("ok", direction=best_direction, period=best_onset["week"],
                  baseline_median=best_onset["baseline_median"],
                  baseline_sigma=best_onset["baseline_sigma"],
                  threshold=best_onset["threshold"],
                  value_at_onset=best_onset["value_at_onset"],
                  persistence_periods=best_onset["persistence_weeks"],
                  baseline_periods=best_onset["baseline_weeks"])


@dataclass(frozen=True)
class Changepoint:
    kpi: str
    label: str
    unit: str
    grain: str
    status: str                       # "ok" | "insufficient_history" | "degenerate_dispersion" | "no_changepoint"
    direction: Optional[str]          # "down" | "up" -- the direction that actually fired
    period: Optional[str]
    baseline_median: Optional[float]
    baseline_sigma: Optional[float]
    threshold: Optional[float]
    value_at_onset: Optional[float]
    persistence_periods: Optional[int]
    baseline_periods: Optional[int]
    # `Series.gaps` carried through: `detect_onset`'s `index` counts held
    # periods, not elapsed calendar time, so a caller judging how far back an
    # onset really was needs to know whether the series has holes in it.
    gaps: Sequence[str]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "kpi": self.kpi, "label": self.label, "unit": self.unit, "grain": self.grain,
            "status": self.status, "direction": self.direction, "period": self.period,
            "baseline_median": safe(self.baseline_median),
            "baseline_sigma": safe(self.baseline_sigma),
            "threshold": safe(self.threshold),
            "value_at_onset": safe(self.value_at_onset),
            "persistence_periods": self.persistence_periods,
            "baseline_periods": self.baseline_periods,
            "gaps": list(self.gaps),
        }
