"""
Arbitrary A/B comparison, significance judgment, and a no-comparison control
band -- the three ways Tier 3 asks "does this change matter?".

`observe.observe` (`engines/observe.py:389`) can only compare a period to
`tf.previous()` or `tf.year_ago()` -- `engines/observe.py:394` is the entire
comparison vocabulary the product has ever had. "Q4 2026 vs Q1 2024" has no
code path. `compare()` below removes that restriction by building each side
through `query.QueryEngine` -- the same scalar getter `query_kpi` uses -- so
any two `TimeFilter`s can be set against each other.

Significance judgment does not get the same freedom, on purpose.
`observe.assess_significance` (`engines/observe.py:136`) works by building a
distribution of *that same transition* across the KPI's history -- the
previous-period delta, or the same-quarter-year-over-year delta -- and asking
where the current change falls in it. That distribution only exists for those
two transitions. `significance()` below still accepts an arbitrary `period_b`,
but only ever answers when it names `period_a`'s adjacent quarter or its
year-ago quarter; any other pair comes back with a typed
`status="unsupported_baseline"` rather than a z-score computed against a
distribution that was never built. This is a deliberate, narrower contract
than `compare()`'s.

`normal_range()` answers "is a 3% move unusual for us" with no comparison at
all, generalising `act.monitoring_threshold` (`engines/act.py:126`) off its
weekly-only restriction and off the templated `rule` sentence it returns.

No tool result in this module carries a sentence. `assess_significance`'s
three prose fields -- `history_note`, `statistical_power`, `dispersion_note`
-- become, respectively: nothing (the existing typed `history_status` already
carries what `history_note` said in words), a typed `power` enum, and a typed
`sigma_floored` boolean. No fact is dropped, only the English.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

from ..config import get_settings
from ..engines.drivers import robust_sigma
from ..engines.metrics import pct_change, safe
from ..engines.observe import assess_significance
from .contract_api import ContractAPI
from .errors import InvalidArgumentError
from .query import QueryEngine
from .series import SERIES_GRAINS, SeriesEngine, Series
from .timefilter import TimeFilter, TimeSelection
from .timefilter import parse as parse_time_filter

SUPPORTED_BASELINES = ("previous_period", "year_over_year")


# ---------------------------------------------------------------------------
# compare()
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PeriodValue:
    value: Optional[float]
    rows: int
    selection: TimeSelection

    def to_payload(self) -> Dict[str, Any]:
        return {"value": safe(self.value), "rows": self.rows,
                "selection": self.selection.to_payload()}


@dataclass(frozen=True)
class Comparison:
    kpi: str
    current: PeriodValue
    baseline: PeriodValue
    change_abs: Optional[float]
    change_pct: Optional[float]
    direction: Optional[str]         # "up" | "down" | "flat" | None (either side NaN)
    is_unfavourable: Optional[bool]
    filters: Mapping[str, Tuple[str, ...]]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "kpi": self.kpi,
            "current": self.current.to_payload(),
            "baseline": self.baseline.to_payload(),
            "change_abs": safe(self.change_abs),
            "change_pct": safe(self.change_pct),
            "direction": self.direction,
            "is_unfavourable": self.is_unfavourable,
            "filters": {k: list(v) for k, v in self.filters.items()},
        }


# ---------------------------------------------------------------------------
# significance()
# ---------------------------------------------------------------------------
def _power(raw: Dict[str, Any]) -> str:
    """Reproduces the four branches at `engines/observe.py:229-240`, from the
    public fields `assess_significance` already returns -- no change to
    `observe.py` is needed to expose this as a typed enum instead of a
    sentence."""
    if raw["history_status"] == "newly_launched":
        return "none"
    if raw["history_status"] != "sufficient_history":
        return "weak"
    if raw["history_points"] < 4:
        return "weak"
    if raw["method"] != "seasonal_robust_z":
        return "moderate"
    return "good"


@dataclass(frozen=True)
class Significance:
    status: str                      # "ok" | "unsupported_baseline"
    kpi: str
    comparison: Optional[str] = None
    current_period: Optional[str] = None
    baseline_period: Optional[str] = None
    method: Optional[str] = None
    history_status: Optional[str] = None
    change_pct: Optional[float] = None
    robust_z: Optional[float] = None
    median_historical_change_pct: Optional[float] = None
    robust_sigma_pct: Optional[float] = None
    median_all_history_pct: Optional[float] = None
    sigma_all_history_pct: Optional[float] = None
    history_points: Optional[int] = None
    same_quarter_points: Optional[int] = None
    z_threshold: Optional[float] = None
    material_threshold_pct: Optional[float] = None
    is_material: Optional[bool] = None
    is_statistically_unusual: Optional[bool] = None
    is_anomaly: Optional[bool] = None
    verdict: Optional[str] = None
    expected_value: Optional[float] = None
    normal_range: Optional[Tuple[Optional[float], Optional[float]]] = None
    historical_changes: Tuple[Dict[str, Any], ...] = ()
    power: Optional[str] = None
    sigma_floored: Optional[bool] = None
    requested_baseline: Any = None
    supported_baselines: Tuple[str, ...] = ()

    @classmethod
    def from_raw(cls, kpi: str, raw: Dict[str, Any]) -> "Significance":
        band = raw.get("normal_range")
        return cls(
            status="ok", kpi=kpi,
            comparison=raw["comparison"],
            current_period=raw["current_period"],
            baseline_period=raw["baseline_period"],
            method=raw["method"],
            history_status=raw["history_status"],
            change_pct=raw["change_pct"],
            robust_z=raw["robust_z"],
            median_historical_change_pct=raw["median_historical_change_pct"],
            robust_sigma_pct=raw["robust_sigma_pct"],
            median_all_history_pct=raw["median_all_history_pct"],
            sigma_all_history_pct=raw["sigma_all_history_pct"],
            history_points=raw["history_points"],
            same_quarter_points=raw["same_quarter_points"],
            z_threshold=raw["z_threshold"],
            material_threshold_pct=raw["material_threshold_pct"],
            is_material=raw["is_material"],
            is_statistically_unusual=raw["is_statistically_unusual"],
            is_anomaly=raw["is_anomaly"],
            verdict=raw["verdict"],
            expected_value=raw["expected_value"],
            normal_range=tuple(band) if band else None,
            historical_changes=tuple(raw["historical_changes"]),
            power=_power(raw),
            sigma_floored=raw["dispersion_note"] is not None,
        )

    @classmethod
    def unsupported(cls, kpi: str, requested_baseline: Any) -> "Significance":
        return cls(status="unsupported_baseline", kpi=kpi,
                   requested_baseline=requested_baseline,
                   supported_baselines=SUPPORTED_BASELINES)

    def to_payload(self) -> Dict[str, Any]:
        payload = {
            "status": self.status,
            "kpi": self.kpi,
            "comparison": self.comparison,
            "current_period": self.current_period,
            "baseline_period": self.baseline_period,
            "method": self.method,
            "history_status": self.history_status,
            "change_pct": safe(self.change_pct),
            "robust_z": safe(self.robust_z),
            "median_historical_change_pct": safe(self.median_historical_change_pct),
            "robust_sigma_pct": safe(self.robust_sigma_pct),
            "median_all_history_pct": safe(self.median_all_history_pct),
            "sigma_all_history_pct": safe(self.sigma_all_history_pct),
            "history_points": self.history_points,
            "same_quarter_points": self.same_quarter_points,
            "z_threshold": self.z_threshold,
            "material_threshold_pct": self.material_threshold_pct,
            "is_material": self.is_material,
            "is_statistically_unusual": self.is_statistically_unusual,
            "is_anomaly": self.is_anomaly,
            "verdict": self.verdict,
            "expected_value": safe(self.expected_value),
            "normal_range": [safe(v) for v in self.normal_range] if self.normal_range else None,
            "historical_changes": [dict(c) for c in self.historical_changes],
            "power": self.power,
            "sigma_floored": self.sigma_floored,
        }
        if self.status == "unsupported_baseline":
            payload["requested_baseline"] = self.requested_baseline
            payload["supported_baselines"] = list(self.supported_baselines)
        return payload


def _quarterly_rows(series: Series) -> List[Dict[str, Any]]:
    """Adapt a quarterly `Series` into the row shape
    `observe.assess_significance` expects -- the same `{year, quarter, period,
    value}` shape `observe.quarterly_series` builds. `safe()` is applied here,
    matching `quarterly_series` (`observe.py:99`), so a ratio KPI's zero-
    denominator NaN becomes `None` before it reaches `assess_significance` --
    exactly as it does today, and required for the regression test to hold
    byte-for-byte."""
    rows = []
    for point in series.points:
        year_str, q_str = point.period.split("-Q")
        rows.append({"year": int(year_str), "quarter": int(q_str),
                     "period": point.period, "value": safe(point.value)})
    return rows


# ---------------------------------------------------------------------------
# normal_range()
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Band:
    median: Optional[float]
    sigma: Optional[float]
    lower: Optional[float]
    upper: Optional[float]

    def to_payload(self) -> Dict[str, Any]:
        return {"median": safe(self.median), "sigma": safe(self.sigma),
                "lower": safe(self.lower), "upper": safe(self.upper)}


@dataclass(frozen=True)
class NormalRange:
    kpi: str
    grain: str
    k: float
    points: int
    status: str                     # "ok" | "insufficient_history"
    level: Optional[Band]           # band over the point values themselves
    change: Optional[Band]          # band over consecutive pct_change between points

    def to_payload(self) -> Dict[str, Any]:
        return {
            "kpi": self.kpi, "grain": self.grain, "k": self.k, "points": self.points,
            "status": self.status,
            "level": self.level.to_payload() if self.level else None,
            "change": self.change.to_payload() if self.change else None,
        }


class ComparisonEngine:
    """`compare_periods` / `assess_significance` / `get_normal_range`, bound
    to one dataset's `ContractAPI`."""

    def __init__(self, api: ContractAPI):
        self._api = api
        self._query = QueryEngine(api)
        self._series = SeriesEngine(api)

    # -- compare() -----------------------------------------------------------
    def compare(self, df: pd.DataFrame, kpi_key: str,
               period_a: Union[Mapping[str, Any], TimeFilter],
               period_b: Union[Mapping[str, Any], TimeFilter],
               filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None
               ) -> Comparison:
        tf_a = period_a if isinstance(period_a, TimeFilter) else parse_time_filter(period_a)
        tf_b = period_b if isinstance(period_b, TimeFilter) else parse_time_filter(period_b)

        current = self._query.query(df, [kpi_key], tf_a, filters=filters, group_by=[])
        baseline = self._query.query(df, [kpi_key], tf_b, filters=filters, group_by=[])
        cur_cell, base_cell = current.cells[0], baseline.cells[0]
        cur_val, base_val = cur_cell.values[kpi_key], base_cell.values[kpi_key]

        both_finite = cur_val == cur_val and base_val == base_val
        change_abs = (cur_val - base_val) if both_finite else float("nan")
        change_pct = pct_change(cur_val, base_val)

        direction: Optional[str] = None
        is_unfavourable: Optional[bool] = None
        if change_abs == change_abs:
            direction = "up" if change_abs > 0 else ("down" if change_abs < 0 else "flat")
            higher_better = self._api.polarity(kpi_key)
            is_unfavourable = (change_abs < 0) if higher_better else (change_abs > 0)

        return Comparison(
            kpi=kpi_key,
            current=PeriodValue(value=cur_val, rows=cur_cell.rows, selection=current.selection),
            baseline=PeriodValue(value=base_val, rows=base_cell.rows, selection=baseline.selection),
            change_abs=change_abs, change_pct=change_pct,
            direction=direction, is_unfavourable=is_unfavourable,
            filters=current.filters,
        )

    # -- significance() -------------------------------------------------------
    def significance(self, df: pd.DataFrame, kpi_key: str,
                     period_a: Union[Mapping[str, Any], TimeFilter],
                     period_b: Optional[Union[Mapping[str, Any], TimeFilter]] = None,
                     filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
                     series: Optional[Series] = None) -> Significance:
        """
        `series`, when given, replaces the quarterly history this call would
        otherwise build itself -- `observe.assess_significance` is a pure
        function over `{year, quarter, period, value}` rows
        (`engines/observe.py:136`), so the series build is the only expensive
        part of this call (measured at ~90% of it). A scan
        (`agent/scan.py`) that already has every KPI's history from one
        batched `SeriesEngine.series_multi` call passes its slice in here
        instead of paying for a second, per-KPI series build. Every branch
        below -- the airlock, the baseline resolution, `Significance.from_raw`
        -- is identical either way, so a scanned verdict is the same object a
        single-KPI call produces.
        """
        tf_a = period_a if isinstance(period_a, TimeFilter) else parse_time_filter(period_a)
        timeframe_a = tf_a.to_timeframe()
        if timeframe_a is None or timeframe_a.quarter is None:
            raise InvalidArgumentError(
                "period_a", tf_a.to_payload(),
                "must resolve to a single quarter (type 'quarter') for a "
                "significance test -- there is no history distribution to "
                "test a full year or a set of periods against.")

        if period_b is None:
            comparison = "previous_period"
        else:
            tf_b = period_b if isinstance(period_b, TimeFilter) else parse_time_filter(period_b)
            timeframe_b = tf_b.to_timeframe()
            if timeframe_b == timeframe_a.previous():
                comparison = "previous_period"
            elif timeframe_b == timeframe_a.year_ago():
                comparison = "year_over_year"
            else:
                return Significance.unsupported(kpi_key, tf_b.to_payload())

        if series is None:
            series = self._series.series(df, kpi_key, grain="quarter",
                                         time_filter={"type": "all"}, filters=filters)
        elif series.grain != "quarter":
            raise InvalidArgumentError(
                "series", series.grain,
                "a precomputed series passed to significance() must be at quarterly grain.")

        rows = _quarterly_rows(series)
        raw = assess_significance(rows, timeframe_a, comparison)
        return Significance.from_raw(kpi_key, raw)

    # -- normal_range() --------------------------------------------------------
    def normal_range(self, df: pd.DataFrame, kpi_key: str, grain: str = "week",
                     filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
                     k: Optional[float] = None, min_points: int = 8) -> NormalRange:
        if grain not in SERIES_GRAINS:
            raise InvalidArgumentError("grain", grain,
                                       "must be one of the supported grains.",
                                       list(SERIES_GRAINS))
        if k is None:
            k = get_settings().anomaly_z_threshold

        series = self._series.series(df, kpi_key, grain=grain,
                                     time_filter={"type": "all"}, filters=filters)
        values = [p.value for p in series.points if p.value == p.value]
        n = len(values)
        if n < min_points:
            return NormalRange(kpi=kpi_key, grain=grain, k=k, points=n,
                               status="insufficient_history", level=None, change=None)

        med, sigma = robust_sigma(values)
        if not (sigma == sigma and sigma > 0):
            return NormalRange(kpi=kpi_key, grain=grain, k=k, points=n,
                               status="insufficient_history", level=None, change=None)
        level = Band(med, sigma, med - k * sigma, med + k * sigma)

        changes = [pct_change(cur, prev) for prev, cur in zip(values, values[1:])]
        changes = [c for c in changes if c == c]
        change_band: Optional[Band] = None
        if changes:
            cmed, csigma = robust_sigma(changes)
            if csigma == csigma and csigma > 0:
                change_band = Band(cmed, csigma, cmed - k * csigma, cmed + k * csigma)

        return NormalRange(kpi=kpi_key, grain=grain, k=k, points=n,
                           status="ok", level=level, change=change_band)
