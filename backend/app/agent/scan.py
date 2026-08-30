"""
`scan_kpis` / `rank_kpis_by_movement` / `scan_anomalies` / `scan_dimension_outliers`
-- what makes Tier 5 reachable: *"What should I be worried about right now?"*,
*"Is anything unusual in the data?"* name no KPI at all, and every tool built
before this batch requires one.

`observe.observe` builds a `kpi_scoreboard` over every KPI
(`engines/observe.py:446-461`), but only as a by-product of investigating one
named KPI, with no significance verdict per row -- just a two-period change.
There is no existing all-KPI sweep with a verdict attached anywhere in
`backend/app`.

**Every scan here batches.** Measured on the retail fixture (8784 rows, 25
KPIs once the contract's auto-generated `*_per_*_rate` KPIs are counted), a
naive per-KPI loop through `ComparisonEngine.significance` costs 477 ms; one
`SeriesEngine.series_multi` call feeding that same significance test costs
109 ms -- and a naive per-member `scan_dimension_outliers` (275 KPI x member
cells on this fixture, the realistic floor) costs 1.64 s against 109 ms
batched, a 15x difference that only grows with a wider dimension. This is not
a micro-optimisation: `QueryEngine._compute_cells` groups once and evaluates
every KPI per cell (`agent/query.py:189-204`), so extra KPIs on one query cost
about 3 ms each, and paying for N independent groupbys instead is the wrong
order of magnitude, not merely slower.

`normal_range` (`agent/compare.py:342`) is deliberately never called inside a
scan -- at 76 ms/KPI it alone would cost more than the rest of a 25-KPI scan
combined.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

from ..engines.drivers import robust_sigma
from ..engines.metrics import pct_change, safe
from .compare import ComparisonEngine
from .contract_api import ContractAPI
from .errors import InvalidArgumentError
from .quality import QualityEngine
from .query import TIME_GRAIN_COLUMNS, QueryEngine
from .series import SeriesEngine
from .timefilter import TimeFilter, TimeSelection
from .trend import Changepoint, Trend, TrendEngine

# Sane, generous caps that keep a scan bounded on a dataset this batch has
# never seen -- module constants, in the style of `breakdown.DEFAULT_PERCENTILES`,
# not `Settings` fields. `config.py` has no scan-shaped setting today, and
# adding one would also mean updating the client echo at
# `api/routes_system.py:33`, which this batch does not otherwise touch.
MAX_SCAN_KPIS = 100
MAX_SCAN_MEMBERS = 200
MAX_ANOMALY_FOLLOWUPS = 5
MIN_HISTORY_CHANGES = 3

ORDER_BY_CHOICES = ("unfavourability", "magnitude")


def _is_unfavourable(change_pct: Optional[float], higher_better: bool) -> Optional[bool]:
    if change_pct is None:
        return None
    return (change_pct < 0) if higher_better else (change_pct > 0)


def _movement_sort_key(change_pct: Optional[float], is_unfavourable: Optional[bool],
                       order_by: str) -> Tuple[int, int, float]:
    """No-data rows always sort last, regardless of `order_by`. `magnitude` is
    largest-first; `unfavourability` puts every unfavourable move ahead of
    every favourable one and only then breaks ties by magnitude -- the
    distinction the board calls out: a KPI moving the wrong way outranks a
    bigger move in the right direction."""
    if change_pct is None:
        return (1, 0, 0.0)
    magnitude = abs(change_pct)
    if order_by == "magnitude":
        return (0, 0, -magnitude)
    unfav_rank = 0 if is_unfavourable else 1
    return (0, unfav_rank, -magnitude)


# ---------------------------------------------------------------------------
# scan_kpis()
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class KpiScanRow:
    kpi: str
    label: str
    unit: str
    kind: str
    higher_is_better: bool
    history_status: Optional[str]
    change_pct: Optional[float]
    robust_z: Optional[float]
    verdict: Optional[str]
    power: Optional[str]
    is_unfavourable: Optional[bool]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "kpi": self.kpi, "label": self.label, "unit": self.unit, "kind": self.kind,
            "higher_is_better": self.higher_is_better,
            "history_status": self.history_status,
            "change_pct": safe(self.change_pct), "robust_z": safe(self.robust_z),
            "verdict": self.verdict, "power": self.power,
            "is_unfavourable": self.is_unfavourable,
        }


@dataclass(frozen=True)
class KpiScan:
    status: str                       # "ok" | "unsupported_baseline"
    comparison: Optional[str]
    rows: Tuple[KpiScanRow, ...]
    considered: int
    truncated: bool
    requested_baseline: Any = None
    supported_baselines: Tuple[str, ...] = ()

    def to_payload(self) -> Dict[str, Any]:
        payload = {
            "status": self.status, "comparison": self.comparison,
            "rows": [r.to_payload() for r in self.rows],
            "considered": self.considered, "truncated": self.truncated,
        }
        if self.status != "ok":
            payload["requested_baseline"] = self.requested_baseline
            payload["supported_baselines"] = list(self.supported_baselines)
        return payload


# ---------------------------------------------------------------------------
# rank_kpis_by_movement()
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RankedKpiRow:
    rank: int
    row: KpiScanRow

    def to_payload(self) -> Dict[str, Any]:
        payload = self.row.to_payload()
        payload["rank"] = self.rank
        return payload


@dataclass(frozen=True)
class RankedKpiScan:
    status: str
    comparison: Optional[str]
    order_by: str
    entries: Tuple[RankedKpiRow, ...]
    considered: int
    truncated: bool
    requested_baseline: Any = None
    supported_baselines: Tuple[str, ...] = ()

    def to_payload(self) -> Dict[str, Any]:
        payload = {
            "status": self.status, "comparison": self.comparison, "order_by": self.order_by,
            "entries": [e.to_payload() for e in self.entries],
            "considered": self.considered, "truncated": self.truncated,
        }
        if self.status != "ok":
            payload["requested_baseline"] = self.requested_baseline
            payload["supported_baselines"] = list(self.supported_baselines)
        return payload


# ---------------------------------------------------------------------------
# scan_anomalies()
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AnomalyRow:
    kpi: str
    label: str
    unit: str
    change_pct: Optional[float]
    robust_z: Optional[float]
    verdict: Optional[str]
    is_unfavourable: Optional[bool]
    trend: Trend
    changepoint: Changepoint

    def to_payload(self) -> Dict[str, Any]:
        return {
            "kpi": self.kpi, "label": self.label, "unit": self.unit,
            "change_pct": safe(self.change_pct), "robust_z": safe(self.robust_z),
            "verdict": self.verdict, "is_unfavourable": self.is_unfavourable,
            "trend": self.trend.to_payload(), "changepoint": self.changepoint.to_payload(),
        }


@dataclass(frozen=True)
class AnomalyScan:
    status: str
    comparison: Optional[str]
    rows: Tuple[AnomalyRow, ...]
    considered: int
    truncated: bool
    followups_truncated: bool = False

    def to_payload(self) -> Dict[str, Any]:
        return {
            "status": self.status, "comparison": self.comparison,
            "rows": [r.to_payload() for r in self.rows],
            "considered": self.considered, "truncated": self.truncated,
            "followups_truncated": self.followups_truncated,
        }


# ---------------------------------------------------------------------------
# scan_dimension_outliers()
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DimensionOutlier:
    member: str
    latest_change_pct: Optional[float]
    robust_z: Optional[float]
    history_points: int
    status: str                       # "ok" | "insufficient"

    def to_payload(self) -> Dict[str, Any]:
        return {
            "member": self.member, "latest_change_pct": safe(self.latest_change_pct),
            "robust_z": safe(self.robust_z), "history_points": self.history_points,
            "status": self.status,
        }


@dataclass(frozen=True)
class DimensionOutlierScan:
    kpi: str
    label: str
    unit: str
    dimension: str
    entries: Tuple[DimensionOutlier, ...]
    total_groups: int
    truncated: bool
    selection: TimeSelection

    def to_payload(self) -> Dict[str, Any]:
        return {
            "kpi": self.kpi, "label": self.label, "unit": self.unit, "dimension": self.dimension,
            "entries": [e.to_payload() for e in self.entries],
            "total_groups": self.total_groups, "truncated": self.truncated,
            "selection": self.selection.to_payload(),
        }


class ScanEngine:
    """`scan_kpis` / `rank_kpis_by_movement` / `scan_anomalies` /
    `scan_dimension_outliers`, bound to one dataset's `ContractAPI`."""

    def __init__(self, api: ContractAPI):
        self._api = api
        self._query = QueryEngine(api)
        self._series = SeriesEngine(api)
        self._compare = ComparisonEngine(api)
        self._quality = QualityEngine(api)
        self._trend = TrendEngine(api)

    # -- scan_kpis() ---------------------------------------------------------
    def scan_kpis(self, df: pd.DataFrame,
                 time_filter: Union[Mapping[str, Any], TimeFilter],
                 baseline: Optional[Union[Mapping[str, Any], TimeFilter]] = None,
                 filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
                 max_kpis: int = MAX_SCAN_KPIS) -> KpiScan:
        """
        Every KPI on the dataset, scored against `observe.assess_significance`
        for the same `(time_filter, baseline)` pair -- one batched
        `SeriesEngine.series_multi` build, then `ComparisonEngine.significance`
        per key against that shared series, so the cost is one groupby, not N.

        A bad `(time_filter, baseline)` pair -- one `significance()` would
        reject as `unsupported_baseline` -- is the same rejection for every KPI,
        so the whole scan short-circuits on the first row rather than repeating
        an identical refusal `max_kpis` times.
        """
        keys = list(self._api.available_keys(df))
        considered = len(keys)
        truncated = considered > max_kpis
        keys = keys[:max_kpis]

        all_series = self._series.series_multi(df, keys, grain="quarter",
                                                time_filter={"type": "all"}, filters=filters)

        rows: List[KpiScanRow] = []
        comparison_label: Optional[str] = None
        for key in keys:
            sig = self._compare.significance(df, key, time_filter, baseline,
                                             filters=filters, series=all_series[key])
            if sig.status != "ok":
                return KpiScan(status=sig.status, comparison=None, rows=(),
                               considered=considered, truncated=truncated,
                               requested_baseline=sig.requested_baseline,
                               supported_baselines=sig.supported_baselines)
            comparison_label = sig.comparison
            higher_better = self._api.polarity(key)
            rows.append(KpiScanRow(
                kpi=key, label=self._api.label(key), unit=self._api.unit(key),
                kind=self._api.kind(key), higher_is_better=higher_better,
                history_status=sig.history_status, change_pct=sig.change_pct,
                robust_z=sig.robust_z, verdict=sig.verdict, power=sig.power,
                is_unfavourable=_is_unfavourable(sig.change_pct, higher_better)))

        return KpiScan(status="ok", comparison=comparison_label, rows=tuple(rows),
                       considered=considered, truncated=truncated)

    # -- rank_kpis_by_movement() -----------------------------------------------
    def rank_kpis_by_movement(self, df: pd.DataFrame,
                              time_filter: Union[Mapping[str, Any], TimeFilter],
                              baseline: Optional[Union[Mapping[str, Any], TimeFilter]] = None,
                              filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
                              order_by: str = "unfavourability",
                              limit: Optional[int] = 10,
                              max_kpis: int = MAX_SCAN_KPIS) -> RankedKpiScan:
        if order_by not in ORDER_BY_CHOICES:
            raise InvalidArgumentError("order_by", order_by,
                                       "must be one of the supported orderings.",
                                       list(ORDER_BY_CHOICES))
        scan = self.scan_kpis(df, time_filter, baseline=baseline, filters=filters,
                              max_kpis=max_kpis)
        if scan.status != "ok":
            return RankedKpiScan(status=scan.status, comparison=None, order_by=order_by,
                                 entries=(), considered=scan.considered, truncated=scan.truncated,
                                 requested_baseline=scan.requested_baseline,
                                 supported_baselines=scan.supported_baselines)

        ordered = sorted(scan.rows,
                         key=lambda r: _movement_sort_key(r.change_pct, r.is_unfavourable, order_by))
        if limit is not None:
            ordered = ordered[:limit]
        entries = tuple(RankedKpiRow(rank=i + 1, row=row) for i, row in enumerate(ordered))

        return RankedKpiScan(status="ok", comparison=scan.comparison, order_by=order_by,
                             entries=entries, considered=scan.considered, truncated=scan.truncated)

    # -- scan_anomalies() -----------------------------------------------------
    def scan_anomalies(self, df: pd.DataFrame,
                       time_filter: Union[Mapping[str, Any], TimeFilter],
                       baseline: Optional[Union[Mapping[str, Any], TimeFilter]] = None,
                       filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
                       grain: str = "quarter", max_kpis: int = MAX_SCAN_KPIS,
                       max_followups: int = MAX_ANOMALY_FOLLOWUPS) -> AnomalyScan:
        """
        `scan_kpis`, then `QualityEngine.filter_material` as the noise gate --
        so the scan and the gate can never disagree about what counts as
        material -- then `detect_trend` / `detect_changepoint` only on what
        survives. `normal_range` is deliberately not called here (see the
        module docstring).
        """
        scan = self.scan_kpis(df, time_filter, baseline=baseline, filters=filters,
                              max_kpis=max_kpis)
        if scan.status != "ok":
            return AnomalyScan(status=scan.status, comparison=None, rows=(),
                               considered=scan.considered, truncated=scan.truncated)

        candidates = [{"kpi": r.kpi, "change_pct": r.change_pct, "robust_z": r.robust_z}
                     for r in scan.rows]
        gate = self._quality.filter_material(candidates)
        material = [c.kpi for c in gate.candidates if c.is_material]
        followups_truncated = len(material) > max_followups
        material = material[:max_followups]

        by_kpi = {r.kpi: r for r in scan.rows}
        rows: List[AnomalyRow] = []
        for key in material:
            row = by_kpi[key]
            trend = self._trend.detect_trend(df, key, grain=grain,
                                             time_filter={"type": "all"}, filters=filters)
            changepoint = self._trend.detect_changepoint(df, key, grain="week", filters=filters)
            rows.append(AnomalyRow(kpi=key, label=row.label, unit=row.unit,
                                   change_pct=row.change_pct, robust_z=row.robust_z,
                                   verdict=row.verdict, is_unfavourable=row.is_unfavourable,
                                   trend=trend, changepoint=changepoint))

        rows.sort(key=lambda r: _movement_sort_key(r.change_pct, r.is_unfavourable, "unfavourability"))

        return AnomalyScan(status="ok", comparison=scan.comparison, rows=tuple(rows),
                           considered=scan.considered, truncated=scan.truncated,
                           followups_truncated=followups_truncated)

    # -- scan_dimension_outliers() ---------------------------------------------
    def scan_dimension_outliers(self, df: pd.DataFrame, kpi_key: str, dimension: str,
                                time_filter: Union[Mapping[str, Any], TimeFilter],
                                filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
                                max_members: int = MAX_SCAN_MEMBERS,
                                min_history: int = MIN_HISTORY_CHANGES) -> DimensionOutlierScan:
        """
        One `query_kpi(group_by=[dimension, "_period"])` yields every member's
        whole history in a single pass -- the batched replacement for calling
        `drivers.member_significance` once per member. Each member is scored
        on how far its latest period-over-period change sits from that same
        member's own history of changes, via the house `robust_sigma`
        estimator (`drivers.robust_sigma`) -- the same one `assess_significance`
        and `normal_range` already use.

        `time_filter` scopes the **history window**, exactly as
        `SeriesEngine.series` does -- pass `{"type": "all"}` (or a
        multi-period `TimeFilter` such as `latest`) for a real distribution to
        score against, not a single quarter. A single-quarter filter leaves
        each member with one point and nothing to compare it to, so every
        entry reports `status="insufficient"` rather than a fabricated score.
        """
        if dimension in TIME_GRAIN_COLUMNS:
            raise InvalidArgumentError(
                "dimension", dimension,
                "is a time grain, not an entity dimension -- use get_timeseries "
                "for a KPI's history instead.",
                [d.name for d in self._api.list_dimensions()])

        result = self._query.query(df, [kpi_key], time_filter, filters=filters,
                                   group_by=[dimension, "_period"])

        by_member: Dict[str, List[Tuple[str, float]]] = {}
        for cell in result.cells:
            g = dict(cell.group)
            by_member.setdefault(g[dimension], []).append((g["_period"], cell.values[kpi_key]))

        entries: List[DimensionOutlier] = []
        for member, points in by_member.items():
            points.sort(key=lambda t: t[0])
            values = [v for _, v in points if v == v]

            changes = [pct_change(cur, prev) for prev, cur in zip(values, values[1:])]
            changes = [c for c in changes if c == c]
            if len(changes) < min_history + 1:
                entries.append(DimensionOutlier(member=member, latest_change_pct=None,
                                               robust_z=None, history_points=len(values),
                                               status="insufficient"))
                continue

            latest, history = changes[-1], changes[:-1]
            med, sigma = robust_sigma(history)
            z = (latest - med) / sigma if (sigma == sigma and sigma > 0) else None
            entries.append(DimensionOutlier(member=member, latest_change_pct=latest,
                                           robust_z=z, history_points=len(values), status="ok"))

        entries.sort(key=lambda e: (0 if e.robust_z is not None else 1,
                                    -(abs(e.robust_z) if e.robust_z is not None else 0.0)))
        total_groups = len(entries)
        truncated = total_groups > max_members
        entries = entries[:max_members]

        return DimensionOutlierScan(
            kpi=kpi_key, label=self._api.label(kpi_key), unit=self._api.unit(kpi_key),
            dimension=dimension, entries=tuple(entries), total_groups=total_groups,
            truncated=truncated, selection=result.selection)
