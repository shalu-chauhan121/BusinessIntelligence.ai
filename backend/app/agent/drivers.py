"""
`attribute_dimensions` / `rank_drivers` / `compute_over_index` -- which axis
explains a KPI's movement, which members drive it, and who moved more than
their size implies.

`engines/drivers.py` already holds the real, prose-free arithmetic this
module needs: `shapley_dimension_attribution` is an exact Shapley split over
the explained sum of squares, and `score_components` / `composite_score` /
`axis_weights` are the published, hand-verifiable composite-score primitives
`docs/RANKING.md` documents and `scripts/validate_rca.py` sweeps. Those four
functions are called **verbatim, unchanged** -- reimplementing the scoring
arithmetic here would risk drifting from the one thing `test_shapley.py` and
`test_rca_ground_truth.py` actually validate.

What is rebuilt is only the *loop* around them, `drivers.rank_drivers`
(`engines/drivers.py:513`), because that loop -- not the scoring it calls --
is the shape defect: it only accepts a quarter-label `Timeframe`
(`member_significance` matches `tf.label` against `f"{y}-Q{q}"` strings, so a
full-year or arbitrary-range comparison can never match), it costs two full
dataset scans per candidate member (`member_significance` +
`member_persistence`, each re-scanning `df` from scratch), and it stamps two
computed-prose `*_note` fields onto every row. Here, per-member history is
built with one batched `QueryEngine` groupby per dimension -- the same
technique `scan_dimension_outliers` (`agent/scan.py:384`) already uses for
exactly this shape of problem -- and an arbitrary `TimeFilter` A/B pair works
because "history" is defined relative to the earliest `_period` label the
requested current period actually covers, not a quarter label match.

`attribute_dimensions` does not reuse `drivers.cell_delta_grid`
(`engines/drivers.py:201`) for the same reason `breakdown.py` rejected it: no
cardinality cap, and a missing cell is silently coerced to a real `0.0`
rather than reported. The cross-dimension cell grid is built through
`QueryEngine.query(group_by=<all dimensions>)` instead, which inherits the
`MAX_GROUPS` cap and ratio-correct aggregation -- the one cell-grid convention
kept unchanged from the legacy grid is that a coordinate present in only one
period contributes that period's value against `0.0` on the missing side,
because a member that did not exist last quarter genuinely delta'd from
nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

from ..engines.analysis import detect_onset
from ..engines.drivers import (
    DISPROPORTIONATE_AT,
    MAX_SHAPLEY_DIMENSIONS,
    axis_weights as _axis_weights,
    composite_score as _composite_score,
    rank_dimensions as _rank_dimensions,
    robust_sigma,
    score_components as _score_components,
    shapley_dimension_attribution,
)
from ..engines.metrics import pct_change, safe
from ..engines.observe import Timeframe, slice_period
from .contract_api import ContractAPI
from .decompose import DecomposeEngine, OthersRollup
from .errors import InvalidArgumentError
from .query import QueryEngine
from .timefilter import TimeFilter, TimeSelection
from .timefilter import apply as apply_time_filter
from .timefilter import parse as parse_time_filter

# A candidate pool this wide would fold most of the dataset into "Other" long
# before reaching a member row anyway; kept generous so `rank_drivers` sees
# every real member `decompose_by_dimension` would otherwise still enumerate,
# not so wide it becomes a way to smuggle an unbounded scan back in.
DEFAULT_RANK_LIMIT = 6


# ---------------------------------------------------------------------------
# attribute_dimensions()
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DimensionShare:
    dimension: str
    phi: Optional[float]
    share: Optional[float]
    rank: int

    def to_payload(self) -> Dict[str, Any]:
        return {"dimension": self.dimension, "phi": safe(self.phi),
               "share": safe(self.share), "rank": self.rank}


@dataclass(frozen=True)
class DimensionAttribution:
    kpi: str
    label: str
    unit: str
    kind: str
    additive: bool
    status: str        # "ok" | "no_variation" | "no_dimensions" | "too_many_dimensions" | "no_usable_cells"
    method: str         # "shapley_ess"
    max_dimensions: Optional[int]
    dimensions: Tuple[DimensionShare, ...]
    total_variation: Optional[float]
    mean_delta: Optional[float]
    n_cells: int
    cells_truncated: bool
    selection: TimeSelection
    baseline_selection: TimeSelection
    filters: Mapping[str, Tuple[str, ...]]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "kpi": self.kpi, "label": self.label, "unit": self.unit,
            "kind": self.kind, "additive": self.additive,
            "status": self.status, "method": self.method,
            "max_dimensions": self.max_dimensions,
            "dimensions": [d.to_payload() for d in self.dimensions],
            "total_variation": safe(self.total_variation),
            "mean_delta": safe(self.mean_delta),
            "n_cells": self.n_cells, "cells_truncated": self.cells_truncated,
            "selection": self.selection.to_payload(),
            "baseline_selection": self.baseline_selection.to_payload(),
            "filters": {k: list(v) for k, v in self.filters.items()},
        }


# ---------------------------------------------------------------------------
# compute_over_index()
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OverIndexRow:
    member: str
    contribution_pct: Optional[float]
    share_of_baseline_pct: Optional[float]
    over_index: Optional[float]
    change_pct: Optional[float]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "member": self.member, "contribution_pct": safe(self.contribution_pct),
            "share_of_baseline_pct": safe(self.share_of_baseline_pct),
            "over_index": safe(self.over_index), "change_pct": safe(self.change_pct),
        }


@dataclass(frozen=True)
class OverIndexResult:
    kpi: str
    label: str
    unit: str
    dimension: str
    kind: str
    disproportionate_at: float
    rows: Tuple[OverIndexRow, ...]
    others: Optional[OthersRollup]
    selection: TimeSelection
    baseline_selection: TimeSelection
    filters: Mapping[str, Tuple[str, ...]]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "kpi": self.kpi, "label": self.label, "unit": self.unit,
            "dimension": self.dimension, "kind": self.kind,
            "disproportionate_at": self.disproportionate_at,
            "rows": [r.to_payload() for r in self.rows],
            "others": self.others.to_payload() if self.others else None,
            "selection": self.selection.to_payload(),
            "baseline_selection": self.baseline_selection.to_payload(),
            "filters": {k: list(v) for k, v in self.filters.items()},
        }


# ---------------------------------------------------------------------------
# rank_drivers()
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DriverComponents:
    magnitude: Optional[float]
    disproportion: Optional[float]
    significance: Optional[float]
    persistence: Optional[float]

    def to_payload(self) -> Dict[str, Any]:
        return {"magnitude": safe(self.magnitude), "disproportion": safe(self.disproportion),
               "significance": safe(self.significance), "persistence": safe(self.persistence)}


@dataclass(frozen=True)
class DriverRow:
    rank: int
    member: str
    dimension: str
    driver_score: float
    member_score: float
    axis_weight: float
    contribution_pct: Optional[float]
    over_index: Optional[float]
    change_abs: Optional[float]
    change_pct: Optional[float]
    robust_z: Optional[float]
    history_points: int
    onset_period: Optional[str]
    periods_outside_band: Optional[int]
    periods_in_window: Optional[int]
    components: DriverComponents
    weight_used: float
    components_used: Tuple[str, ...]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "rank": self.rank, "member": self.member, "dimension": self.dimension,
            "driver_score": safe(self.driver_score), "member_score": safe(self.member_score),
            "axis_weight": safe(self.axis_weight),
            "contribution_pct": safe(self.contribution_pct), "over_index": safe(self.over_index),
            "change_abs": safe(self.change_abs), "change_pct": safe(self.change_pct),
            "robust_z": safe(self.robust_z), "history_points": self.history_points,
            "onset_period": self.onset_period,
            "periods_outside_band": self.periods_outside_band,
            "periods_in_window": self.periods_in_window,
            "components": self.components.to_payload(),
            "weight_used": safe(self.weight_used),
            "components_used": list(self.components_used),
        }


@dataclass(frozen=True)
class DriverRanking:
    kpi: str
    label: str
    unit: str
    axis_weighted: bool
    persistence_available: bool
    rows: Tuple[DriverRow, ...]
    total_candidates: int
    truncated: bool
    mitigations_excluded: int
    selection: TimeSelection
    baseline_selection: TimeSelection
    filters: Mapping[str, Tuple[str, ...]]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "kpi": self.kpi, "label": self.label, "unit": self.unit,
            "axis_weighted": self.axis_weighted,
            "persistence_available": self.persistence_available,
            "rows": [r.to_payload() for r in self.rows],
            "total_candidates": self.total_candidates, "truncated": self.truncated,
            "mitigations_excluded": self.mitigations_excluded,
            "selection": self.selection.to_payload(),
            "baseline_selection": self.baseline_selection.to_payload(),
            "filters": {k: list(v) for k, v in self.filters.items()},
        }


class DriverEngine:
    """`attribute_dimensions` / `rank_drivers` / `compute_over_index`, bound
    to one dataset's `ContractAPI`."""

    def __init__(self, api: ContractAPI):
        self._api = api
        self._query = QueryEngine(api)
        self._decompose = DecomposeEngine(api)

    # -- frame resolution ------------------------------------------------------
    def _select_frame(self, df: pd.DataFrame, tf: TimeFilter,
                      filters: Optional[Mapping[str, Union[str, Sequence[str]]]]
                      ) -> Tuple[pd.DataFrame, TimeSelection]:
        """The same time-then-filter resolution `QueryEngine.query` performs
        (`agent/query.py:143-156`), copied here as `decompose.py` and
        `quality.py` each already do, because this module needs the raw
        filtered frame itself, not an aggregated cell."""
        frame, selection = apply_time_filter(df, tf)
        resolved: Dict[str, Tuple[str, ...]] = {}
        for dim, raw_values in (filters or {}).items():
            self._api.require_dimension(dim)
            values = [raw_values] if isinstance(raw_values, str) else list(raw_values)
            resolved[dim] = tuple(self._api.resolve_member(dim, v, frame) for v in values)
        for dim, members in resolved.items():
            frame = frame[frame[dim].isin(members)]
        return frame, selection

    def _two_frames(self, df: pd.DataFrame, period_a, period_b,
                    filters) -> Tuple[pd.DataFrame, TimeSelection, pd.DataFrame, TimeSelection]:
        tf_a = period_a if isinstance(period_a, TimeFilter) else parse_time_filter(period_a)
        tf_b = period_b if isinstance(period_b, TimeFilter) else parse_time_filter(period_b)
        cur, sel_a = self._select_frame(df, tf_a, filters)
        base, sel_b = self._select_frame(df, tf_b, filters)
        return cur, sel_a, base, sel_b

    def _resolve_dimensions(self, df: pd.DataFrame,
                            dimensions: Optional[Sequence[str]]) -> List[str]:
        if dimensions is not None:
            dims = [self._api.require_dimension(d) for d in dimensions]
        else:
            dims = [d.name for d in self._api.list_dimensions()]
        return [d for d in dims if d in df.columns]

    # -- attribute_dimensions() -------------------------------------------------
    def attribute_dimensions(self, df: pd.DataFrame, kpi_key: str,
                             period_a: Union[Mapping[str, Any], TimeFilter],
                             period_b: Union[Mapping[str, Any], TimeFilter],
                             filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
                             dimensions: Optional[Sequence[str]] = None,
                             limit_cells: Optional[int] = None) -> DimensionAttribution:
        self._api.require(kpi_key)
        cur, sel_a, base, sel_b = self._two_frames(df, period_a, period_b, filters)
        label, unit = self._api.label(kpi_key), self._api.unit(kpi_key)
        kind, additive = self._api.kind(kpi_key), self._api.is_additive(kpi_key)
        filters_out = dict(filters or {})
        dims = self._resolve_dimensions(df, dimensions)

        def _done(status: str, raw: Optional[Dict[str, Any]] = None,
                  cells_truncated: bool = False, n_cells: int = 0) -> DimensionAttribution:
            raw = raw or {}
            ranked = _rank_dimensions({"phi": raw.get("phi") or {}, "shares": raw.get("shares") or {}})
            dim_rows = tuple(DimensionShare(dimension=r["dimension"], phi=r["phi"],
                                            share=r["share"], rank=r["rank"]) for r in ranked)
            return DimensionAttribution(
                kpi=kpi_key, label=label, unit=unit, kind=kind, additive=additive,
                status=status, method="shapley_ess",
                max_dimensions=MAX_SHAPLEY_DIMENSIONS if status == "too_many_dimensions" else None,
                dimensions=dim_rows, total_variation=raw.get("total_variation"),
                mean_delta=raw.get("mean_delta"), n_cells=n_cells, cells_truncated=cells_truncated,
                selection=sel_a, baseline_selection=sel_b, filters=filters_out)

        if not dims:
            return _done("no_dimensions")
        if len(dims) > MAX_SHAPLEY_DIMENSIONS:
            return _done("too_many_dimensions")

        cur_result = self._query.query(cur, [kpi_key], {"type": "all"}, group_by=dims, limit=limit_cells)
        base_result = self._query.query(base, [kpi_key], {"type": "all"}, group_by=dims, limit=limit_cells)
        # NaN cells (a zero-denominator ratio for that coordinate) are treated
        # as absent rather than as a real value -- same convention as a
        # coordinate genuinely missing from one period.
        cur_map = {c.group: c.values[kpi_key] for c in cur_result.cells
                  if c.values[kpi_key] == c.values[kpi_key]}
        base_map = {c.group: c.values[kpi_key] for c in base_result.cells
                   if c.values[kpi_key] == c.values[kpi_key]}
        cells = []
        for key in set(cur_map) | set(base_map):
            current, baseline = cur_map.get(key, 0.0), base_map.get(key, 0.0)
            row = {dim: member for dim, member in key}
            row.update(current=current, baseline=baseline, delta=current - baseline)
            cells.append(row)

        raw = shapley_dimension_attribution(cells, dims)
        truncated = cur_result.truncated or base_result.truncated
        n_cells = raw.get("n_cells", 0)
        if raw.get("skipped") == "no cells with a usable delta":
            return _done("no_usable_cells", raw=raw, cells_truncated=truncated, n_cells=n_cells)
        if raw.get("total_variation") == 0.0:
            return _done("no_variation", raw=raw, cells_truncated=truncated, n_cells=n_cells)
        return _done("ok", raw=raw, cells_truncated=truncated, n_cells=n_cells)

    # -- compute_over_index() ----------------------------------------------
    def compute_over_index(self, df: pd.DataFrame, kpi_key: str, dimension: str,
                           period_a: Union[Mapping[str, Any], TimeFilter],
                           period_b: Union[Mapping[str, Any], TimeFilter],
                           filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
                           limit: Optional[int] = None) -> OverIndexResult:
        decomposition = self._decompose.decompose_by_dimension(
            df, kpi_key, dimension, period_a, period_b, filters=filters, max_items=limit)
        rows = tuple(
            OverIndexRow(member=m.name, contribution_pct=m.contribution_pct,
                        share_of_baseline_pct=m.share_of_baseline_pct,
                        over_index=m.over_index, change_pct=m.change_pct)
            for m in decomposition.members)
        rows = tuple(sorted(rows, key=lambda r: (r.over_index is None, -(r.over_index or 0.0))))
        return OverIndexResult(
            kpi=kpi_key, label=decomposition.label, unit=decomposition.unit,
            dimension=dimension, kind=decomposition.kind, disproportionate_at=DISPROPORTIONATE_AT,
            rows=rows, others=decomposition.others, selection=decomposition.selection,
            baseline_selection=decomposition.baseline_selection, filters=decomposition.filters)

    # -- rank_drivers(): batched per-member history -----------------------
    def _member_dispersion(self, df: pd.DataFrame, dimension: str, kpi_key: str,
                           boundary_period: str) -> Dict[str, Tuple[Optional[float], Optional[float], int]]:
        """
        Median/sigma of every member's own prior period-over-period % changes,
        from one batched `group_by=[dimension, "_period"]` query -- the
        replacement for calling `member_significance` once per member.

        `boundary_period` excludes any `_period` at or after the requested
        current period, generalising `member_significance`'s quarter-label
        match (`engines/drivers.py:284`) to an arbitrary `TimeFilter`: history
        is simply every held change strictly before the earliest period the
        current comparison covers.

        Returns `member -> (median, sigma, history_points)`, with
        `(None, None, n)` when there are fewer than 3 usable history points or
        the history is degenerate (matches `member_significance`'s own
        thresholds) -- never a fabricated score.
        """
        result = self._query.query(df, [kpi_key], {"type": "all"}, group_by=[dimension, "_period"])
        by_member: Dict[str, List[Tuple[str, float]]] = {}
        for cell in result.cells:
            g = dict(cell.group)
            by_member.setdefault(g[dimension], []).append((g["_period"], cell.values[kpi_key]))

        out: Dict[str, Tuple[Optional[float], Optional[float], int]] = {}
        for member, points in by_member.items():
            points.sort(key=lambda t: t[0])
            changes: List[Tuple[str, float]] = []
            for (prev_period, prev_value), (cur_period, cur_value) in zip(points, points[1:]):
                if prev_value == prev_value and cur_value == cur_value:
                    changes.append((cur_period, pct_change(cur_value, prev_value)))
            history = [c for period, c in changes if period < boundary_period and c == c]
            if len(history) < 3:
                out[member] = (None, None, len(history))
                continue
            med, sigma = robust_sigma(history)
            if not (sigma == sigma and sigma > 0):
                out[member] = (None, None, len(history))
                continue
            sigma *= (1.0 + 1.0 / max(len(history), 1)) ** 0.5   # small-sample inflation
            out[member] = (med, sigma, len(history))
        return out

    def _member_persistence(self, df: pd.DataFrame, dimension: str, kpi_key: str,
                            timeframe_a: Timeframe, direction_by_member: Mapping[str, str],
                            lookback_quarters: int = 3
                            ) -> Dict[str, Tuple[Optional[str], int, int]]:
        """
        When each member turned, and whether it stayed turned -- from one
        batched `group_by=[dimension, "_week"]` query over a shared calendar
        window, the replacement for calling `member_persistence` once per
        member. `direction_by_member` is the member's own change sign, already
        computed by `decompose_by_dimension`, so onset direction is exact
        rather than guessed from the weekly series alone.

        Only callable when the requested current period reduces to a single
        `Timeframe` (a quarter or a year) -- `Timeframe.previous()` needs that
        to walk the lookback window. `rank_drivers` skips this entirely
        otherwise and reports `persistence_available: false`.
        """
        start_tf = timeframe_a
        for _ in range(lookback_quarters):
            start_tf = start_tf.previous()
        lo = slice_period(df, start_tf)["_date"].min()
        period_slice = slice_period(df, timeframe_a)
        hi = period_slice["_date"].max()
        period_start = period_slice["_date"].min()
        if lo != lo or hi != hi or period_start != period_start:
            return {}
        baseline_weeks = max(4, int((period_start - lo).days / 7) - 1)

        window = df[(df["_date"] >= lo) & (df["_date"] <= hi)]
        if len(window) == 0:
            return {}
        result = self._query.query(window, [kpi_key], {"type": "all"}, group_by=[dimension, "_week"])

        by_member: Dict[str, List[Tuple[str, float]]] = {}
        for cell in result.cells:
            g = dict(cell.group)
            by_member.setdefault(g[dimension], []).append((g["_week"], cell.values[kpi_key]))

        period_start_label = str(period_start.date())
        out: Dict[str, Tuple[Optional[str], int, int]] = {}
        for member, points in by_member.items():
            points.sort(key=lambda t: t[0])
            weeks_df = pd.DataFrame({"week": [p for p, _ in points], "value": [v for _, v in points]})
            direction = direction_by_member.get(member, "down")
            onset = detect_onset(weeks_df, baseline_weeks=baseline_weeks, direction=direction)

            period_values = [v for w, v in points if w >= period_start_label]
            weeks_in_period = len(period_values)
            outside = 0
            if onset:
                threshold = onset["threshold"]
                for v in period_values:
                    if v != v:
                        continue
                    if (v < threshold) if direction == "down" else (v > threshold):
                        outside += 1
            out[member] = (onset["week"] if onset else None, outside, weeks_in_period)
        return out

    # -- rank_drivers() ------------------------------------------------------
    def rank_drivers(self, df: pd.DataFrame, kpi_key: str,
                     period_a: Union[Mapping[str, Any], TimeFilter],
                     period_b: Union[Mapping[str, Any], TimeFilter],
                     filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
                     dimensions: Optional[Sequence[str]] = None,
                     limit: Optional[int] = DEFAULT_RANK_LIMIT,
                     weights: Optional[Mapping[str, float]] = None) -> DriverRanking:
        self._api.require(kpi_key)
        if limit is not None and limit < 0:
            raise InvalidArgumentError("limit", limit, "must be >= 0.")

        cur, sel_a, base, sel_b = self._two_frames(df, period_a, period_b, filters)
        label, unit = self._api.label(kpi_key), self._api.unit(kpi_key)
        filters_out = dict(filters or {})
        dims = self._resolve_dimensions(df, dimensions)

        attribution = self.attribute_dimensions(df, kpi_key, period_a, period_b,
                                                filters=filters, dimensions=dims)
        phi = {d.dimension: d.phi for d in attribution.dimensions if d.phi is not None}
        axis = _axis_weights({"phi": phi}) if phi else {}
        axis_weighted = attribution.status in ("ok", "no_variation")

        tf_a = period_a if isinstance(period_a, TimeFilter) else parse_time_filter(period_a)
        timeframe_a = tf_a.to_timeframe()
        persistence_available = timeframe_a is not None
        boundary_period = sel_a.periods[0] if sel_a.periods else None

        candidates: List[Dict[str, Any]] = []
        mitigations_excluded = 0
        for dim in dims:
            decomposition = self._decompose.decompose_by_dimension(
                df, kpi_key, dim, period_a, period_b, filters=filters)
            dispersion = (self._member_dispersion(df, dim, kpi_key, boundary_period)
                         if boundary_period is not None else {})
            persistence_map: Dict[str, Tuple[Optional[str], int, int]] = {}
            if persistence_available:
                direction_by_member = {m.name: ("down" if (m.change_abs or 0.0) < 0 else "up")
                                       for m in decomposition.members}
                persistence_map = self._member_persistence(df, dim, kpi_key, timeframe_a,
                                                           direction_by_member)

            for m in decomposition.members:
                # Members contributing against the movement are mitigations,
                # not drivers -- see `engines/drivers.py:518-522`.
                if m.contribution_pct is None or m.contribution_pct <= 0:
                    mitigations_excluded += 1
                    continue

                robust_z: Optional[float] = None
                history_points = 0
                dispersion_row = dispersion.get(m.name)
                if dispersion_row is not None:
                    med, sigma, history_points = dispersion_row
                    if med is not None and m.change_pct is not None:
                        robust_z = (m.change_pct - med) / sigma

                onset_period: Optional[str] = None
                outside: Optional[int] = None
                in_window: Optional[int] = None
                persistence_row = persistence_map.get(m.name)
                if persistence_row is not None:
                    onset_period, outside, in_window = persistence_row

                components = _score_components(m.contribution_pct, m.over_index, robust_z,
                                              outside, in_window)
                scored = _composite_score(dict(components), dict(weights) if weights else None)
                axis_weight = axis.get(dim, 1.0)

                candidates.append({
                    "member": m.name, "dimension": dim,
                    "driver_score": round(scored["score"] * axis_weight, 4),
                    "member_score": round(scored["score"], 4),
                    "axis_weight": round(axis_weight, 4),
                    "contribution_pct": m.contribution_pct, "over_index": m.over_index,
                    "change_abs": m.change_abs, "change_pct": m.change_pct,
                    "robust_z": round(robust_z, 3) if robust_z is not None else None,
                    "history_points": history_points,
                    "onset_period": onset_period, "periods_outside_band": outside,
                    "periods_in_window": in_window,
                    "components": DriverComponents(**{
                        k: (round(v, 4) if v is not None else None) for k, v in components.items()}),
                    "weight_used": round(scored["weight_used"], 3),
                    "components_used": tuple(scored["components_used"]),
                })

        candidates.sort(key=lambda r: (r["driver_score"], r["contribution_pct"] or 0.0), reverse=True)
        total_candidates = len(candidates)
        truncated = limit is not None and total_candidates > limit
        limited = candidates[:limit] if limit is not None else candidates
        rows = tuple(DriverRow(rank=i + 1, **row) for i, row in enumerate(limited))

        return DriverRanking(
            kpi=kpi_key, label=label, unit=unit, axis_weighted=axis_weighted,
            persistence_available=persistence_available, rows=rows,
            total_candidates=total_candidates, truncated=truncated,
            mitigations_excluded=mitigations_excluded,
            selection=sel_a, baseline_selection=sel_b, filters=filters_out)
