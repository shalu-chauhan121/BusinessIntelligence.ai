"""
`rank_entities` / `get_distribution` / `cross_tabulate` -- the three
single-period breakdown shapes `query_kpi` (`agent/query.py`) deliberately
does not build in, because each is genuinely richer than a sorted, grouped
query:

- A rank needs an ordinal, a share of the whole, and a summarised tail --
  `QueryResult` is an ordered list with none of those.
- A distribution is a summary *across the members* of a dimension, not a
  slice of rows -- nothing in the codebase computes a percentile, a quantile,
  a histogram or an IQR anywhere (`analysis.py`, `drivers.py`,
  `kpi/profiling.py` all searched; the closest thing is
  `drivers.robust_sigma`'s median + MAD).
- A cross-tab needs three more queries per call -- row totals, column
  totals, the grand total -- each computed from its own row slice rather than
  by summing cells, which is the only way the reconciliation invariant holds
  for a ratio KPI. (The board's "built on `drivers.cell_delta_grid`" is a
  wrong pointer: that function is a *two-period delta* grid with no
  cardinality cap, and it coerces a NaN group total to a real `0.0`
  (`drivers.py:223`) -- the opposite of the convention this module and
  `series.py` follow. It is not used here.)

All three delegate every number to `QueryEngine.query`
(`agent/query.py:108`), exactly as `SeriesEngine` and `ComparisonEngine` do --
inheriting the validation airlock, the ratio-correctness invariant (a group's
value is always computed from that group's own rows, never derived by
combining other cells) and `TimeSelection` for free. The only genuinely new
arithmetic is the rank/share/histogram bookkeeping layered on top.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..engines.drivers import robust_sigma
from ..engines.metrics import safe
from .contract_api import ContractAPI
from .errors import InvalidArgumentError
from .query import TIME_GRAIN_COLUMNS, QueryEngine
from .timefilter import TimeFilter, TimeSelection

DEFAULT_PERCENTILES = (10, 25, 50, 75, 90)
_ORDERS = ("asc", "desc", "best", "worst")


def _percentile_key(p: float) -> str:
    return f"p{int(p)}" if float(p).is_integer() else f"p{p}"


def _resolve_order(order: str, higher_better: bool) -> str:
    """`best`/`worst` are polarity-aware aliases for `asc`/`desc` -- "the five
    worst regions" is expressible without the caller knowing which direction
    is bad for this particular KPI."""
    if order not in _ORDERS:
        raise InvalidArgumentError("order", order, "must be one of the supported orders.",
                                   list(_ORDERS))
    if order in ("asc", "desc"):
        return order
    if order == "best":
        return "desc" if higher_better else "asc"
    return "asc" if higher_better else "desc"


# ---------------------------------------------------------------------------
# rank_entities()
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RankEntry:
    member: str
    value: Optional[float]
    rows: int
    rank: int
    share_of_total_pct: Optional[float]
    cumulative_share_pct: Optional[float]
    tied_with_previous: bool

    def to_payload(self) -> Dict[str, Any]:
        return {
            "member": self.member, "value": safe(self.value), "rows": self.rows,
            "rank": self.rank,
            "share_of_total_pct": safe(self.share_of_total_pct),
            "cumulative_share_pct": safe(self.cumulative_share_pct),
            "tied_with_previous": self.tied_with_previous,
        }


@dataclass(frozen=True)
class OthersSummary:
    count: int
    value: Optional[float]              # None for a non-additive KPI
    share_of_total_pct: Optional[float]

    def to_payload(self) -> Dict[str, Any]:
        return {"count": self.count, "value": safe(self.value),
                "share_of_total_pct": safe(self.share_of_total_pct)}


@dataclass(frozen=True)
class RankResult:
    kpi: str
    label: str
    unit: str
    dimension: str
    order: str                          # as requested ("best"/"worst"/"asc"/"desc")
    resolved_order: str                 # "asc" | "desc", after the polarity alias
    additive: bool
    kind: str
    entries: Tuple[RankEntry, ...]
    others: Optional[OthersSummary]
    total_groups: int
    truncated: bool
    selection: TimeSelection
    filters: Mapping[str, Tuple[str, ...]]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "kpi": self.kpi, "label": self.label, "unit": self.unit,
            "dimension": self.dimension, "order": self.order,
            "resolved_order": self.resolved_order, "additive": self.additive,
            "kind": self.kind,
            "entries": [e.to_payload() for e in self.entries],
            "others": self.others.to_payload() if self.others else None,
            "total_groups": self.total_groups, "truncated": self.truncated,
            "selection": self.selection.to_payload(),
            "filters": {k: list(v) for k, v in self.filters.items()},
        }


# ---------------------------------------------------------------------------
# get_distribution()
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HistogramBucket:
    lower: float
    upper: float
    count: int

    def to_payload(self) -> Dict[str, Any]:
        return {"lower": safe(self.lower), "upper": safe(self.upper), "count": self.count}


@dataclass(frozen=True)
class Distribution:
    kpi: str
    label: str
    unit: str
    dimension: str
    kind: str
    status: str                         # "ok" | "insufficient"
    count: int
    min: Optional[float]
    max: Optional[float]
    mean: Optional[float]
    median: Optional[float]
    robust_sigma: Optional[float]
    iqr: Optional[float]
    spread: Optional[float]
    percentiles: Mapping[str, float]
    percentile_method: str
    histogram: Tuple[HistogramBucket, ...]
    selection: TimeSelection
    filters: Mapping[str, Tuple[str, ...]]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "kpi": self.kpi, "label": self.label, "unit": self.unit,
            "dimension": self.dimension, "kind": self.kind, "status": self.status,
            "count": self.count,
            "min": safe(self.min), "max": safe(self.max), "mean": safe(self.mean),
            "median": safe(self.median), "robust_sigma": safe(self.robust_sigma),
            "iqr": safe(self.iqr), "spread": safe(self.spread),
            "percentiles": {k: safe(v) for k, v in self.percentiles.items()},
            "percentile_method": self.percentile_method,
            "histogram": [b.to_payload() for b in self.histogram],
            "selection": self.selection.to_payload(),
            "filters": {k: list(v) for k, v in self.filters.items()},
        }


# ---------------------------------------------------------------------------
# cross_tabulate()
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CrossTabCell:
    a: str
    b: str
    value: Optional[float]
    rows: int

    def to_payload(self) -> Dict[str, Any]:
        return {"a": self.a, "b": self.b, "value": safe(self.value), "rows": self.rows}


@dataclass(frozen=True)
class CrossTab:
    kpi: str
    label: str
    unit: str
    dim_a: str
    dim_b: str
    kind: str
    additive: bool
    members_a: Tuple[str, ...]
    members_b: Tuple[str, ...]
    cells: Tuple[CrossTabCell, ...]
    row_totals: Mapping[str, Optional[float]]
    col_totals: Mapping[str, Optional[float]]
    grand_total: Optional[float]
    total_groups: int
    truncated: bool
    selection: TimeSelection
    filters: Mapping[str, Tuple[str, ...]]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "kpi": self.kpi, "label": self.label, "unit": self.unit,
            "dim_a": self.dim_a, "dim_b": self.dim_b, "kind": self.kind,
            "additive": self.additive,
            "members_a": list(self.members_a), "members_b": list(self.members_b),
            "cells": [c.to_payload() for c in self.cells],
            "row_totals": {k: safe(v) for k, v in self.row_totals.items()},
            "col_totals": {k: safe(v) for k, v in self.col_totals.items()},
            "grand_total": safe(self.grand_total),
            "total_groups": self.total_groups, "truncated": self.truncated,
            "selection": self.selection.to_payload(),
            "filters": {k: list(v) for k, v in self.filters.items()},
        }


class BreakdownEngine:
    """`rank_entities` / `get_distribution` / `cross_tabulate`, bound to one
    dataset's `ContractAPI`."""

    def __init__(self, api: ContractAPI):
        self._api = api
        self._query = QueryEngine(api)

    # -- rank_entities() ---------------------------------------------------
    def rank_entities(self, df: pd.DataFrame, kpi_key: str, dimension: str,
                      time_filter: Union[Mapping[str, Any], TimeFilter],
                      order: str = "desc", limit: Optional[int] = 10,
                      filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None
                      ) -> RankResult:
        if dimension in TIME_GRAIN_COLUMNS:
            raise InvalidArgumentError(
                "dimension", dimension,
                "is a time grain, not an entity dimension -- use get_timeseries "
                "for a KPI's history instead.",
                [d.name for d in self._api.list_dimensions()])
        if limit is not None and limit < 1:
            raise InvalidArgumentError("limit", limit, "must be >= 1.")

        higher_better = self._api.polarity(kpi_key)
        resolved_order = _resolve_order(order, higher_better)

        full = self._query.query(df, [kpi_key], time_filter, filters=filters,
                                 group_by=[dimension], sort_by=kpi_key, order=resolved_order)
        grand = self._query.query(df, [kpi_key], time_filter, filters=filters, group_by=[])
        grand_total = grand.cells[0].values[kpi_key]
        additive = self._api.is_additive(kpi_key)
        share_valid = additive and grand_total == grand_total and grand_total != 0

        entries: List[RankEntry] = []
        cumulative = 0.0
        prev_value: Optional[float] = None
        rank = 0
        for cell in full.cells:
            value = cell.values[kpi_key]
            member = dict(cell.group)[dimension]
            share: Optional[float] = None
            if share_valid and value == value:
                share = value / grand_total * 100.0
            tied = (prev_value is not None and value == value
                   and prev_value == prev_value and value == prev_value)
            if not tied:
                rank += 1
            if share is not None:
                cumulative += share
            entries.append(RankEntry(
                member=member, value=value, rows=cell.rows, rank=rank,
                share_of_total_pct=share,
                cumulative_share_pct=cumulative if share is not None else None,
                tied_with_previous=tied))
            prev_value = value

        if limit is None:
            shown, tail = entries, []
        else:
            shown, tail = entries[:limit], entries[limit:]

        others: Optional[OthersSummary] = None
        if tail:
            tail_value = sum(e.value for e in tail if e.value == e.value) if additive else None
            tail_share = (sum(e.share_of_total_pct for e in tail if e.share_of_total_pct is not None)
                         if share_valid else None)
            others = OthersSummary(count=len(tail), value=tail_value, share_of_total_pct=tail_share)

        return RankResult(
            kpi=kpi_key, label=self._api.label(kpi_key), unit=self._api.unit(kpi_key),
            dimension=dimension, order=order, resolved_order=resolved_order,
            additive=additive, kind=self._api.kind(kpi_key),
            entries=tuple(shown), others=others,
            total_groups=full.total_groups, truncated=bool(tail) or full.truncated,
            selection=full.selection, filters=full.filters)

    # -- get_distribution() ------------------------------------------------
    def get_distribution(self, df: pd.DataFrame, kpi_key: str, dimension: str,
                         time_filter: Union[Mapping[str, Any], TimeFilter],
                         filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
                         percentiles: Sequence[float] = DEFAULT_PERCENTILES,
                         bins: int = 10, min_members: int = 3) -> Distribution:
        result = self._query.query(df, [kpi_key], time_filter, filters=filters,
                                   group_by=[dimension])
        values = [c.values[kpi_key] for c in result.cells if c.values[kpi_key] == c.values[kpi_key]]
        count = len(values)
        label, unit, kind = self._api.label(kpi_key), self._api.unit(kpi_key), self._api.kind(kpi_key)

        if count < min_members:
            return Distribution(
                kpi=kpi_key, label=label, unit=unit, dimension=dimension, kind=kind,
                status="insufficient", count=count,
                min=None, max=None, mean=None, median=None, robust_sigma=None,
                iqr=None, spread=None, percentiles={}, percentile_method="numpy_linear",
                histogram=(), selection=result.selection, filters=result.filters)

        arr = np.asarray(values, dtype=float)
        pct_values = np.percentile(arr, list(percentiles))
        pct_map = {_percentile_key(p): float(v) for p, v in zip(percentiles, pct_values)}
        p25, p75 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
        median, sigma = robust_sigma(values)
        counts, edges = np.histogram(arr, bins=bins)
        histogram = tuple(
            HistogramBucket(lower=float(edges[i]), upper=float(edges[i + 1]), count=int(counts[i]))
            for i in range(len(counts)))

        return Distribution(
            kpi=kpi_key, label=label, unit=unit, dimension=dimension, kind=kind,
            status="ok", count=count,
            min=float(arr.min()), max=float(arr.max()), mean=float(arr.mean()),
            median=median, robust_sigma=sigma, iqr=p75 - p25, spread=float(arr.max() - arr.min()),
            percentiles=pct_map, percentile_method="numpy_linear",
            histogram=histogram, selection=result.selection, filters=result.filters)

    # -- cross_tabulate() ------------------------------------------------------
    def cross_tabulate(self, df: pd.DataFrame, kpi_key: str, dim_a: str, dim_b: str,
                       time_filter: Union[Mapping[str, Any], TimeFilter],
                       filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
                       limit_a: Optional[int] = None, limit_b: Optional[int] = None
                       ) -> CrossTab:
        if dim_a == dim_b:
            raise InvalidArgumentError("dim_b", dim_b, "must name a different dimension than dim_a.")

        pivot = self._query.query(df, [kpi_key], time_filter, filters=filters,
                                  group_by=[dim_a, dim_b])
        row_result = self._query.query(df, [kpi_key], time_filter, filters=filters,
                                       group_by=[dim_a])
        col_result = self._query.query(df, [kpi_key], time_filter, filters=filters,
                                       group_by=[dim_b])
        grand_result = self._query.query(df, [kpi_key], time_filter, filters=filters,
                                         group_by=[])
        grand_total = grand_result.cells[0].values[kpi_key]

        # Row/column totals come from their own slice through the airlock --
        # never by summing the pivot's own cells -- which is the only way this
        # stays correct for a ratio KPI (the invariant `series.py` and
        # `query.py` already enforce, carried through here).
        members_a = [dict(c.group)[dim_a] for c in row_result.cells]
        members_b = [dict(c.group)[dim_b] for c in col_result.cells]
        row_totals = {dict(c.group)[dim_a]: c.values[kpi_key] for c in row_result.cells}
        col_totals = {dict(c.group)[dim_b]: c.values[kpi_key] for c in col_result.cells}

        def _select(members: List[str], totals: Mapping[str, Optional[float]],
                   limit: Optional[int]) -> List[str]:
            if limit is None or limit >= len(members):
                return members
            def magnitude(m: str) -> float:
                v = totals[m]
                return abs(v) if v == v else -1.0
            kept = sorted(members, key=magnitude, reverse=True)[:limit]
            return [m for m in members if m in set(kept)]

        selected_a = _select(members_a, row_totals, limit_a)
        selected_b = _select(members_b, col_totals, limit_b)

        pivot_map = {(dict(c.group)[dim_a], dict(c.group)[dim_b]): c for c in pivot.cells}
        cells: List[CrossTabCell] = []
        for a in selected_a:
            for b in selected_b:
                cell = pivot_map.get((a, b))
                if cell is None:
                    cells.append(CrossTabCell(a=a, b=b, value=None, rows=0))
                else:
                    cells.append(CrossTabCell(a=a, b=b, value=cell.values[kpi_key], rows=cell.rows))

        truncated = (pivot.truncated
                    or len(selected_a) < len(members_a)
                    or len(selected_b) < len(members_b))

        return CrossTab(
            kpi=kpi_key, label=self._api.label(kpi_key), unit=self._api.unit(kpi_key),
            dim_a=dim_a, dim_b=dim_b, kind=self._api.kind(kpi_key),
            additive=self._api.is_additive(kpi_key),
            members_a=tuple(selected_a), members_b=tuple(selected_b), cells=tuple(cells),
            row_totals={a: row_totals[a] for a in selected_a},
            col_totals={b: col_totals[b] for b in selected_b},
            grand_total=grand_total, total_groups=pivot.total_groups, truncated=truncated,
            selection=pivot.selection, filters=pivot.filters)
