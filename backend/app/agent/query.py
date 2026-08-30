"""
`query_kpi` -- the generic retrieval engine that answers Tier 1 and Tier 2 of
the question taxonomy on its own: a single value, a range, a predicate set, a
group-by breakdown, a top-N ranking, all through one call shape.

Every number this module returns goes through `ContractAPI.value`
(`agent/contract_api.py`), never `engines.metrics.compute` directly. That is
the whole point: `CompiledKpi.compute` (`kpi/resolver.py:275`) sums a ratio
KPI's numerator and denominator over the group and divides once, and routing
through the one sanctioned facade is the structural reason a group's ratio
cannot regress into `mean-of-member-ratios` the way the pre-Contract engines
occasionally did. A group's value is always computed straight from that
group's own rows -- never derived by combining other cells -- because that is
the only way the reconciliation invariant (grouped totals sum to the ungrouped
total, for an additive KPI) can hold by construction rather than by care.

`group_by` accepts real dimension columns and the five time-grain columns
`engines.metrics.prepare` writes (`_year`, `_quarter`, `_period`, `_month`,
`_week`) as pseudo-dimensions. That is what turns "revenue in every odd year,
by year" into one call instead of one round-trip per year: a `TimeFilter`
narrows *which* rows are in scope, and a grain in `group_by` says how to split
what's left.

Every argument is validated before anything pandas-shaped happens: an unknown
KPI key, dimension, or filter member becomes a typed, recoverable error
(`agent/errors.py`) a model can act on, never a `KeyError` and never a silent
empty result standing in for a hallucinated request.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

from ..engines.metrics import safe
from .contract_api import ContractAPI
from .errors import InvalidArgumentError, UnknownKpiError
from .timefilter import TimeFilter, TimeSelection
from .timefilter import apply as apply_time_filter
from .timefilter import parse as parse_time_filter

# The pseudo-dimensions `group_by` may address in addition to real dimension
# columns -- every period key `engines.metrics.prepare` writes.
TIME_GRAIN_COLUMNS = frozenset({"_year", "_quarter", "_period", "_month", "_week"})

# A `group_by` this wide is almost certainly a mistake, not a question; capped
# so a high-cardinality combination cannot dump thousands of cells on a model
# that asked a simple question.
DEFAULT_MAX_GROUPS = 500


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


@dataclass(frozen=True)
class QueryCell:
    """
    One row of a `query_kpi` result: one combination of `group_by` members
    (empty for the ungrouped, scalar case), each requested KPI's value over
    exactly the rows belonging to that combination, and how many rows that was.
    """

    group: Tuple[Tuple[str, str], ...]
    values: Mapping[str, float]
    rows: int

    def to_payload(self) -> Dict[str, Any]:
        return {
            "group": {dim: member for dim, member in self.group},
            "values": {k: safe(v) for k, v in self.values.items()},
            "rows": self.rows,
        }


@dataclass(frozen=True)
class QueryResult:
    kpis: Tuple[str, ...]
    group_by: Tuple[str, ...]
    cells: Tuple[QueryCell, ...]
    selection: TimeSelection
    filters: Mapping[str, Tuple[str, ...]]
    total_groups: int
    truncated: bool

    def to_payload(self) -> Dict[str, Any]:
        return {
            "kpis": list(self.kpis),
            "group_by": list(self.group_by),
            "cells": [c.to_payload() for c in self.cells],
            "selection": self.selection.to_payload(),
            "filters": {k: list(v) for k, v in self.filters.items()},
            "total_groups": self.total_groups,
            "truncated": self.truncated,
        }


class QueryEngine:
    """A `query_kpi` implementation bound to one dataset's `ContractAPI`."""

    MAX_GROUPS = DEFAULT_MAX_GROUPS

    def __init__(self, api: ContractAPI):
        self._api = api

    def query(self, df: pd.DataFrame, kpi_keys: Union[str, Sequence[str]],
             time_filter: Union[Mapping[str, Any], TimeFilter],
             filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
             group_by: Optional[Sequence[str]] = None,
             sort_by: Optional[str] = None, order: str = "desc",
             limit: Optional[int] = None) -> QueryResult:
        """
        Aggregate `kpi_keys` over `df`, scoped by `time_filter`, `filters`, and
        split by `group_by`. `group_by=[]` (the default) is the scalar getter.

        Raises `EmptyPeriodError` when `time_filter` matches no rows at all --
        that is a request for a period this dataset does not hold. A `filters`
        combination that matches no rows *after* a non-empty time selection is
        not an error: it is a genuine, reportable zero-row result, because a
        named region simply having no orders in a period is a real answer, not
        a malformed request.
        """
        kpi_keys = self._validate_kpi_keys(df, kpi_keys)
        tf = time_filter if isinstance(time_filter, TimeFilter) else parse_time_filter(time_filter)
        group_by = list(group_by or [])
        self._validate_group_by(group_by)
        filters = dict(filters or {})
        for dim in filters:
            self._api.require_dimension(dim)
        valid_sort = set(kpi_keys) | set(group_by)
        if sort_by is not None and sort_by not in valid_sort:
            raise InvalidArgumentError(
                "sort_by", sort_by,
                "must name one of the requested KPIs or group_by columns.",
                sorted(valid_sort))
        if order not in ("asc", "desc"):
            raise InvalidArgumentError("order", order, "must be 'asc' or 'desc'.", ["asc", "desc"])
        if limit is not None and limit < 0:
            raise InvalidArgumentError("limit", limit, "must be >= 0.")

        frame, selection = apply_time_filter(df, tf)
        # Every filter is resolved against the time-selected frame as it stood
        # *before* any filter narrowed it -- resolving against a
        # progressively-narrowed frame would make two individually valid
        # filters (region=North, product=B) spuriously raise
        # `UnknownMemberError` the moment they no longer co-occur, which is a
        # zero-row result, not an unknown value.
        resolved_filters: Dict[str, Tuple[str, ...]] = {}
        for dim, raw_values in filters.items():
            values = [raw_values] if isinstance(raw_values, str) else list(raw_values)
            resolved_filters[dim] = tuple(
                self._api.resolve_member(dim, v, frame) for v in values)
        for dim, resolved in resolved_filters.items():
            frame = frame[frame[dim].isin(resolved)]

        cells = self._compute_cells(frame, kpi_keys, group_by)
        cells = self._sort_cells(cells, group_by, sort_by, order)

        total_groups = len(cells)
        effective_limit = self.MAX_GROUPS if limit is None else limit
        truncated = total_groups > effective_limit
        cells = tuple(cells[:effective_limit])

        return QueryResult(kpis=tuple(kpi_keys), group_by=tuple(group_by), cells=cells,
                           selection=selection, filters=resolved_filters,
                           total_groups=total_groups, truncated=truncated)

    # -- validation ----------------------------------------------------------
    def _validate_kpi_keys(self, df: pd.DataFrame,
                           kpi_keys: Union[str, Sequence[str]]) -> List[str]:
        keys = [kpi_keys] if isinstance(kpi_keys, str) else list(kpi_keys)
        if not keys:
            raise InvalidArgumentError("kpi_keys", keys, "at least one KPI key is required.")
        available = set(self._api.available_keys(df))
        for key in keys:
            if key not in available:
                raise UnknownKpiError(key, sorted(available))
        return keys

    def _validate_group_by(self, group_by: Sequence[str]) -> None:
        for g in group_by:
            if g in TIME_GRAIN_COLUMNS:
                continue
            self._api.require_dimension(g)

    # -- computation -----------------------------------------------------------
    def _compute_cells(self, frame: pd.DataFrame, kpi_keys: Sequence[str],
                       group_by: Sequence[str]) -> List[QueryCell]:
        if not group_by:
            values = {k: self._api.value(frame, k) for k in kpi_keys}
            return [QueryCell(group=(), values=values, rows=int(len(frame)))]

        cells: List[QueryCell] = []
        if len(frame) == 0:
            return cells
        grouped = frame.groupby(list(group_by), dropna=False, observed=True)
        for key, group_df in grouped:
            key_tuple = key if isinstance(key, tuple) else (key,)
            group = tuple((dim, str(member)) for dim, member in zip(group_by, key_tuple))
            values = {k: self._api.value(group_df, k) for k in kpi_keys}
            cells.append(QueryCell(group=group, values=values, rows=int(len(group_df))))
        return cells

    def _sort_cells(self, cells: List[QueryCell], group_by: Sequence[str],
                    sort_by: Optional[str], order: str) -> List[QueryCell]:
        # A stable base ordering by the group key itself makes every later sort
        # deterministic, including the case where `sort_by` is left unset.
        cells = sorted(cells, key=lambda c: tuple(member for _, member in c.group))
        if sort_by is None:
            return cells
        reverse = order == "desc"
        if sort_by in group_by:
            def dim_value(cell: QueryCell) -> str:
                return dict(cell.group)[sort_by]
            return sorted(cells, key=dim_value, reverse=reverse)

        # NaN sorts last regardless of direction: split, sort the finite
        # values, and append the missing ones in their existing (group-key)
        # order rather than letting `reverse` flip which end they land on.
        finite = [c for c in cells if not _is_missing(c.values.get(sort_by))]
        missing = [c for c in cells if _is_missing(c.values.get(sort_by))]
        finite = sorted(finite, key=lambda c: c.values[sort_by], reverse=reverse)
        return finite + missing
