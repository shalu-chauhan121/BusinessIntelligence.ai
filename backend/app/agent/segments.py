"""
`compare_segments` / `compare_cohorts` / `segment_by_behavior` -- how one
named member of a dimension stacks up against another, how one arbitrary
group of rows compares to another, and which members held up versus which
did not.

**`compare_segments` does not wrap `analysis.member_change_table`
(`engines/analysis.py:114`), superseding the board's pointer** -- for the
same reason `breakdown.py` rejected `drivers.cell_delta_grid` and
`correlate.py` rejected `analysis.correlate`: that function returns a bare
DataFrame with no `TimeSelection`, no validation airlock, string-keyed
`{metric}__cur/__base/__chg` columns, and a `resolver` positional C2 only
just made mandatory throughout the codebase. Built instead on one
`QueryEngine.query(group_by=[dimension], filters={dimension: [a, b]})` call
per period -- putting both members inside the *filter*, not just the
`group_by`, means the query only ever produces at most two cells, so
comparing two members of a 500-member dimension costs the same as comparing
two members of a five-member one.

**Antisymmetry is not one field, it is three, because it cannot be one
field.** `pct_change(a, b)` (`engines/metrics.py:454`) is not antisymmetric:
`a/b - 1 != -(b/a - 1)`. So `SegmentGap` carries `gap_abs` (`a - b`, exact and
antisymmetric), `relative_gap_pct` (symmetric percent difference against the
mean magnitude of both sides, also antisymmetric), and `a_vs_b_pct`
(`pct_change(a, b)`, deliberately **not** antisymmetric, because "APAC is 20%
below EMEA" is the phrasing a reader expects and it needs a named
denominator). Swapping `member_a`/`member_b` must negate the first two and
flip `direction`, and must **not** simply negate the third -- both properties
are pinned by tests, not left to be discovered later.

**Cohort overlap is rejected on row-set intersection, not on comparing filter
dicts.** `{region: North}` and `{segment: Enterprise}` share no key yet can
share rows; comparing the dicts would miss exactly that case, and a
"difference" between two groups that are partly the same rows is not a
difference a model should report without caveat. The check runs once per
requested time window (period_a, and again for period_b when a two-period
comparison is asked for), because whether two filters' row sets intersect can
genuinely depend on which period is in scope.

**`segment_by_behavior` must not be built on `decompose_dimension`**
(`engines/observe.py:292`), unlike every other module in this layer that
reuses it. That function coerces an absent member's value to `0.0`
(`observe.py:314-315`) before computing anything, which is exactly the
distinction this tool exists to preserve: a member with zero *rows* in the
baseline period is not the same claim as a member that grew from a real
zero, and `metrics.prepare`'s own `fillna(0.0)` (`metrics.py:344`) already
blurs that distinction one layer down for the raw metric columns. Band
*classification* here is decided purely from cell presence -- a member
missing from one period's grouped cells is `appeared`/`disappeared`,
never scored against a fabricated zero baseline that would report it as
"grew 999900%".

The dollar-value `change_abs` a member contributes is a different question
from its band, and *is* well-defined against an implicit zero, the same way
a genuinely new product's revenue is really, fully, part of the total
change: for an `appeared`/`disappeared` member on an additive KPI,
`change_abs` compares the real side against `0.0` on the missing side, while
`change_pct` is left `None` rather than a percent computed from nothing.
This is what makes the reconciliation invariant hold exactly:
`sum(member.change_abs over every member in every band) == total_change_abs`
for an additive KPI, algebraically, because every member's real value on
each side is counted exactly once, whichever side it came from.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

from ..config import get_settings
from ..engines.metrics import pct_change, safe
from .contract_api import ContractAPI
from .errors import InvalidArgumentError
from .query import TIME_GRAIN_COLUMNS, QueryEngine
from .timefilter import TimeFilter, TimeSelection
from .timefilter import apply as apply_time_filter
from .timefilter import parse as parse_time_filter

BANDS = ("grew", "flat", "declined", "appeared", "disappeared", "undetermined")


def _is_unfavourable(change_abs: Optional[float], higher_better: bool) -> Optional[bool]:
    """`None` when there is no real movement to judge -- never a fabricated
    verdict for an `appeared`/`disappeared`/non-additive member."""
    if change_abs is None or change_abs != change_abs:
        return None
    return (change_abs < 0) if higher_better else (change_abs > 0)


def _gap(a_val: float, b_val: float) -> "SegmentGap":
    """
    The three gap measures shared by `compare_segments` and `compare_cohorts`.

    `a_val`/`b_val` may be NaN (a side absent from this window) -- `direction`
    is `"undetermined"` and every numeric field is `None` in that case, never
    a fabricated gap against a missing side.
    """
    if not (a_val == a_val and b_val == b_val):
        return SegmentGap(gap_abs=None, relative_gap_pct=None, a_vs_b_pct=None,
                          direction="undetermined")
    gap_abs = a_val - b_val
    denom = (abs(a_val) + abs(b_val)) / 2.0
    relative_gap_pct = (gap_abs / denom * 100.0) if denom else None
    a_vs_b_pct = pct_change(a_val, b_val)
    if gap_abs > 0:
        direction = "a_higher"
    elif gap_abs < 0:
        direction = "b_higher"
    else:
        direction = "equal"
    return SegmentGap(
        gap_abs=gap_abs, relative_gap_pct=relative_gap_pct,
        a_vs_b_pct=(a_vs_b_pct if a_vs_b_pct == a_vs_b_pct else None),
        direction=direction)


@dataclass(frozen=True)
class SegmentGap:
    gap_abs: Optional[float]
    relative_gap_pct: Optional[float]
    # `pct_change(a, b)` -- deliberately NOT antisymmetric under swapping a/b.
    a_vs_b_pct: Optional[float]
    direction: str   # "a_higher" | "b_higher" | "equal" | "undetermined"

    def to_payload(self) -> Dict[str, Any]:
        return {"gap_abs": safe(self.gap_abs), "relative_gap_pct": safe(self.relative_gap_pct),
                "a_vs_b_pct": safe(self.a_vs_b_pct), "direction": self.direction}


# ---------------------------------------------------------------------------
# compare_segments()
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SegmentSide:
    member: str
    value: Optional[float]
    rows: int
    present: bool                       # False: zero rows for this member in this window
    baseline_value: Optional[float] = None
    baseline_rows: Optional[int] = None
    baseline_present: Optional[bool] = None   # None when no period_b was requested at all
    change_abs: Optional[float] = None
    change_pct: Optional[float] = None
    is_unfavourable: Optional[bool] = None

    def to_payload(self) -> Dict[str, Any]:
        return {
            "member": self.member, "value": safe(self.value), "rows": self.rows,
            "present": self.present,
            "baseline_value": safe(self.baseline_value), "baseline_rows": self.baseline_rows,
            "baseline_present": self.baseline_present,
            "change_abs": safe(self.change_abs), "change_pct": safe(self.change_pct),
            "is_unfavourable": self.is_unfavourable,
        }


@dataclass(frozen=True)
class SegmentComparison:
    kpi: str
    label: str
    unit: str
    dimension: str
    status: str   # "ok" | "member_absent_a" | "member_absent_b" | "both_absent"
    a: SegmentSide
    b: SegmentSide
    gap: SegmentGap
    divergence_abs: Optional[float]     # (a.change_abs - b.change_abs); None without period_b
    selection: TimeSelection
    baseline_selection: Optional[TimeSelection]
    filters: Mapping[str, Tuple[str, ...]]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "kpi": self.kpi, "label": self.label, "unit": self.unit, "dimension": self.dimension,
            "status": self.status, "a": self.a.to_payload(), "b": self.b.to_payload(),
            "gap": self.gap.to_payload(), "divergence_abs": safe(self.divergence_abs),
            "selection": self.selection.to_payload(),
            "baseline_selection": self.baseline_selection.to_payload()
                                 if self.baseline_selection is not None else None,
            "filters": {k: list(v) for k, v in self.filters.items()},
        }


# ---------------------------------------------------------------------------
# compare_cohorts()
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CohortSide:
    filters: Mapping[str, Tuple[str, ...]]
    value: Optional[float]
    rows: int
    present: bool
    baseline_value: Optional[float] = None
    baseline_rows: Optional[int] = None
    baseline_present: Optional[bool] = None
    change_abs: Optional[float] = None
    change_pct: Optional[float] = None
    is_unfavourable: Optional[bool] = None

    def to_payload(self) -> Dict[str, Any]:
        return {
            "filters": {k: list(v) for k, v in self.filters.items()},
            "value": safe(self.value), "rows": self.rows, "present": self.present,
            "baseline_value": safe(self.baseline_value), "baseline_rows": self.baseline_rows,
            "baseline_present": self.baseline_present,
            "change_abs": safe(self.change_abs), "change_pct": safe(self.change_pct),
            "is_unfavourable": self.is_unfavourable,
        }


@dataclass(frozen=True)
class CohortComparison:
    kpi: str
    label: str
    unit: str
    status: str   # "ok" | "cohort_a_empty" | "cohort_b_empty" | "both_empty"
    a: CohortSide
    b: CohortSide
    gap: SegmentGap
    divergence_abs: Optional[float]
    selection: TimeSelection
    baseline_selection: Optional[TimeSelection]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "kpi": self.kpi, "label": self.label, "unit": self.unit, "status": self.status,
            "a": self.a.to_payload(), "b": self.b.to_payload(), "gap": self.gap.to_payload(),
            "divergence_abs": safe(self.divergence_abs),
            "selection": self.selection.to_payload(),
            "baseline_selection": self.baseline_selection.to_payload()
                                 if self.baseline_selection is not None else None,
        }


# ---------------------------------------------------------------------------
# segment_by_behavior()
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BehaviorMember:
    member: str
    band: str   # one of BANDS
    current: Optional[float]
    baseline: Optional[float]
    change_abs: Optional[float]
    change_pct: Optional[float]
    is_unfavourable: Optional[bool]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "member": self.member, "band": self.band, "current": safe(self.current),
            "baseline": safe(self.baseline), "change_abs": safe(self.change_abs),
            "change_pct": safe(self.change_pct), "is_unfavourable": self.is_unfavourable,
        }


@dataclass(frozen=True)
class BandSummary:
    band: str
    member_count: int
    current: Optional[float]              # None for a non-additive KPI
    baseline: Optional[float]
    change_abs: Optional[float]
    share_of_total_change_pct: Optional[float]
    favourable_count: int
    unfavourable_count: int

    def to_payload(self) -> Dict[str, Any]:
        return {
            "band": self.band, "member_count": self.member_count,
            "current": safe(self.current), "baseline": safe(self.baseline),
            "change_abs": safe(self.change_abs),
            "share_of_total_change_pct": safe(self.share_of_total_change_pct),
            "favourable_count": self.favourable_count,
            "unfavourable_count": self.unfavourable_count,
        }


@dataclass(frozen=True)
class BehaviorSegmentation:
    kpi: str
    label: str
    unit: str
    dimension: str
    kind: str
    additive: bool
    threshold_pct: float
    total_change_abs: Optional[float]
    bands: Mapping[str, BandSummary]
    members: Tuple[BehaviorMember, ...]
    truncated: bool
    total_members: int
    selection: TimeSelection
    baseline_selection: TimeSelection
    filters: Mapping[str, Tuple[str, ...]]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "kpi": self.kpi, "label": self.label, "unit": self.unit, "dimension": self.dimension,
            "kind": self.kind, "additive": self.additive, "threshold_pct": self.threshold_pct,
            "total_change_abs": safe(self.total_change_abs),
            "bands": {k: v.to_payload() for k, v in self.bands.items()},
            "members": [m.to_payload() for m in self.members],
            "truncated": self.truncated, "total_members": self.total_members,
            "selection": self.selection.to_payload(),
            "baseline_selection": self.baseline_selection.to_payload(),
            "filters": {k: list(v) for k, v in self.filters.items()},
        }


class SegmentEngine:
    """`compare_segments` / `compare_cohorts` / `segment_by_behavior`, bound
    to one dataset's `ContractAPI`."""

    def __init__(self, api: ContractAPI):
        self._api = api
        self._query = QueryEngine(api)

    def _select_frame(self, df: pd.DataFrame, tf: TimeFilter,
                      filters: Optional[Mapping[str, Union[str, Sequence[str]]]]
                      ) -> Tuple[pd.DataFrame, TimeSelection]:
        """The same time-then-filter resolution `QueryEngine.query` performs
        (`agent/query.py:143-156`), duplicated here for the same reason
        `quality.py` and `decompose.py` duplicate it: this module needs the
        raw filtered frame itself, to intersect its row index against
        another cohort's, not an aggregated cell."""
        frame, selection = apply_time_filter(df, tf)
        resolved: Dict[str, Tuple[str, ...]] = {}
        for dim, raw_values in (filters or {}).items():
            self._api.require_dimension(dim)
            values = [raw_values] if isinstance(raw_values, str) else list(raw_values)
            resolved[dim] = tuple(self._api.resolve_member(dim, v, frame) for v in values)
        for dim, members in resolved.items():
            frame = frame[frame[dim].isin(members)]
        return frame, selection

    @staticmethod
    def _normalise_filters(filters: Mapping[str, Union[str, Sequence[str]]]
                           ) -> Dict[str, Tuple[str, ...]]:
        out: Dict[str, Tuple[str, ...]] = {}
        for dim, raw in filters.items():
            values = [raw] if isinstance(raw, str) else list(raw)
            out[dim] = tuple(values)
        return out

    def _check_disjoint(self, frame_a: pd.DataFrame, frame_b: pd.DataFrame,
                        filter_b: Mapping[str, Any]) -> None:
        overlap = frame_a.index.intersection(frame_b.index)
        if len(overlap) > 0:
            raise InvalidArgumentError(
                "filter_b", dict(filter_b),
                f"cohorts overlap by {len(overlap)} row(s) in the requested window -- "
                "a cohort cannot be compared against a group that partly contains it. "
                "Narrow filter_a or filter_b so the two cohorts are disjoint.")

    # -- compare_segments() --------------------------------------------------
    def compare_segments(self, df: pd.DataFrame, kpi_key: str, dimension: str,
                         member_a: str, member_b: str,
                         period_a: Union[Mapping[str, Any], TimeFilter],
                         period_b: Optional[Union[Mapping[str, Any], TimeFilter]] = None,
                         filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None
                         ) -> SegmentComparison:
        if dimension in TIME_GRAIN_COLUMNS:
            raise InvalidArgumentError(
                "dimension", dimension,
                "is a time grain, not an entity dimension -- use get_timeseries "
                "for a KPI's history instead.",
                [d.name for d in self._api.list_dimensions()])
        self._api.require_dimension(dimension)

        resolved_a = self._api.resolve_member(dimension, member_a, df)
        resolved_b = self._api.resolve_member(dimension, member_b, df)
        if resolved_a == resolved_b:
            raise InvalidArgumentError("member_b", member_b,
                                       "must name a different member than member_a.")

        label, unit = self._api.label(kpi_key), self._api.unit(kpi_key)
        higher_better = self._api.polarity(kpi_key)
        merged_filters = dict(filters or {})
        merged_filters[dimension] = [resolved_a, resolved_b]

        tf_a = period_a if isinstance(period_a, TimeFilter) else parse_time_filter(period_a)
        result_a = self._query.query(df, [kpi_key], tf_a, filters=merged_filters,
                                     group_by=[dimension])
        cells_a = {dict(c.group)[dimension]: c for c in result_a.cells}

        val_a = cells_a[resolved_a].values[kpi_key] if resolved_a in cells_a else float("nan")
        val_b = cells_a[resolved_b].values[kpi_key] if resolved_b in cells_a else float("nan")
        rows_a = cells_a[resolved_a].rows if resolved_a in cells_a else 0
        rows_b = cells_a[resolved_b].rows if resolved_b in cells_a else 0
        present_a = resolved_a in cells_a
        present_b = resolved_b in cells_a

        baseline_selection = None
        base_val_a = base_val_b = float("nan")
        base_rows_a = base_rows_b = 0
        base_present_a = base_present_b = None
        change_abs_a = change_pct_a = change_abs_b = change_pct_b = None
        unfav_a = unfav_b = None
        divergence_abs = None

        if period_b is not None:
            tf_b = period_b if isinstance(period_b, TimeFilter) else parse_time_filter(period_b)
            result_b = self._query.query(df, [kpi_key], tf_b, filters=merged_filters,
                                         group_by=[dimension])
            cells_b = {dict(c.group)[dimension]: c for c in result_b.cells}
            baseline_selection = result_b.selection

            base_val_a = cells_b[resolved_a].values[kpi_key] if resolved_a in cells_b else float("nan")
            base_val_b = cells_b[resolved_b].values[kpi_key] if resolved_b in cells_b else float("nan")
            base_rows_a = cells_b[resolved_a].rows if resolved_a in cells_b else 0
            base_rows_b = cells_b[resolved_b].rows if resolved_b in cells_b else 0
            base_present_a = resolved_a in cells_b
            base_present_b = resolved_b in cells_b

            if val_a == val_a and base_val_a == base_val_a:
                change_abs_a = val_a - base_val_a
            pct = pct_change(val_a, base_val_a)
            change_pct_a = pct if pct == pct else None
            unfav_a = _is_unfavourable(change_abs_a, higher_better)

            if val_b == val_b and base_val_b == base_val_b:
                change_abs_b = val_b - base_val_b
            pct = pct_change(val_b, base_val_b)
            change_pct_b = pct if pct == pct else None
            unfav_b = _is_unfavourable(change_abs_b, higher_better)

            if change_abs_a is not None and change_abs_b is not None:
                divergence_abs = change_abs_a - change_abs_b

        side_a = SegmentSide(member=resolved_a, value=val_a, rows=rows_a, present=present_a,
                             baseline_value=base_val_a, baseline_rows=base_rows_a,
                             baseline_present=base_present_a, change_abs=change_abs_a,
                             change_pct=change_pct_a, is_unfavourable=unfav_a)
        side_b = SegmentSide(member=resolved_b, value=val_b, rows=rows_b, present=present_b,
                             baseline_value=base_val_b, baseline_rows=base_rows_b,
                             baseline_present=base_present_b, change_abs=change_abs_b,
                             change_pct=change_pct_b, is_unfavourable=unfav_b)

        if present_a and present_b:
            status = "ok"
        elif not present_a and not present_b:
            status = "both_absent"
        elif not present_a:
            status = "member_absent_a"
        else:
            status = "member_absent_b"

        gap = _gap(val_a, val_b)

        return SegmentComparison(
            kpi=kpi_key, label=label, unit=unit, dimension=dimension, status=status,
            a=side_a, b=side_b, gap=gap, divergence_abs=divergence_abs,
            selection=result_a.selection, baseline_selection=baseline_selection,
            filters=result_a.filters)

    # -- compare_cohorts() ---------------------------------------------------
    def compare_cohorts(self, df: pd.DataFrame, kpi_key: str,
                        filter_a: Mapping[str, Union[str, Sequence[str]]],
                        filter_b: Mapping[str, Union[str, Sequence[str]]],
                        period_a: Union[Mapping[str, Any], TimeFilter],
                        period_b: Optional[Union[Mapping[str, Any], TimeFilter]] = None
                        ) -> CohortComparison:
        if not filter_a:
            raise InvalidArgumentError("filter_a", filter_a,
                                       "must name at least one dimension filter.")
        if not filter_b:
            raise InvalidArgumentError("filter_b", filter_b,
                                       "must name at least one dimension filter.")

        label, unit = self._api.label(kpi_key), self._api.unit(kpi_key)
        higher_better = self._api.polarity(kpi_key)

        tf_a = period_a if isinstance(period_a, TimeFilter) else parse_time_filter(period_a)
        frame_a, selection = self._select_frame(df, tf_a, filter_a)
        frame_b, _ = self._select_frame(df, tf_a, filter_b)
        self._check_disjoint(frame_a, frame_b, filter_b)

        val_a = self._api.value(frame_a, kpi_key) if len(frame_a) else float("nan")
        val_b = self._api.value(frame_b, kpi_key) if len(frame_b) else float("nan")
        rows_a, rows_b = len(frame_a), len(frame_b)
        present_a, present_b = rows_a > 0, rows_b > 0

        baseline_selection = None
        base_val_a = base_val_b = float("nan")
        base_rows_a = base_rows_b = 0
        base_present_a = base_present_b = None
        change_abs_a = change_pct_a = change_abs_b = change_pct_b = None
        unfav_a = unfav_b = None
        divergence_abs = None

        if period_b is not None:
            tf_b = period_b if isinstance(period_b, TimeFilter) else parse_time_filter(period_b)
            frame_a_base, baseline_selection = self._select_frame(df, tf_b, filter_a)
            frame_b_base, _ = self._select_frame(df, tf_b, filter_b)
            self._check_disjoint(frame_a_base, frame_b_base, filter_b)

            base_val_a = self._api.value(frame_a_base, kpi_key) if len(frame_a_base) else float("nan")
            base_val_b = self._api.value(frame_b_base, kpi_key) if len(frame_b_base) else float("nan")
            base_rows_a, base_rows_b = len(frame_a_base), len(frame_b_base)
            base_present_a, base_present_b = base_rows_a > 0, base_rows_b > 0

            if val_a == val_a and base_val_a == base_val_a:
                change_abs_a = val_a - base_val_a
            pct = pct_change(val_a, base_val_a)
            change_pct_a = pct if pct == pct else None
            unfav_a = _is_unfavourable(change_abs_a, higher_better)

            if val_b == val_b and base_val_b == base_val_b:
                change_abs_b = val_b - base_val_b
            pct = pct_change(val_b, base_val_b)
            change_pct_b = pct if pct == pct else None
            unfav_b = _is_unfavourable(change_abs_b, higher_better)

            if change_abs_a is not None and change_abs_b is not None:
                divergence_abs = change_abs_a - change_abs_b

        side_a = CohortSide(filters=self._normalise_filters(filter_a), value=val_a, rows=rows_a,
                            present=present_a, baseline_value=base_val_a, baseline_rows=base_rows_a,
                            baseline_present=base_present_a, change_abs=change_abs_a,
                            change_pct=change_pct_a, is_unfavourable=unfav_a)
        side_b = CohortSide(filters=self._normalise_filters(filter_b), value=val_b, rows=rows_b,
                            present=present_b, baseline_value=base_val_b, baseline_rows=base_rows_b,
                            baseline_present=base_present_b, change_abs=change_abs_b,
                            change_pct=change_pct_b, is_unfavourable=unfav_b)

        if present_a and present_b:
            status = "ok"
        elif not present_a and not present_b:
            status = "both_empty"
        elif not present_a:
            status = "cohort_a_empty"
        else:
            status = "cohort_b_empty"

        gap = _gap(val_a, val_b)

        return CohortComparison(
            kpi=kpi_key, label=label, unit=unit, status=status, a=side_a, b=side_b, gap=gap,
            divergence_abs=divergence_abs, selection=selection, baseline_selection=baseline_selection)

    # -- segment_by_behavior() ------------------------------------------------
    def segment_by_behavior(self, df: pd.DataFrame, kpi_key: str, dimension: str,
                            period_a: Union[Mapping[str, Any], TimeFilter],
                            period_b: Union[Mapping[str, Any], TimeFilter],
                            threshold_pct: Optional[float] = None,
                            filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None
                            ) -> BehaviorSegmentation:
        if dimension in TIME_GRAIN_COLUMNS:
            raise InvalidArgumentError(
                "dimension", dimension,
                "is a time grain, not an entity dimension -- use get_timeseries "
                "for a KPI's history instead.",
                [d.name for d in self._api.list_dimensions()])
        self._api.require_dimension(dimension)
        t = threshold_pct if threshold_pct is not None else get_settings().min_material_change_pct
        if t < 0:
            raise InvalidArgumentError("threshold_pct", t, "must be >= 0.")

        label, unit, kind = self._api.label(kpi_key), self._api.unit(kpi_key), self._api.kind(kpi_key)
        additive = self._api.is_additive(kpi_key)
        higher_better = self._api.polarity(kpi_key)

        tf_a = period_a if isinstance(period_a, TimeFilter) else parse_time_filter(period_a)
        tf_b = period_b if isinstance(period_b, TimeFilter) else parse_time_filter(period_b)
        result_a = self._query.query(df, [kpi_key], tf_a, filters=filters, group_by=[dimension])
        result_b = self._query.query(df, [kpi_key], tf_b, filters=filters, group_by=[dimension])
        cells_a = {dict(c.group)[dimension]: c for c in result_a.cells}
        cells_b = {dict(c.group)[dimension]: c for c in result_b.cells}

        total_a = self._query.query(df, [kpi_key], tf_a, filters=filters, group_by=[])
        total_b = self._query.query(df, [kpi_key], tf_b, filters=filters, group_by=[])
        total_a_val = total_a.cells[0].values[kpi_key]
        total_b_val = total_b.cells[0].values[kpi_key]
        total_change_abs = None
        if additive and total_a_val == total_a_val and total_b_val == total_b_val:
            total_change_abs = total_a_val - total_b_val

        all_members = sorted(set(cells_a) | set(cells_b))
        members: List[BehaviorMember] = []
        for m in all_members:
            in_a, in_b = m in cells_a, m in cells_b
            cur = cells_a[m].values[kpi_key] if in_a else float("nan")
            base = cells_b[m].values[kpi_key] if in_b else float("nan")

            if in_a and not in_b:
                band = "appeared"
            elif in_b and not in_a:
                band = "disappeared"
            else:
                pct = pct_change(cur, base)
                if pct != pct:
                    band = "undetermined"
                elif pct >= t:
                    band = "grew"
                elif pct <= -t:
                    band = "declined"
                else:
                    band = "flat"

            if in_a and in_b:
                change_abs = (cur - base) if (cur == cur and base == base) else None
                pct = pct_change(cur, base)
                change_pct = pct if pct == pct else None
            elif additive:
                # The dollar movement is well-defined by comparing the real
                # side against an implicit zero on the missing side -- the
                # same convention `metrics.prepare`'s `fillna(0.0)` applies to
                # the raw metric columns, used here only for the change
                # amount, never for band classification or change_pct.
                cur_for_delta = cur if cur == cur else 0.0
                base_for_delta = base if base == base else 0.0
                change_abs = cur_for_delta - base_for_delta
                change_pct = None
            else:
                change_abs = None
                change_pct = None

            members.append(BehaviorMember(
                member=m, band=band, current=(cur if cur == cur else None),
                baseline=(base if base == base else None),
                change_abs=change_abs, change_pct=change_pct,
                is_unfavourable=_is_unfavourable(change_abs, higher_better)))

        bands: Dict[str, BandSummary] = {}
        for band_name in BANDS:
            band_members = [m for m in members if m.band == band_name]
            count = len(band_members)
            cur_sum = base_sum = change_sum = share = None
            if additive:
                cur_sum = sum(m.current for m in band_members if m.current is not None)
                base_sum = sum(m.baseline for m in band_members if m.baseline is not None)
                change_sum = sum(m.change_abs for m in band_members if m.change_abs is not None)
                if total_change_abs:
                    share = change_sum / total_change_abs * 100.0
            fav = sum(1 for m in band_members if m.is_unfavourable is False)
            unfav = sum(1 for m in band_members if m.is_unfavourable is True)
            bands[band_name] = BandSummary(
                band=band_name, member_count=count, current=cur_sum, baseline=base_sum,
                change_abs=change_sum, share_of_total_change_pct=share,
                favourable_count=fav, unfavourable_count=unfav)

        total_members = max(result_a.total_groups, result_b.total_groups)
        truncated = result_a.truncated or result_b.truncated

        return BehaviorSegmentation(
            kpi=kpi_key, label=label, unit=unit, dimension=dimension, kind=kind,
            additive=additive, threshold_pct=t, total_change_abs=total_change_abs,
            bands=bands, members=tuple(members), truncated=truncated,
            total_members=total_members, selection=result_a.selection,
            baseline_selection=result_b.selection, filters=result_a.filters)
