"""
`find_counterexamples` / `test_consistency_across_dimension` / `test_holdout_segments`
-- does a proposed cause hold up cross-sectionally, or only in aggregate.

The board's headline item for this task -- "remove the hardcoded dimension
whitelist `("region","product","channel","segment")` at `contest.py:125`" --
is already done (`395c1bc`); `contest._cross_sectional_dimension` now scores
every dimension on `schema.dimensions` and `test_cross_sectional_dimension.py`
guards it. What is still broken, and still live, is worse:
`analysis.counterexamples` is polarity-blind. It can only ever find members
whose KPI *fell*, so on a lower-is-better KPI it does not merely miss the
answer, it returns the **inverted** one. Measured on the hospital fixture,
`readmission_rate` (a KPI that is bad when it rises) vs `avg_length_of_stay`:
the legacy function names the one department whose readmissions *improved*
as the counterexample, and misses every department whose readmissions
actually got worse while the proposed driver stayed flat -- and
`score_hypothesis` (`contest.py:382`) then charges a real penalty for that
inverted finding. `find_counterexamples` below reads direction from
`ContractAPI.polarity`, never assumes it.

`test_consistency_across_dimension` is not a duplicate of
`correlate_kpis(mode="cross_sectional")` -- the two disagree in their
*conclusion* on the fixture's own headline pair. Retail `region`,
`revenue ~ fulfillment_rate`: pooled Pearson r is **-0.953** (read naively,
"the cause contradicts the hypothesis"), while all four regions in fact moved
the same way on both KPIs -- the r is ranking magnitudes across a one-point
spread in the cause against a thirty-point spread in the KPI, which is
measuring noise, not a relationship. A **member-level sign-agreement rate**
answers a different, and here more honest, question: does each member's move
agree in direction with what the hypothesis predicts. Both numbers are
reported, deliberately, rather than the tool picking one -- the disagreement
between them is itself a finding a model should see.

`test_holdout_segments` derives its two groups from a metric's own per-member
change (a predicate no dimension-value filter dict can express), then
delegates the actual comparison to `SegmentEngine.compare_cohorts` so the
antisymmetry properties `test_segments.py` already pins are inherited rather
than re-derived -- the same reason this module never touches `SegmentGap`'s
internals directly. What `compare_cohorts` reports is a *level* gap, though,
and that is not the holdout question: measured on the retail product split
(`{Product A}` affected vs `{Product B, Product C}` unaffected), the level
gap is `gap_abs = -1,109,482` -- a group-size artefact, since one group has
one member and the other has two -- while the scale-free
difference-in-differences, `did_pct = a.change_pct - b.change_pct`, is
`-25.30` percentage points. `did_pct` is this module's own headline field,
published alongside the inherited comparison rather than replacing it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

from ..config import get_settings
from ..engines.metrics import pct_change, safe
from .contract_api import ContractAPI
from .correlate import Correlation, CorrelationEngine
from .errors import InvalidArgumentError
from .query import QueryEngine
from .segments import CohortComparison, SegmentEngine
from .timefilter import TimeFilter, TimeSelection
from .timefilter import parse as parse_time_filter

CAUSE_DIRECTIONS = ("up", "down", "any")

# Mirrors `contest.MIN_CROSS_SECTION_MEMBERS` (`engines/contest.py:125`), not
# `correlate.DEFAULT_MIN_N` (6) -- no sample fixture has six cross-sectional
# members (retail `region` has 4, `product` 3; hospital `department` 3-4;
# school `grade` 4), so inheriting the six-member floor would make every one
# of this module's tools report `insufficient_n`/`insufficient_members` on
# every dataset the project owns.
MIN_CONSISTENCY_MEMBERS = 3

DEFAULT_COUNTEREXAMPLE_LIMIT = 20


# ---------------------------------------------------------------------------
# shared: per-member change table, one shape three tools consume
# ---------------------------------------------------------------------------
def _member_changes(query: QueryEngine, df: pd.DataFrame, kpi_key: str, cause_kpi: str,
                    dimension: str, tf_a: TimeFilter, tf_b: TimeFilter,
                    filters: Optional[Mapping[str, Union[str, Sequence[str]]]]
                    ) -> Tuple[List[str], Dict[str, float], Dict[str, float],
                              TimeSelection, TimeSelection, Mapping[str, Tuple[str, ...]]]:
    """
    Every member's percent change in `kpi_key` and `cause_kpi` between the
    two periods -- the same two `QueryEngine.query(group_by=[dimension])`
    calls and the same `pct_change` composition `correlate.
    _cross_sectional_correlation` uses, so this module and `correlate_kpis`
    can never disagree about what a member's change was.
    """
    cur = query.query(df, [kpi_key, cause_kpi], tf_a, filters=filters, group_by=[dimension])
    base = query.query(df, [kpi_key, cause_kpi], tf_b, filters=filters, group_by=[dimension])
    cur_map = {dict(c.group)[dimension]: c.values for c in cur.cells}
    base_map = {dict(c.group)[dimension]: c.values for c in base.cells}
    members = sorted(set(cur_map) | set(base_map))

    kpi_chg: Dict[str, float] = {}
    cause_chg: Dict[str, float] = {}
    for m in members:
        kpi_chg[m] = pct_change(cur_map.get(m, {}).get(kpi_key, float("nan")),
                                base_map.get(m, {}).get(kpi_key, float("nan")))
        cause_chg[m] = pct_change(cur_map.get(m, {}).get(cause_kpi, float("nan")),
                                  base_map.get(m, {}).get(cause_kpi, float("nan")))
    return members, kpi_chg, cause_chg, cur.selection, base.selection, cur.filters


def _cause_moved(chg: float, cause_move_pct: float, cause_direction: str) -> bool:
    if cause_direction == "up":
        return chg >= cause_move_pct
    if cause_direction == "down":
        return chg <= -cause_move_pct
    return abs(chg) >= cause_move_pct   # "any" -- moved at all, either way


# ---------------------------------------------------------------------------
# find_counterexamples()
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Counterexample:
    member: str
    kpi_change_pct: Optional[float]
    cause_change_pct: Optional[float]

    def to_payload(self) -> Dict[str, Any]:
        return {"member": self.member, "kpi_change_pct": safe(self.kpi_change_pct),
               "cause_change_pct": safe(self.cause_change_pct)}


@dataclass(frozen=True)
class CounterexampleSet:
    kpi: str
    cause_kpi: str
    label: str
    cause_label: str
    dimension: str
    status: str                       # "ok" | "no_members" | "insufficient_members" | "no_unfavourable_members"
    kpi_higher_better: bool
    kpi_move_pct: float
    cause_move_pct: float
    cause_direction: str
    n_members: int
    n_comparable: int
    n_kpi_unfavourable: int
    n_counterexamples: int
    counterexample_rate: Optional[float]
    counterexamples: Tuple[Counterexample, ...]
    truncated: bool
    selection: TimeSelection
    baseline_selection: TimeSelection
    filters: Mapping[str, Tuple[str, ...]]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "kpi": self.kpi, "cause_kpi": self.cause_kpi,
            "label": self.label, "cause_label": self.cause_label, "dimension": self.dimension,
            "status": self.status, "kpi_higher_better": self.kpi_higher_better,
            "kpi_move_pct": self.kpi_move_pct, "cause_move_pct": self.cause_move_pct,
            "cause_direction": self.cause_direction,
            "n_members": self.n_members, "n_comparable": self.n_comparable,
            "n_kpi_unfavourable": self.n_kpi_unfavourable,
            "n_counterexamples": self.n_counterexamples,
            "counterexample_rate": safe(self.counterexample_rate),
            "counterexamples": [c.to_payload() for c in self.counterexamples],
            "truncated": self.truncated,
            "selection": self.selection.to_payload(),
            "baseline_selection": self.baseline_selection.to_payload(),
            "filters": {k: list(v) for k, v in self.filters.items()},
        }


# ---------------------------------------------------------------------------
# test_consistency_across_dimension()
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MemberAgreement:
    member: str
    kpi_change_pct: Optional[float]
    cause_change_pct: Optional[float]
    bucket: str                       # "same_sign" | "opposite_sign" | "cause_flat" | "kpi_flat" | "undetermined"

    def to_payload(self) -> Dict[str, Any]:
        return {"member": self.member, "kpi_change_pct": safe(self.kpi_change_pct),
               "cause_change_pct": safe(self.cause_change_pct), "bucket": self.bucket}


@dataclass(frozen=True)
class DimensionConsistency:
    kpi: str
    cause_kpi: str
    label: str
    cause_label: str
    dimension: str
    status: str                       # "ok" | "insufficient_members"
    move_pct: float
    n_members: int
    n_same_sign: int
    n_opposite_sign: int
    n_cause_flat: int
    n_kpi_flat: int
    consistency_rate: Optional[float]
    co_movement_strength: Optional[float]
    direction: Optional[str]          # "positive" | "negative" | "mixed"
    members: Tuple[MemberAgreement, ...]
    pooled: Optional[Correlation]
    selection: Optional[TimeSelection]
    baseline_selection: Optional[TimeSelection]
    filters: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "kpi": self.kpi, "cause_kpi": self.cause_kpi,
            "label": self.label, "cause_label": self.cause_label, "dimension": self.dimension,
            "status": self.status, "move_pct": self.move_pct, "n_members": self.n_members,
            "n_same_sign": self.n_same_sign, "n_opposite_sign": self.n_opposite_sign,
            "n_cause_flat": self.n_cause_flat, "n_kpi_flat": self.n_kpi_flat,
            "consistency_rate": safe(self.consistency_rate),
            "co_movement_strength": safe(self.co_movement_strength),
            "direction": self.direction,
            "members": [m.to_payload() for m in self.members],
            "pooled": self.pooled.to_payload() if self.pooled is not None else None,
            "selection": self.selection.to_payload() if self.selection is not None else None,
            "baseline_selection": self.baseline_selection.to_payload()
                                 if self.baseline_selection is not None else None,
            "filters": {k: list(v) for k, v in self.filters.items()},
        }


# ---------------------------------------------------------------------------
# test_holdout_segments()
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HoldoutComparison:
    kpi: str
    cause_kpi: str
    label: str
    cause_label: str
    dimension: str
    status: str                       # "ok" | "no_affected_members" | "no_unaffected_members"
    cause_move_pct: float
    cause_direction: str
    n_affected: int
    n_unaffected: int
    n_undetermined: int
    did_pct: Optional[float]
    verdict: Optional[str]            # "cause_explains_the_gap" | "kpi_moved_regardless" | "undetermined"
    comparison: Optional[CohortComparison]
    filters: Mapping[str, Tuple[str, ...]]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "kpi": self.kpi, "cause_kpi": self.cause_kpi,
            "label": self.label, "cause_label": self.cause_label, "dimension": self.dimension,
            "status": self.status, "cause_move_pct": self.cause_move_pct,
            "cause_direction": self.cause_direction,
            "n_affected": self.n_affected, "n_unaffected": self.n_unaffected,
            "n_undetermined": self.n_undetermined,
            "did_pct": safe(self.did_pct), "verdict": self.verdict,
            "comparison": self.comparison.to_payload() if self.comparison is not None else None,
            "filters": {k: list(v) for k, v in self.filters.items()},
        }


class ConsistencyEngine:
    """`find_counterexamples` / `test_consistency_across_dimension` /
    `test_holdout_segments`, bound to one dataset's `ContractAPI`."""

    def __init__(self, api: ContractAPI):
        self._api = api
        self._query = QueryEngine(api)
        self._correlate = CorrelationEngine(api)
        self._segments = SegmentEngine(api)

    # -- find_counterexamples() --------------------------------------------
    def find_counterexamples(self, df: pd.DataFrame, kpi_key: str, cause_kpi: str, dimension: str,
                             period_a: Union[Mapping[str, Any], TimeFilter],
                             period_b: Union[Mapping[str, Any], TimeFilter],
                             filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
                             kpi_move_pct: Optional[float] = None,
                             cause_move_pct: Optional[float] = None,
                             cause_direction: str = "any",
                             limit: int = DEFAULT_COUNTEREXAMPLE_LIMIT) -> CounterexampleSet:
        self._api.require(kpi_key)
        self._api.require(cause_kpi)
        self._api.require_dimension(dimension)
        if cause_direction not in CAUSE_DIRECTIONS:
            raise InvalidArgumentError("cause_direction", cause_direction,
                                       "must be one of the supported directions.",
                                       list(CAUSE_DIRECTIONS))

        settings = get_settings()
        kpi_move_pct = kpi_move_pct if kpi_move_pct is not None else settings.min_material_change_pct
        cause_move_pct = (cause_move_pct if cause_move_pct is not None
                         else settings.min_material_change_pct)
        higher_better = self._api.polarity(kpi_key)
        label, cause_label = self._api.label(kpi_key), self._api.label(cause_kpi)

        tf_a = period_a if isinstance(period_a, TimeFilter) else parse_time_filter(period_a)
        tf_b = period_b if isinstance(period_b, TimeFilter) else parse_time_filter(period_b)
        members, kpi_chg, cause_chg, sel_a, sel_b, filters_out = _member_changes(
            self._query, df, kpi_key, cause_kpi, dimension, tf_a, tf_b, filters)

        def _bare(status: str, **kw: Any) -> CounterexampleSet:
            fields = dict(n_comparable=0, n_kpi_unfavourable=0, n_counterexamples=0,
                         counterexample_rate=None, counterexamples=(), truncated=False)
            fields.update(kw)
            return CounterexampleSet(
                kpi=kpi_key, cause_kpi=cause_kpi, label=label, cause_label=cause_label,
                dimension=dimension, status=status, kpi_higher_better=higher_better,
                kpi_move_pct=kpi_move_pct, cause_move_pct=cause_move_pct,
                cause_direction=cause_direction, n_members=len(members),
                selection=sel_a, baseline_selection=sel_b, filters=filters_out, **fields)

        if not members:
            return _bare("no_members")

        comparable = [m for m in members if kpi_chg[m] == kpi_chg[m] and cause_chg[m] == cause_chg[m]]
        if not comparable:
            return _bare("insufficient_members")

        def _kpi_worsened(chg: float) -> bool:
            return chg <= -kpi_move_pct if higher_better else chg >= kpi_move_pct

        unfavourable = [m for m in comparable if _kpi_worsened(kpi_chg[m])]
        if not unfavourable:
            return _bare("no_unfavourable_members", n_comparable=len(comparable))

        counters = sorted(m for m in unfavourable
                          if not _cause_moved(cause_chg[m], cause_move_pct, cause_direction))
        truncated = len(counters) > limit
        shown = counters[:limit]
        examples = tuple(Counterexample(member=m, kpi_change_pct=kpi_chg[m],
                                        cause_change_pct=cause_chg[m]) for m in shown)

        return _bare("ok", n_comparable=len(comparable), n_kpi_unfavourable=len(unfavourable),
                    n_counterexamples=len(counters),
                    counterexample_rate=len(counters) / len(unfavourable),
                    counterexamples=examples, truncated=truncated)

    # -- test_consistency_across_dimension() -------------------------------
    def test_consistency_across_dimension(self, df: pd.DataFrame, kpi_key: str, cause_kpi: str,
                                          dimension: str,
                                          period_a: Union[Mapping[str, Any], TimeFilter],
                                          period_b: Union[Mapping[str, Any], TimeFilter],
                                          filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
                                          move_pct: Optional[float] = None,
                                          min_n: int = MIN_CONSISTENCY_MEMBERS
                                          ) -> DimensionConsistency:
        self._api.require(kpi_key)
        self._api.require(cause_kpi)
        self._api.require_dimension(dimension)

        settings = get_settings()
        move_pct = move_pct if move_pct is not None else settings.min_material_change_pct
        label, cause_label = self._api.label(kpi_key), self._api.label(cause_kpi)

        tf_a = period_a if isinstance(period_a, TimeFilter) else parse_time_filter(period_a)
        tf_b = period_b if isinstance(period_b, TimeFilter) else parse_time_filter(period_b)
        members, kpi_chg, cause_chg, sel_a, sel_b, filters_out = _member_changes(
            self._query, df, kpi_key, cause_kpi, dimension, tf_a, tf_b, filters)

        if len(members) < min_n:
            return DimensionConsistency(
                kpi=kpi_key, cause_kpi=cause_kpi, label=label, cause_label=cause_label,
                dimension=dimension, status="insufficient_members", move_pct=move_pct,
                n_members=len(members), n_same_sign=0, n_opposite_sign=0, n_cause_flat=0,
                n_kpi_flat=0, consistency_rate=None, co_movement_strength=None, direction=None,
                members=(), pooled=None, selection=sel_a, baseline_selection=sel_b,
                filters=filters_out)

        rows: List[MemberAgreement] = []
        n_same = n_opp = n_cause_flat = n_kpi_flat = 0
        for m in members:
            k, c = kpi_chg[m], cause_chg[m]
            if not (k == k and c == c):
                bucket = "undetermined"
            elif abs(c) < move_pct:
                bucket = "cause_flat"; n_cause_flat += 1
            elif abs(k) < move_pct:
                bucket = "kpi_flat"; n_kpi_flat += 1
            elif (k > 0) == (c > 0):
                bucket = "same_sign"; n_same += 1
            else:
                bucket = "opposite_sign"; n_opp += 1
            rows.append(MemberAgreement(member=m, kpi_change_pct=(k if k == k else None),
                                        cause_change_pct=(c if c == c else None), bucket=bucket))

        denom = n_same + n_opp
        rate = (n_same / denom) if denom else None
        strength = abs(2 * rate - 1) if rate is not None else None
        direction = None
        if rate is not None:
            direction = "positive" if rate > 0.5 else "negative" if rate < 0.5 else "mixed"

        pooled = self._correlate.correlate_kpis(
            df, kpi_key, cause_kpi, mode="cross_sectional", period_a=tf_a, period_b=tf_b,
            dimension=dimension, filters=filters, min_n=min_n)

        return DimensionConsistency(
            kpi=kpi_key, cause_kpi=cause_kpi, label=label, cause_label=cause_label,
            dimension=dimension, status="ok", move_pct=move_pct, n_members=len(members),
            n_same_sign=n_same, n_opposite_sign=n_opp, n_cause_flat=n_cause_flat,
            n_kpi_flat=n_kpi_flat, consistency_rate=rate, co_movement_strength=strength,
            direction=direction, members=tuple(rows), pooled=pooled,
            selection=sel_a, baseline_selection=sel_b, filters=filters_out)

    # -- test_holdout_segments() --------------------------------------------
    def test_holdout_segments(self, df: pd.DataFrame, kpi_key: str, cause_kpi: str, dimension: str,
                              period_a: Union[Mapping[str, Any], TimeFilter],
                              period_b: Union[Mapping[str, Any], TimeFilter],
                              filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
                              cause_move_pct: Optional[float] = None,
                              cause_direction: str = "any") -> HoldoutComparison:
        self._api.require(kpi_key)
        self._api.require(cause_kpi)
        self._api.require_dimension(dimension)
        if cause_direction not in CAUSE_DIRECTIONS:
            raise InvalidArgumentError("cause_direction", cause_direction,
                                       "must be one of the supported directions.",
                                       list(CAUSE_DIRECTIONS))

        settings = get_settings()
        cause_move_pct = (cause_move_pct if cause_move_pct is not None
                         else settings.min_material_change_pct)
        label, cause_label = self._api.label(kpi_key), self._api.label(cause_kpi)
        filters_out = dict(filters or {})

        tf_a = period_a if isinstance(period_a, TimeFilter) else parse_time_filter(period_a)
        tf_b = period_b if isinstance(period_b, TimeFilter) else parse_time_filter(period_b)
        members, _kpi_chg, cause_chg, _sel_a, _sel_b, _filters = _member_changes(
            self._query, df, kpi_key, cause_kpi, dimension, tf_a, tf_b, filters)

        affected: List[str] = []
        unaffected: List[str] = []
        undetermined: List[str] = []
        for m in members:
            c = cause_chg[m]
            if not (c == c):
                undetermined.append(m)
            elif _cause_moved(c, cause_move_pct, cause_direction):
                affected.append(m)
            else:
                unaffected.append(m)
        affected.sort()
        unaffected.sort()

        def _bare(status: str) -> HoldoutComparison:
            return HoldoutComparison(
                kpi=kpi_key, cause_kpi=cause_kpi, label=label, cause_label=cause_label,
                dimension=dimension, status=status, cause_move_pct=cause_move_pct,
                cause_direction=cause_direction, n_affected=len(affected),
                n_unaffected=len(unaffected), n_undetermined=len(undetermined),
                did_pct=None, verdict=None, comparison=None, filters=filters_out)

        # Never pass an empty filter dict -- `compare_cohorts({})` raises,
        # which is the wrong signal for "every member's cause moved (or
        # didn't)"; that is a real, reportable outcome, not a bad argument.
        if not affected:
            return _bare("no_affected_members")
        if not unaffected:
            return _bare("no_unaffected_members")

        comparison = self._segments.compare_cohorts(
            df, kpi_key, {dimension: affected}, {dimension: unaffected}, tf_a, tf_b)

        did_pct: Optional[float] = None
        verdict = "undetermined"
        if comparison.a.change_pct is not None and comparison.b.change_pct is not None:
            did_pct = comparison.a.change_pct - comparison.b.change_pct
            verdict = ("kpi_moved_regardless" if abs(did_pct) < settings.min_material_change_pct
                      else "cause_explains_the_gap")

        return HoldoutComparison(
            kpi=kpi_key, cause_kpi=cause_kpi, label=label, cause_label=cause_label,
            dimension=dimension, status="ok", cause_move_pct=cause_move_pct,
            cause_direction=cause_direction, n_affected=len(affected),
            n_unaffected=len(unaffected), n_undetermined=len(undetermined),
            did_pct=did_pct, verdict=verdict, comparison=comparison, filters=filters_out)
