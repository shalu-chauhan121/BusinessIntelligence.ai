"""
`correlate_kpis` / `correlate_kpi_matrix` -- does anything else move with this
KPI, cross-sectionally or over time.

`analysis.correlate` (`engines/analysis.py:139`) is cross-sectional only:
Pearson r over per-member % changes between two periods. It also collapses
three structurally different situations into one identical string --
`n < 3`, a constant `metric_a`, and a constant `metric_b` all return
`"Too few comparable members to measure association."`, which is simply
wrong for the two constant-series cases. This module does not wrap that
function; it recomputes the same arithmetic (a finite-value mask, then
`numpy.corrcoef`) through `QueryEngine`/`SeriesEngine` so a two-KPI table
inherits ratio-correct aggregation and the `MAX_GROUPS` cap, and reports
which of the three happened as a typed `status` instead of one sentence.

Time-series mode is genuinely new: nothing in the codebase correlates two
KPIs' histories directly. It differences both series first, matching
`analysis.lead_lag`'s documented reasoning (`engines/analysis.py:277-282`) --
two independently trending series correlate near 1.0 at every lag on levels,
which is exactly the spurious result the rest of this codebase is written to
avoid (see `agent/trend.py`'s reasoning for Theil-Sen). Only calendar-adjacent
periods are differenced: `SeriesEngine` reports a hole in the middle of a
series as an explicit gap (`agent/series.py`), and a difference taken across
that hole is not a period-over-period change, so such pairs are dropped and
counted rather than silently computed.

`correlate_kpi_matrix` is the one-to-many sweep Tier 4 needs to test many
candidate drivers in one call instead of one round-trip per candidate: it
builds the shared cross-sectional table or the shared multi-series batch
exactly once, then reuses the identical pairwise arithmetic per candidate --
so a row here is byte-identical to calling `correlate_kpis` on that pair
alone, and the whole sweep costs a bounded, small number of underlying
queries rather than one per candidate.

`RelationGraph` labels each candidate's relation to the target KPI
(`agent/relations.py`) so a near-1.0 correlation with a KPI's own formula
component reads as mechanical, not a finding -- the model can discount it
without a second tool call.

No p-value, confidence interval, or Fisher-z CI is computed here. They live
in `agent/significance.py` (X3), which *composes* this module rather than
recomputing `r`, so the two tools can never report different numbers for the
same pair. `paired_sample` below is the seam that makes that composition --
and `agent/confounders.py`'s partial correlation -- possible: it publishes the
aligned vectors the two private builders used to construct inline, so a caller
needing three mutually consistent correlations over one sample can get them
without a second, independently-drifting copy of this arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..engines.metrics import pct_change, safe
from .contract_api import ContractAPI
from .errors import InvalidArgumentError
from .query import QueryEngine
from .relations import RelationGraph
from .series import SERIES_GRAINS, Series, SeriesEngine, period_ordinal
from .timefilter import TimeFilter, TimeSelection
from .timefilter import apply as apply_time_filter
from .timefilter import parse as parse_time_filter

MODES = ("cross_sectional", "time_series")

# `driver_graph.MIN_VERIFY_PERIODS` (`engines/driver_graph.py:50`) -- the same
# floor the existing formula-verification path already uses before trusting a
# correlation between two KPIs.
DEFAULT_MIN_N = 6

# A candidate pool this wide is no longer "a few plausible drivers"; module
# constant in the style of `scan.MAX_SCAN_KPIS`, not a `Settings` field.
MAX_MATRIX_CANDIDATES = 30

# Mirrored the retired `contest.MIN_CROSS_SECTION_MEMBERS` /
# `MAX_CROSS_SECTION_MEMBERS` (`engines/contest.py`, deleted at A9) as a second
# copy of the two numbers rather than an import, since the agent layer never
# depended on that module. Below the floor, a correlation over fewer members
# is not meaningfully different from picking two points and drawing a line;
# above the ceiling, a column that wide is an identifier, not a business
# dimension.
MIN_AUTO_DIMENSION_MEMBERS = 3
MAX_AUTO_DIMENSION_MEMBERS = 200


def pearson_correlate(xs: Sequence[float], ys: Sequence[float],
                      min_n: int) -> Tuple[str, Optional[float], Optional[float], int]:
    """
    Pearson r over paired, finite values only -- the same arithmetic
    `analysis.correlate` uses (a finite mask, then `numpy.corrcoef`), but with
    the three ways it can fail told apart: too few comparable pairs, or either
    series being constant over the pairs that do overlap. Public (unlike the
    rest of this module's private helpers) because `agent/temporal.py` reuses
    it for `cross_correlate_lagged` rather than adding a fifth `numpy.corrcoef`
    call site to the four that already exist across the codebase.
    """
    pairs = [(x, y) for x, y in zip(xs, ys)
            if x == x and y == y and np.isfinite(x) and np.isfinite(y)]
    n = len(pairs)
    if n < min_n:
        return "insufficient_n", None, None, n
    a = np.array([p[0] for p in pairs], dtype=float)
    b = np.array([p[1] for p in pairs], dtype=float)
    if a.std() == 0:
        return "constant_a", None, None, n
    if b.std() == 0:
        return "constant_b", None, None, n
    r = float(np.corrcoef(a, b)[0, 1])
    return "ok", round(r, 3), round(r * r, 3), n


@dataclass(frozen=True)
class PairedSample:
    """
    The aligned vectors a correlation is computed over, published rather than
    built inline.

    `vectors[key]` is one float per entry of `index`, in the same order, with
    `nan` wherever that key has no value for that member/period. **Nothing is
    dropped here.** `pearson_correlate` masks non-finite values *pairwise*, and
    that behaviour is what every existing `Correlation` payload was measured
    with, so this container reproduces the two builders' inputs exactly rather
    than pre-filtering them.

    A caller that needs several correlations to share one sample -- which
    `agent/confounders.py` does, because three pairwise `r`s taken on three
    different overlaps do not form a positive semi-definite matrix and the
    partial-correlation formula can then return `|r| > 1` -- applies listwise
    deletion itself via `listwise()`. Keeping that step out here is deliberate:
    it means adopting this seam changed no existing number.
    """

    basis: str                         # "member" | "period"
    mode: str                          # "cross_sectional" | "time_series"
    transform: str                     # "change"
    index: Tuple[Any, ...]
    vectors: Mapping[str, Tuple[float, ...]]
    dimension: Optional[str] = None
    grain: Optional[str] = None
    gaps: Tuple[str, ...] = ()
    pairs_dropped_at_gaps: int = 0
    # Shared members / shared periods *before* differencing. Distinguishes
    # "the two KPIs' scopes never overlapped" (0 -> `no_overlap`) from
    # "they overlapped but yielded too few usable points" (`insufficient_n`),
    # which the time-series builder used to tell apart with an early return.
    source_entries: int = 0
    selection: Optional[TimeSelection] = None
    baseline_selection: Optional[TimeSelection] = None
    filters: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)

    def listwise(self, keys: Sequence[str]) -> Tuple[Tuple[Any, ...], Dict[str, np.ndarray], int]:
        """
        `(index, {key: array}, dropped)` over only the entries where **every**
        key in `keys` is finite -- the one alignment under which a set of
        pairwise correlations is mutually consistent.
        """
        cols = [self.vectors[k] for k in keys]
        kept = [i for i in range(len(self.index))
                if all(v[i] == v[i] and np.isfinite(v[i]) for v in cols)]
        arrays = {k: np.array([self.vectors[k][i] for i in kept], dtype=float) for k in keys}
        return tuple(self.index[i] for i in kept), arrays, len(self.index) - len(kept)


@dataclass(frozen=True)
class Correlation:
    kpi_a: str
    kpi_b: str
    label_a: str
    label_b: str
    mode: str                          # "cross_sectional" | "time_series"
    transform: str                     # "change"
    basis: str                         # "member" | "period"
    status: str                        # "ok" | "insufficient_n" | "constant_a" | "constant_b" | "no_overlap"
    r: Optional[float]
    r_squared: Optional[float]
    n: int
    min_n: int
    method: str                        # "pearson"
    dimension: Optional[str] = None
    grain: Optional[str] = None
    gaps: Tuple[str, ...] = ()
    pairs_dropped_at_gaps: int = 0
    selection: Optional[TimeSelection] = None
    baseline_selection: Optional[TimeSelection] = None
    # `()` is not a legal stand-in default for an empty mapping -- `.items()`
    # raises `AttributeError` the moment a caller default-constructs this
    # without `filters`. `{}` itself cannot be the literal default (a mutable
    # default is a dataclass error), so `field(default_factory=dict)` is the
    # one way to default this field safely.
    filters: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        payload = {
            "kpi_a": self.kpi_a, "kpi_b": self.kpi_b,
            "label_a": self.label_a, "label_b": self.label_b,
            "mode": self.mode, "transform": self.transform, "basis": self.basis,
            "status": self.status, "r": safe(self.r), "r_squared": safe(self.r_squared),
            "n": self.n, "min_n": self.min_n, "method": self.method,
            "dimension": self.dimension, "grain": self.grain,
            "gaps": list(self.gaps), "pairs_dropped_at_gaps": self.pairs_dropped_at_gaps,
            "filters": {k: list(v) for k, v in self.filters.items()},
        }
        if self.selection is not None:
            payload["selection"] = self.selection.to_payload()
        if self.baseline_selection is not None:
            payload["baseline_selection"] = self.baseline_selection.to_payload()
        return payload


@dataclass(frozen=True)
class CorrelationRow:
    relation: Optional[str]
    correlation: Correlation

    def to_payload(self) -> Dict[str, Any]:
        payload = self.correlation.to_payload()
        payload["relation"] = self.relation
        return payload


@dataclass(frozen=True)
class CorrelationMatrix:
    kpi: str
    label: str
    mode: str
    rows: Tuple[CorrelationRow, ...]
    considered: int
    truncated: bool
    selection: Optional[TimeSelection]
    filters: Mapping[str, Tuple[str, ...]]

    def to_payload(self) -> Dict[str, Any]:
        payload = {
            "kpi": self.kpi, "label": self.label, "mode": self.mode,
            "rows": [r.to_payload() for r in self.rows],
            "considered": self.considered, "truncated": self.truncated,
            "filters": {k: list(v) for k, v in self.filters.items()},
        }
        if self.selection is not None:
            payload["selection"] = self.selection.to_payload()
        return payload


class CorrelationEngine:
    """`correlate_kpis` / `correlate_kpi_matrix`, bound to one dataset's
    `ContractAPI`."""

    def __init__(self, api: ContractAPI):
        self._api = api
        self._query = QueryEngine(api)
        self._series = SeriesEngine(api)

    # -- shared helpers ----------------------------------------------------
    def _auto_dimension(self, df: pd.DataFrame, tf_a: TimeFilter, tf_b: TimeFilter,
                        filters: Optional[Mapping[str, Union[str, Sequence[str]]]]) -> str:
        """
        The dimension with the most members shared by both periods --
        `contest._cross_sectional_dimension`'s rule (`engines/contest.py:130`),
        sourced from every dimension `ContractAPI` knows rather than the
        hardcoded `("region","product","channel","segment")` whitelist at
        `contest.py:125`, which silently disables this check on the hospital
        and school fixtures.

        `filters` is applied to both periods' frames before counting shared
        members, and the count is bounded exactly as `contest`'s chooser
        bounds it. Neither used to happen: measured, `filters={"region":
        "North"}` still auto-selected `region` itself -- the one dimension the
        filter had just collapsed to a single member -- and the correlation
        that followed silently came back `status="insufficient_n"` with
        nothing pointing at why. Sorting dimension names before comparing
        counts makes a tie deterministic, matching `contest`'s alphabetical
        tie-break (pinned by `test_cross_sectional_dimension.py`).
        """
        frame_a, _ = apply_time_filter(df, tf_a)
        frame_b, _ = apply_time_filter(df, tf_b)
        for dim, raw_values in (filters or {}).items():
            self._api.require_dimension(dim)
            values = [raw_values] if isinstance(raw_values, str) else list(raw_values)
            members_a = tuple(self._api.resolve_member(dim, v, frame_a) for v in values)
            members_b = tuple(self._api.resolve_member(dim, v, frame_b) for v in values)
            frame_a = frame_a[frame_a[dim].isin(members_a)]
            frame_b = frame_b[frame_b[dim].isin(members_b)]

        best_dim: Optional[str] = None
        best_count = -1
        for info in sorted(self._api.list_dimensions(df), key=lambda d: d.name):
            dim = info.name
            if dim not in frame_a.columns or dim not in frame_b.columns:
                continue
            shared = set(frame_a[dim].dropna().unique()) & set(frame_b[dim].dropna().unique())
            count = len(shared)
            if count < MIN_AUTO_DIMENSION_MEMBERS or count > MAX_AUTO_DIMENSION_MEMBERS:
                continue
            if count > best_count:
                best_dim, best_count = dim, count
        if best_dim is None:
            raise InvalidArgumentError(
                "dimension", None,
                "could not auto-select a dimension -- none of this dataset's dimensions "
                f"have between {MIN_AUTO_DIMENSION_MEMBERS} and {MAX_AUTO_DIMENSION_MEMBERS} "
                "members shared between the two periods, after applying any filters.",
                [d.name for d in self._api.list_dimensions(df)])
        return best_dim

    # -- aligned samples ---------------------------------------------------
    def _cross_sectional_sample(self, keys: Sequence[str],
                                cur_map: Mapping[Any, Mapping[str, float]],
                                base_map: Mapping[Any, Mapping[str, float]],
                                members: Sequence[Any], dimension: str,
                                selection: TimeSelection, baseline_selection: TimeSelection,
                                filters_out: Mapping[str, Tuple[str, ...]]) -> PairedSample:
        """Each key's per-member % change between the two periods -- the vectors
        `_cross_sectional_correlation` used to build inline, unchanged."""
        vectors = {
            key: tuple(pct_change(cur_map.get(m, {}).get(key, float("nan")),
                                  base_map.get(m, {}).get(key, float("nan"))) for m in members)
            for key in keys}
        return PairedSample(
            basis="member", mode="cross_sectional", transform="change",
            index=tuple(members), vectors=vectors, dimension=dimension,
            source_entries=len(members), selection=selection,
            baseline_selection=baseline_selection, filters=filters_out)

    def _time_series_sample(self, keys: Sequence[str], series_map: Mapping[str, Series],
                            grain: str) -> PairedSample:
        """Each key's period-over-period % change, over the periods every key
        holds. Only calendar-adjacent periods are differenced -- a difference
        taken across a gap is not a period-over-period change -- so such pairs
        are dropped and counted, exactly as before."""
        by_key = {k: {p.period: p.value for p in series_map[k].points} for k in keys}
        periods = sorted(set.intersection(*(set(v) for v in by_key.values()))) if keys else []
        gaps = tuple(sorted(set().union(*(set(series_map[k].gaps) for k in keys)))) if keys else ()
        first = series_map[keys[0]]

        index: List[str] = []
        vectors: Dict[str, List[float]] = {k: [] for k in keys}
        dropped = 0
        prev_period: Optional[str] = None
        prev_ord: Optional[int] = None
        for period in periods:
            ordv = period_ordinal(grain, period)
            if prev_period is not None:
                if ordv - prev_ord == 1:
                    index.append(period)
                    for k in keys:
                        vectors[k].append(pct_change(by_key[k][period], by_key[k][prev_period]))
                else:
                    dropped += 1
            prev_period, prev_ord = period, ordv

        return PairedSample(
            basis="period", mode="time_series", transform="change",
            index=tuple(index), vectors={k: tuple(v) for k, v in vectors.items()},
            grain=grain, gaps=gaps, pairs_dropped_at_gaps=dropped,
            source_entries=len(periods), selection=first.selection, filters=first.filters)

    def _correlation_from_sample(self, sample: PairedSample, kpi_a: str, kpi_b: str,
                                 min_n: int) -> Correlation:
        label_a, label_b = self._api.label(kpi_a), self._api.label(kpi_b)
        common = dict(
            kpi_a=kpi_a, kpi_b=kpi_b, label_a=label_a, label_b=label_b,
            mode=sample.mode, transform=sample.transform, basis=sample.basis,
            min_n=min_n, method="pearson", dimension=sample.dimension, grain=sample.grain,
            gaps=sample.gaps, pairs_dropped_at_gaps=sample.pairs_dropped_at_gaps,
            selection=sample.selection, baseline_selection=sample.baseline_selection,
            filters=sample.filters)

        # No shared scope at all is a different answer from "shared, but too few
        # usable points", and only the time-series builder can produce it.
        if sample.mode == "time_series" and sample.source_entries == 0:
            return Correlation(status="no_overlap", r=None, r_squared=None, n=0, **common)

        status, r, r2, n = pearson_correlate(sample.vectors[kpi_a], sample.vectors[kpi_b], min_n)
        return Correlation(status=status, r=r, r_squared=r2, n=n, **common)

    # -- correlate_kpis() ------------------------------------------------------
    def correlate_kpis(self, df: pd.DataFrame, kpi_a: str, kpi_b: str,
                       mode: str = "cross_sectional",
                       period_a: Optional[Union[Mapping[str, Any], TimeFilter]] = None,
                       period_b: Optional[Union[Mapping[str, Any], TimeFilter]] = None,
                       dimension: Optional[str] = None,
                       time_filter: Optional[Union[Mapping[str, Any], TimeFilter]] = None,
                       grain: str = "quarter",
                       filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
                       min_n: int = DEFAULT_MIN_N) -> Correlation:
        if mode not in MODES:
            raise InvalidArgumentError("mode", mode, "must be one of the supported modes.", list(MODES))
        self._api.require(kpi_a)
        self._api.require(kpi_b)
        if min_n < 2:
            raise InvalidArgumentError("min_n", min_n, "must be >= 2.")

        sample = self.paired_sample(df, [kpi_a, kpi_b], mode=mode, period_a=period_a,
                                    period_b=period_b, dimension=dimension,
                                    time_filter=time_filter, grain=grain, filters=filters)
        return self._correlation_from_sample(sample, kpi_a, kpi_b, min_n)

    # -- paired_sample() -------------------------------------------------------
    def paired_sample(self, df: pd.DataFrame, kpi_keys: Sequence[str],
                      mode: str = "cross_sectional",
                      period_a: Optional[Union[Mapping[str, Any], TimeFilter]] = None,
                      period_b: Optional[Union[Mapping[str, Any], TimeFilter]] = None,
                      dimension: Optional[str] = None,
                      time_filter: Optional[Union[Mapping[str, Any], TimeFilter]] = None,
                      grain: str = "quarter",
                      filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None
                      ) -> PairedSample:
        """
        The aligned vectors `correlate_kpis` correlates, for any number of keys
        and without correlating them.

        `agent/significance.py` and `agent/confounders.py` scope a sample
        through this rather than re-deriving the query pair or the gap-aware
        differencing loop, so a p-value, a partial correlation and the `r` they
        qualify are all computed over the same rows by construction. Argument
        handling is deliberately identical to `correlate_kpis`, which now
        delegates here, so the scope a caller describes cannot mean one thing to
        one tool and something else to another.
        """
        if mode not in MODES:
            raise InvalidArgumentError("mode", mode, "must be one of the supported modes.", list(MODES))
        keys = list(dict.fromkeys(kpi_keys))
        if not keys:
            raise InvalidArgumentError("kpi_keys", kpi_keys, "at least one KPI key is required.")
        for key in keys:
            self._api.require(key)

        if mode == "cross_sectional":
            if period_a is None or period_b is None:
                raise InvalidArgumentError(
                    "period_a", period_a,
                    "cross_sectional mode requires both period_a and period_b.")
            tf_a = period_a if isinstance(period_a, TimeFilter) else parse_time_filter(period_a)
            tf_b = period_b if isinstance(period_b, TimeFilter) else parse_time_filter(period_b)
            if dimension is None:
                dimension = self._auto_dimension(df, tf_a, tf_b, filters)
            else:
                self._api.require_dimension(dimension)

            cur_result = self._query.query(df, keys, tf_a, filters=filters, group_by=[dimension])
            base_result = self._query.query(df, keys, tf_b, filters=filters, group_by=[dimension])
            cur_map = {dict(c.group)[dimension]: c.values for c in cur_result.cells}
            base_map = {dict(c.group)[dimension]: c.values for c in base_result.cells}
            members = sorted(set(cur_map) | set(base_map))
            return self._cross_sectional_sample(
                keys, cur_map, base_map, members, dimension,
                cur_result.selection, base_result.selection, cur_result.filters)

        if grain not in SERIES_GRAINS:
            raise InvalidArgumentError("grain", grain, "must be one of the supported grains.",
                                       list(SERIES_GRAINS))
        tf = time_filter if time_filter is not None else {"type": "all"}
        series_map = self._series.series_multi(df, keys, grain=grain, time_filter=tf,
                                               filters=filters)
        return self._time_series_sample(keys, series_map, grain)

    # -- correlate_kpi_matrix() -------------------------------------------------
    def correlate_kpi_matrix(self, df: pd.DataFrame, kpi_key: str,
                             candidates: Optional[Sequence[str]] = None,
                             mode: str = "cross_sectional",
                             period_a: Optional[Union[Mapping[str, Any], TimeFilter]] = None,
                             period_b: Optional[Union[Mapping[str, Any], TimeFilter]] = None,
                             dimension: Optional[str] = None,
                             time_filter: Optional[Union[Mapping[str, Any], TimeFilter]] = None,
                             grain: str = "quarter",
                             filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
                             min_n: int = DEFAULT_MIN_N,
                             limit: Optional[int] = None) -> CorrelationMatrix:
        if mode not in MODES:
            raise InvalidArgumentError("mode", mode, "must be one of the supported modes.", list(MODES))
        self._api.require(kpi_key)

        graph = RelationGraph(self._api)
        if candidates is None:
            pool = graph.neighbours(kpi_key, depth=1)
            if not pool:
                pool = [k for k in self._api.list_kpi_keys() if k != kpi_key]
        else:
            pool = [c for c in candidates if c != kpi_key]
            for c in pool:
                self._api.require(c)

        considered = len(pool)
        cap = min(MAX_MATRIX_CANDIDATES, limit) if limit is not None else MAX_MATRIX_CANDIDATES
        truncated = considered > cap
        pool = pool[:cap]

        relation_by_kpi: Dict[str, str] = {}
        for relation in graph.related(kpi_key):
            relation_by_kpi.setdefault(relation.source_kpi, relation.relation)

        if mode == "cross_sectional":
            if period_a is None or period_b is None:
                raise InvalidArgumentError(
                    "period_a", period_a,
                    "cross_sectional mode requires both period_a and period_b.")
            tf_a = period_a if isinstance(period_a, TimeFilter) else parse_time_filter(period_a)
            tf_b = period_b if isinstance(period_b, TimeFilter) else parse_time_filter(period_b)
            if dimension is None:
                dimension = self._auto_dimension(df, tf_a, tf_b, filters)
            else:
                self._api.require_dimension(dimension)

            all_keys = [kpi_key] + pool
            cur_result = self._query.query(df, all_keys, tf_a, filters=filters, group_by=[dimension])
            base_result = self._query.query(df, all_keys, tf_b, filters=filters, group_by=[dimension])
            cur_map = {dict(c.group)[dimension]: c.values for c in cur_result.cells}
            base_map = {dict(c.group)[dimension]: c.values for c in base_result.cells}
            members = sorted(set(cur_map) | set(base_map))

            # One sample over every key, but each row's `r` still comes from
            # its own pair's vectors -- `pearson_correlate` masks pairwise, so a
            # matrix row stays byte-identical to the standalone call for that
            # pair. `members` does not depend on the pair, which is what makes
            # the shared sample legitimate here.
            sample = self._cross_sectional_sample(
                all_keys, cur_map, base_map, members, dimension,
                cur_result.selection, base_result.selection, cur_result.filters)
            rows = tuple(
                CorrelationRow(relation=relation_by_kpi.get(cand),
                               correlation=self._correlation_from_sample(sample, kpi_key, cand, min_n))
                for cand in pool)
            selection, filters_out = cur_result.selection, cur_result.filters

        else:
            if grain not in SERIES_GRAINS:
                raise InvalidArgumentError("grain", grain, "must be one of the supported grains.",
                                           list(SERIES_GRAINS))
            tf = time_filter if time_filter is not None else {"type": "all"}
            all_keys = [kpi_key] + pool
            series_map = self._series.series_multi(df, all_keys, grain=grain, time_filter=tf,
                                                   filters=filters)
            target = series_map[kpi_key]
            # Deliberately per-pair, unlike the cross-sectional branch above:
            # the shared-period intersection *does* depend on which candidate is
            # in it, so one sample over every key would silently shrink every
            # row's window to the worst-covered candidate's.
            rows = tuple(
                CorrelationRow(relation=relation_by_kpi.get(cand),
                               correlation=self._correlation_from_sample(
                                   self._time_series_sample([kpi_key, cand], series_map, grain),
                                   kpi_key, cand, min_n))
                for cand in pool)
            selection, filters_out = target.selection, target.filters

        ranked = sorted(rows, key=lambda row: (row.correlation.r is None,
                                               -(abs(row.correlation.r) if row.correlation.r is not None else 0.0)))

        return CorrelationMatrix(kpi=kpi_key, label=self._api.label(kpi_key), mode=mode,
                                 rows=tuple(ranked), considered=considered, truncated=truncated,
                                 selection=selection, filters=filters_out)
