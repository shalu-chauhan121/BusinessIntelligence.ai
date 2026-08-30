"""
`check_temporal_precedence` / `cross_correlate_lagged` / `test_reverse_causation`
-- did the proposed cause move before the KPI, and if it correlates, which
direction actually fits the timing.

`contest.temporal_check` (`engines/contest.py:94`) already does real work here
-- `detect_onset` + `compare_onsets` + `lead_lag` -- but its output is three
sentences of `detail` prose, and its reverse-causation screen
(`mechanism_check`, `contest.py:284`) gates a real score penalty on the
truthiness of a free-text LLM field, `hypothesis["reverse_causation_risk"]`:
any non-empty string counts, including one that says the risk is low, and a
hypothesis whose LLM wrote nothing can never be flagged. This module answers
the same two questions with real arithmetic and no guessing.

**`check_temporal_precedence` composes `TrendEngine`'s already-typed
`changepoint_from_series`, not `analysis.detect_onset` / `compare_onsets`
directly.** `detect_changepoint` already splits `detect_onset`'s bare `None`
into `"insufficient_history" | "degenerate_dispersion" | "no_changepoint"`; a
third copy of that dispersion-estimate line (already living in
`analysis.py:53` and `trend.py`) is not wanted. `changepoint_from_series` is
called twice off a single `SeriesEngine.series_multi` batch, so classifying
both sides costs one query pair, not two.

**The lag is calendar ordinals, not `detect_onset`'s `index`, and the sign is
flipped from `compare_onsets`.** `index` counts *held* periods; a series with
a hole understates the true elapsed lag. And `compare_onsets` computes
`cause.index - kpi.index`; this module uses `kpi_ordinal - cause_ordinal`, so
**positive means the cause turned first** -- the convention `analysis.
lead_lag` and `driver_graph._lagged_r` already use. Two lag conventions
shipping in one batch is the footgun, so this one picks the convention both
lag tools already share, and a cross-tool test pins it.

**Direction must be an argument, and the search window is part of the
answer -- both measured, not assumed.** `detect_changepoint(direction="any")`
returns the *earliest* onset across both directions, which on the retail
sample selects revenue's spring seasonal upturn instead of its collapse if
the caller does not say which direction it means. And `detect_onset` dates
the first breach of the baseline formed by the *opening* `baseline_periods`
of whatever window it is given -- not the move under investigation. Measured
on retail `units_sold` vs `stockout_events`, both filtered to
`product=Product A`: over the full 183-week history the tool reports the
cause leading by **156 weeks** (the cause's baseline was set from three years
before the real event); scoped to a 39-week window around the planted
disruption it reports the KPI leading by 20 weeks instead. Neither number is
usable on its own. So this tool never defaults to all-history, and it flags
`*_onset_at_window_edge` when a side's onset falls within
`baseline_periods + persistence` of the window start -- "already moving when
we started looking", not "started here" -- and degrades the verdict rather
than asserting a precedence the window cannot actually support.

**`cross_correlate_lagged` differences with `.diff()`, not `pct_change`.**
Consolidates two existing lagged-r implementations (`analysis.lead_lag`,
`driver_graph._lagged_r`) plus the contemporaneous-only path in
`agent/correlate.py`'s own time-series mode, reusing `correlate.
pearson_correlate` for the arithmetic core rather than adding a fifth
`numpy.corrcoef` call site. `pct_change` was tried and measured against the
real fixtures first: on retail `revenue ~ stockout_events` -- the pair
carrying the planted supply-disruption signal -- the measured association
halves (`r=-0.583` with `.diff()` vs `r=-0.315` with `pct_change`), and on
hospital weekly `deaths` (zero in 17 of 183 weeks) 8.8% of pairs are
destroyed by a near-zero denominator. Because this tool selects the best lag
by `argmax |r|`, one such blow-up biases *which lag wins*, not merely the
estimate. So the transform here is `"difference"`, deliberately not
`correlate_kpis`'s `"change"` -- the two tools are pinned to agree on the
*pairing* (same `n`, same `pairs_dropped_at_gaps`) at lag 0, never on the
value, since the transforms are different on purpose.

**`test_reverse_causation` reads `RelationGraph.components()`, never
`related()`, for the structural signal.** `related()`'s `derivation` edges
come from a symmetric predicate (`_derivation_related`) and can never say
which KPI is downstream; on the retail sample `related("revenue")` returns
only derivations. `components()` is the one directional signal in the
codebase -- exactly the fact `mechanism_check` was trying to guess from a
sentence. It calls `cross_correlate_lagged` once, not twice: the lag profile
of `(kpi, cause)` at lag `+L` is the profile of `(cause, kpi)` at `-L`,
verified to machine precision, so running the reverse direction would buy
nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

from ..engines.metrics import safe
from .contract_api import ContractAPI
from .correlate import pearson_correlate
from .errors import InvalidArgumentError
from .relations import RelationGraph
from .series import SERIES_GRAINS, SeriesEngine, period_ordinal
from .timefilter import TimeFilter, TimeSelection
from .trend import DIRECTIONS, Changepoint, changepoint_from_series

MIN_LAG_PAIRS = 8              # analysis.lead_lag's per-lag floor (analysis.py:296)
MIN_LAG_SERIES_POINTS = 12     # analysis.lead_lag's merged-rows floor (analysis.py:288)
MAX_LAG_CEILING = 26           # a lag past half a year is not a mechanism worth searching for

# An onset this close to the window's *relative* start is suspect regardless
# of the window's absolute length -- measured on retail `units_sold` vs
# `stockout_events` (`product=Product A`, the planted supply disruption):
# over the full 183-week history the cause's onset lands at week 18 (9.8% of
# the window), well past the strict `baseline_periods + persistence` floor
# below, yet it is still an artifact -- `detect_onset` found the very first
# breach of a baseline built from three years before the real event, and
# reported a 156-week lead. The absolute floor alone (10 weeks, with the
# defaults) does not catch this; the fraction does.
WINDOW_EDGE_FRACTION = 0.2

# The relations `RelationGraph` can attach a direction to -- everything else
# (`derivation`, `shared_source_field`, `semantic_tag`) is symmetric by
# construction and can name a link but never say which side is downstream.
_SHARED_INPUT_RELATIONS = ("derivation", "shared_source_field")


def _onset_at_window_edge(onset_position: int, periods_searched: int,
                          baseline_periods: int, persistence: int) -> bool:
    """
    True when an onset means "already moving when we started looking" rather
    than "started here" -- either it falls within the detector's own
    `baseline_periods + persistence` floor (too little runway to have formed
    a real pre-event baseline at all), or it falls within the first
    `WINDOW_EDGE_FRACTION` of the whole searched window (a long window whose
    "baseline" was built from ancient history, so an early wobble becomes a
    permanent, spurious onset -- see the module-level constant's docstring
    for the measured case this second condition alone was added to catch).
    """
    return (onset_position < baseline_periods + persistence
           or onset_position < WINDOW_EDGE_FRACTION * periods_searched)


# ---------------------------------------------------------------------------
# check_temporal_precedence()
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TemporalPrecedence:
    kpi: str
    cause_kpi: str
    label: str
    cause_label: str
    grain: str
    status: str                       # "ok" | "kpi_undetermined" | "cause_undetermined" | "both_undetermined"
    verdict: Optional[str]            # "cause_precedes_kpi" | "simultaneous" | "kpi_precedes_cause"
    lag_periods: Optional[int]        # > 0: the cause turned first
    kpi_period: Optional[str]
    cause_period: Optional[str]
    kpi_onset_at_window_edge: Optional[bool]
    cause_onset_at_window_edge: Optional[bool]
    gaps_between: int
    detector_resolution_periods: int
    periods_searched: int
    baseline_periods: int
    kpi_changepoint: Changepoint
    cause_changepoint: Changepoint
    selection: TimeSelection
    filters: Mapping[str, Tuple[str, ...]]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "kpi": self.kpi, "cause_kpi": self.cause_kpi,
            "label": self.label, "cause_label": self.cause_label, "grain": self.grain,
            "status": self.status, "verdict": self.verdict, "lag_periods": self.lag_periods,
            "kpi_period": self.kpi_period, "cause_period": self.cause_period,
            "kpi_onset_at_window_edge": self.kpi_onset_at_window_edge,
            "cause_onset_at_window_edge": self.cause_onset_at_window_edge,
            "gaps_between": self.gaps_between,
            "detector_resolution_periods": self.detector_resolution_periods,
            "periods_searched": self.periods_searched, "baseline_periods": self.baseline_periods,
            "kpi_changepoint": self.kpi_changepoint.to_payload(),
            "cause_changepoint": self.cause_changepoint.to_payload(),
            "selection": self.selection.to_payload(),
            "filters": {k: list(v) for k, v in self.filters.items()},
        }


# ---------------------------------------------------------------------------
# cross_correlate_lagged()
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LagPoint:
    lag: int
    r: Optional[float]
    r_squared: Optional[float]
    n: int
    status: str                       # "ok" | "insufficient_n" | "constant_a" | "constant_b"

    def to_payload(self) -> Dict[str, Any]:
        return {"lag": self.lag, "r": safe(self.r), "r_squared": safe(self.r_squared),
               "n": self.n, "status": self.status}


@dataclass(frozen=True)
class LaggedCorrelation:
    kpi: str
    cause_kpi: str
    label: str
    cause_label: str
    grain: str
    status: str                       # "ok" | "no_overlap" | "insufficient_history" | "no_usable_lag"
    verdict: Optional[str]            # "cause_leads" | "simultaneous" | "kpi_leads"
    best_lag: Optional[int]
    r_at_best: Optional[float]
    r_at_lag_0: Optional[float]
    profile: Tuple[LagPoint, ...]
    max_lag: int
    min_n: int
    points: int
    gaps: Tuple[str, ...]
    pairs_dropped_at_gaps: int
    selection: Optional[TimeSelection]
    filters: Mapping[str, Tuple[str, ...]]
    transform: str = "difference"     # deliberately not correlate.py's "change" -- see module docstring
    method: str = "pearson"

    def to_payload(self) -> Dict[str, Any]:
        return {
            "kpi": self.kpi, "cause_kpi": self.cause_kpi,
            "label": self.label, "cause_label": self.cause_label,
            "grain": self.grain, "transform": self.transform, "method": self.method,
            "status": self.status, "verdict": self.verdict,
            "best_lag": self.best_lag, "r_at_best": safe(self.r_at_best),
            "r_at_lag_0": safe(self.r_at_lag_0),
            "profile": [p.to_payload() for p in self.profile],
            "max_lag": self.max_lag, "min_n": self.min_n, "points": self.points,
            "gaps": list(self.gaps), "pairs_dropped_at_gaps": self.pairs_dropped_at_gaps,
            "selection": self.selection.to_payload() if self.selection else None,
            "filters": {k: list(v) for k, v in self.filters.items()},
        }


# ---------------------------------------------------------------------------
# test_reverse_causation()
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ReverseCausation:
    kpi: str
    cause_kpi: str
    label: str
    cause_label: str
    # "cause_derived_from_kpi" | "kpi_derived_from_cause" | "mutual" | "shared_inputs" | "none"
    mechanical: str
    mechanical_relation: Optional[str]
    mechanical_evidence_fields: Tuple[str, ...]
    verdict: str                      # "definitional" | "reverse_supported" | "forward_supported" | "undetermined"
    timing: LaggedCorrelation

    def to_payload(self) -> Dict[str, Any]:
        return {
            "kpi": self.kpi, "cause_kpi": self.cause_kpi,
            "label": self.label, "cause_label": self.cause_label,
            "mechanical": self.mechanical, "mechanical_relation": self.mechanical_relation,
            "mechanical_evidence_fields": list(self.mechanical_evidence_fields),
            "verdict": self.verdict, "timing": self.timing.to_payload(),
        }


class TemporalEngine:
    """`check_temporal_precedence` / `cross_correlate_lagged` /
    `test_reverse_causation`, bound to one dataset's `ContractAPI`."""

    def __init__(self, api: ContractAPI):
        self._api = api
        self._series = SeriesEngine(api)
        self._graph = RelationGraph(api)

    # -- check_temporal_precedence() -------------------------------------------
    def check_temporal_precedence(self, df: pd.DataFrame, kpi_key: str, cause_kpi: str,
                                  grain: str = "week",
                                  time_filter: Optional[Union[Mapping[str, Any], TimeFilter]] = None,
                                  filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
                                  kpi_direction: str = "any", cause_direction: str = "any",
                                  baseline_periods: int = 8, persistence: int = 2,
                                  k_sigma: float = 1.0) -> TemporalPrecedence:
        self._api.require(kpi_key)
        self._api.require(cause_kpi)
        for arg_name, direction in (("kpi_direction", kpi_direction),
                                    ("cause_direction", cause_direction)):
            if direction not in DIRECTIONS:
                raise InvalidArgumentError(arg_name, direction,
                                           "must be one of the supported directions.",
                                           list(DIRECTIONS))

        label, cause_label = self._api.label(kpi_key), self._api.label(cause_kpi)
        series_map = self._series.series_multi(df, [kpi_key, cause_kpi], grain=grain,
                                               time_filter=time_filter, filters=filters)
        kpi_series, cause_series = series_map[kpi_key], series_map[cause_kpi]

        kpi_cp = changepoint_from_series(kpi_series, kpi_key, label, self._api.unit(kpi_key),
                                         grain, kpi_direction, baseline_periods, persistence, k_sigma)
        cause_cp = changepoint_from_series(cause_series, cause_kpi, cause_label,
                                           self._api.unit(cause_kpi), grain, cause_direction,
                                           baseline_periods, persistence, k_sigma)

        kpi_ok, cause_ok = kpi_cp.status == "ok", cause_cp.status == "ok"
        if kpi_ok and cause_ok:
            status = "ok"
        elif not kpi_ok and not cause_ok:
            status = "both_undetermined"
        elif not kpi_ok:
            status = "kpi_undetermined"
        else:
            status = "cause_undetermined"

        periods_searched = len(kpi_series.points)
        kpi_edge = cause_edge = None
        if kpi_ok:
            kpi_start_ord = period_ordinal(grain, kpi_series.points[0].period)
            kpi_edge = _onset_at_window_edge(
                period_ordinal(grain, kpi_cp.period) - kpi_start_ord,
                periods_searched, baseline_periods, persistence)
        if cause_ok:
            cause_start_ord = period_ordinal(grain, cause_series.points[0].period)
            cause_edge = _onset_at_window_edge(
                period_ordinal(grain, cause_cp.period) - cause_start_ord,
                periods_searched, baseline_periods, persistence)

        verdict: Optional[str] = None
        lag_periods: Optional[int] = None
        gaps_between = 0
        if status == "ok":
            kpi_ord = period_ordinal(grain, kpi_cp.period)
            cause_ord = period_ordinal(grain, cause_cp.period)
            lag_periods = kpi_ord - cause_ord
            verdict = ("cause_precedes_kpi" if lag_periods > 0 else
                      "kpi_precedes_cause" if lag_periods < 0 else "simultaneous")
            lo, hi = sorted((kpi_ord, cause_ord))
            all_gaps = set(kpi_series.gaps) | set(cause_series.gaps)
            gaps_between = sum(1 for g in all_gaps if lo < period_ordinal(grain, g) < hi)

        return TemporalPrecedence(
            kpi=kpi_key, cause_kpi=cause_kpi, label=label, cause_label=cause_label, grain=grain,
            status=status, verdict=verdict, lag_periods=lag_periods,
            kpi_period=(kpi_cp.period if kpi_ok else None),
            cause_period=(cause_cp.period if cause_ok else None),
            kpi_onset_at_window_edge=kpi_edge, cause_onset_at_window_edge=cause_edge,
            gaps_between=gaps_between, detector_resolution_periods=persistence,
            periods_searched=periods_searched, baseline_periods=baseline_periods,
            kpi_changepoint=kpi_cp, cause_changepoint=cause_cp,
            selection=kpi_series.selection, filters=dict(kpi_series.filters))

    # -- cross_correlate_lagged() -----------------------------------------------
    def cross_correlate_lagged(self, df: pd.DataFrame, kpi_key: str, cause_kpi: str,
                               grain: str = "week", max_lag: int = 5,
                               time_filter: Optional[Union[Mapping[str, Any], TimeFilter]] = None,
                               filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
                               min_n: int = MIN_LAG_PAIRS) -> LaggedCorrelation:
        self._api.require(kpi_key)
        self._api.require(cause_kpi)
        if grain not in SERIES_GRAINS:
            raise InvalidArgumentError("grain", grain, "must be one of the supported grains.",
                                       list(SERIES_GRAINS))
        if not (1 <= max_lag <= MAX_LAG_CEILING):
            raise InvalidArgumentError("max_lag", max_lag,
                                       f"must be between 1 and {MAX_LAG_CEILING}.")

        label, cause_label = self._api.label(kpi_key), self._api.label(cause_kpi)
        series_map = self._series.series_multi(df, [kpi_key, cause_kpi], grain=grain,
                                               time_filter=time_filter, filters=filters)
        kpi_series, cause_series = series_map[kpi_key], series_map[cause_kpi]
        gaps = tuple(sorted(set(kpi_series.gaps) | set(cause_series.gaps)))

        def _bare(status: str, **kw: Any) -> LaggedCorrelation:
            fields = dict(verdict=None, best_lag=None, r_at_best=None, r_at_lag_0=None,
                         profile=(), points=0, pairs_dropped_at_gaps=0)
            fields.update(kw)
            return LaggedCorrelation(
                kpi=kpi_key, cause_kpi=cause_kpi, label=label, cause_label=cause_label,
                grain=grain, status=status, max_lag=max_lag, min_n=min_n,
                gaps=gaps, selection=kpi_series.selection, filters=dict(kpi_series.filters),
                **fields)

        by_a = {p.period: p.value for p in kpi_series.points}
        by_b = {p.period: p.value for p in cause_series.points}
        periods = sorted(set(by_a) & set(by_b))
        if not periods:
            return _bare("no_overlap")

        # Only calendar-adjacent periods are differenced -- the identical
        # rule `correlate._time_series_correlation` uses -- so a hole in the
        # series drops that one transition rather than differencing across
        # it as if it were a real period-over-period change.
        diffs_a: Dict[int, float] = {}
        diffs_b: Dict[int, float] = {}
        dropped = 0
        prev_period: Optional[str] = None
        prev_ord: Optional[int] = None
        for period in periods:
            ordv = period_ordinal(grain, period)
            if prev_period is not None:
                if ordv - prev_ord == 1:
                    diffs_a[ordv] = by_a[period] - by_a[prev_period]
                    diffs_b[ordv] = by_b[period] - by_b[prev_period]
                else:
                    dropped += 1
            prev_period, prev_ord = period, ordv

        if len(diffs_a) < MIN_LAG_SERIES_POINTS:
            return _bare("insufficient_history", points=len(diffs_a),
                        pairs_dropped_at_gaps=dropped)

        ordinals = sorted(diffs_a)
        profile: List[LagPoint] = []
        for lag in range(-max_lag, max_lag + 1):
            xs: List[float] = []
            ys: List[float] = []
            for t in ordinals:
                src = t - lag
                if src in diffs_b:
                    xs.append(diffs_a[t])
                    ys.append(diffs_b[src])
            status_l, r, r2, n = pearson_correlate(xs, ys, min_n)
            profile.append(LagPoint(lag=lag, r=r, r_squared=r2, n=n, status=status_l))

        ok_points = [p for p in profile if p.status == "ok"]
        if not ok_points:
            return _bare("no_usable_lag", profile=tuple(profile),
                        points=len(diffs_a), pairs_dropped_at_gaps=dropped)

        # Largest |r| wins; ties break toward the smaller absolute lag, then
        # the more negative one -- deliberately not `analysis.lead_lag`'s
        # `max(key=abs)`, which silently favours the most negative lag on a
        # tie because results are built from `-max_lag` upward.
        best = min(ok_points, key=lambda p: (-abs(p.r), abs(p.lag), p.lag))
        r_at_lag_0 = next((p.r for p in profile if p.lag == 0), None)
        verdict = ("cause_leads" if best.lag > 0 else
                  "kpi_leads" if best.lag < 0 else "simultaneous")

        return LaggedCorrelation(
            kpi=kpi_key, cause_kpi=cause_kpi, label=label, cause_label=cause_label, grain=grain,
            status="ok", verdict=verdict, best_lag=best.lag, r_at_best=best.r,
            r_at_lag_0=r_at_lag_0, profile=tuple(profile), max_lag=max_lag, min_n=min_n,
            points=len(diffs_a), gaps=gaps, pairs_dropped_at_gaps=dropped,
            selection=kpi_series.selection, filters=dict(kpi_series.filters))

    # -- test_reverse_causation() ------------------------------------------------
    def test_reverse_causation(self, df: pd.DataFrame, kpi_key: str, cause_kpi: str,
                               grain: str = "week", max_lag: int = 5,
                               time_filter: Optional[Union[Mapping[str, Any], TimeFilter]] = None,
                               filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
                               min_n: int = MIN_LAG_PAIRS) -> ReverseCausation:
        self._api.require(kpi_key)
        self._api.require(cause_kpi)
        label, cause_label = self._api.label(kpi_key), self._api.label(cause_kpi)

        # `components(X)` is "what X's own formula is built from" -- so a hit
        # naming `cause_kpi` inside `components(kpi_key)` means the KPI is
        # downstream of the cause, and vice versa.
        kpi_built_from_cause = next(
            (r for r in self._graph.components(kpi_key) if r.source_kpi == cause_kpi), None)
        cause_built_from_kpi = next(
            (r for r in self._graph.components(cause_kpi) if r.source_kpi == kpi_key), None)

        mechanical: str
        relation: Optional[str] = None
        evidence: Tuple[str, ...] = ()
        if kpi_built_from_cause and cause_built_from_kpi:
            mechanical = "mutual"
            relation, evidence = kpi_built_from_cause.relation, kpi_built_from_cause.evidence_fields
        elif kpi_built_from_cause:
            mechanical = "kpi_derived_from_cause"
            relation, evidence = kpi_built_from_cause.relation, kpi_built_from_cause.evidence_fields
        elif cause_built_from_kpi:
            mechanical = "cause_derived_from_kpi"
            relation, evidence = cause_built_from_kpi.relation, cause_built_from_kpi.evidence_fields
        else:
            shared = next(
                (r for r in self._graph.related(kpi_key, relations=_SHARED_INPUT_RELATIONS)
                if r.source_kpi == cause_kpi), None)
            if shared is None:
                shared = next(
                    (r for r in self._graph.related(cause_kpi, relations=_SHARED_INPUT_RELATIONS)
                    if r.source_kpi == kpi_key), None)
            if shared is not None:
                mechanical = "shared_inputs"
                relation, evidence = shared.relation, shared.evidence_fields
            else:
                mechanical = "none"

        timing = self.cross_correlate_lagged(df, kpi_key, cause_kpi, grain=grain, max_lag=max_lag,
                                             time_filter=time_filter, filters=filters, min_n=min_n)

        if mechanical in ("cause_derived_from_kpi", "kpi_derived_from_cause", "mutual"):
            verdict = "definitional"
        elif timing.status == "ok" and timing.verdict == "kpi_leads":
            verdict = "reverse_supported"
        elif timing.status == "ok" and timing.verdict == "cause_leads":
            verdict = "forward_supported"
        else:
            verdict = "undetermined"

        return ReverseCausation(
            kpi=kpi_key, cause_kpi=cause_kpi, label=label, cause_label=cause_label,
            mechanical=mechanical, mechanical_relation=relation,
            mechanical_evidence_fields=tuple(evidence), verdict=verdict, timing=timing)
