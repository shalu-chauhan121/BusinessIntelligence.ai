"""
`test_statistical_significance` / `check_sample_adequacy` -- could this
correlation be chance, and was the sample ever big enough to ask.

`CorrelationEngine` (`agent/correlate.py`) reports a bare Pearson `r` with an
`n` and nothing else: no p-value, no confidence interval, no standard error.
Its docstring says so and defers the fix here, as `agent/trend.py:11-15` and
`agent/seasonality.py:63-66` each did in turn -- this is the batch that takes
`scipy` and closes all three deferrals. What the gap costs is concrete:
`contest.score_hypothesis` awards `+1.1*|r|` behind a hard `|r| >= 0.5` gate,
and at the sample sizes these fixtures actually have -- retail `region` has
four members, so a cross-sectional correlation runs at `n=4` -- that gate
clears by luck roughly one time in twenty.

**Everything here composes `CorrelationEngine` rather than recomputing `r`.**
`correlate_kpis` and `test_statistical_significance` cannot report different
numbers for the same pair, because the second calls the first. Same reasoning
`agent/consistency.py` gives for delegating to `CorrelationEngine` and
`SegmentEngine` instead of re-deriving their arithmetic.

**Multiplicity is corrected, and both verdicts are published.** A Tier-4 sweep
tests up to `correlate.MAX_MATRIX_CANDIDATES` (30) candidate drivers in one
call. At alpha=0.05 that yields roughly 1.5 "significant" drivers by chance
alone -- precisely the false finding this module exists to prevent, and one a
model reading 30 rows of raw `p` has no way to discount. Benjamini-Hochberg
adjusts them to `q_value`. `significant_raw` and `significant_fdr` are *both*
reported rather than only the corrected verdict, so the model can see when
multiplicity is what killed a candidate rather than the evidence itself.

**Nothing is fabricated where there is no sample.** Fisher's z needs `n >= 4`
(`se = 1/sqrt(n-3)`), so `n=3` yields a real p-value alongside
`ci_status="insufficient_for_ci"` -- never an interval invented from a
degenerate standard error. Every non-`ok` correlation status propagates through
with `p`, `q` and the interval all `None`, the G8 discipline the rest of the
agent layer already follows.

`check_sample_adequacy` answers the question *before* a correlation is run, and
answers it as a **minimum detectable effect** rather than a yes/no: measured on
retail `region`'s four members, only `|r| >= 0.95` is detectable at 80% power.
That is the quantified reason decision 43 set the consistency floor to three
members and published a sign-agreement rate beside the pooled `r` rather than
trusting `r` alone.

One honest limit, stated rather than assumed away: a time-series `n` counts
period pairs, and the p-value treats them as independent. Both time-series
paths here are first-differenced upstream, which removes most of the serial
dependence that would otherwise inflate significance, but no autocorrelation
correction to the effective `n` is applied. `transform` travels on every row so
a caller can see which sample the degrees of freedom came from.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from ..engines.metrics import safe
from .contract_api import ContractAPI
from .correlate import DEFAULT_MIN_N, Correlation, CorrelationEngine, CorrelationMatrix
from .errors import InvalidArgumentError
from .query import QueryEngine
from .series import SERIES_GRAINS, SeriesEngine, period_ordinal
from .timefilter import TimeFilter, TimeSelection

# Module constants in the style of `correlate.MAX_MATRIX_CANDIDATES` and
# `scan.MAX_SCAN_KPIS`, not `Settings` fields: these are properties of the
# statistics, not knobs a deployment tunes.
DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.80

# A Pearson t-test needs `df = n - 2 >= 1`; Fisher's z needs `n - 3 >= 1` for a
# finite standard error. The two floors differ, which is exactly why a result
# can legitimately carry a real p-value and no interval.
MIN_N_FOR_P = 3
MIN_N_FOR_CI = 4

ADEQUACY_VERDICTS = ("adequate", "marginal", "inadequate")


# ---------------------------------------------------------------------------
# The statistics. Module-level, so `agent/confounders.py` qualifies a partial
# correlation with this arithmetic rather than a second copy free to drift --
# the same reason `trend.theil_sen` is a shared function.
# ---------------------------------------------------------------------------
def p_value_for_r(r: Optional[float], n: int, controls: int = 0) -> Optional[float]:
    """
    Two-sided p for a Pearson (or partial) `r`, from the t-statistic
    `r*sqrt(df/(1-r^2))` with `df = n - 2 - controls`.

    Identical to `scipy.stats.pearsonr`'s p when `controls=0` -- pinned by a
    test against the raw arrays -- but computable from `(r, n)` alone, which is
    what lets this module qualify a `Correlation` it did not compute itself.

    Deliberately fed the *published* `r`, which `pearson_correlate` has already
    rounded to three places, so a reader recomputing p from the r in the payload
    gets the number in the payload. The one visible consequence: an `|r|` that
    rounds to 1.0 reports `p = 0.0` rather than the ~1e-150 the unrounded r
    would give. Both mean the same thing, and internal consistency between the
    two published figures is worth more here than a tail digit.
    """
    if r is None or n - 2 - controls < 1:
        return None
    if abs(r) >= 1.0:
        return 0.0
    df = n - 2 - controls
    t = r * np.sqrt(df / (1.0 - r * r))
    return float(2.0 * stats.t.sf(abs(t), df))


def fisher_interval(r: Optional[float], n: int, alpha: float = DEFAULT_ALPHA,
                    controls: int = 0) -> Tuple[Optional[float], Optional[float], str]:
    """
    `(low, high, status)` -- the Fisher z-transformed confidence interval, back
    through `tanh` so it respects the [-1, 1] bound instead of running off the
    end of the scale the way a naive `r +- 1.96*se` does.

    `status` is `"ok"` or `"insufficient_for_ci"`, the second whenever
    `n - 3 - controls < 1` leaves the standard error undefined -- so an interval
    is never invented at the sample sizes where one would matter most.
    """
    if r is None or n - 3 - controls < 1:
        return None, None, "insufficient_for_ci"
    if abs(r) >= 1.0:
        edge = float(np.sign(r))
        return edge, edge, "ok"
    se = 1.0 / np.sqrt(n - 3 - controls)
    z = np.arctanh(r)
    crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    return float(np.tanh(z - crit * se)), float(np.tanh(z + crit * se)), "ok"


def benjamini_hochberg(p_values: Sequence[Optional[float]]) -> List[Optional[float]]:
    """
    Benjamini-Hochberg adjusted q-values, returned in the input's order.

    `q_i = min over j >= i of (p_j * m / j)` across the p-values sorted
    ascending, enforced monotone and clipped at 1. `None` entries -- a candidate
    whose correlation had no usable sample -- carry through as `None` and are
    excluded from the family size `m`: a test that was never run cannot consume
    a share of the false-discovery budget.
    """
    indexed = [(i, p) for i, p in enumerate(p_values) if p is not None]
    out: List[Optional[float]] = [None] * len(p_values)
    m = len(indexed)
    if m == 0:
        return out
    indexed.sort(key=lambda pair: pair[1])
    running = 1.0
    for rank in range(m, 0, -1):
        idx, p = indexed[rank - 1]
        running = min(running, p * m / rank)
        out[idx] = float(min(1.0, running))
    return out


def min_detectable_r(n: int, alpha: float = DEFAULT_ALPHA,
                     power: float = DEFAULT_POWER) -> Optional[float]:
    """
    The smallest `|r|` a sample of `n` could detect at `alpha` with `power`, via
    the Fisher-z normal approximation
    `r = tanh((z_{1-alpha/2} + z_power) / sqrt(n-3))`.

    The honest form of "is n big enough": a number the caller compares against
    the effect it actually cares about, rather than a bare verdict.
    """
    if n - 3 < 1:
        return None
    crit = float(stats.norm.ppf(1.0 - alpha / 2.0) + stats.norm.ppf(power))
    return float(np.tanh(crit / np.sqrt(n - 3)))


def _rounded(value: Optional[float], places: int = 4) -> Optional[float]:
    return None if value is None else round(value, places)


def _round_p(p: Optional[float]) -> Optional[float]:
    """Four significant figures, not four decimal places.

    A p of 1.4e-11 rounded to decimals prints as `0.0`, which reads as a
    placeholder rather than as overwhelming evidence -- and this module's whole
    job is that the number a model reads means what it says.
    """
    if p is None:
        return None
    if p == 0.0:
        return 0.0
    return float(f"{p:.4g}")


def _adequacy_verdict(n: int, min_n: int) -> str:
    """`min_n` is the floor below which a correlation is refused outright;
    twice `MIN_N_FOR_CI` is the point at which the interval stops spanning
    most of the scale. Between them the sample supports a number without
    supporting much confidence in it."""
    if n < min_n:
        return "inadequate"
    if n < MIN_N_FOR_CI * 2:
        return "marginal"
    return "adequate"


# ---------------------------------------------------------------------------
# test_statistical_significance()
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Significance:
    kpi: str
    cause_kpi: str
    label: str
    cause_label: str
    mode: str
    transform: str
    basis: str
    method: str                        # "pearson"
    status: str                        # the Correlation's own status, propagated
    r: Optional[float]
    r_squared: Optional[float]
    n: int
    degrees_of_freedom: Optional[int]
    p_value: Optional[float]
    q_value: Optional[float]           # Benjamini-Hochberg adjusted
    ci_status: str                     # "ok" | "insufficient_for_ci"
    ci_low: Optional[float]
    ci_high: Optional[float]
    ci_excludes_zero: Optional[bool]
    alpha: float
    significant_raw: Optional[bool]
    significant_fdr: Optional[bool]
    min_detectable_r: Optional[float]
    relation: Optional[str] = None     # RelationGraph's label, on a sweep row
    dimension: Optional[str] = None
    grain: Optional[str] = None

    def to_payload(self) -> Dict[str, Any]:
        return {
            "kpi": self.kpi, "cause_kpi": self.cause_kpi,
            "label": self.label, "cause_label": self.cause_label,
            "mode": self.mode, "transform": self.transform, "basis": self.basis,
            "method": self.method, "status": self.status,
            "r": safe(self.r), "r_squared": safe(self.r_squared), "n": self.n,
            "degrees_of_freedom": self.degrees_of_freedom,
            "p_value": safe(self.p_value), "q_value": safe(self.q_value),
            "ci_status": self.ci_status,
            "ci_low": safe(self.ci_low), "ci_high": safe(self.ci_high),
            "ci_excludes_zero": self.ci_excludes_zero, "alpha": self.alpha,
            "significant_raw": self.significant_raw,
            "significant_fdr": self.significant_fdr,
            "min_detectable_r": safe(self.min_detectable_r),
            "relation": self.relation,
            "dimension": self.dimension, "grain": self.grain,
        }


@dataclass(frozen=True)
class SignificanceSet:
    kpi: str
    label: str
    mode: str
    alpha: float
    family_size: int
    considered: int
    truncated: bool
    rows: Tuple[Significance, ...]
    ordered_by: str
    selection: Optional[TimeSelection] = None
    filters: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        payload = {
            "kpi": self.kpi, "label": self.label, "mode": self.mode,
            "alpha": self.alpha, "family_size": self.family_size,
            "considered": self.considered, "truncated": self.truncated,
            "ordered_by": self.ordered_by,
            "rows": [r.to_payload() for r in self.rows],
            "filters": {k: list(v) for k, v in self.filters.items()},
        }
        if self.selection is not None:
            payload["selection"] = self.selection.to_payload()
        return payload


# ---------------------------------------------------------------------------
# check_sample_adequacy()
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AdequacyAxis:
    basis: str                         # "member" | "period"
    scope: str                         # the dimension name, or the grain
    units: int                         # members present / periods held
    usable_points: int                 # what a correlation would actually get
    gaps: int
    min_detectable_r: Optional[float]
    verdict: str                       # one of ADEQUACY_VERDICTS

    def to_payload(self) -> Dict[str, Any]:
        return {
            "basis": self.basis, "scope": self.scope, "units": self.units,
            "usable_points": self.usable_points, "gaps": self.gaps,
            "min_detectable_r": safe(self.min_detectable_r), "verdict": self.verdict,
        }


@dataclass(frozen=True)
class SampleAdequacy:
    kpi: str
    label: str
    alpha: float
    power: float
    min_n: int
    axes: Tuple[AdequacyAxis, ...]
    best_basis: Optional[str]
    best_scope: Optional[str]
    selection: Optional[TimeSelection] = None
    filters: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        payload = {
            "kpi": self.kpi, "label": self.label,
            "alpha": self.alpha, "power": self.power, "min_n": self.min_n,
            "axes": [a.to_payload() for a in self.axes],
            "best_basis": self.best_basis, "best_scope": self.best_scope,
            "filters": {k: list(v) for k, v in self.filters.items()},
        }
        if self.selection is not None:
            payload["selection"] = self.selection.to_payload()
        return payload


class SignificanceEngine:
    """`test_statistical_significance` / `check_sample_adequacy`, bound to one
    dataset's `ContractAPI`."""

    def __init__(self, api: ContractAPI):
        self._api = api
        self._correlate = CorrelationEngine(api)
        self._query = QueryEngine(api)
        self._series = SeriesEngine(api)

    # -- test_statistical_significance() ---------------------------------------
    def qualify(self, corr: Correlation, alpha: float = DEFAULT_ALPHA,
                relation: Optional[str] = None, controls: int = 0) -> Significance:
        """
        One `Correlation` plus its p-value, interval and detectability.

        `q_value` is deliberately left `None` here and filled once the family is
        known -- a q-value is a property of the family, not of the pair, and
        computing one per call is what makes an uncorrected sweep look
        corrected. Public because `agent/confounders.py` qualifies a partial
        correlation through the identical path, passing `controls` so the
        degrees of freedom shrink by the number of variables held constant.
        """
        p = p_value_for_r(corr.r, corr.n, controls)
        lo, hi, ci_status = fisher_interval(corr.r, corr.n, alpha, controls)
        excludes = None if lo is None or hi is None else bool(lo > 0.0 or hi < 0.0)
        df = corr.n - 2 - controls
        return Significance(
            kpi=corr.kpi_a, cause_kpi=corr.kpi_b,
            label=corr.label_a, cause_label=corr.label_b,
            mode=corr.mode, transform=corr.transform, basis=corr.basis, method=corr.method,
            status=corr.status, r=corr.r, r_squared=corr.r_squared, n=corr.n,
            degrees_of_freedom=df if (corr.r is not None and df >= 1) else None,
            p_value=_round_p(p), q_value=None,
            ci_status=ci_status, ci_low=_rounded(lo), ci_high=_rounded(hi),
            ci_excludes_zero=excludes, alpha=alpha,
            significant_raw=None if p is None else bool(p < alpha),
            significant_fdr=None,
            min_detectable_r=_rounded(min_detectable_r(corr.n, alpha)),
            relation=relation, dimension=corr.dimension, grain=corr.grain)

    def test_statistical_significance(
            self, df: pd.DataFrame, kpi_key: str,
            cause_kpi: Optional[str] = None,
            candidates: Optional[Sequence[str]] = None,
            mode: str = "cross_sectional",
            period_a: Optional[Union[Mapping[str, Any], TimeFilter]] = None,
            period_b: Optional[Union[Mapping[str, Any], TimeFilter]] = None,
            dimension: Optional[str] = None,
            time_filter: Optional[Union[Mapping[str, Any], TimeFilter]] = None,
            grain: str = "quarter",
            filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
            min_n: int = DEFAULT_MIN_N,
            alpha: float = DEFAULT_ALPHA) -> SignificanceSet:
        """
        Is the association between `kpi_key` and one `cause_kpi` -- or each of
        `candidates` -- distinguishable from chance?

        Exactly one of `cause_kpi` / `candidates` is required. The sweep form is
        not a convenience: testing candidates one call at a time makes the
        family size unknowable, and an uncorrected family of thirty is the
        failure this tool exists to catch. Every scoping argument is passed
        straight through to `CorrelationEngine`, which stays the single owner of
        what "this scope" means.
        """
        if (cause_kpi is None) == (candidates is None):
            raise InvalidArgumentError(
                "cause_kpi", cause_kpi,
                "exactly one of cause_kpi or candidates is required.")
        if not 0.0 < alpha < 1.0:
            raise InvalidArgumentError("alpha", alpha, "must be strictly between 0 and 1.")
        self._api.require(kpi_key)

        shared = dict(mode=mode, period_a=period_a, period_b=period_b, dimension=dimension,
                      time_filter=time_filter, grain=grain, filters=filters, min_n=min_n)

        if cause_kpi is not None:
            self._api.require(cause_kpi)
            corr = self._correlate.correlate_kpis(df, kpi_key, cause_kpi, **shared)
            rows = [self.qualify(corr, alpha)]
            considered, truncated = 1, False
            selection, filters_out = corr.selection, corr.filters
        else:
            matrix: CorrelationMatrix = self._correlate.correlate_kpi_matrix(
                df, kpi_key, candidates=candidates, **shared)
            rows = [self.qualify(row.correlation, alpha, row.relation) for row in matrix.rows]
            considered, truncated = matrix.considered, matrix.truncated
            selection, filters_out = matrix.selection, matrix.filters

        q_values = benjamini_hochberg([row.p_value for row in rows])
        rows = [replace(row, q_value=_round_p(q),
                        significant_fdr=None if q is None else bool(q < alpha))
                for row, q in zip(rows, q_values)]
        family_size = sum(1 for row in rows if row.p_value is not None)

        # A row with no usable sample has no p-value to sort on and belongs at
        # the bottom whatever else is true; the key breaks ties so the order is
        # stable across runs (G5).
        ranked = sorted(rows, key=lambda row: (row.p_value is None,
                                               row.p_value if row.p_value is not None else 1.0,
                                               row.cause_kpi))
        return SignificanceSet(
            kpi=kpi_key, label=self._api.label(kpi_key), mode=mode, alpha=alpha,
            family_size=family_size, considered=considered, truncated=truncated,
            rows=tuple(ranked), ordered_by="p_value", selection=selection,
            filters=filters_out)

    # -- check_sample_adequacy() ------------------------------------------------
    def check_sample_adequacy(
            self, df: pd.DataFrame, kpi_key: str,
            dimension: Optional[str] = None,
            time_filter: Optional[Union[Mapping[str, Any], TimeFilter]] = None,
            grain: str = "quarter",
            filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
            min_n: int = DEFAULT_MIN_N,
            alpha: float = DEFAULT_ALPHA,
            power: float = DEFAULT_POWER) -> SampleAdequacy:
        """
        How many points a correlation on this KPI would actually get on each
        axis available, and the smallest `|r|` that many points could detect.

        Both bases are reported because they answer differently, and a model
        choosing between `mode="cross_sectional"` and `mode="time_series"` needs
        to see which one its dataset can support: on every fixture here the
        period axis is an order of magnitude larger than the member axis, and
        nothing in the system says so today.
        """
        self._api.require(kpi_key)
        if not 0.0 < alpha < 1.0:
            raise InvalidArgumentError("alpha", alpha, "must be strictly between 0 and 1.")
        if not 0.0 < power < 1.0:
            raise InvalidArgumentError("power", power, "must be strictly between 0 and 1.")
        if grain not in SERIES_GRAINS:
            raise InvalidArgumentError("grain", grain, "must be one of the supported grains.",
                                       list(SERIES_GRAINS))
        tf = time_filter if time_filter is not None else {"type": "all"}

        axes: List[AdequacyAxis] = []
        selection: Optional[TimeSelection] = None
        filters_out: Mapping[str, Tuple[str, ...]] = {}

        # -- member axis: one entry per dimension, or just the named one ------
        dims = ([self._api.require_dimension(dimension)] if dimension is not None
                else [d.name for d in self._api.list_dimensions(df)])
        for dim in sorted(dims):
            result = self._query.query(df, [kpi_key], tf, filters=filters, group_by=[dim])
            selection, filters_out = result.selection, result.filters
            units = len(result.cells)
            axes.append(AdequacyAxis(
                basis="member", scope=dim, units=units, usable_points=units, gaps=0,
                min_detectable_r=_rounded(min_detectable_r(units, alpha, power)),
                verdict=_adequacy_verdict(units, min_n)))

        # -- period axis: usable points are *adjacent* pairs, not points ------
        # A correlation over a differenced series gets one point per
        # calendar-adjacent pair, so a series with holes yields fewer than
        # `len(points) - 1`. Counting points here would overstate the sample
        # this tool exists to be honest about.
        series = self._series.series(df, kpi_key, grain=grain, time_filter=tf, filters=filters)
        periods = [p.period for p in series.points]
        pairs = sum(1 for i in range(1, len(periods))
                    if period_ordinal(grain, periods[i])
                    - period_ordinal(grain, periods[i - 1]) == 1)
        axes.append(AdequacyAxis(
            basis="period", scope=grain, units=len(periods), usable_points=pairs,
            gaps=len(series.gaps),
            min_detectable_r=_rounded(min_detectable_r(pairs, alpha, power)),
            verdict=_adequacy_verdict(pairs, min_n)))

        # Sorted key, not just `max`, so a tie resolves the same way every run.
        best = max(axes, key=lambda a: (a.usable_points, a.scope)) if axes else None
        return SampleAdequacy(
            kpi=kpi_key, label=self._api.label(kpi_key), alpha=alpha, power=power,
            min_n=min_n, axes=tuple(axes),
            best_basis=None if best is None else best.basis,
            best_scope=None if best is None else best.scope,
            selection=selection if selection is not None else series.selection,
            filters=filters_out or series.filters)
