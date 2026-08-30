"""
`test_confounders` / `test_spurious_correlation` / `rank_competing_explanations`
-- is the driver real, or a third variable, a shared trend, or the weakest of
several candidates that all look alike in isolation.

**Every relationship in this system is bivariate.** Nothing anywhere controls
for a third variable, and nothing checks whether two series correlate only
because both trend. On a business dataset -- where revenue, cost, headcount and
volume all grow together -- that is the default failure mode, not an edge case:
two independently rising series correlate near 1.0 on levels, which is the
reason `analysis.lead_lag` first-differences (`engines/analysis.py:277-282`) and
the reason `agent/trend.py` fits Theil-Sen rather than least squares. This
module is where that reasoning finally becomes a question the model can ask.

**One aligned sample, or the arithmetic is not merely imprecise but invalid.**
Partial correlation reads three pairwise `r`s. Taken on three *different*
overlaps -- which is what happens if each pair is correlated independently --
they do not form a positive semi-definite matrix, and
`(r_ab - r_ac*r_bc)/sqrt((1-r_ac^2)(1-r_bc^2))` can then return a number
outside [-1, 1]: an impossible correlation, reported with no sign that anything
went wrong. So every `r` here comes from one `correlate.PairedSample` under
listwise deletion (`PairedSample.listwise`), and `dropped_for_alignment` is
published because that `n` can legitimately differ from a standalone
`correlate_kpis` call on the same pair -- better the model sees the difference
than discovers it.

**p-values and intervals come from `agent/significance.py`, not a second copy.**
`p_value_for_r` and `fisher_interval` both take a `controls` count, so a partial
correlation's degrees of freedom shrink by the number of variables held
constant. The same composition rule X3 follows for `CorrelationEngine`.

**`rank_competing_explanations` publishes no score.** Decision 45 records that
the agent tools deliberately produce no confidence number -- that judgment is
the model's now -- and this is the tool whose name most invites breaking that.
It assembles one row per candidate on one shared basis (correlation, p and q
from X3; precedence lag from X1; cross-sectional agreement from X2; and each
candidate's partial correlation controlling for **every other candidate**), and
orders them by a single declared, reproducible key published in the payload as
`ordered_by`. Mutual control is the point: "is it pricing, cost, or mix" is
asking which candidate survives the others, and that is not answerable by three
separate calls.

`test_spurious_correlation` is time-series only, and says so rather than
returning a meaningless number: a cross-sectional sample of members has no time
axis and therefore no trend to remove. It reports `r` on all three transforms
of the same pair -- levels, Theil-Sen residuals, and first differences -- because
the finding is the *gap* between them, not any one of the three. Detrending
reuses `trend.theil_sen`, the identical fit `detect_trend` and
`detect_seasonality` already use, rather than a fourth hand-rolled copy.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..engines.metrics import safe
from .consistency import ConsistencyEngine
from .contract_api import ContractAPI
from .correlate import (DEFAULT_MIN_N, MODES, Correlation, CorrelationEngine,
                        PairedSample, pearson_correlate)
from .errors import InvalidArgumentError
from .relations import RelationGraph
from .series import SERIES_GRAINS, SeriesEngine, period_ordinal
from .significance import (DEFAULT_ALPHA, Significance, SignificanceEngine,
                           _round_p, _rounded, fisher_interval, p_value_for_r)
from .temporal import TemporalEngine
from .timefilter import TimeFilter, TimeSelection
from .trend import theil_sen

# `rank_competing_explanations` is O(k) in X1/X2 calls and O(k^2) in controls,
# so its cap is far below `correlate.MAX_MATRIX_CANDIDATES` (30). A field of
# candidates wider than this is a sweep -- `correlate_kpi_matrix` followed by
# `test_statistical_significance` -- not a comparison of rival explanations.
MAX_COMPETING_CANDIDATES = 8

# Controls this numerous exhaust the degrees of freedom of every sample these
# fixtures hold: a joint partial on 4 members with 5 controls has df < 1.
MAX_CONTROLS = 10

# Attenuation bands for the `effect` verdict, in percent of the baseline |r|.
# Published in the payload rather than hidden, so a caller can see the cut
# points the classification used.
EXPLAINED_AWAY_ATTENUATION_PCT = 50.0
ATTENUATED_ATTENUATION_PCT = 15.0
# Below this the partial is indistinguishable from no relationship, so a sign
# on it is noise and must not be reported as a flip.
NEGLIGIBLE_R = 0.05

EFFECTS = ("unchanged", "attenuated", "explained_away", "strengthened", "sign_flipped")
CONTROL_STATUSES = ("ok", "collinear_controls", "insufficient_n_for_controls",
                    "insufficient_n", "undefined_sample")
SPURIOUS_VERDICTS = ("survives_detrending", "trend_driven", "insufficient")
COMPETING_SORT_KEYS = ("abs_partial_r", "abs_r", "p_value")


# ---------------------------------------------------------------------------
# the maths
# ---------------------------------------------------------------------------
def first_order_partial(r_ab: float, r_ac: float, r_bc: float) -> Optional[float]:
    """
    `r_ab.c` -- the correlation between a and b with c held constant.

    Returns `None` when a control is perfectly collinear with either variable,
    which drives the denominator to zero: that is "the control explains one of
    them entirely", a typed answer, not a value.
    """
    denom = (1.0 - r_ac * r_ac) * (1.0 - r_bc * r_bc)
    if denom <= 0.0:
        return None
    return (r_ab - r_ac * r_bc) / float(np.sqrt(denom))


def joint_partial(matrix: np.ndarray, i: int, j: int) -> Optional[float]:
    """
    `r_ij.rest` from the inverse correlation matrix (the precision matrix):
    `-P_ij / sqrt(P_ii * P_jj)`.

    Generalises `first_order_partial` to any number of simultaneous controls.
    Returns `None` when the matrix is singular or too ill-conditioned to invert
    meaningfully -- collinear controls, common when a candidate is a formula
    component of the KPI -- so the caller gets a typed status rather than a
    `LinAlgError` or a number built out of floating-point noise.
    """
    if not np.all(np.isfinite(matrix)):
        return None
    try:
        if np.linalg.cond(matrix) > 1e10:
            return None
        precision = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return None
    denom = precision[i, i] * precision[j, j]
    if denom <= 0.0:
        return None
    return float(-precision[i, j] / np.sqrt(denom))


def _supports_controls(n: int, k: int) -> bool:
    """
    Can `n` observations support a partial correlation holding `k` variables
    constant at once? Needs `df = n - 2 - k >= 1`.

    Told apart from collinearity deliberately. A 5-variable matrix estimated
    from 4 members is rank-deficient by construction, and `numpy.linalg.cond`
    reports that as ill-conditioning -- indistinguishable, from the outside,
    from two genuinely redundant candidates. They call for opposite responses:
    drop a *candidate* for collinearity, change the *axis* for too few
    observations. Measured on the retail fixture, ranking four candidates
    cross-sectionally over `region`'s four members hits this case every time,
    and reporting it as collinearity would send a model hunting for a redundancy
    that is not there.
    """
    return n - 2 - k >= 1


def _clamp(r: Optional[float]) -> Tuple[Optional[float], bool]:
    """Correlations are bounded on [-1, 1]; a float artefact that prints as
    1.0000000002 is a bug the caller should see flagged, not silently rounded
    away."""
    if r is None or not np.isfinite(r):
        return None, False
    if r > 1.0:
        return 1.0, True
    if r < -1.0:
        return -1.0, True
    return float(r), False


def _classify_effect(r_baseline: Optional[float], r_partial: Optional[float]
                     ) -> Tuple[Optional[str], Optional[float]]:
    """`(effect, attenuation_pct)` -- how much of the baseline association
    survived the control, and what that shift is called."""
    if r_baseline is None or r_partial is None or abs(r_baseline) < NEGLIGIBLE_R:
        return None, None
    attenuation = (abs(r_baseline) - abs(r_partial)) / abs(r_baseline) * 100.0
    if abs(r_partial) >= NEGLIGIBLE_R and np.sign(r_partial) != np.sign(r_baseline):
        effect = "sign_flipped"
    elif attenuation >= EXPLAINED_AWAY_ATTENUATION_PCT:
        effect = "explained_away"
    elif attenuation >= ATTENUATED_ATTENUATION_PCT:
        effect = "attenuated"
    elif attenuation <= -ATTENUATED_ATTENUATION_PCT:
        effect = "strengthened"
    else:
        effect = "unchanged"
    return effect, round(attenuation, 2)


def _correlation_matrix(sample: PairedSample, keys: Sequence[str]
                        ) -> Tuple[Optional[np.ndarray], int, int]:
    """
    `(R, n, dropped)` over the entries where every key is finite.

    Listwise, not pairwise: this is the whole reason `PairedSample` exists as a
    published seam. A key that is constant over the surviving rows makes `R`
    undefined, which is returned as `None` rather than a matrix of NaN.
    """
    _, arrays, dropped = sample.listwise(keys)
    n = len(next(iter(arrays.values()))) if arrays else 0
    if n < 3:
        return None, n, dropped
    stacked = np.vstack([arrays[k] for k in keys])
    if np.any(stacked.std(axis=1) == 0):
        return None, n, dropped
    return np.corrcoef(stacked), n, dropped


# ---------------------------------------------------------------------------
# test_confounders()
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ControlResult:
    control_kpi: Optional[str]         # None on the joint row
    control_label: Optional[str]
    controls: Tuple[str, ...]
    n_controls: int
    relation: Optional[str]            # RelationGraph's label for the control
    status: str                        # "ok" | "collinear_controls" | "insufficient_n" | "undefined_sample"
    r_partial: Optional[float]
    clamped: bool
    attenuation_pct: Optional[float]
    effect: Optional[str]
    degrees_of_freedom: Optional[int]
    p_value: Optional[float]
    ci_status: str
    ci_low: Optional[float]
    ci_high: Optional[float]
    significant: Optional[bool]
    r_control_kpi: Optional[float] = None     # r between the control and the KPI
    r_control_cause: Optional[float] = None   # r between the control and the cause

    def to_payload(self) -> Dict[str, Any]:
        return {
            "control_kpi": self.control_kpi, "control_label": self.control_label,
            "controls": list(self.controls), "n_controls": self.n_controls,
            "relation": self.relation, "status": self.status,
            "r_partial": safe(self.r_partial), "clamped": self.clamped,
            "attenuation_pct": safe(self.attenuation_pct), "effect": self.effect,
            "degrees_of_freedom": self.degrees_of_freedom,
            "p_value": safe(self.p_value), "ci_status": self.ci_status,
            "ci_low": safe(self.ci_low), "ci_high": safe(self.ci_high),
            "significant": self.significant,
            "r_control_kpi": safe(self.r_control_kpi),
            "r_control_cause": safe(self.r_control_cause),
        }


@dataclass(frozen=True)
class ConfounderTest:
    kpi: str
    cause_kpi: str
    label: str
    cause_label: str
    mode: str
    transform: str
    basis: str
    status: str                        # "ok" | "insufficient_n" | "undefined_sample"
    r_baseline: Optional[float]
    n: int
    dropped_for_alignment: int
    alpha: float
    baseline: Optional[Significance]
    rows: Tuple[ControlResult, ...]
    joint: Optional[ControlResult]
    thresholds: Mapping[str, float]
    dimension: Optional[str] = None
    grain: Optional[str] = None
    selection: Optional[TimeSelection] = None
    filters: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        payload = {
            "kpi": self.kpi, "cause_kpi": self.cause_kpi,
            "label": self.label, "cause_label": self.cause_label,
            "mode": self.mode, "transform": self.transform, "basis": self.basis,
            "status": self.status, "r_baseline": safe(self.r_baseline), "n": self.n,
            "dropped_for_alignment": self.dropped_for_alignment, "alpha": self.alpha,
            "baseline": None if self.baseline is None else self.baseline.to_payload(),
            "rows": [r.to_payload() for r in self.rows],
            "joint": None if self.joint is None else self.joint.to_payload(),
            "thresholds": dict(self.thresholds),
            "dimension": self.dimension, "grain": self.grain,
            "filters": {k: list(v) for k, v in self.filters.items()},
        }
        if self.selection is not None:
            payload["selection"] = self.selection.to_payload()
        return payload


# ---------------------------------------------------------------------------
# test_spurious_correlation()
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TransformCorrelation:
    transform: str                     # "level" | "detrended" | "difference"
    status: str
    r: Optional[float]
    n: int
    p_value: Optional[float]
    ci_status: str
    ci_low: Optional[float]
    ci_high: Optional[float]
    significant: Optional[bool]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "transform": self.transform, "status": self.status, "r": safe(self.r),
            "n": self.n, "p_value": safe(self.p_value), "ci_status": self.ci_status,
            "ci_low": safe(self.ci_low), "ci_high": safe(self.ci_high),
            "significant": self.significant,
        }


@dataclass(frozen=True)
class SpuriousTest:
    kpi: str
    cause_kpi: str
    label: str
    cause_label: str
    grain: str
    status: str
    verdict: Optional[str]             # one of SPURIOUS_VERDICTS
    alpha: float
    method: str                        # "theil_sen"
    level: TransformCorrelation
    detrended: TransformCorrelation
    difference: TransformCorrelation
    kpi_slope_per_period: Optional[float]
    cause_slope_per_period: Optional[float]
    shared_trend_direction: Optional[bool]
    attenuation_pct: Optional[float]
    gaps: Tuple[str, ...]
    selection: Optional[TimeSelection] = None
    filters: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        payload = {
            "kpi": self.kpi, "cause_kpi": self.cause_kpi,
            "label": self.label, "cause_label": self.cause_label,
            "grain": self.grain, "status": self.status, "verdict": self.verdict,
            "alpha": self.alpha, "method": self.method,
            "level": self.level.to_payload(),
            "detrended": self.detrended.to_payload(),
            "difference": self.difference.to_payload(),
            "kpi_slope_per_period": safe(self.kpi_slope_per_period),
            "cause_slope_per_period": safe(self.cause_slope_per_period),
            "shared_trend_direction": self.shared_trend_direction,
            "attenuation_pct": safe(self.attenuation_pct),
            "gaps": list(self.gaps),
            "filters": {k: list(v) for k, v in self.filters.items()},
        }
        if self.selection is not None:
            payload["selection"] = self.selection.to_payload()
        return payload


# ---------------------------------------------------------------------------
# rank_competing_explanations()
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CompetingCandidate:
    cause_kpi: str
    cause_label: str
    relation: Optional[str]
    significance: Optional[Significance]
    r_partial_vs_rivals: Optional[float]
    partial_status: str
    partial_effect: Optional[str]
    attenuation_pct: Optional[float]
    rivals_controlled: Tuple[str, ...]
    lag_periods: Optional[int]
    precedence_status: str
    precedence_verdict: Optional[str]
    consistency_rate: Optional[float]
    consistency_status: str
    consistency_direction: Optional[str]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "cause_kpi": self.cause_kpi, "cause_label": self.cause_label,
            "relation": self.relation,
            "significance": None if self.significance is None else self.significance.to_payload(),
            "r_partial_vs_rivals": safe(self.r_partial_vs_rivals),
            "partial_status": self.partial_status,
            "partial_effect": self.partial_effect,
            "attenuation_pct": safe(self.attenuation_pct),
            "rivals_controlled": list(self.rivals_controlled),
            "lag_periods": self.lag_periods,
            "precedence_status": self.precedence_status,
            "precedence_verdict": self.precedence_verdict,
            "consistency_rate": safe(self.consistency_rate),
            "consistency_status": self.consistency_status,
            "consistency_direction": self.consistency_direction,
        }


@dataclass(frozen=True)
class CompetingExplanations:
    kpi: str
    label: str
    mode: str
    alpha: float
    ordered_by: str
    family_size: int
    considered: int
    truncated: bool
    n: int
    dropped_for_alignment: int
    rows: Tuple[CompetingCandidate, ...]
    dimension: Optional[str] = None
    grain: Optional[str] = None
    selection: Optional[TimeSelection] = None
    filters: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        payload = {
            "kpi": self.kpi, "label": self.label, "mode": self.mode, "alpha": self.alpha,
            "ordered_by": self.ordered_by, "family_size": self.family_size,
            "considered": self.considered, "truncated": self.truncated,
            "n": self.n, "dropped_for_alignment": self.dropped_for_alignment,
            "rows": [r.to_payload() for r in self.rows],
            "dimension": self.dimension, "grain": self.grain,
            "filters": {k: list(v) for k, v in self.filters.items()},
        }
        if self.selection is not None:
            payload["selection"] = self.selection.to_payload()
        return payload


class ConfounderEngine:
    """`test_confounders` / `test_spurious_correlation` /
    `rank_competing_explanations`, bound to one dataset's `ContractAPI`."""

    def __init__(self, api: ContractAPI):
        self._api = api
        self._correlate = CorrelationEngine(api)
        self._significance = SignificanceEngine(api)
        self._series = SeriesEngine(api)
        self._graph = RelationGraph(api)

    def _relation_labels(self, kpi_key: str) -> Dict[str, str]:
        """`RelationGraph`'s label per neighbour, built the way
        `correlate_kpi_matrix` builds it (`correlate.py`): a control that is a
        formula component of the KPI is mechanical, not a confounder, and the
        model needs to see that without a second call."""
        out: Dict[str, str] = {}
        for relation in self._graph.related(kpi_key):
            out.setdefault(relation.source_kpi, relation.relation)
        return out

    # -- test_confounders() ----------------------------------------------------
    def test_confounders(self, df: pd.DataFrame, kpi_key: str, cause_kpi: str,
                         candidates: Sequence[str],
                         mode: str = "cross_sectional",
                         period_a: Optional[Union[Mapping[str, Any], TimeFilter]] = None,
                         period_b: Optional[Union[Mapping[str, Any], TimeFilter]] = None,
                         dimension: Optional[str] = None,
                         time_filter: Optional[Union[Mapping[str, Any], TimeFilter]] = None,
                         grain: str = "quarter",
                         filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
                         min_n: int = DEFAULT_MIN_N,
                         alpha: float = DEFAULT_ALPHA) -> ConfounderTest:
        """
        Does the `kpi ~ cause` association survive holding each candidate
        confounder constant -- and all of them at once?

        Every correlation is taken over one listwise-aligned sample, which is
        what makes the three `r`s a partial correlation reads mutually
        consistent. `n` and `dropped_for_alignment` are published because that
        `n` can differ from a standalone `correlate_kpis` call on the same pair.
        """
        if mode not in MODES:
            raise InvalidArgumentError("mode", mode, "must be one of the supported modes.",
                                       list(MODES))
        if not 0.0 < alpha < 1.0:
            raise InvalidArgumentError("alpha", alpha, "must be strictly between 0 and 1.")
        self._api.require(kpi_key)
        self._api.require(cause_kpi)

        controls = [c for c in dict.fromkeys(candidates) if c not in (kpi_key, cause_kpi)]
        if not controls:
            raise InvalidArgumentError(
                "candidates", list(candidates),
                "at least one candidate confounder distinct from kpi_key and cause_kpi "
                "is required.")
        if len(controls) > MAX_CONTROLS:
            raise InvalidArgumentError(
                "candidates", len(controls),
                f"at most {MAX_CONTROLS} candidate confounders may be tested at once.")
        for c in controls:
            self._api.require(c)

        keys = [kpi_key, cause_kpi] + controls
        sample = self._correlate.paired_sample(
            df, keys, mode=mode, period_a=period_a, period_b=period_b, dimension=dimension,
            time_filter=time_filter, grain=grain, filters=filters)
        baseline_corr = self._correlate._correlation_from_sample(sample, kpi_key, cause_kpi, min_n)
        baseline_sig = self._significance.qualify(baseline_corr, alpha)
        relations = self._relation_labels(kpi_key)

        common = dict(
            kpi=kpi_key, cause_kpi=cause_kpi,
            label=self._api.label(kpi_key), cause_label=self._api.label(cause_kpi),
            mode=sample.mode, transform=sample.transform, basis=sample.basis,
            alpha=alpha, baseline=baseline_sig,
            thresholds={"explained_away_attenuation_pct": EXPLAINED_AWAY_ATTENUATION_PCT,
                        "attenuated_attenuation_pct": ATTENUATED_ATTENUATION_PCT,
                        "negligible_r": NEGLIGIBLE_R},
            dimension=sample.dimension, grain=sample.grain,
            selection=sample.selection, filters=sample.filters)

        # -- per-control first-order partials, each on its own three-key sample.
        # Scoping alignment to the three variables a given partial actually
        # reads is deliberate: dropping an entry because some *unrelated*
        # control is missing there would shrink every row to the worst-covered
        # candidate's coverage, which is the failure the time-series branch of
        # `correlate_kpi_matrix` avoids for the same reason.
        rows: List[ControlResult] = []
        for control in controls:
            triple = [kpi_key, cause_kpi, control]
            matrix, n_c, dropped_c = _correlation_matrix(sample, triple)
            label = self._api.label(control)
            if matrix is None or n_c < min_n or not _supports_controls(n_c, 1):
                rows.append(ControlResult(
                    control_kpi=control, control_label=label, controls=(control,), n_controls=1,
                    relation=relations.get(control),
                    status=("insufficient_n" if n_c < min_n
                            else "undefined_sample" if matrix is None
                            else "insufficient_n_for_controls"),
                    r_partial=None, clamped=False, attenuation_pct=None, effect=None,
                    degrees_of_freedom=None, p_value=None, ci_status="insufficient_for_ci",
                    ci_low=None, ci_high=None, significant=None))
                continue

            r_ab, r_ac, r_bc = matrix[0, 1], matrix[0, 2], matrix[1, 2]
            raw = first_order_partial(float(r_ab), float(r_ac), float(r_bc))
            r_partial, clamped = _clamp(raw)
            if r_partial is None:
                rows.append(ControlResult(
                    control_kpi=control, control_label=label, controls=(control,), n_controls=1,
                    relation=relations.get(control), status="collinear_controls",
                    r_partial=None, clamped=False, attenuation_pct=None, effect=None,
                    degrees_of_freedom=None, p_value=None, ci_status="insufficient_for_ci",
                    ci_low=None, ci_high=None, significant=None,
                    r_control_kpi=round(float(r_ac), 3), r_control_cause=round(float(r_bc), 3)))
                continue

            effect, attenuation = _classify_effect(float(r_ab), r_partial)
            p = p_value_for_r(r_partial, n_c, controls=1)
            lo, hi, ci_status = fisher_interval(r_partial, n_c, alpha, controls=1)
            rows.append(ControlResult(
                control_kpi=control, control_label=label, controls=(control,), n_controls=1,
                relation=relations.get(control), status="ok",
                r_partial=round(r_partial, 3), clamped=clamped,
                attenuation_pct=attenuation, effect=effect,
                degrees_of_freedom=n_c - 3 if n_c - 3 >= 1 else None,
                p_value=_round_p(p),
                ci_status=ci_status, ci_low=_rounded(lo), ci_high=_rounded(hi),
                significant=None if p is None else bool(p < alpha),
                r_control_kpi=round(float(r_ac), 3), r_control_cause=round(float(r_bc), 3)))

        # -- joint partial, every control held at once ------------------------
        matrix, n_all, dropped_all = _correlation_matrix(sample, keys)
        k = len(controls)
        if matrix is None or n_all < min_n or not _supports_controls(n_all, k):
            joint = ControlResult(
                control_kpi=None, control_label=None, controls=tuple(controls), n_controls=k,
                relation=None,
                status=("insufficient_n" if n_all < min_n
                        else "undefined_sample" if matrix is None
                        else "insufficient_n_for_controls"),
                r_partial=None, clamped=False, attenuation_pct=None, effect=None,
                degrees_of_freedom=None, p_value=None, ci_status="insufficient_for_ci",
                ci_low=None, ci_high=None, significant=None)
        else:
            raw = joint_partial(matrix, 0, 1)
            r_joint, clamped = _clamp(raw)
            if r_joint is None:
                joint = ControlResult(
                    control_kpi=None, control_label=None, controls=tuple(controls), n_controls=k,
                    relation=None, status="collinear_controls",
                    r_partial=None, clamped=False, attenuation_pct=None, effect=None,
                    degrees_of_freedom=None, p_value=None, ci_status="insufficient_for_ci",
                    ci_low=None, ci_high=None, significant=None)
            else:
                effect, attenuation = _classify_effect(float(matrix[0, 1]), r_joint)
                p = p_value_for_r(r_joint, n_all, controls=k)
                lo, hi, ci_status = fisher_interval(r_joint, n_all, alpha, controls=k)
                joint = ControlResult(
                    control_kpi=None, control_label=None, controls=tuple(controls), n_controls=k,
                    relation=None, status="ok",
                    r_partial=round(r_joint, 3), clamped=clamped,
                    attenuation_pct=attenuation, effect=effect,
                    degrees_of_freedom=n_all - 2 - k if n_all - 2 - k >= 1 else None,
                    p_value=_round_p(p),
                    ci_status=ci_status, ci_low=_rounded(lo), ci_high=_rounded(hi),
                    significant=None if p is None else bool(p < alpha))

        status = "ok" if baseline_corr.status == "ok" else baseline_corr.status
        return ConfounderTest(
            status=status, r_baseline=baseline_corr.r, n=baseline_corr.n,
            dropped_for_alignment=dropped_all, rows=tuple(rows), joint=joint, **common)

    # -- test_spurious_correlation() -------------------------------------------
    def test_spurious_correlation(self, df: pd.DataFrame, kpi_key: str, cause_kpi: str,
                                  grain: str = "quarter",
                                  time_filter: Optional[Union[Mapping[str, Any], TimeFilter]] = None,
                                  filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
                                  mode: str = "time_series",
                                  min_n: int = DEFAULT_MIN_N,
                                  alpha: float = DEFAULT_ALPHA) -> SpuriousTest:
        """
        Does the association survive removing each series' own trend?

        Time-series only. A cross-sectional sample of members has no time axis
        and therefore no trend to remove, so `mode="cross_sectional"` is
        refused rather than answered with a number that would mean nothing.
        """
        if mode != "time_series":
            raise InvalidArgumentError(
                "mode", mode,
                "test_spurious_correlation is time-series only -- a cross-sectional sample "
                "of members has no time axis and therefore no trend to remove.",
                ["time_series"])
        if grain not in SERIES_GRAINS:
            raise InvalidArgumentError("grain", grain, "must be one of the supported grains.",
                                       list(SERIES_GRAINS))
        if not 0.0 < alpha < 1.0:
            raise InvalidArgumentError("alpha", alpha, "must be strictly between 0 and 1.")
        self._api.require(kpi_key)
        self._api.require(cause_kpi)

        tf = time_filter if time_filter is not None else {"type": "all"}
        series_map = self._series.series_multi(df, [kpi_key, cause_kpi], grain=grain,
                                               time_filter=tf, filters=filters)
        kpi_series, cause_series = series_map[kpi_key], series_map[cause_kpi]
        by_kpi = {p.period: p.value for p in kpi_series.points}
        by_cause = {p.period: p.value for p in cause_series.points}
        periods = sorted(set(by_kpi) & set(by_cause))
        gaps = tuple(sorted(set(kpi_series.gaps) | set(cause_series.gaps)))

        common = dict(
            kpi=kpi_key, cause_kpi=cause_kpi,
            label=self._api.label(kpi_key), cause_label=self._api.label(cause_kpi),
            grain=grain, alpha=alpha, method="theil_sen", gaps=gaps,
            selection=kpi_series.selection, filters=kpi_series.filters)

        def _empty(transform: str) -> TransformCorrelation:
            return TransformCorrelation(transform=transform, status="no_overlap", r=None, n=0,
                                        p_value=None, ci_status="insufficient_for_ci",
                                        ci_low=None, ci_high=None, significant=None)

        if len(periods) < min_n:
            return SpuriousTest(
                status="insufficient_history" if periods else "no_overlap",
                verdict="insufficient",
                level=_empty("level"), detrended=_empty("detrended"),
                difference=_empty("difference"),
                kpi_slope_per_period=None, cause_slope_per_period=None,
                shared_trend_direction=None, attenuation_pct=None, **common)

        xs = np.array([period_ordinal(grain, p) for p in periods], dtype=float)
        kpi_levels = np.array([by_kpi[p] for p in periods], dtype=float)
        cause_levels = np.array([by_cause[p] for p in periods], dtype=float)

        def _qualified(transform: str, a: Sequence[float], b: Sequence[float]
                       ) -> TransformCorrelation:
            status, r, _r2, n = pearson_correlate(a, b, min_n)
            p = p_value_for_r(r, n)
            lo, hi, ci_status = fisher_interval(r, n, alpha)
            return TransformCorrelation(
                transform=transform, status=status, r=r, n=n,
                p_value=_round_p(p), ci_status=ci_status,
                ci_low=_rounded(lo), ci_high=_rounded(hi),
                significant=None if p is None else bool(p < alpha))

        level = _qualified("level", kpi_levels, cause_levels)

        kpi_slope, kpi_intercept = theil_sen(xs, kpi_levels)
        cause_slope, cause_intercept = theil_sen(xs, cause_levels)
        detrended = _qualified("detrended",
                               kpi_levels - (kpi_slope * xs + kpi_intercept),
                               cause_levels - (cause_slope * xs + cause_intercept))

        # Only calendar-adjacent periods are differenced -- the rule
        # `correlate._time_series_sample` and `temporal.cross_correlate_lagged`
        # both follow, since a difference across a hole is not a
        # period-over-period change.
        d_kpi: List[float] = []
        d_cause: List[float] = []
        for i in range(1, len(periods)):
            if xs[i] - xs[i - 1] == 1:
                d_kpi.append(kpi_levels[i] - kpi_levels[i - 1])
                d_cause.append(cause_levels[i] - cause_levels[i - 1])
        difference = _qualified("difference", d_kpi, d_cause)

        # The finding is the gap between levels and residuals: a large `r` on
        # levels that vanishes once each series' own drift is removed is two
        # things growing side by side, not one moving the other.
        verdict: Optional[str]
        attenuation: Optional[float]
        if level.r is None or detrended.r is None:
            verdict, attenuation = "insufficient", None
        else:
            attenuation = (round((abs(level.r) - abs(detrended.r)) / abs(level.r) * 100.0, 2)
                           if abs(level.r) >= NEGLIGIBLE_R else None)
            verdict = ("survives_detrending" if detrended.significant
                       else "trend_driven" if level.significant
                       else "insufficient")

        shared_direction = (None if kpi_slope == 0.0 or cause_slope == 0.0
                            else bool(np.sign(kpi_slope) == np.sign(cause_slope)))
        return SpuriousTest(
            status="ok", verdict=verdict, level=level, detrended=detrended,
            difference=difference,
            kpi_slope_per_period=_rounded(kpi_slope, 6),
            cause_slope_per_period=_rounded(cause_slope, 6),
            shared_trend_direction=shared_direction, attenuation_pct=attenuation, **common)

    # -- rank_competing_explanations() -----------------------------------------
    def rank_competing_explanations(
            self, df: pd.DataFrame, kpi_key: str, candidate_causes: Sequence[str],
            mode: str = "cross_sectional",
            period_a: Optional[Union[Mapping[str, Any], TimeFilter]] = None,
            period_b: Optional[Union[Mapping[str, Any], TimeFilter]] = None,
            dimension: Optional[str] = None,
            time_filter: Optional[Union[Mapping[str, Any], TimeFilter]] = None,
            grain: str = "quarter",
            precedence_grain: str = "week",
            filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
            min_n: int = DEFAULT_MIN_N,
            alpha: float = DEFAULT_ALPHA,
            ordered_by: str = "abs_partial_r",
            include_precedence: bool = True,
            include_consistency: bool = True) -> CompetingExplanations:
        """
        Several candidate causes, laid side by side on one identical evidence
        basis: correlation with p and q (X3), each candidate's partial
        correlation controlling for **every rival** (X4), temporal precedence
        (X1) and cross-sectional agreement (X2).

        **No composite score is produced.** Decision 45 records that the agent
        tools deliberately emit no confidence number; the model reads the row
        and judges. `ordered_by` names the single sort key applied and travels
        in the payload, so the ordering is reproducible and never a hidden
        weighting.

        Mutual control is what makes this different from N separate calls: "is
        it pricing, cost, or mix" asks which candidate survives the others, and
        that question does not exist for a candidate examined alone.
        """
        if mode not in MODES:
            raise InvalidArgumentError("mode", mode, "must be one of the supported modes.",
                                       list(MODES))
        if ordered_by not in COMPETING_SORT_KEYS:
            raise InvalidArgumentError("ordered_by", ordered_by,
                                       "must be one of the supported sort keys.",
                                       list(COMPETING_SORT_KEYS))
        if not 0.0 < alpha < 1.0:
            raise InvalidArgumentError("alpha", alpha, "must be strictly between 0 and 1.")
        self._api.require(kpi_key)

        pool = [c for c in dict.fromkeys(candidate_causes) if c != kpi_key]
        if len(pool) < 2:
            raise InvalidArgumentError(
                "candidate_causes", list(candidate_causes),
                "at least two distinct candidate causes are required -- ranking one "
                "explanation against nothing is test_statistical_significance.")
        for c in pool:
            self._api.require(c)
        considered = len(pool)
        truncated = considered > MAX_COMPETING_CANDIDATES
        pool = pool[:MAX_COMPETING_CANDIDATES]

        # -- one shared sample, one shared significance family -----------------
        sample = self._correlate.paired_sample(
            df, [kpi_key] + pool, mode=mode, period_a=period_a, period_b=period_b,
            dimension=dimension, time_filter=time_filter, grain=grain, filters=filters)
        sig_set = self._significance.test_statistical_significance(
            df, kpi_key, candidates=pool, mode=mode, period_a=period_a, period_b=period_b,
            dimension=dimension, time_filter=time_filter, grain=grain, filters=filters,
            min_n=min_n, alpha=alpha)
        sig_by_key = {row.cause_kpi: row for row in sig_set.rows}
        relations = self._relation_labels(kpi_key)

        matrix, n_all, dropped_all = _correlation_matrix(sample, [kpi_key] + pool)
        temporal = TemporalEngine(self._api) if include_precedence else None
        consistency = ConsistencyEngine(self._api) if include_consistency else None

        rows: List[CompetingCandidate] = []
        for position, cand in enumerate(pool, start=1):
            rivals = tuple(c for c in pool if c != cand)

            # -- partial vs every rival ---------------------------------------
            if matrix is None or n_all < min_n or not _supports_controls(n_all, len(rivals)):
                r_partial = None
                partial_status = ("insufficient_n" if n_all < min_n
                                  else "undefined_sample" if matrix is None
                                  else "insufficient_n_for_controls")
            else:
                keep = [0, position] + [pool.index(r) + 1 for r in rivals]
                sub = matrix[np.ix_(keep, keep)]
                r_partial, clamped = _clamp(joint_partial(sub, 0, 1))
                partial_status = "ok" if r_partial is not None else "collinear_controls"
            sig = sig_by_key.get(cand)
            effect, attenuation = _classify_effect(None if sig is None else sig.r, r_partial)

            # -- X1: did it move first ----------------------------------------
            lag, prec_status, prec_verdict = None, "not_requested", None
            if temporal is not None:
                try:
                    prec = temporal.check_temporal_precedence(
                        df, kpi_key, cand, grain=precedence_grain,
                        time_filter=time_filter, filters=filters)
                    lag, prec_status, prec_verdict = (prec.lag_periods, prec.status, prec.verdict)
                except Exception:
                    # A sub-check that cannot run degrades this *cell*; it never
                    # drops the candidate's row or fails the whole comparison.
                    prec_status = "unavailable"

            # -- X2: does it hold across members ------------------------------
            rate, cons_status, cons_direction = None, "not_requested", None
            if consistency is not None and mode == "cross_sectional" \
                    and period_a is not None and period_b is not None:
                try:
                    dim = sample.dimension or dimension
                    cons = consistency.test_consistency_across_dimension(
                        df, kpi_key, cand, dim, period_a, period_b, filters=filters)
                    rate, cons_status, cons_direction = (cons.consistency_rate, cons.status,
                                                         cons.direction)
                except Exception:
                    cons_status = "unavailable"
            elif consistency is not None:
                cons_status = "not_applicable"

            rows.append(CompetingCandidate(
                cause_kpi=cand, cause_label=self._api.label(cand),
                relation=relations.get(cand), significance=sig,
                r_partial_vs_rivals=None if r_partial is None else round(r_partial, 3),
                partial_status=partial_status, partial_effect=effect,
                attenuation_pct=attenuation, rivals_controlled=rivals,
                lag_periods=lag, precedence_status=prec_status, precedence_verdict=prec_verdict,
                consistency_rate=None if rate is None else round(rate, 3),
                consistency_status=cons_status, consistency_direction=cons_direction))

        def _key(row: CompetingCandidate):
            """A missing value always sorts last, and `cause_kpi` breaks every
            tie, so the order is total and reproducible (G5)."""
            if ordered_by == "p_value":
                p = None if row.significance is None else row.significance.p_value
                return (p is None, p if p is not None else 1.0, row.cause_kpi)
            value = (row.r_partial_vs_rivals if ordered_by == "abs_partial_r"
                     else (None if row.significance is None else row.significance.r))
            return (value is None, -abs(value) if value is not None else 0.0, row.cause_kpi)

        return CompetingExplanations(
            kpi=kpi_key, label=self._api.label(kpi_key), mode=mode, alpha=alpha,
            ordered_by=ordered_by, family_size=sig_set.family_size,
            considered=considered, truncated=truncated,
            n=n_all, dropped_for_alignment=dropped_all,
            rows=tuple(sorted(rows, key=_key)),
            dimension=sample.dimension, grain=sample.grain,
            selection=sample.selection, filters=sample.filters)
