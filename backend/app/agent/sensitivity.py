"""
`test_sensitivity_to_outliers` / `estimate_effect_size` -- is a cross-sectional
finding the artefact of one member, and how wide is the effect's real interval.

X3 (`agent/significance.py`) asks whether a correlation could be chance and X4
(`agent/confounders.py`) whether a third variable explains it. Neither asks the
question a business reviewer asks first: **is this whole finding just one big
account?** On a dataset where `region` has four members
(`agent/significance.py:38-42`), a single member's move can carry a
cross-sectional `r` from near zero to near one, and nothing in the shipped
tools would show it.

**Leave-one-out, not a rule of thumb.** `test_sensitivity_to_outliers` refits
the correlation with each member removed in turn and reports the member whose
removal moves `|r|` the most -- the single most influential point -- alongside
the finding with that point gone. The verdict is whether dropping it alone
overturns the association: a sign flip, a collapse past
`OUTLIER_DRIVEN_ATTENUATION_PCT` of the baseline `|r|`, or a loss of
significance.

**The influential point and the disproportionate contributor are different
questions, and both travel in the payload.**
`ConcentrationEngine.find_outlier_contributors` (`agent/concentration.py:339`)
ranks members by a robust z-score on their over-index -- how much a member moved
relative to its size. That is not the same as leverage on the cross-sectional
correlation: a member can dominate the *change* without being the point the
*line* pivots on. `flagged_outlier_member` carries the concentration tool's top
material pick and `flagged_matches_influential` says whether the two agree, so
the model can tell "one member drove it and it was the obvious one" from "one
member drove it and it was a quiet one".

**`estimate_effect_size` gives the interval, distribution-free.** X3's
`fisher_interval` is a parametric CI that assumes bivariate normality; on the
member counts these fixtures hold that assumption is thin. `scipy.stats.bootstrap`
resamples the aligned pairs and reports a BCa interval beside the Fisher one, so
the model sees both and can notice when they disagree. `r_squared` -- the share
of variance the two move together -- travels too, because "significant" and
"large" are different claims and only the second is an effect size.
`magnitude_thresholds` publishes the cut points rather than a word, the
`agent/confounders.py` discipline.

**The bootstrap is seeded.** `scipy.stats.bootstrap` draws pseudo-random
resamples; an unseeded call returns a different interval every run and would
fail G5 (identical input -> identical output). `BOOTSTRAP_SEED` is fixed at 0,
a deliberate choice of reproducibility over per-call entropy -- the interval is
a property of the sample, and two readers computing it should get the same
number.

Nothing here recomputes a correlation from row data. Every `r` comes from
`CorrelationEngine` over one `PairedSample` -- the composition rule X3 and X4
follow -- so a finding's `r` cannot mean one thing to `correlate_kpis` and
another to its own sensitivity check. The leave-one-out refits are the one
piece of arithmetic this module owns, and they are `numpy.corrcoef` over a
subset of the vectors that `PairedSample` already published.

Provisional by design: the master plan (task X5) marks the robustness family
for a build-or-cut decision against A8's tool-usage report. `test_sensitivity_to_period`
and `test_subsample_stability` are deferred pending that evidence; these two
are the ones the plan marks unconditional.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import DegenerateDataWarning

from ..engines.metrics import safe
from .concentration import ConcentrationEngine
from .contract_api import ContractAPI
from .correlate import DEFAULT_MIN_N, MODES, CorrelationEngine, PairedSample
from .errors import InvalidArgumentError
from .significance import (DEFAULT_ALPHA, _round_p, _rounded, fisher_interval,
                           p_value_for_r)
from .timefilter import TimeFilter, TimeSelection

# Below this |r| there is no association to be sensitive about, and a sign on a
# value this small is noise. Local, like `confounders.NEGLIGIBLE_R` -- a
# threshold, not shared maths.
NEGLIGIBLE_R = 0.05

# Percent of the baseline |r| that must vanish when the single most influential
# member is dropped for the finding to read as carried by that member.
# Published in the payload, not hidden -- the `confounders.py` discipline.
OUTLIER_DRIVEN_ATTENUATION_PCT = 50.0

# A leave-one-out refit needs at least three members left to correlate, so the
# population must hold at least four before the question means anything.
MIN_MEMBERS_FOR_DROP_ONE = 4

# Fisher's z (`fisher_interval`) needs n >= 4; a BCa bootstrap of a correlation
# needs enough distinct resamples that the jackknife acceleration is defined.
# Eight is the floor below which the interval is not worth reporting -- and is
# where `agent/significance.py`'s `_adequacy_verdict` also stops calling a
# sample "marginal".
MIN_N_FOR_BOOTSTRAP = 8

# `scipy.stats.bootstrap` resample count and seed. The seed is load-bearing for
# G5 (identical input -> byte-identical output) -- see the module docstring.
BOOTSTRAP_RESAMPLES = 4000
BOOTSTRAP_SEED = 0

OUTLIER_SENSITIVITY_VERDICTS = ("robust", "outlier_driven", "no_association", "insufficient")
BOOTSTRAP_STATUSES = ("ok", "insufficient_for_bootstrap", "degenerate")


# ---------------------------------------------------------------------------
# the maths this module owns
# ---------------------------------------------------------------------------
def _pearson(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """
    Pearson `r`, or `None` when either side has no variance.

    A vertical or horizontal scatter has no correlation -- not a NaN to
    propagate into a verdict. The `< 3` guard is the same floor
    `pearson_correlate` enforces; below it `numpy.corrcoef` is drawing a line
    through two points.
    """
    if len(x) < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    r = float(np.corrcoef(x, y)[0, 1])
    return r if np.isfinite(r) else None


def _bootstrap_r_ci(a: np.ndarray, b: np.ndarray, alpha: float
                    ) -> Tuple[Optional[float], Optional[float], str]:
    """
    `(low, high, status)` -- a BCa bootstrap confidence interval for Pearson
    `r` over the paired sample `(a, b)`, seeded so the interval is reproducible.

    `status` is `"ok"` or `"degenerate"`. BCa raises outright when the
    jackknife acceleration is undefined (every resample identical, or the
    statistic pinned at an extreme), and a resample that happens to be constant
    yields a `nan` the interval must not carry through -- both collapse to
    `"degenerate"` rather than a fabricated bound.
    """
    def _stat(x: np.ndarray, y: np.ndarray) -> float:
        if np.std(x) == 0.0 or np.std(y) == 0.0:
            return np.nan
        return float(np.corrcoef(x, y)[0, 1])

    try:
        with warnings.catch_warnings():
            # A degenerate sample is handled below on `isfinite` -- scipy's
            # warning about it is not news, and must not reach a caller.
            warnings.simplefilter("ignore", DegenerateDataWarning)
            result = stats.bootstrap(
                (a, b), _stat, paired=True, vectorized=False,
                n_resamples=BOOTSTRAP_RESAMPLES, method="BCa",
                confidence_level=1.0 - alpha,
                rng=np.random.default_rng(BOOTSTRAP_SEED))
        low = float(result.confidence_interval.low)
        high = float(result.confidence_interval.high)
    except (ValueError, RuntimeError):
        return None, None, "degenerate"
    if not (np.isfinite(low) and np.isfinite(high)):
        return None, None, "degenerate"
    return max(-1.0, low), min(1.0, high), "ok"


# ---------------------------------------------------------------------------
# test_sensitivity_to_outliers()
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OutlierSensitivity:
    kpi: str
    cause_kpi: str
    label: str
    cause_label: str
    dimension: Optional[str]
    transform: str
    basis: str
    # "ok" | a propagated Correlation status | "insufficient_population" | "constant_series"
    status: str
    verdict: str                       # one of OUTLIER_SENSITIVITY_VERDICTS
    r_full: Optional[float]
    n_full: int
    alpha: float
    p_value_full: Optional[float]
    significant_full: Optional[bool]
    most_influential_member: Optional[str]
    r_without_most_influential: Optional[float]
    n_without: int
    delta_r: Optional[float]
    attenuation_pct: Optional[float]
    p_value_without: Optional[float]
    significant_without: Optional[bool]
    significance_lost: Optional[bool]
    sign_flipped: Optional[bool]
    flagged_outlier_member: Optional[str]
    flagged_outlier_status: str        # find_outlier_contributors' own status, or "not_run"/"unavailable"
    flagged_matches_influential: Optional[bool]
    r_without_flagged: Optional[float]
    thresholds: Mapping[str, float]
    selection: Optional[TimeSelection] = None
    baseline_selection: Optional[TimeSelection] = None
    filters: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        payload = {
            "kpi": self.kpi, "cause_kpi": self.cause_kpi,
            "label": self.label, "cause_label": self.cause_label,
            "dimension": self.dimension, "transform": self.transform, "basis": self.basis,
            "status": self.status, "verdict": self.verdict,
            "r_full": safe(self.r_full), "n_full": self.n_full, "alpha": self.alpha,
            "p_value_full": safe(self.p_value_full), "significant_full": self.significant_full,
            "most_influential_member": self.most_influential_member,
            "r_without_most_influential": safe(self.r_without_most_influential),
            "n_without": self.n_without, "delta_r": safe(self.delta_r),
            "attenuation_pct": safe(self.attenuation_pct),
            "p_value_without": safe(self.p_value_without),
            "significant_without": self.significant_without,
            "significance_lost": self.significance_lost,
            "sign_flipped": self.sign_flipped,
            "flagged_outlier_member": self.flagged_outlier_member,
            "flagged_outlier_status": self.flagged_outlier_status,
            "flagged_matches_influential": self.flagged_matches_influential,
            "r_without_flagged": safe(self.r_without_flagged),
            "thresholds": dict(self.thresholds),
            "filters": {k: list(v) for k, v in self.filters.items()},
        }
        if self.selection is not None:
            payload["selection"] = self.selection.to_payload()
        if self.baseline_selection is not None:
            payload["baseline_selection"] = self.baseline_selection.to_payload()
        return payload


# ---------------------------------------------------------------------------
# estimate_effect_size()
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EffectSize:
    kpi: str
    cause_kpi: str
    label: str
    cause_label: str
    mode: str
    transform: str
    basis: str
    status: str                        # the Correlation's own status, propagated
    r: Optional[float]
    r_squared: Optional[float]
    n: int
    dropped_for_alignment: int
    alpha: float
    p_value: Optional[float]
    fisher_ci_status: str              # "ok" | "insufficient_for_ci"
    fisher_ci_low: Optional[float]
    fisher_ci_high: Optional[float]
    bootstrap_status: str              # one of BOOTSTRAP_STATUSES, or a propagated status
    bootstrap_method: str              # "BCa"
    bootstrap_resamples: int
    ci_low: Optional[float]
    ci_high: Optional[float]
    ci_excludes_zero: Optional[bool]
    magnitude_thresholds: Mapping[str, float]
    dimension: Optional[str] = None
    grain: Optional[str] = None
    selection: Optional[TimeSelection] = None
    baseline_selection: Optional[TimeSelection] = None
    filters: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        payload = {
            "kpi": self.kpi, "cause_kpi": self.cause_kpi,
            "label": self.label, "cause_label": self.cause_label,
            "mode": self.mode, "transform": self.transform, "basis": self.basis,
            "status": self.status, "r": safe(self.r), "r_squared": safe(self.r_squared),
            "n": self.n, "dropped_for_alignment": self.dropped_for_alignment,
            "alpha": self.alpha, "p_value": safe(self.p_value),
            "fisher_ci_status": self.fisher_ci_status,
            "fisher_ci_low": safe(self.fisher_ci_low),
            "fisher_ci_high": safe(self.fisher_ci_high),
            "bootstrap_status": self.bootstrap_status,
            "bootstrap_method": self.bootstrap_method,
            "bootstrap_resamples": self.bootstrap_resamples,
            "ci_low": safe(self.ci_low), "ci_high": safe(self.ci_high),
            "ci_excludes_zero": self.ci_excludes_zero,
            "magnitude_thresholds": dict(self.magnitude_thresholds),
            "dimension": self.dimension, "grain": self.grain,
            "filters": {k: list(v) for k, v in self.filters.items()},
        }
        if self.selection is not None:
            payload["selection"] = self.selection.to_payload()
        if self.baseline_selection is not None:
            payload["baseline_selection"] = self.baseline_selection.to_payload()
        return payload


class SensitivityEngine:
    """`test_sensitivity_to_outliers` / `estimate_effect_size`, bound to one
    dataset's `ContractAPI`."""

    def __init__(self, api: ContractAPI):
        self._api = api
        self._correlate = CorrelationEngine(api)
        self._concentration = ConcentrationEngine(api)

    # -- test_sensitivity_to_outliers() --------------------------------------
    def test_sensitivity_to_outliers(
            self, df: pd.DataFrame, kpi_key: str, cause_kpi: str,
            period_a: Union[Mapping[str, Any], TimeFilter],
            period_b: Union[Mapping[str, Any], TimeFilter],
            dimension: Optional[str] = None,
            filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
            min_n: int = DEFAULT_MIN_N,
            alpha: float = DEFAULT_ALPHA) -> OutlierSensitivity:
        """
        Does the cross-sectional `kpi ~ cause` association survive dropping the
        single member that most influences it?

        Cross-sectional only: the question is about members, and a time-series
        sample has none. The refit is leave-one-out over the members
        `PairedSample` already aligned -- the most influential is the one whose
        removal moves `|r|` the furthest, not the one that moved the most
        (`find_outlier_contributors` answers that, and its pick travels
        alongside for the comparison).
        """
        if not 0.0 < alpha < 1.0:
            raise InvalidArgumentError("alpha", alpha, "must be strictly between 0 and 1.")
        if min_n < 3:
            raise InvalidArgumentError("min_n", min_n, "must be >= 3.")
        self._api.require(kpi_key)
        self._api.require(cause_kpi)

        sample = self._correlate.paired_sample(
            df, [kpi_key, cause_kpi], mode="cross_sectional",
            period_a=period_a, period_b=period_b, dimension=dimension, filters=filters)
        corr = self._correlate._correlation_from_sample(sample, kpi_key, cause_kpi, min_n)

        index, arrays, _dropped = sample.listwise([kpi_key, cause_kpi])
        a, b = arrays[kpi_key], arrays[cause_kpi]
        n_full = len(a)

        common = dict(
            kpi=kpi_key, cause_kpi=cause_kpi,
            label=self._api.label(kpi_key), cause_label=self._api.label(cause_kpi),
            dimension=sample.dimension, transform=sample.transform, basis=sample.basis,
            alpha=alpha,
            thresholds={"outlier_driven_attenuation_pct": OUTLIER_DRIVEN_ATTENUATION_PCT,
                        "negligible_r": NEGLIGIBLE_R,
                        "min_members_for_drop_one": float(MIN_MEMBERS_FOR_DROP_ONE)},
            selection=sample.selection, baseline_selection=sample.baseline_selection,
            filters=sample.filters)

        def _blank(status: str) -> OutlierSensitivity:
            return OutlierSensitivity(
                status=status, verdict="insufficient",
                r_full=corr.r, n_full=n_full, p_value_full=None, significant_full=None,
                most_influential_member=None, r_without_most_influential=None,
                n_without=0, delta_r=None, attenuation_pct=None,
                p_value_without=None, significant_without=None, significance_lost=None,
                sign_flipped=None, flagged_outlier_member=None,
                flagged_outlier_status="not_run", flagged_matches_influential=None,
                r_without_flagged=None, **common)

        if corr.status != "ok":
            return _blank(corr.status)
        if n_full < MIN_MEMBERS_FOR_DROP_ONE:
            return _blank("insufficient_population")

        r_full = _pearson(a, b)
        if r_full is None:
            return _blank("constant_series")

        # -- leave-one-out: whose removal moves |r| the most ------------------
        candidates: List[Tuple[float, str, float]] = []
        for i, member in enumerate(index):
            keep = np.arange(n_full) != i
            r_i = _pearson(a[keep], b[keep])
            if r_i is not None:
                candidates.append((abs(abs(r_full) - abs(r_i)), str(member), r_i))
        if not candidates:
            return _blank("constant_series")
        candidates.sort(key=lambda t: (-t[0], t[1]))
        _, infl_member, r_without = candidates[0]

        p_full = p_value_for_r(round(r_full, 3), n_full)
        p_without = p_value_for_r(round(r_without, 3), n_full - 1)
        sig_full = None if p_full is None else bool(p_full < alpha)
        sig_without = None if p_without is None else bool(p_without < alpha)
        sig_lost = (bool(sig_full and not sig_without)
                    if sig_full is not None and sig_without is not None else None)

        atten = (round((abs(r_full) - abs(r_without)) / abs(r_full) * 100.0, 2)
                 if abs(r_full) >= NEGLIGIBLE_R else None)
        flipped = bool(abs(r_without) >= NEGLIGIBLE_R
                       and np.sign(r_without) != np.sign(r_full))

        if abs(r_full) < NEGLIGIBLE_R:
            verdict = "no_association"
        elif flipped or (atten is not None and atten >= OUTLIER_DRIVEN_ATTENUATION_PCT) \
                or bool(sig_lost):
            verdict = "outlier_driven"
        else:
            verdict = "robust"

        # -- cross-check: is the leverage point also a disproportionate mover?
        flagged_member: Optional[str] = None
        flagged_status = "not_run"
        try:
            outliers = self._concentration.find_outlier_contributors(
                df, kpi_key, sample.dimension, period_a, period_b, filters=filters)
            flagged_status = outliers.status
            material = [row for row in outliers.rows if row.material]
            if material:
                flagged_member = str(material[0].member)
        except Exception:                                  # pragma: no cover - defensive
            flagged_status = "unavailable"

        r_without_flagged: Optional[float] = None
        members = [str(m) for m in index]
        if flagged_member is not None and flagged_member in members:
            keep = np.array([m != flagged_member for m in members])
            r_without_flagged = _pearson(a[keep], b[keep])
        matches = None if flagged_member is None else bool(flagged_member == infl_member)

        return OutlierSensitivity(
            status="ok", verdict=verdict,
            r_full=round(r_full, 3), n_full=n_full,
            p_value_full=_round_p(p_full), significant_full=sig_full,
            most_influential_member=infl_member,
            r_without_most_influential=round(r_without, 3), n_without=n_full - 1,
            delta_r=round(r_full - r_without, 3), attenuation_pct=atten,
            p_value_without=_round_p(p_without), significant_without=sig_without,
            significance_lost=sig_lost, sign_flipped=flipped,
            flagged_outlier_member=flagged_member, flagged_outlier_status=flagged_status,
            flagged_matches_influential=matches,
            r_without_flagged=None if r_without_flagged is None else round(r_without_flagged, 3),
            **common)

    # -- estimate_effect_size() --------------------------------------------------
    def estimate_effect_size(
            self, df: pd.DataFrame, kpi_key: str, cause_kpi: str,
            mode: str = "cross_sectional",
            period_a: Optional[Union[Mapping[str, Any], TimeFilter]] = None,
            period_b: Optional[Union[Mapping[str, Any], TimeFilter]] = None,
            dimension: Optional[str] = None,
            time_filter: Optional[Union[Mapping[str, Any], TimeFilter]] = None,
            grain: str = "quarter",
            filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
            min_n: int = DEFAULT_MIN_N,
            alpha: float = DEFAULT_ALPHA) -> EffectSize:
        """
        The magnitude of the `kpi ~ cause` association and a distribution-free
        interval for it: `r`, its shared-variance `r_squared`, a BCa bootstrap
        CI, and the parametric Fisher CI beside it for comparison.

        The bootstrap is seeded (`BOOTSTRAP_SEED`) so the interval is
        reproducible -- G5. Below `MIN_N_FOR_BOOTSTRAP` aligned points the
        bootstrap is refused (`bootstrap_status="insufficient_for_bootstrap"`)
        rather than run on a sample too small for the jackknife acceleration to
        mean anything; `r` is still reported.
        """
        if mode not in MODES:
            raise InvalidArgumentError("mode", mode, "must be one of the supported modes.",
                                       list(MODES))
        if not 0.0 < alpha < 1.0:
            raise InvalidArgumentError("alpha", alpha, "must be strictly between 0 and 1.")
        if min_n < 3:
            raise InvalidArgumentError("min_n", min_n, "must be >= 3.")
        self._api.require(kpi_key)
        self._api.require(cause_kpi)

        sample = self._correlate.paired_sample(
            df, [kpi_key, cause_kpi], mode=mode, period_a=period_a, period_b=period_b,
            dimension=dimension, time_filter=time_filter, grain=grain, filters=filters)
        corr = self._correlate._correlation_from_sample(sample, kpi_key, cause_kpi, min_n)

        _index, arrays, dropped = sample.listwise([kpi_key, cause_kpi])
        a, b = arrays[kpi_key], arrays[cause_kpi]
        n = min(len(a), corr.n)

        lo, hi, fisher_status = fisher_interval(corr.r, corr.n, alpha)
        p = p_value_for_r(corr.r, corr.n)

        ci_low: Optional[float] = None
        ci_high: Optional[float] = None
        if corr.status != "ok":
            boot_status = corr.status
        elif n < MIN_N_FOR_BOOTSTRAP:
            boot_status = "insufficient_for_bootstrap"
        else:
            ci_low, ci_high, boot_status = _bootstrap_r_ci(a, b, alpha)

        excludes = (None if ci_low is None or ci_high is None
                    else bool(ci_low > 0.0 or ci_high < 0.0))

        return EffectSize(
            kpi=kpi_key, cause_kpi=cause_kpi,
            label=self._api.label(kpi_key), cause_label=self._api.label(cause_kpi),
            mode=sample.mode, transform=sample.transform, basis=sample.basis,
            status=corr.status, r=corr.r, r_squared=corr.r_squared, n=corr.n,
            dropped_for_alignment=dropped, alpha=alpha, p_value=_round_p(p),
            fisher_ci_status=fisher_status,
            fisher_ci_low=_rounded(lo), fisher_ci_high=_rounded(hi),
            bootstrap_status=boot_status, bootstrap_method="BCa",
            bootstrap_resamples=BOOTSTRAP_RESAMPLES if boot_status == "ok" else 0,
            ci_low=_rounded(ci_low), ci_high=_rounded(ci_high),
            ci_excludes_zero=excludes,
            magnitude_thresholds={"small": 0.1, "medium": 0.3, "large": 0.5},
            dimension=sample.dimension, grain=sample.grain,
            selection=sample.selection, baseline_selection=sample.baseline_selection,
            filters=sample.filters)
