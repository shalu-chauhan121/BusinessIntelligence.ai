"""
`measure_concentration` / `find_outlier_contributors` -- how concentrated a
KPI is across a dimension's members, and which members are cross-sectional
outliers in one A/B change.

Today's only notion of concentration is a single `max(|contribution_pct|)`
over one arbitrarily-chosen dimension (`engines/observe.py:437-442`), picked
by `next(iter(drivers))` -- a dict-ordering read, not a deliberate choice, and
a maximum is not an index. "80% of revenue comes from 3 accounts" -- exactly
what a reviewer notices first (the Tier-5 trace in the plan) -- has no code
path today.

`measure_concentration` computes no arithmetic of its own over row data: it
is built entirely on `BreakdownEngine.rank_entities` (`agent/breakdown.py:242`)
called with `order="desc", limit=None`, which already returns every member
sorted descending with `share_of_total_pct` / `cumulative_share_pct` computed
against a grand total obtained from its *own* ungrouped query
(`breakdown.py:261`) rather than by summing cells. The only new code here is
HHI / normalised HHI / effective-members / a small-sample-corrected Gini /
top-k share, layered on those already-correct shares.

`order="desc"` is passed literally, not `"best"`/`"worst"` -- `_resolve_order`
(`breakdown.py:53`) flips direction for a lower-is-better KPI, and a
concentration ranking wants largest-first regardless of polarity.

HHI here is on `[0, 1]`, not the US DOJ `0..10000` scale -- do not compare it
against a 1800-style threshold. `hhi_normalized` matters, not decoration: raw
HHI on an n-member dimension is >= 1/n *by construction*, so a 3-member
dimension reads as "concentrated" under raw HHI even when perfectly even --
exactly the school fixture's `campus` dimension. `hhi_normalized` divides
that construction effect out.

The gate for whether a share means anything is `ContractAPI.is_additive`, not
`kind() == "sum"`: that is the literal condition `rank_entities` itself uses
to decide whether `share_of_total_pct` is a real number or `None`
(`breakdown.py:264`), and this module's `top_k_shares` reads that same field,
so gating on anything else could disagree with the numbers actually returned.

`find_outlier_contributors` answers a distinct question from three shipped
tools that all touch "which member matters":
`DriverEngine.compute_over_index` (`agent/drivers.py:355`) reports every
member's over-index against a fixed `DISPROPORTIONATE_AT = 1.2`;
`ScanEngine.scan_dimension_outliers` (`agent/scan.py:384`) scores each member
against **its own history**; `DriverEngine.rank_drivers` is a four-part
composite. This tool is cross-sectional and population-relative: whose
over-index is an outlier *within this dimension's members* for one named A/B
change, scored with `robust_sigma` (`engines/drivers.py:175`) -- the same
median+MAD estimator `assess_significance`, `normal_range` and
`scan_dimension_outliers` already use. Three tools now answer "which member
matters"; that is a tool-*selection* risk for A2's registry, flagged here for
A8's tool-usage report to settle on evidence, alongside X5.

Built on `DecomposeEngine.decompose_by_dimension` (`agent/decompose.py:355`)
with an explicit `max_items`, exactly as `compute_over_index` already does
(`agent/drivers.py:360-361`) -- reusing the code path that carries I1's core
invariant (member contributions sum to the total change) rather than
re-deriving `contribution_pct` / `over_index` from `engines/observe.py:292-365`
a third time. `observe.decompose_dimension` costs one boolean mask + a
`compute()` call per member (`observe.py:305-315`), the same N-scans pattern
`scan.py` and `agent/drivers.py` document and batch around; there is no
batched replacement for that specific arithmetic yet, so the scored
population is capped at `MAX_OUTLIER_MEMBERS`. That cap is not a compromise:
`decompose_dimension` sorts by `|change_abs|` before folding the tail
(`observe.py:366`), so the scored population is always the members that moved
the most in absolute terms -- precisely the members a finding could plausibly
be about.

The materiality gate is the point of the tool, not an afterthought: a member
holding 0.1% of baseline that moves 400% has a wild over-index and a
near-zero contribution -- the classic false positive a naive over-index scan
produces. Rows below `min_material_change_pct` (`config.py:41`) on
`abs(contribution_pct)` are flagged `material=False` and sorted after the
material rows, never dropped -- a model that wants the small, wild movers can
still see them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

from ..config import get_settings
from ..engines.drivers import DISPROPORTIONATE_AT, robust_sigma
from ..engines.metrics import safe
from .breakdown import BreakdownEngine
from .contract_api import ContractAPI
from .decompose import DecomposeEngine
from .query import QueryEngine
from .timefilter import TimeFilter, TimeSelection

# The k values `top_k_shares` reports cumulative share for, capped at the
# dimension's own member count when it is smaller.
DEFAULT_TOP_K: Tuple[int, ...] = (1, 3, 5, 10)

# How the Gini coefficient here is computed, echoed in the payload so the
# number is reproducible without reading this module.
GINI_METHOD = "sample_corrected"

# Mirrors `scan.MAX_SCAN_MEMBERS` -- a generous, bounded population for the
# per-member `decompose_dimension` scan `find_outlier_contributors` costs.
MAX_OUTLIER_MEMBERS = 200

# A median-absolute-deviation z-score over fewer than this many members is
# noise, not a signal -- report `insufficient` rather than a fabricated score.
MIN_OUTLIER_MEMBERS = 4


def _gini_sample_corrected(values: Sequence[float]) -> Optional[float]:
    """
    Small-sample-corrected Gini coefficient.

    The population estimator `G = (2*Sum(i*v_i))/(n*Sum(v)) - (n+1)/n` (values
    sorted ascending, `i` a 1-based rank) is biased toward 0 on a small `n` --
    it can never reach 1.0 even for total dominance by one member out of a
    handful. The `n/(n-1)` correction (Deltas & Mehran's small-sample Gini) is
    what makes single-member dominance among n=3 members read as exactly 1.0
    rather than 0.667, which is the number a caller comparing across
    dimensions of different cardinality actually needs.

    `None` for n <= 1 or a zero total -- there is no inequality to measure
    between one thing and itself, and no meaningful ratio when nothing summed
    to anything.
    """
    n = len(values)
    if n <= 1:
        return None
    total = sum(values)
    if total == 0:
        return None
    ordered = sorted(values)
    weighted = sum((i + 1) * v for i, v in enumerate(ordered))
    g_biased = (2.0 * weighted) / (n * total) - (n + 1.0) / n
    return g_biased * n / (n - 1)


# ---------------------------------------------------------------------------
# measure_concentration()
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Concentration:
    kpi: str
    label: str
    unit: str
    dimension: str
    kind: str
    # "ok" | "empty" | "not_additive" | "too_many_members" | "has_negative_members"
    status: str
    member_count: int
    grand_total: Optional[float]
    hhi: Optional[float]
    hhi_normalized: Optional[float]
    effective_members: Optional[float]
    gini: Optional[float]
    gini_method: str
    # keyed "top_1" / "top_3" / ... -- cumulative share of the top k members
    top_k_shares: Mapping[str, Optional[float]]
    members_for_50pct: Optional[int]
    members_for_80pct: Optional[int]
    members_for_90pct: Optional[int]
    truncated: bool
    negative_members: Tuple[str, ...]
    selection: TimeSelection
    filters: Mapping[str, Tuple[str, ...]]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "kpi": self.kpi, "label": self.label, "unit": self.unit,
            "dimension": self.dimension, "kind": self.kind, "status": self.status,
            "member_count": self.member_count, "grand_total": safe(self.grand_total),
            "hhi": safe(self.hhi), "hhi_normalized": safe(self.hhi_normalized),
            "effective_members": safe(self.effective_members),
            "gini": safe(self.gini), "gini_method": self.gini_method,
            "top_k_shares": {k: safe(v) for k, v in self.top_k_shares.items()},
            "members_for_50pct": self.members_for_50pct,
            "members_for_80pct": self.members_for_80pct,
            "members_for_90pct": self.members_for_90pct,
            "truncated": self.truncated, "negative_members": list(self.negative_members),
            "selection": self.selection.to_payload(),
            "filters": {k: list(v) for k, v in self.filters.items()},
        }


# ---------------------------------------------------------------------------
# find_outlier_contributors()
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OutlierContributor:
    member: str
    contribution_pct: Optional[float]
    share_of_baseline_pct: Optional[float]
    over_index: Optional[float]
    change_pct: Optional[float]
    # Population-relative robust z-score of `over_index` across this
    # dimension's members -- `None` when the population is degenerate
    # (identical over-indices, MAD and std both zero), never a fabricated
    # score and never `inf`.
    robust_z: Optional[float]
    # `over_index >= DISPROPORTIONATE_AT` (1.2) -- the fixed-threshold
    # judgment `compute_over_index` reports, kept alongside the
    # population-relative one so the two stay visibly distinct.
    is_disproportionate: bool
    # `abs(contribution_pct) >= min_material_change_pct` -- False rows are
    # not dropped, only sorted after the material ones.
    material: bool

    def to_payload(self) -> Dict[str, Any]:
        return {
            "member": self.member, "contribution_pct": safe(self.contribution_pct),
            "share_of_baseline_pct": safe(self.share_of_baseline_pct),
            "over_index": safe(self.over_index), "change_pct": safe(self.change_pct),
            "robust_z": safe(self.robust_z), "is_disproportionate": self.is_disproportionate,
            "material": self.material,
        }


@dataclass(frozen=True)
class OutlierContributors:
    kpi: str
    label: str
    unit: str
    dimension: str
    kind: str
    status: str                       # "ok" | "insufficient"
    population_size: int
    truncated: bool
    min_material_change_pct: float
    rows: Tuple[OutlierContributor, ...]
    selection: TimeSelection
    baseline_selection: TimeSelection
    filters: Mapping[str, Tuple[str, ...]]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "kpi": self.kpi, "label": self.label, "unit": self.unit,
            "dimension": self.dimension, "kind": self.kind, "status": self.status,
            "population_size": self.population_size, "truncated": self.truncated,
            "min_material_change_pct": self.min_material_change_pct,
            "rows": [r.to_payload() for r in self.rows],
            "selection": self.selection.to_payload(),
            "baseline_selection": self.baseline_selection.to_payload(),
            "filters": {k: list(v) for k, v in self.filters.items()},
        }


class ConcentrationEngine:
    """`measure_concentration` / `find_outlier_contributors`, bound to one
    dataset's `ContractAPI`."""

    def __init__(self, api: ContractAPI):
        self._api = api
        self._query = QueryEngine(api)
        self._breakdown = BreakdownEngine(api)
        self._decompose = DecomposeEngine(api)

    # -- measure_concentration() --------------------------------------------
    def measure_concentration(self, df: pd.DataFrame, kpi_key: str, dimension: str,
                              time_filter: Union[Mapping[str, Any], TimeFilter],
                              filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
                              top_k: Sequence[int] = DEFAULT_TOP_K) -> Concentration:
        label, unit, kind = self._api.label(kpi_key), self._api.unit(kpi_key), self._api.kind(kpi_key)
        additive = self._api.is_additive(kpi_key)

        rank = self._breakdown.rank_entities(df, kpi_key, dimension, time_filter,
                                             order="desc", limit=None, filters=filters)
        member_count = len(rank.entries)
        negative_members = tuple(e.member for e in rank.entries
                                 if e.value == e.value and e.value < 0)

        def _degenerate(status: str, top_k_shares: Mapping[str, Optional[float]] = None,
                        members_for: Mapping[int, Optional[int]] = None) -> Concentration:
            grand = self._query.query(df, [kpi_key], time_filter, filters=filters, group_by=[])
            grand_total = grand.cells[0].values[kpi_key]
            mf = members_for or {50: None, 80: None, 90: None}
            return Concentration(
                kpi=kpi_key, label=label, unit=unit, dimension=dimension, kind=kind,
                status=status, member_count=member_count, grand_total=grand_total,
                hhi=None, hhi_normalized=None, effective_members=None,
                gini=None, gini_method=GINI_METHOD,
                top_k_shares=dict(top_k_shares or {}),
                members_for_50pct=mf.get(50), members_for_80pct=mf.get(80),
                members_for_90pct=mf.get(90),
                truncated=rank.truncated, negative_members=negative_members,
                selection=rank.selection, filters=rank.filters)

        if member_count == 0:
            return _degenerate("empty")
        if not additive:
            return _degenerate("not_additive")

        # Shares and cumulative shares are read straight off `rank`'s own
        # entries -- never recomputed -- so `top_k_shares` and
        # `members_for_Npct` can never disagree with what `rank_entities`
        # itself reports for the same call.
        share_valid = rank.entries[0].share_of_total_pct is not None
        top_k_shares: Dict[str, Optional[float]] = {}
        members_for: Dict[int, Optional[int]] = {50: None, 80: None, 90: None}
        if share_valid:
            for k in top_k:
                idx = min(k, member_count) - 1
                top_k_shares[f"top_{k}"] = rank.entries[idx].cumulative_share_pct
            for i, entry in enumerate(rank.entries, start=1):
                cum = entry.cumulative_share_pct or 0.0
                for threshold in (50, 80, 90):
                    if members_for[threshold] is None and cum >= threshold:
                        members_for[threshold] = i

        if rank.truncated:
            return _degenerate("too_many_members", top_k_shares, members_for)
        if negative_members:
            return _degenerate("has_negative_members", top_k_shares, members_for)
        if not share_valid:
            # Additive, untruncated, no negative members, yet the grand total
            # is zero or undefined -- there is genuinely nothing to measure a
            # share of.
            return _degenerate("empty", top_k_shares, members_for)

        grand = self._query.query(df, [kpi_key], time_filter, filters=filters, group_by=[])
        grand_total = grand.cells[0].values[kpi_key]
        values = [e.value for e in rank.entries if e.value == e.value]
        fractions = [v / grand_total for v in values]
        hhi = sum(f * f for f in fractions)
        n = member_count
        hhi_normalized = ((hhi - 1.0 / n) / (1.0 - 1.0 / n)) if n > 1 else None
        effective_members = (1.0 / hhi) if hhi else None
        gini = _gini_sample_corrected(values)

        return Concentration(
            kpi=kpi_key, label=label, unit=unit, dimension=dimension, kind=kind,
            status="ok", member_count=member_count, grand_total=grand_total,
            hhi=hhi, hhi_normalized=hhi_normalized, effective_members=effective_members,
            gini=gini, gini_method=GINI_METHOD, top_k_shares=top_k_shares,
            members_for_50pct=members_for[50], members_for_80pct=members_for[80],
            members_for_90pct=members_for[90],
            truncated=False, negative_members=(),
            selection=rank.selection, filters=rank.filters)

    # -- find_outlier_contributors() ----------------------------------------
    def find_outlier_contributors(self, df: pd.DataFrame, kpi_key: str, dimension: str,
                                  period_a: Union[Mapping[str, Any], TimeFilter],
                                  period_b: Union[Mapping[str, Any], TimeFilter],
                                  filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
                                  max_members: int = MAX_OUTLIER_MEMBERS,
                                  min_members: int = MIN_OUTLIER_MEMBERS) -> OutlierContributors:
        decomposition = self._decompose.decompose_by_dimension(
            df, kpi_key, dimension, period_a, period_b, filters=filters, max_items=max_members)

        population_size = len(decomposition.members)
        truncated = decomposition.others is not None
        material_threshold = get_settings().min_material_change_pct

        if population_size < min_members:
            rows = tuple(
                OutlierContributor(
                    member=m.name, contribution_pct=m.contribution_pct,
                    share_of_baseline_pct=m.share_of_baseline_pct, over_index=m.over_index,
                    change_pct=m.change_pct, robust_z=None, is_disproportionate=False,
                    material=False)
                for m in decomposition.members)
            return OutlierContributors(
                kpi=kpi_key, label=decomposition.label, unit=decomposition.unit,
                dimension=dimension, kind=decomposition.kind, status="insufficient",
                population_size=population_size, truncated=truncated,
                min_material_change_pct=material_threshold, rows=rows,
                selection=decomposition.selection,
                baseline_selection=decomposition.baseline_selection, filters=decomposition.filters)

        over_indices = [m.over_index for m in decomposition.members if m.over_index is not None]
        med, sigma = robust_sigma(over_indices)

        rows: List[OutlierContributor] = []
        for m in decomposition.members:
            z: Optional[float] = None
            if m.over_index is not None and sigma == sigma and sigma > 0:
                z = (m.over_index - med) / sigma
            material = m.contribution_pct is not None and abs(m.contribution_pct) >= material_threshold
            disproportionate = m.over_index is not None and m.over_index >= DISPROPORTIONATE_AT
            rows.append(OutlierContributor(
                member=m.name, contribution_pct=m.contribution_pct,
                share_of_baseline_pct=m.share_of_baseline_pct, over_index=m.over_index,
                change_pct=m.change_pct, robust_z=z, is_disproportionate=disproportionate,
                material=material))

        rows.sort(key=lambda r: (0 if r.material else 1, r.robust_z is None,
                                 -(abs(r.robust_z) if r.robust_z is not None else 0.0)))

        return OutlierContributors(
            kpi=kpi_key, label=decomposition.label, unit=decomposition.unit,
            dimension=dimension, kind=decomposition.kind, status="ok",
            population_size=population_size, truncated=truncated,
            min_material_change_pct=material_threshold, rows=tuple(rows),
            selection=decomposition.selection,
            baseline_selection=decomposition.baseline_selection, filters=decomposition.filters)
