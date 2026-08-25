"""
Conflict and ambiguity detection.

The rule this module exists to enforce: **the system never silently resolves a
conflicting business definition or assumption.** Where two definitions disagree,
where a name promises a relationship the data does not support, or where the
system cannot tell which column a concept means, it says so and stops.

Severity is what makes that mechanical rather than advisory:

    blocking  a human must choose before the affected KPI can be approved
    warning   the KPI is usable but the caveat travels with it
    info      recorded for the audit trail

`service.approve_*` refuses while any blocking conflict stands, so a contract
cannot become authoritative with an unresolved contradiction inside it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from .contract import ConflictFlag, KpiDefinition, ResolutionOption
from .library import LibraryMatch
from .profiling import DatasetProfile
from .resolver import CompiledKpi, KpiResolutionError

# Below five quarters the significance test in Observe cannot separate a signal
# from normal variation. The existing schema warning says so; the contract makes
# it a first-class caveat attached to every KPI.
MIN_QUARTERS_FOR_SIGNIFICANCE = 5


def _cid(kind: str, *parts: str) -> str:
    return "cf_" + "_".join([kind, *[p.replace(" ", "-") for p in parts if p]])[:120]


def detect_conflicts(
    kpis: Sequence[KpiDefinition],
    profile: DatasetProfile,
    matches: Sequence[LibraryMatch],
    df: Optional[pd.DataFrame] = None,
    resolver: Optional[Dict[str, CompiledKpi]] = None,
    quarters: int = 0,
) -> List[ConflictFlag]:
    out: List[ConflictFlag] = []

    out += _duplicate_and_disagreeing_definitions(kpis)
    out += _ambiguous_concept_bindings(matches, kpis)
    out += _hierarchy_violations(profile)
    out += _failed_subset_promises(kpis, profile)
    out += _unconfirmed_grain(kpis, profile)
    out += _calendar_disagreement(kpis)
    if quarters:
        out += _insufficient_history(kpis, quarters)
    if df is not None and resolver:
        out += _implausible_units(kpis, resolver, df)
    return out


# ---------------------------------------------------------------------------
# duplicate / disagreeing definitions
# ---------------------------------------------------------------------------
def _signature(k: KpiDefinition) -> str:
    f = k.formula
    return f"{f.kind}|{f.expression}|{f.numerator_expression}|{f.denominator_expression}|{f.scale}"


def _duplicate_and_disagreeing_definitions(kpis: Sequence[KpiDefinition]) -> List[ConflictFlag]:
    """
    Two KPIs claiming the same name with different maths is the classic
    'different definitions of the same KPI across sources' problem, and it is
    blocking: whichever one a report happens to use changes the answer.
    """
    out: List[ConflictFlag] = []

    by_name: Dict[str, List[KpiDefinition]] = {}
    for k in kpis:
        by_name.setdefault(k.name.strip().lower(), []).append(k)
    for name, group in by_name.items():
        if len(group) < 2:
            continue
        signatures = {_signature(k) for k in group}
        ids = [k.kpi_id for k in group]
        if len(signatures) > 1:
            out.append(ConflictFlag(
                conflict_id=_cid("formula_disagreement", name),
                kind="formula_disagreement", severity="blocking",
                detail=(f"{len(group)} KPIs are named '{group[0].name}' but compute it "
                        f"differently. Downstream analysis cannot tell which definition a "
                        f"number came from. Choose the authoritative one, or rename the others."),
                affected_kpis=ids,
                resolution_options=[
                    ResolutionOption(option_id=f"keep:{k.kpi_id}",
                                     label=f"Treat '{k.kpi_id}' as authoritative",
                                     detail=(k.formula.expression
                                             or f"{k.formula.numerator_expression} / "
                                                f"{k.formula.denominator_expression}"),
                                     effect={"keep": k.kpi_id,
                                             "reject": [o for o in ids if o != k.kpi_id]})
                    for k in group
                ],
            ))
        else:
            out.append(ConflictFlag(
                conflict_id=_cid("duplicate_definition", name),
                kind="duplicate_definition", severity="warning",
                detail=(f"{len(group)} KPIs share the name '{group[0].name}' and the same "
                        f"formula. Keeping both duplicates every chart and comparison."),
                affected_kpis=ids,
                resolution_options=[
                    ResolutionOption(option_id=f"keep:{ids[0]}",
                                     label=f"Keep only '{ids[0]}'",
                                     effect={"keep": ids[0], "reject": ids[1:]}),
                    ResolutionOption(option_id="keep_all", label="Keep both",
                                     effect={"keep_all": True}),
                ],
            ))

    by_sig: Dict[str, List[KpiDefinition]] = {}
    for k in kpis:
        by_sig.setdefault(_signature(k), []).append(k)
    for sig, group in by_sig.items():
        names = {k.name.strip().lower() for k in group}
        if len(group) > 1 and len(names) > 1:
            out.append(ConflictFlag(
                conflict_id=_cid("duplicate_definition", "same-maths", group[0].kpi_id),
                kind="duplicate_definition", severity="info",
                detail=(f"{', '.join(k.name for k in group)} are computed identically. "
                        f"They may be the same KPI under different names."),
                affected_kpis=[k.kpi_id for k in group],
            ))
    return out


# ---------------------------------------------------------------------------
# ambiguous concept binding
# ---------------------------------------------------------------------------
def _ambiguous_concept_bindings(matches: Sequence[LibraryMatch],
                                kpis: Sequence[KpiDefinition]) -> List[ConflictFlag]:
    """Two columns matched one concept equally well — the system must not pick."""
    known = {k.kpi_id for k in kpis}
    out: List[ConflictFlag] = []
    for m in matches:
        if m.library.id not in known:
            continue
        for concept in m.ambiguous_concepts:
            binding = m.bindings[concept]
            options = [binding.field_name] + [n for n, s in binding.alternatives
                                              if abs(s - binding.score) < 1e-9]
            out.append(ConflictFlag(
                conflict_id=_cid("aggregation_ambiguity", m.library.id, concept),
                kind="aggregation_ambiguity", severity="blocking",
                detail=(f"'{m.library.name}' needs a field for the concept '{concept}', and "
                        f"{len(options)} columns match it equally well: {', '.join(options)}. "
                        f"Which one is authoritative changes the number."),
                affected_kpis=[m.library.id], affected_fields=options,
                resolution_options=[
                    ResolutionOption(option_id=f"bind:{o}", label=f"Use '{o}'",
                                     effect={"kpi_id": m.library.id, "concept": concept,
                                             "field": o})
                    for o in options
                ],
            ))
    return out


# ---------------------------------------------------------------------------
# hierarchies
# ---------------------------------------------------------------------------
def _hierarchy_violations(profile: DatasetProfile) -> List[ConflictFlag]:
    """
    A child value mapping to more than one parent breaks every roll-up along
    that hierarchy. Repairing it silently would quietly change reported totals.
    """
    out: List[ConflictFlag] = []
    for h in profile.hierarchies:
        if h.clean:
            continue
        out.append(ConflictFlag(
            conflict_id=_cid("hierarchy_violation", h.child, h.parent),
            kind="hierarchy_violation", severity="blocking",
            detail=(f"'{h.child}' looks like it rolls up into '{h.parent}', but "
                    f"{h.violations} of {h.checked_members} '{h.child}' values appear under "
                    f"more than one '{h.parent}'. Aggregating along this hierarchy would "
                    f"double-count or mis-attribute those members."),
            affected_fields=[h.child, h.parent],
            resolution_options=[
                ResolutionOption(option_id="treat_as_hierarchy",
                                 label=f"Treat {h.child} → {h.parent} as a hierarchy anyway",
                                 detail="Roll-ups will attribute each member to its most "
                                        "frequent parent.",
                                 effect={"hierarchy": [h.child, h.parent], "accept": True}),
                ResolutionOption(option_id="treat_as_independent",
                                 label=f"Treat {h.child} and {h.parent} as independent",
                                 detail="No roll-up between them; both remain slicing "
                                        "dimensions.",
                                 effect={"hierarchy": None}),
            ],
        ))
    return out


# ---------------------------------------------------------------------------
# a name promising a relationship the data does not support
# ---------------------------------------------------------------------------
def _failed_subset_promises(kpis: Sequence[KpiDefinition],
                            profile: DatasetProfile) -> List[ConflictFlag]:
    """
    A library rate binds on names. If the bound numerator is not actually
    contained by the bound denominator on the rows, the KPI will produce values
    above 100% and the definition is wrong for this dataset — or the data is.
    """
    out: List[ConflictFlag] = []
    for k in kpis:
        if k.formula.kind != "ratio" or k.unit != "percent":
            continue
        num_fields = [f for f in k.source_fields
                      if "{" + f + "}" in (k.formula.numerator_expression or "")]
        den_fields = [f for f in k.source_fields
                      if "{" + f + "}" in (k.formula.denominator_expression or "")]
        if len(num_fields) != 1 or len(den_fields) != 1:
            continue
        num, den = profile.get(num_fields[0]), profile.get(den_fields[0])
        if not num or not den or num.name == den.name:
            continue
        # Margin-shaped ratios subtract inside the numerator; containment of the
        # subtrahend was already checked when the candidate was generated.
        if "-" in (k.formula.numerator_expression or ""):
            continue
        if den.name in num.subset_of:
            continue
        checks = num.evidence.get("subset_checks", {}).get(den.name)
        rows_within = checks.get("rows_within") if isinstance(checks, dict) else None
        out.append(ConflictFlag(
            conflict_id=_cid("subset_check_failed", k.kpi_id),
            kind="subset_check_failed", severity="blocking",
            detail=(f"'{k.name}' is defined as {num.name} as a share of {den.name}, but "
                    f"{num.name} is not within {den.name} on the actual rows"
                    + (f" (it holds on only {rows_within:.0%} of them)"
                       if isinstance(rows_within, (int, float)) else "")
                    + ". The rate will exceed 100%. Either the definition does not fit this "
                      "dataset or the data has a quality problem."),
            affected_kpis=[k.kpi_id], affected_fields=[num.name, den.name],
            resolution_options=[
                ResolutionOption(option_id="accept", label="Use the definition anyway",
                                 detail="Values above 100% will be reported as measured.",
                                 effect={"accept": True}),
                ResolutionOption(option_id="reject", label="Drop this KPI",
                                 effect={"reject": k.kpi_id}),
            ],
        ))
    return out


# ---------------------------------------------------------------------------
# grain
# ---------------------------------------------------------------------------
def _unconfirmed_grain(kpis: Sequence[KpiDefinition],
                       profile: DatasetProfile) -> List[ConflictFlag]:
    out: List[ConflictFlag] = []
    if not profile.row_grain_is_unique and profile.row_grain:
        out.append(ConflictFlag(
            conflict_id=_cid("grain_mismatch", "row-grain"),
            kind="grain_mismatch", severity="warning",
            detail=(f"The key {profile.row_grain} does not uniquely identify a row — "
                    f"{profile.duplicate_row_groups} duplicate groups remain. Every KPI's "
                    f"grain is therefore uncertain, and totals may double-count."),
            affected_fields=list(profile.row_grain),
        ))
    unconfirmed = [k.kpi_id for k in kpis if k.granularity.requires_confirmation]
    if unconfirmed:
        out.append(ConflictFlag(
            conflict_id=_cid("grain_mismatch", "unconfirmed"),
            kind="grain_mismatch", severity="warning",
            detail=(f"{len(unconfirmed)} KPI(s) have a granularity the system inferred but "
                    f"could not verify. Confirm the intended grain before relying on "
                    f"comparisons or roll-ups."),
            affected_kpis=unconfirmed,
        ))
    return out


def _calendar_disagreement(kpis: Sequence[KpiDefinition]) -> List[ConflictFlag]:
    calendars = {k.time_semantics.calendar_id for k in kpis if k.time_semantics.calendar_id}
    if len(calendars) <= 1:
        return []
    return [ConflictFlag(
        conflict_id=_cid("calendar_mismatch", "contract"),
        kind="calendar_mismatch", severity="blocking",
        detail=(f"KPIs in this contract use {len(calendars)} different calendars "
                f"({', '.join(sorted(calendars))}). A period-over-period comparison across "
                f"them compares different spans of time."),
        affected_kpis=[k.kpi_id for k in kpis
                       if k.time_semantics.calendar_id in calendars],
    )]


def _insufficient_history(kpis: Sequence[KpiDefinition], quarters: int) -> List[ConflictFlag]:
    if quarters >= MIN_QUARTERS_FOR_SIGNIFICANCE:
        return []
    return [ConflictFlag(
        conflict_id=_cid("insufficient_history", "contract"),
        kind="insufficient_history", severity="warning",
        detail=(f"The dataset covers {quarters} quarter(s). The significance test needs at "
                f"least {MIN_QUARTERS_FOR_SIGNIFICANCE} to separate a real signal from normal "
                f"variation, and 8 or more to model seasonality. Every KPI here inherits that "
                f"limitation."),
        affected_kpis=[k.kpi_id for k in kpis],
    )]


# ---------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------
def _implausible_units(kpis: Sequence[KpiDefinition], resolver: Dict[str, CompiledKpi],
                       df: pd.DataFrame) -> List[ConflictFlag]:
    """A percentage that is not in 0-100 over the whole dataset is mis-declared."""
    out: List[ConflictFlag] = []
    for k in kpis:
        if k.unit != "percent":
            continue
        compiled = resolver.get(k.kpi_id)
        if compiled is None:
            continue
        try:
            value = compiled.compute(df)
        except KpiResolutionError:
            continue
        if value != value:            # NaN
            continue
        if -0.001 <= value <= 100.001:
            continue
        out.append(ConflictFlag(
            conflict_id=_cid("unit_mismatch", k.kpi_id),
            kind="unit_mismatch", severity="warning",
            detail=(f"'{k.name}' is declared as a percentage but evaluates to {value:,.1f} "
                    f"across the whole dataset. Either the unit or the formula is wrong."),
            affected_kpis=[k.kpi_id],
            resolution_options=[
                ResolutionOption(option_id="unit_ratio", label="Change the unit to ratio",
                                 effect={"kpi_id": k.kpi_id, "unit": "ratio"}),
                ResolutionOption(option_id="accept", label="Keep it as a percentage",
                                 effect={"accept": True}),
            ],
        ))
    return out
