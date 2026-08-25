"""
KPI Contract lifecycle: discover -> propose -> edit -> resolve -> approve.

This module is the only place that assembles a `KpiContract`, and the only place
that decides a KPI has become authoritative. Two entry points matter:

`discover(...)`
    Runs the full pipeline — profile, bind the library, derive candidates,
    screen them, detect conflicts — and produces a **draft** whose KPIs are
    `proposed` or `needs_confirmation`. Nothing here is authoritative until a
    human approves it.

`bootstrap(...)`
    Produces a **provisional** contract with the same KPIs already marked
    approved. This is the migration path: a user who has never opened the KPI
    Studio, and every dataset uploaded before this layer existed, still gets a
    working product with values identical to the hard-coded registry. The
    `provisional` status is what tells the UI that no human has reviewed it.

The approval rules are enforced here rather than in the API, so they hold
whatever calls them:

  * a KPI with an unresolved **blocking** conflict cannot be approved;
  * a KPI whose granularity was inferred but not confirmed cannot be approved;
  * a contract with any unresolved blocking conflict cannot be approved.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from ..db.repositories import KpiContractRepository, new_id, now_iso
from .conflicts import detect_conflicts
from .contract import (
    AggregationSpec,
    ApprovalState,
    CalendarSpec,
    ComparabilityEntry,
    ConflictFlag,
    ContainmentEvidence,
    FormulaSpec,
    GranularitySpec,
    HierarchySpec,
    KpiContract,
    KpiDefinition,
    Provenance,
    RejectedCandidate,
    SourceBinding,
    SourceCompleteness,
    TIME_GRAIN_ORDER,
    TimeSemantics,
    UnavailableKpi,
    ValidationRule,
)
from .derivation import (
    DerivedCandidate,
    build_atomic_candidates,
    build_derived_candidates,
    library_signatures,
)
from .domain import DomainContext, detect_domain_context
from .explanation import compose_explanation
from .library import (LibraryMatch, bind_library, detect_domains, display_name,
                      resolve_expression)
from .profiling import DatasetProfile, profile_dataset
from .resolver import compile_contract
from .screening import screen_candidates


class KpiContractError(ValueError):
    """A lifecycle operation was refused — usually because a human must decide."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _time_grain_from_schema_grain(grain: str) -> str:
    """Map the ingestion layer's detected cadence onto a contract time grain."""
    mapping = {"daily": "day", "weekly": "week", "monthly": "month",
               "quarterly": "quarter", "yearly": "year"}
    if grain in mapping:
        return mapping[grain]
    m = re.match(r"^(\d+)-day$", grain or "")
    if m:
        days = int(m.group(1))
        if days <= 3:
            return "day"
        if days <= 10:
            return "week"
        if days <= 45:
            return "month"
        return "quarter"
    return "month"


def _valid_rollups(time_grain: str) -> List[str]:
    if time_grain not in TIME_GRAIN_ORDER:
        return ["quarter", "year"]
    return TIME_GRAIN_ORDER[TIME_GRAIN_ORDER.index(time_grain):]


def _rollup_policy(kind: str, additivity_hint: str = "flow") -> str:
    """
    How a value may be carried to a coarser time grain.

    A ratio is `recompute_from_components`, never `mean`: averaging four weekly
    margins does not give the monthly margin. This is the rule the old registry
    satisfied by accident of implementation and the contract now states.
    """
    if kind == "ratio":
        return "recompute_from_components"
    if kind == "mean":
        return "mean"
    return "sum"


def _unique_id(base: str, taken: set) -> str:
    if base not in taken:
        taken.add(base)
        return base
    n = 2
    while f"{base}_{n}" in taken:
        n += 1
    taken.add(f"{base}_{n}")
    return f"{base}_{n}"


def _source_binding(dataset_id: str, dataset_label: str, fields: Sequence[str],
                    profile: DatasetProfile, schema: Any) -> SourceBinding:
    return SourceBinding(
        dataset_id=dataset_id,
        source_label=dataset_label,
        fields=list(fields),
        native_grain=list(profile.row_grain),
        calendar_id="gregorian",
        refresh_cadence=getattr(schema, "grain", "unknown") or "unknown",
        completeness=SourceCompleteness(
            date_min=getattr(schema, "date_min", "") or "",
            date_max=getattr(schema, "date_max", "") or "",
            row_count=int(getattr(schema, "row_count", 0) or 0),
            null_rates={f.name: round(f.null_rate, 4)
                        for f in profile.fields if f.name in set(fields)},
        ),
        precedence=0,
    )


def _granularity(profile: DatasetProfile, time_grain: str, kind: str,
                 requires_confirmation: bool, declared_by: str = "inferred") -> GranularitySpec:
    return GranularitySpec(
        entity_grain=[],                       # whole business, sliced by dimensions
        time_grain=time_grain,
        native_row_grain=list(profile.row_grain),
        valid_rollups=_valid_rollups(time_grain),
        declared_by=declared_by,
        requires_confirmation=requires_confirmation,
        confidence=0.4 if requires_confirmation else 0.7,
        note=("The dataset's own cadence was used. Confirm the level this KPI is meant "
              "to be reported at."),
    )


def _validation_rules(unit: str, kind: str) -> List[ValidationRule]:
    rules: List[ValidationRule] = []
    if unit == "percent" and kind == "ratio":
        rules.append(ValidationRule(kind="range", min=0.0, max=100.0,
                                    detail="A percentage outside 0-100 indicates the "
                                           "numerator is not contained by the denominator.",
                                    severity="warning"))
    if unit in ("currency", "count"):
        rules.append(ValidationRule(kind="non_negative",
                                    detail="A negative value here usually means a "
                                           "reversal or a data-quality problem.",
                                    severity="info"))
    return rules


# ---------------------------------------------------------------------------
# building definitions
# ---------------------------------------------------------------------------
def _definition_from_library(match: LibraryMatch, kpi_id: str, profile: DatasetProfile,
                             source: SourceBinding, time_grain: str, date_column: str,
                             proposed: bool, domain_ctx: DomainContext) -> KpiDefinition:
    lib = match.library
    fm = match.field_map
    kind = lib.kind
    formula = FormulaSpec(
        expression=resolve_expression(lib.expression, fm),
        kind=kind,
        numerator_expression=resolve_expression(lib.numerator, fm) or None,
        denominator_expression=resolve_expression(lib.denominator, fm) or None,
        scale=lib.scale,
    )
    fields = match.source_fields
    needs_confirm = proposed and kind == "ratio"
    name = display_name(match)
    # The library's own `definition`/`relevance` are a single hand-written
    # sentence per library id, identical whichever dataset activated the KPI.
    # `compose_explanation` writes fresh text grounded in the domain this
    # specific dataset was detected to belong to and the fields it actually
    # bound — that is what makes the same library id ("cost_of_goods") read
    # differently on a retailer than on a hospital.
    business_definition, relevance = compose_explanation(
        name, kind, lib.unit, list(lib.semantic_tags), lib.higher_is_better,
        fields, domain_ctx)
    return KpiDefinition(
        kpi_id=kpi_id,
        name=name,
        business_definition=business_definition,
        kpi_type="general",
        formula=formula,
        computation_note=(f"Computed as {formula.numerator_expression} / "
                          f"{formula.denominator_expression}"
                          + (f", scaled by {lib.scale:g}" if lib.scale != 1 else "")
                          if kind == "ratio" else
                          f"{'Summed' if kind == 'sum' else 'Averaged'} over the period: "
                          f"{formula.expression}"),
        sources=[source.model_copy(update={"fields": list(fields)})],
        source_fields=list(fields),
        dimensions=[d.name for d in profile.dimensions],
        unit=lib.unit,
        higher_is_better=lib.higher_is_better,
        aggregation=AggregationSpec(
            method="ratio" if kind == "ratio" else kind,
            rollup_policy=_rollup_policy(kind),
            cross_sectional_policy=_rollup_policy(kind),
            note=("Recomputed from its numerator and denominator at every level; a rate "
                  "is never averaged." if kind == "ratio" else ""),
        ),
        granularity=_granularity(profile, time_grain, kind, needs_confirm),
        time_semantics=TimeSemantics(date_field=date_column, date_role="period_start"),
        hierarchy_refs=[],
        validation_rules=_validation_rules(lib.unit, kind),
        relevance=relevance,
        semantic_tags=list(lib.semantic_tags),
        provenance=Provenance(
            origin="general_library",
            derived_from=[lib.id],
            computability_evidence={"bound_concepts": fm, "domain": lib.domain},
            screened_by="deterministic",
            created_at=now_iso(),
            notes=[f"Business context: {domain_ctx.domain} "
                  f"(confidence {domain_ctx.confidence:g}). "
                  + " ".join(domain_ctx.evidence)],
        ),
        confidence=0.85,
        status="needs_confirmation" if needs_confirm else ("proposed" if proposed else "approved"),
        approval=ApprovalState(approved=not proposed),
    )


def _definition_from_candidate(cand: DerivedCandidate, kpi_id: str, profile: DatasetProfile,
                               source: SourceBinding, time_grain: str, date_column: str,
                               proposed: bool, domain_ctx: DomainContext) -> KpiDefinition:
    kind = cand.kind
    formula = FormulaSpec(
        expression=cand.expression,
        kind=kind,
        numerator_expression=cand.numerator or None,
        denominator_expression=cand.denominator or None,
        scale=cand.scale,
    )
    kpi_type = "derived" if cand.rule not in ("atomic_measure",) else "atomic"
    needs_confirm = proposed and (kind == "ratio" or cand.verdict == "questionable")
    grain = _granularity(profile, time_grain, kind, needs_confirm)
    suggested = cand.evidence.get("suggested_time_grain")
    if suggested in TIME_GRAIN_ORDER:
        grain.time_grain = suggested
        grain.valid_rollups = _valid_rollups(suggested)
    suggested_entity = cand.evidence.get("suggested_entity_grain")
    if suggested_entity:
        grain.entity_grain = list(suggested_entity)

    origin = "llm_suggested" if cand.rule == "llm_proposed" else (
        "dynamic_derived" if kpi_type == "derived" else "dynamic_atomic")

    llm_screened = bool(cand.evidence.get("llm_verdict"))
    if llm_screened and cand.definition and cand.relevance:
        # The screening model reviewed this specific candidate and was instructed
        # (KPI_DISCOVERY_SYSTEM) to write to the same four questions using the
        # same detected domain context handed to it — trust its text over the
        # deterministic template, which never saw this candidate at all.
        business_definition, relevance = cand.definition, cand.relevance
    else:
        business_definition, relevance = compose_explanation(
            cand.name, kind, cand.unit, list(cand.semantic_tags), cand.higher_is_better,
            cand.source_fields, domain_ctx)
        if cand.rule == "atomic_measure" and cand.verdict == "questionable" and cand.definition:
            # A data-quality caveat (e.g. "this column is already a rate") the
            # domain narrative has no way to know on its own — keep it, appended.
            business_definition = f"{business_definition} {cand.definition}"

    return KpiDefinition(
        kpi_id=kpi_id,
        name=cand.name,
        business_definition=business_definition,
        kpi_type=kpi_type,
        formula=formula,
        computation_note=(cand.numerator + " / " + cand.denominator) if kind == "ratio"
                         else cand.expression,
        sources=[source.model_copy(update={"fields": list(cand.source_fields)})],
        source_fields=list(cand.source_fields),
        dimensions=[d.name for d in profile.dimensions],
        unit=cand.unit,
        higher_is_better=cand.higher_is_better,
        aggregation=AggregationSpec(
            method="ratio" if kind == "ratio" else kind,
            rollup_policy=_rollup_policy(kind),
            cross_sectional_policy=_rollup_policy(kind),
        ),
        granularity=grain,
        time_semantics=TimeSemantics(date_field=date_column, date_role="period_start"),
        validation_rules=_validation_rules(cand.unit, kind),
        relevance=relevance,
        semantic_tags=list(cand.semantic_tags),
        provenance=Provenance(
            origin=origin,
            derived_from=list(cand.source_fields),
            derivation_rule=cand.rule,
            computability_evidence=dict(cand.evidence),
            screened_by=str(cand.evidence.get("llm_verdict") and "llm" or "deterministic"),
            screening_verdict=cand.verdict,
            created_at=now_iso(),
            notes=([] if llm_screened else
                  [f"Business context: {domain_ctx.domain} "
                   f"(confidence {domain_ctx.confidence:g}). "
                   + " ".join(domain_ctx.evidence)]),
        ),
        confidence=cand.confidence,
        status="needs_confirmation" if needs_confirm else ("proposed" if proposed else "approved"),
        approval=ApprovalState(approved=not proposed),
    )


# ---------------------------------------------------------------------------
# comparability
# ---------------------------------------------------------------------------
def compute_comparability(kpis: Sequence[KpiDefinition]) -> None:
    """
    Fill in, for every KPI, which others it may legitimately be compared with.

    Downstream code today puts every KPI on the same scoreboard over the same
    slice and correlates any pair, regardless of grain or calendar. The contract
    publishes the answer so that consumers can stop doing that; this layer does
    not change their behaviour, it just makes the constraint knowable.
    """
    for a in kpis:
        entries: List[ComparabilityEntry] = []
        for b in kpis:
            if a.kpi_id == b.kpi_id:
                continue
            reasons: List[str] = []
            if a.granularity.time_grain != b.granularity.time_grain:
                reasons.append(
                    f"different time grain ({a.granularity.time_grain} vs "
                    f"{b.granularity.time_grain})")
            if a.granularity.entity_grain != b.granularity.entity_grain:
                reasons.append(
                    f"different entity grain ({a.granularity.label} vs {b.granularity.label})")
            if a.time_semantics.calendar_id != b.time_semantics.calendar_id:
                reasons.append(
                    f"different calendar ({a.time_semantics.calendar_id} vs "
                    f"{b.time_semantics.calendar_id})")
            a_cadence = a.sources[0].refresh_cadence if a.sources else "unknown"
            b_cadence = b.sources[0].refresh_cadence if b.sources else "unknown"
            if a_cadence != b_cadence:
                reasons.append(f"different refresh cadence ({a_cadence} vs {b_cadence})")
            entries.append(ComparabilityEntry(kpi_id=b.kpi_id, comparable=not reasons,
                                              reasons=reasons))
        a.comparability = entries


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------
def build_contract(uid: str, dataset: Dict[str, Any], df: pd.DataFrame, schema: Any,
                   proposed: bool, llm: Any = None) -> KpiContract:
    """
    The shared discovery pipeline.

    `proposed=True`  -> a draft for human review (the KPI Studio flow)
    `proposed=False` -> a provisional contract that works immediately (migration)
    """
    dataset_id = dataset.get("_id") or dataset.get("id") or "unknown"
    dataset_label = dataset.get("filename", "") or ""
    date_column = schema.date_column
    measures = list(schema.base_metrics) + list(schema.extra_metrics)

    profile = profile_dataset(df, date_column, schema.dimensions, measures)
    matches, misses = bind_library(profile)
    # A field is "already covered" only when a library KPI exposes it *directly*.
    # A column that merely feeds a ratio — `fulfilled_orders` inside fulfilment
    # rate — still deserves a KPI of its own, which is what the registry this
    # replaces did with every numeric column it did not recognise.
    covered = [m.source_fields[0] for m in matches if len(m.source_fields) == 1]

    # The most plausible business context for this DATASET, from the concepts
    # that actually bound. Every KPI's explanation is written through this one
    # lens below — including "general" KPIs like revenue, so the same library
    # id reads differently on a hospital than on a retailer. This never
    # influences what binds, what is computable, or any computed value.
    domain_ctx = detect_domain_context(matches, profile)

    atomic = build_atomic_candidates(profile, covered)
    derived, rejected = build_derived_candidates(profile, exclude=library_signatures(matches))
    candidates = atomic + derived

    screening = screen_candidates(profile, candidates, llm=llm if proposed else None,
                                  domain_ctx=domain_ctx)
    candidates = candidates + screening.new_candidates

    time_grain = _time_grain_from_schema_grain(getattr(schema, "grain", "") or "")
    all_fields = sorted({f.name for f in profile.measures})
    source = _source_binding(dataset_id, dataset_label, all_fields, profile, schema)

    taken: set = set()
    kpis: List[KpiDefinition] = []
    for match in matches:
        kpi_id = _unique_id(match.library.id, taken)
        kpis.append(_definition_from_library(match, kpi_id, profile, source,
                                             time_grain, date_column, proposed, domain_ctx))
    for cand in candidates:
        if (cand.evidence.get("llm_verdict") == "reject" and proposed
                and cand.rule != "atomic_measure"):
            # The model judged it meaningless. Keep it out of the review queue,
            # but record that it was considered and why it was dropped.
            rejected.append(RejectedCandidate(
                label=cand.name,
                expression=cand.expression or f"{cand.numerator} / {cand.denominator}",
                rule=cand.rule,
                reason=("Screened out as not a meaningful KPI for this business: "
                        + str(cand.evidence.get("llm_reason") or "no reason given")),
            ))
            continue
        kpi_id = _unique_id(cand.candidate_id, taken)
        kpis.append(_definition_from_candidate(cand, kpi_id, profile, source,
                                               time_grain, date_column, proposed, domain_ctx))

    hierarchies = [
        HierarchySpec(
            name=f"{h.child}_to_{h.parent}",
            levels=[h.child, h.parent],
            level_fields={h.child: h.child, h.parent: h.parent},
            containment_evidence=[ContainmentEvidence(
                child=h.child, parent=h.parent, violations=h.violations,
                checked_members=h.checked_members, clean=h.clean)],
            rollup_allowed=h.clean,
        )
        for h in profile.hierarchies
    ]
    for kpi in kpis:
        kpi.hierarchy_refs = [h.name for h in hierarchies if h.rollup_allowed]

    compute_comparability(kpis)

    contract = KpiContract(
        contract_id=new_id("kc"),
        uid=uid,
        dataset_id=dataset_id,
        version=1,
        is_current=True,
        status="draft" if proposed else "provisional",
        dataset_label=dataset_label,
        detected_domains=detect_domains(matches),
        domain_confidence=domain_ctx.confidence,
        domain_uncertain=domain_ctx.is_uncertain,
        domain_evidence=list(domain_ctx.evidence),
        calendars=[CalendarSpec()],
        hierarchies=hierarchies,
        sources=[source],
        kpis=kpis,
        unavailable=[
            UnavailableKpi(library_id=m.library.id, name=m.library.name,
                           domain=m.library.domain, why_unavailable=m.why,
                           missing_concepts=m.missing_concepts)
            for m in misses
        ],
        rejected_candidates=rejected,
        field_profiles=profile.to_dict_list(),
        screened_by=screening.screened_by,
        discovery_note=" ".join(screening.notes) if screening.notes else "",
        created_at=now_iso(),
        updated_at=now_iso(),
    )

    resolver = compile_contract(contract, include_unapproved=True)
    contract.conflicts = detect_conflicts(
        kpis, profile, matches, df=df, resolver=resolver,
        quarters=len(getattr(schema, "quarters", []) or []),
    )
    if screening.discarded_proposals:
        contract.discovery_note = (
            contract.discovery_note
            + f" {len(screening.discarded_proposals)} model proposal(s) were discarded for "
              f"naming columns that do not exist."
        ).strip()
    return contract


def discover(uid: str, dataset: Dict[str, Any], df: pd.DataFrame, schema: Any,
             llm: Any = None) -> KpiContract:
    """Full discovery producing a draft for human review."""
    return build_contract(uid, dataset, df, schema, proposed=True, llm=llm)


def bootstrap(uid: str, dataset: Dict[str, Any], df: pd.DataFrame, schema: Any) -> KpiContract:
    """
    A provisional contract that works without anyone having reviewed anything.

    Deliberately deterministic — no LLM — so that an existing dataset behaves
    identically to how it did before this layer existed.
    """
    return build_contract(uid, dataset, df, schema, proposed=False, llm=None)


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------
def _to_doc(contract: KpiContract) -> Dict[str, Any]:
    doc = contract.model_dump(mode="json")
    doc["_id"] = contract.contract_id
    return doc


def _from_doc(doc: Dict[str, Any]) -> KpiContract:
    payload = {k: v for k, v in doc.items() if k != "_id"}
    payload.setdefault("contract_id", doc.get("_id", ""))
    return KpiContract.model_validate(payload)


def save(contract: KpiContract) -> KpiContract:
    contract.updated_at = now_iso()
    KpiContractRepository().save(contract.uid, _to_doc(contract))
    return contract


def load_current(uid: str, dataset_id: str) -> Optional[KpiContract]:
    """The live contract — what the analysis engines resolve against."""
    doc = KpiContractRepository().current(uid, dataset_id)
    return _from_doc(doc) if doc else None


def load_editable(uid: str, dataset_id: str) -> Optional[KpiContract]:
    """
    The contract the KPI Studio edits: the draft under review if there is one,
    otherwise the live contract.

    Keeping these separate is what lets a reviewer approve KPIs one at a time
    without the dashboard changing under everyone else's feet mid-review.
    """
    repo = KpiContractRepository()
    doc = repo.draft(uid, dataset_id) or repo.current(uid, dataset_id)
    return _from_doc(doc) if doc else None


def load(uid: str, contract_id: str) -> Optional[KpiContract]:
    doc = KpiContractRepository().get(uid, contract_id)
    return _from_doc(doc) if doc else None


def get_or_bootstrap(uid: str, dataset: Dict[str, Any], df: pd.DataFrame,
                     schema: Any) -> KpiContract:
    """
    The contract the analysis engines resolve against.

    If the user has one, it is used. If not — an existing dataset, or one
    uploaded before the KPI Studio was visited — a provisional contract is
    generated and persisted so that behaviour is unchanged and the Studio has
    something to show.
    """
    dataset_id = dataset.get("_id") or dataset.get("id") or "unknown"
    repo = KpiContractRepository()
    existing = load_current(uid, dataset_id)
    if existing:
        return existing

    # Versions exist but none is flagged current — promote the newest usable one
    # rather than generating another. Bootstrapping again would quietly discard
    # whatever review had already happened.
    previous = [d for d in repo.versions(uid, dataset_id)
                if d.get("status") in ("approved", "provisional")]
    if previous:
        newest = max(previous, key=lambda d: int(d.get("version", 0)))
        repo.make_current(uid, dataset_id, newest["_id"])
        return _from_doc(repo.get(uid, newest["_id"]) or newest)

    contract = bootstrap(uid, dataset, df, schema)
    contract.version = repo.next_version(uid, dataset_id)
    save(contract)
    repo.make_current(uid, dataset_id, contract.contract_id)
    return contract


def replace_with_discovery(uid: str, dataset: Dict[str, Any], df: pd.DataFrame,
                           schema: Any, llm: Any = None) -> KpiContract:
    """
    Run discovery and store it as a new draft version.

    Any KPI the user already approved in the current contract keeps its
    definition and its approval — re-running discovery must never silently
    revert a human decision.
    """
    dataset_id = dataset.get("_id") or dataset.get("id") or "unknown"
    repo = KpiContractRepository()
    current = load_current(uid, dataset_id)
    editable = load_editable(uid, dataset_id)

    draft = discover(uid, dataset, df, schema, llm=llm)
    draft.version = repo.next_version(uid, dataset_id)
    # A draft is never live. Until it is approved the previously current
    # contract keeps driving every dashboard and investigation.
    draft.is_current = False

    if current or editable:
        carried = 0
        by_id = {k.kpi_id: k for k in draft.kpis}
        # Only a contract a HUMAN approved carries decisions forward. A
        # provisional contract's approvals are granted by the bootstrap for
        # backwards compatibility, not chosen by anyone — treating them as
        # decisions would let a dataset skip review entirely.
        human_reviewed = bool(current and current.status == "approved")
        for old in (current.kpis if current else []):
            if old.provenance.origin == "user_defined":
                continue
            if not human_reviewed or old.status != "approved":
                continue
            if old.kpi_id in by_id:
                by_id[old.kpi_id] = old
                carried += 1
        # User-defined KPIs are never regenerated by discovery, so carry them over.
        # User-defined KPIs are explicit requirements. Preserve them from the
        # editable draft when present, rather than silently losing an analyst's
        # in-progress definition when discovery is re-run.
        user_defined = [k for k in (editable.kpis if editable else [])
                        if k.provenance.origin == "user_defined"]
        draft.kpis = list(by_id.values()) + user_defined
        if carried or user_defined:
            draft.discovery_note = (
                f"{draft.discovery_note} Carried forward {carried} previously approved "
                f"and {len(user_defined)} user-defined KPI(s) from version "
                f"{(editable or current).version}.").strip()
        compute_comparability(draft.kpis)

    # Deliberately not made current: a draft is under review, not live. It goes
    # live when `approve_contract` succeeds, which is also the point at which
    # every blocking conflict must have been resolved by a person.
    save(draft)
    return draft


# ---------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------
def _require_kpi(contract: KpiContract, kpi_id: str) -> KpiDefinition:
    kpi = contract.kpi(kpi_id)
    if kpi is None:
        raise KpiContractError(f"This contract has no KPI '{kpi_id}'.")
    return kpi


EDITABLE_FIELDS = {
    "name", "business_definition", "unit", "higher_is_better", "relevance",
    "semantic_tags", "dimensions", "filters", "formula", "aggregation",
    "granularity", "time_semantics", "business_rules", "validation_rules",
    "hierarchy_refs", "depends_on", "computation_note",
}


def update_kpi(contract: KpiContract, kpi_id: str, patch: Dict[str, Any],
               actor: str) -> KpiDefinition:
    """
    Apply a user override. Any edit resets approval — a definition that changed
    has not been approved in its new form.
    """
    kpi = _require_kpi(contract, kpi_id)
    unknown = set(patch) - EDITABLE_FIELDS
    if unknown:
        raise KpiContractError(f"These fields cannot be edited: {', '.join(sorted(unknown))}.")

    # Round-trip through the model so nested structures in the patch are
    # validated rather than trusted — a granularity or formula arriving as a
    # raw dict from the API must be a real one before it is stored.
    data = kpi.model_dump()
    data.update(patch)
    updated = KpiDefinition.model_validate(data)

    if "formula" in patch:
        from .resolver import compile_kpi
        compile_kpi(updated)          # reject a malformed formula at the door

    if "granularity" in patch:
        updated.granularity.declared_by = "user"
        updated.granularity.requires_confirmation = False
        updated.granularity.confidence = 1.0
        updated.approval.user_confirmed_granularity = True
    if "formula" in patch or "business_definition" in patch:
        updated.approval.user_confirmed_definition = True

    if updated.provenance.origin != "user_defined":
        updated.kpi_type = "user_defined" if ("formula" in patch) else updated.kpi_type
    updated.provenance.edited_by = list(updated.provenance.edited_by) + [actor]
    updated.provenance.notes = list(updated.provenance.notes) + [
        f"Edited {', '.join(sorted(patch))} at {now_iso()}."]
    updated.approval.approved = False
    updated.approval.approved_by = None
    updated.approval.approved_at = None
    updated.status = "proposed"

    contract.kpis = [updated if k.kpi_id == kpi_id else k for k in contract.kpis]
    compute_comparability(contract.kpis)
    return updated


def add_user_kpi(contract: KpiContract, payload: Dict[str, Any], actor: str) -> KpiDefinition:
    """Create a KPI the business defines itself, which discovery cannot infer."""
    taken = {k.kpi_id for k in contract.kpis}
    base = (payload.get("kpi_id") or payload.get("name", "kpi")).strip().lower()
    kpi_id = _unique_id(re.sub(r"[^a-z0-9_]+", "_", base).strip("_") or "kpi", taken)

    source = contract.sources[0] if contract.sources else SourceBinding(
        dataset_id=contract.dataset_id)
    formula = FormulaSpec.model_validate(payload["formula"])
    from .resolver import compile_kpi
    # Parse first, then validate every referenced field against this contract's
    # actual bound source fields. A user-requested KPI is mandatory when its
    # inputs exist; when they do not, reject it explicitly instead of storing a
    # definition that could vanish from the resolver later.
    referenced = set(compile_kpi(KpiDefinition(
        kpi_id="_validation", name="_validation", formula=formula,
        provenance=Provenance(origin="user_defined"),
    )).source_fields)
    available = {field for binding in contract.sources for field in binding.fields}
    unavailable = referenced - available
    if unavailable:
        raise KpiContractError(
            "User-defined KPI requires unavailable source field(s): "
            + ", ".join(sorted(unavailable)))
    kind = formula.kind
    definition = KpiDefinition(
        kpi_id=kpi_id,
        name=payload.get("name") or kpi_id,
        business_definition=payload.get("business_definition", ""),
        kpi_type="user_defined",
        formula=formula,
        computation_note=payload.get("computation_note", ""),
        sources=[source],
        source_fields=sorted(referenced),
        dimensions=list(payload.get("dimensions", [])),
        filters=payload.get("filters", []),
        unit=payload.get("unit", "count"),
        higher_is_better=bool(payload.get("higher_is_better", True)),
        aggregation=AggregationSpec.model_validate(payload["aggregation"])
        if payload.get("aggregation") else
        AggregationSpec(method="ratio" if kind == "ratio" else kind,
                        rollup_policy=_rollup_policy(kind)),
        granularity=GranularitySpec.model_validate(payload["granularity"])
        if payload.get("granularity") else GranularitySpec(declared_by="user"),
        time_semantics=TimeSemantics.model_validate(payload["time_semantics"])
        if payload.get("time_semantics") else
        TimeSemantics(date_field=contract.sources[0].native_grain[0]
                      if contract.sources and contract.sources[0].native_grain else ""),
        business_rules=payload.get("business_rules", []),
        validation_rules=payload.get("validation_rules", []),
        relevance=payload.get("relevance", ""),
        semantic_tags=list(payload.get("semantic_tags", [])),
        provenance=Provenance(origin="user_defined", screened_by="user",
                              created_at=now_iso(), edited_by=[actor]),
        confidence=1.0,
        required=True,
        status="proposed",
        approval=ApprovalState(user_confirmed_definition=True),
    )
    # Compile now so a malformed formula is rejected at the door, not at render.
    compile_kpi(definition)

    contract.kpis = list(contract.kpis) + [definition]
    compute_comparability(contract.kpis)
    return definition


def delete_kpi(contract: KpiContract, kpi_id: str) -> None:
    _require_kpi(contract, kpi_id)
    dependents = [k.kpi_id for k in contract.kpis if kpi_id in k.depends_on]
    if dependents:
        raise KpiContractError(
            f"'{kpi_id}' cannot be removed because {', '.join(dependents)} depend on it.")
    contract.kpis = [k for k in contract.kpis if k.kpi_id != kpi_id]
    compute_comparability(contract.kpis)


def approve_kpi(contract: KpiContract, kpi_id: str, actor: str) -> KpiDefinition:
    """
    Make one KPI authoritative — refused while anything is genuinely undecided.
    """
    kpi = _require_kpi(contract, kpi_id)
    blocking = contract.blocking_conflicts_for(kpi_id)
    if blocking:
        raise KpiContractError(
            f"'{kpi.name}' has {len(blocking)} unresolved blocking conflict(s): "
            + "; ".join(c.detail for c in blocking)
        )
    if kpi.granularity.requires_confirmation:
        raise KpiContractError(
            f"'{kpi.name}' needs its granularity confirmed before it can be approved. "
            f"The system inferred {kpi.granularity.label} but could not verify it."
        )
    kpi.status = "approved"
    kpi.approval.approved = True
    kpi.approval.approved_by = actor
    kpi.approval.approved_at = now_iso()
    kpi.approval.rejected_reason = None
    return kpi


def reject_kpi(contract: KpiContract, kpi_id: str, actor: str, reason: str = "") -> KpiDefinition:
    kpi = _require_kpi(contract, kpi_id)
    kpi.status = "rejected"
    kpi.approval.approved = False
    kpi.approval.approved_by = None
    kpi.approval.approved_at = None
    kpi.approval.rejected_reason = reason or "Rejected by the reviewer."
    kpi.provenance.edited_by = list(kpi.provenance.edited_by) + [actor]
    return kpi


def resolve_conflict(contract: KpiContract, conflict_id: str, option_id: str,
                     rationale: str, actor: str) -> ConflictFlag:
    """
    Record a human decision on an ambiguity, and act on it where the option says to.

    The rationale is kept on the conflict and echoed into the provenance of every
    KPI it touched, so the audit trail explains not just what the definition is
    but why it is that and not the alternative.
    """
    conflict = next((c for c in contract.conflicts if c.conflict_id == conflict_id), None)
    if conflict is None:
        raise KpiContractError(f"No conflict '{conflict_id}' in this contract.")
    option = next((o for o in conflict.resolution_options if o.option_id == option_id), None)
    if option is None and conflict.resolution_options:
        raise KpiContractError(
            f"'{option_id}' is not one of the options for this conflict: "
            + ", ".join(o.option_id for o in conflict.resolution_options))

    effect = dict(option.effect) if option else {}
    for rejected_id in effect.get("reject", []) or ([effect["reject"]]
                                                    if isinstance(effect.get("reject"), str)
                                                    else []):
        if contract.kpi(rejected_id):
            reject_kpi(contract, rejected_id, actor,
                       reason=f"Superseded by conflict resolution: {rationale}".strip())
    if effect.get("unit") and effect.get("kpi_id"):
        target = contract.kpi(effect["kpi_id"])
        if target:
            target.unit = effect["unit"]
    if effect.get("hierarchy") is None and "hierarchy" in effect:
        contract.hierarchies = [h for h in contract.hierarchies
                                if h.name not in {f"{a}_to_{b}" for a, b in
                                                  [(conflict.affected_fields + ["", ""])[:2]]}]

    conflict.resolved = True
    conflict.resolved_by = actor
    conflict.resolved_at = now_iso()
    conflict.resolution_option_id = option_id
    conflict.resolution_rationale = rationale

    for kpi_id in conflict.affected_kpis:
        kpi = contract.kpi(kpi_id)
        if kpi:
            kpi.provenance.notes = list(kpi.provenance.notes) + [
                f"Conflict '{conflict.kind}' resolved as '{option_id}' by {actor}: {rationale}"]
    return conflict


def approve_contract(contract: KpiContract, actor: str) -> KpiContract:
    """
    Make the whole contract authoritative.

    Refused while any blocking conflict stands. This is the point at which the
    contract becomes the source of truth for every downstream stage, so it is
    also the point at which every unresolved ambiguity has to have been decided
    by a person.
    """
    blocking = contract.blocking_conflicts
    if blocking:
        raise KpiContractError(
            f"{len(blocking)} blocking conflict(s) must be resolved before this contract can "
            f"be approved: " + "; ".join(c.detail for c in blocking)
        )
    approved = [k for k in contract.kpis if k.status == "approved"]
    if not approved:
        raise KpiContractError(
            "Approve at least one KPI before approving the contract — a contract with no "
            "authoritative KPI cannot drive any analysis."
        )
    contract.status = "approved"
    contract.approved_at = now_iso()
    contract.approved_by = actor
    # Approval is what takes a draft live.
    contract.is_current = True
    return contract


def resolver_for(contract: KpiContract) -> Dict[str, Any]:
    """The compiled, approved KPIs — what the analysis engines consume."""
    return compile_contract(contract)
