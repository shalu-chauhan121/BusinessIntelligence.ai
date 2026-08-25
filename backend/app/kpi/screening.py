"""
LLM semantic screening of KPI candidates.

The deterministic layers decide what is *computable*. This layer asks a model
what is *meaningful* — which of the computable candidates a real organisation of
this kind would actually put on a report, what the business calls it, and why it
matters. The model also gets to propose KPIs the rules did not think of, which
is how an unusual domain gets covered without shipping a pack for it.

Two hard rules, both enforced here in code rather than trusted to the prompt:

  1. **The model never computes.** It is handed the dataset's shape — column
     names, semantic types, summary statistics — and never a row.
  2. **The model never invents a field.** Every proposal is re-validated against
     the profiled field list; one naming a column that does not exist is
     discarded and counted, not repaired.

With no API key the whole module degrades to `deterministic_screening`, which
leaves the rules' own verdicts in place and marks everything for review. The
product's no-key guarantee holds.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field, ValidationError

from .derivation import DerivedCandidate
from .domain import DomainContext
from .profiling import DatasetProfile

log = logging.getLogger(__name__)

VALID_VERDICTS = {"valid", "questionable", "reject"}
VALID_KINDS = {"sum", "mean", "ratio"}
VALID_UNITS = {"currency", "count", "percent", "ratio", "duration", "score"}
VALID_TIME_GRAINS = {"day", "week", "month", "quarter", "year"}


# ---------------------------------------------------------------------------
# response schema
#
# The shared `_extract_json` in the LLM client is a regex with no validation.
# The contract is an authoritative artefact, so the response is parsed through
# these models and anything malformed is dropped rather than absorbed.
# ---------------------------------------------------------------------------
class ScreenedCandidate(BaseModel):
    candidate_id: str
    verdict: str = "questionable"
    name: str = ""
    definition: str = ""
    why_relevant: str = ""
    suggested_time_grain: Optional[str] = None
    suggested_entity_grain: List[str] = Field(default_factory=list)
    semantic_tags: List[str] = Field(default_factory=list)
    ambiguities: List[str] = Field(default_factory=list)


class ProposedKpi(BaseModel):
    name: str
    definition: str = ""
    kind: str = "sum"
    field: Optional[str] = None
    minus_field: Optional[str] = None
    numerator_field: Optional[str] = None
    denominator_field: Optional[str] = None
    scale: float = 1.0
    unit: str = "count"
    higher_is_better: bool = True
    why_relevant: str = ""
    semantic_tags: List[str] = Field(default_factory=list)


class ScreeningResponse(BaseModel):
    domain: str = ""
    candidates: List[ScreenedCandidate] = Field(default_factory=list)
    additional_kpis: List[ProposedKpi] = Field(default_factory=list)


class ScreeningResult:
    """What screening produced, plus enough detail to explain what it rejected."""

    def __init__(self) -> None:
        self.screened_by: str = "deterministic"
        self.domain: str = ""
        self.applied: int = 0
        self.rejected_by_model: List[str] = []
        self.discarded_proposals: List[str] = []
        self.new_candidates: List[DerivedCandidate] = []
        self.notes: List[str] = []


# ---------------------------------------------------------------------------
# fact sheet
# ---------------------------------------------------------------------------
def dataset_facts(profile: DatasetProfile, domain_ctx: Optional[DomainContext] = None
                  ) -> Dict[str, Any]:
    """The shape of the dataset — never its contents."""
    business_context = {
        "detected_domain": domain_ctx.domain if domain_ctx else "unknown",
        "confidence_0_to_1": domain_ctx.confidence if domain_ctx else 0.0,
        "is_uncertain": domain_ctx.is_uncertain if domain_ctx else True,
        "evidence": list(domain_ctx.evidence) if domain_ctx else [],
    }
    return {
        "business_context": business_context,
        "row_count": profile.row_count,
        "row_grain": profile.row_grain,
        "date_column": profile.date_column,
        "dimensions": [
            {"name": f.name, "distinct_values": f.distinct_count}
            for f in profile.dimensions
        ],
        "measures": [
            {
                "name": f.name,
                "semantic_type": f.semantic_type,
                "additivity": f.additivity,
                "aggregates_by": f.default_aggregation,
                "is_integer": f.is_integer,
                "is_non_negative": f.is_non_negative,
                "contained_by": f.subset_of,
            }
            for f in profile.measures
        ],
        "detected_hierarchies": [
            {"child": h.child, "parent": h.parent, "clean": h.clean}
            for h in profile.hierarchies
        ],
    }


def candidate_facts(candidates: Sequence[DerivedCandidate]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for c in candidates:
        out.append({
            "candidate_id": c.candidate_id,
            "proposed_name": c.name,
            "kind": c.kind,
            "expression": c.expression or f"{c.numerator} / {c.denominator}",
            "source_fields": c.source_fields,
            "generated_by_rule": c.rule,
            "deterministic_verdict": c.verdict,
        })
    return out


# ---------------------------------------------------------------------------
# validation of model output
# ---------------------------------------------------------------------------
def _validate_proposal(proposal: ProposedKpi,
                       measures: Sequence[str]) -> Tuple[Optional[DerivedCandidate], str]:
    """Turn a model proposal into a candidate, or say why it was discarded."""
    known = set(measures)
    kind = proposal.kind if proposal.kind in VALID_KINDS else None
    if kind is None:
        return None, f"'{proposal.name}': unknown kind '{proposal.kind}'"
    unit = proposal.unit if proposal.unit in VALID_UNITS else "count"

    if kind == "ratio":
        num, den = proposal.numerator_field, proposal.denominator_field
        if not num or not den:
            return None, f"'{proposal.name}': a ratio needs both a numerator and a denominator"
        missing = [f for f in (num, den) if f not in known]
        if missing:
            return None, f"'{proposal.name}': names column(s) {missing} that do not exist"
        if num == den:
            return None, f"'{proposal.name}': numerator and denominator are the same column"
        return DerivedCandidate(
            candidate_id=f"llm_{num}_per_{den}",
            name=proposal.name[:60],
            definition=proposal.definition,
            kind="ratio",
            numerator="{" + num + "}", denominator="{" + den + "}",
            scale=float(proposal.scale or 1.0), unit=unit,
            higher_is_better=bool(proposal.higher_is_better),
            source_fields=[num, den], rule="llm_proposed",
            verdict="questionable", confidence=0.35,
            semantic_tags=list(proposal.semantic_tags),
            relevance=proposal.why_relevant,
            evidence={"origin": "llm_suggested", "revalidated_against_field_list": True},
        ), ""

    base = proposal.field
    if not base:
        return None, f"'{proposal.name}': a {kind} KPI needs a field"
    fields = [base] + ([proposal.minus_field] if proposal.minus_field else [])
    missing = [f for f in fields if f not in known]
    if missing:
        return None, f"'{proposal.name}': names column(s) {missing} that do not exist"
    expression = "{" + base + "}"
    if proposal.minus_field:
        expression += " - {" + proposal.minus_field + "}"
    return DerivedCandidate(
        candidate_id=f"llm_{base}" + (f"_less_{proposal.minus_field}" if proposal.minus_field else ""),
        name=proposal.name[:60],
        definition=proposal.definition,
        kind=kind, expression=expression, unit=unit,
        higher_is_better=bool(proposal.higher_is_better),
        source_fields=[f for f in fields if f], rule="llm_proposed",
        verdict="questionable", confidence=0.35,
        semantic_tags=list(proposal.semantic_tags),
        relevance=proposal.why_relevant,
        evidence={"origin": "llm_suggested", "revalidated_against_field_list": True},
    ), ""


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def deterministic_screening(candidates: Sequence[DerivedCandidate]) -> ScreeningResult:
    """The no-API-key path: the rules' own verdicts stand, and a human reviews."""
    result = ScreeningResult()
    result.screened_by = "deterministic"
    result.notes.append(
        "No language model was used. Candidates carry the deterministic rules' own "
        "verdicts and every one requires review before it becomes authoritative."
    )
    return result


def screen_candidates(profile: DatasetProfile,
                      candidates: List[DerivedCandidate],
                      llm: Any = None,
                      domain_ctx: Optional[DomainContext] = None) -> ScreeningResult:
    """
    Apply semantic screening in place over `candidates`, returning what happened.

    `domain_ctx` is the dataset's deterministically detected business context
    (`app.kpi.domain`) — handed to the model so its `definition`/`why_relevant`
    text is grounded in the same evidence the no-key path uses, rather than the
    model guessing an industry independently.

    Any failure — no key, a transport error, malformed JSON — degrades to the
    deterministic result. Discovery must never fail because the model was
    unavailable.
    """
    if llm is None or not getattr(llm, "enabled", False):
        return deterministic_screening(candidates)

    try:
        raw = llm.screen_kpi_candidates(dataset_facts(profile, domain_ctx),
                                        candidate_facts(candidates))
        parsed = ScreeningResponse.model_validate(raw)
    except ValidationError as exc:
        log.warning("KPI screening returned an unusable shape: %s", exc)
        result = deterministic_screening(candidates)
        result.notes.append("The screening model returned malformed output; it was ignored.")
        return result
    except Exception as exc:                          # transport, key, timeout
        log.warning("KPI screening unavailable: %s", exc)
        result = deterministic_screening(candidates)
        result.notes.append(f"The screening model was unavailable ({exc}); it was ignored.")
        return result

    result = ScreeningResult()
    result.screened_by = f"llm:{getattr(llm, 'model', 'anthropic')}"
    result.domain = parsed.domain

    by_id = {c.candidate_id: c for c in candidates}
    for verdict in parsed.candidates:
        cand = by_id.get(verdict.candidate_id)
        if cand is None or verdict.verdict not in VALID_VERDICTS:
            continue
        if verdict.verdict == "reject":
            result.rejected_by_model.append(cand.candidate_id)
            cand.verdict = "questionable"
            cand.confidence = min(cand.confidence, 0.2)
            cand.evidence["llm_verdict"] = "reject"
            cand.evidence["llm_reason"] = verdict.why_relevant or verdict.definition
            result.applied += 1
            continue

        # The model may improve the prose and the grain hint. It may never
        # change the maths — the formula is not in its gift.
        if verdict.name:
            cand.name = verdict.name[:60]
        if verdict.definition:
            cand.definition = verdict.definition
        if verdict.why_relevant:
            cand.relevance = verdict.why_relevant
        if verdict.semantic_tags:
            cand.semantic_tags = list(verdict.semantic_tags)
        if verdict.suggested_time_grain in VALID_TIME_GRAINS:
            cand.evidence["suggested_time_grain"] = verdict.suggested_time_grain
        known_dims = {d.name for d in profile.dimensions}
        entity = [d for d in verdict.suggested_entity_grain if d in known_dims]
        if entity:
            cand.evidence["suggested_entity_grain"] = entity
        if verdict.ambiguities:
            cand.evidence["ambiguities"] = list(verdict.ambiguities)
        cand.verdict = verdict.verdict
        cand.evidence["llm_verdict"] = verdict.verdict
        result.applied += 1

    measures = [m.name for m in profile.measures]
    existing = {c.signature for c in candidates}
    for proposal in parsed.additional_kpis:
        candidate, why = _validate_proposal(proposal, measures)
        if candidate is None:
            result.discarded_proposals.append(why)
            log.info("Discarded an LLM KPI proposal: %s", why)
            continue
        if candidate.signature in existing:
            continue
        existing.add(candidate.signature)
        result.new_candidates.append(candidate)
    return result
