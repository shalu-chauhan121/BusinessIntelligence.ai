"""
Contracts for question interpretation and the KPI driver graph.

These models exist so that each boundary here is typed and testable rather
than a loose dictionary.

The 4-stage pipeline's own contracts (`MaterialSignals`, `HypothesisPrediction`,
`LlmHypothesis`, `LlmHypothesisSet`, `RecommendationCore`) were retired at A9
along with the engines that produced and consumed them
(`engines/{signals,hypotheses,llm_hypotheses,act,contest,investigate}.py`).
What remains — question interpretation and the driver graph — is still load-
bearing: `/api/questions/interpret` and `agent/relations.py`/`agent/kpi_search.py`
both depend on it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# query understanding
# ---------------------------------------------------------------------------
PeriodSource = Literal["explicit", "inferred_default", "dataset_latest"]
KpiRole = Literal["outcome", "comparison", "filter_context"]
MatchBasis = Literal["exact_name", "semantic_tag", "concept_alias",
                     "business_definition", "relevance_text",
                     "llm_proposed_verified"]
IntentType = Literal["change_explanation", "comparison", "ranking",
                     "threshold_check", "unsupported"]
AmbiguityKind = Literal["outcome_unresolved", "outcome_multiple", "period_vague",
                        "comparison_assumed", "dimension_unresolved",
                        "unsupported_by_dataset"]


class PeriodSpec(BaseModel):
    """The period an investigation runs over, and how confidently it was known."""

    year: int
    quarter: Optional[int] = None
    comparison: str = "previous_period"
    source: PeriodSource = "inferred_default"
    # Shown to the reader whenever the period was not stated outright, so an
    # assumed timeframe is never mistaken for one the question asked for.
    assumption_note: str = ""


class KpiRef(BaseModel):
    """A KPI the question referred to, and the evidence for that reading."""

    kpi_key: str
    label: str = ""
    role: KpiRole = "outcome"
    match_basis: MatchBasis = "exact_name"
    confidence: float = 0.0
    matched_text: str = ""                       # the words in the question that matched


class Ambiguity(BaseModel):
    """
    Something the interpreter could not settle.

    `blocking` is the whole point, and the line it draws is between *nothing*
    and *several*. An outcome KPI that resolves to nothing stops the
    investigation and asks, because guessing produces a confident answer to a
    question nobody asked. An outcome that resolves to several proceeds on the
    best-scoring one with `assumed` naming it and `candidates` listing the rest,
    because every one of them is a measure this dataset really has. A vague
    period proceeds on a stated default for the same reason.
    """

    kind: AmbiguityKind
    blocking: bool
    message: str
    candidates: List[KpiRef] = Field(default_factory=list)
    assumed: str = ""


class InvestigationIntent(BaseModel):
    """A business question, resolved against the KPI Contract."""

    question: str
    intent_type: IntentType = "change_explanation"
    outcome: Optional[KpiRef] = None
    comparison_kpis: List[KpiRef] = Field(default_factory=list)
    period: Optional[PeriodSpec] = None
    dimension_hints: List[str] = Field(default_factory=list)
    entity_filters: Dict[str, str] = Field(default_factory=dict)
    ambiguities: List[Ambiguity] = Field(default_factory=list)
    # Words that looked like they named something measurable but bound to
    # nothing in the contract. Surfaced rather than silently dropped, because
    # they are usually the most interesting thing about a failed interpretation.
    unmapped_terms: List[str] = Field(default_factory=list)
    llm_used: bool = False
    grounding_note: str = ""

    @property
    def blocked(self) -> bool:
        return any(a.blocking for a in self.ambiguities)

    @property
    def assumptions(self) -> List[str]:
        """Every non-blocking assumption, for disclosure to the reader."""
        out = [a.message for a in self.ambiguities if not a.blocking]
        if self.period and self.period.assumption_note:
            out.append(self.period.assumption_note)
        return out

    def all_kpi_keys(self) -> List[str]:
        keys = [self.outcome.kpi_key] if self.outcome else []
        return keys + [k.kpi_key for k in self.comparison_kpis]


# ---------------------------------------------------------------------------
# driver graph
# ---------------------------------------------------------------------------
EdgeRelation = Literal["formula_numerator", "formula_denominator", "formula_term",
                       "shared_source_field", "derivation", "semantic_tag",
                       "not_comparable", "llm_proposed"]
EdgeOrigin = Literal["deterministic", "llm"]


class DriverEdge(BaseModel):
    """
    A candidate relationship between two KPIs.

    Deterministic edges are structural facts read out of the contract — a KPI
    that appears in another's formula genuinely drives it. LLM-proposed edges
    are guesses until `verified` is set by a correlation and lead/lag check, and
    an unverified edge carries `weight = 0.0` so it can suggest a hypothesis
    without ever contributing to that hypothesis's score.
    """

    source_kpi: str
    target_kpi: str
    relation: EdgeRelation
    origin: EdgeOrigin = "deterministic"
    weight: float = 0.0
    verified: bool = False
    correlation: Optional[float] = None
    lead_lag_periods: Optional[int] = None
    verification_note: str = ""
    evidence_fields: List[str] = Field(default_factory=list)
    note: str = ""


class DriverGraph(BaseModel):
    """What could plausibly move the outcome KPI, and on what basis."""

    outcome_kpi: str
    edges: List[DriverEdge] = Field(default_factory=list)
    # Dimension members that actually moved the KPI, from `observe.determine_focus`.
    dimension_drivers: List[Dict[str, Any]] = Field(default_factory=list)
    # Dimensions that proved to carry real explanatory signal for this KPI.
    # The contract copies every dataset dimension onto every KPI, so relevance
    # has to be established empirically rather than read off the definition.
    relevant_dimensions: List[str] = Field(default_factory=list)
    unverified_llm_edges: List[DriverEdge] = Field(default_factory=list)
    note: str = ""

    def driver_metrics(self) -> List[str]:
        """Every KPI that may legitimately appear in a hypothesis prediction."""
        return sorted({e.source_kpi for e in self.edges})
