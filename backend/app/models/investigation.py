"""
The contracts between the stages of a question-driven investigation.

These models exist so that each boundary in the pipeline is typed and testable
rather than a loose dictionary. Two of them carry the load-bearing guarantees of
the whole architecture:

  * `MaterialSignals` is the only channel by which observed facts reach the
    language model. Everything in it has already passed a deterministic
    significance test, so the model is never in a position to mistake noise for
    a finding, or to be asked to explain a change that did not happen.

  * `HypothesisPrediction` is how the model proposes evidence. It states what it
    expects a metric to have done, and the data decides whether that expectation
    held. The model never asserts a number, a direction that was observed, or a
    verdict.

The remaining models carry the question's interpretation, the KPI relationships
the interpretation is grounded in, and the persona-invariant substrate that
recommendations are reframed from.
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
    comparison: str = "previous_period"          # matches schemas.Comparison
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
    # Dimension members that actually moved the KPI, from `determine_focus`.
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


# ---------------------------------------------------------------------------
# the material-signal boundary
# ---------------------------------------------------------------------------
class MaterialSignals(BaseModel):
    """
    The observed facts, filtered to what actually matters.

    This is the only path by which numbers reach the hypothesis prompt. A KPI
    that moved within normal variation is carried as context flagged
    `moved=False` — never as a change to be explained — because a question like
    "why did profit fall even though CAC was flat" needs the flat metric present
    in order to be answerable at all.

    When `nothing_material` is true the pipeline reports that there is nothing to
    explain and generates no hypotheses. Asking a language model to explain noise
    reliably produces an explanation, which is precisely the failure to avoid.
    """

    outcome_kpi: str
    material: List[Dict[str, Any]] = Field(default_factory=list)
    context: List[Dict[str, Any]] = Field(default_factory=list)
    focus_members: List[Dict[str, Any]] = Field(default_factory=list)
    signals_considered: int = 0
    signals_retained: int = 0
    nothing_material: bool = False
    filter_note: str = ""


# ---------------------------------------------------------------------------
# LLM hypothesis contract
# ---------------------------------------------------------------------------
Direction = Literal["up", "down"]


class HypothesisPrediction(BaseModel):
    """
    A falsifiable claim about a metric, made before the data is consulted.

    The generator says what it expects; `engines.hypotheses.evidence` measures
    what happened and flips the stance to "contradicting" when the metric moved
    the other way or did not move at all. A hypothesis therefore cannot claim
    support it does not have, however plausible its wording.
    """

    metric: str
    expected_direction: Direction
    scope: Optional[Dict[str, str]] = None
    # The change that would count as full strength for this prediction, in
    # percent. Mirrors the `reference` argument the evidence helper already uses.
    reference_pct: float = 5.0
    weight: float = 1.0
    rationale: str = ""


class LlmHypothesis(BaseModel):
    """One proposed mechanism, in the shape the deterministic engines consume."""

    key: str
    title: str
    statement: str
    mechanism: str = ""
    family: str = "other"
    predictions: List[HypothesisPrediction] = Field(default_factory=list, max_length=6)
    cause_metric: Optional[str] = None
    cause_direction: Optional[Direction] = None
    reverse_causation_risk: str = ""
    # What would have to be true for this hypothesis to be wrong. Supplied per
    # hypothesis so the disconfirmation search is driven by the hypothesis
    # itself rather than by a lookup table of known template keys.
    contradiction_queries: List[str] = Field(default_factory=list, max_length=4)
    rag_queries: List[str] = Field(default_factory=list, max_length=4)
    missing: List[str] = Field(default_factory=list)
    # Whether this came from reasoning about the detected industry, or from the
    # general business mechanisms that apply to any operation. Both categories
    # are asked for; neither is fabricated when the data cannot support it.
    domain_specific: bool = False


class LlmHypothesisSet(BaseModel):
    hypotheses: List[LlmHypothesis] = Field(default_factory=list, max_length=10)
    generation_note: str = ""


# ---------------------------------------------------------------------------
# persona-invariant recommendation substrate
# ---------------------------------------------------------------------------
class RecommendationCore(BaseModel):
    """
    The facts a recommendation must rest on, before any persona sees it.

    Every persona's advice is generated from this same object, so two readers
    can be told to do different things but never for different reasons. The
    cause metric and evidence references are the constraint: a reframing that
    reaches outside them is not a reframing, it is an invention.
    """

    hypothesis_key: str
    hypothesis_title: str
    cause_metric: Optional[str] = None
    cause_direction: Optional[str] = None
    confidence: float = 0.0
    confidence_label: str = ""
    causal_claim: str = ""
    supporting_evidence: List[str] = Field(default_factory=list)
    contradicting_evidence: List[str] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)
    affected_areas: List[str] = Field(default_factory=list)
    monitoring_threshold: Optional[Dict[str, Any]] = None
    what_would_change_this: List[str] = Field(default_factory=list)
