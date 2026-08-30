"""Request/response models — the API contract the frontend develops against."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

Role = Literal["data_analyst", "business_leader"]
# Presentation, not authorisation. A persona reframes the explanation and the
# recommendations; it can never widen what the server is willing to send, which
# stays governed by `Role` alone. See `app.personas`.
Persona = Literal["business_analyst", "business_manager", "business_leader",
                  "domain_specialist", "operational_user"]
Comparison = Literal["previous_period", "year_over_year"]


class RegisterRequest(BaseModel):
    role: Role = "business_leader"
    display_name: str = ""
    organisation: str = ""


class RoleUpdate(BaseModel):
    role: Role


class PersonaUpdate(BaseModel):
    persona: Persona


class DemoLoginRequest(BaseModel):
    email: str
    display_name: str = ""
    role: Role = "business_leader"


class UserOut(BaseModel):
    uid: str
    email: str
    display_name: str
    role: Role
    persona: Persona = "business_analyst"
    organisation: str = ""
    created_at: Optional[str] = None
    token_verified: bool = False
    permissions: Dict[str, bool] = Field(default_factory=dict)


class QuestionRequest(BaseModel):
    """
    A business question, in the user's own words, for `/api/questions/interpret`.

    Reading a question (which KPI, which period) is the only thing this model
    still serves after A9 retired the 4-stage pipeline it used to kick off —
    `persona` and `persist` went with that pipeline, since interpretation alone
    neither frames an answer nor produces anything to save.
    """

    question: str = Field(min_length=1, max_length=500,
                          description="e.g. 'Why did profit fall in Q4 even though revenue held?'")
    dataset_id: Optional[str] = None
    use_llm: bool = True


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


# ---------------------------------------------------------------------------
# agentic question answering  (POST /api/questions/ask)
#
# Typed both ways, unlike the four stage endpoints in the same router. The
# reason is the one given for the KPI contract below: the shape is the artefact
# a client depends on. `agent.loop.AgentAnswer` is already a frozen dataclass
# with a fixed set of fields, and `status` is a closed set the client must
# branch on — both belong in /openapi.json rather than being discovered by
# reading route code.
# ---------------------------------------------------------------------------
AgentStatus = Literal["ok", "max_turns_exhausted", "truncated", "refused", "llm_required"]


class AgentQuestionRequest(BaseModel):
    """
    A business question for the agent loop.

    Deliberately narrower than the old `QuestionRequest`. It has no `use_llm`
    (the loop has no deterministic fallback by design, so `False` would only
    mean "return `llm_required` on purpose") and no `persona`
    (`prompts.agent_system` takes a seed dict and has no persona seam — a field
    the server accepted and then ignored would be worse than an absent one).

    `persist`, unlike those two, is a legitimate lever: `InvestigationRepository`
    is a generic `{**payload}` store, so an `AgentAnswer` (question/answer/
    evidence/kpis_used/periods_used/engine) persists exactly as it is, with no
    kpi/verdict/hypothesis shape imposed on it.

    `max_turns` is deliberately not exposed either: it is a cost lever a client
    should not hold, and `settings.llm_max_turns` already governs it.
    """

    question: str = Field(min_length=1, max_length=500,
                          description="e.g. 'What should I be worried about right now?'")
    dataset_id: Optional[str] = None
    persist: bool = False


class AgentEvidenceStep(BaseModel):
    """One tool call, exactly as `LLMClient._call_with_tools` recorded it."""

    step: int
    tool: str
    args: Dict[str, Any] = Field(default_factory=dict)
    # Already JSON-safe: `registry._deep_safe` applies `metrics.safe` to every
    # leaf of every tool result before it reaches the trace.
    result: Any = None
    is_error: bool = False


class AgentAnswerResponse(BaseModel):
    """`agent.loop.AgentAnswer`, serialised, plus the blocks every analysis response carries."""

    status: AgentStatus
    question: str
    answer: str
    evidence: List[AgentEvidenceStep] = Field(default_factory=list)
    kpis_used: List[str] = Field(default_factory=list)
    periods_used: List[Any] = Field(default_factory=list)
    engine: Dict[str, Any] = Field(default_factory=dict)
    dataset: Dict[str, Any] = Field(default_factory=dict)
    view: Dict[str, Any] = Field(default_factory=dict)
    # Set only when the request asked to persist and the answer was saved —
    # the id `GET /api/investigations/{id}` and `DELETE` take.
    investigation_id: Optional[str] = None
    telemetry: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# KPI contract
#
# Unlike the analysis endpoints, which return `Dict[str, Any]`, the KPI contract
# is a formal artefact: its shape is the thing downstream systems depend on, so
# it is typed both ways. The response models are the contract models themselves,
# in `app.kpi.contract`.
# ---------------------------------------------------------------------------
class KpiDiscoverRequest(BaseModel):
    dataset_id: Optional[str] = None
    use_llm: bool = Field(default=True,
                          description="Run LLM semantic screening. Without an API key the "
                                      "deterministic rules decide on their own.")


class KpiUpdateRequest(BaseModel):
    """A partial override of one KPI definition. Any edit resets its approval."""

    patch: Dict[str, Any] = Field(
        description="Fields to override, e.g. {'granularity': {...}, 'business_definition': '...'}"
    )


class KpiCreateRequest(BaseModel):
    """A KPI the business defines itself, which discovery cannot infer."""

    name: str
    formula: Dict[str, Any] = Field(
        description="{'kind': 'sum'|'mean'|'ratio', 'expression': '{a} - {b}', "
                    "'numerator_expression': ..., 'denominator_expression': ..., 'scale': 1.0}"
    )
    kpi_id: Optional[str] = None
    business_definition: str = ""
    computation_note: str = ""
    source_fields: List[str] = Field(default_factory=list)
    dimensions: List[str] = Field(default_factory=list)
    filters: List[Dict[str, Any]] = Field(default_factory=list)
    unit: str = "count"
    higher_is_better: bool = True
    aggregation: Optional[Dict[str, Any]] = None
    granularity: Optional[Dict[str, Any]] = None
    time_semantics: Optional[Dict[str, Any]] = None
    business_rules: List[Dict[str, Any]] = Field(default_factory=list)
    validation_rules: List[Dict[str, Any]] = Field(default_factory=list)
    relevance: str = ""
    semantic_tags: List[str] = Field(default_factory=list)


class KpiRejectRequest(BaseModel):
    reason: str = ""


class ConflictResolveRequest(BaseModel):
    option_id: str
    rationale: str = Field(default="", description="Why this resolution was chosen. Kept "
                                                   "in the provenance of every KPI it touched.")


class StageResponse(BaseModel):
    """Documented shape of a stage response (contents are stage-specific)."""
    stage: str
    data: Dict[str, Any]


class ErrorResponse(BaseModel):
    detail: str
