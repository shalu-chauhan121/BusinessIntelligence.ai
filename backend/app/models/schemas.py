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
    A business question, in the user's own words.

    Replaces KPI selection as the way an investigation starts. Which KPI, which
    period and which comparison are resolved from the question against the
    dataset's KPI contract rather than chosen from a dropdown.
    """

    question: str = Field(min_length=1, max_length=500,
                          description="e.g. 'Why did profit fall in Q4 even though revenue held?'")
    dataset_id: Optional[str] = None
    persona: Optional[Persona] = Field(
        default=None,
        description="Overrides the user's stored persona for this run. Presentation only.")
    use_llm: bool = True
    persist: bool = True


class AnalysisRequest(BaseModel):
    kpi: Optional[str] = Field(default=None, description="KPI key, e.g. 'revenue'. Defaults to revenue.")
    year: Optional[int] = Field(default=None, description="Analysis year. Defaults to the latest in the data.")
    quarter: Optional[int] = Field(default=None, ge=1, le=4,
                                   description="1-4, or null for the full year.")
    comparison: Comparison = "previous_period"
    # A per-stage caller may target the KPI directly (above) OR ask a business
    # question and let the stage resolve its own KPI/period from the contract,
    # the same way `/api/questions/investigate` does. When both are supplied,
    # the question wins — it is the more specific instruction.
    question: Optional[str] = Field(
        default=None, max_length=500,
        description="Resolve the KPI/period from a question instead of the fields above.")
    persona: Optional[Persona] = None
    dataset_id: Optional[str] = None
    use_llm: bool = True
    persist: bool = True


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


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
