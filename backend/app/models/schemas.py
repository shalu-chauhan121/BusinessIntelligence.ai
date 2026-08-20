"""Request/response models — the API contract the frontend develops against."""
from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

Role = Literal["data_analyst", "business_leader"]
Comparison = Literal["previous_period", "year_over_year"]


class RegisterRequest(BaseModel):
    role: Role = "business_leader"
    display_name: str = ""
    organisation: str = ""


class RoleUpdate(BaseModel):
    role: Role


class DemoLoginRequest(BaseModel):
    email: str
    display_name: str = ""
    role: Role = "business_leader"


class UserOut(BaseModel):
    uid: str
    email: str
    display_name: str
    role: Role
    organisation: str = ""
    created_at: Optional[str] = None
    token_verified: bool = False
    permissions: Dict[str, bool] = Field(default_factory=dict)


class AnalysisRequest(BaseModel):
    kpi: Optional[str] = Field(default=None, description="KPI key, e.g. 'revenue'. Defaults to revenue.")
    year: Optional[int] = Field(default=None, description="Analysis year. Defaults to the latest in the data.")
    quarter: Optional[int] = Field(default=None, ge=1, le=4,
                                   description="1-4, or null for the full year.")
    comparison: Comparison = "previous_period"
    dataset_id: Optional[str] = None
    use_llm: bool = True
    persist: bool = True


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


class StageResponse(BaseModel):
    """Documented shape of a stage response (contents are stage-specific)."""
    stage: str
    data: Dict[str, Any]


class ErrorResponse(BaseModel):
    detail: str
