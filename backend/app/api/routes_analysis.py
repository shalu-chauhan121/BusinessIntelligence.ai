"""The four investigation stages, the dashboard, and saved investigations."""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..db.repositories import DatasetRepository, InvestigationRepository
from ..deps import active_dataset, current_user
from ..engines.contest import contest as contest_stage
from ..engines.investigate import investigate as investigate_stage
from ..engines.act import act as act_stage
from ..engines.observe import available_timeframes
from ..llm.client import get_llm
from ..models.schemas import AnalysisRequest
from ..services import dataset_service, pipeline
from .redact import (
    is_analyst,
    redact_contest,
    redact_investigation,
    redact_observation,
    redact_result,
)

router = APIRouter(prefix="/api", tags=["analysis"])


def _dataset_for(user: Dict[str, Any], dataset_id: Optional[str]) -> Dict[str, Any]:
    repo = DatasetRepository()
    ds = repo.get(user["uid"], dataset_id) if dataset_id else repo.active(user["uid"])
    if not ds:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "No dataset available. Upload a business metrics CSV on the Data page first.",
        )
    return ds


def _guard(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except (ValueError, dataset_service.DatasetError) as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc


# ---------------------------------------------------------------------------
# dashboard
# ---------------------------------------------------------------------------
@router.get("/dashboard")
def dashboard(year: Optional[int] = Query(default=None),
              quarter: Optional[int] = Query(default=None, ge=1, le=4),
              kpi: Optional[str] = Query(default=None),
              comparison: str = Query(default="previous_period"),
              user: Dict[str, Any] = Depends(current_user),
              dataset: Dict[str, Any] = Depends(active_dataset)) -> Dict[str, Any]:
    """Everything the dashboard needs for one (year, quarter) selection."""
    df, schema = dataset_service.load(dataset)
    observation = _guard(pipeline.run_observe, dataset, kpi, year, quarter, comparison)
    return {
        "dataset": {"id": dataset["_id"], "filename": dataset.get("filename"),
                    "rows": schema.row_count, "grain": schema.grain},
        "timeframes": available_timeframes(df),
        "schema": schema.to_dict(),
        "observe": redact_observation(observation, is_analyst(user)),
        "view": {"role": user.get("role"), "analyst_detail_included": is_analyst(user)},
    }


@router.get("/meta/timeframes")
def timeframes(dataset: Dict[str, Any] = Depends(active_dataset)) -> Dict[str, Any]:
    df, schema = dataset_service.load(dataset)
    return {"timeframes": available_timeframes(df), "kpis": schema.to_dict()["kpi_catalogue"]}


# ---------------------------------------------------------------------------
# stage 1 — observe
# ---------------------------------------------------------------------------
@router.post("/observe")
def observe_endpoint(body: AnalysisRequest,
                     user: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    ds = _dataset_for(user, body.dataset_id)
    observation = _guard(pipeline.run_observe, ds, body.kpi, body.year, body.quarter, body.comparison)
    return {"stage": "observe", "observe": redact_observation(observation, is_analyst(user))}


# ---------------------------------------------------------------------------
# stage 2 — investigate
# ---------------------------------------------------------------------------
@router.post("/investigate")
def investigate_endpoint(body: AnalysisRequest,
                         user: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    ds = _dataset_for(user, body.dataset_id)
    df, schema = dataset_service.load(ds)
    observation = _guard(pipeline.run_observe, ds, body.kpi, body.year, body.quarter, body.comparison)
    llm = get_llm() if body.use_llm else None
    investigation = investigate_stage(df, schema, observation, user["uid"], llm=llm)
    analyst = is_analyst(user)
    return {"stage": "investigate",
            "observe": redact_observation(observation, analyst),
            "investigate": redact_investigation(investigation, analyst)}


# ---------------------------------------------------------------------------
# stage 3 — contest
# ---------------------------------------------------------------------------
@router.post("/contest")
def contest_endpoint(body: AnalysisRequest,
                     user: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    ds = _dataset_for(user, body.dataset_id)
    df, schema = dataset_service.load(ds)
    observation = _guard(pipeline.run_observe, ds, body.kpi, body.year, body.quarter, body.comparison)
    llm = get_llm() if body.use_llm else None
    investigation = investigate_stage(df, schema, observation, user["uid"], llm=llm)
    contested = contest_stage(df, schema, observation, investigation, user["uid"], llm=llm)
    analyst = is_analyst(user)
    return {"stage": "contest",
            "observe": redact_observation(observation, analyst),
            "investigate": redact_investigation(investigation, analyst),
            "contest": redact_contest(contested, analyst)}


# ---------------------------------------------------------------------------
# stage 4 — act
# ---------------------------------------------------------------------------
@router.post("/act")
def act_endpoint(body: AnalysisRequest,
                 user: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    ds = _dataset_for(user, body.dataset_id)
    df, schema = dataset_service.load(ds)
    observation = _guard(pipeline.run_observe, ds, body.kpi, body.year, body.quarter, body.comparison)
    llm = get_llm() if body.use_llm else None
    investigation = investigate_stage(df, schema, observation, user["uid"], llm=llm)
    contested = contest_stage(df, schema, observation, investigation, user["uid"], llm=llm)
    action = act_stage(df, observation, investigation, contested, llm=llm)
    return {"stage": "act", "act": action}


# ---------------------------------------------------------------------------
# full pipeline
# ---------------------------------------------------------------------------
@router.post("/investigations/run")
def run_investigation(body: AnalysisRequest,
                      user: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    """OBSERVE -> INVESTIGATE -> CONTEST -> ACT in one call. Used by the UI."""
    ds = _dataset_for(user, body.dataset_id)
    result = _guard(pipeline.run_full, user["uid"], ds, body.kpi, body.year, body.quarter,
                    body.comparison, body.persist, body.use_llm)
    return redact_result(result, user)


@router.get("/investigations")
def list_investigations(user: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    items = InvestigationRepository().list_for_user(user["uid"])
    return {
        "investigations": [
            {
                "id": i["_id"],
                "kpi": i.get("kpi"), "kpi_label": i.get("kpi_label"),
                "timeframe": i.get("timeframe"), "baseline_timeframe": i.get("baseline_timeframe"),
                "change_pct": i.get("change_pct"), "verdict": i.get("verdict"),
                "headline": i.get("headline"),
                "leading_hypothesis": i.get("leading_hypothesis"),
                "leading_confidence": i.get("leading_confidence"),
                "created_at": i.get("created_at"),
            }
            for i in items
        ]
    }


@router.get("/investigations/{investigation_id}")
def get_investigation(investigation_id: str,
                      user: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    doc = InvestigationRepository().get(user["uid"], investigation_id)
    if not doc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Investigation not found.")
    return redact_result(doc["result"], user)


@router.delete("/investigations/{investigation_id}")
def delete_investigation(investigation_id: str,
                         user: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    if not InvestigationRepository().delete(user["uid"], investigation_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Investigation not found.")
    return {"deleted": investigation_id}
