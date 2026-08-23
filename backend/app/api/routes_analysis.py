"""The four investigation stages, the dashboard, and saved investigations."""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..db.repositories import DatasetRepository, InvestigationRepository, TelemetryRepository
from ..deps import active_dataset, current_user
from ..engines.contest import contest as contest_stage
from ..engines.investigate import investigate as investigate_stage
from ..engines.act import act as act_stage
from ..engines.observe import available_timeframes
from ..llm.client import get_llm
from ..models.schemas import AnalysisRequest
from ..services import dataset_service, pipeline
from ..services.telemetry import request_telemetry, track_processing_step
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
    with request_telemetry(user["uid"], "/api/dashboard") as telemetry:
        df, schema = dataset_service.load(dataset)
        observation = _guard(pipeline.run_observe, dataset, kpi, year, quarter, comparison)
    return {
        "dataset": {"id": dataset["_id"], "filename": dataset.get("filename"),
                    "rows": schema.row_count, "grain": schema.grain},
        "timeframes": available_timeframes(df),
        "schema": schema.to_dict(),
        "observe": redact_observation(observation, is_analyst(user)),
        "view": {"role": user.get("role"), "analyst_detail_included": is_analyst(user)},
        "telemetry": telemetry.saved,
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
    with request_telemetry(user["uid"], "/api/observe") as telemetry:
        observation = _guard(pipeline.run_observe, ds, body.kpi, body.year, body.quarter, body.comparison)
    return {"stage": "observe", "observe": redact_observation(observation, is_analyst(user)),
            "telemetry": telemetry.saved}


# ---------------------------------------------------------------------------
# stage 2 — investigate
# ---------------------------------------------------------------------------
@router.post("/investigate")
def investigate_endpoint(body: AnalysisRequest,
                         user: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    ds = _dataset_for(user, body.dataset_id)
    with request_telemetry(user["uid"], "/api/investigate") as telemetry:
        df, schema = dataset_service.load(ds)
        observation = _guard(pipeline.run_observe, ds, body.kpi, body.year, body.quarter, body.comparison)
        llm = get_llm() if body.use_llm else None
        with track_processing_step("Investigate", "Non-LLM Processing"):
            investigation = investigate_stage(df, schema, observation, user["uid"], llm=llm)
    analyst = is_analyst(user)
    return {"stage": "investigate",
            "observe": redact_observation(observation, analyst),
            "investigate": redact_investigation(investigation, analyst), "telemetry": telemetry.saved}


# ---------------------------------------------------------------------------
# stage 3 — contest
# ---------------------------------------------------------------------------
@router.post("/contest")
def contest_endpoint(body: AnalysisRequest,
                     user: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    ds = _dataset_for(user, body.dataset_id)
    with request_telemetry(user["uid"], "/api/contest") as telemetry:
        df, schema = dataset_service.load(ds)
        observation = _guard(pipeline.run_observe, ds, body.kpi, body.year, body.quarter, body.comparison)
        llm = get_llm() if body.use_llm else None
        with track_processing_step("Investigate", "Non-LLM Processing"):
            investigation = investigate_stage(df, schema, observation, user["uid"], llm=llm)
        with track_processing_step("Contest", "Non-LLM Processing"):
            contested = contest_stage(df, schema, observation, investigation, user["uid"], llm=llm)
    analyst = is_analyst(user)
    return {"stage": "contest",
            "observe": redact_observation(observation, analyst),
            "investigate": redact_investigation(investigation, analyst),
            "contest": redact_contest(contested, analyst), "telemetry": telemetry.saved}


# ---------------------------------------------------------------------------
# stage 4 — act
# ---------------------------------------------------------------------------
@router.post("/act")
def act_endpoint(body: AnalysisRequest,
                 user: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    ds = _dataset_for(user, body.dataset_id)
    with request_telemetry(user["uid"], "/api/act") as telemetry:
        df, schema = dataset_service.load(ds)
        observation = _guard(pipeline.run_observe, ds, body.kpi, body.year, body.quarter, body.comparison)
        llm = get_llm() if body.use_llm else None
        with track_processing_step("Investigate", "Non-LLM Processing"):
            investigation = investigate_stage(df, schema, observation, user["uid"], llm=llm)
        with track_processing_step("Contest", "Non-LLM Processing"):
            contested = contest_stage(df, schema, observation, investigation, user["uid"], llm=llm)
        with track_processing_step("Act", "Non-LLM Processing"):
            action = act_stage(df, observation, investigation, contested, llm=llm)
    return {"stage": "act", "act": action, "telemetry": telemetry.saved}


# ---------------------------------------------------------------------------
# full pipeline
# ---------------------------------------------------------------------------
@router.post("/investigations/run")
def run_investigation(body: AnalysisRequest,
                      user: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    """OBSERVE -> INVESTIGATE -> CONTEST -> ACT in one call. Used by the UI."""
    with request_telemetry(user["uid"], "/api/investigations/run") as telemetry:
        ds = _dataset_for(user, body.dataset_id)
        result = _guard(pipeline.run_full, user["uid"], ds, body.kpi, body.year, body.quarter,
                        body.comparison, body.persist, body.use_llm)
    result["telemetry"] = telemetry.saved
    return redact_result(result, user)


@router.get("/telemetry/summary")
def telemetry_summary(user: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    records = TelemetryRepository().list_for_user(user["uid"])
    count = len(records)
    latencies = sorted(r.get("latency_ms", 0) for r in records)
    p95 = latencies[max(0, int((count - 1) * 0.95))] if count else 0
    total = lambda key: sum(r.get(key, 0) or 0 for r in records)
    successes = sum(r.get("status") == "success" for r in records)
    return {
        "total_requests": count, "successful_requests": successes, "failed_requests": count - successes,
        "average_latency_ms": round(total("latency_ms") / count) if count else 0, "p95_latency_ms": p95,
        "total_model_calls": total("model_calls"), "average_model_calls": round(total("model_calls") / count, 2) if count else 0,
        "total_input_tokens": total("input_tokens"), "total_output_tokens": total("output_tokens"), "total_tokens": total("total_tokens"),
        "total_prompt_tokens": total("prompt_tokens") or total("input_tokens"),
        "total_completion_tokens": total("completion_tokens") or total("output_tokens"),
        "average_input_tokens": round(total("input_tokens") / count) if count else 0,
        "average_output_tokens": round(total("output_tokens") / count) if count else 0,
        "average_total_tokens": round(total("total_tokens") / count) if count else 0,
        "estimated_total_cost": round(total("estimated_cost"), 6),
        "average_cost_per_request": round(total("estimated_cost") / count, 6) if count else 0,
        "total_llm_duration_ms": sum((r.get("processing", {}).get("llm", {}).get("duration_ms") or 0) for r in records),
        "total_non_llm_duration_ms": sum((r.get("processing", {}).get("non_llm", {}).get("duration_ms") or 0) for r in records),
        "cost_estimated": True,
    }


@router.get("/telemetry/recent")
def telemetry_recent(limit: int = Query(default=10, ge=1, le=50),
                     user: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    records = TelemetryRepository().list_for_user(user["uid"], limit=limit)
    fields = ("trace_id", "timestamp", "start_time", "end_time", "duration_ms", "endpoint", "model_name", "model_calls",
              "input_tokens", "output_tokens", "prompt_tokens", "completion_tokens", "total_tokens", "estimated_cost", "latency_ms", "llm_latency_ms",
              "processing", "status", "error_type", "errors")
    return {"telemetry": [{key: item.get(key) for key in fields} for item in records]}


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
