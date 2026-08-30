"""
The dashboard, question interpretation, the agent loop, saved answers, and
telemetry.

Most routes here return `Dict[str, Any]` with no `response_model`: a payload
is a deep, engine-shaped dict, so pinning it in OpenAPI would have cost more
than it bought. `/questions/ask` is the one exception and declares a real
response model, for the reason `routes_kpi.py` gives about the KPI contract —
its shape is already frozen (`AgentAnswer`) and is the artefact a client
depends on.

The four-stage pipeline (`observe -> investigate -> contest -> act`) and its
endpoints, `/investigations/run` and `/questions/investigate`, were retired at
A9: the agent loop behind `/questions/ask` is now the only way this API
answers a business question. `/questions/interpret` survives — it is a cheap,
independently-tested read of the KPI Contract grounding layer with no
narrative template in it, and nothing downstream depends on the pipeline it
used to feed.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..agent import loop
from ..agent.context import AgentContext
from ..db.repositories import DatasetRepository, InvestigationRepository, TelemetryRepository, new_id
from ..deps import active_dataset, current_user
from ..engines.observe import available_timeframes
from ..llm.client import LLMTransportError, get_llm
from ..models.schemas import AgentAnswerResponse, AgentQuestionRequest, QuestionRequest
from ..query import interpret_question_cached
from ..services import dashboard as dashboard_service
from ..services import dataset_service
from ..services.telemetry import request_telemetry, track_processing_step
from .redact import is_analyst, redact_observation

log = logging.getLogger(__name__)

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


def _llm_guard(fn, *args, **kwargs):
    """
    `_guard`, plus the tool loop's one failure mode that is not about the
    request at all.

    A provider timeout or rate limit is neither a malformed request (422) nor a
    bug in this codebase (500), and it must not be laundered into a 200 with a
    typed status either — that would make an outage indistinguishable from a
    successful answer to `/api/telemetry/summary`. The SDK's own message goes to
    the log, not to the caller.
    """
    try:
        return fn(*args, **kwargs)
    except LLMTransportError as exc:
        log.warning("agent loop transport failure: %s", exc)
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "The reasoning provider is unavailable. Try again in a moment.") from exc
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
        with track_processing_step("Load dataset", "Non-LLM Processing"):
            df, schema = dataset_service.load(dataset, user["uid"])
        observation = _guard(dashboard_service.run_observe, dataset, kpi, year, quarter,
                             comparison, user["uid"])
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
def timeframes(user: Dict[str, Any] = Depends(current_user),
               dataset: Dict[str, Any] = Depends(active_dataset)) -> Dict[str, Any]:
    df, schema = dataset_service.load(dataset, user["uid"])
    return {"timeframes": available_timeframes(df), "kpis": schema.to_dict()["kpi_catalogue"]}


# ---------------------------------------------------------------------------
# question interpretation
# ---------------------------------------------------------------------------
@router.post("/questions/interpret")
def interpret(body: QuestionRequest,
              user: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    """
    How the system reads a question, without running anything.

    Cheap on purpose: it powers the panel that shows which KPI and period were
    understood, so a misreading can be corrected before an investigation runs.
    """
    ds = _dataset_for(user, body.dataset_id)
    df, schema = _guard(dataset_service.load, ds, user["uid"])
    intent = interpret_question_cached(body.question, schema, df, ds["_id"],
                                       llm=get_llm() if body.use_llm else None)
    return {"intent": intent.model_dump(), "blocked": intent.blocked,
            "assumptions": intent.assumptions}


# ---------------------------------------------------------------------------
# agentic question answering
# ---------------------------------------------------------------------------
@router.post("/questions/ask", response_model=AgentAnswerResponse)
def ask_question(body: AgentQuestionRequest,
                 user: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    """
    Ask a business question and get a prose answer plus the complete trail of
    tool calls that produced it.

    Every ending of the agent loop is HTTP 200 with a typed `status`: a
    question the loop could not converge on, a safety refusal, and the absence
    of an API key are all things a client renders, not errors it retries. Only
    a missing dataset (409), a broken one (422) and a provider outage (503) are
    errors.

    Persisted only when `body.persist` is set — `InvestigationRepository` is a
    generic store, so the `AgentAnswer` is saved as-is, with no kpi/verdict
    shape imposed on it. `GET /api/investigations` lists it; `GET
    /api/investigations/{id}` returns it unredacted, for the same reason given
    below.

    The evidence trail is complete for every role — `redact.py` is deliberately
    not applied here. Showing how the answer was reached is the point of the
    endpoint, not an analyst privilege; `view.redaction` says so explicitly.
    """
    ds = _dataset_for(user, body.dataset_id)
    with request_telemetry(user["uid"], "/api/questions/ask") as telemetry:
        with track_processing_step("Build agent context", "Non-LLM Processing"):
            # Loads the dataset, compiles the contract facade and builds all 56
            # tool schemas — the only substantial non-LLM work in the request.
            ctx = _guard(AgentContext.build, user["uid"], ds)
        # `llm` is passed rather than left to `loop.answer`'s own lazy lookup,
        # giving tests the same seam this endpoint uses.
        result = _llm_guard(loop.answer, body.question, ctx, llm=get_llm())

    payload = {
        # Field by field rather than `dataclasses.asdict`: that deep-copies
        # recursively, and `evidence[*]["result"]` is an arbitrarily large
        # nested tool payload. Serialising is this endpoint's whole job, so it
        # is written out where it can be read.
        "status": result.status,
        "question": result.question,
        "answer": result.answer,
        "evidence": result.evidence,
        "kpis_used": result.kpis_used,
        "periods_used": result.periods_used,
        "engine": result.engine,
        "dataset": {"id": ds["_id"], "filename": ds.get("filename")},
        # `analyst_detail_included` keeps the meaning it has everywhere else in
        # this API — whether the caller holds the analyst role — so a client can
        # read it uniformly. `redaction` is what says this particular endpoint
        # withheld nothing from anyone.
        "view": {"role": user.get("role"), "analyst_detail_included": is_analyst(user),
                 "redaction": "none"},
        "telemetry": telemetry.saved,
        "investigation_id": None,
    }

    if body.persist and result.status == "ok":
        # Reserved before the write, not read back after it: `payload["result"]`
        # is `payload` itself, so patching `investigation_id` in after
        # `create()` would mutate the very dict just handed to the store,
        # racing whatever `create()` already did with it (serialise it to
        # disk, hold a reference, or both) rather than reliably landing in
        # what gets persisted.
        payload["investigation_id"] = new_id("inv")
        InvestigationRepository().create(user["uid"], {
            "_id": payload["investigation_id"],
            "question": result.question,
            "answer": result.answer,
            "kpis_used": result.kpis_used,
            "periods_used": result.periods_used,
            "engine": result.engine,
            "dataset_id": ds["_id"],
            "status": result.status,
            "result": payload,
        })

    return payload


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


# ---------------------------------------------------------------------------
# saved agent answers
# ---------------------------------------------------------------------------
_PREVIEW_CHARS = 160


@router.get("/investigations")
def list_investigations(user: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    """
    Saved agent answers, newest first.

    A row predating this batch has no `answer` field (it was a saved run of the
    retired 4-stage pipeline) — it is listed with `legacy_format: true` and no
    preview, rather than guessing at a shape it does not have.
    """
    items = InvestigationRepository().list_for_user(user["uid"])
    rows = []
    for i in items:
        answer = i.get("answer")
        if answer is None:
            rows.append({"id": i["_id"], "legacy_format": True,
                         "question": i.get("question") or i.get("kpi_label") or i.get("kpi") or "",
                         "created_at": i.get("created_at")})
            continue
        preview = answer if len(answer) <= _PREVIEW_CHARS else answer[:_PREVIEW_CHARS].rstrip() + "…"
        rows.append({
            "id": i["_id"],
            "legacy_format": False,
            "question": i.get("question") or "",
            "answer_preview": preview,
            "kpis_used": i.get("kpis_used") or [],
            "turns": (i.get("engine") or {}).get("turns"),
            "created_at": i.get("created_at"),
        })
    return {"investigations": rows}


@router.get("/investigations/{investigation_id}")
def get_investigation(investigation_id: str,
                      user: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    """
    A saved agent answer, unredacted — matching `/questions/ask`'s own
    `redaction: "none"`, since this is exactly what that endpoint returned.

    A pre-A9 row has no `result` in the new shape; it is reported as
    `legacy_format` rather than rendered, since nothing left in this API knows
    how to interpret the retired pipeline's payload.
    """
    doc = InvestigationRepository().get(user["uid"], investigation_id)
    if not doc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Investigation not found.")
    if "answer" not in doc:
        return {"status": "legacy_format", "id": doc["_id"],
                "question": doc.get("question") or doc.get("kpi_label") or doc.get("kpi") or "",
                "created_at": doc.get("created_at")}
    return doc["result"]


@router.delete("/investigations/{investigation_id}")
def delete_investigation(investigation_id: str,
                         user: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    if not InvestigationRepository().delete(user["uid"], investigation_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Investigation not found.")
    return {"deleted": investigation_id}
