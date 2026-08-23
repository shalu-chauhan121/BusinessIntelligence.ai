"""Small, best-effort runtime telemetry for LLM-backed insight requests."""
from __future__ import annotations

import contextvars
import json
import logging
import math
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, Optional
from uuid import uuid4

from ..config import get_settings
from ..db.repositories import TelemetryRepository

log = logging.getLogger(__name__)

# Defaults are deliberately central and displayed as estimates. Override without
# code changes with LLM_PRICING_JSON, e.g. {"model":{"input_per_1m":3,"output_per_1m":15}}.
MODEL_PRICING = {"claude-sonnet-4-5": {"input_per_1m": 3.0, "output_per_1m": 15.0}}
_current: contextvars.ContextVar[Optional["TelemetrySession"]] = contextvars.ContextVar("telemetry", default=None)


def _pricing(model: str) -> Optional[Dict[str, float]]:
    configured = get_settings().llm_pricing_json.strip()
    try:
        prices = {**MODEL_PRICING, **(json.loads(configured) if configured else {})}
        value = prices.get(model)
        if value and "input_per_1m" in value and "output_per_1m" in value:
            return value
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        log.warning("Ignoring invalid LLM_PRICING_JSON: %s", exc)
    return None


def estimate_tokens(value: str) -> int:
    """A dependency-free fallback, used only when provider usage is absent."""
    return int(math.ceil(len(value or "") / 4))


class TelemetrySession:
    def __init__(self, uid: str, endpoint: str):
        self.uid, self.endpoint = uid, endpoint
        self.trace_id = f"trc_{uuid4().hex[:16]}"
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.start_time = self.timestamp
        self.started = time.perf_counter()
        self.calls: list[Dict[str, Any]] = []
        self.steps: list[Dict[str, Any]] = []
        self.failed_error: Optional[str] = None
        self.saved: Optional[Dict[str, Any]] = None

    def record_call(self, *, model: str, system: str, user: str, response: Any = None,
                    error: Optional[BaseException] = None, started: float) -> None:
        ended = time.perf_counter()
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        estimated = False
        if error:
            input_tokens = output_tokens = 0
        elif input_tokens is None or output_tokens is None:
            estimated = True
        if input_tokens is None:
            input_tokens = estimate_tokens(system + user)
        if output_tokens is None:
            text = "".join(getattr(part, "text", "") for part in getattr(response, "content", []) or [])
            output_tokens = estimate_tokens(text)
        price = _pricing(model)
        cost = None if not price else (
            input_tokens / 1_000_000 * float(price["input_per_1m"])
            + output_tokens / 1_000_000 * float(price["output_per_1m"])
        )
        self.calls.append({
            "model_name": model, "input_tokens": int(input_tokens), "output_tokens": int(output_tokens),
            "total_tokens": int(input_tokens) + int(output_tokens), "tokens_estimated": estimated,
            "estimated_cost": cost, "cost_estimated": True, "latency_ms": round((ended - started) * 1000),
            "status": "failed" if error else "success", "error_type": type(error).__name__ if error else None,
            "processing_type": "LLM Processing", "step": "Model API call",
        })

    def record_step(self, *, name: str, processing_type: str, started: float,
                    llm_duration_ms: int = 0, error: Optional[BaseException] = None) -> None:
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        self.steps.append({
            "name": name, "processing_type": processing_type,
            "duration_ms": max(0, elapsed_ms - llm_duration_ms),
            "elapsed_ms": elapsed_ms,
            "status": "failed" if error else "success",
            "error_type": type(error).__name__ if error else None,
        })

    def persist(self, error: Optional[BaseException] = None) -> Dict[str, Any]:
        if error:
            self.failed_error = type(error).__name__
        calls = self.calls
        ended_at = datetime.now(timezone.utc).isoformat()
        duration_ms = round((time.perf_counter() - self.started) * 1000)
        llm_steps = [*calls]
        non_llm_steps = self.steps
        errors = [
            {"step": item.get("step", item.get("name")), "processing_type": item["processing_type"],
             "error_type": item["error_type"]}
            for item in [*llm_steps, *non_llm_steps] if item.get("error_type")
        ]
        if error and not errors:
            errors.append({"step": "Request", "processing_type": "Non-LLM Processing", "error_type": type(error).__name__})
        doc = {
            "trace_id": self.trace_id, "timestamp": self.timestamp, "endpoint": self.endpoint, "uid": self.uid,
            "start_time": self.start_time, "end_time": ended_at, "duration_ms": duration_ms,
            "model_name": calls[-1]["model_name"] if calls else None,
            "model_names": sorted({call["model_name"] for call in calls}), "model_calls": len(calls),
            "input_tokens": sum(c["input_tokens"] for c in calls), "output_tokens": sum(c["output_tokens"] for c in calls),
            "total_tokens": sum(c["total_tokens"] for c in calls),
            "estimated_cost": round(sum(c["estimated_cost"] or 0 for c in calls), 8),
            "cost_estimated": True, "tokens_estimated": any(c["tokens_estimated"] for c in calls),
            "latency_ms": duration_ms,
            "llm_latency_ms": sum(c["latency_ms"] for c in calls), "status": "failed" if error else "success",
            "error_type": self.failed_error, "errors": errors, "calls": calls,
            "steps": [*non_llm_steps, *llm_steps],
            "processing": {
                "llm": {"label": "LLM Processing", "step_count": len(llm_steps),
                        "duration_ms": sum(item["latency_ms"] for item in llm_steps)},
                "non_llm": {"label": "Non-LLM Processing", "step_count": len(non_llm_steps),
                            "duration_ms": sum(item["duration_ms"] for item in non_llm_steps)},
            },
        }
        doc["prompt_tokens"] = doc["input_tokens"]
        doc["completion_tokens"] = doc["output_tokens"]
        try:
            self.saved = TelemetryRepository().create(self.uid, doc)
        except Exception as exc:  # telemetry must never change the business response
            log.warning("Could not persist telemetry trace %s: %s", self.trace_id, exc)
            self.saved = doc
        log.info("Telemetry trace=%s endpoint=%s status=%s duration_ms=%s llm_ms=%s non_llm_ms=%s model_calls=%s tokens=%s estimated_cost=%s errors=%s",
                 self.trace_id, self.endpoint, doc["status"], doc["duration_ms"], doc["llm_latency_ms"],
                 doc["processing"]["non_llm"]["duration_ms"], doc["model_calls"], doc["total_tokens"],
                 doc["estimated_cost"], [item["error_type"] for item in errors])
        return self.saved


@contextmanager
def request_telemetry(uid: str, endpoint: str) -> Iterator[TelemetrySession]:
    session = TelemetrySession(uid, endpoint)
    token = _current.set(session)
    try:
        yield session
    except BaseException as exc:
        session.persist(exc)
        raise
    else:
        session.persist()
    finally:
        _current.reset(token)


def track_llm_call(*, model: str, system: str, user: str, response: Any = None,
                   error: Optional[BaseException] = None, started: float) -> None:
    session = _current.get()
    if session:
        session.record_call(model=model, system=system, user=user, response=response, error=error, started=started)


@contextmanager
def track_processing_step(name: str, processing_type: str = "Non-LLM Processing") -> Iterator[None]:
    """Measure a workflow step when a request telemetry session is active."""
    started = time.perf_counter()
    session_at_start = _current.get()
    call_index = len(session_at_start.calls) if session_at_start else 0

    def record(error: Optional[BaseException] = None) -> None:
        session = _current.get()
        if session:
            nested_llm_ms = sum(call["latency_ms"] for call in session.calls[call_index:])
            session.record_step(name=name, processing_type=processing_type, started=started,
                                llm_duration_ms=nested_llm_ms, error=error)

    try:
        yield
    except BaseException as exc:
        record(exc)
        raise
    else:
        record()
