"""Health and capability reporting."""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter

from ..auth.firebase_auth import auth_mode
from ..config import get_settings
from ..db.repositories import get_store
from ..llm.client import get_llm

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "service": "BusinessIntelligence.ai API"}


@router.get("/system/status")
def system_status() -> Dict[str, Any]:
    s = get_settings()
    store = get_store()
    llm = get_llm()
    return {
        "app": {"env": s.app_env, "pipeline": ["observe", "investigate", "contest", "act"]},
        "database": {"backend": store.backend_name, "reachable": store.ping(),
                     "atlas_ready": bool(s.mongodb_uri)},
        "auth": {"mode": auth_mode(), "firebase_project": s.firebase_project_id or None},
        "llm": llm.status,
        "analysis": {
            "anomaly_z_threshold": s.anomaly_z_threshold,
            "min_material_change_pct": s.min_material_change_pct,
            "max_drivers_per_dimension": s.max_drivers_per_dimension,
            "structured_engine": "pandas / NumPy (deterministic)",
            "retrieval_engine": "BM25 over per-user document chunks",
        },
    }
