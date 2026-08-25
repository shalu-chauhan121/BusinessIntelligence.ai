"""
BusinessIntelligence.ai — API entry point.

    USER -> FRONTEND -> REST API -> BACKEND
                                      |-- structured analysis (pandas/NumPy)
                                      |-- RAG retrieval (per-user documents)
                                      '-- LLM reasoning (Anthropic Claude)
                                            |
                                 OBSERVE -> INVESTIGATE -> CONTEST -> ACT
                                            |
                                        FRONTEND
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api import routes_analysis, routes_auth, routes_data, routes_kpi, routes_system
from .config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

settings = get_settings()

DESCRIPTION = """
An evidence-backed KPI storytelling and root-cause investigation API.

**The four stages**

| Stage | Question | How it is answered |
|---|---|---|
| `OBSERVE` | What actually changed? | Deterministic statistics over the uploaded dataset |
| `INVESTIGATE` | What could explain it? | Competing hypotheses, each tested with structured data + retrieved documents |
| `CONTEST` | What would disprove it? | Temporal precedence, cross-sectional consistency, counterexamples, contradictory retrieval |
| `ACT` | What should we do? | Evidence-linked recommendations with monitoring thresholds |

Numbers are never produced by the language model. The analysis layer computes,
retrieval quotes, the model reasons and writes.
"""

app = FastAPI(
    title="BusinessIntelligence.ai API",
    version="1.0.0",
    description=DESCRIPTION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_system.router)
app.include_router(routes_auth.router)
app.include_router(routes_data.router)
app.include_router(routes_kpi.router)
app.include_router(routes_analysis.router)


@app.exception_handler(ValueError)
async def value_error_handler(_: Request, exc: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/", tags=["system"])
def root():
    return {
        "service": "BusinessIntelligence.ai",
        "docs": "/docs",
        "health": "/api/health",
        "pipeline": ["observe", "investigate", "contest", "act"],
    }
