# Observability audit

Audit date: 2026-08-23

## Scope and method

The complete repository inventory was scanned, followed by a content scan for telemetry, logging, monitoring, tracing, analytics, middleware, timing, token/cost accounting and LLM client calls. Source, tests, frontend, configuration and documentation were reviewed. No external telemetry, APM, tracing or analytics SDK is installed; the only middleware is FastAPI CORS.

## Existing observability paths

| File(s) | Existing functionality |
|---|---|
| `backend/app/main.py` | Process-level Python logging configuration and CORS middleware only. |
| `backend/app/services/pipeline.py` | Existing per-stage elapsed seconds and total elapsed seconds on full investigation results. |
| `backend/app/llm/client.py` | The single Anthropic call wrapper used by all three model roles: investigation framing, contest stance classification and act narrative. |
| `backend/app/services/telemetry.py` | Uncommitted, in-progress request telemetry session: trace ID, request duration, per-call latency, provider token usage with character-count fallback, configurable per-model pricing, cost estimate, success/failure status and best-effort persistence. |
| `backend/app/db/repositories.py` | Uncommitted `TelemetryRepository`, using the existing document-store abstraction and per-user isolation. |
| `backend/app/api/routes_analysis.py` | Uncommitted telemetry scopes for investigate/contest/act/full run, telemetry summary and recent-trace endpoints. |
| `frontend/src/lib/api.js`, `frontend/src/pages/Dashboard.jsx`, `frontend/src/pages/InvestigationPage.jsx` | Uncommitted telemetry API calls, aggregate dashboard metrics and full-run request metrics. |
| `backend/tests/test_telemetry.py` | Uncommitted tests for usage capture, fallback estimation, storage failure tolerance and aggregate summary. |
| `README.md`, `docs/ARCHITECTURE.md`, `docs/API_CONTRACT.md`, `SettingsPage.jsx`, `InvestigationPage.jsx` | Clear conceptual separation: deterministic analysis and RAG versus LLM reasoning. The UI has an LLM-enabled badge and an engine footer, but no structured per-step breakdown. |

## Feature assessment

### 1. LLM versus non-LLM processing

Partly present only as architecture documentation, LLM status, and a stage-level UI badge. The persisted telemetry has per-call data but does not classify workflow steps, calculate class totals, emit a structured classification to logs, or show a request's LLM/non-LLM breakdown in the UI.

### 2. Runtime telemetry

Substantially present in the uncommitted implementation. It records request latency, model calls, provider prompt/completion/total token usage (with fallback), estimated cost, and success/failure. It exposes aggregate metrics and a request card in the UI.

Confirmed gaps:

* explicit request start and end timestamps are absent;
* stages/other non-LLM work are not individually measured or classified;
* the full-run response has telemetry, but individual stage responses do not;
* the UI omits model name, prompt tokens, completion tokens and an explicit failure/error field;
* an LLM failure is recorded as a call failure but a recovered request is recorded as successful, which is useful but needs to be clearly surfaced as a call-level error;
* telemetry currently begins after some setup work in analysis routes, so its `latency_ms` should be described as measured workflow duration rather than full HTTP lifecycle duration.

## Duplicate implementations to avoid

Do not add an APM/analytics dependency, a second Anthropic wrapper, a second persistence collection, or broad HTTP middleware. Extend `TelemetrySession`, call it from the existing `LLMClient._call`, and measure the existing pipeline stages. Continue using `TelemetryRepository`, the existing JSON/Mongo store interface and frontend API utility.

## Validation baseline

* Frontend build cannot run because local `node_modules` are absent (`vite` is not found); no dependency installation was performed.
* Backend tests could not create their temporary upload directory under the sandboxed Windows Temp location. This is an environment permission limitation, not an assertion failure; it will be retried with the approved execution context after implementation.
