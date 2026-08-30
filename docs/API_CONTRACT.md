# API contract

Base URL: `http://localhost:8000` · Interactive docs: `/docs` · OpenAPI: `/openapi.json`

All endpoints except `/api/health`, `/api/system/status`, `/api/auth/config` and
`/api/auth/demo-login` require:

```
Authorization: Bearer <firebase-id-token | demo-token>
```

Errors are `{"detail": "<human-readable message>"}` with the HTTP status:
`401` not signed in · `403` role not permitted · `409` no dataset uploaded ·
`422` bad input or unusable data · `404` not found.

---

## System

### `GET /api/health`
```json
{ "status": "ok", "service": "BusinessIntelligence.ai API" }
```

### `GET /api/system/status`
Reports what this deployment is actually running — used by the Settings page.
```json
{
  "app":      { "env": "development", "pipeline": ["observe","investigate","contest","act"] },
  "database": { "backend": "json", "reachable": true, "atlas_ready": false },
  "auth":     { "mode": "demo", "firebase_project": null },
  "llm":      { "provider": "anthropic", "model": null, "enabled": false,
                "mode": "deterministic_fallback", "note": "…" },
  "analysis": { "anomaly_z_threshold": 2.0, "min_material_change_pct": 3.0,
                "structured_engine": "pandas / NumPy (deterministic)",
                "retrieval_engine": "BM25 over per-user document chunks" }
}
```

---

## Authentication & roles

### `GET /api/auth/config` — public
Tells the frontend which authentication path is active.
```json
{
  "mode": "demo",
  "firebase_configured": false,
  "demo_login_enabled": true,
  "roles": [ { "key": "business_leader", "label": "Business Leader", "description": "…" },
             { "key": "data_analyst",    "label": "Data Analyst",    "description": "…" } ],
  "personas": [ { "key": "business_analyst", "label": "Business Analyst",
                  "description": "…", "action_horizon": "analysis_cycle" }, … ],
  "notice": "Running in demo auth mode: … signatures are not verified."
}
```

### Roles and personas are different things

A **role** is authorisation: which fields the server is willing to send at all. There are two
(`data_analyst`, `business_leader`) and they drive server-side redaction.

A **persona** is presentation: how findings are framed and what the reader is advised to do. There
are five (`business_analyst`, `business_manager`, `business_leader`, `domain_specialist`,
`operational_user`), any user may choose any of them, and choosing one grants nothing.

`PATCH /api/auth/persona` with `{ "persona": "operational_user" }` changes it; the persona also
defaults sensibly from the role for a user who never picks one, and an unknown persona falls back to
neutral analyst framing rather than a guess.

**The invariant:** persona changes the explanation and the recommendations. It never changes the
evidence, the hypothesis ranking, the confidence scores or the causal verdicts — those are computed
once, before any persona is consulted, and are returned identically to every reader in
`act.persona_view.recommendation_basis`. A recommendation that does not trace back to a cause metric
in that basis is dropped rather than shown.

### `POST /api/auth/demo-login` — only when `mode = demo`
```json
// request
{ "email": "leader@acme.com", "display_name": "Priya", "role": "business_leader" }
// response
{ "token": "demo.eyJ1a…", "user": { … }, "mode": "demo" }
```

### `POST /api/auth/register` — after Firebase sign-up
```json
// request
{ "role": "data_analyst", "display_name": "Priya", "organisation": "Acme" }
```

### `GET /api/auth/me`
```json
{
  "uid": "…", "email": "…", "display_name": "…", "role": "data_analyst",
  "organisation": "Acme", "token_verified": true,
  "permissions": {
    "view_dashboard": true, "run_investigation": true,
    "view_statistical_detail": true, "view_evidence_ledger": true,
    "view_reasoning_trail": true, "view_raw_driver_tables": true,
    "search_documents": true, "manage_data": true
  }
}
```

### `PATCH /api/auth/role`
`{ "role": "business_leader" }` → the updated user object.

---

## Structured data

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/datasets` | List this user's datasets |
| `POST` | `/api/datasets` | Upload a CSV (multipart, field `file`); becomes active |
| `POST` | `/api/datasets/load-sample` | Load the bundled sample company |
| `GET` | `/api/datasets/template` | Download the CSV template |
| `GET` | `/api/datasets/active` | Active dataset + detected schema + available timeframes |
| `POST` | `/api/datasets/{id}/activate` | Make a dataset active |
| `DELETE` | `/api/datasets/{id}` | Delete a dataset |

### `GET /api/datasets/active`
```json
{
  "id": "ds_…", "filename": "business_metrics_sample.csv", "is_active": true,
  "timeframes": [ { "year": 2023, "quarter": 1, "label": "2023-Q1", "rows": 624,
                    "start": "2023-01-02", "end": "2023-03-27" }, … ],
  "schema": {
    "date_column": "date",
    "dimensions": ["region","product","channel","segment"],
    "base_metrics": ["revenue","units_sold", …],
    "available_kpis": ["revenue","orders","customers", …],
    "kpi_catalogue": [ { "key": "revenue", "label": "Revenue", "unit": "currency",
                         "higher_is_better": true, "description": "…" }, … ],
    "row_count": 8784, "grain": "weekly",
    "date_min": "2023-01-02", "date_max": "2026-06-29",
    "quarters": ["2023-Q1", …], "years": [2023,2024,2025,2026],
    "warnings": []
  }
}
```

---

## Documents (RAG corpus)

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/documents` | List documents + chunk counts |
| `POST` | `/api/documents` | Upload documents (multipart, repeated field `files`) |
| `POST` | `/api/documents/load-samples` | Load the bundled document corpus |
| `DELETE` | `/api/documents/{id}` | Delete a document and its chunks |
| `POST` | `/api/documents/search` | **Analyst only** — inspect retrieval for a query |

### `POST /api/documents/search` → `403` for non-analysts
```json
// request
{ "query": "supply disruption stockout allocation", "top_k": 5 }
// response
{
  "query": "…", "indexed_chunks": 21,
  "method": "BM25 lexical retrieval over per-user document chunks",
  "results": [ { "chunk_id": "doc_…_c0002", "document_name": "ops_supply_incident_report_2026.md",
                 "heading": "1. Incident summary", "text": "…",
                 "score": 7.77, "relevance": 1.0, "strength": 0.863 } ]
}
```

---

## Analysis

**A9 retired the fixed 4-stage pipeline** (`observe → investigate → contest → act`) and the six
endpoints built around it: the four stage endpoints (`/observe`, `/investigate`, `/contest`,
`/act`), `/api/investigations/run`, and `/api/questions/investigate`. `POST /api/questions/ask`
— the agent loop — is now the only way this API answers a business question. What survives
alongside it: `/api/dashboard` (still the `observe` engine, for the KPI-driven dashboard view),
`/api/questions/interpret` (question grounding alone, with nothing run), and `/api/investigations`
(now a history of saved agent answers, not saved pipeline runs).

| Method | Path | Returns |
|---|---|---|
| `GET` | `/api/dashboard?year=&quarter=&kpi=&comparison=` | Everything the dashboard needs |
| `GET` | `/api/meta/timeframes` | Available periods + KPI catalogue |
| `POST` | `/api/questions/interpret` | How a question is read, without running it |
| `POST` | `/api/questions/ask` | Ask a business question of the agent loop — prose answer + full evidence trail |
| `GET` | `/api/investigations` | Saved agent answers, newest first |
| `GET` | `/api/investigations/{id}` | A saved agent answer, unredacted |
| `DELETE` | `/api/investigations/{id}` | Delete one |

### `POST /api/questions/interpret`

Reads a question against the dataset's KPI contract — which KPI, which period — without running
anything. Cheap by design.

```
POST /api/questions/interpret
{ "question": "Why did profit fall in Q4 even though revenue held?",
  "dataset_id": null, "use_llm": true }
```

```json
{ "intent": { "outcome": { "kpi_key": "gross_profit", "label": "Gross profit" }, "...": "…" },
  "blocked": false, "assumptions": [] }
```

**A model can never introduce a KPI.** Any key it returns is validated against the dataset's
resolver; an unrecognised one is discarded rather than believed.

### Agentic question answering — `POST /api/questions/ask`

A model receives the question and all 56 analysis tools, decides which to call and in what order,
and writes the answer itself. There are no narrative templates behind it — **the prose in `answer`
is the entire answer**, and `evidence` is the complete list of tool calls that produced it.

```
POST /api/questions/ask
{ "question": "What was revenue in all odd-numbered years?", "dataset_id": null, "persist": false }
```

No `persona` or `use_llm`: the loop has no persona seam and no deterministic fallback.
`persist` (default `false`) saves the answer — see below.

```json
{
  "status": "ok",
  "question": "What was revenue in all odd-numbered years?",
  "answer": "Revenue across 2023 and 2025 — the odd-numbered years your data covers — was £31.63m …",
  "evidence": [
    { "step": 1, "tool": "query_kpi",
      "args": { "kpi_keys": "revenue", "time_filter": { "type": "years", "values": [2023, 2025] } },
      "result": { "rows": [ … ], "total": 31631471.07 },
      "is_error": false }
  ],
  "kpis_used": ["revenue"],
  "periods_used": ["2023,2025"],
  "engine": { "turns": 2, "model": "claude-opus-5", "seconds": 3.1 },
  "dataset": { "id": "ds_…", "filename": "…" },
  "view": { "role": "business_leader", "analyst_detail_included": false, "redaction": "none" },
  "telemetry": { … },
  "investigation_id": null
}
```

**Every ending of the loop is HTTP 200 with a typed `status`.** Check `status` before reading
`answer`:

| `status` | What happened | What to render |
|---|---|---|
| `ok` | The model finished and wrote an answer | `answer` + the evidence panel |
| `max_turns_exhausted` | It ran out of turns before concluding | The partial trail, and that it did not converge |
| `truncated` | The final turn hit the output-token ceiling | The partial answer, marked incomplete |
| `refused` | A safety classifier declined the question | The refusal; do not retry unchanged |
| `llm_required` | No model is configured on this deployment | A configuration message. `evidence` is empty and **no dataset work was done** |

**Every role receives the complete evidence trail.** `redact.py` is not applied here, and
`view.redaction` is always `"none"` to say so. `view.analyst_detail_included` keeps the meaning it
has everywhere else — whether the caller holds the Data Analyst role — so a client can read it
uniformly; it simply does not govern anything on this endpoint. Showing how an answer was reached
is the product here, not an analyst privilege.

**`persist: true` saves the answer**, only when `status` is `"ok"` — a truncated or refused run has
no durable answer worth keeping. `investigation_id` is set on the response and the row appears
under `GET /api/investigations`; without it, nothing is written.

**This endpoint declares a response model**, unlike the retired stage endpoints, for the reason
given under `## KPI contract` below: the shape is already frozen and is the artefact a client
depends on, so it belongs in `/openapi.json`. The five `status` values are published as an enum
there.

| Status | When |
|---|---|
| `409` | No dataset named and none active |
| `422` | The dataset cannot be loaded, or `question` is empty or over 500 characters |
| `503` | The reasoning provider is unavailable (timeout, rate limit, auth) |

A question the loop could not answer is **never** one of these — it is a 200 with a `status`.

### `GET /api/investigations` / `GET /api/investigations/{id}`

```json
{ "investigations": [
  { "id": "inv_…", "legacy_format": false,
    "question": "What was revenue in all odd-numbered years?",
    "answer_preview": "Revenue across 2023 and 2025 — the odd-numbered years your data covers …",
    "kpis_used": ["revenue"], "turns": 2, "created_at": "2026-08-30T10:00:00+00:00" }
] }
```

`GET /api/investigations/{id}` returns the full saved `/questions/ask` response body, unredacted
for every role — the same `redaction: "none"` reasoning as the live endpoint. A row saved before
A9 (a run of the retired pipeline) has no `answer` field; it is reported as
`{"status": "legacy_format", "id", "question", "created_at"}` rather than rendered.

---

## KPI contract

The authoritative KPI definitions for a dataset. Reads are open to any signed-in
user; **every write requires the Data Analyst role** (`manage_kpi_contract`).

Unlike the analysis endpoints, these declare real response models — the contract
is the artefact downstream systems depend on, so its shape is in `/openapi.json`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/kpi/contract` | The contract being edited (draft if one exists, else live) |
| `POST` | `/api/kpi/contract/discover` | Profile the data and propose KPIs, as a new draft |
| `GET` | `/api/kpi/contract/proposals` | Proposals grouped by origin, plus conflicts |
| `POST` | `/api/kpi/contract/kpis` | Define a KPI yourself |
| `PATCH` | `/api/kpi/contract/kpis/{id}` | Override any part of a definition |
| `DELETE` | `/api/kpi/contract/kpis/{id}` | Remove a KPI |
| `POST` | `/api/kpi/contract/kpis/{id}/preview` | Compute it over recent periods without approving |
| `POST` | `/api/kpi/contract/kpis/{id}/approve` · `/reject` | Per-KPI decision |
| `POST` | `/api/kpi/contract/conflicts/{id}/resolve` | Record a human decision on an ambiguity |
| `POST` | `/api/kpi/contract/approve` | Make the contract authoritative (version++, goes live) |
| `GET` | `/api/kpi/contract/versions` | Audit history |
| `GET` | `/api/kpi/library` | Library entries, and which bound to this dataset |

All accept an optional `?dataset_id=` and default to the active dataset.

**Statuses.** A contract is `draft` (under review), `provisional` (generated
automatically, never reviewed), `approved` (authoritative) or `superseded`.
A draft is deliberately **not live**: approving one KPI inside it does not change
what the dashboard shows. The draft goes live only when the contract as a whole
is approved.

**Refusals.** `409` when a rule blocks the action — a KPI with an unresolved
blocking conflict, a KPI whose granularity was inferred but never confirmed, or a
contract with any blocking conflict outstanding. `422` for a malformed formula.

### `GET /api/kpi/contract/proposals`

```json
{
  "summary": { "version": 2, "status": "draft", "kpi_count": 25,
               "counts_by_status": {"proposed": 12, "needs_confirmation": 13},
               "blocking_conflicts": 0, "approvable": true,
               "detected_domains": ["general", "retail_ecommerce"] },
  "groups": {
    "general_library": [ { "kpi_id": "gross_margin_pct", "name": "Gross margin %",
      "kpi_type": "general", "status": "needs_confirmation",
      "business_definition": "Gross profit as a share of revenue.",
      "formula": { "kind": "ratio", "numerator_expression": "{revenue} - {cost_of_goods}",
                   "denominator_expression": "{revenue}", "scale": 100.0 },
      "unit": "percent", "higher_is_better": true,
      "source_fields": ["revenue", "cost_of_goods"],
      "granularity": { "entity_grain": [], "time_grain": "week",
                       "native_row_grain": ["date","region","product","channel","segment"],
                       "valid_rollups": ["week","month","quarter","year"],
                       "declared_by": "inferred", "requires_confirmation": true },
      "aggregation": { "method": "ratio", "rollup_policy": "recompute_from_components" },
      "time_semantics": { "date_field": "date", "calendar_id": "gregorian",
                          "comparison_default": "previous_period" },
      "sources": [ { "dataset_id": "ds_…", "fields": ["revenue","cost_of_goods"],
                     "native_grain": [...], "refresh_cadence": "weekly" } ],
      "relevance": "Separates a revenue problem from a pricing or cost problem.",
      "semantic_tags": ["profitability","efficiency"],
      "comparability": [ { "kpi_id": "orders", "comparable": true, "reasons": [] } ],
      "provenance": { "origin": "general_library", "derived_from": ["gross_margin_pct"],
                      "screened_by": "deterministic" },          // analyst only
      "confidence": 0.85,                                         // analyst only
      "approval": { "approved": false, "user_confirmed_granularity": false } } ],
    "discovered_atomic": [ … ], "discovered_derived": [ … ],
    "llm_suggested": [ … ],     "user_defined": [ … ]
  },
  "conflicts": [ { "conflict_id": "cf_…", "kind": "subset_check_failed",
                   "severity": "blocking", "detail": "…",
                   "affected_kpis": ["return_rate"],
                   "resolution_options": [ {"option_id": "accept", "label": "…"} ],
                   "resolved": false } ],
  "unavailable": [ { "library_id": "gross_margin_pct", "name": "Gross margin %",
                     "why_unavailable": "no field in this dataset matched the concept(s) 'revenue'" } ],
  "rejected_candidates": [ { "expression": "units_sold / orders", "rule": "outcome_rate",
                             "reason": "'units_sold' is not contained by 'orders' on the rows…" } ],
  "field_profiles": [ … ],          // analyst only
  "screened_by": "deterministic",
  "analyst_detail_included": true
}
```

`conflict.kind` ∈ `duplicate_definition` · `formula_disagreement` · `grain_mismatch` ·
`calendar_mismatch` · `hierarchy_violation` · `aggregation_ambiguity` ·
`unit_mismatch` · `source_disagreement` · `subset_check_failed` · `insufficient_history`.

### Effect on the analysis endpoints

`observe` gains two fields, and `kpi_scoreboard` entries gain the same:

```json
{ "granularity": "company-week", "contract_status": "approved" }
```

Only **approved** KPIs are authoritative. A dataset with no contract is given a
`provisional` one generated from the general library, so behaviour is unchanged
for anything uploaded before this layer existed.
