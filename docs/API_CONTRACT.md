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

Shared request body (all fields optional). **`/api/questions/ask` does not take this body** —
it has its own, narrower one; see below.

```json
{
  "kpi": "revenue",                    // defaults to revenue, or the first available KPI
  "year": 2026,                        // defaults to the latest period in the data
  "quarter": 2,                        // 1-4, or null for the full year
  "comparison": "previous_period",     // or "year_over_year"
  "dataset_id": null,                  // defaults to the active dataset
  "use_llm": true,                     // false forces the deterministic reasoner
  "persist": true                      // save to investigation history
}
```

| Method | Path | Returns |
|---|---|---|
| `GET` | `/api/dashboard?year=&quarter=&kpi=&comparison=` | Everything the dashboard needs |
| `GET` | `/api/meta/timeframes` | Available periods + KPI catalogue |
| `POST` | `/api/observe` | Stage 1 only — KPI or `question` |
| `POST` | `/api/investigate` | Stages 1–2 — KPI or `question` |
| `POST` | `/api/contest` | Stages 1–3 — KPI or `question` |
| `POST` | `/api/act` | Stage 4 (runs 1–3 internally) — KPI or `question`, reframed for `persona` |
| `POST` | `/api/investigations/run` | All four stages for an explicitly named KPI, persisted |
| `POST` | `/api/questions/interpret` | How a question is read, without running it |
| `POST` | `/api/questions/investigate` | Ask a business question — **used by the UI** |
| `POST` | `/api/questions/ask` | Ask a business question of the agent loop — prose answer + full evidence trail |
| `GET` | `/api/investigations` | History |
| `GET` | `/api/investigations/{id}` | A saved investigation |
| `DELETE` | `/api/investigations/{id}` | Delete one |

Each stage endpoint recomputes the stages before it, so any one can be called standalone —
this is what let the frontend be built against a stable contract while the engines were still
being written.

**Every single-stage endpoint accepts either an explicit `kpi`/`year`/`quarter` (the original
contract) or a `question`** (resolved against the KPI contract exactly as `/api/questions/investigate`
resolves one). When both are given the question wins. An unresolvable question returns
`{"status": "needs_clarification", ...}` rather than the stage's normal shape — check `status` before
reading `observe`/`investigate`/`contest`/`act`. All four route through the same
`pipeline.prepare_stage_context` the full pipeline uses, so a single-stage call and
`/api/questions/investigate` never disagree about what counts as a material signal or a driver.

### Question-driven investigation

An investigation starts from a question, not a KPI selection. Which KPI, which period and which
comparison are resolved from the question against the dataset's KPI contract.

```
POST /api/questions/investigate
{ "question": "Why did profit fall in Q4 even though revenue held?",
  "dataset_id": null, "persona": null, "use_llm": true, "persist": true }
```

Two shapes come back. A resolved question returns the usual
`{observe, investigate, contest, act, engine}` envelope plus `question`, `intent`, `comparisons`
and `assumptions`, with `status: "ok"`.

A question that cannot be resolved returns **HTTP 200** with:

```json
{
  "status": "needs_clarification",
  "intent": { "...": "the partial reading" },
  "ambiguities": [
    { "kind": "outcome_unresolved", "blocking": true,
      "message": "No KPI in this dataset matches what the question is asking about. This dataset measures: Admissions, Recovery rate, ...",
      "candidates": [ { "kpi_key": "recovery_rate", "label": "Recovery rate" } ] }
  ]
}
```

This is a normal branch, not an error. `blocking` draws one line, and it is between *nothing* and
*several*:

* **Nothing matched** (`outcome_unresolved`), or a period the dataset does not hold
  (`unsupported_by_dataset`) — the run stops and asks. There is no answer to give, and investigating
  the nearest KPI produces a confident answer to a question the user did not ask.
* **Several matched** (`outcome_multiple`) — the run proceeds. Every candidate is a KPI the dataset
  genuinely measures, so the best-scoring one is used, `assumed` names the key that was taken,
  `candidates` lists the others, and the disclosure appears in `assumptions`. When a model is
  available it breaks the tie, constrained to the tied candidates. **This no longer blocks.**
* **A vague period** (`period_vague`, `comparison_assumed`) — proceeds on a stated default, disclosed
  the same way.

A non-blocking tie looks like this, inside the ordinary `status: "ok"` envelope:

```json
{ "kind": "outcome_multiple", "blocking": false, "assumed": "bed_occupancy_rate",
  "message": "Read as 'Bed occupancy rate'. It scored level with 'Recovery rate', so ask again naming the measure if that is not what you meant.",
  "candidates": [ { "kpi_key": "bed_occupancy_rate", "label": "Bed occupancy rate" } ] }
```

`POST /api/questions/interpret` returns `{intent, blocked, assumptions}` and runs nothing. It backs
the panel showing how the question was read, so a misreading can be corrected before an
investigation runs.

**A model can never introduce a KPI.** Any key it returns is validated against the dataset's
resolver; an unrecognised one degrades to a clarification rather than to an investigation of
something the dataset does not measure.

### Agentic question answering — `POST /api/questions/ask`

The four stages above run a fixed sequence for one KPI. This endpoint does not: a model receives
the question and all 56 analysis tools, decides which to call and in what order, and writes the
answer itself. There are no narrative templates behind it — **the prose in `answer` is the entire
answer**, and `evidence` is the complete list of tool calls that produced it.

```
POST /api/questions/ask
{ "question": "What was revenue in all odd-numbered years?", "dataset_id": null }
```

No `persona`, `use_llm` or `persist`: the loop has no persona seam, no deterministic fallback, and
saves nothing.

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
  "telemetry": { … }
}
```

**Every ending of the loop is HTTP 200 with a typed `status`**, the same way
`/questions/investigate` returns `needs_clarification` at 200. Check `status` before reading
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
is the product here, not an analyst privilege. This is the one place `## The black-box principle`
below is deliberately inverted: the box is open.

**Nothing is persisted.** An answer has no KPI, verdict or leading hypothesis — the fields a saved
investigation is shaped around — so this does not appear under `GET /api/investigations`.

**This endpoint declares a response model**, unlike its four stage neighbours, for the reason given
under `## KPI contract` below: the shape is already frozen and is the artefact a client depends on,
so it belongs in `/openapi.json`. The five `status` values are published as an enum there.

| Status | When |
|---|---|
| `409` | No dataset named and none active |
| `422` | The dataset cannot be loaded, or `question` is empty or over 500 characters |
| `503` | The reasoning provider is unavailable (timeout, rate limit, auth) |

A question the loop could not answer is **never** one of these — it is a 200 with a `status`.

### Stage 1 — `observe`

```json
{
  "kpi": "revenue", "kpi_label": "Revenue", "unit": "currency", "higher_is_better": true,
  "timeframe":          { "year": 2026, "quarter": 2, "label": "2026-Q2", "pretty": "Q2 2026" },
  "baseline_timeframe": { "year": 2026, "quarter": 1, "label": "2026-Q1", "pretty": "Q1 2026" },
  "comparison": "previous_period",
  "current_value": 3671325.4, "baseline_value": 4433063.2,
  "change_abs": -761737.8, "change_pct": -17.18,
  "direction": "down", "is_unfavourable": true,
  "anomaly": true, "verdict": "meaningful_signal",

  "significance": {
    "method": "seasonal_robust_z",          // analyst-only from here …
    "robust_z": -9.2, "median_historical_change_pct": 6.31, "robust_sigma_pct": 2.55,
    "history_points": 12, "same_quarter_points": 3,
    "z_threshold": 2.0, "material_threshold_pct": 3.0,
    "is_material": true, "is_statistically_unusual": true, "is_anomaly": true,
    "expected_value": 4712594.3, "normal_range": [4486222.1, 4938966.6],
    "historical_changes": [ { "period": "2024-Q2", "from": "2024-Q1", "change_pct": 6.31 } ],
    "statistical_power": "Good: the comparison is restricted to …",
    "dispersion_note": "The same-quarter sample was too small …",
    "explanation": "…"                      // … leader-only replacement for the above
  },

  "drivers": {                                // analyst only; {} for a leader
    "region": [ { "name": "North", "current": 780281.3, "baseline": 1271465.9,
                  "change_abs": -491184.6, "change_pct": -38.63,
                  "contribution_pct": 64.5, "share_of_current_pct": 21.3,
                  "share_of_baseline_pct": 28.7, "over_index": 2.24,
                  "effects": null } ]
  },
  "top_drivers": [ { "dimension": "product", "name": "Product A",
                     "contribution_pct": 78.3, "over_index": 1.85,
                     "is_disproportionate": true, … } ],
  "driver_concentration_pct": 64.5,

  "series": { "quarterly": [ { "period": "2023-Q1", "year": 2023, "quarter": 1, "value": … } ],
              "weekly":    [ { "week": "2025-12-29", "value": … } ] },

  "kpi_scoreboard": [ { "key": "orders", "label": "Orders", "unit": "count",
                        "higher_is_better": true, "current": 4027, "baseline": 4879,
                        "change_pct": -17.46, "is_primary": false } ],

  "rows_analysed": 624, "baseline_rows": 624, "data_warnings": []
}
```

`verdict` ∈ `meaningful_signal` · `within_normal_variation` · `statistically_unusual_but_immaterial`.

### Stage 2 — `investigate`

```json
{
  "focus": { "region": "North", "product": "Product A" },
  "focus_label": "North / Product A",
  "hypotheses": [ {
    "key": "supply_constraint",
    "title": "Supply constraint limited what could be sold",
    "family": "supply",
    "statement": "…",
    "evidence": [ { "type": "structured", "stance": "supporting",
                    "metric": "fulfillment_rate", "label": "Fulfilment rate",
                    "scope": "whole business", "baseline": 100.0, "current": 92.93,
                    "change_abs": -7.07, "change_pct": -7.07, "unit": "percent",
                    "strength": 1.0, "weight": 1.3,          // analyst only
                    "detail": "Fulfilment rate in whole business: 100.0% → 92.9% (-7.1 pts …).",
                    "note": "", "source": "Structured data analysis (uploaded dataset)" } ],
    "documentary_evidence": [ { "type": "unstructured", "stance": "supporting",
                                "source": "ops_supply_incident_report_2026.md",
                                "doc_type": "operations_report",
                                "section": "1. Incident summary",
                                "quote": "On 4 May 2026 our tier-1 component supplier …",
                                "relevance": 1.0, "strength": 0.863, "score": 7.77,
                                "chunk_id": "doc_…_c0002", "document_id": "doc_…" } ],
    "missing": [ "Supplier lead-time and on-time-in-full data is not present …" ],
    "cause_metric": "stockout_rate", "cause_direction": "up",
    "prior_support": 4.32, "prior_against": 0.0
  } ],
  "considered_count": 10,                 // analyst only
  "not_carried_forward": [ … ],           // analyst only
  "documents_indexed": 21, "rag_available": true,
  "llm_used": false, "llm_note": null,
  "method_note": "Hypotheses are generated from a library of business explanations …"
}
```

### Stage 3 — `contest`

```json
{
  "ranking": [ { "rank": 1, "key": "demand_contraction",
                 "title": "Underlying customer demand contracted",
                 "confidence": 71, "band": "strong",
                 "status": "well_supported", "causally_consistent": true } ],
  "hypotheses": [ {
    "…all Stage 2 fields…",
    "contest": {
      "temporal": {
        "status": "kpi_precedes_cause",     // cause_precedes_kpi | simultaneous | undetermined | not_applicable
        "detail": "The KPI began moving in 2026-04-06, 4 week(s) BEFORE the proposed cause moved …",
        "lag_weeks": 4,
        "kpi_onset_week": "2026-04-06", "cause_onset_week": "2026-05-04",
        "cause_metric": "stockout_rate", "cause_metric_label": "Stockout rate",
        "scope": "North / Product A"
      },
      "temporal_series": { "kpi": [ { "week": "…", "value": … } ],
                           "cause": [ … ] },              // analyst only
      "consistency": {
        "status": "checked", "dimension": "region",
        "correlation": { "r": 0.38, "r_squared": 0.144, "n": 4,
                         "interpretation": "Across 4 members … not proof of causation." },
        "counterexamples": [ { "member": "South", "kpi_change_pct": -6.1,
                               "cause_change_pct": 1.2 } ],
        "members": [ … ],                                  // analyst only
        "detail": "…"
      },
      "mechanism": {
        "declared_risk": "Marketing spend is frequently budgeted as a percentage of revenue …",
        "lead_lag": { "verdict": "simultaneous", "best": { "lag_weeks": 0, "r": 0.78, "n": 25 },
                      "r_at_lag_0": 0.78, "profile": [ … ], "detail": "…" },
        "reverse_causation_suspected": true,
        "conclusion": "Reverse causation cannot be ruled out …"
      },
      "contradictory_evidence": [ { "…document evidence…",
                                    "contrast_markers": ["does not explain","before any"],
                                    "classified_by": "lexical_contrast_heuristic" } ]
    },
    "scoring": {
      "confidence": 55, "confidence_band": "moderate", "status": "partially_supported",
      "confidence_label": "Evidence-based confidence (not a probability)",
      "support_score": 6.94, "against_score": 3.20, "missing_penalty": 0.30,  // analyst only
      "score_ledger": [ { "side": "for", "source": "structured",
                          "label": "Fulfilment rate", "value": 1.3 } ],        // analyst only
      "cap_reason": "Capped: the KPI began moving before this cause did …",
      "causal_claim": "Association only. …",
      "causally_consistent": false
    },
    "reasoning_trail": [ { "step": "Temporal precedence checked", "detail": "…" } ]
  } ],
  "ambiguity_note": "The evidence does not separate 'A' from 'B' (58% vs 56%) …",
  "unresolved_questions": [ … ],
  "confidence_disclaimer": "Confidence scores are evidence-strength scores … not probabilities …"
}
```

### Stage 4 — `act`

```json
{
  "narrative": {
    "headline": "Revenue fell -17.2% in Q2 2026 versus Q1 2026.",
    "what_changed": "Revenue moved from 4,433,063 in Q1 2026 to 3,671,325 in Q2 2026 (-17.2%).",
    "how_significant": "This is outside revenue's normal variation: the move is 9.2 robust standard deviations …",
    "what_drove_it": "The change is concentrated in Product A (product, 78% of the change, 1.8x its size) …",
    "leading_explanation": "The strongest-evidenced explanation is “…” at 71% …",
    "what_we_are_not_sure_about": [ "…" ]
  },
  "recommendations": [ {
    "id": "rec_demand_contraction",
    "title": "Address the demand shortfall directly",
    "based_on": { "hypothesis": "Underlying customer demand contracted", "key": "demand_contraction",
                  "confidence": 71, "band": "strong", "status": "well_supported" },
    "priority": "high", "horizon": "This quarter", "owner": "Sales / commercial leadership",
    "actions": [ "Run a win-back and retention programme against the accounts in North / Product A …" ],
    "rationale": "…",
    "supporting_evidence": [ "Orders in North: 1,395 → 866 (-37.9%)." ],
    "documentary_evidence": [ "customer_feedback_q2_2026.md: \"…\"" ],
    "counter_evidence": [ "…" ],
    "monitoring": [ { "metric": "orders", "label": "Orders", "unit": "count",
                      "weekly_median": 348.0, "robust_sigma": 31.13,
                      "upper_alert": 410.27, "lower_alert": 285.73,
                      "rule": "Alert when the weekly orders moves outside 285.73 – 410.27 …" } ],
    "what_would_change_this": [ "…" ]
  } ],
  "ranking": [ … ],
  "llm_story": null,                       // populated when ANTHROPIC_API_KEY is set
  "limits": [ "This analysis explains the data that was uploaded. …" ]
}
```

### `POST /api/investigations/run`

```json
{
  "investigation_id": "inv_…",
  "dataset": { "id": "ds_…", "filename": "…", "rows": 8784 },
  "observe": { … }, "investigate": { … }, "contest": { … }, "act": { … },
  "engine": {
    "llm": { "provider": "anthropic", "enabled": false, "mode": "deterministic_fallback", … },
    "stage_seconds": { "observe": 0.42, "investigate": 0.31, "contest": 0.55, "act": 0.12 },
    "total_seconds": 1.40,
    "pipeline": ["observe","investigate","contest","act"]
  },
  "telemetry": {
    "trace_id": "trc_…",
    "start_time": "2026-08-23T10:00:00+00:00",
    "end_time": "2026-08-23T10:00:01+00:00",
    "duration_ms": 1400,
    "model_name": "claude-sonnet-4-5",
    "model_calls": 3,
    "prompt_tokens": 1200, "completion_tokens": 400, "total_tokens": 1600,
    "estimated_cost": 0.0096,
    "status": "success",
    "processing": {
      "llm": { "label": "LLM Processing", "step_count": 3, "duration_ms": 800 },
      "non_llm": { "label": "Non-LLM Processing", "step_count": 4, "duration_ms": 600 }
    },
    "errors": []
  },
  "view": { "role": "data_analyst", "analyst_detail_included": true }
}
```

---

## The black-box principle

The stage contracts above were fixed before the engines were written. Any stage can be served
by mock data, by the deterministic engine, or by the full engine plus the LLM, and the
frontend cannot tell the difference. That is what allowed the frontend, backend and analysis
work to proceed in parallel — and it is why swapping BM25 for a vector index, or the JSON
store for Atlas, changes nothing above this line.

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
