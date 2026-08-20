# Task board

The seed brief asks for 40–60 atomic tasks rather than vague ones like "build RAG". This is
the board this prototype was actually built from: 58 tasks, each independently executable,
testable, and small enough for one developer.

Columns: **Backlog → To do → In progress → Review / testing → Done.**
Status here reflects the submitted prototype.

---

## Phase 0 — Foundation

| # | Task | Status |
|---|---|---|
| 1 | Create the GitHub repository and directory structure | ✅ |
| 2 | Scaffold the FastAPI application with health and CORS | ✅ |
| 3 | Environment configuration with working defaults (`config.py`, `.env.example`) | ✅ |
| 4 | Define the document-store interface (`Collection`, `DocumentStore`) | ✅ |
| 5 | Implement the file-backed JSON store with atomic writes | ✅ |
| 6 | Implement the MongoDB/Atlas store behind the same interface | ✅ |
| 7 | Write the repositories: users, datasets, documents, investigations | ✅ |
| 8 | Scaffold Vite + React + Tailwind with design tokens | ✅ |
| 9 | Write the API client with a single bearer-token attach point | ✅ |
| 10 | Fix the four stage API contracts before writing any engine | ✅ |

## Phase 1 — Auth and roles

| # | Task | Status |
|---|---|---|
| 11 | Verify Firebase ID tokens with `firebase-admin` | ✅ |
| 12 | Add demo auth mode + local login for running without Firebase | ✅ |
| 13 | FastAPI dependencies: identity → user → analyst gate | ✅ |
| 14 | Role selection (analyst / leader) at sign-up; persist the profile | ✅ |
| 15 | Permission matrix per role, returned by `/api/auth/me` | ✅ |
| 16 | Server-side response redaction for non-analyst roles | ✅ |
| 17 | Sign-in / sign-up page with the role picker | ✅ |
| 18 | Auth context: session, profile, role change, sign-out | ✅ |

## Phase 2 — Data ingestion

| # | Task | Status |
|---|---|---|
| 19 | Detect the schema of an uploaded CSV (date, dimensions, metrics) | ✅ |
| 20 | Build the metric registry with per-metric aggregation rules | ✅ |
| 21 | Store an upload per user; activate/deactivate/delete datasets | ✅ |
| 22 | Cache prepared DataFrames on path + mtime | ✅ |
| 23 | Generate the demo dataset with a planted, documented scenario | ✅ |
| 24 | Write the CSV template and the data-format reference | ✅ |
| 25 | Data page: drag-drop upload, format reference, sample loader | ✅ |
| 26 | Report data-quality warnings (short history, no dimensions) | ✅ |

## Phase 3 — Observe slice

| # | Task | Status |
|---|---|---|
| 27 | Timeframe model: year/quarter, previous period, year-ago | ✅ |
| 28 | Quarterly and weekly KPI series | ✅ |
| 29 | Robust z significance against the KPI's own history | ✅ |
| 30 | Seasonal comparison using the same quarter transition | ✅ |
| 31 | Small-sample sigma inflation and dispersion floor | ✅ |
| 32 | Driver decomposition for additive KPIs | ✅ |
| 33 | Rate/mix decomposition for ratio KPIs | ✅ |
| 34 | Over-index scoring to separate drivers from size effects | ✅ |
| 35 | Dashboard: timeframe picker, KPI tiles, trend chart with normal band | ✅ |
| 36 | Driver visualisation with a diverging scale | ✅ |

## Phase 4 — RAG slice

| # | Task | Status |
|---|---|---|
| 37 | Extract text from pdf / md / txt / csv | ✅ |
| 38 | Heading-aware chunking with overlap | ✅ |
| 39 | BM25 index built per user | ✅ |
| 40 | Absolute retrieval strength so weak matches cannot pose as evidence | ✅ |
| 41 | Clean markdown out of quoted passages | ✅ |
| 42 | Documents page: upload, list, delete, sample loader | ✅ |
| 43 | Analyst-only retrieval inspector | ✅ |

## Phase 5 — Investigate slice

| # | Task | Status |
|---|---|---|
| 44 | Hypothesis JSON schema and evidence item schema | ✅ |
| 45 | Hypothesis template library with applicability gating | ✅ |
| 46 | Structured probes per template, scoped to the drivers | ✅ |
| 47 | Direction contracts: auto-relabel evidence that points the other way | ✅ |
| 48 | Attach retrieved documentary evidence per hypothesis | ✅ |
| 49 | Hypothesis cards and evidence panels in the UI | ✅ |

## Phase 6 — Contest slice

| # | Task | Status |
|---|---|---|
| 50 | Weekly onset detection and temporal-precedence comparison | ✅ |
| 51 | Cross-sectional consistency and counterexample search | ✅ |
| 52 | Contradictory retrieval with contradiction markers | ✅ |
| 53 | Reverse-causation screen with differenced lead/lag correlation | ✅ |
| 54 | Evidence-based confidence scoring, ledger and caps | ✅ |
| 55 | Contest UI: ranking, contradictions, onset chart, reasoning trail | ✅ |

## Phase 7 — Act slice and polish

| # | Task | Status |
|---|---|---|
| 56 | Recommendation playbook with monitoring thresholds from history | ✅ |
| 57 | Anthropic reasoning layer with guardrails and a deterministic fallback | ✅ |
| 58 | Test suite, README, architecture, API contract, demo script | ✅ |

---

## Next up (post-submission backlog)

| Task | Why |
|---|---|
| Hybrid retrieval: BM25 + embeddings | Find paraphrased evidence lexical search misses |
| Scheduled monitoring and alerting | The thresholds are already computed; wire them to a schedule |
| Warehouse connectors (BigQuery, Snowflake, Postgres) | Remove the CSV step for real deployments |
| Multi-KPI investigations | Explain revenue and margin together with shared drivers |
| Analyst feedback on hypotheses | Tune evidence weights against real human judgements |
| Export an investigation to PDF | Circulate the finding, not a screenshot |
