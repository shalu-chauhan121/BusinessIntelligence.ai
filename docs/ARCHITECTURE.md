# Architecture

This document explains every component, **why it exists**, and the trade-offs taken.

---

## 1. The organising principle

> The LLM proposes and reasons. The data engine calculates. RAG retrieves.
> Contest challenges. The backend orchestrates. The frontend communicates.

The failure mode this design exists to prevent is the one that makes "AI for BI" demos
untrustworthy: a language model looks at a KPI and writes a confident, fluent, unverifiable
explanation. Fluency is not evidence.

So responsibilities are split by *what kind of thing* each layer is good at:

| Layer | Produces | Never does |
|---|---|---|
| Structured analysis (pandas/NumPy) | Every business number | Interpret, or write prose |
| RAG retrieval (BM25) | Verbatim quotations with citations | Summarise or paraphrase |
| LLM reasoning (Claude) | Framing, stance classification, narrative | Produce or alter a number |
| Backend (FastAPI) | Orchestration, auth, persistence | Analysis |
| Frontend (React) | Presentation | Business arithmetic |

The consequence worth noticing: **the whole four-stage pipeline completes with no API key.**
The model improves the writing, not the conclusions. That is what makes the output auditable.

---

## 2. The four stages

They are separate because each answers a different question, and because merging them is
exactly how a plausible-but-wrong explanation survives.

```
OBSERVE        What actually changed?          deterministic statistics
   │
INVESTIGATE    What could explain it?          competing hypotheses + evidence
   │
CONTEST        What would disprove it?         adversarial checks
   │
ACT            What should we do?              recommendations + monitoring
```

### Stage 1 — OBSERVE (`app/engines/observe.py`)

**Significance.** The current period-over-period change is scored against the distribution of
the KPI's own historical changes.

* **Robust statistics, not mean and standard deviation.** Median and MAD (× 1.4826) are used
  because the anomaly we are looking for would otherwise inflate the yardstick used to detect
  it. One −17% quarter in a twelve-quarter history moves the mean and inflates the standard
  deviation enough to hide itself.
* **Seasonality by construction.** When at least three same-quarter transitions exist, the
  comparison is restricted to them (`Q1→Q2` in previous years). This absorbs seasonality
  without fitting a seasonal model to a series that is far too short to fit one to.
* **Small-sample correction.** Three same-quarter observations that happen to cluster produce
  a near-zero MAD and therefore a z-score in the tens — technically correct, and useless. Two
  corrections apply: sigma is inflated by `√(1 + 1/n)`, and the seasonal sigma is floored at
  half the KPI's all-history dispersion. The floor is reported to the user when it binds.
* **Material and unusual.** A change must clear both an absolute materiality floor and the
  statistical threshold. A 0.4% move that is statistically strange is not a business event.

**Driver decomposition.**

* Additive KPIs: `contribution_i = Δ_i / Δ_total`.
* Ratio KPIs: a rate/mix decomposition,
  `ΔR = Σ wᵢ,cur·(rᵢ,cur − rᵢ,base) + Σ rᵢ,base·(wᵢ,cur − wᵢ,base)`.
  A margin can fall because members got worse (rate) or because the blend shifted toward
  weaker members (mix). Those need different responses, so they are reported separately.
* **Over-indexing.** Contribution alone is misleading: the biggest segment contributes the
  most to any decline, by arithmetic. Each member therefore carries
  `over_index = contribution% ÷ share-of-baseline%`. Only members above 1.2 are treated as
  drivers, and only those scope the rest of the investigation. This is the difference between
  "Enterprise, 60% of the decline" (meaningless — it is 60% of the business) and "North, 65%
  of the decline at 2.2× its size" (a real finding).

### Stage 2 — INVESTIGATE (`app/engines/investigate.py`, `hypotheses.py`)

Ten hypothesis templates: supply, demand, competitive, pricing, mix, channel execution,
service quality, marketing, seasonality, data artefact.

* **Applicability gating.** A template is only offered if the dataset has the columns needed
  to test it. The system does not propose explanations it cannot examine.
* **Probes, not prose.** Each template runs measurements over the selected and baseline
  periods, scoped to the drivers Observe identified.
* **Direction contracts.** Every probe declares the direction it predicts. If the metric moves
  the other way, the evidence is automatically re-labelled *contradicting*. A hypothesis
  cannot cite a number that points against it.
* **Declared gaps.** Each template states what evidence would be needed but is absent.
* **Retrieval.** Queries are built from the hypothesis plus the driver entities Observe found
  ("North", "Product A"), so retrieval is specific rather than topical.

The LLM, when enabled, may only re-word titles and statements and suggest further evidence to
seek. It cannot add, remove or re-rank hypotheses.

### Stage 3 — CONTEST (`app/engines/contest.py`, `analysis.py`)

The differentiator. Five checks, all deterministic:

1. **Temporal precedence.** Onset detection on weekly series: the first week a series leaves
   its own robust band and stays outside it. Run for the KPI and for the proposed cause, then
   compared. *A cause cannot explain a change that started before it.* When the KPI moves
   first, confidence is capped at 55% and the reason is shown. This is the check that demotes
   the best-documented explanation in the sample scenario.
2. **Cross-sectional consistency.** Pearson *r* between per-member changes in the KPI and in
   the proposed cause. Always described as association; never as causation.
3. **Counterexample search.** Members where the KPI fell but the proposed cause did not move.
   The cleanest contradiction structured data can produce.
4. **Contradictory retrieval.** Queries written to surface passages that argue *against* the
   hypothesis, filtered by explicit contradiction markers ("does not explain", "before any",
   "unaffected", "unverified"). With an API key, the LLM re-classifies each passage as a
   skeptical reviewer, and the classifier used is shown in the UI.
5. **Reverse-causation screen.** Marketing spend budgeted as a share of revenue *falls because
   revenue fell*. Realised price is computed from the revenue being explained. Tickets per
   order rise when orders fall. These correlate near-perfectly and explain nothing. Templates
   declare the risk; a lead/lag cross-correlation over **differenced** weekly series (so a
   shared trend cannot manufacture the correlation) tests which moves first; suspected cases
   are penalised and capped at 45%.

**Confidence scoring.**

```
confidence = support / (support + against + missing_penalty + prior)
```

`support` and `against` are weighted sums of measured evidence strengths — structured probes,
document relevance, temporal verdict, cross-sectional consistency, counterexamples, mechanism
screen. The constant `prior` prevents a hypothesis with one weak piece of evidence from
scoring highly merely because nothing contradicts it yet. Bands: strong ≥ 65, moderate ≥ 45,
weak ≥ 25, otherwise insufficient.

It is labelled **evidence-based confidence** everywhere and never called a probability,
because the system does not implement a probabilistic model and should not pretend to. Where
the top two are within 12 points, the system says the evidence does not separate them.

### Stage 4 — ACT (`app/engines/act.py`)

Recommendations come from a playbook keyed on hypothesis family, scoped to the drivers
Observe found, and each carries:

* the hypothesis it rests on and that hypothesis's confidence,
* the specific evidence behind it,
* **monitoring thresholds computed from the user's own weekly history** (median ± 2 robust
  sigma) — the sample scenario's operations report notes that no stockout early-warning
  existed, and this is the concrete fix,
* **what would change this advice** — the missing evidence, the counterexamples, the timing
  question.

The narrative keeps the uncertainty. A leader who acts on false certainty is worse off than
one who acts knowing the evidence is mixed.

---

## 3. Backend

**FastAPI**, chosen for typed request/response models, automatic OpenAPI docs (useful when
frontend and backend are built in parallel), and native async.

```
app/
  main.py       application, CORS, routers
  config.py     pydantic-settings; every value has a working default
  deps.py       current_identity → current_user → require_analyst / active_dataset
  api/          routes_auth · routes_data · routes_analysis · routes_system · redact
  services/     dataset_service (load + cache) · pipeline (stage orchestration)
```

The pipeline is a **plain deterministic sequence**, not an agent loop. The order of the stages
is the product's core idea; leaving it to a model to decide would be handing away the thing
that makes the output trustworthy. Agentic tool selection could be added later where it
genuinely helps — dynamic probe selection is the obvious candidate — but it is not needed for
correctness, so it is not here.

**Per-user isolation.** Every dataset, document, chunk and investigation carries a `uid` and
every repository query filters on it. There is no cross-user read path.

**Dataset caching.** Prepared DataFrames are cached on `(path, mtime)`, so repeated analysis of
the same upload does not re-parse the CSV.

---

## 4. Storage — and swapping the database

The application only ever talks to a document-shaped interface: collections of JSON documents,
dict filters with `$in` / `$gte` / `$exists`, and repositories on top.

```
app/db/
  base.py          Collection / DocumentStore interfaces + filter evaluation
  json_store.py    file-backed, atomic writes, thread-safe   (default)
  mongo_store.py   pymongo against a local mongod or Atlas
  repositories.py  Users · Datasets · Documents · Investigations
```

**Why a JSON store by default.** A judge, a teammate, or a fresh laptop can run the project
with `pip install -r requirements.txt` and nothing else. No database server, no connection
string, no Docker.

### Swapping the database

```bash
# backend/.env
DB_BACKEND=mongo
MONGODB_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority
MONGODB_DB=businessintelligence
```

Restart the backend. No application code changes, because the interface is already
document-shaped. Confirm on the Settings page or at `GET /api/system/status`, which reports
the active backend and whether it is reachable.

---

## 5. Authentication and authorisation

**Authentication is Firebase.** The browser signs in with the Firebase JS SDK (email/password
or Google) and sends the ID token as a bearer token. The backend verifies it with
`firebase-admin`.

**Demo mode.** With no service-account credentials the backend cannot verify signatures, so it
says so — `/api/auth/config` returns a visible notice, and the UI displays it. In that mode a
local `/api/auth/demo-login` endpoint is enabled so the product can be run and reviewed before
a Firebase project exists. Real Firebase tokens are still accepted, decoded but marked
`verified: false`. The mode is never silently upgraded, and demo tokens are rejected outright
once verification is available.

**Authorisation is ours.** Firebase says *who* you are; the role in our own user store says
what you may see. `app/api/redact.py` removes analyst-only fields from the response for other
roles:

| Field | Leader | Analyst |
|---|---|---|
| Headline change, verdict, drivers, hypotheses, evidence, reasoning trail, recommendations | ✅ | ✅ |
| Robust z, sigma, historical change distribution, method name | plain-language sentence only | ✅ |
| Full driver tables per dimension | top drivers only | ✅ |
| Evidence ledger behind each confidence score | ❌ | ✅ |
| Per-member consistency tables, weekly onset series | ❌ | ✅ |
| Retrieval inspector | ❌ | ✅ |

Doing this on the server is what makes it authorisation rather than presentation.

---

## 6. RAG

```
app/rag/
  extract.py    pdf / md / txt / csv → text, with document-type classification
  chunker.py    heading-aware chunking, ~170 words with 40-word overlap
  index.py      BM25 (k1 = 1.5, b = 0.75), per user, in-process
  retriever.py  the only interface the engines use
```

**Why BM25 rather than a vector database.** No embedding API key, no service to run, no cold
start — so the demo is reproducible anywhere. And business evidence retrieval is heavily
entity-driven ("North", "Product A", "stockout", the competitor's name), which is exactly
where exact-term matching is strong.

**Absolute strength, not relative relevance.** Relevance normalised to the best hit for a query
is always 1.0 for *something* — so a hypothesis with no relevant document would appear to have
a perfectly relevant one. Chunks therefore carry a saturating absolute score, and passages
below an absolute floor are dropped entirely.

**Markdown cleanup.** Chunks are cleaned before being quoted (table pipes, heading hashes,
emphasis markers, blockquote arrows), because evidence is quoted verbatim in the UI and raw
markdown reads as source code rather than as something a person wrote.

**Upgrade path.** `Retriever` is the only class the engines import. Chroma (simple) or
PostgreSQL + pgvector (integrated) can replace `index.py` behind the same two methods.

---

## 7. LLM layer

`app/llm/` — one model, three roles, three prompts.

| Role | Given | Returns |
|---|---|---|
| **Investigate** | Facts already computed, hypotheses already generated | Better titles and statements, further evidence to seek |
| **Contest** | A hypothesis and retrieved passages | A stance verdict per passage, as a skeptical reviewer |
| **Act** | The completed investigation | The leader narrative |

Every prompt carries the same guardrail block: do not compute, estimate or invent any figure;
do not claim causation; do not resolve genuine ambiguity by picking a side; never describe
confidence scores as probabilities; return JSON only.

The model is only ever handed a **fact sheet** of values the analysis layer already computed —
it is never given the raw dataset, so it has nothing to compute from even if it tried.

Every call is wrapped: a failure degrades to the deterministic output and records why, and the
UI states which mode produced what you are reading.

---

## 8. Frontend

**Vite + React + Tailwind**, matching the intended stack, with Recharts for charts.

* **Responsive.** Single-column on mobile, sidebar from `lg`, charts fluid, tables scroll.
* **Theming.** Design tokens as CSS custom properties; SVG resolves them natively, so light
  and dark themes swap in one place.
* **Charts follow a colour method rather than a taste.** Categorical hues are assigned in a
  fixed, colour-vision-deficiency-checked order and never cycled; the driver chart uses a
  diverging pair around zero because its job is polarity; status colours are reserved and never
  reused as a series; every chart has a table view so identity is never carried by colour alone.
* **No dual-axis charts.** The onset chart has two measures on wildly different scales — a
  revenue series and a stockout rate. Rather than two y-axes, both are standardised against
  their own pre-period normal and plotted in robust sigma units. One axis, both readable, and
  the units match the onset test that produced the markers.
* **Loading, empty and error states** everywhere, including the two that matter: no dataset
  uploaded, and the API not running.

---

## 9. Trade-offs taken

| Decision | Alternative | Why |
|---|---|---|
| Deterministic four-stage pipeline | Autonomous agents | The stage order *is* the product idea. Reliability first; agentic tool selection can come later |
| BM25 retrieval | Embeddings + vector DB | No key, no service, no cold start; entity-driven queries suit lexical search. Interface allows the swap |
| JSON document store default | Require MongoDB | Anyone can run the project immediately; Atlas is one env var away |
| Evidence-strength score | Probabilistic model | We do not implement one, so calling it a probability would be dishonest |
| Hypothesis library + probes | Free-form LLM hypotheses | Every hypothesis must be *testable against the uploaded data* |
| Server-side role redaction | Hide in the UI | Hiding in the UI is not authorisation |
| Robust (median/MAD) statistics | Mean/σ, or a fitted seasonal model | Outlier-resistant, and honest about 12 quarters of history |
| Confidence caps | Let the score speak | A cause that started after the effect must not out-rank one that did not, whatever its documentation |

## 10. What would come next

1. Vector retrieval alongside BM25 (hybrid), so paraphrased evidence is found too.
2. Scheduled monitoring: run Observe nightly and alert when a KPI leaves its band — the
   thresholds are already computed.
3. Multi-KPI investigations: explain revenue and margin together, with shared drivers.
4. Warehouse connectors (BigQuery, Snowflake, Postgres) behind the same dataset interface.
5. Analyst feedback captured on hypotheses, to tune the evidence weights on real judgements.
