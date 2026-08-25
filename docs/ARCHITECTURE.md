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

## 2. The stages

They are separate because each answers a different question, and because merging them is
exactly how a plausible-but-wrong explanation survives.

```
UNDERSTAND     What is being asked?            question -> KPI contract  (app/query/)
   │
OBSERVE        What actually changed?          deterministic statistics
   │           ── material-signal boundary ──  only real movements pass here
INVESTIGATE    What could explain it?          competing hypotheses + evidence
   │
CONTEST        What would disprove it?         adversarial checks
   │
ACT            What should we do?              recommendations + monitoring
                                               reframed per persona
```

### Stage −1 — UNDERSTAND (`app/query/`)

An investigation starts from a business question, not a KPI dropdown. Grounding is deterministic
first: KPI names, semantic tags and the concept library's field aliases resolve most questions with
no model involved. A model is consulted only when that is genuinely ambiguous, and the key it
returns is validated against the resolver — **it may choose among the KPIs that exist and cannot
introduce one.**

Where the reading is uncertain the system says so rather than guessing. An unresolvable KPI, or a
period the dataset does not hold, blocks and asks; a merely vague period proceeds on a stated
default and discloses it. Investigating the nearest KPI would produce a confident answer to a
question nobody asked.

### The material-signal boundary (`app/engines/signals.py`)

Everything a model learns about the numbers passes through `material_signals`. A model handed the
full observation sees every KPI that wobbled by a percent and will, reliably, explain each one.
Only movements a deterministic significance test already called real are offered as findings;
anything else is carried as context flagged `moved: false` — which is what makes
*"why did occupancy fall even though admissions were flat"* answerable. When nothing is material,
no hypotheses are generated at all.

### Where hypotheses come from

The retail template library is gone. It could not do the job: a template written around orders,
stockouts and discounting fired on any dataset with a dimension column, so a hospital's declining
margin was explained as a competitor taking volume. Two sources replace it, both measured by the
same machinery:

- **contract-derived, deterministic** — a ratio cannot move unless its numerator or denominator
  moved; an additive KPI moves with its terms; a change concentrated in one dimension member is
  localised. True of every business, needs no model, always available.
- **domain-aware, model-proposed** — mechanisms specific to how this kind of operation works, asked
  for in both a domain-specific and a general-business category, neither forced.

**Neither source asserts evidence.** Both emit *predictions* — this metric should have moved this
way — and `hypotheses.evidence(..., expect=)` measures each against the data and flips a prediction
that did not hold into evidence *against* the hypothesis that made it. A hypothesis cannot claim
support it does not have, however plausible its wording. A prediction naming a metric the dataset
does not measure is dropped, and a hypothesis left with none is discarded: this is why a hospital
can no longer be told about competitor pricing, and why that is now structural rather than
discouraged.

### Stage 0 — the KPI CONTRACT (`app/kpi/`)

Before any stage can run, the system has to know what a KPI *is*. That used to be
a frozen 20-entry dictionary matched to uploaded columns by literal name, which
meant a hospital dataset got a retail dashboard. The KPI Contract replaces it.

```
profiling  -> what does each column MEAN?      semantic type, additivity, containment
library    -> which broadly applicable KPIs does this data support?   binds on CONCEPT
derivation -> which derived KPIs are semantically valid?              typed rules + data checks
screening  -> (optional) a model judges semantics; it never computes
conflicts  -> what is ambiguous, and must a human decide?
contract   -> the artefact: one versioned, approved document per (user, dataset)
resolver   -> the artefact, compiled into something that computes
```

* **Concept binding, not column names.** A library entry declares the concepts it
  needs (`revenue`, `cost`) with aliases and a required semantic type, so
  `net_sales`, `turnover` and `billed_amount` all satisfy revenue. **An entry
  activates only when every concept binds to a real field** — the reason a
  hospital dataset never grows a gross-margin tile, and the reason the absence is
  reported (`no field matched the concept 'revenue'`) rather than silent.
* **Computable is not meaningful.** Any two numeric columns can be divided.
  A rate is only proposed where the numerator is genuinely contained by the
  denominator *on the actual rows*, and where the denominator reads as a
  population rather than as another measure that happens to be bigger. That is
  what separates `recovered / discharges` from `orders / revenue`. Combinations
  the rules decline are returned with the reason, so a user can see the system
  considered them.
* **Granularity is first-class.** Every KPI declares its entity grain, time grain,
  native row grain and roll-up policy. A ratio's policy is
  `recompute_from_components`, never `mean`: averaging four weekly margins is not
  the quarterly margin. A grain that was inferred but not confirmed blocks
  approval, because a KPI compared at the wrong grain is simply wrong.
* **Ambiguity is never silently resolved.** Two definitions of one name, a
  concept two columns match equally well, a hierarchy whose members roll up to
  two parents, a rate whose containment fails on the rows — each becomes a
  `blocking` conflict that prevents approval until a person chooses and records
  why. The rationale lands in the provenance of every KPI it touched.
* **Formulas are parsed, never executed.** Definitions arrive from the API as
  text and are compiled into a small typed AST over an allowlist of column names.
  There is no `eval`.
* **The LLM judges, it does not compute.** Screening sees column names, semantic
  types and summary statistics — never a row. Every proposal it makes is
  re-validated against the field list; one naming a column that does not exist is
  discarded. With no API key the deterministic rules decide alone and everything
  is marked for review.

**Migration.** A dataset with no contract is given a `provisional` one generated
from the library, marked as never reviewed. The test suite asserts it reproduces
the old registry's numbers value-for-value, for every KPI in every quarter, so
making the contract the source of truth changed no number anywhere.

**A draft is not live.** Approving one KPI inside a draft does not change what the
dashboard shows; the draft goes live only when the contract as a whole is
approved, which is also when every blocking conflict must have been resolved.

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
  api/          routes_auth · routes_data · routes_kpi · routes_analysis · routes_system · redact
  kpi/          contract · profiling · library · derivation · screening · conflicts · resolver · service
  services/     dataset_service (load + cache + contract) · pipeline (stage orchestration)
```

`app/kpi/` depends on nothing in `app/engines/`, so the dependency runs one way:
the engines resolve KPIs through a compiled contract, and the contract layer knows
nothing about the four stages.

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
  repositories.py  Users · Datasets · Documents · Investigations · KpiContracts
```

KPI contracts are versioned the way datasets are activated: a new document per
version with `is_current` flipped, never mutation in place. The version that
produced a saved investigation is still on disk, so an approval is auditable and
reversible.

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

### Personas are the presentation half, and are not authorisation

Role decides what may be sent. **Persona** decides how it reads and what the reader is advised to
do. The five (`business_analyst`, `business_manager`, `business_leader`, `domain_specialist`,
`operational_user`) are free for any user to choose, because choosing one grants nothing — it writes
a different field, and `redact.py` never looks at it.

| | Business Analyst | Business Manager | Business Leader | Domain Specialist | Operational User |
|---|---|---|---|---|---|
| Recommends | analytical follow-ups | operational interventions | decisions and priorities | domain-technical actions | immediate actions |
| Horizon | next analysis cycle | this quarter | strategic | 2–4 weeks | next shift |

**The invariant, enforced in code.** `personas.reframe` builds every persona's advice from one
`RecommendationCore` per hypothesis — cause metric, confidence, causal claim, supporting and
contradicting evidence — computed before any persona is consulted. Evidence, ranking, confidence and
causal verdicts are therefore identical for every reader; only framing and advice differ. A
recommendation whose `based_on` is not in that core is dropped rather than shown, so a reframing
cannot become an invention. Persona differentiation also works with no API key, via
`_deterministic_reframe` — it degrades in eloquence, not in existence.

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
