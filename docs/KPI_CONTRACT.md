# KPI Definition & KPI Contract Layer

## Context

**The problem.** Every KPI in this product is hardcoded. `backend/app/engines/metrics.py` holds a frozen 20-entry `METRICS: Dict[str, MetricSpec]` dict — `revenue`, `gross_margin_pct`, `stockout_rate`, `avg_order_value` — matched to uploaded CSVs by **literal column name**. `detect_schema()` classifies columns only as date / numeric / categorical; unknown numeric columns become nameless "extra metrics" that are always summed. There is no KPI persistence, no CRUD, no user override, no semantic layer, no granularity model, and no calendar or hierarchy model anywhere in the codebase.

The consequence: upload a hospital dataset and the product finds `revenue` and `cost_of_goods`, computes gross margin, and is blind to admissions, length of stay, readmission rate and recovery rate. It renders a generic retail dashboard for a hospital. That is precisely the failure this layer exists to remove.

**The outcome.** A **KPI Contract** — one versioned, user-approved document per (user, dataset) that is the authoritative analytical definition of every KPI: what it means, how it is computed, from which fields and sources, at what grain, under which calendar and hierarchy, with which business rules, and with what provenance and approval state. Downstream — Observe, Investigate, Contest, Act — resolves KPIs through the contract instead of through a Python literal.

**Decisions taken** (confirmed with the user before writing this plan):
1. The contract **becomes the source of truth**. `METRICS` is demoted to a seed library; the engines resolve through a compiled contract.
2. Ship **backend + a KPI Studio UI** — requirement §2's "user must review and confirm" needs a real surface.
3. Discovery is **deterministic candidate generation + LLM semantic screening**, with a full deterministic fallback when no API key is set.
4. Multi-source is **modelled now, single-source enforced at runtime**.

**Non-negotiable constraint inherited from the codebase.** `docs/ARCHITECTURE.md` §1: *the LLM never produces a business number.* In this layer the model judges semantics, names KPIs and writes definitions. It never computes, and it can never reference a field that is not in the profiled field list — every LLM proposal is re-validated deterministically before it can be stored.

---

## Architecture

```
CSV upload ──► profiling.py ──► FieldProfile[]  (semantic type, additivity, hierarchy hints)
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
              library.py        derivation.py      (user input)
          concept binding      typed derived         §2 defines
          general + domain      candidates            or overrides
                    └─────────────────┬─────────────────┘
                                      ▼
                              screening.py  (Claude: valid? name? why? ambiguous?)
                                      ▼
                              conflicts.py  (grain / calendar / hierarchy / duplicate)
                                      ▼
                     ┌────────► KpiContract (status: draft → proposed → approved)
                     │                │
              KPI Studio UI ──────────┘ user reviews, edits, resolves conflicts, approves
                                      │
                                      ▼
                              resolver.py  ──► CompiledKpi (drop-in for MetricSpec)
                                      ▼
                    observe / hypotheses / contest / act  (unchanged call shapes)
```

### New backend module: `backend/app/kpi/`

| File | Responsibility |
|---|---|
| `contract.py` | The Pydantic models. **This file is the deliverable** — everything else serves it. |
| `profiling.py` | `FieldProfile` + semantic-role inference + row-grain and hierarchy detection. |
| `library.py` | The general KPI library and domain packs, as data. |
| `derivation.py` | Typed rules that generate *valid* derived candidates and reject merely-computable ones. |
| `screening.py` | LLM semantic screening; deterministic fallback. |
| `conflicts.py` | Conflict/ambiguity detection and severity. |
| `resolver.py` | Compiles a contract into executable `CompiledKpi` objects. Safe AST, no `eval`. |
| `service.py` | Lifecycle: discover → propose → edit → resolve conflicts → approve → version. |

Following existing conventions, two pieces live **outside** this module in their established homes:
- `KpiContractRepository` → appended to `backend/app/db/repositories.py` ("the only DB surface the API uses").
- `KPI_DISCOVERY_SYSTEM` prompt → `backend/app/llm/prompts.py`; `screen_kpi_candidates()` → `backend/app/llm/client.py`, matching `frame_hypotheses` / `classify_stance` / `write_story`.

---

## 1. Field profiling — the enabler

`backend/app/kpi/profiling.py`. Everything downstream depends on knowing what a column *means*, which nothing in the codebase currently does.

```python
@dataclass
class FieldProfile:
    name: str
    role: Literal["measure", "dimension", "time", "identifier", "unknown"]
    semantic_type: Literal["money", "count", "rate_pct", "ratio", "duration",
                           "score", "flag", "id", "category", "date"]
    additivity: Literal["flow", "stock", "rate", "non_additive"]
    # flow  → summable across time (revenue, admissions)
    # stock → a level; mean/last across time, never summed (inventory_units, beds_occupied)
    # rate  → never summed and never averaged; recompute from components
    null_rate: float
    distinct_count: int
    min / max / median: Optional[float]
    is_integer: bool
    is_non_negative: bool
    name_tokens: List[str]          # normalised tokens for library alias matching
    subset_of: List[str]            # measures m where this <= m on >=95% of rows
    evidence: Dict[str, Any]        # every check that produced the above, for audit
```

Inference combines **name tokens**, **distribution** and **row-level relational checks**. The relational checks matter most: `subset_of` is what lets the system know `recovered_patients ⊆ admissions` and therefore that `recovered/admissions` is a rate, while `revenue/cost_of_goods` is not a subset relation and therefore not a rate.

Two further outputs:

- **Row grain** — the smallest column subset (date + dimensions) that is unique per row. This is the *native* grain and the floor for every KPI's granularity.
- **Hierarchy candidates** — for each ordered dimension pair, test whether each child value maps to exactly one parent value. Clean containment → a hierarchy level; violations → a `hierarchy_violation` conflict, never a silent fix.

The existing `DIMENSION_HINTS` constant at `backend/app/engines/metrics.py:17` is declared but never used — profiling gives it a real job as a name-token source.

---

## 2. General KPI library

`backend/app/kpi/library.py`. The library binds on **concept**, not on literal column name — this is the single change that makes it work across domains.

```python
LibraryKpi(
    id="profit",
    name="Profit",
    domain="general",
    definition="Revenue less total cost over the period.",
    concepts=[
        ConceptRef("revenue", aliases=["revenue", "sales", "net_sales", "turnover",
                                       "gross_revenue", "billed_amount"],
                   semantic_type="money", additivity="flow"),
        ConceptRef("cost", aliases=["cost", "cogs", "cost_of_goods", "expense",
                                    "total_cost", "operating_cost"],
                   semantic_type="money", additivity="flow"),
    ],
    formula="{revenue} - {cost}",
    unit="currency", aggregation="sum", higher_is_better=True,
    default_time_grain="month",
    semantic_tags=["profitability"],
)
```

**A library KPI activates only when every concept binds to a real field.** Unbound concepts are reported (`why_unavailable: "no field matched concept 'cost'"`) but never fabricated — requirement §1.

Packs, all data, no code changes to add one: `general` (revenue, cost, profit, profit margin, orders, customers, growth rate, average value per unit, cost per unit), plus `retail_ecommerce` (where the current 20 `METRICS` entries land), `healthcare`, `logistics`, `saas_subscription`, `manufacturing`.

The healthcare pack is the proof the design works: admissions, discharges, recovery rate, readmission rate, average length of stay, bed occupancy, mortality rate, cost per patient-day — all binding on concept, all inert on a retail CSV.

---

## 3. Derived-KPI discovery

`backend/app/kpi/derivation.py`. The requirement is explicit: *do not assume every mathematically possible combination is a meaningful KPI.* The rules are typed on semantic role, and every candidate must pass a **data check**, not just a type check.

| Rule | Precondition | Produces |
|---|---|---|
| `money(flow) − money(flow)` | one field reads as inflow, the other as outflow | surplus / profit |
| `A / B × 100` | `A.subset_of` contains `B`, both counts, verified ≥95% of rows | a rate (recovery rate, fulfilment rate) |
| `money / count` | count is a population or event count | unit economics (AOV, cost per patient) |
| `duration` mean | semantic_type = duration | average length of stay, avg handling time |
| `money(flow) / money(flow)` | subset relation holds (e.g. discount ⊆ gross) | share-of ratio |
| `(A − B) / A` | B.subset_of A | margin-shaped ratio |

**Explicitly rejected**, and recorded with the reason so the user can see the system considered and declined it:
- `rate / rate`, `rate` summed, any ratio whose denominator is not a population for its numerator
- `count / count` with no subset relation → emitted as `questionable`, blocked from auto-approval, requires user confirmation
- any combination where the subset check *fails on the actual rows* even though the names suggest it should hold — this is a data-quality finding, surfaced as a conflict

Each candidate carries `derivation_rule` and `computability_evidence` (the checks that passed, with numbers) so the proposal is auditable.

---

## 4. LLM semantic screening

`backend/app/kpi/screening.py` + a new `KPI_DISCOVERY_SYSTEM` prompt in `backend/app/llm/prompts.py` reusing the existing `GUARDRAIL` block verbatim, and `LLMClient.screen_kpi_candidates()` in `backend/app/llm/client.py` following the existing `_call(system, user)` pattern.

**Input:** field profiles (names, semantic types, additivity, cardinality — *no raw rows*, matching the existing fact-sheet discipline at `client.py:96`) plus the deterministic candidates.

**Output, per candidate:** `verdict: valid | questionable | reject`, business `name`, `definition`, `why_relevant`, `suggested_time_grain`, `suggested_entity_grain`, `semantic_tags`, `ambiguities[]`.

**It may also propose additional domain KPIs** — this is how a hospital dataset gets domain-specific metrics the library did not anticipate. Every such proposal is re-validated deterministically against the field list before storage; one naming an absent field is discarded and logged. All LLM-originated proposals land as `status=proposed`, `confidence=low`, `origin=llm_suggested`, and require explicit user approval.

**No API key:** deterministic verdicts only, every candidate `needs_review`, no names or prose beyond what the library supplies. The pipeline completes — consistent with the product's stated no-key guarantee.

⚠️ Note: `_extract_json` at `client.py:24` is regex-based with no schema validation. For this layer, parse the response through a Pydantic model and discard malformed entries rather than `.get()`-ing into it — the contract is an authoritative artefact and cannot absorb garbage.

---

## 5. The KPI Contract

`backend/app/kpi/contract.py`. Two levels.

### `KpiContract` — one per (uid, dataset_id), versioned

```python
_id, uid, dataset_id, version: int, is_current: bool
status: draft | proposed | approved | superseded
calendars: List[CalendarSpec]          # dataset-level, per-KPI overridable
hierarchies: List[HierarchySpec]
sources: List[SourceBinding]
kpis: List[KpiDefinition]
conflicts: List[ConflictFlag]
provenance, created_at, approved_at, approved_by
```

### `KpiDefinition` — every field required by §6

```python
kpi_id, name, business_definition
kpi_type: general | atomic | derived | user_defined
formula: FormulaSpec                    # typed AST, see §6 below
computation_note: str                   # plain-English restatement
sources: List[SourceBinding]            # dataset_id + fields + native grain + cadence
source_fields: List[str]
dimensions: List[str]                   # dimensions this KPI is valid to slice by
filters: List[FilterSpec]               # e.g. status != 'cancelled'
unit: currency | count | percent | ratio | duration
aggregation: AggregationSpec
granularity: GranularitySpec
time_semantics: TimeSemantics
hierarchy_refs: List[str]
business_rules: List[BusinessRule]      # statement + enforced: bool + machine_check | null
depends_on: List[str]                   # other kpi_ids — derived KPIs reference, not inline
validation_rules: List[ValidationRule]
relevance: str                          # why this KPI matters for this business
semantic_tags: List[str]
provenance: Provenance
confidence: float
status: proposed | approved | rejected | needs_confirmation
approval: ApprovalState
```

### Granularity is first-class (§3)

```python
class GranularitySpec:
    entity_grain: List[str]        # ["hospital"], ["product","region"], [] = whole business
    time_grain: day|week|month|quarter|year
    native_row_grain: List[str]    # from profiling — the floor
    rollup_policy: sum | mean | last | recompute_from_components
    valid_rollups: List[str]       # time grains this may legally be rolled up to
    declared_by: inferred | user
    requires_confirmation: bool
    confidence: float
```

`rollup_policy` is the field that fixes a real latent bug: **a ratio must never be averaged across periods.** Today `compute()` at `metrics.py:274` recomputes ratios from summed components by accident of implementation; the contract makes it a declared, enforced rule. A KPI whose grain cannot be reliably inferred — any ratio, or any dataset where the row grain is not unique — is marked `requires_confirmation` and cannot be approved until the user sets it.

### Time & calendar semantics (§4)

```python
class TimeSemantics:
    date_field: str
    date_role: event_date | period_start | period_end | posting_date
    calendar_id: str                    # "gregorian" | "fiscal:acme_apr"
    fiscal_year_start_month: int
    week_start: monday | sunday
    period_alignment: calendar | 4-4-5 | custom
    comparison_default: previous_period | year_over_year
    restatement_window_days: int        # data may be revised after landing
    timezone: str
```

`Timeframe` at `backend/app/engines/observe.py:37` hardcodes calendar quarters. The contract records the fiscal offset; the resolver applies it. Two KPIs on different calendars are flagged `calendar_mismatch`, not silently compared.

### Hierarchies (§4)

```python
class HierarchySpec:
    name: str                      # "geography"
    levels: List[str]              # leaf → root: ["store","region","country","global"]
    level_fields: Dict[str, str]
    containment_evidence: List[...]  # per parent/child pair, violation count
    rollup_allowed: bool
```

### Conflicts — never silently resolved (§4)

```python
class ConflictFlag:
    id, kind, severity: blocking | warning | info
    detail, affected: List[str]
    resolution_options: List[ResolutionOption]
    resolved_by, resolution, resolved_at
```

Kinds: `duplicate_definition`, `formula_disagreement`, `grain_mismatch`, `calendar_mismatch`, `hierarchy_violation`, `aggregation_ambiguity`, `unit_mismatch`, `source_disagreement`, `subset_check_failed`, `insufficient_history`.

**The rule that makes §4 mechanical: a KPI carrying an unresolved `blocking` conflict cannot reach `status=approved`, and a contract with any blocking conflict cannot be approved as a whole.** The user resolves by choosing an option and recording a rationale, which lands in provenance.

### Multi-source (§5) — modelled, single-source enforced

```python
class SourceBinding:
    dataset_id, source_label
    fields: List[str]
    native_grain: List[str]
    calendar_id: str
    refresh_cadence: str            # "weekly" | "daily" | "adhoc"
    completeness: {date_min, date_max, null_rates}
    precedence: int                 # which source wins on disagreement
```

`KpiContract.sources` and `KpiDefinition.sources` are lists today constrained by the service to length 1. Alongside them, a computed **comparability matrix** per KPI pair: `comparable_with: [kpi_id]` and `blocked_by: [{kpi_id, reason}]`, derived from grain, calendar and cadence. This is what downstream needs and does not have — today `kpi_scoreboard` at `observe.py:404` computes every KPI over the same slice regardless of grain, and `contest.py` correlates any two metrics without checking whether they are comparable at all. The contract publishes the answer; wiring the refusal into those call sites is noted below as a follow-on, not done here.

---

## 6. Resolver — how the contract drives computation

`backend/app/kpi/resolver.py`. Formulas are **not** `eval`'d. A small typed AST over an allowlist of bound fields:

```
FieldRef | Difference | Sum | Ratio | Scale | Constant | Filtered(expr, FilterSpec)
```

parsed from a restricted grammar (`{revenue} - {cost}`, `{recovered} / {admissions} * 100`). This covers everything the current `METRICS` registry does — including `numerator_expr="revenue-cost_of_goods"` at `metrics.py:47` — plus filters, which the registry cannot express at all.

```python
@dataclass
class CompiledKpi:
    key, label, unit, kind, scale, higher_is_better    # ← same surface as MetricSpec
    granularity, rollup_policy, sources, depends_on
    def compute(self, df) -> float
    def components(self, df) -> Tuple[float | None, float | None]

def compile_contract(contract) -> Dict[str, CompiledKpi]
```

`CompiledKpi` deliberately exposes `.key .label .unit .kind .scale .higher_is_better` — the exact attribute surface of `MetricSpec`. Every downstream `METRICS.get(metric)` becomes `resolver.get(metric)` and the surrounding code is untouched. That includes the two places that reach into spec internals: `observe.decompose_dimension` (`observe.py:271, 306`) and `metric_components` (`metrics.py:293`).

### Threading the resolver — the smallest viable change

Explicit parameter, no global state, no `contextvars`. Signatures gaining an optional `resolver` argument (defaulting to the legacy `METRICS` dict so nothing breaks mid-migration):

- `metrics.py`: `compute`, `metric_components`, `metric_label`, `metric_unit`, `higher_is_better`
- `observe.py`: `quarterly_series`, `weekly_series`, `decompose_dimension`, `observe`
- `hypotheses.py`: `Context` gains a `resolver` field — `evidence()` and every template already reach it through `ctx`
- `contest.py`, `act.py`: pass through from the `schema`/`observation` they already hold

`DatasetSchema` gains `contract_resolver: Optional[Dict[str, CompiledKpi]]`, populated by `dataset_service.load()`. Roughly 8 signature changes and ~20 call sites, all inside `backend/app/engines/`.

### Migration — nothing breaks

`dataset_service.load()` loads the current approved contract and compiles it. **If none exists, `service.bootstrap_contract()` generates a provisional one from library binding**, marked `status=provisional`, functionally identical to today's `METRICS` behaviour on the sample dataset. Existing users, existing datasets and `docs/ui-preview.html` all keep working, and the test in §Verification below asserts value-for-value parity.

---

## 7. API

New `backend/app/api/routes_kpi.py`, prefix `/api/kpi`, registered in `main.py` alongside the other four routers.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/kpi/contract` | Current contract for the active dataset |
| `POST` | `/api/kpi/contract/discover` | Run discovery → draft contract (`{dataset_id?, use_llm}`) |
| `GET` | `/api/kpi/contract/proposals` | Proposed KPIs + conflicts awaiting review |
| `POST` | `/api/kpi/contract/kpis` | Create a fully user-defined KPI (§2) |
| `PATCH` | `/api/kpi/contract/kpis/{kpi_id}` | Edit or override any definition field (§2) |
| `DELETE` | `/api/kpi/contract/kpis/{kpi_id}` | Remove a KPI |
| `POST` | `/api/kpi/contract/kpis/{kpi_id}/preview` | Dry-run compute over recent periods, no approval |
| `POST` | `/api/kpi/contract/kpis/{kpi_id}/approve` · `/reject` | Per-KPI decision |
| `POST` | `/api/kpi/contract/conflicts/{id}/resolve` | Choose an option + rationale |
| `POST` | `/api/kpi/contract/approve` | Approve whole contract → version++, becomes current |
| `GET` | `/api/kpi/contract/versions` | Audit history |
| `GET` | `/api/kpi/library` | Library entries + which bound, and why the rest did not |

Reads use `Depends(current_user)`; **all writes use `Depends(require_analyst)`** (`deps.py:41`) and a new `manage_kpi_contract` entry in the `PERMISSIONS` matrix at `routes_auth.py:16`.

⚠️ Deliberate, contained deviation: unlike every other route in this codebase, these endpoints declare real Pydantic **response models**. The contract is the artefact — its shape must be typed, not a bare `Dict[str, Any]`.

Storage: new `kpi_contracts` collection, `KpiContractRepository` in `db/repositories.py`, `uid`-scoped like every other repository. Versioning follows the existing `DatasetRepository.set_active` pattern (`repositories.py:116`): a new document per version with `is_current` flipped, so approvals are auditable and reversible.

---

## 8. Frontend — KPI Studio

New route `/kpi` in `App.jsx` and a nav entry in the `NAV` array at `components/Layout.jsx:20`.

**`frontend/src/pages/KpiStudioPage.jsx`** plus `frontend/src/components/kpi/`:

| Component | Role |
|---|---|
| `ContractHeader.jsx` | Version, status, approved/proposed/rejected counts, "Approve contract" (disabled while blocking conflicts stand, with the reason shown) |
| `KpiProposalList.jsx` | Grouped: General library · Discovered atomic · Discovered derived · User-defined. Per row: name, formula, grain, confidence, why relevant, source fields |
| `KpiDefinitionEditor.jsx` | Full editor drawer over every contract field (§2) |
| `GranularityPicker.jsx` | Entity grain (dimension multi-select) + time grain + rollup policy, inferred value pre-filled, explicit confirm affordance |
| `ConflictPanel.jsx` | Conflicts by severity with resolution choices and a rationale box |
| `KpiPreview.jsx` | Dry-run values across recent periods so a definition can be sanity-checked before approval |

Reuse, don't reinvent — the frontend has no form abstraction, so follow the established patterns:
- `components/ui.jsx` primitives: `SectionTitle`, `Callout`, `Badge` (tones `neutral|good|warning|serious|critical`), `EmptyState`, `ErrorState`, `LoadingCard`, `AnalystOnly`
- Design tokens and `.card` / `.card-pad` / `.field` / `.label` / `.btn-primary` / `.btn-ghost` from `src/index.css`
- The multi-select layout pattern from `components/TimeframePicker.jsx`
- The single-`form`-object + curried setter pattern from `pages/AuthPage.jsx`
- `lib/format.js` `formatValue(value, unit)` for previews

Touched existing files:
- `lib/api.js` — add ~12 methods to the flat `api` object, in a new `// kpi contract` group
- `pages/DataPage.jsx` — after upload, a `Callout` linking to `/kpi` ("N KPIs proposed — review before analysing"); **replace the hardcoded `REQUIRED` / `DIMENSIONS` / `METRICS` const arrays at lines 7–30 with the live library-binding report**, which is the visible payoff of this whole layer
- `components/TimeframePicker.jsx` — `kpiOptions` now sourced from the contract; show each KPI's grain
- `pages/Dashboard.jsx` — KPI scoreboard cards show grain and `provisional` status where applicable

⚠️ `AnalystOnly` is presentation, not authorisation. Any analyst-gated contract field must also be stripped server-side in `backend/app/api/redact.py`, which holds `ANALYST_ONLY_SIGNIFICANCE` / `ANALYST_ONLY_SCORING` — add a `ANALYST_ONLY_CONTRACT` list for computability evidence and confidence internals.

---

## 9. Explicitly out of scope

State these in the PR description so the boundary is clear:

- **Multi-dataset activation and cross-source joins.** Modelled in the contract, enforced to one source. Ingestion and `active_dataset` are untouched.
- **Making `hypotheses.py` domain-agnostic.** The 10 templates still gate on hardcoded keys (`ctx.has("stockout_rate")`). This layer *publishes* `semantic_tags` so a later phase can rewrite `applies()` against tags instead of literal keys — that rewrite is not done here.
- **Contract-driven anomaly thresholds**, warehouse connectors, contract-aware refusal inside `contest.py` correlation. The comparability matrix is computed and served; consuming it is follow-on work.

---

## 10. Files

**New — backend:** `app/kpi/{__init__,contract,profiling,library,derivation,screening,conflicts,resolver,service}.py`, `app/api/routes_kpi.py`, `tests/test_kpi_contract.py`, `tests/fixtures/hospital_sample.csv`

**New — frontend:** `src/pages/KpiStudioPage.jsx`, `src/components/kpi/{ContractHeader,KpiProposalList,KpiDefinitionEditor,GranularityPicker,ConflictPanel,KpiPreview}.jsx`

**New — docs:** `docs/KPI_CONTRACT.md` (spec, field reference, full JSON example)

**Modified — backend:** `app/engines/metrics.py` (registry → seed library, resolver-aware helpers), `app/engines/observe.py` · `hypotheses.py` · `contest.py` · `act.py` (resolver threading), `app/services/dataset_service.py` (load + compile contract), `app/services/pipeline.py` (validate KPI against contract), `app/db/repositories.py` (+`KpiContractRepository`), `app/llm/{client,prompts}.py` (+screening), `app/api/{routes_auth,redact}.py`, `app/main.py`

**Modified — frontend:** `src/App.jsx`, `src/components/Layout.jsx`, `src/lib/api.js`, `src/pages/DataPage.jsx`, `src/pages/Dashboard.jsx`, `src/components/TimeframePicker.jsx`

**Modified — docs:** `docs/API_CONTRACT.md`, `docs/ARCHITECTURE.md`, `sample_data/DATA_FORMAT.md`

---

## Verification

Tests go in `backend/tests/test_kpi_contract.py` on the existing `EngineTestCase` base (`tests/base.py`) — stdlib `unittest`, no API key, no network, matching the suite's existing discipline.

```bash
make test
# or: cd backend && python -m unittest discover -s tests -t .
```

**Migration safety — the test that must pass first.** For every KPI in the bootstrap contract on `sample_data/business_metrics_sample.csv`, assert `resolver.compute(df, key) == ` the legacy `METRICS`-based value, for every quarter. If this holds, the source-of-truth swap changed no number anywhere.

**Correctness:**
- Profiling assigns the right `semantic_type` / `additivity` on the sample CSV — notably `inventory_units` as `stock`, not `flow`
- A revenue-only CSV activates revenue KPIs and **never** fabricates margin or AOV (§1)
- A synthetic hospital CSV (`admissions, discharges, recovered, bed_days, length_of_stay, department, date`) yields recovery rate, occupancy and avg LOS, and yields **no** `gross_margin_pct`
- `revenue / cost_of_goods` is not proposed — no subset relation
- Subset check rejects a rate when the data violates it (`returns > customers` on some rows)
- Ratio rollup: a ratio over two quarters equals recompute-from-components, **not** the mean of the two quarter ratios
- Row-grain uniqueness detection; a KPI whose grain is not inferable is `requires_confirmation` and cannot be approved
- Duplicate concept definitions raise a `blocking` `duplicate_definition` conflict; approval is refused while it stands
- A fiscal calendar with April start places a given date in the correct fiscal quarter
- Approval lifecycle: proposed → edited → approved bumps `version`, retains the prior version, records provenance
- Discovery with no `ANTHROPIC_API_KEY` completes and never emits a KPI referencing an absent field
- Contracts are `uid`-scoped — a second user sees none of the first's

**End-to-end, manually:**
```bash
make backend                       # http://localhost:8000/docs
make frontend                      # http://localhost:5173
```
1. Sign in (demo mode), Business data → **Load sample company**
2. New **KPI contract** page → Discover → review proposals, grains, conflicts
3. Edit one KPI's granularity, add a user-defined KPI with a filter, Preview it
4. Resolve any blocking conflict, Approve the contract
5. Dashboard → scoreboard reflects the approved contract; values match the pre-change screenshots in `docs/screenshots/dashboard-light.png`
6. Run a full investigation → all four stages complete unchanged
7. Upload a hospital-shaped CSV → discovery proposes clinical KPIs, no retail dashboard

**No-key check:** unset `ANTHROPIC_API_KEY`, repeat steps 2–6. Discovery must complete deterministically with every candidate marked `needs_review`.
