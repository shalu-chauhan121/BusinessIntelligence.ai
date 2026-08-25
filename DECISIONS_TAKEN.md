# Decisions Taken — Question-Driven Investigation

Review log for decisions made autonomously during implementation of the
question-driven Investigation architecture. Newest entries appended at the end.

Priority order used throughout:
1. Explicit requirements in the approved plan
2. Existing project architecture and conventions
3. Deterministic, simple, maintainable implementation
4. Minimal change and reuse of existing infrastructure
5. Reasonable assumptions where necessary

---

## 1. Rehydrate `DomainContext` from the contract rather than re-detecting it

**Context/problem.** `detect_domain_context` (`kpi/domain.py:112`) runs once at discovery
time (`kpi/service.py:405`) and its result is flattened onto the contract as four scalar
fields (`detected_domains`, `domain_confidence`, `domain_uncertain`, `domain_evidence`).
The rich `DomainContext` object — in particular `vocab: DomainVocab` — is discarded. The
analysis engines therefore have no access to the business domain, which is the root cause
of generic, non-domain-aware explanations.

**Options considered.**
- (a) Re-run `detect_domain_context` at analysis time. Requires `LibraryMatch` + `DatasetProfile`,
  neither of which is available post-discovery; would mean re-profiling on every request.
- (b) Persist the full `DomainContext` as a new nested object on the contract. Cleaner model,
  but a schema change requiring migration of existing contracts.
- (c) Rehydrate `DomainContext` from the four flat fields already stored, looking `vocab` up
  from `DOMAIN_VOCAB` by domain key.

**Choice made.** (c) — new `domain.domain_context_from_contract(contract)` helper, called in
`dataset_service.attach_contract` and hung on `DatasetSchema.domain`.

**Reason.** Zero migration, zero new LLM/profiling cost, and no new persisted state. The
four flat fields plus the static `DOMAIN_VOCAB` table are together sufficient to reconstruct
the object losslessly, because `vocab` is a pure function of `domain`.

**Consequence/trade-off.** `detected_domains` is a list while `DomainContext.domain` is a
scalar; the helper takes the first entry and treats an empty list as `"uncertain"`. If the
contract ever legitimately carries multiple co-equal domains, this reduction loses that
nuance — acceptable because `detect_domain_context` already collapses ties to `"uncertain"`.

---

## 2. Expose KPI semantics through `DatasetSchema` accessors, not by widening the engines' inputs

**Context/problem.** `KpiDefinition` (business_definition, relevance, semantic_tags, formula,
provenance) is reachable only via `CompiledKpi.definition` on the resolver. Engines receive
`schema` but consistently reach for `metric_label`/`metric_unit` helpers rather than the
resolver directly, so semantic fields never travel downstream.

**Options considered.**
- (a) Pass the resolver or the contract explicitly into every engine signature.
- (b) Add accessors on `DatasetSchema` that delegate to `contract_resolver`.

**Choice made.** (b) — `DatasetSchema.kpi_definition(key)` and `DatasetSchema.domain`.

**Reason.** `DatasetSchema` is already threaded through every engine, so this adds capability
without touching a single call signature. It matches the existing convention of
`metric_label`/`metric_unit`/`metric_description` helpers reading through the resolver.

**Consequence/trade-off.** `DatasetSchema` grows a dependency on the KPI contract types. Kept
tolerable by returning `Optional` everywhere and importing lazily inside the accessor, so a
dataset with no contract still works exactly as before.

---

## 3. Persona is a separate field from role, defaulting from it

**Context/problem.** The spec named five personas; the system has two auth roles
(`data_analyst`, `business_leader`) with a 10-capability permission matrix driving
server-side redaction.

**Options considered.**
- (a) Extend `Role` to five values. Conflates authorization with presentation and forces a
  permissions decision per persona.
- (b) A `persona` field independent of `role`.

**Choice made.** (b). `UserRepository.PERSONAS`, `set_persona`, `PATCH /api/auth/persona`,
`Persona` literal in schemas, and `app/personas/profiles.py` as the single source of truth.
`redact.py` is untouched and still keys on `role` alone.

**Reason.** Security must not move when a user changes how they want text framed. Keeping the
two orthogonal means `set_persona` provably cannot widen access — it writes a different key.

**Consequence/trade-off.** Two fields to reason about instead of one, and the settings UI must
explain the difference. Mitigated by defaulting persona from role (`ROLE_DEFAULT_PERSONA`), so a
user who never touches it sees sensible framing.

---

## 4. Unknown persona degrades to analyst framing, not to a guess

**Context/problem.** The plan requires "neutral business-analysis presentation rather than
inventing a persona" when none is reliably available.

**Choice made.** `persona_for(None)` and any unrecognised key return `BUSINESS_ANALYST`.
`shape_user` re-validates a stored persona and falls back to the role default if the stored
value is not recognised.

**Reason.** The analyst profile withholds nothing and asserts no decision context. Defaulting to
`business_leader` instead would silently strip method and evidence from a reader who may need it.

**Consequence/trade-off.** A leader whose persona failed to persist gets a denser page than they
want. Preferable to the reverse, which would hide material caveats.

---

## 5. Relative periods resolve against the dataset, never the wall clock

**Context/problem.** "last quarter" has to mean something specific. Using `datetime.now()` would
make the same question return different answers over time, and return nothing at all for a
dataset whose data ended a year ago.

**Choice made.** `resolve_period` derives every relative expression from
`observe.available_timeframes(df)` — its last entry is "latest", and "previous quarter" is
`Timeframe.previous()` of that.

**Reason.** Determinism (a stated requirement: the same question must yield the same numbers) and
correctness for historical datasets.

**Consequence/trade-off.** "this quarter" means the dataset's newest quarter even if that is
historical. Disclosed explicitly through `PeriodSpec.assumption_note`, which the UI surfaces.

---

## 6. A period the dataset does not hold blocks rather than falls back

**Context/problem.** A question naming Q4 2026 against a dataset ending Q2 2026 could either
snap to the nearest available quarter or refuse.

**Options considered.** (a) Snap to latest and disclose. (b) Block and list what is available.

**Choice made.** (b) — a blocking `Ambiguity(kind="unsupported_by_dataset")` naming the periods
that do exist.

**Reason.** The plan's ambiguity policy says never substitute a proxy. Answering about Q2 when
asked about Q4 produces a confidently wrong answer, which is the specific failure mode this
architecture exists to prevent. A missing period is a different case from a *vague* one, and only
the vague case has a defensible default.

**Consequence/trade-off.** Slightly more friction for typos. Acceptable: the message lists the
available periods, so recovery is one edit.

---

## 7. Phrase matching is token-bounded, not substring

**Context/problem.** Initial grounding matched "admissions" inside "readmissions", so any question
about admissions also scored readmissions — two clinically distinct measures. Compounded by the
tokeniser treating `_` as a word character, which collapsed `bed_occupancy_rate` into one token and
prevented "bed occupancy" from matching at all.

**Choice made.** `_contains_phrase` normalises both sides, splits on underscores, and requires a
contiguous whole-token subsequence.

**Reason.** Substring matching on domain vocabulary produces false positives that are hard to
notice and clinically misleading.

**Consequence/trade-off.** Morphological variants ("admission" vs "admissions") no longer match by
accident. Handled deliberately by the concept-alias table, which is the right place for it.

---

## 8. Contrast clauses become comparison KPIs, filtered to the top match

**Context/problem.** "Why did profit fall even though CAC was up" must investigate profit, not CAC.
Separately, alias matching bound "admissions" to both `admissions` and `cost_per_admission`
(a shared source field), adding a spurious second contrast.

**Choice made.** Split the question on the first contrast marker; the main clause supplies the
outcome and the contrast clause supplies comparison KPIs. Comparison candidates are then filtered
to those within 0.25 confidence of the best one.

**Reason.** A contrast clause names one measure. Keeping every KPI that shares a source field with
it turns one contrast into a family and dilutes the observation set.

**Consequence/trade-off.** A question genuinely contrasting two measures in one clause keeps only
the strongest unless they score closely. Acceptable; the alternative admits far more noise.

---

## 9. The deterministic floor is broader than formula decomposition

**Context/problem.** The approved plan retired all 10 retail templates and added one
contract-derived formula-decomposition generator as the no-LLM floor (flagged as a deviation). In
practice that floor was too narrow: it only fires for ratio KPIs with numerator/denominator
components. An additive KPI such as `revenue`, or any base metric with no component KPIs, produced
**zero** hypotheses with no LLM configured — `test_pipeline_runs_without_an_llm_key` failed outright.

**Options considered.**
- (a) Accept zero hypotheses without an LLM. Rejected: the product's stated design is that the
  investigation completes with or without an API key.
- (b) Reinstate some templates. Rejected: contradicts the approved decision.
- (c) Widen the deterministic floor with generators that are still contract-derived and
  domain-neutral.

**Choice made.** (c). Three deterministic generators in `engines/hypotheses.py`:
`formula_decomposition_candidates` (ratio numerator/denominator, plus additive per-term),
`related_driver_candidates` (from driver-graph association edges), and `concentration_candidate`
(the movement is concentrated in one dimension member, reusing the `determine_focus` over-index test).

**Reason.** Each is derived from the KPI contract or the dataset's own dimensions, carries no
industry assumption, and is falsifiable. Together they guarantee at least one testable explanation
for any dataset with either a decomposable KPI or a dimension.

**Consequence/trade-off.** More deterministic hypotheses than the plan described, so the shortlist
has more competition for its five slots. Mitigated by `llm_hypotheses.shortlist`, which reserves a
place for each category. These explanations are also modest by design — they say *which* component
or segment moved, not *why* — and each states that limitation in its own text.

---

## 10. Related-driver hypotheses predict same-direction movement

**Context/problem.** Hypotheses built from `shared_source_field` edges need a falsifiable
prediction, but the sign of such a relationship is not known deterministically. Setting `expect` to
whatever direction the metric actually moved would be circular — the evidence would always confirm.

**Choice made.** Predict that a related measure moved in the **same direction as the outcome**, and
let the existing `expect` auto-flip adjudicate. A related measure that moved the other way produces
contradicting evidence against its own hypothesis.

**Reason.** A genuine falsifiable prediction rather than a restatement of the data, and it keeps a
single adjudication mechanism for every hypothesis regardless of source.

**Consequence/trade-off.** A genuinely inverse driver is scored as contradicted rather than
recognised as an inverse relationship. Acceptable: the hypothesis text is explicit that shared
inputs make two measures related without making one the cause, and `missing` says the dataset
cannot separate the two readings.

---

## 11. The driver graph reads the seed registry when there is no contract

**Context/problem.** `build_driver_graph` was written against `CompiledKpi` (parsed formula ASTs).
Datasets with no KPI contract — including every existing engine test fixture — resolve KPIs through
the `metrics.METRICS` seed registry, so the graph returned zero edges and the deterministic floor
collapsed for them.

**Choice made.** `_spec_map` prefers the contract resolver and falls back to `METRICS`; `_all_fields`
and `_numerator_denominator_fields` read both shapes (`source_fields`/ASTs, and
`requires`/`numerator`/`denominator`/`numerator_expr`).

**Reason.** Mirrors the precedence `metrics.metric_spec` already establishes, so behaviour is
consistent with the rest of the engine layer and datasets predating the contract keep working.

**Consequence/trade-off.** Two spec shapes to support, contained to two small helpers.

---

## 12. LLM-proposed edges are verified on differenced quarterly changes

**Context/problem.** The plan said to verify proposed edges with `analysis.correlate` and
`analysis.lead_lag`. Their real signatures did not fit: `correlate(table, metric_a, metric_b)`
expects a member-change table with `{metric}__chg` columns, and `lead_lag` needs weekly frames with
12+ merged rows and a `_week` column, and computes without a resolver so contract KPIs do not resolve.

**Choice made.** New `driver_graph.period_change_table` builds the exact frame shape `correlate`
expects, with one row per period-over-period change instead of one per dimension member, so
`correlate` is reused unchanged. Ordering is checked with `_lagged_r`, comparing correlation at lag 0
against the cause leading and the outcome leading.

**Reason.** Reuses the existing statistic rather than adding a second correlation implementation.
Differencing rather than levels follows the reasoning already documented in `lead_lag` — two trending
series correlate at any lag, which is the spurious result being guarded against.

**Consequence/trade-off.** Period-level correlation needs `MIN_VERIFY_PERIODS = 6` changes to mean
anything, so short datasets verify nothing and every proposed edge stays weightless. That is the
correct failure direction: unverified edges may still suggest a hypothesis, they simply cannot lend
it weight.

---

## 13. Scoreboard KPIs use a magnitude test, not a significance verdict

**Context/problem.** `material_signals` filters on `verdict == "meaningful_signal"`, but only the
outcome KPI has a verdict — `kpi_scoreboard` entries are plain two-period comparisons with no
baseline distribution behind them.

**Choice made.** Scoreboard entries are included when `abs(change_pct) >= 5.0`, capped at six and
sorted by magnitude, each labelled "Two-period comparison; not independently significance-tested."

**Reason.** Some threshold is needed or the model sees every fractional wobble. Being explicit that
these are weaker than the outcome's verdict keeps the distinction visible rather than implying a
significance test that did not happen.

**Consequence/trade-off.** 5% is a judgement call, not a derived threshold. It is a single named
constant and the label is honest about what the number is.

---

## 14. Recommendations are regenerated per persona from an invariant core

**Context/problem.** Recommendations must be persona-aware, not shared — but persona must not be able
to change the facts, the ranking, the confidence, or turn correlation into causation.

**Choice made.** Split into `RecommendationCore` (persona-invariant: cause metric, confidence, causal
claim, supporting/contradicting/missing evidence, affected areas, monitoring threshold) and
`personas.reframe`, which generates each persona's explanation and actions **from** that core.
`_validate_actions` drops any recommendation whose `based_on` is not a cause metric or hypothesis key
present in the core.

**Reason.** Makes the invariant structural rather than a matter of prompt compliance. A model that
tries to recommend "cut prices to beat the competitor" on a hospital dataset has that recommendation
dropped, because `competitor_price_index` is not in the core. Covered by
`test_a_model_may_not_smuggle_in_an_unsupported_action`.

**Consequence/trade-off.** A legitimate action phrased with an unrecognised `based_on` is dropped.
Preferred over the alternative, which is advice with borrowed authority.

---

## 15. Personas differ without an LLM too

**Context/problem.** If persona framing only existed in the LLM path, the product would show
identical text for all five personas whenever no API key is configured.

**Choice made.** `_deterministic_reframe` plus `_ACTION_TEMPLATES` give each persona its own action
verb, owner and horizon, and `detail_level` controls summary depth, all without a model.

**Reason.** Persona is a product behaviour, not an LLM feature. It should degrade in eloquence, not
disappear.

**Consequence/trade-off.** The templated actions are generic compared with generated ones. They are
correct, attributed and differentiated, which is what the guarantee requires.

---

## 16. `run_question` returns a clarification rather than raising

**Context/problem.** A blocked intent is a normal outcome, not an error, but the existing route helper
`_guard` converts `ValueError` into HTTP 422.

**Choice made.** `pipeline.run_question` returns `{"status": "needs_clarification", intent,
ambiguities}` with HTTP 200; only genuine failures raise.

**Reason.** The clarification carries candidate KPIs the user can choose from — a step in the
conversation, not a failure. An error status would make the frontend treat a normal branch as a fault.

**Consequence/trade-off.** Callers must check `status` rather than the HTTP code. The field is always
present on this endpoint.

---

## 17. `run_full` is kept as a KPI-driven entry point

**Context/problem.** The plan removes KPI selection as the *primary* workflow. Deleting `run_full`
outright would break the dashboard, saved-investigation replay, and every existing test.

**Choice made.** `run_question` interprets a question and then delegates to `run_full`, which gains
optional `persona`, `question` and `intent` parameters. `POST /api/investigations/run` stays,
documented as superseded.

**Reason.** One pipeline, one set of stages, two entry points. The dashboard legitimately drills in
from an already-named KPI, which needs no question.

**Consequence/trade-off.** Two entry points to maintain. `run_question` is a thin wrapper, so no stage
logic is duplicated.

---

## 18. Fiscal calendars are routed through one function that refuses

**Context/problem.** `CalendarSpec.fiscal_year_start_month` has always existed and no engine reads it;
all period logic uses the calendar `_year`/`_quarter` columns from `metrics.prepare`.

**Choice made.** `periods.to_timeframe(year, quarter, calendar)` is the single conversion point. For
the January default it is the identity, so behaviour is unchanged. For any other start month it raises
`NotImplementedError` with an explanation.

**Reason.** Silently mislabelling a fiscal quarter as a calendar one is a correctness bug that would be
very hard to notice. Refusing is honest, and the offset now has exactly one home for when the
frame-level work is done.

**Consequence/trade-off.** Non-January fiscal years remain unsupported. They already were; the
difference is the system now says so instead of quietly getting it wrong.

---

## 19. Three existing pipeline tests were rewritten, not deleted

**Context/problem.** `test_competing_hypotheses_are_generated`,
`test_documentary_evidence_is_quoted_with_a_source` and
`test_temporal_precedence_contradicts_the_supply_story` all asserted on `supply_constraint` and
`demand_contraction` — templates the approved plan deletes.

**Choice made.** Rewritten to assert the underlying properties: that several competing hypotheses are
produced and each is attributable to an identifiable source; that whatever hypotheses retrieved
documents quote them with a filename; and that *any* hypothesis dated after the KPI moved is
confidence-capped. Added `test_every_hypothesis_seeks_disconfirming_evidence`.

**Reason.** The behaviours those tests protected are still required; only the names that produced them
are gone. Deleting them would have lost real coverage.

**Consequence/trade-off.** The temporal test now skips when the fixture produces no late-dated
hypothesis, rather than failing. It asserts unconditionally when one exists.

---

## 20. `act()` gained a `persona` parameter rather than a separate stage

**Context/problem.** Persona output could have been a fifth pipeline stage or an argument to the
existing fourth.

**Choice made.** `act(..., persona=None)` returns `persona_view` alongside the existing `narrative`,
`recommendations` and `llm_story`. Omitting the argument reproduces the previous behaviour exactly.

**Reason.** Persona reframing consumes precisely what `act` has already assembled. A separate stage
would need the same inputs and add an orchestration step for no gain, and the default keeps every
existing caller working unchanged.

**Consequence/trade-off.** `act` does two things — deterministic narrative and persona view. The
persona work is delegated to `personas.reframe`, so `act` only wires it.

---

## 21. `TimeframePicker` keeps the KPI selector behind a flag

**Context/problem.** The plan removes KPI selection from Investigation, but `TimeframePicker` and
`useAnalysisSettings` (localStorage key `bi-analysis-settings`) are shared with `Dashboard.jsx`.
Deleting the KPI column outright would break the dashboard's own browsing.

**Choice made.** `showKpi` prop, defaulting to `true`. Investigation no longer renders
`TimeframePicker` at all; Dashboard is untouched and keeps its selector. `useAnalysisSettings` is
unchanged, so no stored settings migrate.

**Reason.** Browsing the dashboard by KPI is a legitimate use that needs no question. Removing KPI
selection as the *investigation entry point* was the requirement, not removing it everywhere.

**Consequence/trade-off.** Two ways to reach an investigation remain (a question, or a dashboard
drill-in). Both converge on `run_full`, so there is one pipeline.

---

## 22. Example questions are generated from the dataset's own KPIs

**Context/problem.** The question bar needs examples, but hardcoded ones ("why did revenue fall")
teach a hospital user nothing about what they can ask — the same generic-content failure this whole
change is meant to fix.

**Choice made.** `exampleQuestions(meta)` in `InvestigationPage.jsx` builds prompts from the first
two entries of the live `kpi_catalogue`.

**Reason.** The examples are then always answerable against the loaded dataset, and they teach the
contrast phrasing ("even though") the grounding layer understands.

**Consequence/trade-off.** Examples are only as good as the KPI labels. Acceptable — those labels
are the contract's own, and are what the user will be reading everywhere else.

---

## 23. Clarification candidates become a question rather than a bare KPI id

**Context/problem.** When the user picks a candidate KPI from a clarification prompt, the system
could either re-run with a confirmed intent object or synthesise a new question.

**Choice made.** `askAbout` composes `Why did {label} change?` and resubmits through the normal
question path.

**Reason.** One code path, and the question bar then shows what was actually asked, so the history
entry and the intent panel stay consistent with each other. The `confirmed_intent` parameter exists
on `run_question` for a future richer correction flow.

**Consequence/trade-off.** The user's original phrasing (period, contrast) is lost when they pick a
candidate. Worth revisiting if clarification turns out to be common; the plumbing is already there.

---

## 24. `depends_on` is derived on the `CompiledKpi`, not written back to the stored `KpiDefinition`

**Context/problem.** Gap review found the plan's "revive `depends_on` in `resolver.compile_contract`"
item unimplemented; `driver_graph.py` had instead grown its own parallel field-overlap logic
(`_spec_map`/`_all_fields`), so the contract's own `depends_on` stayed dead everywhere else that
might read it (KPI Studio UI, future tooling).

**Options considered.**
- (a) Mutate `KpiDefinition.depends_on` in place during compilation. Rejected: the definition is a
  persisted Pydantic model tracked by contract version; mutating it during a read-path compile would
  give `compile_contract` a side effect on data that should change only through an explicit edit +
  save, and risks the derived value leaking into a subsequent `contract.save()`.
- (b) Populate `CompiledKpi.depends_on` (already a field, already copied from the definition, never
  filled) via a structural pass over every compiled KPI's fields.

**Choice made.** (b). New `resolver._derive_structural_depends_on(compiled)`, called at the end of
`compile_contract`. For each KPI, any other compiled KPI whose full field set is a subset of this
one's numerator/denominator/expression fields is added to `depends_on`; hand-declared entries are
kept, structural ones are unioned in.

**Reason.** Matches the plan's intent — `depends_on` becomes real — without touching persisted state
during a compile. `CompiledKpi` is already the object every engine reads; enriching it is the correct
layer.

**Consequence/trade-off.** `KpiDefinition.depends_on` (the persisted field) still reads empty; only
the compiled, in-memory `CompiledKpi.depends_on` is populated. Any consumer that reads the definition
directly rather than through the resolver still sees nothing. `driver_graph.py` was left as it was —
it still computes its own richer, multi-relation graph (formula/derivation/shared-field/semantic-tag)
directly from field sets, because `depends_on` alone only captures the strongest of those relations
and driver_graph needs the others too.

---

## 25. The single-stage endpoints accept a question, with the KPI-driven contract untouched

**Context/problem.** `/api/observe|investigate|contest|act` were flagged as a gap: the plan called
for repurposing them as intent-driven, but they still only accepted `AnalysisRequest.kpi/year/quarter`
and never received `signals`/`graph`, meaning a single-stage call and the full pipeline could
silently diverge on what counted as a material signal.

**Choice made.** `AnalysisRequest` gained optional `question` and `persona` fields. A new
`_resolve_stage_target` helper resolves the actual KPI/period either from `body.question` (via
`interpret_question_cached`) or from `body.kpi/year/quarter` when no question is given — the question
wins when both are supplied, as the more specific instruction. All four endpoints now call the same
`pipeline.prepare_stage_context` that `run_full` uses, so `/api/investigate` for a given KPI and
`run_full` for that KPI can no longer disagree about material signals or the driver graph. `/api/act`
also gained `persona` handling identical to `run_full`'s.

**Reason.** Preserves every existing caller (an `AnalysisRequest` with only `kpi` behaves exactly as
before — verified by `TestKpiDrivenStagesStillWork`), while giving the endpoints the intent-driven
capability the plan asked for, without a second request model.

**Consequence/trade-off.** `_resolve_stage_target` duplicates a small amount of the blocking logic
also present in `pipeline.run_question` (checking `intent.blocked`), because the single-stage
endpoints need to return a stage-shaped clarification (`{"stage": ..., "status": ...}`) rather than
`run_full`'s envelope. Kept small and covered by
`test_an_unresolvable_question_returns_a_clarification_not_an_error`.

---

## 26. Persona switcher added to Settings, driven by `/api/auth/config`

**Context/problem.** `api.setPersona` and the backend endpoint existed with no UI control — a user
could not actually change persona.

**Choice made.** `SettingsPage.jsx` fetches `personas` from the existing `/api/auth/config` response
(already returning the list) and renders a selector matching the existing role-selector's visual
pattern, calling a new `changePersona` on `AuthContext` (mirroring `changeRole`).

**Reason.** No new endpoint needed — `/api/auth/config` already carried `personas` from the original
implementation pass; only the frontend was missing. Mirroring the role selector's pattern kept the
page visually consistent rather than introducing a new interaction style for one field.

**Consequence/trade-off.** Persona icons are a frontend-only lookup table (`PERSONA_ICONS`) keyed by
persona, since icon choice is presentational and doesn't belong in the API response.

---

## 27. Intent interpretation is cached in-process, keyed on contract version rather than explicitly invalidated

**Context/problem.** The plan called for caching intent by
`(question_normalized, dataset_id, contract_version)` but this was never built — every
`/questions/interpret` and `/questions/investigate` call re-ran grounding (and, when triggered, a
full LLM round-trip) from scratch even for an identical repeated question.

**Options considered.**
- (a) A TTL-based cache with a fixed expiry.
- (b) A cache keyed on `(question, dataset_id, contract_version)`, relying on the version already
  changing whenever a KPI contract is edited and re-attached.

**Choice made.** (b). `interpret_question_cached` in `app/query/understanding.py`, module-level dict
mirroring the existing `dataset_service._CACHE` pattern (same file, `_CACHE: Dict[...] = {}`, an
explicit `clear_cache()` escape hatch). FIFO eviction past 256 entries. Every cache hit returns
`.model_copy(deep=True)` rather than the stored object, so a caller mutating its own returned
`InvestigationIntent` can never corrupt what the next caller receives — verified directly by
`test_repeated_questions_are_served_from_cache_without_corruption`.

**Reason.** A TTL invents a staleness window that doesn't correspond to anything real — the contract
is either unchanged (cache is exactly correct, forever) or changed (cache must miss, immediately).
Version-keying gets both for free: no wall-clock guess, and no explicit invalidation call needed on
the KPI Studio approval path, because the next `dataset_service.load()` after an edit already bumps
`schema.contract_version`.

**Consequence/trade-off.** The cache is per-process and unbounded except by the 256-entry FIFO cap —
acceptable at the scale the rest of the app's in-process caching already assumes (the dataset cache
has no cap at all). `clear_intent_cache()` is exposed but not wired to any event, since version-keying
makes explicit invalidation unnecessary for the one case that matters (a contract edit).
