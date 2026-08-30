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

---

## 28. HHI and Gini refuse a truncated member set; top-k share does not

**Context/problem.** `measure_concentration` (`agent/concentration.py`) is built on
`BreakdownEngine.rank_entities`, which caps at `QueryEngine.MAX_GROUPS` (500) and reports
`truncated` when a dimension has more members than that. An HHI or Gini computed over only the
top 500 of, say, 10,000 members is not an approximation of the true index — it systematically
overstates concentration, because every excluded member is by construction smaller than every
included one. Top-k share and "members for N% of total" are a different case: `rank_entities`
sorts descending *before* truncating, and the grand total comes from an independent ungrouped
query, so the top-k members and their cumulative share are exactly correct regardless of how
many long-tail members were never fetched.

**Options considered.**
- (a) Report every index as `None` whenever `truncated` is true, including top-k share.
- (b) Compute every index over the truncated set anyway, with `truncated` as a caveat flag.
- (c) Gate HHI/Gini/effective-members off entirely under `too_many_members`, while top-k share
  and `members_for_Npct` stay real numbers, because they remain exact under truncation.

**Choice made.** (c).

**Reason.** (a) throws away numbers that are still correct; (b) reports numbers that are
actively wrong without saying so strongly enough — `truncated: true` sitting next to a
confident-looking `hhi: 0.34` is exactly the kind of caveat a model skims past. Splitting the
gate by whether truncation actually breaks the specific number keeps every returned field
either exact or absent, never approximate-and-unlabelled.

**Consequence/trade-off.** A caller asking "how concentrated is this?" on a very wide dimension
gets `hhi: null` but a real `top_k_shares`, which means the tool cannot answer "is this a
concentrated market" as a single index for a 10,000+-member dimension — it can still answer
"what share do the top 10 hold," which covers most of what a question like that actually wants.

---

## 29. `measure_concentration`'s additivity gate reads `ContractAPI.is_additive`, not `kind() == "sum"`

**Context/problem.** The board's task description for I5 suggested gating "not_additive" on
`kind(key) != "sum"`. But `share_of_total_pct` — the field every one of `measure_concentration`'s
numbers is built from — is `None` on `rank_entities`' own entries whenever
`share_valid = additive and grand_total == grand_total and grand_total != 0`
(`agent/breakdown.py:264`) is false, and `additive` there is `ContractAPI.is_additive`, not
`kind()`. Gating this module's own status on a different condition than the one that actually
determines whether the numbers it reads exist would let the two disagree.

**Options considered.**
- (a) Gate on `kind(key) == "sum"`, as originally sketched.
- (b) Gate on `is_additive(key)`, the literal condition `rank_entities` uses.

**Choice made.** (b).

**Reason.** A KPI could in principle be `kind() == "mean"` while still being additive in the
sense that matters here, or vice versa; the only fact that matters is whether
`share_of_total_pct` is a real number or `None` on the entries this module reads, and that fact
is `is_additive`, by construction of `rank_entities` itself. Gating on anything else risks a
status of `"ok"` whose `top_k_shares` are silently all `None`, or a status of `"not_additive"`
that undersells a KPI whose shares were actually computable.

**Consequence/trade-off.** None functionally — `kind() == "sum"` and `is_additive() == True`
coincide on every KPI in every sample fixture today, so this only matters if a future contract
introduces a KPI where the two diverge, at which point this choice is the one that stays
correct.

---

## 30. Cohort overlap in `compare_cohorts` is rejected on row-set intersection, not on comparing filter dicts

**Context/problem.** `compare_cohorts` (`agent/segments.py`) takes two filter-shaped cohorts
that can span different dimensions — `{region: [APAC]}` vs `{segment: [Enterprise]}` — which
share no key yet can still share rows. A "difference" between two groups that are partly the
same rows is not a clean difference, and a model reading `gap_abs` without knowing that would
over-claim.

**Options considered.**
- (a) Reject when the two filter mappings are structurally equal (catches only the exact
  self-comparison case).
- (b) Resolve both cohorts to their actual row index against the requested time window and
  reject on a non-empty intersection.
- (c) Always compute the comparison, carrying `overlap_rows`/`overlap_pct` as typed fields so
  the model can discount it itself.

**Choice made.** (b), confirmed with the user directly (see the batch-10 planning transcript).

**Reason.** (a) misses the real case entirely — `{region: [North]}` vs `{segment: [Enterprise]}`
are structurally unequal dicts that still overlap by row. (c) is more permissive but risks a
model asserting a confident-sounding gap between two groups that are 40% the same underlying
rows; the house convention throughout this layer (`compare_segments`'s own member-identity
check, `cross_tabulate`'s `dim_a == dim_b` rejection) is to refuse a request whose comparison
is not well-posed rather than answer it with a caveat attached. The check is repeated once per
requested time window (`period_a`, and again for `period_b` when given), because whether two
filters' row sets intersect can genuinely depend on which period is in scope — a pair of
filters with zero overlapping rows in one quarter can overlap in another.

**Consequence/trade-off.** A model asking to compare two cohorts it did not realize overlap gets
a typed, recoverable `InvalidArgumentError` naming the overlapping row count rather than an
answer — it must narrow one of the two filters and re-ask, which costs a turn of the agent loop
but keeps every `CohortComparison` the tool does return well-posed by construction.

---

## 31. `compare_segments` publishes three gap measures, not one

**Context/problem.** A single "how does A compare to B" question wants both a phrasing a reader
expects ("APAC is 20% below EMEA") and a well-behaved number a model can reason about
symmetrically (swap A and B and get the negated answer). `pct_change(a, b)`
(`engines/metrics.py:454`) supplies the first and is not antisymmetric —
`a/b - 1 != -(b/a - 1)` — so it cannot supply the second.

**Options considered.**
- (a) Return only `pct_change(a, b)`, matching how `ComparisonEngine.compare`'s `change_pct`
  already works for a period-over-period delta.
- (b) Return only an antisymmetric measure (absolute gap, or a symmetric percent difference),
  dropping the reader-expected phrasing.
- (c) Return all three: `gap_abs` (exact, antisymmetric), `relative_gap_pct` (symmetric percent
  difference against the mean magnitude of both sides, antisymmetric), and `a_vs_b_pct`
  (`pct_change(a, b)`, deliberately not antisymmetric), each field's antisymmetry status
  documented and pinned by a test.

**Choice made.** (c).

**Reason.** (a) alone would make `direction` and the sign of the reported percent disagree
under a naive swap in a way that is easy to get wrong downstream (the model prompt, or a caller
composing two calls); (b) alone loses the phrasing that answers what was actually asked.
Publishing all three, each labeled by what it is, lets the model pick "APAC is 20% below EMEA"
for prose while `gap_abs`/`relative_gap_pct` remain safe to combine algebraically (sum, sort,
threshold) without the sign convention silently depending on argument order.

**Consequence/trade-off.** Three numbers for one comparison is more surface area than a single
field, and a model or a future caller could reach for `a_vs_b_pct` in a context that assumes
antisymmetry. The module docstring and a dedicated test
(`TestGapAntisymmetry.test_a_vs_b_pct_is_deliberately_not_antisymmetric`) exist specifically so
that risk is documented and checked, not merely hoped around.

---

## 32. `compare_segments` does not wrap `analysis.member_change_table`

**Context/problem.** The integration plan's board pointed I6's `compare_segments` at wrapping
`analysis.member_change_table` (`engines/analysis.py:114`), the way `decompose_by_dimension`
wraps `observe.decompose_dimension`. That function returns a bare `DataFrame` with no
`TimeSelection`, no validation airlock, string-keyed `{metric}__cur/__base/__chg` columns, and a
`resolver` positional argument that C2 only just made mandatory at every call site across the
codebase.

**Options considered.**
- (a) Wrap `member_change_table` as the board specified, reshaping its DataFrame output into the
  typed dataclasses this layer uses.
- (b) Build `compare_segments` directly on `QueryEngine.query`, the same primitive
  `breakdown.py`, `series.py`, and `drivers.py` already build on.

**Choice made.** (b), superseding the board's pointer — the same call `breakdown.py` made about
`drivers.cell_delta_grid` and `correlate.py` made about `analysis.correlate`.

**Reason.** Every other tool in this layer inherits the validation airlock, ratio-correct
aggregation, and `TimeSelection` reporting by going through `QueryEngine`; wrapping
`member_change_table` instead would mean re-deriving all three around a function that was never
built to provide them, for no arithmetic this module actually needs (`member_change_table`
scans every member of a dimension to build a full comparison table, where `compare_segments`
only ever needs the two named members). Putting both members inside the query's own `filters`,
not just `group_by`, means the query produces at most two cells regardless of the dimension's
total cardinality — comparing two members of a 500-member dimension costs the same as two
members of a five-member one.

**Consequence/trade-off.** `analysis.member_change_table` remains unwrapped by this layer; it
stays in use by `engines/contest.py`'s existing cross-sectional consistency check (retired at
X2/A9, not this batch) and by `test_resolver_propagation.py`'s regression coverage for the C2
ratio-KPI bug. No behavior of the legacy function changes.

---

## 33. `compare_to_seasonal_norm` bands the level, not the transition distribution

**Context/problem.** The integration plan's board pointed I7's `compare_to_seasonal_norm` at
exposing `observe.py:189-193` -- the same-quarter-transition path `assess_significance` already
uses internally to decide whether a change is unusual. But `ComparisonEngine.significance`
(`agent/compare.py`) already publishes that entire result (`method="seasonal_robust_z"`,
`same_quarter_points`, `expected_value`, `normal_range`, `robust_z`, `power`).

**Options considered.**
- (a) Wrap `observe.py:189-193` as the board specified, giving the model a second tool name for
  fields `significance()` already returns.
- (b) Build a genuinely different tool: detrend with the shared `trend.theil_sen`, compute a
  robust seasonal index per phase-of-cycle, and band the requested period's *level* against
  trend-plus-season, independent of `significance()`'s transition-distribution approach.

**Choice made.** (b), superseding the board's pointer -- the same call made about
`compare_segments` in decision 32.

**Reason.** `assess_significance` distributes over *changes*: it asks whether this quarter's
delta from the previous one is unusual against the history of that same delta. It has no way to
answer "is this quarter's *value* normal for this point in the cycle" -- a KPI that dips by the
same amount every Q3 has a perfectly ordinary Q2->Q3 change every year, and (b) is needed to
answer whether the resulting *level* is itself normal. Measured directly
(`TestCompareToSeasonalNormIsNotSignificance`): a KPI with a clean, perfectly regular +60 bump
every Q3 that fails to bump in its final year is called "normal" by `significance()`, because a
perfectly regular historical bump has zero historical variance, so the seasonal-transition test
degenerates (`sig_seasonal <= 0`) and falls back to the noisy all-transitions distribution, which
does not notice one missing jump. `compare_to_seasonal_norm` flags the same period directly by
comparing its level against the fitted trend-plus-seasonal-index expectation. Publishing a thin
re-expose of (a) alongside `significance()`'s already-overlapping fields would also be exactly the
kind of tool-selection hazard `concentration.py`'s own header warns about.

**Consequence/trade-off.** Two tools now judge "is this normal" from different statistical bases
(a transition distribution vs. a level band), and a model composing both must understand they can
legitimately disagree -- which is the point, not a defect, but it is one more thing a system
prompt has to explain. `compare_to_seasonal_norm` also duplicates the trend/season fit
`detect_seasonality` performs rather than sharing a cached result across a single tool call,
since each tool call is independent and there is no cross-call cache in this layer yet.

---

## 34. Seasonality is gated on autocorrelation; seasonal strength is descriptive only

**Context/problem.** The obvious way to detect seasonality is an STL-style strength ratio,
`1 - sigma(residual after removing trend and season) / sigma(residual after removing trend
only)`, with a fixed cutoff. Before writing `detect_seasonality`, this was implemented and
calibrated against a Monte Carlo null and the real sample fixtures.

**Options considered.**
- (a) Gate on the in-sample strength ratio with a fixed cutoff (e.g. `>= 0.60`), the STL-style
  default.
- (b) Gate on the autocorrelation of the detrended series at the cycle lag; report strength
  separately, computed leave-one-cycle-out, as a descriptive magnitude only.
- (c) Add `scipy`/`statsmodels` now (a real periodogram, ACF with significance testing, or
  `seasonal_decompose`), pulling the dependency forward from where the master plan sequences it
  (X3).

**Choice made.** (b).

**Reason.** (a) fails on two independent measurements. First, size: a Monte Carlo null of pure
Gaussian noise (300 trials per cell) shows a `strength >= 0.60` gate firing on 32% of pure noise
at n=12 quarters (three cycles -- an ordinary ask), and its power on a genuine strong seasonal
signal *falls* from 0.69 to 0.57 as n grows from 14 to 24, because more observations per phase
shrink the in-sample ratio -- a threshold whose power decreases with sample size is not
defensible at any cutoff. Second, and more concretely: retail's own quarterly `revenue` --
which has essentially zero real seasonality (autocorrelation at the cycle lag ~0.035) -- scores
an in-sample strength of 0.668, above every commonly cited STL cutoff. A fixed strength gate
would announce, on the exact dataset this product demos on, that revenue is seasonal when it is
not: the precise credibility failure this task exists to prevent, inverted. The ACF gate
(threshold `0.30`) was calibrated against the same null and measured at <=7% false-positive
across n in {12, 14, 24, 42, 182} with 0.84-0.99 power on genuine seasonality, and correctly
returns `is_seasonal=False` for retail `revenue`. Leave-one-cycle-out strength was chosen over
in-sample for the same reason the gate changed: each point's index comes from every *other*
observation of its phase, so a phase median fitted on nothing but noise cannot inflate the
reported number -- it drops retail `revenue`'s reported strength to 0.335, agreeing with the ACF
verdict instead of contradicting it. (c) was rejected on the same grounds `agent/trend.py`
already declined `scipy` for Theil-Sen: `requirements.txt` carries only numpy and pandas, and the
master plan sequences p-values, confidence intervals, and multivariate statistics at X3
specifically so that dependency is added once, deliberately, not piecemeal six tasks early. A
seeded permutation test was also measured as a fourth option: correct size (0.05-0.09) but only
0.60 power at n=14 against the ACF gate's 0.97, at 99x the per-call cost, with determinism
resting on NumPy's RNG stream staying stable across versions -- strictly dominated by (b).

**Consequence/trade-off.** `seasonal_strength` is published but must never be used as a
detection condition by any future caller -- `TestStrengthIsDescriptiveNotAGate` pins this by
asserting retail `revenue` is *not* seasonal despite a strength that would pass a naive cutoff,
specifically so a "helpful" refactor back toward gating on strength fails immediately and on the
demo dataset itself. A calibrated p-value (replacing the fixed ACF threshold with something
statistically principled) is deferred to X3, where `scipy` is already scheduled to arrive.

---

## 35. A bridge step for a ratio KPI is `rate_effect + mix_effect`; the others step is summed, not rescaled

**Context/problem.** `bridge_periods` (I8) needed a per-member "step" value that sums exactly to
a KPI's total change, and a way to fold a truncated tail of members into a single "others" step
without breaking that sum. The obvious choices -- `MemberContribution.change_abs` for the member
step, and `OthersRollup.change_abs` or a `contribution_pct` rescale for the tail -- were measured
against the real retail fixture before being rejected.

**Options considered, for the member step.**
- (a) `MemberContribution.change_abs` for every KPI kind, matching the field `decompose.py`
  already surfaces per member.
- (b) `change_abs` for `sum`-kind KPIs; `effects.rate_effect + effects.mix_effect` for
  `ratio`-kind KPIs.

**Options considered, for the others step.**
- (a) `OthersRollup.change_abs`, the field `decompose_by_dimension` already computes for the tail.
- (b) `contribution_pct / 100 * total_change`, arithmetically exact against `observe`'s internal
  `total_delta`.
- (c) Ask `decompose_by_dimension` for every member (`max_items=10**9`), compute each member's
  own true step value (per the member-step choice above), rank and split head/tail here, and sum
  the tail's own step values exactly.

**Choice made.** (b) for the member step; (c) for the others step.

**Reason.** For a ratio KPI, `change_abs` is `(rate_cur - rate_base) * scale` -- that member's
own rate change, which does not sum to the KPI's total change. What sums, exactly, is the
Bennet-style split `w_cur*(r_cur - r_base) + r_base*(w_cur - w_base) = w_cur*r_cur -
w_base*r_base`, whose members telescope to `R_cur - R_base` with no interaction term needed.
Measured on `fulfillment_rate` by `product`, 2026-Q2 vs 2026-Q1: two of three members have
`change_abs` of exactly `0.0` while their `rate_effect + mix_effect` is `+4.38` and `+3.12` -- (a)
would render two of three bars as zero, and on `gross_margin_pct` by `product` the naive sum is
`+0.308` against a true change of `-0.342`, so the waterfall would show the margin *rising* when
it fell. For the others step, (a) inherits the same defect (measured: `0.040` reported against a
true `1.219` on the same KPI, a 30x error). (b) is arithmetically exact but goes through
`observe`'s internally-computed `total_delta` rather than `DimensionDecomposition.change_abs` --
the two agree today but nothing pins them together -- silently drops any member whose
`contribution_pct` was `None` (`observe.py`'s rollup does `sum(r["contribution_pct"] or 0 ...)`),
and inherits a head/tail cut sorted on `abs(change_abs)`, which for a ratio ranks by rate change
rather than waterfall contribution -- the two zero-`change_abs` members above would sort into the
tail *first* under that cut. (c) costs nothing extra, since `decompose_by_dimension` already
computes every member and only truncates afterwards; it fixes the ranking and makes the others
step exact by construction rather than by a rescaling identity.

**Consequence/trade-off.** `bridge.py` calls `decompose_by_dimension` with `max_items=10**9`
rather than `None` to get every member, specifically to avoid a duplication bug fixed alongside
this batch in `observe.decompose_dimension` itself: `rows[:None]` and `rows[None:]` both return
every row, so `max_items=None` used to fold the entire member list into a same-named "Other" row
on top of the members it was meant to summarise. A member whose step value cannot be computed (a
`None` from a zero-denominator ratio component) is excluded from both the shown steps and the
others sum, so its contribution is folded into `unexplained` rather than fabricated or silently
dropped from the total.

---

## 36. `unexplained` is a terminal step, not only a top-level field

**Context/problem.** `decompose_formula`'s convention publishes `unexplained` as a top-level
scalar residual. `bridge_periods` needed to decide whether to keep that convention, or also
surface the residual as the last entry in the step sequence itself.

**Options considered.**
- (a) Top-level field only, matching `decompose_formula` exactly.
- (b) Both: a terminal `BridgeStep` with `kind="unexplained"`, in addition to the top-level field.

**Choice made.** (b).

**Reason.** Decision 37 (mean-kind KPIs) surfaced a residual that is three times the total change
with the opposite sign when an unrefused tool is forced through a dimension whose members do not
sum to the total. A waterfall's entire purpose is to be read visually, step by step, until it
reaches the end value -- a residual that large has to be visible in the sequence a reader actually
looks at, not only in a sibling scalar a renderer might not surface. Emitting it always when
`status == "ok"`, even at exactly `0.0`, keeps the payload shape stable so a renderer can hide a
zero-valued bar trivially rather than branching on whether the field is present.

**Consequence/trade-off.** Every successful bridge has one more step than its named components,
and a consumer summing `steps` to reconstruct `end_value` no longer needs to separately add
`unexplained` -- but a consumer that already knows about `decompose_formula`'s top-level-only
convention has to learn the new field exists as a step too. The top-level field is kept
regardless, so a caller reading only that (to match `decompose_formula`'s shape) still gets the
right number.

---

## 37. Mean-kind KPIs are refused on the dimension axis of `bridge_periods`

**Context/problem.** `observe.decompose_dimension` -- the arithmetic `decompose_by_dimension`
wraps and `bridge_periods` builds on -- routes `kind in ("sum", "mean")` down the identical
additive branch, silently treating a mean-kind KPI's per-member group averages as if they summed
to the KPI's overall value. They do not: sum(group means) != overall mean in general.

**Options considered.**
- (a) Leave the dimension axis open to every KPI kind, inheriting whatever `decompose_by_dimension`
  returns, and let the (real, but unexplained-only) residual absorb the error.
- (b) Refuse the dimension axis outright for `kind() == "mean"`, with a typed `unsupported_kind`
  status, and point the caller at the formula axis instead.
- (c) Refuse whenever `is_additive()` is false, which also covers mean-kind KPIs.

**Choice made.** (b).

**Reason.** Measured on retail `inventory_units` (kind `mean`) by `region`: the total change is
`-12.48` while the naive member sum is `-49.92` -- the residual an unrefused (a) would report is
*three times the total change, with the opposite sign*, which decision 36 exists specifically to
make visible rather than hide, but a residual that badly wrong is not a "step-by-step
reconciliation" a reader can trust at all; refusing outright is more honest than surfacing a
technically-unconditional but practically meaningless waterfall. (c) was rejected because
`is_additive()` is `False` for ratio KPIs too, and the dimension axis is exactly the one this
module gets *right* for a ratio KPI (decision 35's rate/mix split reconciles exactly) --
gating on `is_additive()` would refuse the axis's best-supported case along with its worst one.
Gating on `kind() == "mean"` specifically refuses only the KPIs that are actually wrong, matching
the precedent decision 29 already set (`measure_concentration`'s additivity gate reads
`ContractAPI.is_additive`, not `kind() == "sum"`) by choosing the predicate that actually answers
the question being asked, rather than reaching for whichever one happens to be already in scope.

**Consequence/trade-off.** A mean-kind KPI can still be bridged along its formula axis (most
mean-kind KPIs in the sample fixtures are single bare fields, so that resolves to `single_term`
rather than a real decomposition) -- the dimension axis is the only one closed off, and only for
this one KPI kind. The underlying bug in `observe.decompose_dimension` is not fixed here: it is
inherited by every other caller of that function (`agent/drivers.py`, `agent/concentration.py`,
`engines/observe.observe`'s own per-dimension driver loop), all of which currently consume
`contribution_pct` from the same wrong branch. Recorded here so it is not silently rediscovered;
fixing `observe.decompose_dimension` itself is out of scope for this batch.

---

## 38. `cross_correlate_lagged` differences with `.diff()`, not `pct_change`

**Context/problem.** X1's `cross_correlate_lagged` needed a transform to difference two KPI
series before correlating them at each candidate lag. `agent/correlate.py`'s own `correlate_kpis`
already differences with `pct_change` and publishes `transform="change"`; the legacy
`analysis.lead_lag` uses absolute first differences (`.diff()`). Using `pct_change` was tried
first, on the theory that it would keep the layer's meaning of "change" consistent and make the
tool's lag-0 answer numerically equal to `correlate_kpis(mode="time_series")`.

**Options considered.**
- (a) `pct_change`, matching `correlate_kpis`.
- (b) `.diff()`, matching legacy `analysis.lead_lag`.
- (c) A `transform` parameter exposing both.

**Choice made.** (b), reversing the initial choice of (a) once it was measured against the real
fixtures.

**Reason.** Measured on retail `revenue ~ stockout_events` -- the pair carrying the planted
supply-disruption signal -- the association *halves* under `pct_change` (`r=-0.583` with `.diff()`
vs `r=-0.315`). On hospital weekly `deaths` (zero in 17 of 183 weeks), `pct_change` destroys 16 of
182 pairs at every lag from near-zero denominators, with excursions up to 300%. Because this tool
selects the best lag by `argmax |r|`, one such blow-up biases *which lag wins*, not merely the
estimate -- a selection bias, not added noise. The "scale-free" argument for `pct_change` buys
nothing here either: Pearson r is already invariant to positive affine rescaling of either series,
so the two transforms differ only through a time-varying denominator, which is exactly the source
of the blow-up. (c) was rejected as a knob the model has no basis to choose between, and doubles
the test matrix for a choice the fixtures already settle.

**Consequence/trade-off.** `cross_correlate_lagged` is no longer numerically equal to
`correlate_kpis(mode="time_series")` at lag 0 -- the two tools now deliberately disagree about
*how* a period-over-period change is measured. What is pinned instead is the *pairing*: same `n`,
same `pairs_dropped_at_gaps`, same `status`, at lag 0. `transform` publishes `"difference"` here,
distinct from `correlate.py`'s `"change"`, so a caller reading both fields is never told the same
word means two different computations.

---

## 39. `check_temporal_precedence` measures lag in calendar ordinals; positive means the cause led

**Context/problem.** `compare_onsets` (`engines/analysis.py:81`) computes its lag as
`cause_onset["index"] - kpi_onset["index"]`, where `index` counts *held* periods in whatever frame
`detect_onset` was given -- a series with a hole understates the true elapsed lag. `analysis.
lead_lag` and `driver_graph._lagged_r`, independently, both use the opposite sign: a positive lag
means the cause moved first.

**Options considered.**
- (a) Reuse `compare_onsets`'s own `cause.index - kpi.index` convention.
- (b) Compute the lag from `period_ordinal(grain, period)` differencing, with the sign flipped to
  match the two existing lagged-correlation tools: `kpi_ordinal - cause_ordinal`, positive when the
  cause is earlier.

**Choice made.** (b).

**Reason.** Ordinals are gap-honest where `index` is not -- verified, `period_ordinal("week",
"2026-05-04") - period_ordinal("week", "2026-04-06") == 4`, the true calendar span, regardless of
any hole elsewhere in either series. Having `check_temporal_precedence` and `cross_correlate_lagged`
agree on what "positive" means is more valuable than matching a legacy convention that only one
(soon-to-be-superseded) function uses -- a model reasoning about both tools' output in the same
turn must not have to remember two sign conventions for the same word.

**Consequence/trade-off.** This module's `lag_periods` carries the opposite sign from
`compare_onsets`'s `lag_weeks` during the migration window covered by decision 44. A cross-tool
test (`TestSignConventionAgrees`) pins that `check_temporal_precedence` and `cross_correlate_lagged`
agree with each other, since that is the invariant that matters going forward.

---

## 40. `check_temporal_precedence` requires an explicit direction and a scoped window

**Context/problem.** `TrendEngine.detect_changepoint(direction="any")` returns the *earliest*
onset across both directions, and `detect_onset` dates the first breach of the baseline formed by
the *opening* `baseline_periods` of whatever window it is handed -- not the move under
investigation. Both were measured as live traps before this tool's signature was finalised.

**Options considered.**
- (a) Default both `kpi_direction`/`cause_direction` to `"any"` and default `time_filter` to
  all-history, matching `detect_changepoint`'s own defaults, for the simplest possible call.
- (b) Require explicit directions (still defaulting to `"any"` if the caller genuinely does not
  know, but documented as a trap) and never default `time_filter` to all-history internally --
  the caller must scope the window itself -- plus a per-side `onset_at_window_edge` flag.

**Choice made.** (b).

**Reason.** Measured on retail: `direction="any"` on `revenue` selects the 2026-03-02 seasonal
upturn instead of the 2026-04-20 decline the caller almost certainly means. Separately, on
`units_sold` vs `stockout_events` (`product=Product A`), the full 183-week history reports the
cause leading the KPI by **156 weeks**, while the same pair scoped to a 39-week window around the
planted disruption reports the KPI leading by 20 weeks instead -- opposite verdicts, same KPIs,
same grain. Neither number is trustworthy without knowing the window it came from. `_onset_at_
window_edge` catches this: the cause's onset in the full-history case sits at week 18 of 183 (9.8%
into the window), well past the strict `baseline_periods + persistence` floor (10) but still
clearly an artifact of a baseline built from three years before the real event -- hence the
`WINDOW_EDGE_FRACTION = 0.2` relative test alongside the absolute one.

**Consequence/trade-off.** Every caller of `check_temporal_precedence` must pass directions (or
consciously accept `"any"`'s documented risk) and think about the search window rather than
getting a default all-history answer for free. `onset_at_window_edge` is a heuristic, not a proof
-- it does not catch every possible spurious onset, only the two measured failure modes (too little
runway, and too early a fraction of a long window).

---

## 41. `test_reverse_causation` reads `RelationGraph.components()`, not `related()`

**Context/problem.** `mechanism_check` (`engines/contest.py:284`) gates a real score penalty and a
confidence cap on `bool(hypothesis.get("reverse_causation_risk"))` -- a free-text LLM field used
only for its truthiness. The structural fact it is trying to approximate -- are the two KPIs
algebraically linked -- already exists in the contract.

**Options considered.**
- (a) `RelationGraph.related(key)`, the broader traversal that also surfaces `derivation` and
  `shared_source_field` relations.
- (b) `RelationGraph.components(key)`, the narrower formula-only traversal.

**Choice made.** (b) for the *directional* mechanical check, with (a) restricted to
`("derivation", "shared_source_field")` as a fallback for the direction-less "shared inputs" case.

**Reason.** `related()`'s `derivation` edges come from the symmetric predicate
`_derivation_related(spec, other_spec)` and can never say which KPI is downstream -- measured,
`related("revenue")` on the retail sample returns five `derivation` edges and nothing else.
`components(key)` is the one relation in the codebase that is genuinely directional: a hit naming
`cause_kpi` inside `components(kpi_key)` means the KPI is built from the cause's formula, and vice
versa. Verified against the contract: `revenue` vs `avg_order_value` resolves to
`cause_derived_from_kpi` / `kpi_derived_from_cause` depending on argument order (`avg_order_value`'s
formula has `revenue` as its numerator), and `gross_profit` / `gross_margin_pct` resolves to
`mutual` (the equivalence rule makes each a component of the other).

**Consequence/trade-off.** A pair linked only by `derivation` or `shared_source_field` (measured:
`revenue_less_marketing_spend` vs `gross_profit`) is reported as `"shared_inputs"`, not
`"definitional"` -- it names a mechanical link honestly without claiming a direction it cannot
support. The timing signal (`cross_correlate_lagged`) is computed once, not once per direction: the
lag profile of `(kpi, cause)` at `+L` is the profile of `(cause, kpi)` at `-L`, verified to machine
precision, so a second call would buy nothing and is asserted against in
`test_the_timing_call_is_made_exactly_once`.

---

## 42. `find_counterexamples` is polarity-aware; `analysis.counterexamples` is fixed in place

**Context/problem.** `analysis.counterexamples` selects members with `kpi_chg <= kpi_drop_pct`
(`kpi_drop_pct` negative) -- it can only ever find members whose KPI *fell*. For a lower-is-better
KPI, "moved badly" means the value *rose*, so on `readmission_rate`, `mortality_rate`,
`return_rate` and similar KPIs it looks in the wrong tail entirely.

**Options considered.**
- (a) Leave `analysis.counterexamples` untouched; have the new `find_counterexamples` reimplement
  polarity-aware selection independently.
- (b) Add a `kpi_higher_better: bool = True` parameter to `analysis.counterexamples` itself
  (default preserves every existing caller's behaviour byte-for-byte), have
  `contest.consistency_check` pass `higher_is_better(kpi, schema.contract_resolver)`, and have
  `find_counterexamples` read the same polarity via `ContractAPI.polarity`.

**Choice made.** (b).

**Reason.** Measured on the hospital fixture, `readmission_rate` (a KPI that is bad when it rises)
vs `avg_length_of_stay`: the unfixed function returns `[]` -- it finds *no* counterexamples on data
where every department's readmission rate in fact worsened. This is not a missing capability in a
new tool, it is a wrong number already live in `contest.consistency_check`, feeding a real
`score_hypothesis` penalty (`against += 0.6 * n_counter`) today. Fixing it in place costs one
parameter and one caller-side line, changes no payload shape, and removes the wrong number from the
pipeline that is still running (decision 44) rather than leaving it to be fixed only when that
pipeline is eventually retired.

**Consequence/trade-off.** `analysis.counterexamples`'s default (`kpi_higher_better=True`) means
every *other* undocumented caller (there are none besides `contest.py` and its own test) keeps
today's behaviour unless it opts in. `find_counterexamples` itself does not call the legacy
function at all -- it recomputes the same selection over `ContractAPI`-sourced changes, matching
the layering precedent from decision 32.

---

## 43. `test_consistency_across_dimension` reports a three-bucket rate and a pooled r; `min_n` defaults to 3

**Context/problem.** With exactly two periods, a per-member "relationship strength" can only mean
sign agreement between the KPI's and the cause's change. The obvious design collapses this into a
single "consistency rate" and reuses `correlate.DEFAULT_MIN_N` (6) for the pooled correlation.

**Options considered.**
- (a) A two-bucket rate (agree / disagree) plus a pooled r at `min_n=6`.
- (b) A three-bucket rate (`same_sign` / `opposite_sign`, with `cause_flat` and `kpi_flat` excluded
  from the rate's denominator) plus a pooled r at a dataset-appropriate `min_n=3`.

**Choice made.** (b).

**Reason.** Measured on retail `product`, `fulfillment_rate` is exactly `0.00` for two of three
members -- the cause said nothing there, which is `find_counterexamples`' business, not a sign
disagreement; folding it into a two-bucket rate would double-count one finding as two. Separately,
no sample fixture has six cross-sectional members (retail `region` 4, `product` 3; hospital
`department` 3-4; school `grade` 4), so `min_n=6` would make this tool report `insufficient_n` on
every dataset the project owns; `min_n=3` mirrors `contest.MIN_CROSS_SECTION_MEMBERS`. The two
statistics are reported together, deliberately, because they can and do disagree: retail `region`,
`revenue ~ fulfillment_rate` gives `consistency_rate=1.0` (every region moved the same way on both
KPIs) alongside `pooled.r=-0.953` (ranking magnitudes across a one-point cause spread against a
thirty-point KPI spread -- noise, not a real association). Collapsing to one number would hide
whichever one the model did not see.

**Consequence/trade-off.** A caller must read both `consistency_rate` and `pooled.r` and understand
they answer different questions; `co_movement_strength = abs(2*rate - 1)` exists specifically so a
rate of `0.0` (consistently inverse -- the mechanically correct relationship for e.g. margin vs
cost) is not misread as "inconsistent".

---

## 44. `test_holdout_segments` delegates to `compare_cohorts` but publishes its own `did_pct`

**Context/problem.** A holdout split ("the cause moved here" vs "it did not") is a predicate over a
metric's own per-member change, which no dimension-value filter dict can express -- but the
*result* of that split, two member lists, is a legal `compare_cohorts` filter pair. What
`compare_cohorts` reports by default is a level gap between the two groups.

**Options considered.**
- (a) Derive the split, call `compare_cohorts`, and report its `gap` as the holdout finding.
- (b) Derive the split, delegate to `compare_cohorts` for the arithmetic and antisymmetry
  guarantees, but add the module's own `did_pct = a.change_pct - b.change_pct` as the headline
  field.

**Choice made.** (b).

**Reason.** Measured on the retail product split (`{Product A}` affected vs
`{Product B, Product C}` unaffected): `gap.gap_abs = -1,109,482`, which is a group-size artefact (one
member against two, at very different revenue scales), not a measure of whether the cause explains
anything. The scale-free difference-in-differences, `did_pct = -25.30` percentage points, is the
actual answer to "did the KPI move differently where the cause was present". Delegating the
underlying comparison to `compare_cohorts` (rather than reimplementing gap arithmetic) inherits
`SegmentGap`'s pinned antisymmetry and the row-index disjointness check for free -- the latter
becomes a trivial assertion here, since a derived split is disjoint by construction.

**Consequence/trade-off.** The tool never constructs an empty filter dict when one side of the
split is empty (`compare_cohorts({})` raises `InvalidArgumentError`, the wrong signal for a real,
reportable "every member's cause moved" outcome) -- `no_affected_members` / `no_unaffected_members`
are returned directly instead. `verdict` is derived purely from `|did_pct|` against
`min_material_change_pct`, not from each side's `is_unfavourable` sign, because the measured
retail case has *both* groups moving unfavourably (revenue fell in both) at very different
magnitudes -- gating on the unfavourable sign alone would have misclassified it as
`"kpi_moved_regardless"` when the differential is in fact the whole finding.

---

## 45. X1/X2 are additive; `contest.py`'s legacy checks are retired at A9, not this batch

**Context/problem.** `engines/contest.py`'s `temporal_check`, `consistency_check`,
`mechanism_check` and `score_hypothesis` are the functions X1 and X2 supersede. The master plan's
A9 is the designated deletion batch, but each batch since Layer O has had the option to retire what
it replaces early.

**Options considered.**
- (a) Retire the three checks and `score_hypothesis`'s associated penalties now, repointing
  `contest.py` at the new agent tools.
- (b) Ship `agent/temporal.py` and `agent/consistency.py` as pure additions; leave `contest.py`
  untouched.

**Choice made.** (b).

**Reason.** Measured blast radius of (a): `frontend/src/components/HypothesisCard.jsx` reads eleven
distinct legacy fields (`temporal.detail`, `consistency.correlation.interpretation`,
`mechanism.conclusion`, `mechanism.lead_lag.detail`, among others), `OnsetChart.jsx` reads the onset
series shape, `act.py:220` reads `consistency.counterexamples`, and `test_pipeline.py` /
`test_rca_ground_truth.py` / `scripts/validate_rca.py` are the project's only end-to-end causal
correctness harness. `contest()` still needs *some* `score_hypothesis` to produce a confidence
number, and the new tools deliberately produce none -- that judgment is the model's job now, and
this batch has no replacement for the ledger. The three legacy checks are called from nowhere but
`contest.contest()` and two tests, so leaving them costs nothing ongoing. This mirrors decision 32's
precedent exactly (`analysis.member_change_table` also left standing, "retired at X2/A9, not this
batch").

**Consequence/trade-off.** `analysis.counterexamples`, `analysis.correlate`, `analysis.lead_lag`,
`compare_onsets` and `member_change_table` are now superseded but load-bearing, alongside their
X1/X2 replacements, until A9's `HypothesisCard` rewrite retires them. The one exception taken now is
decision 42's in-place polarity fix to `analysis.counterexamples`, because it corrects a wrong
number in a pipeline that is still running, at zero cost to any consumer's payload shape.

## 46. `scipy` is adopted at X3; the robust estimators stay hand-rolled

**Context/problem.** Three modules deferred the same dependency to this batch in turn:
`agent/trend.py:11-15` hand-fitted Theil-Sen rather than take `scipy` "six batches before the master
plan calls for one"; `agent/seasonality.py:63-66` skipped a permutation p-value calling a calibrated
one "the natural `scipy`-backed task at X3"; `agent/correlate.py:40-42` shipped a bare `r` with "no
p-value, confidence interval, or Fisher-z CI ... sequenced at X3". X3 cannot be built without
deciding.

**Options considered.**
- (a) Add `scipy`, and use exact `stats.t` / `stats.norm` distributions.
- (b) Stay on numpy: Fisher-z with a normal approximation for both p and CI, hand-rolling the normal
  CDF from `math.erf`.

**Choice made.** (a). `scipy==1.18.0` in `backend/requirements.txt`, pinned to the version the
environment already resolves alongside the existing `numpy==2.2.1` / `pandas==2.2.3` pins.

**Reason.** (b) is only accurate asymptotically, and the samples these fixtures actually hold are
the opposite of asymptotic: the retail `region` axis has four members, the quarterly axis thirteen
differenced pairs. A normal approximation is least trustworthy exactly where this batch's honesty
guarantees matter most. `scipy.stats.pearsonr` also becomes the test oracle -- `p_value_for_r`
matches it to twelve decimal places across 25 random samples -- which no hand-rolled version could
be checked against without reimplementing the incomplete beta function to check it with.

**Consequence/trade-off.** One more dependency in a deliberately thin analysis stack, and X5's
bootstrap CIs and I7's calibrated seasonality p-value now have a natural home. `trend.theil_sen`
and `drivers.robust_sigma` are **not** rewritten onto `scipy`: they are robust estimators chosen for
their breakdown behaviour, not approximations of something `scipy` does better, and swapping them
would change measured outputs that existing tests pin.

## 47. Significance composes `CorrelationEngine`; it never recomputes `r`

**Context/problem.** `test_statistical_significance` needs `r` and `n`. It could compute them from
the frame directly -- it has the same `QueryEngine` and `SeriesEngine` available -- or ask
`CorrelationEngine` for them.

**Options considered.**
- (a) Recompute, so significance owns its own arithmetic end to end.
- (b) Call `correlate_kpis` / `correlate_kpi_matrix` and qualify what comes back.

**Choice made.** (b), and `agent/confounders.py` follows the same rule for `p_value_for_r` and
`fisher_interval`.

**Reason.** Under (a) a model that calls both tools on the same pair can be handed two different
`r`s -- from a scoping default drifting apart, an auto-dimension choice differing, or a gap rule
diverging -- and has no way to tell which to believe. Under (b) that is structurally impossible.
This is decision 32's reasoning applied one layer up, and the same rule `agent/consistency.py`
follows in delegating to `CorrelationEngine` and `SegmentEngine` rather than re-deriving them.
`test_a_sweep_row_matches_the_standalone_call_for_that_pair` pins it.

**Consequence/trade-off.** X3 inherits every scoping decision `CorrelationEngine` makes, including
`_auto_dimension`'s choice, and inherits its `pearson_correlate` rounding: `p` is computed from the
published three-decimal `r`, so an `|r|` that rounds to 1.0 reports `p = 0.0` rather than ~1e-150.
Internal consistency between the two published figures was judged worth more than the tail digit,
and the docstring says so.

## 48. Multiplicity is corrected with BH-FDR, and **both** verdicts are published

**Context/problem.** `correlate_kpi_matrix` sweeps up to `MAX_MATRIX_CANDIDATES = 30` candidates in
one call. At alpha = 0.05, thirty tests against a true null produce roughly 1.5 "significant"
results by chance -- and the legacy `contest.score_hypothesis` gate (`|r| >= 0.5`, worth
`+1.1*|r|`) has no notion of a family at all.

**Options considered.**
- (a) Raw p-values only; leave multiplicity to the model's judgment.
- (b) Benjamini-Hochberg, reporting only the corrected verdict.
- (c) Benjamini-Hochberg, reporting `significant_raw` and `significant_fdr` side by side.

**Choice made.** (c). `q_value`, `family_size` and `alpha` all travel in the payload; a single-pair
call is `family_size = 1` and `q == p`.

**Reason.** (a) reliably hands the model a false driver flagged as significant --
`test_thirty_null_candidates_produce_raw_hits_and_no_fdr_hits` measures exactly that on 30 uniform
p-values. (b) hides *why* a candidate failed: a model cannot tell an unimpressive effect from a
real one that lost to the size of the family it was tested in, and those warrant different next
moves (drop it, versus test it alone). A `None` p-value -- a candidate whose correlation had no
usable sample -- is excluded from `m` rather than counted, since a test that never ran cannot
consume a share of the false-discovery budget.

**Consequence/trade-off.** `q_value` is only meaningful relative to the family the caller chose, so
a model that sweeps candidates in two calls of fifteen gets weaker correction than one call of
thirty. `family_size` is published so that is visible rather than implicit.

## 49. Partial correlation requires one listwise-aligned sample, published as `PairedSample`

**Context/problem.** `first_order_partial` reads three pairwise `r`s. Computed independently, each
lands on its own pairwise-finite overlap -- which is what `pearson_correlate` does, correctly, for a
single correlation. Three `r`s from three different overlaps do not form a positive semi-definite
matrix, and `(r_ab - r_ac*r_bc)/sqrt((1-r_ac^2)(1-r_bc^2))` can then return a value outside
[-1, 1]: an impossible correlation with nothing marking it as impossible.

**Options considered.**
- (a) Let `confounders.py` rebuild the vectors itself from `QueryEngine` / `SeriesEngine`.
- (b) Change `pearson_correlate` to delete listwise.
- (c) Publish the vectors `correlate.py` already builds as `PairedSample`, and apply listwise
  deletion at the call site that needs it.

**Choice made.** (c). `paired_sample()` is a public seam; `correlate_kpis` now delegates to it, and
`PairedSample.listwise()` is opt-in.

**Reason.** (a) is a second copy of the gap-aware differencing loop, free to drift from the one
`correlate_kpis` uses -- the failure `series.period_ordinal` and `trend.theil_sen` are shared
functions to prevent. (b) would change every existing `Correlation` payload, since pairwise masking
is what they were all measured with. (c) changed no number: `test_correlation.py`'s existing tests
and the new `TestPairedSampleIsTheSeamNotAChange` pin that the published vectors reproduce the
reported `r` exactly, and that widening a sample with more keys leaves the existing vectors
untouched.

**Consequence/trade-off.** The aligned `n` can legitimately differ from a standalone
`correlate_kpis` call on the same pair, so `n` and `dropped_for_alignment` are both published rather
than leaving the model to discover the gap. One asymmetry is deliberate: each first-order partial
aligns over only the three variables *it* reads, so an unrelated candidate being missing somewhere
cannot shrink every row to the worst-covered candidate's coverage -- the same reason
`correlate_kpi_matrix`'s time-series branch samples per pair rather than once across all keys.

## 50. "Too few observations" is a different status from "collinear controls"

**Context/problem.** `joint_partial` inverts a correlation matrix and refuses an ill-conditioned
one. Two structurally different situations reach that refusal: genuinely redundant candidates, and
a matrix with more variables than the sample can support. `numpy.linalg.cond` cannot tell them
apart. Measured on the retail fixture, ranking four candidates cross-sectionally over `region`'s
four members hits the second case on *every* row.

**Options considered.**
- (a) Report both as `collinear_controls`.
- (b) Check `df = n - 2 - k >= 1` first and report `insufficient_n_for_controls` separately.

**Choice made.** (b), via `_supports_controls`, applied to the first-order rows, the joint row, and
`rank_competing_explanations`.

**Reason.** They call for opposite responses -- drop a *candidate* for collinearity, change the
*axis* for too few observations -- so collapsing them sends a model hunting for a redundancy that is
not there. This is the same defect `agent/trend.py` calls out in `detect_onset` returning a bare
`None` for three different reasons, and `agent/correlate.py` in `analysis.correlate` collapsing
`n < 3` and two constant-series cases into one sentence. `TestTooFewObservationsIsNotCollinearity`
pins both halves: the four-member axis reports `insufficient_n_for_controls`, and the same
candidates on the thirteen-pair quarterly axis all report `ok`.

**Consequence/trade-off.** None beyond one more status in `CONTROL_STATUSES`. It also makes the
`check_sample_adequacy` recommendation actionable: the tool that says "the period axis is the
adequate one" and the tool that says "this axis cannot support four controls" now agree in
vocabulary.

## 51. `rank_competing_explanations` publishes an evidence table and no score

**Context/problem.** Decision 45 records that the new agent tools "deliberately produce none -- that
judgment is the model's job now", of the confidence scalar `contest.score_hypothesis` produces. This
is the tool whose name most invites breaking that rule.

**Options considered.**
- (a) Fold correlation, precedence, consistency and partial correlation into one 0-1 support number.
- (b) Publish one row per candidate carrying each measure separately, ordered by a single declared
  key named in the payload as `ordered_by`.

**Choice made.** (b). Sort keys are `abs_partial_r` (default), `abs_r`, `p_value`; missing values
sort last and `cause_kpi` breaks every tie.

**Reason.** (a) is `score_hypothesis`'s weighted ledger rebuilt under a new name -- the thing this
rewrite exists to delete -- and it buries exactly the disagreements decisions 31 and 43 went out of
their way to surface (a pooled `r` of -0.953 against a member agreement rate of 1.0 on the same
pair). An ordering the caller cannot reproduce from the published fields *is* a hidden weighting,
so `test_the_ordering_is_reproducible_from_the_published_key` recomputes it from the payload alone,
and `TestNoScoreIsPublished` asserts structurally that no `score` / `confidence` / `support` /
`rank` / `weight` key exists anywhere in the result.

**Consequence/trade-off.** The model must read a table rather than a number, and a sub-check that
cannot run degrades that *cell* to a typed status rather than dropping the row or failing the call.
Mutual control is what earns the tool its place: each candidate's partial holds every rival
constant, which is the question "is it pricing, cost, or mix" is actually asking and which no number
of separate `test_confounders` calls answers.

## 52. `test_spurious_correlation` is time-series only and names its three transforms

**Context/problem.** Detrending needs a time axis. A cross-sectional sample is one point per member
with no ordering, so "does it survive detrending" has no meaning there. Separately, this codebase
now carries three different differencing conventions: `correlate.py`'s `transform="change"`
(`pct_change`), `temporal.py`'s `transform="difference"` (`.diff()`, decision 38), and this tool's
detrended residuals.

**Options considered.**
- (a) Accept `mode="cross_sectional"` and silently correlate the raw member values.
- (b) Refuse it with a typed `InvalidArgumentError` naming `time_series` as the valid alternative.

**Choice made.** (b), and all three transforms are published side by side with explicit
`transform` values `"level"`, `"detrended"`, `"difference"`.

**Reason.** (a) returns a number that reads like an answer and means nothing. On the transforms:
decision 38 already established that the choice of differencing materially changes the measured
association (`r=-0.583` with `.diff()` against `-0.315` with `pct_change` on the same pair), so a
tool whose entire finding is *the gap between transforms* must name which is which. Measured on two
independent random walks sharing an injected trend: level `r = 1.0` (p ~ 0), detrended `r = -0.069`
(p = 0.50), verdict `trend_driven`. On the retail fixture's planted supply disruption
(`revenue ~ stockout_events`, weekly): level `r = -0.580`, detrended `r = -0.916`, verdict
`survives_detrending`. Detrending reuses `trend.theil_sen` rather than a fourth hand-rolled fit.

**Consequence/trade-off.** A caller wanting a cross-sectional robustness check must reach for
`test_confounders` instead, which is the right tool for that axis. The verdict is derived from
whether the *detrended* correlation retains significance, so a pair with too little history returns
`"insufficient"` rather than a guess -- the G8 discipline, applied to a verdict rather than a number.

## 53. `LLMClient._call_with_tools` is a sibling to `_call`, not a replacement, and moves to Opus 5

**Context/problem.** Batch A1 needed the LLM client to run a multi-turn tool-use loop for the first
time. `_call` (`client.py:95`) is a single non-streaming `messages.create` with a hardcoded
one-element message list and no `tools=`; its response reader
(`"".join(getattr(b, "text", "") for b in resp.content)`) silently drops a `tool_use` block and then
`_extract_json` raises `ValueError("model did not return JSON")`. Separately, `anthropic==0.42.0`
(requirements.txt) and `anthropic_model="claude-sonnet-4-5"` (config.py) predate adaptive thinking,
`output_config.effort`, and the current tool-use surface.

**Options considered.**
- (a) Extend `_call`'s signature to accept an optional `tools=`/`messages=` and branch internally.
- (b) A new sibling method, `_call_with_tools`, leaving `_call` and its six role methods untouched.

**Choice made.** (b). Also: bumped `anthropic` to `1.2.0` (audited every call site against the 0.x ->
1.x breaking-change list first -- no `.with_raw_response`, no `completions.create`, no
`temperature`/`top_p`/`top_k`, no `httpx` object handed to the SDK; the only change needed was the
pin itself, since the SDK's HTTP layer moving to `httpx2` is transitive and Python is already 3.12,
above the new 3.10 floor); moved `anthropic_model` to `claude-opus-5`; added `llm_tool_max_tokens`
(16000), `llm_max_turns` (12) and `llm_effort` ("high") to `Settings`, leaving `llm_max_tokens`
(4000) as the ceiling for the six existing single-turn roles; added the `claude-opus-5` entry to
`telemetry.MODEL_PRICING` (a model absent from that table silently prices at `$0`, and
`test_llm_client_caches_identical_prompt_without_new_usage` already asserts a positive cost against
`client.model`).

**Reason.** (a) means every one of the six existing role methods and both fakes in
`test_telemetry.py` would need to reason about a widened `_call` signature for a code path none of
them use; (b) keeps them provably unchanged (that test file is untouched and still green) while the
new loop gets its own return shape. `_call_with_tools` returns a frozen `ToolLoopResult` --
`{status, text, content, trace, turns, stop_reason}` -- where `status` is one of `ok` /
`max_turns_exhausted` / `truncated` / `refused`: non-convergence is a typed result, never an
exception, matching the `agent.errors` philosophy the tool boundary already uses. `resp.content` is
appended to history *whole*, never rebuilt, so `thinking` blocks (Opus 5 runs adaptive thinking by
default) and `tool_use` block `id`s survive intact across turns; every `tool_result` for one
assistant turn is returned in a single user message, never split across several.

**Consequence/trade-off.** The cache key for the tool loop (`_tool_cache_key`) hashes the *entire*
message history plus `tools`/`tool_choice`/`max_tokens` under a separate `TOOL_CACHE_VERSION`
constant, rather than reusing `_cache_key`'s `(system, user)` shape or bumping the existing
`CACHE_VERSION` -- a v1-style key would collide turn 3 of one question with turn 3 of another
sharing the same system prompt. A refusal (`stop_reason == "refusal"`) is deliberately never
cached, since it is a safety-classifier decision, not a stable answer to memoize. There is still no
LLM test double anywhere else in this repo (every other LLM path is tested by disabling the LLM
entirely); this batch introduces the first one, `tests/fakes.py`'s `ScriptedAnthropic`, assigned
directly to `client._client` the same way `test_telemetry.py:74` already fakes the single-turn path.

## 54. The tool-use loop is hand-written, not the SDK's beta Tool Runner

**Context/problem.** The `anthropic` Python SDK ships a beta `client.beta.messages.tool_runner(...)`
that drives the agentic loop automatically given `@beta_tool`-decorated functions, so A1 did not have
to write a loop at all if it deferred to that helper.

**Options considered.**
- (a) `client.beta.messages.tool_runner(tools=[...])`, wrapping each registry tool as a `@beta_tool`.
- (b) A hand-written `_call_with_tools` loop, as recorded in decision 53.

**Choice made.** (b), confirmed with the user ahead of implementation.

**Reason.** The runner owns the entire loop and takes `tools=` once at construction, which forecloses
three things this package's design already commits to: A5's evidence-trace accumulation (the runner
does not expose a per-turn hook shaped like `trace.append(...)` without reaching into its internals),
a turn budget the caller can inspect mid-loop (`max_iterations` bounds it but does not surface *why*
it stopped as a typed status the way `ToolLoopResult.status` does), and progressive tool-group
disclosure (loading only the Observe orient/retrieve tools until the model commits to an explanatory
question, per the master plan's tool-count mitigation) -- the runner's tool list is fixed for the
call, not something a per-turn hook can narrow. A hand-written loop also takes no beta dependency.

**Consequence/trade-off.** `_call_with_tools` re-implements what the runner would give for free:
turn iteration, `stop_reason` branching, and re-sending tool results. This is bounded, contained code
(the method plus its two private helpers) reviewed and tested in this same batch
(`test_tool_transport.py`), not a growing maintenance surface -- and it is exactly the surface A5
needs to extend for progressive disclosure, which the runner does not offer a seam for.

## 55. Tool JSON Schemas are auto-generated from resolved type hints, with a small name-keyed override table

**Context/problem.** A2 needed an Anthropic `input_schema` for every one of 56 tools drawn from
twenty engine classes' public methods. Every engine module has `from __future__ import annotations`
(PEP 563), so `inspect.signature(fn).parameter.annotation` hands back the *string* source of each
annotation (`'Optional[Sequence[str]]'`) rather than a resolved type -- `typing.get_type_hints(fn)`
is required to get an object `typing.get_origin`/`get_args` can actually introspect.

**Options considered.**
- (a) Hand-write all 56 schemas.
- (b) Auto-generate from `typing.get_type_hints`, with a small table for the cases a bare annotation
  cannot express, keyed by parameter name (and, for the handful of genuine collisions, by
  `(tool_name, param_name)`).

**Choice made.** (b).

**Reason.** ~85% of parameters across the twenty engines are a plain `str`/`int`/`float`/`bool` or a
`Sequence[...]` of one, fully described by the resolved annotation alone. What the annotation cannot
express, confirmed by reading each module's own validation rather than assumed from the name:
`TimeFilter`'s eight-shape union is expressed once as a **flat** object schema (`TIME_FILTER_SCHEMA`)
rather than an 8-branch discriminated `anyOf` -- `timefilter.parse` already raises a typed,
recoverable `MalformedTimeFilterError` naming the valid shapes on a bad combination, which is cheaper
in tokens across ~40 tool schemas than carrying the full union in every one; KPI-key and dimension
parameters get their `enum` injected per request from `api.available_keys(df)` /
`api.list_dimensions(df)`, so schemas are built per request, not at import; module constants
(`MODES`, `SERIES_GRAINS`, `_ORDERS`, `ORDER_BY_CHOICES`, `CAUSE_DIRECTIONS`, `DIRECTIONS`,
`COMPETING_SORT_KEYS`) supply enums a bare `str` annotation cannot. Two genuine collisions surfaced
under this design and needed `(tool_name, param_name)`-keyed overrides rather than the shared
per-name table: `order` means `("asc","desc")` in `query.QueryEngine.query` but `("asc","desc",
"best","worst")` in `breakdown.rank_entities`; `candidates` means a list of KPI keys everywhere
except `quality.filter_material`, the one engine method with no `df` at all, where it means a list of
already-computed candidate dicts from a prior tool call. A third, `compare.significance(series=...)`,
is an internal already-computed-`Series` passthrough with no JSON shape at all and was added to the
existing omitted-knob set (`max_kpis`, `max_members`, `limit_cells`, `min_points`,
`baseline_periods`, `k_sigma`, `weights`) rather than schematised.

**Consequence/trade-off.** `test_the_schema_cannot_drift_from_its_signature`
(`test_registry.py`) is the guard this design needs to keep working: every schema property must
exist on the bound callable's own parameters, and every parameter without a default must be
required, so a later signature change with no matching registry update fails a test rather than
silently mismatching. A model that sends a wrong-shaped `TimeFilter` or a `filter_material`
`candidates` list without the expected keys is caught by the engine's own validation
(`timefilter.parse`, `AgentToolError`), one recoverable turn later, rather than by JSON Schema
`strict` validation up front -- consistent with this package's existing preference for a typed,
recoverable error over front-loaded schema strictness (decision 52's `test_spurious_correlation`
refusal is the same shape of choice, one layer down).

## 56. `get_kpi_definition` excludes the KPI Contract's human-authored `business_definition`/`relevance`

**Context/problem.** `ContractAPI.business_definition(key)` and `.relevance(key)` return real
contract-authored prose describing what a KPI means in this business's terms -- not a template a
tool rephrases, but content a human wrote once, in KPI Studio. `introspect.get_kpi_definition` (A2)
initially included both, alongside `.description(key)`, since a "what does this KPI mean" tool seems
like the natural place for them. `test_g1_no_tool_result_leaks_a_sentence` failed immediately: at 207
characters, `description` reads exactly like the prose the whole rewrite exists to keep out from
behind a tool boundary.

**Options considered.**
- (a) Add `description`/`business_definition`/`relevance` to `PROSE_ALLOWLIST` (`base.py`) alongside
  `label`/`unit`/etc., since this prose is contract data rather than a manufactured narrative.
- (b) Trim `get_kpi_definition`'s payload to structured facts only -- unit, direction, aggregation
  kind, tags, source fields -- leaving the prose fields out of the tool boundary entirely.

**Choice made.** (b), matching the master plan's own Part 2 field list for this exact tool
("formula, unit, aggregation, tags, source_fields, depends_on" -- no business-definition prose).

**Reason.** `PROSE_ALLOWLIST` is shared, mechanical, and load-bearing for every other agent-layer test
in the suite; widening it to admit two more sentence-length fields weakens the guard for every tool
that will ever be checked against it, to accommodate one tool's convenience. The one rule this whole
rewrite is built on ("every tool returns numbers and facts only ... if there is no template left to
rephrase, the answer cannot be a rephrased template") reads as being about the tool boundary, not
about where the prose originated -- human-authored contract text relayed verbatim by the model is
still a sentence crossing that boundary. `kpi/explanation.py`'s `compose_explanation` remains the
sanctioned place this prose is authored and consumed (KPI Studio, a different feature, explicitly
kept out of scope by Part 4 of the master plan); nothing here touches it.

**Consequence/trade-off.** A leader-facing "what does gross margin mean" answer must come from the
model's own words over the structured facts this tool now returns, rather than being lifted from
contract prose -- consistent with "the model writes all prose; there are no templates left" for
everywhere else in this rewrite, so this is not a new asymmetry, just this tool catching up to it.

## 57. The validation airlock validates against the exact schema already sent to the model

**Context/problem.** A3 needed to close the gap between "the engines validate semantics, first,
before any pandas work" (`query.py:118-141` is the canonical example) and "nothing checked an
argument's *type and shape* before that." Rather than guess the scope, the live registry was
fuzzed first. Three genuine gaps, not hypothetical ones: `search_kpis(text=True)` executed and
returned a plausible empty answer (`{"matches": []}`, `is_error=False`) rather than being rejected;
`query_kpi(filters=["region"])` reached pandas and leaked "dictionary update sequence element #0
has length 6; 2 is required"; `test_confounders(candidates="not_a_list")` had its string iterated
character by character and reported `unknown_kpi: 'n'`, actively misdirecting the model about what
was actually wrong. Everything else already degraded correctly and nothing raised.

**Options considered.**
- (a) A hand-rolled type checker walking each schema dict by hand.
- (b) Pydantic models per tool, alongside the JSON Schema `registry.py` already generates.
- (c) `jsonschema`, validating directly against the exact schema already sent to the model.

**Choice made.** (c), confirmed with the user ahead of implementation. `agent/airlock.py`'s
`validate(tool_name, schema, args)` runs before `registry.dispatch` binds any context or calls the
engine, converting a `jsonschema.ValidationError` into `InvalidArgumentError` -- the class
`errors.py` already reserved for exactly this ("reused by the validation airlock in front of the
full tool registry").

**Reason.** (a) reimplements a solved problem by hand, and a bug in it is a silently-accepted bad
argument -- precisely what the airlock exists to prevent. (b) creates a second, independently
maintained schema per tool that can drift from the one the model was actually shown; (c) has one
schema, so a schema a model was shown and a schema it is checked against cannot drift apart. Fuzzing
first, rather than deriving scope from first principles, is what surfaced a second and unrelated
finding: `kpi_keys`'s generated schema was array-only even though `QueryEngine.query`'s own
annotation is `Union[str, Sequence[str]]` and the engine has always accepted a bare string -- the
airlock rejecting `"revenue"` as `kpi_keys` on its first real exercise is what caught an A2 schema
bug that had been silently working by accident (nothing validated the array-only claim before).
Fixed by special-casing `kpi_keys` to `anyOf: [string-enum, array-of-string-enum]`;
`candidates`/`candidate_causes` stay array-only, since those really are `Sequence[str]`-only
everywhere they appear.

**Consequence/trade-off.** Three existing `test_registry.py` assertions changed meaning, not
behaviour: a hallucinated KPI key, an unknown dimension, and a bogus `TimeFilter` `type` are now
caught by the airlock (`invalid_argument`) *before* the engine's own `UnknownKpiError` /
`UnknownDimensionError` / `MalformedTimeFilterError` can fire, because all three are schema `enum`s
built from `api.available_keys(df)` / `api.list_dimensions(df)` / the eight known `TimeFilter`
shapes. Equally recoverable either way -- `valid_alternatives` still carries the real choices,
extracted from the failing subschema's own `enum` (including through an `anyOf`, for `kpi_keys`) --
and reachable *earlier*, which is a strengthening of "never execute" rather than a regression. See
decision 58 for the one case (a legal `type` with a missing required key) where the engine's own
check remains the sole authority and stays reachable through `dispatch`.

## 58. `TimeFilter`'s schema stays permissive; a legal-type-but-incomplete shape still reaches `timefilter.parse`

**Context/problem.** Decision 55 chose a flat `TimeFilter` schema over an 8-branch discriminated
`anyOf` specifically so `timefilter.parse`'s own `MalformedTimeFilterError` (naming valid shapes)
stayed the recoverable-turn mechanism for "which keys does this `type` need" -- restating that union
in the schema was the cost that decision declined to pay. Decision 57 established that the airlock
now catches a bogus `type` *value* earlier, via the schema's own `enum`. The question this left open:
does the airlock's own type/shape checking end up re-deciding what a legal `type` requires too,
quietly reintroducing the union decision 55 avoided?

**Options considered.**
- (a) Let `jsonschema` validate the full nested `TimeFilter` shape (all eight `type`-conditional key
  requirements), matching each `type` to its own `required`/`properties` via a discriminated
  sub-schema.
- (b) Keep `TIME_FILTER_SCHEMA` exactly as decision 55 left it -- `type` as an enum, its sibling keys
  (`year`, `quarter`, `values`, `start`, `end`, `grain`, `n`) each typed but none of them required
  except `type` itself -- so a legal `type` with a missing key for that shape passes the airlock and
  reaches `timefilter.parse`, which has always been the one place that particular rule lived.

**Choice made.** (b). No schema change was needed at all -- `TIME_FILTER_SCHEMA` already had this
shape from A2; A3 only had to confirm the airlock's own JSON-Schema-only view of it does not
accidentally supersede `timefilter.parse` for the one thing the flat schema deliberately left out.
Pinned by `test_a_quarter_time_filter_missing_its_quarter_key_still_reaches_malformed_time_filter`
(`test_registry.py`): `{"type": "quarter", "year": 2024}` (a legal type, missing the `quarter` key
that specific type requires) still returns `malformed_time_filter` with `valid_types`, not
`invalid_argument`.

**Reason.** The division of labour this produces is clean rather than incidental: the airlock owns
"is this a legal `type` at all, and are the fields present of the right primitive type" (bool, list,
wrong nesting -- the argument-shape class of error A3 exists for); `timefilter.parse` owns "given a
legal `type`, is this specific combination of keys coherent for it" -- a domain rule the flat schema
was never meant to encode. Restating the second half in JSON Schema would be exactly the token cost
decision 55 traded away.

**Consequence/trade-off.** Two failure modes now report through two different error codes for what
might look, superficially, like "the same kind of mistake" -- a bogus `type` value is
`invalid_argument`; a legal `type` with a missing key is `malformed_time_filter`. Both carry a
`valid_alternatives`/`valid_types` list the model can act on identically, so this is a distinction
without a difference to the model reading the tool result; it matters only to a human or a test
reading the error `code` field, which is exactly where it belongs.

## 59. `AgentContext.build` shares the dataframe by reference and reports a turn budget rather than enforcing one

**Context/problem.** A4 needed a per-request handle wrapping `dataset_service.load` +
`ContractAPI` + `registry.build`, one call rather than three, for the loop to thread everywhere.
Two things needed a decision: whether the handle should own its own copy of the dataframe, and
whether a "turn budget" concept on the context does anything `LLMClient._call_with_tools`'s own
`max_turns` enforcement does not already do.

**Options considered.**
- (a) `AgentContext.build` copies `df` (`.copy()`) so nothing outside the context can affect it.
- (b) Share `dataset_service.load`'s cached frame by reference, as every other caller of `load`
  already does, and document `df` as read-only on the dataclass rather than defend against mutation.
- For the budget: (i) `Budget` re-enforces `max_turns` as a second gate the loop must check before
  calling `_call_with_tools`, or (ii) `Budget.record(turns)` is populated *after* a
  `_call_with_tools` call returns, purely for reporting `remaining_turns`/`exhausted`.

**Choice made.** (b) and (ii).

**Reason.** `dataset_service.load` already returns the cached frame itself
(`dataset_service.py:397`), not a copy, and every existing caller (the whole pipeline, every engine)
already depends on that for cost -- copying only inside `AgentContext` would make this one path
inconsistent with the rest of the codebase for no real safety gain, since every registered tool
already treats `df` as read-only by convention (confirmed across all twenty engines: `df` is read,
never assigned into). Routes in this codebase are sync `def`, so FastAPI runs them in a threadpool;
this makes the read-only contract a real concurrency surface, not a hypothetical one, so it is
stated explicitly in the dataclass's own docstring rather than left implicit. On the budget: `(i)`
would be a second enforcement path duplicating what `_call_with_tools`'s own loop already does
correctly and would need to stay in lockstep with it; `(ii)` gives A5's loop (and, later, A7) a
place to answer "how much of this conversation's turn allowance is left" without re-deriving it from
a `ToolLoopResult` the caller may not still be holding, at zero risk of the two paths disagreeing
about what "exhausted" means.

**Consequence/trade-off.** A future in-place mutation of `ctx.df` by any tool would corrupt every
other request sharing that `(path, mtime)` cache entry -- exactly the existing hazard
`dataset_service.py` already carries for every other caller, not a new one this batch introduced.
`Budget` currently has nothing upstream of it consuming `remaining_turns` before the fact (this
batch's `loop.answer` makes exactly one `_call_with_tools` call per question), so its enforcement
value is entirely forward-looking -- for a future multi-shot conversation feature reusing one
`AgentContext` across several questions, this batch's headless tests (`test_agent_context.py`)
pin the reporting contract now rather than leave it to be designed under that future pressure.

## 60. The agent loop returns a typed `llm_required` status and exposes all 56 tools every turn

**Context/problem.** A5 needed to decide two things with no existing precedent in this codebase to
copy: what happens when there is no usable LLM (the deterministic pipeline has its own fallback that
completes without one; the tool-use loop, by design, has none), and whether to expose the registry's
full 56-tool surface every turn or narrow it via `registry.by_group(...)` (already built, unused).

**Options considered, no key.**
- (a) Let `LLMClient._call_with_tools` raise `RuntimeError("LLM disabled")` as it already does, and
  push the decision to A7's endpoint.
- (b) Route to `pipeline.run_question` when there is no key, preserving the old no-key guarantee.
- (c) Return a typed `AgentAnswer(status="llm_required", ...)` before touching the dataset at all.

**Options considered, tool exposure.**
- (i) All 56 tools, every turn.
- (ii) Progressive disclosure via `by_group`: orient/retrieve first, widen once the model commits to
  an explanatory question.

**Choice made.** (c) and (i), both confirmed with the user ahead of implementation.

**Reason.** (a) would surface at `routes_analysis._guard` (`ValueError`-derived, since
`RuntimeError` is not even a `ValueError`) as an unhandled 500, not the 422 with structure A7 needs;
(b) would couple the new loop to `pipeline.py`, which A9 is going to delete, and would silently
reintroduce the templated prose this whole rewrite exists to remove the moment a key is absent.
`AgentAnswer(status="llm_required")` matches the master plan's own premise ("no deterministic
fallback is required") and, checked as the very first line of `answer()`, means the no-key path
calls no tool at all -- not even the two seeding calls (`describe_dataset`/`list_kpis`) -- pinned by
`test_a_disabled_client_returns_llm_required_without_calling_any_tool`. On tool exposure: the tool
array already carries the loop's one cache breakpoint (decision 57's registry work), so it is the
largest stable prefix of every request in a run; narrowing or widening it turn-to-turn would forfeit
that prefix for the rest of the run. `by_group` remains built, deliberately unused -- the master
plan's own words are "revisit if eval shows tool-selection errors," and A8 is what produces that
evidence, not a guess made here.

**Consequence/trade-off.** `kpis_used` and `periods_used` on `AgentAnswer` are derived by walking
`ToolLoopResult.trace`'s own recorded tool arguments, never by reading the model's final prose --
pinned by `test_kpis_used_reflects_actual_tool_arguments_not_the_final_answer_text`, which scripts a
final answer that *names* a KPI the model never actually queried and asserts it does not leak into
`kpis_used`. A future eval (A8) may find the 56-tool surface causes measurable tool-selection error
at some question complexity; that is the evidence this decision explicitly defers to, not a risk
accepted silently.

## 61. `/api/questions/ask` returns HTTP 200 for every ending of the loop, and declares a response model

**Context/problem.** A7 had to map a five-value `AgentAnswer.status` union onto HTTP, and decide
whether to follow `routes_analysis.py`'s convention (every route `-> Dict[str, Any]`, no
`response_model`) or `routes_kpi.py`'s (real response models, justified at `routes_kpi.py:9-12`).

**Options considered, status mapping.**
- (a) 200 for `ok`; a 4xx/5xx for `refused`, `truncated`, `max_turns_exhausted`, and 503 for
  `llm_required`.
- (b) 200 with a typed `status` for all five, the way `/questions/investigate` already returns
  `needs_clarification` at 200.

**Options considered, response model.** (i) `Dict[str, Any]`, matching the four stage endpoints in
the same router. (ii) A declared `AgentAnswerResponse`.

**Choice made.** (b) and (ii), both confirmed with the user ahead of implementation.

**Reason.** On status: `API_CONTRACT.md:244` already establishes the house reading of these — "a
normal branch, not an error". A model that declined to answer, or ran out of turns, has produced a
*result the client renders*, complete with the partial evidence trail; an HTTP error would discard
that trail and invite a retry that changes nothing. On the response model: the two in-repo
precedents split on whether a shape is discovered or fixed, and this one is fixed —
`AgentAnswer` is a frozen dataclass whose own docstring says "A7 serialises this rather than
reshaping it", and `status` is a closed set every client must branch on. Publishing that as a
`Literal` enum in `/openapi.json` is checkable in a way a markdown table is not, and
`test_the_openapi_document_publishes_all_five_answer_statuses` checks it.

**Consequence/trade-off.** FastAPI's `response_model` *filters*: a key added to the route dict and
not to `AgentAnswerResponse` disappears from the response with no error at all. That is a real
regression vector, and the only defence is
`test_the_response_carries_exactly_the_documented_top_level_keys`, which asserts the exact
top-level key set on a live response rather than trusting the model to stay in step. It also means
`routes_analysis.py` now carries two conventions; the module docstring says which and why, so the
inconsistency reads as deliberate rather than as drift.

## 62. `LLMTransportError` is raised from `_call_with_tools`, so a provider outage is a 503

**Context/problem.** `_call_with_tools` recorded telemetry and then re-raised the raw Anthropic SDK
exception (`client.py:248-253`). `routes_analysis._guard` catches only `(ValueError,
DatasetError)`, and an SDK error is neither -- so a rate limit or timeout would have surfaced as an
unhandled 500. Decision 60 anticipated exactly this ("would surface at `routes_analysis._guard` as
an unhandled 500, not the 422 with structure A7 needs") and left it to A7.

**Options considered.**
- (a) `except Exception` around `loop.answer` in the route.
- (b) `loop.answer` catches it and returns a new `AgentAnswer(status="llm_error")` at 200.
- (c) A `RuntimeError` subclass raised from `_call_with_tools`, caught in a new `_llm_guard`.

**Choice made.** (c), confirmed with the user.

**Reason.** (a) would launder genuine defects -- a `TypeError` in loop code, a `KeyError` in the
registry -- into "the reasoning provider is unavailable", which is both the wrong status and a way
to hide bugs permanently; the route deliberately lets anything unrecognised become a 500, because a
bug should look like a bug. (b) would make a provider outage indistinguishable from a successful
answer to monitoring: `TelemetryRepository` and `/api/telemetry/summary` count status, and an
outage recorded as a success is a metric that lies. It would also re-open A5's status enum, which
decision 61 commits to serialising rather than extending. (c) keeps `llm/client.py` the only module
that names the SDK's exception types, matching how it is already the only module that *imports* the
SDK, and being a `RuntimeError` rather than a `ValueError` means `_guard`'s 422 branch cannot
swallow it by accident.

**Consequence/trade-off.** This modifies A1's artefact, which A7 strictly speaking does not own.
The blast radius is deliberately minimal: `_call` is untouched, so the four single-turn roles the
deterministic pipeline depends on keep their exact behaviour, and nothing in the suite pinned the
old bare re-raise. The original exception is chained (`from exc`) so its message is still available
to the log; it is simply not returned to the caller, since an SDK message naming an internal
endpoint is not a user-facing string.

## 63. `kpis_used` counts only tool calls that ran, and the eval harness re-derives it anyway

**Context/problem.** `loop._derive_kpis_used` walked every trace entry regardless of `is_error`, so
a KPI key the airlock had *rejected* still appeared in `AgentAnswer.kpis_used` -- and A7 is the task
that would have serialised it straight to the UI's "KPIs used" chip. `test_agent_loop.py:87` already
scripted such a trace without asserting on the field. A8's `Observed` needed the same derivation.

**Options considered.**
- (a) Leave `loop.py` alone; have the eval harness compute its own error-filtered `kpis_used_ok`.
- (b) Fix `loop.py` and have the harness import the fixed function -- one source of truth.
- (c) Fix `loop.py` **and** keep the harness's independent derivation.

**Choice made.** (c), confirmed with the user.

**Reason.** (a) alone would ship a known-wrong field: the derivation exists specifically to make
`kpis_used` a fact about what ran rather than a claim the prose makes about itself, and a rejected
call ran nothing. The fix is one predicate (`_succeeded`) applied to both `_derive_kpis_used` and
`_derive_periods_used`, since a period from a refused call is wrong for the identical reason. The
harness keeping its own derivation looks like duplication and is not: an eval that imported the
function under test would *inherit* a future regression instead of detecting it. Keeping both makes
the disagreement visible, which is why `score` publishes a `loop_derivation_agrees` check comparing
the two -- a green tier eval now also certifies that this fix is still in place.

**Consequence/trade-off.** `_KPI_ARG_NAMES` is imported by the harness, so the two derivations share
the argument-name table even though they do not share the walk. That is the right seam: the names
are data about the registry's 56 signatures and would be actively misleading to duplicate, whereas
the filtering rule is the logic under test. One more consequence worth naming: an answer to a
question the dataset cannot support now correctly reports `kpis_used: []`, which is what
`t1_out_of_extent` asserts -- an answer resting on no data should say so.

## 64. A9 retirement: three modules the plan named for deletion were kept, on measured test coupling

**Context/problem.** A9 is the designated deletion batch: retire the 4-stage pipeline
(`engines/{act,contest,hypotheses,investigate,llm_hypotheses,signals}.py`, `services/pipeline.py`),
its six endpoints, its prompts, its redaction paths, and collapse `InvestigationPage.jsx` to one
`answer` + evidence panel. Batches 1-16 (Layers C/O/I/X/A1-A8) were already complete and untouched
by this one. The plan going in named `personas/reframe.py`, `query/understanding.py` +
`query/grounding.py` + `query/periods.py`, and most of `engines/analysis.py` as candidates for
deletion alongside the pipeline proper.

**Options considered, per module.**
- `personas/`: delete the whole package (the plan's initial read) vs. delete only `reframe.py` and
  keep `profiles.py`/`__init__.py`.
- `query/understanding.py` + `query/grounding.py` (+ `periods.py`, which `understanding.py` needs):
  delete along with `/questions/investigate` vs. keep them and keep `/questions/interpret` as a
  surviving route.
- `engines/analysis.py`: prune `correlate`/`lead_lag`/`counterexamples`/`member_change_table`/
  `coverage_report` down to only what `validate_rca.py` and `agent/` still call, vs. leave the whole
  module untouched.

**Choice made, per module.**
- `personas/`: kept `profiles.py` + `__init__.py`; deleted only `reframe.py`.
- `query/`: kept `understanding.py`, `grounding.py`, `periods.py` in full; kept `/questions/interpret`.
- `engines/analysis.py`: left entirely untouched.

**Reason.** In each case the plan's assumption -- that the module existed only to feed the retired
pipeline -- did not survive contact with the actual test suite. `routes_auth.py` imports
`persona_for_role`/`persona_options` from `personas/__init__.py` for the Settings page persona
switcher (decision 26), a live, unrelated feature; only `reframe.py`'s reframing engine was
pipeline-only. `query/grounding.py` has two independent, substantial test files
(`test_dimension_catalogue.py`, `test_kpi_search.py`) exercising `ground_question`/`_normalise`/
`_split_contrast` that have nothing to do with `investigate`/`contest`; `understanding.py` needs
`periods.py`'s `resolve_period` to keep working, so keeping one meant keeping the other two.
`engines/analysis.py`'s `correlate`, `member_change_table` and `counterexamples` are imported
directly by `test_resolver_propagation.py` (the C2 resolver-propagation regression test) and
`test_consistency.py` (X2's consistency tooling) -- both protecting live correctness invariants,
not the narrative pipeline. Deleting any of the three would have cost working, independently-valued
test coverage for zero benefit: none of the three is prose, none of the three violates "every tool
returns numbers and facts only", and nothing downstream needed them gone.

**Consequence/trade-off.** `/api/questions/interpret` survives as a seventh live endpoint alongside
`/dashboard`, `/meta/timeframes`, `/questions/ask` and the three `/investigations` routes -- one
more endpoint than the plan's minimal end state, in exchange for zero test breakage on functionality
the agent loop does not supersede (C5's `search_kpis` replaces the *blocking* behaviour
`understanding.py` used to have, not the grounding computation itself). `engines/analysis.py`
carries two genuinely-dead functions (`lead_lag`, `coverage_report`, zero callers anywhere) as an
accepted small residue rather than a risk to a still-load-bearing module.

## 65. `/questions/ask` gained `persist`, breaking A7's original "nothing is persisted" design

**Context/problem.** A7 (decision 61) deliberately shipped `/api/questions/ask` with no `persist`
field: an `AgentAnswer` has no kpi/verdict/hypothesis, the shape a saved investigation was built
around, so persistence was left for a later decision. A9's frontend collapse needs History to keep
working, and `InvestigationRepository` has no consumer left once `run_full`/`run_question` are gone.

**Options considered.** (a) persist agent answers via a schema migration on
`InvestigationRepository`; (b) drop History entirely; (c) keep `/investigations` serving only
pre-existing rows, read-only. Put to the user directly; (a) was chosen.

**Reason.** `InvestigationRepository.create` is already a generic `{**payload}` store with no
schema enforcement (`db/repositories.py:256`), so an `AgentAnswer` (`question`/`answer`/`evidence`/
`kpis_used`/`periods_used`/`engine`) persists exactly as it is -- no migration is actually needed,
only a new `persist: bool = False` field on `AgentQuestionRequest` and a conditional write in the
route. Defaulting to `false` (rather than A7's implicit "never") keeps the endpoint's existing
callers unaffected and makes saving an explicit, opt-in act, matching what a "History" feature
means to a user asking one-off questions.

**A real bug caught and fixed during implementation.** The first version built the response
`payload` dict, embedded it as `"result": payload` in the document passed to `create()`, and only
*afterwards* set `payload["investigation_id"]`. Because `"result": payload` is the same object
reference as `payload` itself, patching it after the write raced whatever `create()` had already
done with the reference (serialise it, hold it, or both) -- the persisted `result.investigation_id`
would come back `None`. Fixed by reserving the id upfront (`new_id("inv")`) before either the
response or the persisted document is built, so both are fully formed before either is used. Caught
by a new integration test (`test_a_persist_field_in_the_body_saves_the_answer`) rather than by
inspection.

**Consequence/trade-off.** `GET /api/investigations` and `GET /api/investigations/{id}` now serve
two shapes: rows saved by the retired pipeline (no `answer` key) render as
`{"status": "legacy_format", ...}`; rows saved by the agent loop render the full
`AgentAnswerResponse` body, unredacted, matching `/questions/ask`'s own `redaction: "none"`.
`HistoryPage.jsx` and `InvestigationPage.jsx` both branch on this explicitly rather than assuming
one shape.

## 66. `determine_focus` moves to `engines/observe.py`, not deleted with `investigate.py`

**Context/problem.** `scripts/validate_rca.py`'s temporal-ordering check (the one asserting a
decline's cause postdates the decline itself) scopes its onset comparison to "the focus the engine
itself selected" -- `investigate.determine_focus`, a pure 25-line function reading only
`observation["drivers"]`. Deleting `investigate.py` wholesale would silently cost this check, the
project's only end-to-end causal-correctness gate for temporal ordering.

**Choice made.** Move `determine_focus` (and its two thresholds, renamed
`MIN_FOCUS_CONTRIBUTION_PCT`/`MIN_FOCUS_OVER_INDEX` to avoid colliding with any future `observe.py`
constant) verbatim into `engines/observe.py`. `validate_rca.py`'s `run_pipeline()` now calls it
directly instead of running the retired `investigate()` stage.

**Reason.** `determine_focus` has no dependency on anything else in `investigate.py` -- it takes an
`observation` dict and returns a `Dict[str, str]`, nothing more. `observe.py` already produces the
`observation` this function reads, so the move puts the function next to its only real input rather
than leaving a one-function shim module alive for a single caller.

**Consequence/trade-off.** A static import-gate test (`test_legacy_retirement.py`) that walks the
whole `app/` and `scripts/` tree with `ast` caught a second, easy-to-miss call site during this
batch: `driver_graph._dimension_drivers` had its own lazy
`from .investigate import MIN_CONTRIBUTION_PCT, MIN_OVER_INDEX`, undetected by grepping call sites
of `determine_focus` alone since it imported the threshold constants directly rather than the
function. Repointed at the moved constants in `observe.py`. This is the argument for the
AST-walking guard test existing at all: a substring search would have found the docstring's
*mention* of `determine_focus` and stopped there.

## 67. `docs/API_CONTRACT.md` rewritten for the six retired endpoints; `RANKING.md`/`ARCHITECTURE.md`/`TASK_BOARD.md`/`DEMO_SCRIPT.md` left as-is

**Context/problem.** A9's plan named doc updates in scope. `API_CONTRACT.md`'s `## Analysis`
section (~390 lines) documented the four stage endpoints, `/investigations/run` and
`/questions/investigate` in full request/response detail; `RANKING.md`, `ARCHITECTURE.md` and
`DEMO_SCRIPT.md` reference `contest.py`/`score_hypothesis`/the four-stage flow narratively, at a
combined ~1,900 lines across four files.

**Choice made.** Rewrote `API_CONTRACT.md`'s endpoint table and the `## Analysis` section in full,
including the response shape for `GET /api/investigations` (both the live and `legacy_format`
cases) and `/questions/ask`'s new `persist`/`investigation_id`. Left `RANKING.md`, `ARCHITECTURE.md`,
`TASK_BOARD.md` and `DEMO_SCRIPT.md` untouched.

**Reason.** `API_CONTRACT.md` is the one doc a client integrator actually depends on being
accurate -- a stale endpoint table actively misleads in a way a stale architecture narrative does
not. The other four are long-form prose (a driver-ranking methodology writeup, a system-design
narrative, a historical phase-by-phase task log already marked complete, a demo walkthrough) where
a correct but shallow pass would misrepresent how much of their content actually changed, and a
genuinely thorough pass is a rewrite-sized task of its own, disproportionate to a batch whose
deliverable is the code retirement.

**Consequence/trade-off.** `RANKING.md:334,502,523` still link to `engines/contest.py`, which no
longer exists, and its causal-consistency narrative still describes `score_hypothesis`'s ledger.
`TASK_BOARD.md` was already confirmed stale before this batch (a pre-A9 Phase 0-7 record, not the
C/O/I/X/A board these decisions track). Flagged here rather than silently left inconsistent;
whoever next edits those docs for an unrelated reason should reconcile this in passing rather than
this batch attempting it speculatively.

## 68. X5 ships two of its four tools; the sensitivity/subsample pair is deferred pending A8's live report

**Context/problem.** Task X5 (the Contest robustness family: `test_sensitivity_to_outliers`,
`estimate_effect_size`, `test_sensitivity_to_period`, `test_subsample_stability`) is marked
provisional in the master plan itself: "the Part 2.5 trace exercise never called any of these across
all 15 questions ... before building, run the eval suite from A8 and cut whatever the agent does not
actually reach for." `scripts/validate_agent.coverage()` produces exactly that `never_fired` report,
but it needs a live `ANTHROPIC_API_KEY` and real spend, and the 17 scripted `CASES` were never
written to invite a robustness check, so their silence is weak evidence either way.

**Options considered.**
- (a) Build all four now, let a later eval cut what does not earn its place.
- (b) Run the live gate first, build only the survivors.
- (c) Build the two the plan marks unconditional (`test_sensitivity_to_outliers`,
  `estimate_effect_size`), defer the two the plan already flags as weakest pending the gate.

**Choice made.** (c). The live gate (option b) is the right process and remains the next step, but
it is not runnable from this environment; (a) would ship tools with a documented ~0% reach rate.

**Reason.** The plan's own analysis already separates the four: the outlier-drop test answers the
question a reviewer asks first ("is this just one big account") and `find_outlier_contributors`
(I5) hands it the candidate member for free; `estimate_effect_size` gives magnitude and a
distribution-free interval that `significance.fisher_interval` (X3) cannot, since Fisher's z assumes
a normality these member counts do not support. `test_subsample_stability` is close to meaningless
on the fixtures that exist -- split-half on retail `region`'s four members -- and `estimate_effect_size`'s
bootstrap CI already carries the instability signal it would report. `test_sensitivity_to_period`
is cheap (`ComparisonEngine.compare` is already generalised to arbitrary A/B) but needs the gate's
evidence that the model reaches for window-choice robustness at all. Deferring two is not the same
as cutting them; the gate decides.

**Consequence/trade-off.** The registry goes 56 -> 58 tools, both in the `contest` group.
Progressive disclosure stays off (decision 60): the tool array is still the loop's one cache
breakpoint and narrowing it per-turn forfeits that. `sensitivity.py` composes `CorrelationEngine`
and `find_outlier_contributors` and owns only the leave-one-out refits (`numpy.corrcoef` over a
subset of `PairedSample`'s published vectors) and the `scipy.stats.bootstrap` call. That bootstrap
is seeded at a fixed `BOOTSTRAP_SEED = 0` -- a deliberate choice of G5 reproducibility over per-call
entropy, since the interval is a property of the sample; `test_sensitivity.py::TestTheBootstrapIsSeeded`
is the regression that an unseeded call would fail. `test_tier_eval.py`'s coverage assertions were
already written against `len(TOOL_SPECS)` and stay green; only a stale "56" in a method name and two
docstrings was corrected.
