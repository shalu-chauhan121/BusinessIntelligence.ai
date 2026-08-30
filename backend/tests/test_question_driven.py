"""
The question-driven investigation: grounding, driver derivation, signal
filtering, and the guarantees that keep the language model honest.

The acceptance test in `TestDomainAwareness` is the one that matters most. The
system used to answer a question about a hospital by proposing that a competitor
had taken volume — a sentence hardcoded into a retail template that fired on any
dataset with a dimension column. That must now be impossible, not merely
unlikely, and the test asserts it directly.
"""
from __future__ import annotations

import unittest
from dataclasses import replace

import pandas as pd

from app.engines import metrics, observe
from app.engines.driver_graph import (attach_proposed_edges, build_driver_graph,
                                      period_change_table, verify_edge)
from app.engines.hypotheses import Context
from app.engines.investigate import determine_focus, investigate
from app.engines.llm_hypotheses import build_llm_candidates, category_summary, shortlist
from app.engines.signals import material_signals
from app.kpi import service as kpi_service
from app.kpi.domain import domain_context_from_contract
from app.models.investigation import DriverEdge
from app.agent.dimensions import MemberCatalogue
from app.query.grounding import ground_question
from app.query.periods import resolve_period
from app.query.understanding import interpret_question

HOSPITAL_CSV = "../sample_data/hospital_kpi_smoke_sample.csv"

# Vocabulary that only belongs to a retail explanation. If any of it reaches a
# hospital investigation, a template has leaked back in.
RETAIL_LEAKAGE = ("competitor", "competitive pressure", "discount", "stockout",
                  "assortment", "shopper", "basket", "merchandis")


def _hospital_dataset():
    """The hospital sample, with its KPI contract bootstrapped and attached."""
    df = metrics.normalise_columns(pd.read_csv(HOSPITAL_CSV))
    schema = metrics.detect_schema(df)
    df = metrics.prepare(df, schema)
    contract = kpi_service.get_or_bootstrap(
        "test_uid", {"_id": "ds_hospital", "filename": "hospital.csv"}, df, schema)
    resolver = kpi_service.resolver_for(contract)
    schema.contract_resolver = resolver
    schema.available_kpis = [k for k, c in resolver.items()
                             if set(c.source_fields) <= set(df.columns)]
    schema.domain = domain_context_from_contract(contract)
    return df, schema, contract


def _context(df, schema, metric):
    tfs = observe.available_timeframes(df)
    tf = observe.Timeframe(tfs[-1]["year"], tfs[-1]["quarter"])
    obs = observe.observe(df, schema, metric, tf, "previous_period")
    ctx = Context(df=df, schema=schema, metric=metric,
                  cur=observe.slice_period(df, tf),
                  base=observe.slice_period(df, tf.previous()),
                  observation=obs, focus=determine_focus(obs))
    return ctx, obs


class HospitalTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.schema, cls.contract = _hospital_dataset()


# ---------------------------------------------------------------------------
class TestDomainAwareness(HospitalTestCase):
    def test_the_domain_is_reachable_at_analysis_time(self):
        """Detection happens at discovery; the engines need it much later."""
        self.assertEqual(self.schema.domain_key, "healthcare")
        self.assertIn("hospital", self.schema.domain.vocab.label)
        self.assertIn("beds", self.schema.domain.vocab.capacity_note)

    def test_a_hospital_investigation_never_blames_a_competitor(self):
        """
        The acceptance test for this whole change.

        Every hypothesis must be derived from what this dataset measures. There
        is no competitor-price column in a hospital admissions export, so no
        competitor explanation can survive — it has nothing to be tested against.
        """
        ctx, obs = _context(self.df, self.schema, "readmission_rate")
        result = investigate(self.df, self.schema, obs, "test_uid", llm=None)

        blob = " ".join(
            f"{h.get('title', '')} {h.get('statement', '')} {h.get('mechanism', '')}"
            for h in result["hypotheses"]).lower()
        self.assertTrue(result["hypotheses"], "a real movement produced no explanation")
        for term in RETAIL_LEAKAGE:
            self.assertNotIn(term, blob, f"retail vocabulary '{term}' leaked into a hospital "
                                         f"investigation")

    def test_hypotheses_name_the_hospitals_own_measures(self):
        ctx, obs = _context(self.df, self.schema, "readmission_rate")
        result = investigate(self.df, self.schema, obs, "test_uid", llm=None)
        causes = {h.get("cause_metric") for h in result["hypotheses"]} - {None}
        self.assertTrue(causes <= set(self.schema.available_kpis),
                        "a hypothesis cited a measure this dataset does not have")
        self.assertIn("readmissions", causes,
                      "the numerator of the KPI that moved should be a candidate driver")


# ---------------------------------------------------------------------------
class TestQueryGrounding(HospitalTestCase):
    def test_a_named_kpi_resolves_without_a_model(self):
        intent = interpret_question("Why did the recovery rate drop?", self.schema,
                                    self.df, llm=None, contract=self.contract)
        self.assertIsNotNone(intent.outcome)
        self.assertEqual(intent.outcome.kpi_key, "recovery_rate")
        self.assertFalse(intent.llm_used)
        self.assertFalse(intent.blocked)

    def test_a_contrast_clause_becomes_a_comparison_not_the_outcome(self):
        intent = interpret_question(
            "Why did bed occupancy fall even though admissions rose?",
            self.schema, self.df, llm=None, contract=self.contract)
        self.assertEqual(intent.outcome.kpi_key, "bed_occupancy_rate")
        self.assertIn("admissions", [k.kpi_key for k in intent.comparison_kpis])

    def test_admissions_does_not_match_readmissions(self):
        """Substring matching on clinical vocabulary is a correctness bug."""
        grounded = ground_question("Why did admissions fall?", self.schema)
        keys = [c.kpi_key for c in grounded.outcome_candidates]
        self.assertIn("admissions", keys)
        self.assertNotIn("readmissions", keys)

    def test_an_unresolvable_question_blocks_rather_than_guessing(self):
        intent = interpret_question("Why did everything get worse?", self.schema,
                                    self.df, llm=None, contract=self.contract)
        self.assertTrue(intent.blocked)
        self.assertIsNone(intent.outcome)
        blocking = [a for a in intent.ambiguities if a.blocking]
        self.assertTrue(blocking[0].candidates, "a block should offer the reader a way forward")

    def test_a_period_the_dataset_lacks_blocks_rather_than_substituting(self):
        intent = interpret_question("Why did readmissions rise in Q3 2019?", self.schema,
                                    self.df, llm=None, contract=self.contract)
        self.assertTrue(intent.blocked)
        self.assertTrue(any(a.kind == "unsupported_by_dataset" for a in intent.ambiguities))

    def test_a_vague_period_proceeds_and_discloses(self):
        intent = interpret_question("Why did readmissions rise?", self.schema,
                                    self.df, llm=None, contract=self.contract)
        self.assertFalse(intent.blocked)
        self.assertEqual(intent.period.source, "dataset_latest")
        self.assertTrue(intent.assumptions, "an assumed period must be disclosed")

    def test_relative_periods_follow_the_dataset_not_the_clock(self):
        timeframes = [{"year": 2026, "quarter": q, "label": f"2026-Q{q}"} for q in (1, 2, 3, 4)]
        period, _ = resolve_period("Why did sales drop last quarter?", timeframes)
        self.assertEqual((period.year, period.quarter), (2026, 4))

    def test_repeated_questions_are_served_from_cache_without_corruption(self):
        """
        Interpretation is cached per (question, dataset, contract version) so a
        repeated question does not re-run grounding or an LLM round-trip. The
        cache must never let one caller's mutation of its own returned intent
        leak into what a later caller receives.
        """
        from app.query import clear_intent_cache, interpret_question_cached
        from app.query.understanding import _INTENT_CACHE

        clear_intent_cache()
        q = "Why did readmissions rise?"
        first = interpret_question_cached(q, self.schema, self.df, "ds_hospital", llm=None)
        self.assertEqual(len(_INTENT_CACHE), 1)

        second = interpret_question_cached(q, self.schema, self.df, "ds_hospital", llm=None)
        self.assertEqual(len(_INTENT_CACHE), 1, "a repeated question grew the cache")
        self.assertIsNot(first, second, "a cache hit must return an independent copy")
        self.assertEqual(first.outcome.kpi_key, second.outcome.kpi_key)

        second.unmapped_terms.append("CORRUPTED")
        third = interpret_question_cached(q, self.schema, self.df, "ds_hospital", llm=None)
        self.assertNotIn("CORRUPTED", third.unmapped_terms,
                         "mutating a returned intent corrupted the cached entry")

        # Normalisation: case/whitespace differences are the same question.
        interpret_question_cached("  Why DID readmissions   rise?  ", self.schema, self.df,
                                  "ds_hospital", llm=None)
        self.assertEqual(len(_INTENT_CACHE), 1)

        # A different dataset is a different entry.
        interpret_question_cached(q, self.schema, self.df, "ds_other", llm=None)
        self.assertEqual(len(_INTENT_CACHE), 2)

        clear_intent_cache()

    def test_a_contract_version_bump_invalidates_the_cache(self):
        """
        A KPI Studio edit changes the contract version, which is exactly the
        signal a cached interpretation must not survive — the question could
        now resolve to a different KPI than it did before the edit.
        """
        from app.query import clear_intent_cache, interpret_question_cached
        from app.query.understanding import _INTENT_CACHE

        original_version = self.schema.contract_version
        try:
            clear_intent_cache()
            q = "Why did readmissions rise?"
            self.schema.contract_version = 1
            interpret_question_cached(q, self.schema, self.df, "ds_versioned", llm=None)
            self.assertEqual(len(_INTENT_CACHE), 1)

            self.schema.contract_version = 2
            interpret_question_cached(q, self.schema, self.df, "ds_versioned", llm=None)
            self.assertEqual(len(_INTENT_CACHE), 2, "a version bump should be a cache miss")
        finally:
            self.schema.contract_version = original_version
            clear_intent_cache()

    def test_a_model_may_not_invent_a_kpi(self):
        class Inventing:
            enabled = True

            def understand_question(self, *a, **k):
                return {"outcome_kpi_key": "profit_margin_that_does_not_exist",
                        "confidence": 0.99}

        intent = interpret_question("Why did the thing change?", self.schema, self.df,
                                    llm=Inventing(), contract=self.contract)
        self.assertIsNone(intent.outcome)
        self.assertTrue(intent.blocked)


# ---------------------------------------------------------------------------
class TestDriverGraph(HospitalTestCase):
    def test_a_ratio_resolves_to_its_numerator_and_denominator(self):
        graph = build_driver_graph(self.schema, "readmission_rate")
        by_relation = {e.relation: e.source_kpi for e in graph.edges}
        self.assertEqual(by_relation.get("formula_numerator"), "readmissions")
        self.assertEqual(by_relation.get("formula_denominator"), "discharges")

    def test_formula_edges_outrank_thematic_ones(self):
        graph = build_driver_graph(self.schema, "readmission_rate")
        formula = [e for e in graph.edges if e.relation.startswith("formula_")]
        thematic = [e for e in graph.edges if e.relation == "semantic_tag"]
        self.assertTrue(formula)
        if thematic:
            self.assertGreater(min(e.weight for e in formula),
                               max(e.weight for e in thematic))

    def test_an_unverified_proposed_edge_carries_no_weight(self):
        """
        A model may suggest a relationship; only the data may confer weight on it.
        """
        graph = build_driver_graph(self.schema, "readmission_rate")
        proposed = [DriverEdge(source_kpi="admissions", target_kpi="readmission_rate",
                               relation="llm_proposed", origin="llm", weight=0.9)]
        graph = attach_proposed_edges(graph, proposed, self.df, self.schema)
        for edge in graph.unverified_llm_edges:
            self.assertFalse(edge.verified)
            self.assertEqual(edge.weight, 0.0)
            self.assertTrue(edge.verification_note)
        for edge in graph.edges:
            if edge.origin == "llm":
                self.assertTrue(edge.verified, "an unverified edge reached the graph")

    def test_depends_on_is_populated_from_the_formula(self):
        """
        `depends_on` used to be dead — hand-authored, never populated by
        discovery. `compile_contract` now derives it structurally, so it is a
        real, reusable relationship rather than an always-empty field.
        """
        resolver = self.schema.contract_resolver
        self.assertEqual(set(resolver["readmission_rate"].depends_on),
                         {"readmissions", "discharges"})
        self.assertEqual(set(resolver["bed_occupancy_rate"].depends_on),
                         {"bed_days", "beds_available"})
        self.assertEqual(resolver["admissions"].depends_on, [])

    def test_period_change_table_is_differenced_not_levels(self):
        table = period_change_table(self.df, ["readmission_rate", "admissions"],
                                    self.schema.contract_resolver)
        self.assertIn("readmission_rate__chg", table.columns)
        self.assertIn("admissions__chg", table.columns)
        self.assertEqual(len(table), len(observe.available_timeframes(self.df)) - 1)


# ---------------------------------------------------------------------------
class TestMaterialSignals(HospitalTestCase):
    def setUp(self):
        self.ctx, self.obs = _context(self.df, self.schema, "readmission_rate")

    def test_noise_is_not_offered_as_something_to_explain(self):
        signals = material_signals(self.obs, {}, self.ctx.focus)
        self.assertLess(signals.signals_retained, signals.signals_considered,
                        "nothing was filtered, so the boundary is doing no work")
        for entry in signals.material:
            self.assertTrue(entry["moved"])

    def test_a_named_but_flat_comparison_is_kept_as_context(self):
        """
        "Why did X fall even though Y was flat" is unanswerable without Y.
        """
        tfs = observe.available_timeframes(self.df)
        tf = observe.Timeframe(tfs[-1]["year"], tfs[-1]["quarter"])
        comparisons = {"admissions": observe.observe(self.df, self.schema, "admissions",
                                                     tf, "previous_period")}
        signals = material_signals(self.obs, comparisons, self.ctx.focus)
        carried = {e["kpi"]: e for e in signals.material + signals.context}
        self.assertIn("admissions", carried)
        self.assertTrue(carried["admissions"]["named_in_question"])

    def test_nothing_material_produces_no_hypotheses(self):
        """Asked to explain noise, the honest answer is that there is nothing to explain."""
        flat = dict(self.obs)
        flat["verdict"] = "within_normal_variation"
        flat["kpi_scoreboard"] = []
        signals = material_signals(flat, {}, {})
        self.assertTrue(signals.nothing_material)

        result = investigate(self.df, self.schema, self.obs, "test_uid",
                             llm=None, signals=signals)
        self.assertTrue(result["nothing_to_explain"])
        self.assertEqual(result["hypotheses"], [])


# ---------------------------------------------------------------------------
class TestLlmHypothesisContract(HospitalTestCase):
    def setUp(self):
        self.ctx, self.obs = _context(self.df, self.schema, "readmission_rate")
        self.allowed = list(self.schema.available_kpis)

    def _proposal(self, **overrides):
        base = {
            "key": "example", "title": "An example", "statement": "Something happened.",
            "mechanism": "How it happened.", "family": "quality", "domain_specific": True,
            "predictions": [{"metric": "readmissions", "expected_direction": "up",
                             "reference_pct": 10, "weight": 1.0, "rationale": "r"}],
            "cause_metric": "readmissions", "cause_direction": "up",
            "contradiction_queries": ["readmissions stable"], "rag_queries": [], "missing": [],
        }
        base.update(overrides)
        return {"hypotheses": [base]}

    def test_a_prediction_on_a_metric_that_does_not_exist_is_dropped(self):
        raw = self._proposal(
            key="competitor_pricing", title="Competitor pricing",
            predictions=[{"metric": "competitor_price_index", "expected_direction": "down",
                          "reference_pct": 5, "weight": 1.0, "rationale": "r"}],
            cause_metric="competitor_price_index")
        self.assertEqual(build_llm_candidates(self.ctx, raw, self.allowed), [],
                         "an untestable hypothesis reached the reader")

    def test_a_prediction_that_did_not_hold_becomes_evidence_against(self):
        """
        The model states expectations; the data decides what they support.
        """
        raw = self._proposal(
            predictions=[{"metric": "discharges", "expected_direction": "up",
                          "reference_pct": 5, "weight": 1.0, "rationale": "r"}],
            cause_metric="discharges")
        built = build_llm_candidates(self.ctx, raw, self.allowed)
        self.assertTrue(built)
        stances = {e["stance"] for e in built[0]["evidence"]}
        self.assertIn("contradicting", stances,
                      "discharges did not rise, so the prediction must count against")

    def test_an_unknown_family_is_normalised_and_still_recommendable(self):
        from app.engines.act import PLAYBOOK, generic_play

        raw = self._proposal(family="clinical_capacity_nonsense")
        built = build_llm_candidates(self.ctx, raw, self.allowed)
        self.assertEqual(built[0]["family"], "other")
        play = PLAYBOOK.get(built[0]["family"]) or generic_play(built[0])
        self.assertTrue(play["actions"], "an unknown family produced no recommendation")

    def test_braces_in_a_model_authored_action_do_not_raise(self):
        from app.engines.act import generic_play

        play = generic_play({"cause_metric": "readmissions"})
        rendered = [a.replace("{focus}", "Emergency {ward}") for a in play["actions"]]
        self.assertTrue(all(isinstance(r, str) for r in rendered))

    def test_malformed_output_degrades_rather_than_raising(self):
        self.assertEqual(build_llm_candidates(self.ctx, {"nonsense": True}, self.allowed), [])
        self.assertEqual(build_llm_candidates(self.ctx, None, self.allowed), [])

    def test_shortlist_reserves_a_place_for_each_category(self):
        domain = [{"key": f"d{i}", "domain_specific": True, "prior_support": 5.0 - i,
                   "prior_against": 0.0, "testable": True} for i in range(4)]
        general = [{"key": "g1", "domain_specific": False, "prior_support": 0.4,
                    "prior_against": 0.0, "testable": True}]
        picked = shortlist(domain + general, limit=3)
        self.assertIn("g1", [h["key"] for h in picked],
                      "the general category was crowded out entirely")

    def test_coverage_is_reported_honestly_when_a_category_is_empty(self):
        only_general = [{"key": "g1", "domain_specific": False}]
        summary = category_summary(only_general)
        self.assertEqual(summary["domain_specific_count"], 0)
        self.assertTrue(summary["coverage_note"])


# ---------------------------------------------------------------------------
class TestDeterminism(HospitalTestCase):
    def test_the_same_question_yields_the_same_numbers(self):
        runs = []
        for _ in range(3):
            ctx, obs = _context(self.df, self.schema, "readmission_rate")
            result = investigate(self.df, self.schema, obs, "test_uid", llm=None)
            runs.append((
                round(obs["change_pct"], 6),
                [(h["key"], h["prior_support"], h["prior_against"])
                 for h in result["hypotheses"]],
            ))
        self.assertEqual(runs[0], runs[1])
        self.assertEqual(runs[1], runs[2])

    def test_the_pipeline_completes_with_no_model_available(self):
        ctx, obs = _context(self.df, self.schema, "readmission_rate")
        result = investigate(self.df, self.schema, obs, "test_uid", llm=None)
        self.assertFalse(result["llm_used"])
        self.assertTrue(result["hypotheses"],
                        "the deterministic floor produced nothing without a model")


# ---------------------------------------------------------------------------
class TestPersonaReframing(HospitalTestCase):
    """
    The same findings, different advice.

    Persona may change what a reader is told to do and how much detail they are
    given. It may not change what the evidence says, how confident the system
    is, or which explanation ranked first — if it could, two colleagues could
    read one investigation and disagree about what happened.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from app.engines.act import act
        from app.engines.contest import contest

        ctx, cls.obs = _context(cls.df, cls.schema, "readmission_rate")
        cls.investigation = investigate(cls.df, cls.schema, cls.obs, "test_uid", llm=None)
        cls.contested = contest(cls.df, cls.schema, cls.obs, cls.investigation,
                                "test_uid", llm=None)
        cls.views = {}
        for persona in ("business_analyst", "business_manager", "business_leader",
                        "domain_specialist", "operational_user"):
            result = act(cls.df, cls.obs, cls.investigation, cls.contested, llm=None,
                         resolver=cls.schema.contract_resolver, persona=persona)
            cls.views[persona] = result["persona_view"]

    def test_the_evidence_is_identical_for_every_persona(self):
        bases = [
            [(c["hypothesis_key"], c["confidence"], c["causal_claim"],
              tuple(c["supporting_evidence"]), tuple(c["contradicting_evidence"]))
             for c in view["recommendation_basis"]]
            for view in self.views.values()
        ]
        first = bases[0]
        for other in bases[1:]:
            self.assertEqual(first, other,
                             "a persona changed the evidence, the ranking or the confidence")

    def test_the_recommendations_genuinely_differ(self):
        actions = {p: v["recommendations"][0]["action"] for p, v in self.views.items()}
        self.assertEqual(len(set(actions.values())), len(actions),
                         f"two personas were given identical advice: {actions}")

    def test_each_persona_gets_its_own_horizon_and_owner(self):
        horizons = {v["recommendations"][0]["timeframe"] for v in self.views.values()}
        owners = {v["recommendations"][0]["owner"] for v in self.views.values()}
        self.assertGreaterEqual(len(horizons), 4)
        self.assertGreaterEqual(len(owners), 4)
        self.assertEqual(self.views["operational_user"]["recommendations"][0]["timeframe"],
                         "immediately")

    def test_no_persona_may_act_on_an_unestablished_driver(self):
        allowed = {c["cause_metric"] for c in
                   self.views["business_analyst"]["recommendation_basis"]}
        allowed |= {c["hypothesis_key"] for c in
                    self.views["business_analyst"]["recommendation_basis"]}
        for persona, view in self.views.items():
            for rec in view["recommendations"]:
                basis = rec.get("based_on")
                if basis:
                    self.assertIn(basis, allowed,
                                  f"{persona} was advised to act on something the evidence "
                                  f"did not establish")

    def test_depth_follows_the_persona(self):
        analyst = len(self.views["business_analyst"]["summary"])
        leader = len(self.views["business_leader"]["summary"])
        self.assertGreater(analyst, leader,
                           "the analyst summary should carry more than the leader's")

    def test_an_unknown_persona_falls_back_to_neutral_framing(self):
        from app.personas import persona_for

        self.assertEqual(persona_for(None).key, "business_analyst")
        self.assertEqual(persona_for("chief_vibes_officer").key, "business_analyst")

    def test_a_model_may_not_smuggle_in_an_unsupported_action(self):
        """A reframing that reaches outside the evidence is dropped, not shown."""
        from app.personas.reframe import reframe
        from app.personas.profiles import BUSINESS_LEADER

        class Inventing:
            enabled = True

            def write_for_persona(self, *a, **k):
                return {"summary": "s", "recommendations": [
                    {"action": "Cut prices to beat the competitor.", "why": "w",
                     "owner": "o", "timeframe": "t", "based_on": "competitor_price_index"},
                    {"action": "Review readmissions.", "why": "w", "owner": "o",
                     "timeframe": "t", "based_on": "readmissions"},
                ]}

        view = reframe(BUSINESS_LEADER, self.obs, self.investigation, self.contested,
                       [], llm=Inventing())
        actions = [r["action"] for r in view["recommendations"]]
        self.assertNotIn("Cut prices to beat the competitor.", actions)
        self.assertIn("Review readmissions.", actions)


# ---------------------------------------------------------------------------
class TestQuestionGroundingBindsDimensionMembers(unittest.TestCase):
    """
    C4. `_match_dimensions` read `schema.dimension_members`, an attribute that
    never existed, so `entity_filters` was permanently `{}` and no question has
    ever bound a filter. These pin the binding, and equally pin the guards that
    stop a live index from binding things it should not.
    """

    @classmethod
    def setUpClass(cls):
        from .base import setup_environment
        from app.services import dataset_service

        state = setup_environment()
        cls.df, cls.schema = dataset_service.load(state["dataset"], state["uid"])

    def test_a_question_naming_a_region_binds_that_region_as_a_filter(self):
        """The headline case, and the one that has never worked."""
        grounded = ground_question("Why did revenue fall in North?", self.schema)
        self.assertEqual(grounded.entity_filters, {"region": "North"})
        self.assertIn("region", grounded.dimension_hints)

    def test_a_member_word_the_question_already_spent_on_a_kpi_does_not_also_become_a_filter(self):
        """
        `returns` is both a KPI and, on some datasets, a member value. The
        measure the question is *about* has first claim on the word.
        """
        frame = self.df.copy()
        frame["channel"] = "Returns"
        schema = replace(self.schema)
        schema.member_catalogue = MemberCatalogue(frame, schema.dimensions)

        grounded = ground_question("Why did returns rise?", schema)
        self.assertIn("returns", [c.kpi_key for c in grounded.outcome_candidates])
        self.assertNotIn("channel", grounded.entity_filters)

    def test_a_bare_year_is_never_read_as_a_dimension_member(self):
        frame = self.df.copy()
        frame["segment"] = "2024"
        schema = replace(self.schema)
        schema.member_catalogue = MemberCatalogue(frame, schema.dimensions)

        grounded = ground_question("What was revenue in 2024?", schema)
        self.assertEqual(grounded.entity_filters, {})

    def test_a_member_that_is_also_a_stopword_does_not_bind(self):
        frame = self.df.copy()
        frame["segment"] = "All"
        frame.loc[frame.index[:10], "segment"] = "Down"
        schema = replace(self.schema)
        schema.member_catalogue = MemberCatalogue(frame, schema.dimensions)

        grounded = ground_question("Why is revenue down for all products?", schema)
        self.assertNotIn("segment", grounded.entity_filters)

    def test_two_members_of_one_dimension_bind_the_longer_match_deterministically(self):
        """
        A column holding both "North" and "North East" must bind the longer
        phrase, and must bind the same one on every run -- the old code took
        whichever value `unique()` happened to yield first.
        """
        frame = self.df.copy()
        frame["region"] = "North"
        frame.loc[frame.index[:10], "region"] = "North East"
        schema = replace(self.schema)
        schema.member_catalogue = MemberCatalogue(frame, schema.dimensions)

        readings = {ground_question("Why did revenue fall in North East?",
                                    schema).entity_filters.get("region")
                    for _ in range(5)}
        self.assertEqual(readings, {"North East"})

    def test_a_value_held_by_two_dimensions_binds_neither(self):
        frame = self.df.copy()
        frame["channel"] = frame["region"]
        schema = replace(self.schema)
        schema.member_catalogue = MemberCatalogue(frame, schema.dimensions)

        grounded = ground_question("Why did revenue fall in North?", schema)
        self.assertEqual(grounded.entity_filters, {},
                         "an ambiguous member was silently assigned to one column")

    def test_the_same_binding_works_for_a_department_and_for_a_grade(self):
        """
        Three domains, one behaviour. A retail-only assumption fails here rather
        than in front of a user -- the precedent `test_cross_sectional_dimension`
        sets for every dimension-aware feature.
        """
        cases = [
            ("../sample_data/hospital_kpi_smoke_sample.csv",
             "Why did admissions fall in Cardiology?", "department", "Cardiology"),
            ("../sample_data/school_kpi_smoke_sample.csv",
             "Why did attendance fall in Grade_7?", "grade", "Grade_7"),
            ("../sample_data/school_kpi_smoke_sample.csv",
             "Why did attendance fall at North Campus?", "campus", "North Campus"),
        ]
        for path, question, dimension, member in cases:
            with self.subTest(question=question):
                raw = metrics.normalise_columns(pd.read_csv(path))
                schema = metrics.detect_schema(raw)
                frame = metrics.prepare(raw, schema)
                schema.member_catalogue = MemberCatalogue(frame, schema.dimensions)

                grounded = ground_question(question, schema)
                self.assertEqual(grounded.entity_filters.get(dimension), member)

    def test_a_bound_member_is_not_reported_as_an_unmapped_term(self):
        """
        `unmapped_terms` means "looked measurable, bound to nothing". A word
        that bound a filter is the opposite of that, and multi-word members
        must have every token accounted for.
        """
        grounded = ground_question("Why did revenue fall for Product A?", self.schema)
        self.assertEqual(grounded.entity_filters, {"product": "Product A"})
        self.assertNotIn("product", grounded.unmapped_terms)

    def test_a_dataset_with_no_catalogue_grounds_exactly_as_before(self):
        """The catalogue is additive; a schema without one must not regress."""
        schema = replace(self.schema)
        schema.member_catalogue = None
        grounded = ground_question("Why did revenue fall in North?", schema)
        self.assertEqual(grounded.entity_filters, {})
        self.assertTrue(grounded.outcome_candidates)


# ---------------------------------------------------------------------------
class TestATieIsAnsweredNotRefused(HospitalTestCase):
    """
    C5. A tie between two KPIs used to be a blocking `outcome_multiple` -- the
    "please pick a KPI" wall. Every candidate in a tie is a measure this dataset
    genuinely has, so the best-scoring one is used and the rest are disclosed.

    The distinction that survives: an outcome resolving to *nothing* still
    blocks, because there is no answer to give.
    """

    TIED_QUESTION = "Why did the rate change?"

    def _tie(self, llm=None):
        return interpret_question(self.TIED_QUESTION, self.schema, self.df,
                                  llm=llm, contract=self.contract)

    def test_a_tie_resolves_to_the_top_candidate_rather_than_blocking(self):
        intent = self._tie()
        self.assertFalse(intent.blocked)
        self.assertIsNotNone(intent.outcome)
        self.assertIn(intent.outcome.kpi_key, self.schema.available_kpis)

    def test_a_tie_discloses_the_candidates_it_did_not_pick(self):
        intent = self._tie()
        tie = next(a for a in intent.ambiguities if a.kind == "outcome_multiple")
        self.assertFalse(tie.blocking)
        self.assertGreaterEqual(len(tie.candidates), 2)
        self.assertIn(tie.message, intent.assumptions,
                      "a tie must be disclosed to the reader, not swallowed")

    def test_a_tie_records_the_chosen_key_as_an_assumption(self):
        intent = self._tie()
        tie = next(a for a in intent.ambiguities if a.kind == "outcome_multiple")
        self.assertEqual(tie.assumed, intent.outcome.kpi_key)

    def test_a_tie_consults_the_model_which_the_old_path_never_did(self):
        """
        The LLM fallback only ever ran on an empty candidate list, so the one
        case a model is best placed to settle was the one it was never asked
        about.
        """
        baseline = self._tie().outcome.kpi_key
        alternative = next(
            c.kpi_key for c in next(a for a in self._tie().ambiguities
                                    if a.kind == "outcome_multiple").candidates
            if c.kpi_key != baseline)

        class Deciding:
            enabled = True

            def understand_question(self, *a, **k):
                return {"outcome_kpi_key": alternative, "confidence": 0.95}

        intent = self._tie(llm=Deciding())
        self.assertEqual(intent.outcome.kpi_key, alternative)
        self.assertTrue(intent.llm_used)
        self.assertFalse(intent.blocked)

    def test_a_model_that_names_an_untied_kpi_is_ignored_and_the_top_candidate_stands(self):
        """
        The model may break the tie; it may not step outside it. An
        unconstrained pick would let it choose a KPI that scored nothing.
        """
        baseline = self._tie().outcome.kpi_key
        outside = next(k for k in self.schema.available_kpis
                       if k not in {c.kpi_key for c in
                                    next(a for a in self._tie().ambiguities
                                         if a.kind == "outcome_multiple").candidates})

        class Wandering:
            enabled = True

            def understand_question(self, *a, **k):
                return {"outcome_kpi_key": outside, "confidence": 0.99}

        intent = self._tie(llm=Wandering())
        self.assertEqual(intent.outcome.kpi_key, baseline)

    def test_no_blocking_ambiguity_for_a_kpi_tie_survives_anywhere(self):
        """
        The C5 guarantee, swept across three domains: `outcome_unresolved` is
        now the only blocking outcome ambiguity there is.
        """
        questions = [
            "Why did the rate change?", "What changed?", "Why did the count move?",
            "Why did everything get worse?", "Why did the total fall?",
            "How did performance change last quarter?", "Why is the average down?",
        ]
        for path in ("../sample_data/business_metrics_sample.csv",
                     "../sample_data/hospital_kpi_smoke_sample.csv",
                     "../sample_data/school_kpi_smoke_sample.csv"):
            raw = metrics.normalise_columns(pd.read_csv(path))
            schema = metrics.detect_schema(raw)
            frame = metrics.prepare(raw, schema)
            contract = kpi_service.get_or_bootstrap(
                "test_uid", {"_id": f"ds_{path}", "filename": path}, frame, schema)
            resolver = kpi_service.resolver_for(contract)
            schema.contract_resolver = resolver
            schema.available_kpis = [k for k, c in resolver.items()
                                     if set(c.source_fields) <= set(frame.columns)]
            for question in questions:
                with self.subTest(path=path, question=question):
                    intent = interpret_question(question, schema, frame,
                                                llm=None, contract=contract)
                    offenders = [a.kind for a in intent.ambiguities
                                 if a.blocking and a.kind == "outcome_multiple"]
                    self.assertEqual(offenders, [])

    def test_an_unresolvable_outcome_still_blocks(self):
        """The distinction C5 preserves: nothing matched is not a tie."""
        intent = interpret_question("Why did everything get worse?", self.schema,
                                    self.df, llm=None, contract=self.contract)
        self.assertTrue(intent.blocked)
        self.assertIsNone(intent.outcome)
        self.assertTrue(any(a.kind == "outcome_unresolved" and a.blocking
                            for a in intent.ambiguities))


if __name__ == "__main__":
    unittest.main()
