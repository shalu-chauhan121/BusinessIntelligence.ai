"""
Question grounding, driver-graph derivation, and dimension-member binding.

`TestDomainAwareness`, `TestMaterialSignals`, `TestLlmHypothesisContract`,
`TestDeterminism` and `TestPersonaReframing` were retired at A9 along with
`engines/{hypotheses,investigate,llm_hypotheses,signals}.py` and
`personas/reframe.py`, the modules they exercised. What remains here —
`interpret_question`/`ground_question` (`query/`) and the driver graph
(`engines/driver_graph.py`) — is independent of that retired pipeline.
"""
from __future__ import annotations

import unittest
from dataclasses import replace

import pandas as pd

from app.engines import metrics, observe
from app.engines.driver_graph import (attach_proposed_edges, build_driver_graph,
                                      period_change_table)
from app.kpi import service as kpi_service
from app.kpi.domain import domain_context_from_contract
from app.models.investigation import DriverEdge
from app.agent.dimensions import MemberCatalogue
from app.query.grounding import ground_question
from app.query.periods import resolve_period
from app.query.understanding import interpret_question

HOSPITAL_CSV = "../sample_data/hospital_kpi_smoke_sample.csv"


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


class HospitalTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.schema, cls.contract = _hospital_dataset()


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
