"""
KPI relevance search -- the ranked-candidate service that replaced a wall.

`understanding._resolve_outcome` used to treat two KPIs scoring within
`TIE_MARGIN` as an impasse and refuse to answer. The evidence ladder that
produced those candidates was sound; what was wrong was the conclusion drawn
from a tie. The ladder now lives here, returns a ranked list, and never refuses.

The pair of tests that matter most are `TestGroundingDelegatesRatherThanDuplicating`:
the ladder moved out of `grounding` rather than being copied, and these assert
that the two agree candidate for candidate and that no scoring constant survived
in the old home. Two copies of a seven-tier ladder drift within a release.
"""
from __future__ import annotations

import unittest
from pathlib import Path

from app.agent.contract_api import ContractAPI
from app.agent.kpi_search import (SCORE_DEFINITION, SCORE_EXACT_NAME,
                                  SCORE_PHRASE_IN_NAME, SCORE_RELEVANCE,
                                  SCORE_SEMANTIC_TAG, SCORE_WORD_IN_NAME,
                                  KpiMatch, KpiSearch, concept_alias_index)
from .base import EngineTestCase, contracted as _contracted

SAMPLES = Path(__file__).resolve().parents[2] / "sample_data"


class SearchTestCase(unittest.TestCase):
    CSV = "hospital_kpi_smoke_sample.csv"

    @classmethod
    def setUpClass(cls):
        cls.df, cls.schema = _contracted(cls.CSV)
        cls.search = KpiSearch(ContractAPI(cls.schema))

    def scores(self, text):
        return {m.kpi_key: m.score for m in self.search.search(text).matches}

    def bases(self, text):
        return {m.kpi_key: m.basis for m in self.search.search(text).matches}


class TestTheEvidenceLadderIsOrdered(SearchTestCase):
    def test_an_exact_kpi_name_scores_one(self):
        self.assertEqual(self.scores("admissions").get("admissions"), SCORE_EXACT_NAME)
        self.assertEqual(self.bases("admissions").get("admissions"), "exact_name")

    def test_a_multi_word_fragment_of_a_label_outscores_a_single_word_one(self):
        """"bed occupancy" is strong evidence; "rate" must not claim every rate."""
        phrase = self.scores("bed occupancy").get("bed_occupancy_rate")
        self.assertEqual(phrase, SCORE_PHRASE_IN_NAME)

        single = self.search.search("rate").matches
        for match in single:
            self.assertLessEqual(match.score, SCORE_WORD_IN_NAME)

    def test_a_semantic_tag_outscores_a_concept_alias(self):
        self.assertGreater(SCORE_SEMANTIC_TAG, 0.7)

    def test_a_business_definition_overlap_is_weaker_than_any_name_match(self):
        self.assertLess(SCORE_DEFINITION, SCORE_WORD_IN_NAME)
        self.assertLess(SCORE_DEFINITION, SCORE_PHRASE_IN_NAME)

    def test_admissions_does_not_reach_readmissions(self):
        """Substring matching on clinical vocabulary is a correctness bug."""
        keys = set(self.scores("admissions"))
        self.assertIn("admissions", keys)
        self.assertNotIn("readmissions", keys)


class TestRankingIsStableAndTotal(SearchTestCase):
    def test_the_same_text_ranks_identically_across_repeated_searches(self):
        runs = {tuple((m.kpi_key, m.score, m.basis)
                      for m in self.search.search("readmission rate").matches)
                for _ in range(5)}
        self.assertEqual(len(runs), 1, "the same question ranked two ways")

    def test_two_kpis_on_the_same_score_are_ordered_by_key(self):
        matches = self.search.search("rate of admissions and discharges").matches
        for earlier, later in zip(matches, matches[1:]):
            if earlier.score == later.score:
                self.assertLess(earlier.kpi_key, later.kpi_key)

    def test_results_descend_by_score(self):
        matches = self.search.search("readmission rate").matches
        self.assertEqual([m.score for m in matches],
                         sorted((m.score for m in matches), reverse=True))

    def test_a_limit_takes_the_best_candidates_not_an_arbitrary_slice(self):
        everything = self.search.search("rate", limit=None).matches
        capped = self.search.search("rate", limit=2).matches
        self.assertLessEqual(len(capped), 2)
        self.assertEqual(list(capped), list(everything[:len(capped)]))


class TestAnUnmatchedTermIsEmptyNotAnError(SearchTestCase):
    def test_a_term_no_kpi_matches_returns_no_candidates(self):
        result = self.search.search("photosynthesis")
        self.assertEqual(result.matches, ())
        self.assertEqual(result.matched_grams, ())

    def test_an_empty_question_returns_no_candidates(self):
        for text in ("", "   ", "?!"):
            self.assertEqual(self.search.search(text).matches, (), repr(text))

    def test_a_dataset_with_no_available_kpis_returns_no_candidates(self):
        from dataclasses import replace

        empty = replace(self.schema)
        empty.available_kpis = []
        self.assertEqual(KpiSearch(ContractAPI(empty)).search("admissions").matches, ())

    def test_a_tie_returns_both_candidates_rather_than_raising(self):
        """
        The whole point of C5. Two KPIs matching a phrase equally well is a
        ranked list with two rows, not an impasse.
        """
        result = self.search.search("rate", limit=None)
        near = [m for m in result.matches
                if result.matches and result.matches[0].score - m.score < 0.15]
        self.assertGreaterEqual(len(near), 2)


class TestTheSearchGoesThroughTheFacade(unittest.TestCase):
    def test_the_module_never_reads_contract_resolver_directly(self):
        """
        C1's rule: new code reaches the contract through `ContractAPI`, so
        there is no per-call resolver argument anyone can forget to pass.
        """
        import re
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        source = (root / "app/agent/kpi_search.py").read_text(encoding="utf-8")
        # The docstring names the attribute to say it is not touched; what has
        # to stay clean is the executable code.
        code = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
        code = re.sub(r"#.*", "", code)
        self.assertNotIn("contract_resolver", code)
        self.assertNotIn("available_kpis", code)


class TestNoProseCrossesTheBoundary(SearchTestCase):
    """Tools return facts; the model writes the sentences."""

    ALLOWED_STRING_FIELDS = {"kpi_key", "label", "basis", "matched_gram"}
    MAX_LENGTH = 64

    def test_a_match_carries_no_sentence(self):
        from dataclasses import fields as dataclass_fields

        for match in self.search.search("readmission rate").matches:
            for f in dataclass_fields(match):
                value = getattr(match, f.name)
                if isinstance(value, str):
                    self.assertIn(f.name, self.ALLOWED_STRING_FIELDS)
                    self.assertLess(len(value), self.MAX_LENGTH)

    def test_the_matched_gram_is_never_a_sentence(self):
        for match in self.search.search("readmission rate").matches:
            self.assertNotIn(" ", match.matched_gram, "a space means a sentence")

    def test_no_note_or_detail_field_exists_on_a_match(self):
        names = {f.name for f in KpiMatch.__dataclass_fields__.values()}
        for banned in ("note", "detail", "message", "statement", "mechanism"):
            self.assertNotIn(banned, names)


class TestGroundingDelegatesRatherThanDuplicating(unittest.TestCase):
    """
    The ladder moved down a layer; it was not copied. Two implementations of a
    seven-tier scoring ladder drift within a release, and "scores are stable"
    is untestable if two of them are allowed to disagree.
    """

    QUESTIONS = (
        "Why did admissions fall?",
        "Why did the recovery rate drop?",
        "Why did bed occupancy fall even though admissions rose?",
        "What is driving readmissions?",
        "Why did revenue fall in North?",
        "Why did revenue rise but margin fall?",
        "How did the fulfilment rate change?",
        "Why did attendance fall at North Campus?",
        "Why did everything get worse?",
        "photosynthesis",
        "",
        "Why did the cost per admission rise?",
    )

    @classmethod
    def setUpClass(cls):
        cls.datasets = [_contracted(name) for name in (
            "business_metrics_sample.csv",
            "hospital_kpi_smoke_sample.csv",
            "school_kpi_smoke_sample.csv",
        )]

    def test_grounding_and_the_search_service_agree_on_every_candidate_for_a_battery_of_questions(self):
        from app.query.grounding import ground_question

        for _, schema in self.datasets:
            search = KpiSearch(ContractAPI(schema))
            for question in self.QUESTIONS:
                with self.subTest(question=question, kpis=len(schema.available_kpis)):
                    grounded = ground_question(question, schema)
                    # `ground_question` splits on a contrast marker, so compare
                    # against the same main clause the service would be given.
                    from app.query.grounding import _split_contrast

                    main, _ = _split_contrast(question)
                    expected = [(m.kpi_key, m.basis, m.score)
                                for m in search.search(main, limit=None).matches]
                    actual = [(c.kpi_key, c.match_basis, c.confidence)
                              for c in grounded.outcome_candidates]
                    self.assertEqual(actual, expected)

    def test_no_scoring_constant_survives_in_grounding(self):
        import inspect

        from app.query import grounding

        source = inspect.getsource(grounding._match_clause)
        for constant in ("0.85", "0.55", "0.8", "0.7", "0.65", "0.35"):
            self.assertNotIn(constant, source,
                             f"scoring constant {constant} is still in grounding")

    def test_the_alias_index_is_built_once_however_many_questions_are_asked(self):
        from app.query.grounding import ground_question

        concept_alias_index.cache_clear()
        _, schema = self.datasets[1]
        for _ in range(5):
            ground_question("Why did admissions fall?", schema)
        info = concept_alias_index.cache_info()
        self.assertEqual(info.misses, 1, "the concept library was walked more than once")
        self.assertGreaterEqual(info.hits, 4)


class TestRelevanceTextIsScoredAtLast(EngineTestCase):
    """
    `relevance` has been carried on every KPI definition and never scored
    against -- a live dead branch. It is the weakest tier on purpose: relevance
    prose says why a KPI matters, which is poor evidence about which KPI was
    named.
    """

    def test_relevance_prose_scores_below_the_business_definition(self):
        self.assertLess(SCORE_RELEVANCE, SCORE_DEFINITION)

    def test_a_relevance_match_alone_never_clears_the_outcome_confidence_floor(self):
        from app.query.understanding import MIN_OUTCOME_CONFIDENCE

        self.assertLess(SCORE_RELEVANCE, MIN_OUTCOME_CONFIDENCE)
        self.assertLess(SCORE_DEFINITION, MIN_OUTCOME_CONFIDENCE)

    def test_a_phrase_only_in_the_relevance_text_is_matched_and_reported(self):
        from dataclasses import replace

        from app.services import dataset_service

        _, schema = dataset_service.load(self.dataset, self.uid)
        key = schema.available_kpis[0]
        definition = schema.kpi_definition(key)
        if definition is None:                       # pragma: no cover - contract-less
            self.skipTest("dataset carries no KPI definitions")

        schema = replace(schema)
        search = KpiSearch(ContractAPI(schema))
        entry = next(e for e in search.entries() if e.key == key)
        object.__setattr__(entry, "relevance", "a wholly distinctive marker phrase")
        object.__setattr__(entry, "definition", "")

        result = search.search("distinctive marker phrase")
        by_key = {m.kpi_key: m for m in result.matches}
        self.assertIn(key, by_key)
        self.assertEqual(by_key[key].basis, "relevance_text")
        self.assertEqual(by_key[key].score, SCORE_RELEVANCE)


if __name__ == "__main__":
    unittest.main()
