"""
The dimension member catalogue -- the index `query/grounding.py` has always read
and never found.

`_match_dimensions` reads `schema.dimension_members` to turn a named entity in a
question into a filter. That attribute never existed, so `entity_filters` was
permanently `{}` and a question naming a region bound nothing. These tests pin
the index that closes it, and equally pin the two things that must *not* happen
now that members can bind: a catalogue object reaching the persisted schema
document, and a whole high-cardinality column being dumped into a lookup.
"""
from __future__ import annotations

import unittest

import pandas as pd

from app.agent.dimensions import (DEFAULT_PAGE_SIZE, MAX_INDEXED_MEMBERS,
                                  MemberCatalogue, MemberMatch, MemberPage, normalise)
from app.agent.errors import UnknownDimensionError

from .base import EngineTestCase


def _wide_frame(members: int = 10_000) -> pd.DataFrame:
    """A frame with one narrow column and one far past the index cap."""
    return pd.DataFrame({
        "narrow": ["North", "South"] * (members // 2),
        "wide": [f"Entity {i:05d}" for i in range(members)],
    })


class TestTheCatalogueMirrorsTheFrame(EngineTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.catalogue = MemberCatalogue(cls.df, cls.schema.dimensions)

    def test_every_member_of_every_dimension_matches_the_frames_own_uniques(self):
        for dimension in self.catalogue.dimensions():
            page = self.catalogue.members(dimension, limit=MAX_INDEXED_MEMBERS)
            expected = sorted(self.df[dimension].astype(str).unique())
            self.assertEqual(list(page.members), expected)

    def test_distinct_count_is_the_uncapped_cardinality_even_when_members_are_capped(self):
        catalogue = MemberCatalogue(_wide_frame(), ["narrow", "wide"], max_indexed=50)
        self.assertEqual(catalogue.count("wide"), 10_000)
        self.assertFalse(catalogue.is_indexed("wide"))
        # The page reports the whole, not the slice it handed back.
        page = catalogue.members("wide", limit=10)
        self.assertEqual(len(page.members), 10)
        self.assertEqual(page.total, 10_000)

    def test_a_dimension_the_frame_does_not_hold_is_absent_rather_than_empty(self):
        catalogue = MemberCatalogue(self.df, list(self.schema.dimensions) + ["not_a_column"])
        self.assertNotIn("not_a_column", catalogue.dimensions())
        with self.assertRaises(UnknownDimensionError) as ctx:
            catalogue.count("not_a_column")
        self.assertEqual(ctx.exception.to_payload()["error"], "unknown_dimension")
        self.assertTrue(ctx.exception.valid_alternatives)


class TestHighCardinalityIsPaginatedNotDumped(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalogue = MemberCatalogue(_wide_frame(), ["narrow", "wide"], max_indexed=50)

    def test_a_ten_thousand_member_dimension_returns_one_page_not_ten_thousand_names(self):
        page = self.catalogue.members("wide")
        self.assertEqual(len(page.members), DEFAULT_PAGE_SIZE)
        self.assertEqual(page.total, 10_000)
        self.assertFalse(page.indexed)

    def test_paging_through_a_large_dimension_visits_every_member_exactly_once(self):
        seen, offset = [], 0
        while True:
            page = self.catalogue.members("wide", offset=offset, limit=500)
            if not page.members:
                break
            seen.extend(page.members)
            offset += 500
        self.assertEqual(len(seen), 10_000)
        self.assertEqual(len(set(seen)), 10_000)

    def test_a_page_past_the_end_is_empty_rather_than_an_error(self):
        page = self.catalogue.members("wide", offset=999_999)
        self.assertEqual(page.members, ())
        self.assertEqual(page.total, 10_000)

    def test_a_dimension_past_the_index_cap_is_not_offered_for_free_text_lookup(self):
        """Its values stay listable; they simply cannot be found by scanning."""
        self.assertIsNotNone(self.catalogue.resolve("wide", "entity 00007"))
        self.assertEqual(self.catalogue.lookup("Entity 00007"), ())
        # The narrow column beside it still binds.
        self.assertTrue(self.catalogue.lookup("North"))


class TestLookupIgnoresCaseAndSpacing(EngineTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.catalogue = MemberCatalogue(cls.df, cls.schema.dimensions)

    def test_north_and_upper_north_and_padded_north_all_find_the_same_member(self):
        for spelling in ("North", "north", "NORTH", "  North  "):
            matches = self.catalogue.lookup(spelling)
            self.assertEqual([m.member for m in matches], ["North"], spelling)
            self.assertEqual(matches[0].dimension, "region")

    def test_a_multi_word_member_is_found_from_its_spaced_form(self):
        matches = self.catalogue.lookup("product a")
        self.assertEqual([(m.dimension, m.member) for m in matches], [("product", "Product A")])

    def test_a_value_present_in_two_dimensions_returns_both_rather_than_choosing(self):
        frame = pd.DataFrame({"alpha": ["Shared", "Alpha only"],
                              "beta": ["Shared", "Beta only"]})
        catalogue = MemberCatalogue(frame, ["alpha", "beta"])
        matches = catalogue.lookup("shared")
        self.assertEqual(sorted(m.dimension for m in matches), ["alpha", "beta"])

    def test_an_unknown_value_returns_no_matches_rather_than_raising(self):
        self.assertEqual(self.catalogue.lookup("Atlantis"), ())
        self.assertEqual(self.catalogue.lookup(""), ())
        self.assertIsNone(self.catalogue.resolve("region", "Atlantis"))


class TestMembersThatMustNeverBeIndexed(unittest.TestCase):
    """
    Both exclusions are language-neutral, which is why they live in the agent
    layer at all. A numeric value collides with every year and quantity a
    question contains; a two-character value collides with everything.
    """

    def setUp(self):
        frame = pd.DataFrame({
            "coded": ["2023", "2024", "7"],
            "short": ["AB", "CD", "Northwest"],
        })
        self.catalogue = MemberCatalogue(frame, ["coded", "short"])

    def test_a_purely_numeric_member_is_never_offered_as_a_filter(self):
        self.assertEqual(self.catalogue.lookup("2023"), ())
        # Still a real member: listable, and resolvable when named outright.
        self.assertIn("2023", self.catalogue.members("coded").members)
        self.assertEqual(self.catalogue.resolve("coded", "2023"), "2023")

    def test_a_two_character_member_is_not_indexed(self):
        self.assertEqual(self.catalogue.lookup("AB"), ())
        self.assertTrue(self.catalogue.lookup("Northwest"))


class TestNothingIsComputedUntilItIsAskedFor(unittest.TestCase):
    """
    The catalogue is built on the dataset load path. `detect_schema` admits a
    column as a dimension at up to 20% of row count, so eager `nunique()` over
    every dimension of a large frame would be paid on every cold load.
    """

    def test_constructing_the_catalogue_touches_no_column(self):
        touched = []

        class Watched(pd.DataFrame):
            @property
            def _constructor(self):
                return Watched

            def __getitem__(self, key):
                touched.append(key)
                return super().__getitem__(key)

        frame = Watched({"region": ["North", "South"], "product": ["A", "B"]})
        catalogue = MemberCatalogue(frame, ["region", "product"])
        self.assertEqual(touched, [], "construction read a column")

        catalogue.count("region")
        self.assertEqual(touched, ["region"], "asking about one column read another")


class TestNoProseCrossesTheBoundary(EngineTestCase):
    """
    Tools in the agent layer return numbers and facts; the model writes the
    sentences. A `note` or `detail` field here is how templates creep back.
    """

    ALLOWED_STRING_FIELDS = {"dimension", "member", "normalised"}

    def test_the_member_dataclasses_have_no_free_text_field(self):
        from dataclasses import fields as dataclass_fields

        catalogue = MemberCatalogue(self.df, self.schema.dimensions)
        for match in catalogue.lookup("North"):
            for f in dataclass_fields(match):
                value = getattr(match, f.name)
                if isinstance(value, str):
                    self.assertIn(f.name, self.ALLOWED_STRING_FIELDS)
                    self.assertLess(len(value), 64)

        page = catalogue.members("region")
        for f in dataclass_fields(page):
            value = getattr(page, f.name)
            if isinstance(value, str):
                self.assertIn(f.name, self.ALLOWED_STRING_FIELDS)


class TestNoDomainVocabularyInTheCatalogue(unittest.TestCase):
    """The same guard `test_formula_ast` applies to `formula.py`."""

    def test_no_business_vocabulary_appears_in_the_module(self):
        import re
        from pathlib import Path

        from .test_formula_ast import TestNoDomainVocabulary

        root = Path(__file__).resolve().parents[1]
        source = (root / "app/agent/dimensions.py").read_text(encoding="utf-8")
        code = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
        code = re.sub(r"#.*", "", code)
        found = sorted(t for t in TestNoDomainVocabulary._domain_terms()
                       if re.search(rf"\b{re.escape(t)}\b", code, re.IGNORECASE))
        self.assertEqual(found, [],
                         f"dimensions.py references business vocabulary {found}; "
                         "domain knowledge belongs in kpi/library.py")


class TestNormalisationAgreesWithGrounding(EngineTestCase):
    """
    The index is searched with n-grams `query.grounding` produces. If the two
    normalisers disagree on any input, a member is filed under a key no question
    can ever generate and the lookup silently never fires.
    """

    def test_the_two_normalisers_agree_on_every_member_and_a_corpus(self):
        from app.query.grounding import _normalise

        corpus = ["North", "Product A", "Grade_6", "North Campus", "  padded  ",
                  "Mixed-Case Value", "with/punct.uation", "2024", ""]
        for dimension in self.schema.dimensions:
            corpus.extend(self.df[dimension].astype(str).unique().tolist())
        for text in corpus:
            self.assertEqual(normalise(text), _normalise(text), text)


class TestTheCatalogueIsBuiltOnceAndNeverPersisted(EngineTestCase):
    """
    `to_dict` serialises `self.__dict__` minus an exclusion list, and its output
    is written into the dataset document and returned by the datasets API. A
    catalogue object reaching it is a `TypeError` on the JSON backend the whole
    suite runs on, and silent corruption of `meta["schema"]` elsewhere.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from app.services import dataset_service

        cls.loaded_df, cls.loaded_schema = dataset_service.load(cls.dataset, cls.uid)

    def test_a_dataset_load_attaches_a_catalogue(self):
        self.assertIsInstance(self.loaded_schema.member_catalogue, MemberCatalogue)
        self.assertIn("region", self.loaded_schema.member_catalogue.dimensions())

    def test_a_second_load_of_an_unchanged_dataset_reuses_the_same_catalogue(self):
        from app.services import dataset_service

        _, again = dataset_service.load(self.dataset, self.uid)
        self.assertIs(again.member_catalogue, self.loaded_schema.member_catalogue,
                      "an unchanged file rebuilt its catalogue")

    def test_the_catalogue_never_reaches_the_persisted_schema_document(self):
        import json

        payload = self.loaded_schema.to_dict()
        self.assertNotIn("member_catalogue", payload, sorted(payload))
        for key, value in payload.items():
            self.assertNotIsInstance(value, MemberCatalogue, key)
        json.dumps(payload, default=str)          # must not raise

    def test_replacing_the_schema_shares_the_catalogue_without_copying_it(self):
        """
        `dataset_service.load` calls `dataclasses.replace`, a shallow copy, so
        the catalogue is shared by reference. That is correct only while it is
        effectively immutable -- which is what the frozen dataclasses buy.
        """
        from dataclasses import replace

        copy = replace(self.loaded_schema)
        self.assertIs(copy.member_catalogue, self.loaded_schema.member_catalogue)

    def test_the_dimension_members_property_is_not_a_serialised_field(self):
        members = self.loaded_schema.dimension_members
        self.assertIn("region", members)
        self.assertIn("North", members["region"])
        self.assertNotIn("dimension_members", self.loaded_schema.__dict__)
        self.assertNotIn("dimension_members", self.loaded_schema.to_dict())

    def test_a_schema_with_no_catalogue_reports_no_members_rather_than_failing(self):
        from app.engines.metrics import DatasetSchema

        bare = DatasetSchema(date_column="date")
        self.assertEqual(bare.dimension_members, {})


if __name__ == "__main__":
    unittest.main()
