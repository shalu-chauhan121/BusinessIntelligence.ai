"""
KPI relationships derived from the formula AST rather than from field overlap.

The headline test is a direct comparison: on one fixture, `build_driver_graph`
and `RelationGraph.components` are asked the same question and the legacy answer
is asserted to be *incomplete*. That framing matters -- it proves the AST path is
strictly better rather than merely different, and it will fail loudly if someone
later "fixes" containment and makes this group redundant.

The rest of the suite runs the same assertions in retail and clinical
vocabulary, and confirms that the non-formula relations -- shared source fields,
semantic tags, recorded derivations -- still agree with `driver_graph`, since
those are reused from it rather than reimplemented.
"""
from __future__ import annotations

import unittest
from dataclasses import fields as dataclass_fields

from app.agent.contract_api import ContractAPI
from app.agent.relations import Relation, RelationGraph
from app.engines.driver_graph import build_driver_graph
from app.engines.metrics import detect_schema
from app.kpi.resolver import compile_contract

from .fixtures.shape_contract import (
    cyclic_contract,
    dual_role_contract,
    nested_contract,
    nested_frame,
)
from .test_formula_ast import HospitalContractTestCase
from .base import EngineTestCase


def schema_for(contract, df):
    resolver = compile_contract(contract)
    schema = detect_schema(df)
    schema.contract_resolver = resolver
    schema.available_kpis = list(resolver)
    return schema


# ---------------------------------------------------------------------------
# the headline: what containment cannot reach
# ---------------------------------------------------------------------------
class TestAstBeatsContainment(unittest.TestCase):
    """
    `chi = {gamma}/{alpha}`, `gamma = {alpha}-{beta}`.

    `chi`'s referenced fields are `{gamma, alpha}`. `beta` is not among them, and
    neither is `gamma` a subset of them -- `gamma`'s own fields are
    `{alpha, beta}`. So containment reaches neither, while the AST reaches both.
    """

    @classmethod
    def setUpClass(cls):
        cls.df = nested_frame()
        cls.schema = schema_for(nested_contract(), cls.df)
        cls.graph = RelationGraph(ContractAPI(cls.schema))
        cls.found = {r.source_kpi for r in cls.graph.components("chi")}
        cls.legacy = {e.source_kpi for e in build_driver_graph(cls.schema, "chi").edges}

    def test_containment_misses_the_transitive_component(self):
        """The red-first baseline. If this ever passes, the diagnosis was wrong."""
        self.assertNotIn("beta", self.legacy)

    def test_containment_even_misses_the_direct_numerator(self):
        self.assertNotIn("gamma", self.legacy)

    def test_the_ast_finds_the_transitive_component(self):
        self.assertIn("beta", self.found)

    def test_the_ast_finds_the_direct_numerator(self):
        self.assertIn("gamma", self.found)

    def test_the_ast_is_a_strict_superset_of_containment(self):
        self.assertTrue(self.legacy < self.found,
                        f"legacy {self.legacy} must be a strict subset of {self.found}")

    def test_the_transitive_component_carries_its_sign(self):
        """`beta` is subtracted inside `gamma`, so it moves `chi` the other way.
        Containment has no sign at all, which is how a driver ends up quoted
        with the direction of its effect reversed."""
        beta = next(r for r in self.graph.components("chi") if r.source_kpi == "beta")
        self.assertEqual(beta.sign, -1)
        self.assertEqual(beta.depth, 1)

    def test_the_direct_numerator_is_labelled_as_the_numerator(self):
        gamma = next(r for r in self.graph.components("chi") if r.source_kpi == "gamma")
        self.assertEqual(gamma.relation, "formula_numerator")
        self.assertEqual(gamma.depth, 0)

    def test_a_deeper_component_weighs_less_than_a_direct_one(self):
        by_key = {r.source_kpi: r for r in self.graph.components("chi")}
        self.assertLess(by_key["beta"].weight, by_key["gamma"].weight)


class TestDualRole(unittest.TestCase):
    """A field in both halves of a ratio, with no business vocabulary attached."""

    @classmethod
    def setUpClass(cls):
        cls.schema = schema_for(dual_role_contract(), nested_frame())
        cls.graph = RelationGraph(ContractAPI(cls.schema))

    def test_both_positions_are_reported(self):
        relations = {r.relation for r in self.graph.components("delta")
                     if r.source_kpi == "alpha"}
        self.assertEqual(relations, {"formula_numerator", "formula_denominator"})

    def test_containment_reports_only_one_of_them(self):
        legacy = {e.relation for e in build_driver_graph(self.schema, "delta").edges
                  if e.source_kpi == "alpha"}
        self.assertEqual(legacy, {"formula_numerator"},
                         "the baseline this group exists to improve on")

    def test_the_subtracted_term_is_negative(self):
        beta = next(r for r in self.graph.components("delta") if r.source_kpi == "beta")
        self.assertEqual(beta.sign, -1)


class TestCycleTermination(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.graph = RelationGraph(ContractAPI(
            schema_for(cyclic_contract(), nested_frame())))

    def test_traversal_of_a_cyclic_contract_terminates(self):
        self.assertEqual(self.graph.neighbours("ping", depth=5), ["pong"])

    def test_depth_limiting_terminates(self):
        for depth in (1, 2, 3):
            self.assertIsInstance(self.graph.related("ping", depth=depth), list)


# ---------------------------------------------------------------------------
# domain one: retail
# ---------------------------------------------------------------------------
class TestRetailRelations(EngineTestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.graph = RelationGraph(ContractAPI(cls.schema))
        cls.components = cls.graph.components("gross_margin_pct")

    def test_a_field_in_both_halves_yields_two_relations(self):
        relations = {r.relation for r in self.components if r.source_kpi == "revenue"}
        self.assertEqual(relations, {"formula_numerator", "formula_denominator"})

    def test_the_subtracted_component_carries_a_negative_sign(self):
        cogs = [r for r in self.components if r.source_kpi == "cost_of_goods"]
        self.assertTrue(cogs)
        self.assertEqual({r.sign for r in cogs}, {-1})

    def test_a_kpi_equal_to_the_numerator_is_found_without_being_named(self):
        """`gross_profit` is `revenue - cost_of_goods`, which is exactly this
        ratio's numerator -- but it is never named in the formula. The
        equivalence rule finds it by matching signed terms."""
        gross_profit = [r for r in self.components if r.source_kpi == "gross_profit"]
        self.assertTrue(gross_profit)
        self.assertEqual(gross_profit[0].relation, "formula_numerator")

    def test_origin_reports_the_seed_path(self):
        self.assertEqual({r.origin for r in self.components}, {"seed"})


# ---------------------------------------------------------------------------
# domain two: healthcare
# ---------------------------------------------------------------------------
class TestHealthcareRelations(HospitalContractTestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.graph = RelationGraph(cls.api)

    def test_ratio_halves_are_labelled(self):
        """The same assertion as the retail block, sharing no vocabulary."""
        by_key = {r.source_kpi: r for r in self.graph.components("readmission_rate")}
        self.assertEqual(by_key["readmissions"].relation, "formula_numerator")
        self.assertEqual(by_key["discharges"].relation, "formula_denominator")

    def test_origin_reports_the_contract_path(self):
        relations = self.graph.components("readmission_rate")
        self.assertEqual({r.origin for r in relations}, {"contract"})

    def test_semantic_tag_relations_still_agree_with_driver_graph(self):
        """Non-formula edges are reused from `driver_graph`, not reimplemented,
        so they must not drift."""
        mine = {r.source_kpi for r in self.graph.related("readmission_rate")
                if r.relation == "semantic_tag"}
        legacy = {e.source_kpi for e in
                  build_driver_graph(self.h_schema, "readmission_rate").edges
                  if e.relation == "semantic_tag"}
        self.assertEqual(mine, legacy)

    def test_an_unrelated_kpi_gets_no_field_relation(self):
        """`mortality_rate` shares no source field with `readmission_rate`, so
        it must relate only by business theme, never by field overlap."""
        relations = {r.relation for r in self.graph.related("readmission_rate")
                     if r.source_kpi == "mortality_rate"}
        self.assertNotIn("shared_source_field", relations)

    def test_every_available_kpi_can_be_traversed(self):
        for key in self.h_schema.available_kpis:
            self.assertIsInstance(self.graph.related(key), list)


# ---------------------------------------------------------------------------
# the prose-leak guard (G1, applied where the field would be introduced)
# ---------------------------------------------------------------------------
class TestNoProseCrossesTheBoundary(unittest.TestCase):
    """
    `DriverEdge` carries `note` and `verification_note`, assembled from
    f-strings (`driver_graph.py:166`, `:410`) and read back out by the narrative
    layer. The agent tool boundary returns numbers and facts; the model writes
    the sentences. The dataclass simply has nowhere to put a sentence, which is
    what keeps templates from creeping back in.
    """

    ALLOWED_STRING_FIELDS = {"source_kpi", "target_kpi", "relation", "origin"}
    MAX_LENGTH = 48

    def test_the_dataclass_has_no_free_text_field(self):
        names = {f.name for f in dataclass_fields(Relation)}
        self.assertNotIn("note", names)
        self.assertNotIn("verification_note", names)

    def test_no_returned_string_is_long_enough_to_be_a_sentence(self):
        graph = RelationGraph(ContractAPI(
            schema_for(nested_contract(), nested_frame())))
        for key in ("chi", "gamma", "alpha"):
            for relation in graph.related(key, depth=2):
                for field in dataclass_fields(relation):
                    value = getattr(relation, field.name)
                    if isinstance(value, str):
                        self.assertIn(field.name, self.ALLOWED_STRING_FIELDS)
                        self.assertLess(len(value), self.MAX_LENGTH)
                        self.assertNotIn(" ", value, "a space means a sentence")


class TestNoDomainVocabularyInRelations(unittest.TestCase):
    """The same guard `test_formula_ast` applies to `formula.py`."""

    def test_no_business_vocabulary_appears_in_the_module(self):
        import re
        from pathlib import Path

        from .test_formula_ast import TestNoDomainVocabulary

        root = Path(__file__).resolve().parents[1]
        source = (root / "app/agent/relations.py").read_text(encoding="utf-8")
        code = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
        code = re.sub(r"#.*", "", code)
        found = sorted(t for t in TestNoDomainVocabulary._domain_terms()
                       if re.search(rf"\b{re.escape(t)}\b", code, re.IGNORECASE))
        self.assertEqual(found, [],
                         f"relations.py references business vocabulary {found}; "
                         "domain knowledge belongs in kpi/library.py")


if __name__ == "__main__":
    unittest.main()
