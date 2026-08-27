"""
Reading a KPI's formula structurally, across two unrelated business domains.

The suite is deliberately written twice over: once in retail vocabulary against
the seed registry, once in clinical vocabulary against a discovered contract.
The *same* assertions appear in both, because that is what "domain agnostic"
actually means here -- one implementation, two vocabularies with nothing in
common, identical structural results. A third block covers formula shapes that
no discovered contract produces (nested, cyclic, malformed), which is the only
reason a hand-built fixture is needed at all.

What is being proved is that sign, position and depth survive, since those are
exactly the three things `driver_graph.py:162-179`'s field-set containment
discards.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from app.agent import formula as F
from app.agent.contract_api import ContractAPI
from app.agent.errors import MalformedFormulaError
from app.engines.metrics import detect_schema, prepare
from app.kpi import service as kpi_service
from app.kpi.resolver import compile_contract

from .base import EngineTestCase
from .fixtures.shape_contract import (
    cyclic_contract,
    dual_role_contract,
    malformed_contract,
    nested_contract,
    nested_frame,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def api_for(contract, df) -> ContractAPI:
    """A ContractAPI over an in-memory contract, the way `dataset_service` builds one."""
    resolver = compile_contract(contract)
    schema = detect_schema(df)
    schema.contract_resolver = resolver
    schema.available_kpis = list(resolver)
    return ContractAPI(schema)


# ---------------------------------------------------------------------------
# domain one: retail, through the seed registry (no contract)
# ---------------------------------------------------------------------------
class TestRetailSeedFormulas(EngineTestCase):
    """`gross_margin_pct` is `(revenue - cost_of_goods) / revenue` -- one KPI
    carrying a subtractive term and a field in both halves of a ratio."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.api = ContractAPI(cls.schema)      # no contract attached: seed path

    def test_the_seed_path_is_reported_as_seed(self):
        struct = F.structure(self.api, "gross_margin_pct")
        self.assertEqual(struct.source, "seed")

    def test_an_additive_formula_yields_signed_terms(self):
        struct = F.structure(self.api, "gross_profit")
        signs = {t.field: t.sign for t in struct.terms}
        self.assertEqual(signs["revenue"], 1)
        self.assertEqual(signs["cost_of_goods"], -1,
                         "a subtracted term must carry a negative sign")

    def test_a_field_in_both_halves_reports_both_roles(self):
        """The assertion set containment cannot make: `driver_graph.py:162`
        tests the numerator first and `continue`s, losing the denominator."""
        struct = F.structure(self.api, "gross_margin_pct")
        self.assertEqual(F.roles_of(struct, "revenue"),
                         {F.NUMERATOR, F.DENOMINATOR})

    def test_a_subtractive_term_is_never_reported_as_additive(self):
        struct = F.structure(self.api, "gross_margin_pct")
        self.assertEqual(F.signs_of(struct, "cost_of_goods"), {-1})

    def test_ratio_halves_are_separated(self):
        struct = F.structure(self.api, "avg_order_value")
        self.assertEqual(F.roles_of(struct, "revenue"), {F.NUMERATOR})
        self.assertEqual(F.roles_of(struct, "orders"), {F.DENOMINATOR})

    def test_scale_is_carried(self):
        self.assertEqual(F.structure(self.api, "gross_margin_pct").scale, 100.0)

    def test_every_available_kpi_resolves(self):
        for key in self.schema.available_kpis:
            struct = F.structure(self.api, key)
            self.assertTrue(struct.terms, f"{key} produced no terms")


# ---------------------------------------------------------------------------
# domain two: healthcare, through a discovered contract
# ---------------------------------------------------------------------------
class HospitalContractTestCase(EngineTestCase):
    """A dataset whose KPIs are entirely absent from the retail seed registry."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        raw = pd.read_csv(FIXTURES / "hospital_sample.csv")
        cls.h_schema = detect_schema(raw)
        cls.h_df = prepare(raw, cls.h_schema)
        contract = kpi_service.bootstrap(
            "u", {"_id": "hosp1", "filename": "hospital.csv"}, cls.h_df, cls.h_schema)
        cls.h_contract = contract
        resolver = compile_contract(contract)
        cls.h_schema.contract_resolver = resolver
        cls.h_schema.available_kpis = [
            k for k, c in resolver.items()
            if set(c.source_fields) <= set(cls.h_df.columns)]
        cls.api = ContractAPI(cls.h_schema)
        # If the fixture stopped producing clinical ratios this suite proves nothing.
        assert "readmission_rate" in cls.h_schema.available_kpis


class TestHealthcareContractFormulas(HospitalContractTestCase):

    def test_the_contract_path_is_reported_as_contract(self):
        struct = F.structure(self.api, "readmission_rate")
        self.assertEqual(struct.source, "contract")

    def test_ratio_halves_are_separated(self):
        """The same assertion as the retail block, in a vocabulary that shares
        no term with it."""
        struct = F.structure(self.api, "readmission_rate")
        self.assertEqual(F.roles_of(struct, "readmissions"), {F.NUMERATOR})
        self.assertEqual(F.roles_of(struct, "discharges"), {F.DENOMINATOR})

    def test_a_mean_kpi_resolves_without_ratio_handling(self):
        struct = F.structure(self.api, "avg_length_of_stay")
        self.assertEqual(struct.kind, "mean")
        self.assertEqual(len(struct.terms), 1)
        self.assertEqual(struct.terms[0].sign, 1)

    def test_a_sum_kpi_resolves(self):
        struct = F.structure(self.api, "admissions")
        self.assertEqual(struct.kind, "sum")
        self.assertEqual([t.field for t in struct.terms], ["admissions"])

    def test_a_contract_kpi_shadowing_a_seed_key_uses_the_contract_formula(self):
        """`cost_of_goods` is a seed key; here the contract binds it to
        `treatment_cost`. The contract must win."""
        struct = F.structure(self.api, "cost_of_goods")
        self.assertEqual(struct.source, "contract")
        self.assertEqual([t.field for t in struct.terms], ["treatment_cost"])

    def test_every_available_kpi_resolves(self):
        for key in self.h_schema.available_kpis:
            struct = F.structure(self.api, key)
            self.assertTrue(struct.terms, f"{key} produced no terms")

    def test_a_clean_contract_has_nothing_unparsable(self):
        self.assertEqual(F.unparsable_kpis(self.h_contract), [])


# ---------------------------------------------------------------------------
# shapes no discovered contract produces
# ---------------------------------------------------------------------------
class TestNestedFormulas(unittest.TestCase):
    """`chi = {gamma}/{alpha}` where `gamma = {alpha}-{beta}`.

    `chi`'s own referenced fields are `{gamma, alpha}`, so `beta` is not in them
    at any remove -- the transitive case containment cannot reach.
    """

    @classmethod
    def setUpClass(cls):
        cls.df = nested_frame()
        cls.api = api_for(nested_contract(), cls.df)

    def test_a_component_one_level_down_is_reached(self):
        struct = F.structure(self.api, "chi")
        self.assertIn("beta", [t.field for t in struct.terms])

    def test_that_component_carries_its_sign_and_depth(self):
        struct = F.structure(self.api, "chi")
        beta = next(t for t in struct.terms if t.field == "beta")
        self.assertEqual(beta.sign, -1)
        self.assertEqual(beta.depth, 1)
        self.assertEqual(beta.via, ("gamma",))

    def test_expansion_preserves_the_outer_position(self):
        """`alpha` is `chi`'s denominator directly, and its numerator through
        `gamma`. Both must survive."""
        struct = F.structure(self.api, "chi")
        self.assertEqual(F.roles_of(struct, "alpha"), {F.NUMERATOR, F.DENOMINATOR})

    def test_component_kpis_records_the_expanded_away_key(self):
        struct = F.structure(self.api, "chi")
        self.assertIn("gamma", struct.component_kpis)

    def test_expansion_can_be_switched_off(self):
        struct = F.structure(self.api, "chi", expand=False)
        self.assertNotIn("beta", [t.field for t in struct.terms])
        self.assertIn("gamma", [t.field for t in struct.terms])

    def test_depth_limit_truncates_rather_than_recursing(self):
        struct = F.structure(self.api, "chi", max_depth=0)
        self.assertTrue(struct.truncated)


class TestDualRoleFormula(unittest.TestCase):
    """The retail dual-role case with the business vocabulary stripped off."""

    @classmethod
    def setUpClass(cls):
        cls.api = api_for(dual_role_contract(), nested_frame())

    def test_both_positions_are_reported(self):
        struct = F.structure(self.api, "delta")
        self.assertEqual(F.roles_of(struct, "alpha"), {F.NUMERATOR, F.DENOMINATOR})

    def test_the_subtracted_term_is_negative(self):
        struct = F.structure(self.api, "delta")
        self.assertEqual(F.signs_of(struct, "beta"), {-1})


class TestCyclicFormula(unittest.TestCase):
    """Two KPIs referencing each other. Nothing legitimate produces this; the
    requirement is simply that traversal ends."""

    @classmethod
    def setUpClass(cls):
        cls.api = api_for(cyclic_contract(), nested_frame())

    def test_a_cycle_terminates_and_is_flagged(self):
        struct = F.structure(self.api, "ping")
        self.assertTrue(struct.truncated,
                        "a cycle must be reported, not silently absorbed")

    def test_a_cycle_still_records_its_component(self):
        struct = F.structure(self.api, "ping")
        self.assertIn("pong", struct.component_kpis)


class TestMalformedFormulas(unittest.TestCase):
    """`compile_contract` (`resolver.py:397-400`) drops an unparsable KPI
    silently, so it is simply missing and nothing says why."""

    def test_compilation_still_drops_them(self):
        self.assertEqual(compile_contract(malformed_contract()), {})

    def test_unparsable_kpis_names_every_one_with_a_reason(self):
        reported = dict(F.unparsable_kpis(malformed_contract()))
        self.assertEqual(set(reported), {"sound", "broken"})
        for key, reason in reported.items():
            self.assertTrue(reason.strip(), f"{key} was reported with no reason")

    def test_introspecting_a_bad_expression_raises_a_recoverable_error(self):
        with self.assertRaises(MalformedFormulaError) as caught:
            F._parse("sound", "{alpha} +")
        payload = caught.exception.to_payload()
        self.assertEqual(payload["error"], "malformed_formula")
        self.assertEqual(payload["requested"], "sound")
        self.assertIn("reason", payload)


# ---------------------------------------------------------------------------
# the guard that keeps the module domain-free
# ---------------------------------------------------------------------------
class TestNoDomainVocabulary(unittest.TestCase):
    """
    `formula.py` must never learn what a field *means*.

    Domain knowledge belongs to `kpi/library.py`'s six packs and runs at
    discovery time. If a business term ever appears in the introspection layer,
    some dataset outside that vocabulary is about to get a worse answer -- so
    this fails the build instead.
    """

    MODULES = ("app/agent/formula.py",)

    @staticmethod
    def _domain_terms():
        """
        Whole KPI ids from every library pack, plus the concept field names they
        bind to. Whole ids rather than word fragments: splitting `return_rate`
        would flag the keyword `return`, and `avg_order_value` the word `value`,
        which says nothing about domain leakage.
        """
        from app.kpi.library import ALL_LIBRARY_KPIS
        terms = set()
        for kpi in ALL_LIBRARY_KPIS:
            terms.add(kpi.id.lower())
            for concept in kpi.concepts:
                name = getattr(concept, "concept", None) or getattr(concept, "name", "")
                if name:
                    terms.add(str(name).lower())
        return {t for t in terms if len(t) > 3}

    def test_no_business_vocabulary_appears_in_the_module(self):
        import re
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        terms = self._domain_terms()
        for relative in self.MODULES:
            source = (root / relative).read_text(encoding="utf-8")
            # Docstrings cite real KPI names to explain the bug being fixed;
            # what must stay clean is the executable code.
            code = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
            code = re.sub(r"#.*", "", code)
            found = sorted(t for t in terms
                           if re.search(rf"\b{re.escape(t)}\b", code, re.IGNORECASE))
            self.assertEqual(found, [],
                             f"{relative} references business vocabulary {found}; "
                             "domain knowledge belongs in kpi/library.py")


if __name__ == "__main__":
    unittest.main()
