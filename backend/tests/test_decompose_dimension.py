"""
`decompose_by_dimension` / `decompose_rate_mix` / `decompose_nested` --
`agent/decompose.py`.

These wrap `observe.decompose_dimension` (`engines/observe.py:292`) rather
than re-deriving it -- that function is the exact code `test_rca_ground_truth.py`
checks against the retail sample's planted three-factor scenario, and
`TestWrapperReproducesTheLegacyFunction` is the guard that wrapping it changed
no arithmetic. The tests that matter otherwise are about the shape fix: the
templated `"Other (N members)"` row becomes a typed `others` sibling that is
never in `members`, and its roll-up is exact for the fields it can sum and
honestly `None` for the ones it cannot.
"""
from __future__ import annotations

import unittest

from app.agent.contract_api import ContractAPI
from app.agent.decompose import DecomposeEngine
from app.engines import observe

from .base import EngineTestCase, FIXTURES, assert_json_safe, assert_no_prose_leak, contracted


class DecomposeTestCase(EngineTestCase):
    """The shared retail fixture, with a compiled contract."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.rdf, cls.rschema = contracted("business_metrics_sample.csv")
        cls.api = ContractAPI(cls.rschema)
        cls.de = DecomposeEngine(cls.api)
        cls.period_a = {"type": "quarter", "year": 2026, "quarter": 2}
        cls.period_b = {"type": "quarter", "year": 2026, "quarter": 1}


# ---------------------------------------------------------------------------
# decompose_by_dimension()
# ---------------------------------------------------------------------------
class TestMemberContributionsSumToTheTotalChange(DecomposeTestCase):
    def test_every_member_accounted_for_sums_exactly(self):
        result = self.de.decompose_by_dimension(
            self.rdf, "revenue", "region", self.period_a, self.period_b, max_items=10)
        self.assertIsNone(result.others)   # region has 4 members, well under max_items
        total = sum(m.change_abs for m in result.members)
        self.assertAlmostEqual(total, result.change_abs, places=4)

    def test_with_a_rollup_the_others_change_abs_completes_the_sum(self):
        result = self.de.decompose_by_dimension(
            self.rdf, "revenue", "region", self.period_a, self.period_b, max_items=2)
        self.assertIsNotNone(result.others)
        total = sum(m.change_abs for m in result.members) + result.others.change_abs
        self.assertAlmostEqual(total, result.change_abs, places=4)


class TestWrapperReproducesTheLegacyFunction(DecomposeTestCase):
    """Applying `TimeFilter` + resolving the periods must produce the exact
    same two frames -- and the exact same rows -- `decompose_dimension` would
    over hand-sliced frames."""

    def test_field_for_field_against_the_hand_sliced_call(self):
        tf_a = observe.Timeframe(2026, 2)
        tf_b = observe.Timeframe(2026, 1)
        cur = observe.slice_period(self.rdf, tf_a)
        base = observe.slice_period(self.rdf, tf_b)
        expected = observe.decompose_dimension(cur, base, "region", "revenue", 10,
                                               self.rschema.contract_resolver)

        result = self.de.decompose_by_dimension(
            self.rdf, "revenue", "region", self.period_a, self.period_b, max_items=10)

        self.assertEqual(len(result.members), len(expected))
        for member, row in zip(result.members, expected):
            self.assertEqual(member.name, row["name"])
            self.assertEqual(member.current, row["current"])
            self.assertEqual(member.baseline, row["baseline"])
            self.assertEqual(member.change_abs, row["change_abs"])
            self.assertEqual(member.contribution_pct, row["contribution_pct"])
            self.assertEqual(member.over_index, row["over_index"])


class TestRateMixDecomposition(DecomposeTestCase):
    def test_rate_and_mix_effects_reconcile_to_the_ratio_change(self):
        result = self.de.decompose_rate_mix(
            self.rdf, "gross_margin_pct", "region", self.period_a, self.period_b)
        self.assertEqual(result.status, "ok")
        full = self.de.decompose_by_dimension(
            self.rdf, "gross_margin_pct", "region", self.period_a, self.period_b)
        self.assertAlmostEqual(result.total_rate_effect + result.total_mix_effect,
                               full.change_abs, places=4)

    def test_an_additive_kpi_has_no_mix_to_shift(self):
        result = self.de.decompose_rate_mix(
            self.rdf, "revenue", "region", self.period_a, self.period_b)
        self.assertEqual(result.status, "not_a_ratio")
        self.assertEqual(result.members, ())


class TestOthersRollupIsExactNotLossy(DecomposeTestCase):
    def test_the_four_summable_fields_are_exact_and_the_rest_are_none(self):
        result = self.de.decompose_by_dimension(
            self.rdf, "revenue", "region", self.period_a, self.period_b, max_items=1)
        self.assertIsNotNone(result.others)
        self.assertEqual(result.others.member_count, 3)

        full = self.de.decompose_by_dimension(
            self.rdf, "revenue", "region", self.period_a, self.period_b, max_items=10)
        tail_names = {m.name for m in full.members} - {result.members[0].name}
        tail = [m for m in full.members if m.name in tail_names]

        self.assertAlmostEqual(result.others.current, sum(m.current for m in tail), places=4)
        self.assertAlmostEqual(result.others.baseline, sum(m.baseline for m in tail), places=4)
        self.assertAlmostEqual(result.others.change_abs, sum(m.change_abs for m in tail), places=4)
        self.assertAlmostEqual(result.others.contribution_pct,
                               sum(m.contribution_pct for m in tail), places=4)

    def test_no_row_carries_a_templated_name(self):
        result = self.de.decompose_by_dimension(
            self.rdf, "revenue", "region", self.period_a, self.period_b, max_items=1)
        for member in result.members:
            self.assertNotIn("Other (", member.name)
        payload_text = str(result.to_payload())
        self.assertNotIn("Other (", payload_text)
        self.assertIsInstance(result.others.member_count, int)


# ---------------------------------------------------------------------------
# decompose_nested()
# ---------------------------------------------------------------------------
class TestNestedDrill(DecomposeTestCase):
    def test_inner_change_abs_sums_to_the_outer_members_own_change_abs(self):
        outer = self.de.decompose_by_dimension(
            self.rdf, "units_sold", "region", self.period_a, self.period_b, max_items=10)
        north = next(m for m in outer.members if m.name == "North")

        nested = self.de.decompose_nested(
            self.rdf, "units_sold", "region", "North", "product",
            self.period_a, self.period_b, max_items=10)

        self.assertAlmostEqual(sum(m.change_abs for m in nested.members), north.change_abs, places=4)

    def test_the_drill_into_north_surfaces_the_planted_product_a_locus(self):
        """Ground truth: `supply_disruption_product_a` is planted specifically
        inside the North revenue decline's neighbourhood -- Product A should
        be the dominant contributor once drilled into."""
        nested = self.de.decompose_nested(
            self.rdf, "units_sold", "region", "North", "product",
            self.period_a, self.period_b, max_items=10)
        top = max(nested.members, key=lambda m: abs(m.change_abs or 0))
        self.assertEqual(top.name, "Product A")


# ---------------------------------------------------------------------------
# integrity / prose guard / determinism / portability
# ---------------------------------------------------------------------------
class TestNumericIntegrityAndProseGuard(DecomposeTestCase):
    def test_the_payload_survives_json_dumps_with_no_nan_or_inf(self):
        result = self.de.decompose_by_dimension(
            self.rdf, "gross_margin_pct", "region", self.period_a, self.period_b)
        assert_json_safe(result.to_payload())

    def test_no_string_field_outside_the_allowlist_looks_like_prose(self):
        result = self.de.decompose_by_dimension(
            self.rdf, "revenue", "region", self.period_a, self.period_b, max_items=1)
        assert_no_prose_leak(result.to_payload())


class TestDeterminism(DecomposeTestCase):
    def test_identical_input_gives_identical_output_across_runs(self):
        first = self.de.decompose_by_dimension(
            self.rdf, "revenue", "region", self.period_a, self.period_b).to_payload()
        second = self.de.decompose_by_dimension(
            self.rdf, "revenue", "region", self.period_a, self.period_b).to_payload()
        self.assertEqual(first, second)


class TestPortabilityAcrossDatasets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.schema = contracted("hospital_sample.csv", FIXTURES)
        assert cls.schema.dimensions == ["department"]
        cls.de = DecomposeEngine(ContractAPI(cls.schema))
        assert "recovery_rate" in cls.schema.available_kpis

    def test_decompose_works_on_a_single_dimension_and_a_contract_only_ratio_kpi(self):
        result = self.de.decompose_by_dimension(
            self.df, "recovery_rate", "department",
            {"type": "quarter", "year": 2025, "quarter": 2},
            {"type": "quarter", "year": 2025, "quarter": 1})
        self.assertTrue(result.members)

        rm = self.de.decompose_rate_mix(
            self.df, "recovery_rate", "department",
            {"type": "quarter", "year": 2025, "quarter": 2},
            {"type": "quarter", "year": 2025, "quarter": 1})
        self.assertEqual(rm.status, "ok")


if __name__ == "__main__":
    unittest.main()
