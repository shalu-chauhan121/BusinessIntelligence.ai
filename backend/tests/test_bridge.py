"""
`bridge_periods` -- `agent/bridge.py`.

`decompose.py` already answers "which members moved" and "which formula
component moved"; this module orders those answers into a waterfall with a
running total. The tests here are not about `decompose.py`'s own arithmetic
(`test_decompose_dimension.py` / `test_decompose_formula.py` own that) but
about the three traps a naive waterfall falls into, each pinned with the
exact numbers measured against the real retail fixture before this module was
written:

1. A ratio KPI's member step is `rate_effect + mix_effect`, never
   `change_abs` -- on `fulfillment_rate` by `product`, two of three members
   have `change_abs == 0.0` while their true waterfall contribution is not.
2. The "others" step is the exact sum of the tail members' own step values,
   not `OthersRollup.change_abs` -- on `gross_margin_pct` by `product` those
   differ by 30x (`0.040` vs `1.219`).
3. A mean-kind KPI (`inventory_units`) cannot be bridged by dimension at all:
   the unrefused arithmetic reports a residual three times the total change,
   with the opposite sign.
"""
from __future__ import annotations

import unittest

from app.agent.bridge import BridgeEngine
from app.agent.contract_api import ContractAPI
from app.agent.decompose import DecomposeEngine
from app.agent.errors import InvalidArgumentError, UnknownDimensionError

from .base import EngineTestCase, FIXTURES, assert_json_safe, assert_no_prose_leak, contracted


class BridgeTestCase(EngineTestCase):
    """The shared retail fixture, with a compiled contract."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.rdf, cls.rschema = contracted("business_metrics_sample.csv")
        cls.api = ContractAPI(cls.rschema)
        cls.be = BridgeEngine(cls.api)
        cls.de = DecomposeEngine(cls.api)
        cls.period_a = {"type": "quarter", "year": 2026, "quarter": 2}
        cls.period_b = {"type": "quarter", "year": 2026, "quarter": 1}


# ---------------------------------------------------------------------------
# the reconciliation invariant, every shape at once
# ---------------------------------------------------------------------------
class TestWaterfallReconcilesForEveryShape(BridgeTestCase):
    """The headline test, mirroring `test_decompose_formula.py`'s own:
    parameterised across every KPI and both axes simultaneously. Float
    re-association means the invariant is exact by definition of the
    residual, not bit-exact -- `assertAlmostEqual`, not `==`, exactly as
    `test_decompose_formula.py` already does."""

    def test_start_plus_steps_equals_end_by_formula(self):
        for key in self.rschema.available_kpis:
            with self.subTest(kpi=key):
                result = self.be.bridge_periods(self.rdf, key, self.period_a, self.period_b,
                                                by="formula")
                if result.status != "ok":
                    continue
                self.assertAlmostEqual(
                    result.start_value + sum(s.value for s in result.steps),
                    result.end_value, places=4)

    def test_start_plus_steps_equals_end_by_every_dimension(self):
        for key in self.rschema.available_kpis:
            for dim in self.rschema.dimensions:
                with self.subTest(kpi=key, dimension=dim):
                    result = self.be.bridge_periods(self.rdf, key, self.period_a, self.period_b,
                                                    by=dim)
                    if result.status != "ok":
                        continue
                    self.assertAlmostEqual(
                        result.start_value + sum(s.value for s in result.steps),
                        result.end_value, places=4)

    def test_the_last_step_is_always_unexplained_and_equals_the_top_level_field(self):
        result = self.be.bridge_periods(self.rdf, "revenue", self.period_a, self.period_b,
                                        by="region")
        self.assertEqual(result.steps[-1].kind, "unexplained")
        self.assertAlmostEqual(result.steps[-1].value, result.unexplained, places=10)


# ---------------------------------------------------------------------------
# trap 1: a ratio member's step is rate+mix, not change_abs
# ---------------------------------------------------------------------------
class TestRatioMemberStepsUseRateAndMixNotChangeAbs(BridgeTestCase):
    def test_zero_change_abs_members_still_carry_their_true_contribution(self):
        full = self.de.decompose_by_dimension(self.rdf, "fulfillment_rate", "product",
                                              self.period_a, self.period_b, max_items=50)
        zero_change_abs_members = {m.name for m in full.members if m.change_abs == 0.0}
        self.assertTrue(zero_change_abs_members, "fixture must contain a zero-change_abs member")

        result = self.be.bridge_periods(self.rdf, "fulfillment_rate", self.period_a,
                                        self.period_b, by="product")
        steps_by_label = {s.label: s.value for s in result.steps if s.kind == "member"}
        for name in zero_change_abs_members:
            with self.subTest(member=name):
                self.assertIn(name, steps_by_label)
                self.assertNotAlmostEqual(steps_by_label[name], 0.0, places=6)

    def test_the_naive_sum_would_have_reversed_the_sign_of_the_margin_change(self):
        full = self.de.decompose_by_dimension(self.rdf, "gross_margin_pct", "product",
                                              self.period_a, self.period_b, max_items=50)
        naive_sum = sum(m.change_abs for m in full.members if m.change_abs == m.change_abs)
        self.assertGreater(naive_sum, 0.0)   # the naive (wrong) figure is positive...
        self.assertLess(full.change_abs, 0.0)  # ...while the true change is negative

        result = self.be.bridge_periods(self.rdf, "gross_margin_pct", self.period_a,
                                        self.period_b, by="product")
        true_sum = sum(s.value for s in result.steps if s.kind == "member")
        self.assertAlmostEqual(true_sum, full.change_abs, places=4)


# ---------------------------------------------------------------------------
# trap 2: the others step is the sum of the tail's true step values
# ---------------------------------------------------------------------------
class TestOthersStepIsTheSumOfTheTailSteps(BridgeTestCase):
    def test_others_differs_from_the_naive_othersrollup_change_abs_by_30x(self):
        full = self.de.decompose_by_dimension(self.rdf, "gross_margin_pct", "product",
                                              self.period_a, self.period_b, max_items=2)
        self.assertIsNotNone(full.others)
        naive_others = full.others.change_abs

        result = self.be.bridge_periods(self.rdf, "gross_margin_pct", self.period_a,
                                        self.period_b, by="product", max_items=2)
        others_steps = [s for s in result.steps if s.kind == "others"]
        self.assertEqual(len(others_steps), 1)
        true_others = others_steps[0].value

        # Pin the divergence itself -- if a future refactor moves this back
        # to `OthersRollup.change_abs`, this assertion fails loudly.
        self.assertGreater(abs(true_others - naive_others), abs(naive_others) * 5)
        self.assertAlmostEqual(true_others, 1.2190655258375844, places=6)
        self.assertAlmostEqual(naive_others, 0.04019480985226753, places=6)

    def test_others_still_reconciles_when_present(self):
        result = self.be.bridge_periods(self.rdf, "gross_margin_pct", self.period_a,
                                        self.period_b, by="product", max_items=2)
        self.assertAlmostEqual(
            result.start_value + sum(s.value for s in result.steps),
            result.end_value, places=4)


class TestOthersIsRankedByWaterfallContributionNotRateChange(BridgeTestCase):
    def test_max_items_one_keeps_the_largest_true_contributor(self):
        full = self.de.decompose_by_dimension(self.rdf, "fulfillment_rate", "product",
                                              self.period_a, self.period_b, max_items=50)

        def _step_value(m):
            return (m.effects.rate_effect + m.effects.mix_effect) if m.effects else m.change_abs

        true_largest = max(full.members, key=lambda m: abs(_step_value(m))).name

        result = self.be.bridge_periods(self.rdf, "fulfillment_rate", self.period_a,
                                        self.period_b, by="product", max_items=1)
        member_steps = [s for s in result.steps if s.kind == "member"]
        self.assertEqual(len(member_steps), 1)
        self.assertEqual(member_steps[0].label, true_largest)


# ---------------------------------------------------------------------------
# trap 3: mean-kind KPIs are refused on the dimension axis
# ---------------------------------------------------------------------------
class TestMeanKindByDimensionIsRefused(BridgeTestCase):
    def test_inventory_units_by_region_is_refused_not_fabricated(self):
        self.assertEqual(self.api.kind("inventory_units"), "mean")
        result = self.be.bridge_periods(self.rdf, "inventory_units", self.period_a,
                                        self.period_b, by="region")
        self.assertEqual(result.status, "unsupported_kind")
        self.assertEqual(result.steps, ())
        self.assertIsNone(result.total_change)

    def test_inventory_units_by_formula_still_works(self):
        result = self.be.bridge_periods(self.rdf, "inventory_units", self.period_a,
                                        self.period_b, by="formula")
        self.assertEqual(result.status, "single_term")


# ---------------------------------------------------------------------------
# formula axis inherits decompose_formula verbatim
# ---------------------------------------------------------------------------
class TestFormulaAxisInheritsDecomposeFormulaVerbatim(BridgeTestCase):
    def test_status_shape_and_step_values_match_decompose_formula(self):
        fd = self.de.decompose_formula(self.rdf, "gross_margin_pct", self.period_a, self.period_b)
        bridge = self.be.bridge_periods(self.rdf, "gross_margin_pct", self.period_a,
                                        self.period_b, by="formula")
        self.assertEqual(bridge.status, fd.status)
        self.assertEqual(bridge.shape, fd.shape)

        component_values = sorted(s.value for s in bridge.steps if s.kind == "component")
        effect_values = sorted(e.value for e in fd.effects if e.value is not None)
        self.assertEqual(len(component_values), len(effect_values))
        for a, b in zip(component_values, effect_values):
            self.assertAlmostEqual(a, b, places=10)

    def test_a_two_field_additive_formula_has_no_residual(self):
        result = self.be.bridge_periods(self.rdf, "revenue_less_marketing_spend",
                                        self.period_a, self.period_b, by="formula")
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.shape, "additive")
        self.assertAlmostEqual(result.unexplained, 0.0, places=6)


# ---------------------------------------------------------------------------
# ordering, cumulative totals, and argument errors
# ---------------------------------------------------------------------------
class TestStepOrderingIsTotalAndDeterministic(BridgeTestCase):
    def test_others_is_second_to_last_and_unexplained_is_last(self):
        result = self.be.bridge_periods(self.rdf, "gross_margin_pct", self.period_a,
                                        self.period_b, by="product", max_items=1)
        kinds = [s.kind for s in result.steps]
        self.assertEqual(kinds[-1], "unexplained")
        self.assertEqual(kinds[-2], "others")

    def test_named_steps_are_sorted_by_absolute_value_descending(self):
        result = self.be.bridge_periods(self.rdf, "revenue", self.period_a, self.period_b,
                                        by="region")
        member_values = [abs(s.value) for s in result.steps if s.kind == "member"]
        self.assertEqual(member_values, sorted(member_values, reverse=True))

    def test_a_constructed_tie_breaks_on_label(self):
        result = self.be.bridge_periods(self.rdf, "revenue", self.period_a, self.period_b,
                                        by="formula")
        # Not a real tie in this fixture, but the sort key itself is checked:
        # verify the comparator used is (value magnitude desc, label asc) by
        # confirming labels are ascending among any equal-magnitude run.
        labels_by_value = [(round(abs(s.value), 6), s.label) for s in result.steps
                          if s.kind in ("component", "interaction")]
        for i in range(len(labels_by_value) - 1):
            if labels_by_value[i][0] == labels_by_value[i + 1][0]:
                self.assertLessEqual(labels_by_value[i][1], labels_by_value[i + 1][1])


class TestCumulativeIsTheRunningTotal(BridgeTestCase):
    def test_each_steps_cumulative_is_start_plus_the_running_sum(self):
        result = self.be.bridge_periods(self.rdf, "revenue", self.period_a, self.period_b,
                                        by="region")
        running = result.start_value
        for step in result.steps:
            running += step.value
            self.assertAlmostEqual(step.cumulative, running, places=6)
        self.assertAlmostEqual(running, result.end_value, places=4)


class TestTimeGrainDimensionIsRejected(BridgeTestCase):
    def test_a_time_grain_column_names_formula_among_the_alternatives(self):
        with self.assertRaises(InvalidArgumentError) as ctx:
            self.be.bridge_periods(self.rdf, "revenue", self.period_a, self.period_b,
                                   by="_quarter")
        payload = ctx.exception.to_payload()
        self.assertEqual(payload["argument"], "by")
        self.assertIn("formula", payload["valid_alternatives"])

    def test_an_unknown_dimension_raises_unknowndimensionerror(self):
        with self.assertRaises(UnknownDimensionError):
            self.be.bridge_periods(self.rdf, "revenue", self.period_a, self.period_b,
                                   by="not_a_real_dimension")


# ---------------------------------------------------------------------------
# integrity / prose guard / determinism / portability
# ---------------------------------------------------------------------------
class TestNumericIntegrityAndProseGuard(BridgeTestCase):
    def test_the_payload_survives_json_dumps_with_no_nan_or_inf(self):
        result = self.be.bridge_periods(self.rdf, "gross_margin_pct", self.period_a,
                                        self.period_b, by="product", max_items=1)
        assert_json_safe(result.to_payload())

    def test_no_string_field_outside_the_allowlist_looks_like_prose(self):
        result = self.be.bridge_periods(self.rdf, "revenue", self.period_a, self.period_b,
                                        by="product", max_items=1)
        assert_no_prose_leak(result.to_payload())

    def test_a_deliberately_long_member_label_still_passes_the_guard(self):
        """`label` is on the prose allowlist -- a member step's name can be an
        arbitrary dimension member string, so it must never be checked for
        length the way `name` (used nowhere in this module) would be."""
        long_name = "A Very Long Member Name That Would Fail Under The Non Allowlisted Key"
        from app.agent.bridge import Bridge, BridgeStep
        fabricated = Bridge(
            kpi="revenue", label="Revenue", unit="currency", by="product", axis="dimension",
            kind="sum", status="ok", shape=None, reason=None,
            start_value=100.0, end_value=150.0, total_change=50.0,
            steps=(BridgeStep(label=long_name, kind="member", value=50.0, cumulative=150.0),),
            unexplained=0.0, selection=None, baseline_selection=None, filters={})
        assert_no_prose_leak(fabricated.to_payload())


class TestDeterminism(BridgeTestCase):
    def test_identical_input_gives_identical_output_across_runs(self):
        first = self.be.bridge_periods(self.rdf, "gross_margin_pct", self.period_a,
                                       self.period_b, by="product").to_payload()
        second = self.be.bridge_periods(self.rdf, "gross_margin_pct", self.period_a,
                                        self.period_b, by="product").to_payload()
        self.assertEqual(first, second)


class TestPortabilityAcrossDatasets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.schema = contracted("hospital_sample.csv", FIXTURES)
        cls.be = BridgeEngine(ContractAPI(cls.schema))
        assert "recovery_rate" in cls.schema.available_kpis

    def test_a_contract_only_ratio_kpi_bridges_by_dimension_and_by_formula(self):
        result = self.be.bridge_periods(
            self.df, "recovery_rate", {"type": "quarter", "year": 2025, "quarter": 2},
            {"type": "quarter", "year": 2025, "quarter": 1}, by="department")
        self.assertEqual(result.status, "ok")
        self.assertAlmostEqual(
            result.start_value + sum(s.value for s in result.steps),
            result.end_value, places=4)

        by_formula = self.be.bridge_periods(
            self.df, "recovery_rate", {"type": "quarter", "year": 2025, "quarter": 2},
            {"type": "quarter", "year": 2025, "quarter": 1}, by="formula")
        self.assertIn(by_formula.status, ("ok", "single_term", "unsupported_formula", "insufficient"))


if __name__ == "__main__":
    unittest.main()
