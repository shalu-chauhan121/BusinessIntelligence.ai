"""
`decompose_formula` -- `agent/decompose.py`. Genuinely new maths: today
`formula_decomposition_candidates` (`engines/hypotheses.py:218`) reads a KPI's
formula and writes a sentence about which half of a ratio moved, computing no
attribution arithmetic at all.

The headline invariant is unconditional: `sum(effects) + interaction +
unexplained == total_change`, exactly, for every formula shape, because
`unexplained` is a residual, not an independent computation. What varies by
shape is whether that residual is genuinely zero. For an additive formula and
a ratio it is (row-wise `evaluate(...).sum()` is linear); for a `sum`-kind
row-wise product it is not, because `Σ(aᵢ·bᵢ) ≠ Σa · Σb`, and
`TestMultiplicativeShape` asserts the gap is real and reported rather than
hidden.

Checked against every KPI in every sample fixture before writing this: the
only operator any formula uses is `-`. There is no multiplicative KPI
anywhere in the repo, so the multiplicative branch is exercised through a
hand-assembled `CompiledKpi`, the way `test_contract_api.py` already builds
synthetic contract entries for a case the sample data does not cover.
"""
from __future__ import annotations

import unittest

from app.agent.contract_api import ContractAPI
from app.agent.decompose import DecomposeEngine
from app.engines.analysis import price_volume_decomposition
from app.engines.observe import Timeframe, slice_period
from app.kpi.resolver import CompiledKpi, parse_expression

from .base import EngineTestCase, FIXTURES, assert_json_safe, assert_no_prose_leak, contracted


class DecomposeFormulaTestCase(EngineTestCase):
    """The shared retail fixture, with a compiled contract."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.rdf, cls.rschema = contracted("business_metrics_sample.csv")
        cls.api = ContractAPI(cls.rschema)
        cls.de = DecomposeEngine(cls.api)
        cls.period_a = {"type": "quarter", "year": 2026, "quarter": 2}
        cls.period_b = {"type": "quarter", "year": 2026, "quarter": 1}


class TestReconciliationIsUnconditional(DecomposeFormulaTestCase):
    """The file's headline test: parameterised across every KPI in the
    contract at once, every shape simultaneously."""

    def test_effects_plus_interaction_plus_unexplained_equals_total_change(self):
        for key in self.rschema.available_kpis:
            with self.subTest(kpi=key):
                result = self.de.decompose_formula(self.rdf, key, self.period_a, self.period_b)
                if result.status != "ok":
                    continue
                explained = sum(e.value for e in result.effects) + result.interaction
                self.assertAlmostEqual(explained + result.unexplained, result.total_change,
                                       places=4)


class TestAdditiveShape(DecomposeFormulaTestCase):
    def test_gross_profit_splits_into_two_signed_exact_effects(self):
        result = self.de.decompose_formula(self.rdf, "gross_profit", self.period_a, self.period_b)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.shape, "additive")
        by_name = {e.name: e.value for e in result.effects}
        self.assertEqual(set(by_name), {"revenue", "cost_of_goods"})
        self.assertEqual(result.interaction, 0.0)
        self.assertAlmostEqual(result.unexplained, 0.0, places=4)

        revenue_change = self.api.value(
            slice_period(self.rdf, Timeframe(2026, 2)), "revenue") - self.api.value(
            slice_period(self.rdf, Timeframe(2026, 1)), "revenue")
        self.assertAlmostEqual(by_name["revenue"], revenue_change, places=4)
        # cost_of_goods is subtractive in gross_profit's formula
        cost_change = self.api.value(
            slice_period(self.rdf, Timeframe(2026, 2)), "cost_of_goods") - self.api.value(
            slice_period(self.rdf, Timeframe(2026, 1)), "cost_of_goods")
        self.assertAlmostEqual(by_name["cost_of_goods"], -cost_change, places=4)


class TestRatioShape(DecomposeFormulaTestCase):
    def test_numerator_and_denominator_effects_reconcile_to_the_ratio_change(self):
        result = self.de.decompose_formula(self.rdf, "gross_margin_pct",
                                           self.period_a, self.period_b)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.shape, "ratio")
        self.assertAlmostEqual(result.unexplained, 0.0, places=4)

    def test_the_numerators_own_terms_break_out_beneath_it(self):
        result = self.de.decompose_formula(self.rdf, "gross_margin_pct",
                                           self.period_a, self.period_b)
        numerator = next(e for e in result.effects if e.name == "numerator")
        self.assertIsNotNone(numerator.components)
        self.assertEqual({c.name for c in numerator.components}, {"revenue", "cost_of_goods"})
        self.assertAlmostEqual(sum(c.value for c in numerator.components), numerator.value,
                               places=4)


class TestSingleTermShape(DecomposeFormulaTestCase):
    def test_a_bare_field_kpi_reports_single_term_not_a_fabricated_effect(self):
        result = self.de.decompose_formula(self.rdf, "revenue", self.period_a, self.period_b)
        self.assertEqual(result.status, "single_term")
        self.assertEqual(result.effects, ())
        self.assertIsNotNone(result.total_change)   # the fact itself is still reported


class TestMultiplicativeShape(unittest.TestCase):
    """No multiplicative KPI exists in any sample fixture -- exercised through
    a hand-assembled `CompiledKpi`, as `test_contract_api.py:137` already does
    for a case the sample data does not cover."""

    @classmethod
    def setUpClass(cls):
        cls.df, cls.schema = contracted("business_metrics_sample.csv")
        cls.schema.contract_resolver["orders_x_units"] = CompiledKpi(
            key="orders_x_units", label="Orders x units", unit="count", kind="sum",
            expression_ast=parse_expression("{orders} * {units_sold}"),
            source_fields=["orders", "units_sold"])
        cls.schema.available_kpis = list(cls.schema.available_kpis) + ["orders_x_units"]
        cls.api = ContractAPI(cls.schema)
        cls.de = DecomposeEngine(cls.api)
        cls.period_a = {"type": "quarter", "year": 2026, "quarter": 2}
        cls.period_b = {"type": "quarter", "year": 2026, "quarter": 1}

    def test_two_symmetric_effects_plus_an_interaction_are_reported(self):
        result = self.de.decompose_formula(self.df, "orders_x_units", self.period_a, self.period_b)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.shape, "multiplicative")
        self.assertEqual({e.name for e in result.effects}, {"orders", "units_sold"})
        self.assertNotEqual(result.interaction, 0.0)

    def test_effects_and_interaction_reconcile_to_the_aggregate_totals_delta(self):
        """Δ(a·b) computed from aggregate totals is always exact -- this is
        the identity that makes `unexplained` a real, non-tautological signal
        rather than just soaking up an arithmetic mistake."""
        result = self.de.decompose_formula(self.df, "orders_x_units", self.period_a, self.period_b)
        a_c = self.api.value(slice_period(self.df, Timeframe(2026, 2)), "orders")
        a_b = self.api.value(slice_period(self.df, Timeframe(2026, 1)), "orders")
        b_c = self.api.value(slice_period(self.df, Timeframe(2026, 2)), "units_sold")
        b_b = self.api.value(slice_period(self.df, Timeframe(2026, 1)), "units_sold")
        aggregate_delta = (a_c * b_c) - (a_b * b_b)

        explained = sum(e.value for e in result.effects) + result.interaction
        self.assertAlmostEqual(explained, aggregate_delta, places=4)

    def test_a_row_wise_product_reports_a_real_nonzero_unexplained(self):
        """Sigma(a_i * b_i) != (Sigma a) * (Sigma b) -- the gap this
        decomposition cannot close, surfaced rather than hidden."""
        result = self.de.decompose_formula(self.df, "orders_x_units", self.period_a, self.period_b)
        self.assertGreater(abs(result.unexplained), 1.0)


class TestReconcilesWithLegacyPriceVolumeDecomposition(EngineTestCase):
    """`price_volume_decomposition`'s one-sided convention
    (`price_effect = q_current * delta_p`) makes its two effects sum exactly
    to total_change only by absorbing the cross-term into the price effect.
    The symmetric convention this module uses is the identical arithmetic
    with that cross-term pulled back out: `legacy_price_effect ==
    symmetric_price_effect + interaction`. No real KPI in this codebase is a
    row-wise product of an aggregate price and an aggregate quantity (price
    is a ratio, not a column), so this identity is checked directly rather
    than through `decompose_formula`."""

    def test_symmetric_effects_plus_interaction_equal_the_legacy_effects(self):
        cur = slice_period(self.df, Timeframe(2026, 2))
        base = slice_period(self.df, Timeframe(2026, 1))
        legacy = price_volume_decomposition(cur, base)

        p_c, p_b = legacy["avg_price_current"], legacy["avg_price_baseline"]
        q_c, q_b = legacy["units_current"], legacy["units_baseline"]

        volume_effect = (q_c - q_b) * p_b          # Delta(q) * p_base
        price_effect = q_b * (p_c - p_b)            # q_base * Delta(p)
        interaction = (q_c - q_b) * (p_c - p_b)

        self.assertAlmostEqual(volume_effect, legacy["volume_effect"], places=6)
        self.assertAlmostEqual(price_effect + interaction, legacy["price_effect"], places=6)


# ---------------------------------------------------------------------------
# integrity / prose guard / determinism / portability
# ---------------------------------------------------------------------------
class TestProseGuard(DecomposeFormulaTestCase):
    BANNED = {"narrative", "statement", "title", "note"}

    def _assert_no_banned_keys(self, payload, path: str = "$") -> None:
        if isinstance(payload, dict):
            for key, value in payload.items():
                self.assertNotIn(key, self.BANNED, f"{path}.{key} is a banned prose field")
                self._assert_no_banned_keys(value, f"{path}.{key}")
        elif isinstance(payload, list):
            for i, item in enumerate(payload):
                self._assert_no_banned_keys(item, f"{path}[{i}]")

    def test_no_banned_prose_key_survives(self):
        result = self.de.decompose_formula(self.rdf, "gross_margin_pct",
                                           self.period_a, self.period_b)
        self._assert_no_banned_keys(result.to_payload())

    def test_no_string_field_outside_the_allowlist_looks_like_prose(self):
        result = self.de.decompose_formula(self.rdf, "gross_profit", self.period_a, self.period_b)
        assert_no_prose_leak(result.to_payload())

    def test_the_payload_survives_json_dumps_with_no_nan_or_inf(self):
        result = self.de.decompose_formula(self.rdf, "gross_margin_pct",
                                           self.period_a, self.period_b)
        assert_json_safe(result.to_payload())


class TestDeterminism(DecomposeFormulaTestCase):
    def test_identical_input_gives_identical_output_across_runs(self):
        first = self.de.decompose_formula(
            self.rdf, "gross_margin_pct", self.period_a, self.period_b).to_payload()
        second = self.de.decompose_formula(
            self.rdf, "gross_margin_pct", self.period_a, self.period_b).to_payload()
        self.assertEqual(first, second)


class TestPortabilityAcrossDatasets(unittest.TestCase):
    def test_decompose_formula_works_on_every_kpi_in_two_other_domains(self):
        for csv_name, samples_dir in [("hospital_sample.csv", FIXTURES),
                                      ("school_kpi_smoke_sample.csv", None)]:
            df, schema = (contracted(csv_name, samples_dir) if samples_dir
                         else contracted(csv_name))
            api = ContractAPI(schema)
            de = DecomposeEngine(api)
            period_a = {"type": "quarter", "year": 2025, "quarter": 2}
            period_b = {"type": "quarter", "year": 2025, "quarter": 1}
            for key in schema.available_kpis:
                with self.subTest(dataset=csv_name, kpi=key):
                    result = de.decompose_formula(df, key, period_a, period_b)
                    self.assertIn(result.status,
                                 ("ok", "single_term", "unsupported_formula", "insufficient"))


if __name__ == "__main__":
    unittest.main()
