"""
`test_statistical_significance` / `check_sample_adequacy` --
`agent/significance.py`.

The property that matters most is that the p-value is the *real* one, not an
approximation that happens to look right: `TestPValueMatchesScipy` pins it
against `scipy.stats.pearsonr` on raw arrays, which is the whole justification
for taking the dependency at this batch rather than hand-rolling a Fisher-z
normal approximation as `agent/trend.py` did for its slope.

Second is that multiplicity is actually corrected.
`TestMultiplicityIsCorrected` encodes the measured failure the design exists to
prevent: 30 independent p-values drawn uniformly -- exactly what a Tier-4 sweep
of 30 unrelated candidates produces -- yield raw "significant" hits at
alpha=0.05 while the FDR-adjusted set is empty.

Third is that nothing is fabricated where there is no sample. Fisher's z needs
`n >= 4`, so `n = 3` must produce a real p-value *and* no interval; a
correlation whose own status was never `ok` must produce neither. Measured on
the retail fixture this is not hypothetical: every member axis the dataset has
is too small to detect anything below `|r| = 0.99`, and
`TestSampleAdequacyOnTheRealFixture` pins that number.
"""
from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd
from scipy import stats

from app.agent.contract_api import ContractAPI
from app.agent.correlate import CorrelationEngine
from app.agent.errors import InvalidArgumentError
from app.agent.significance import (DEFAULT_ALPHA, MIN_N_FOR_CI, SignificanceEngine,
                                    benjamini_hochberg, fisher_interval,
                                    min_detectable_r, p_value_for_r)

from .base import EngineTestCase, assert_json_safe, assert_no_prose_leak, contracted

# The prose fields the legacy contest path published; none may reappear here.
BANNED = {"interpretation", "detail", "narrative", "note", "conclusion",
          "history_note", "dispersion_note", "statistical_power"}


def _assert_no_banned_keys(payload, path: str = "$") -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            assert key not in BANNED, f"{path}.{key} is a banned prose field"
            _assert_no_banned_keys(value, f"{path}.{key}")
    elif isinstance(payload, list):
        for i, item in enumerate(payload):
            _assert_no_banned_keys(item, f"{path}[{i}]")


class SignificanceTestCase(EngineTestCase):
    """The retail fixture with its contract attached, so a contract-only ratio
    KPI is queryable through every tool here."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.cdf, cls.cschema = contracted("business_metrics_sample.csv")
        cls.api = ContractAPI(cls.cschema)
        cls.engine = SignificanceEngine(cls.api)
        cls.correlate = CorrelationEngine(cls.api)


# ---------------------------------------------------------------------------
# the statistics themselves
# ---------------------------------------------------------------------------
class TestPValueMatchesScipy(unittest.TestCase):
    """`p_value_for_r` computes from `(r, n)` alone; it must still be the number
    `scipy.stats.pearsonr` computes from the arrays."""

    def test_it_matches_pearsonr_on_a_fixture(self):
        x = np.array([1., 2, 3, 4, 5, 7, 9, 11])
        y = np.array([2., 1, 4, 3, 6, 7, 8, 12])
        r, expected = stats.pearsonr(x, y)
        self.assertAlmostEqual(p_value_for_r(float(r), len(x)), float(expected), places=12)

    def test_it_matches_pearsonr_across_many_random_samples(self):
        rng = np.random.default_rng(3)
        for _ in range(25):
            n = int(rng.integers(5, 40))
            x = rng.normal(size=n)
            y = 0.4 * x + rng.normal(size=n)
            r, expected = stats.pearsonr(x, y)
            self.assertAlmostEqual(p_value_for_r(float(r), n), float(expected), places=12)

    def test_a_negative_r_gives_the_same_two_sided_p_as_its_positive(self):
        self.assertAlmostEqual(p_value_for_r(-0.6, 20), p_value_for_r(0.6, 20), places=15)

    def test_controls_consume_degrees_of_freedom(self):
        # A partial correlation on the same r and n is less significant than the
        # raw one, because a control costs a degree of freedom.
        self.assertGreater(p_value_for_r(0.6, 20, controls=3), p_value_for_r(0.6, 20))

    def test_no_p_value_where_there_are_no_degrees_of_freedom(self):
        self.assertIsNone(p_value_for_r(0.9, 2))
        self.assertIsNone(p_value_for_r(None, 40))
        self.assertIsNone(p_value_for_r(0.9, 5, controls=4))


class TestConfidenceInterval(unittest.TestCase):
    def test_the_interval_contains_the_point_estimate(self):
        lo, hi, status = fisher_interval(0.62, 30)
        self.assertEqual(status, "ok")
        self.assertLess(lo, 0.62)
        self.assertGreater(hi, 0.62)

    def test_the_interval_narrows_monotonically_as_n_rises(self):
        widths = []
        for n in (6, 10, 20, 50, 200):
            lo, hi, _ = fisher_interval(0.5, n)
            widths.append(hi - lo)
        self.assertEqual(widths, sorted(widths, reverse=True))

    def test_it_never_runs_off_the_end_of_the_scale(self):
        # The reason for transforming through Fisher's z at all: a naive
        # `r +- 1.96*se` on a near-perfect r produces a bound above 1.
        lo, hi, _ = fisher_interval(0.98, 8)
        self.assertGreaterEqual(lo, -1.0)
        self.assertLessEqual(hi, 1.0)

    def test_three_points_is_a_typed_status_not_an_invented_interval(self):
        lo, hi, status = fisher_interval(0.9, 3)
        self.assertEqual(status, "insufficient_for_ci")
        self.assertIsNone(lo)
        self.assertIsNone(hi)

    def test_a_control_raises_the_floor_for_an_interval(self):
        self.assertEqual(fisher_interval(0.9, 4)[2], "ok")
        self.assertEqual(fisher_interval(0.9, 4, controls=1)[2], "insufficient_for_ci")


class TestBenjaminiHochberg(unittest.TestCase):
    def test_it_matches_a_hand_computed_family(self):
        # m=4; sorted p = .01, .03, .04, .20 -> raw q = .04, .06, .0533, .20,
        # then enforced monotone from the largest down.
        got = benjamini_hochberg([0.01, 0.04, 0.03, 0.20])
        for value, expected in zip(got, [0.04, 0.05333333, 0.05333333, 0.20]):
            self.assertAlmostEqual(value, expected, places=6)

    def test_q_is_never_below_its_own_p(self):
        rng = np.random.default_rng(5)
        ps = list(rng.uniform(size=40))
        for p, q in zip(ps, benjamini_hochberg(ps)):
            self.assertGreaterEqual(q, p - 1e-12)

    def test_q_is_monotone_in_p(self):
        ps = [0.001, 0.01, 0.02, 0.2, 0.5, 0.9]
        qs = benjamini_hochberg(ps)
        self.assertEqual(qs, sorted(qs))

    def test_a_family_of_one_leaves_p_untouched(self):
        self.assertAlmostEqual(benjamini_hochberg([0.031])[0], 0.031, places=12)

    def test_a_test_that_never_ran_is_carried_through_and_leaves_the_family(self):
        # `None` must not consume a share of the false-discovery budget: with
        # one real test the family size is 1, so q == p.
        got = benjamini_hochberg([None, 0.04, None])
        self.assertIsNone(got[0])
        self.assertIsNone(got[2])
        self.assertAlmostEqual(got[1], 0.04, places=12)

    def test_an_all_none_family_returns_all_none(self):
        self.assertEqual(benjamini_hochberg([None, None]), [None, None])


class TestMultiplicityIsCorrected(unittest.TestCase):
    """The measured failure the whole FDR decision exists to prevent."""

    def test_thirty_null_candidates_produce_raw_hits_and_no_fdr_hits(self):
        # 30 independent tests against a true null: uniform p-values, which is
        # exactly what a Tier-4 sweep of 30 unrelated candidates produces.
        rng = np.random.default_rng(19)
        ps = list(rng.uniform(size=30))
        qs = benjamini_hochberg(ps)
        self.assertGreaterEqual(sum(1 for p in ps if p < DEFAULT_ALPHA), 1,
                                "the null family should produce at least one raw hit")
        self.assertEqual(sum(1 for q in qs if q < DEFAULT_ALPHA), 0,
                         "no candidate should survive FDR correction against a true null")

    def test_a_genuine_signal_still_survives_a_family_of_thirty(self):
        # Correction must not be so blunt that a real effect is lost.
        ps = [1e-8] + list(np.linspace(0.1, 0.99, 29))
        self.assertLess(benjamini_hochberg(ps)[0], DEFAULT_ALPHA)


class TestMinimumDetectableEffect(unittest.TestCase):
    def test_it_falls_as_the_sample_grows(self):
        values = [min_detectable_r(n) for n in (5, 10, 30, 100, 500)]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_it_is_undefined_below_four_points(self):
        self.assertIsNone(min_detectable_r(3))

    def test_a_four_member_dimension_can_only_detect_a_near_perfect_r(self):
        self.assertGreater(min_detectable_r(4), 0.95)


# ---------------------------------------------------------------------------
# test_statistical_significance()
# ---------------------------------------------------------------------------
class TestSignificanceComposesCorrelate(SignificanceTestCase):
    """The composition rule: two tools may never report different numbers for
    the same pair."""

    def test_r_and_n_are_identical_to_a_direct_correlate_kpis_call(self):
        args = dict(mode="time_series", grain="quarter")
        direct = self.correlate.correlate_kpis(self.cdf, "revenue", "units_sold", **args)
        via = self.engine.test_statistical_significance(
            self.cdf, "revenue", cause_kpi="units_sold", **args).rows[0]
        self.assertEqual(via.r, direct.r)
        self.assertEqual(via.n, direct.n)
        self.assertEqual(via.status, direct.status)

    def test_a_sweep_row_matches_the_standalone_call_for_that_pair(self):
        args = dict(mode="time_series", grain="quarter")
        alone = self.engine.test_statistical_significance(
            self.cdf, "revenue", cause_kpi="units_sold", **args).rows[0]
        swept = {row.cause_kpi: row for row in self.engine.test_statistical_significance(
            self.cdf, "revenue", candidates=["units_sold", "marketing_spend"],
            **args).rows}["units_sold"]
        # p is a property of the pair; only q is a property of the family.
        self.assertEqual((alone.r, alone.n, alone.p_value),
                         (swept.r, swept.n, swept.p_value))

    def test_a_family_of_one_has_q_equal_to_p(self):
        result = self.engine.test_statistical_significance(
            self.cdf, "revenue", cause_kpi="units_sold", mode="time_series", grain="quarter")
        self.assertEqual(result.family_size, 1)
        self.assertEqual(result.rows[0].q_value, result.rows[0].p_value)


class TestSignificancePayload(SignificanceTestCase):
    def test_a_significant_pair_reports_an_interval_excluding_zero(self):
        row = self.engine.test_statistical_significance(
            self.cdf, "revenue", cause_kpi="units_sold",
            mode="time_series", grain="quarter").rows[0]
        self.assertEqual(row.status, "ok")
        self.assertTrue(row.significant_raw)
        self.assertEqual(row.ci_status, "ok")
        self.assertTrue(row.ci_excludes_zero)
        self.assertLessEqual(row.ci_low, row.r)
        self.assertGreaterEqual(row.ci_high, row.r)

    def test_both_verdicts_are_published(self):
        result = self.engine.test_statistical_significance(
            self.cdf, "gross_margin_pct",
            candidates=["cost_of_goods", "revenue", "units_sold", "marketing_spend"],
            mode="time_series", grain="quarter")
        for row in result.rows:
            self.assertIn("significant_raw", row.to_payload())
            self.assertIn("significant_fdr", row.to_payload())

    def test_rows_are_ordered_by_the_key_the_payload_declares(self):
        result = self.engine.test_statistical_significance(
            self.cdf, "gross_margin_pct",
            candidates=["cost_of_goods", "revenue", "units_sold", "marketing_spend"],
            mode="time_series", grain="quarter")
        self.assertEqual(result.ordered_by, "p_value")
        ps = [row.p_value for row in result.rows]
        self.assertEqual(ps, sorted(ps))

    def test_a_correlation_with_no_usable_sample_yields_no_statistics(self):
        # `min_n` above the fixture's own history forces `insufficient_n`
        # through the underlying correlation; nothing may be invented from it.
        row = self.engine.test_statistical_significance(
            self.cdf, "revenue", cause_kpi="units_sold", mode="time_series",
            grain="quarter", min_n=500).rows[0]
        self.assertEqual(row.status, "insufficient_n")
        self.assertIsNone(row.r)
        self.assertIsNone(row.p_value)
        self.assertIsNone(row.q_value)
        self.assertIsNone(row.significant_raw)
        self.assertIsNone(row.significant_fdr)
        self.assertEqual(row.ci_status, "insufficient_for_ci")


class TestSignificanceArguments(SignificanceTestCase):
    def test_naming_neither_a_cause_nor_candidates_is_rejected(self):
        with self.assertRaises(InvalidArgumentError):
            self.engine.test_statistical_significance(self.cdf, "revenue")

    def test_naming_both_a_cause_and_candidates_is_rejected(self):
        with self.assertRaises(InvalidArgumentError):
            self.engine.test_statistical_significance(
                self.cdf, "revenue", cause_kpi="units_sold", candidates=["orders"])

    def test_an_alpha_outside_the_unit_interval_is_rejected(self):
        for bad in (0.0, 1.0, -0.1, 4.0):
            with self.assertRaises(InvalidArgumentError):
                self.engine.test_statistical_significance(
                    self.cdf, "revenue", cause_kpi="units_sold", alpha=bad)

    def test_a_hallucinated_kpi_never_reaches_pandas(self):
        from app.agent.errors import UnknownKpiError
        with self.assertRaises(UnknownKpiError):
            self.engine.test_statistical_significance(
                self.cdf, "revenue", cause_kpi="synergy_index")


# ---------------------------------------------------------------------------
# check_sample_adequacy()
# ---------------------------------------------------------------------------
class TestSampleAdequacyOnTheRealFixture(SignificanceTestCase):
    """The measured reason this tool exists: on the dataset this product demos
    on, *every* member axis is too small to detect anything short of a
    near-perfect correlation, and nothing in the system says so today."""

    def test_every_member_axis_of_the_retail_fixture_is_inadequate(self):
        result = self.engine.check_sample_adequacy(self.cdf, "revenue")
        member_axes = [a for a in result.axes if a.basis == "member"]
        self.assertTrue(member_axes)
        for axis in member_axes:
            self.assertEqual(axis.verdict, "inadequate", f"{axis.scope} unexpectedly adequate")

    def test_the_four_member_region_axis_can_only_detect_a_near_perfect_r(self):
        result = self.engine.check_sample_adequacy(self.cdf, "revenue", dimension="region")
        region = next(a for a in result.axes if a.scope == "region")
        self.assertEqual(region.units, 4)
        self.assertGreater(region.min_detectable_r, 0.95)

    def test_the_period_axis_is_the_one_that_supports_a_correlation(self):
        result = self.engine.check_sample_adequacy(self.cdf, "revenue")
        period = next(a for a in result.axes if a.basis == "period")
        self.assertEqual(period.verdict, "adequate")
        self.assertEqual(result.best_basis, "period")
        self.assertLess(period.min_detectable_r, 0.95)

    def test_usable_points_are_adjacent_pairs_not_points(self):
        # A differenced series gets one point per calendar-adjacent pair, so
        # counting points would overstate the sample this tool is honest about.
        result = self.engine.check_sample_adequacy(self.cdf, "revenue")
        period = next(a for a in result.axes if a.basis == "period")
        self.assertEqual(period.usable_points, period.units - 1 - period.gaps)

    def test_a_named_dimension_reports_only_that_dimension(self):
        result = self.engine.check_sample_adequacy(self.cdf, "revenue", dimension="product")
        scopes = {a.scope for a in result.axes if a.basis == "member"}
        self.assertEqual(scopes, {"product"})

    def test_a_finer_grain_yields_a_smaller_detectable_effect(self):
        quarterly = self.engine.check_sample_adequacy(self.cdf, "revenue", grain="quarter")
        weekly = self.engine.check_sample_adequacy(self.cdf, "revenue", grain="week")
        q = next(a for a in quarterly.axes if a.basis == "period")
        w = next(a for a in weekly.axes if a.basis == "period")
        self.assertGreater(w.usable_points, q.usable_points)
        self.assertLess(w.min_detectable_r, q.min_detectable_r)

    def test_an_out_of_range_power_is_rejected(self):
        for bad in (0.0, 1.0, 1.5):
            with self.assertRaises(InvalidArgumentError):
                self.engine.check_sample_adequacy(self.cdf, "revenue", power=bad)


# ---------------------------------------------------------------------------
# cross-cutting guards
# ---------------------------------------------------------------------------
class TestNumericIntegrityAndProseGuard(SignificanceTestCase):
    def _payloads(self):
        return [
            self.engine.test_statistical_significance(
                self.cdf, "revenue", cause_kpi="units_sold",
                mode="time_series", grain="quarter").to_payload(),
            self.engine.test_statistical_significance(
                self.cdf, "gross_margin_pct",
                candidates=["cost_of_goods", "units_sold", "marketing_spend"],
                mode="time_series", grain="quarter").to_payload(),
            self.engine.check_sample_adequacy(self.cdf, "revenue").to_payload(),
        ]

    def test_every_payload_survives_json_dumps_with_no_nan_or_inf(self):
        for payload in self._payloads():
            assert_json_safe(payload)

    def test_no_string_field_outside_the_allowlist_looks_like_prose(self):
        for payload in self._payloads():
            assert_no_prose_leak(payload)

    def test_no_banned_prose_key_survives(self):
        for payload in self._payloads():
            _assert_no_banned_keys(payload)


class TestDeterminism(SignificanceTestCase):
    def test_identical_input_gives_identical_output_across_runs(self):
        call = lambda: self.engine.test_statistical_significance(
            self.cdf, "gross_margin_pct",
            candidates=["cost_of_goods", "units_sold", "marketing_spend", "revenue"],
            mode="time_series", grain="quarter").to_payload()
        self.assertEqual(json.dumps(call(), sort_keys=True),
                         json.dumps(call(), sort_keys=True))

    def test_candidate_order_does_not_change_the_result(self):
        forward = self.engine.test_statistical_significance(
            self.cdf, "gross_margin_pct", candidates=["cost_of_goods", "units_sold"],
            mode="time_series", grain="quarter")
        reverse = self.engine.test_statistical_significance(
            self.cdf, "gross_margin_pct", candidates=["units_sold", "cost_of_goods"],
            mode="time_series", grain="quarter")
        self.assertEqual([r.cause_kpi for r in forward.rows],
                         [r.cause_kpi for r in reverse.rows])
        self.assertEqual([r.q_value for r in forward.rows], [r.q_value for r in reverse.rows])


class TestPortabilityAcrossDatasets(unittest.TestCase):
    """G3: the hospital and school fixtures have different KPI names and
    different dimensions, and both must run every tool here."""

    def test_both_tools_run_on_the_hospital_and_school_fixtures(self):
        for csv in ("hospital_kpi_smoke_sample.csv", "school_kpi_smoke_sample.csv"):
            df, schema = contracted(csv)
            api = ContractAPI(schema)
            engine = SignificanceEngine(api)
            keys = api.available_keys(df)
            self.assertGreaterEqual(len(keys), 2, csv)

            adequacy = engine.check_sample_adequacy(df, keys[0])
            assert_json_safe(adequacy.to_payload())
            assert_no_prose_leak(adequacy.to_payload())
            self.assertTrue(adequacy.axes, csv)

            result = engine.test_statistical_significance(
                df, keys[0], candidates=list(keys[1:4]), mode="time_series", grain="quarter")
            assert_json_safe(result.to_payload())
            assert_no_prose_leak(result.to_payload())
            for row in result.rows:
                if row.p_value is not None:
                    self.assertGreaterEqual(row.q_value, row.p_value - 1e-12, csv)


if __name__ == "__main__":
    unittest.main()
