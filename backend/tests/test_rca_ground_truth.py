"""
Root-cause analysis against known ground truth.

The demo dataset has a three-factor scenario planted in it, and
sample_data/ground_truth.json records what was planted along with each factor's
TRUE effect on Q2-2026 revenue, measured by re-simulating with that factor
switched off. This suite runs the real engines over the real data and asserts
that the analysis recovers the planted factors: finds all three, ranks them in
the right order, attributes them within tolerance, dates their onsets, and gets
the temporal ordering between them the right way round.

This is the test that would fail if driver ranking silently regressed. The
scorecard it asserts is the same one `scripts/validate_rca.py` prints, so the
number in docs/RANKING.md and the number the build checks cannot diverge.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
for path in (str(BACKEND), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from scripts import validate_rca                                # noqa: E402

from .base import EngineTestCase                                # noqa: E402

TRUTH_PATH = ROOT / "sample_data" / "ground_truth.json"


class TestGroundTruthManifest(unittest.TestCase):
    """The measuring instrument has to be sound before it can measure anything."""

    @classmethod
    def setUpClass(cls):
        cls.truth = json.loads(TRUTH_PATH.read_text())

    def test_manifest_exists_and_names_three_factors(self):
        self.assertEqual(len(self.truth["planted_factors"]), 3)
        self.assertEqual(
            {f["id"] for f in self.truth["planted_factors"]},
            {"north_demand_erosion", "east_demand_erosion", "supply_disruption_product_a"})

    def test_shapley_impacts_sum_to_the_planted_delta(self):
        """Efficiency, on the ground truth itself: the parts must be the whole."""
        total = sum(f["true_impact_revenue"]["shapley"] for f in self.truth["planted_factors"])
        self.assertAlmostEqual(total / self.truth["totals"]["planted_delta"], 1.0, places=6)

    def test_observed_delta_is_planted_effect_plus_natural_drift(self):
        t = self.truth
        self.assertAlmostEqual(
            t["totals"]["planted_delta"] + t["totals"]["natural_drift"],
            t["observed"]["change_abs"], places=2)

    def test_natural_drift_is_positive_and_therefore_not_ignorable(self):
        """
        Trend and seasonality push Q2 UP while the planted factors push it down,
        so the planted effect is larger than the observed movement. Any scoring
        that conflated the two denominators would be wrong; this pins the fact.
        """
        self.assertGreater(self.truth["totals"]["natural_drift"], 0)
        self.assertGreater(self.truth["totals"]["planted_share_of_observed_delta_pct"], 100)

    def test_expected_ranks_are_contiguous(self):
        ranks = sorted(f["expected_rank"] for f in self.truth["planted_factors"])
        self.assertEqual(ranks, [1, 2, 3])


class TestRcaRecoversThePlantedFactors(unittest.TestCase):
    """The scorecard, asserted check by check."""

    @classmethod
    def setUpClass(cls):
        cls.truth = json.loads(TRUTH_PATH.read_text())
        cls.pipeline = validate_rca.run_pipeline()
        cls.checks, cls.detail = validate_rca.score(cls.truth, cls.pipeline)
        cls.by_name = {c["check"]: c for c in cls.checks}

    def assertCheck(self, name):
        check = self.by_name.get(name)
        self.assertIsNotNone(check, f"no check named {name!r}")
        self.assertTrue(check["pass"],
                        f"{name}: {check['detail']} (value={check['value']}, bar={check['bar']})")

    def test_the_movement_is_detected_as_a_meaningful_signal(self):
        self.assertCheck("detection")

    def test_all_three_planted_loci_are_found(self):
        self.assertCheck("locus_recall")

    def test_the_top_of_the_ranking_is_not_padded_with_non_factors(self):
        self.assertCheck("precision_at_3")

    def test_the_ranking_order_matches_true_impact(self):
        self.assertCheck("rank_correlation")

    def test_the_decomposition_arithmetic_is_exact(self):
        self.assertCheck("decomposition_exact")

    def test_attribution_is_within_tolerance_of_the_counterfactual_truth(self):
        self.assertCheck("attribution_mae")

    def test_each_factors_onset_is_dated_within_two_weeks(self):
        self.assertCheck("onset_dating")

    def test_the_kpi_moved_before_the_best_documented_cause(self):
        """
        The finding the CONTEST stage exists for: the supply disruption is real
        and well evidenced, but the decline started four weeks before it did.
        """
        self.assertCheck("temporal_ordering")

    def test_a_demand_led_decline_reads_as_volume_not_price(self):
        self.assertCheck("mechanism_volume_led")

    def test_every_check_passes(self):
        failed = [c["check"] for c in self.checks if not c["pass"]]
        self.assertEqual(failed, [], f"failed checks: {failed}")


class TestRankingIsDrivenByEvidenceNotWeights(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.truth = json.loads(TRUTH_PATH.read_text())
        cls.pipeline = validate_rca.run_pipeline()

    def test_top_three_survives_a_fifty_percent_swing_in_every_weight(self):
        result = validate_rca.weight_sensitivity(self.pipeline, self.truth)
        unstable = [v for v in result["variants"] if not v["top3_unchanged"]]
        self.assertTrue(result["stable"],
                        f"the weights are doing the work: {unstable}")


class TestDriverRankingContract(EngineTestCase):
    """Shape guarantees the API contract and the frontend depend on."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from app.engines.observe import Timeframe, observe
        cls.observation = observe(cls.df, cls.schema, "revenue", Timeframe(2026, 2))

    def test_ranks_are_contiguous_and_start_at_one(self):
        ranks = [d["rank"] for d in self.observation["top_drivers"]]
        self.assertEqual(ranks, list(range(1, len(ranks) + 1)))

    def test_scores_are_monotonically_non_increasing(self):
        scores = [d["driver_score"] for d in self.observation["top_drivers"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_every_driver_publishes_the_arithmetic_behind_its_score(self):
        for d in self.observation["top_drivers"]:
            self.assertIn("score_components", d)
            self.assertEqual(set(d["score_components"]),
                             {"magnitude", "disproportion", "significance", "persistence"})
            self.assertIsNotNone(d.get("member_score"))
            self.assertIsNotNone(d.get("axis_weight"))
            self.assertIn("method", d)

    def test_backward_compatible_fields_are_untouched(self):
        """Investigate, Contest, signals and the API contract all read these."""
        for d in self.observation["top_drivers"]:
            for key in ("dimension", "name", "contribution_pct", "over_index",
                        "is_disproportionate", "change_abs", "change_pct"):
                self.assertIn(key, d)

    def test_size_effect_members_are_not_ranked_as_drivers(self):
        """
        Enterprise is ~60% of the business and carries ~60% of any movement. It
        must not appear above a genuine driver just for being large -- that is
        arithmetic, not a finding.
        """
        top3 = {d["name"] for d in self.observation["top_drivers"][:3]}
        for passenger in ("Enterprise", "SMB", "Online", "Retail Partner"):
            self.assertNotIn(passenger, top3)

    def test_dimension_shapley_shares_sum_to_one_on_the_real_grid(self):
        """The efficiency axiom, on real data rather than a hand-built grid."""
        shares = (self.observation["dimension_shapley"] or {}).get("shares") or {}
        self.assertTrue(shares)
        self.assertAlmostEqual(sum(shares.values()), 1.0, places=9)

    def test_dimension_shapley_total_equals_the_grids_sum_of_squares(self):
        from app.engines.drivers import cell_delta_grid
        from app.engines.observe import Timeframe, slice_period

        tf = Timeframe(2026, 2)
        cells = cell_delta_grid(slice_period(self.df, tf), slice_period(self.df, tf.previous()),
                                self.schema.dimensions, "revenue", self.schema.contract_resolver)
        deltas = [c["delta"] for c in cells]
        mean = sum(deltas) / len(deltas)
        tss = sum((d - mean) ** 2 for d in deltas)
        self.assertAlmostEqual(
            self.observation["dimension_shapley"]["total_variation"] / tss, 1.0, places=9)

    def test_the_movement_is_shaped_by_product_and_region_not_channel_or_segment(self):
        shares = self.observation["dimension_shapley"]["shares"]
        self.assertGreater(shares["product"] + shares["region"], 0.85)
        self.assertLess(shares["channel"], 0.10)
        self.assertLess(shares["segment"], 0.10)


if __name__ == "__main__":
    unittest.main()
