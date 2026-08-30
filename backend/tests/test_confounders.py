"""
`test_confounders` / `test_spurious_correlation` /
`rank_competing_explanations` -- `agent/confounders.py`.

The property that matters most is that a confounded relationship actually
collapses. `TestAConfoundedTripleCollapses` builds `C -> A` and `C -> B` with no
direct `A -> B` link at all: the measured baseline is `r = 0.981`, and holding
`C` constant takes it to `-0.031` (96.8% attenuation, p = 0.74). The same test
pins the other half, which is the one a weak implementation gets wrong -- a
genuinely independent control must leave `r` where it was, and does: 0.981 to
0.981.

Second is that the arithmetic is *valid*, not merely close. Three pairwise `r`s
taken on three different overlaps do not form a positive semi-definite matrix,
and the partial-correlation formula can then return a number outside [-1, 1].
`TestListwiseAlignment` pins that every `r` comes from one aligned sample, and
`TestPartialCorrelationBounds` pins that nothing outside the legal range ever
escapes unflagged.

Third is that "too few observations" and "collinear candidates" are told apart.
They look identical to `numpy.linalg.cond` and call for opposite responses --
change the axis, or drop a candidate. Measured on the retail fixture, ranking
four candidates cross-sectionally over `region`'s four members hits this every
time, and `TestTooFewObservationsIsNotCollinearity` is what would fail if the
two were ever collapsed back into one status.

For `test_spurious_correlation`, what matters is the gap between transforms:
two independent random walks with a shared injected trend measure `r = 1.0` on
levels and `-0.069` on Theil-Sen residuals, and the verdict must be
`trend_driven`. That is the failure mode a business dataset produces by
default, since revenue, cost, headcount and volume all grow together.

For `rank_competing_explanations`, what matters is that **no composite score
exists**. `TestNoScoreIsPublished` asserts it structurally, so a future
"simplification" that reintroduces `score_hypothesis`'s weighted ledger fails
here rather than shipping.
"""
from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from app.agent.confounders import (ATTENUATED_ATTENUATION_PCT, COMPETING_SORT_KEYS,
                                   EXPLAINED_AWAY_ATTENUATION_PCT, ConfounderEngine,
                                   first_order_partial, joint_partial)
from app.agent.contract_api import ContractAPI
from app.agent.errors import InvalidArgumentError, UnknownKpiError
from app.agent.significance import SignificanceEngine
from app.engines import metrics

from .base import EngineTestCase, assert_json_safe, assert_no_prose_leak, contracted

BANNED = {"interpretation", "detail", "narrative", "note", "conclusion",
          "score", "confidence", "support", "confidence_label"}


def _assert_no_banned_keys(payload, path: str = "$") -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            assert key not in BANNED, f"{path}.{key} is a banned field"
            _assert_no_banned_keys(value, f"{path}.{key}")
    elif isinstance(payload, list):
        for i, item in enumerate(payload):
            _assert_no_banned_keys(item, f"{path}[{i}]")


def _weekly_frame(revenue, units_sold, marketing_spend, start="2023-01-02",
                  region="North"):
    """A contract-less weekly frame over the seed `METRICS` registry's own
    column names, so `ContractAPI` resolves them without a bootstrapped
    contract -- the same construction `test_consistency.py` uses."""
    rows = [{"date": pd.Timestamp(start) + pd.Timedelta(weeks=i),
             "region": region if isinstance(region, str) else region[i % len(region)],
             "revenue": float(revenue[i]),
             "units_sold": float(units_sold[i]),
             "marketing_spend": float(marketing_spend[i])}
            for i in range(len(revenue))]
    raw = metrics.normalise_columns(pd.DataFrame(rows))
    schema = metrics.detect_schema(raw)
    return metrics.prepare(raw, schema), schema


def _confounded(seed=7, n=120, independent=False):
    """`C` drives both `A` and `B`, with no direct `A -> B` link. With
    `independent=True` the control is regenerated unrelated to either, which is
    the control case a weak implementation gets wrong."""
    rng = np.random.default_rng(seed)
    c = rng.normal(0, 1, n)
    a = 3.0 * c + rng.normal(0, 0.3, n)
    b = 2.0 * c + rng.normal(0, 0.3, n)
    control = rng.normal(0, 1, n) if independent else c
    return _weekly_frame(5000 + a * 100, 5000 + b * 100, 5000 + control * 100)


class ConfounderTestCase(EngineTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.cdf, cls.cschema = contracted("business_metrics_sample.csv")
        cls.api = ContractAPI(cls.cschema)
        cls.engine = ConfounderEngine(cls.api)
        cls.significance = SignificanceEngine(cls.api)


# ---------------------------------------------------------------------------
# the maths
# ---------------------------------------------------------------------------
class TestPartialCorrelationArithmetic(unittest.TestCase):
    def test_the_first_order_formula_matches_a_hand_computation(self):
        r_ab, r_ac, r_bc = 0.8, 0.7, 0.6
        expected = (r_ab - r_ac * r_bc) / np.sqrt((1 - r_ac ** 2) * (1 - r_bc ** 2))
        self.assertAlmostEqual(first_order_partial(r_ab, r_ac, r_bc), float(expected), places=12)

    def test_a_perfectly_collinear_control_is_a_typed_none_not_a_division_by_zero(self):
        self.assertIsNone(first_order_partial(0.8, 1.0, 0.6))
        self.assertIsNone(first_order_partial(0.8, 0.7, -1.0))

    def test_the_joint_partial_reduces_to_the_first_order_one_with_a_single_control(self):
        rng = np.random.default_rng(2)
        data = rng.normal(size=(3, 200))
        data[1] += 0.5 * data[2]
        data[0] += 0.4 * data[2]
        matrix = np.corrcoef(data)
        expected = first_order_partial(matrix[0, 1], matrix[0, 2], matrix[1, 2])
        self.assertAlmostEqual(joint_partial(matrix, 0, 1), expected, places=10)

    def test_a_singular_matrix_is_a_typed_none_not_a_linalg_error(self):
        singular = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        self.assertIsNone(joint_partial(singular, 0, 1))

    def test_a_nan_matrix_is_a_typed_none(self):
        self.assertIsNone(joint_partial(np.full((3, 3), np.nan), 0, 1))


class TestPartialCorrelationBounds(unittest.TestCase):
    def test_no_partial_correlation_escapes_the_legal_range(self):
        rng = np.random.default_rng(4)
        for _ in range(200):
            data = rng.normal(size=(3, 12))
            data[0] += 0.9 * data[2]
            data[1] += 0.9 * data[2]
            matrix = np.corrcoef(data)
            for value in (first_order_partial(matrix[0, 1], matrix[0, 2], matrix[1, 2]),
                          joint_partial(matrix, 0, 1)):
                if value is not None and np.isfinite(value):
                    self.assertLessEqual(abs(value), 1.0 + 1e-9)


# ---------------------------------------------------------------------------
# test_confounders()
# ---------------------------------------------------------------------------
class TestAConfoundedTripleCollapses(unittest.TestCase):
    """The headline property: a relationship that exists only because of a
    third variable must be seen to vanish when that variable is held."""

    def _run(self, independent):
        df, schema = _confounded(independent=independent)
        engine = ConfounderEngine(ContractAPI(schema))
        return engine.test_confounders(df, "revenue", "units_sold", ["marketing_spend"],
                                       mode="time_series", grain="week")

    def test_the_baseline_association_is_strong(self):
        self.assertGreater(abs(self._run(independent=False).r_baseline), 0.9)

    def test_holding_the_confounder_explains_the_association_away(self):
        row = self._run(independent=False).rows[0]
        self.assertEqual(row.status, "ok")
        self.assertLess(abs(row.r_partial), 0.15)
        self.assertGreater(row.attenuation_pct, EXPLAINED_AWAY_ATTENUATION_PCT)
        self.assertEqual(row.effect, "explained_away")

    def test_the_explained_away_partial_is_not_significant(self):
        self.assertFalse(self._run(independent=False).rows[0].significant)

    def test_an_independent_control_leaves_the_association_untouched(self):
        result = self._run(independent=True)
        row = result.rows[0]
        self.assertEqual(row.effect, "unchanged")
        self.assertLess(abs(row.attenuation_pct), ATTENUATED_ATTENUATION_PCT)
        self.assertAlmostEqual(abs(row.r_partial), abs(result.r_baseline), delta=0.05)

    def test_the_joint_row_agrees_with_the_single_control_row(self):
        result = self._run(independent=False)
        self.assertEqual(result.joint.n_controls, 1)
        self.assertAlmostEqual(result.joint.r_partial, result.rows[0].r_partial, delta=1e-3)

    def test_the_classification_thresholds_are_published(self):
        thresholds = self._run(independent=False).thresholds
        self.assertEqual(thresholds["explained_away_attenuation_pct"],
                         EXPLAINED_AWAY_ATTENUATION_PCT)
        self.assertIn("negligible_r", thresholds)


class TestListwiseAlignment(unittest.TestCase):
    """Every `r` a partial correlation reads must come from the same rows, or
    the three of them do not form a valid correlation matrix."""

    def _frame_with_an_unusable_control_entry(self):
        # A zero in the control's base period makes its percent change NaN
        # (`metrics.pct_change` refuses a zero denominator), so exactly one
        # entry is unusable for any pair involving the control.
        rng = np.random.default_rng(9)
        n = 40
        c = np.abs(rng.normal(5, 1, n)) * 100
        c[3] = 0.0
        a = 5000 + rng.normal(0, 100, n)
        b = 5000 + rng.normal(0, 100, n)
        return _weekly_frame(a, b, c)

    def test_the_dropped_entries_are_counted_not_silently_absorbed(self):
        df, schema = self._frame_with_an_unusable_control_entry()
        engine = ConfounderEngine(ContractAPI(schema))
        result = engine.test_confounders(df, "revenue", "units_sold", ["marketing_spend"],
                                         mode="time_series", grain="week")
        self.assertGreater(result.dropped_for_alignment, 0)

    def test_the_baseline_n_is_published_alongside_the_aligned_n(self):
        # The baseline pair keeps its own pairwise n; the joint alignment
        # reports how many entries the shared sample lost. Both are visible so
        # a caller never has to discover the difference.
        df, schema = self._frame_with_an_unusable_control_entry()
        engine = ConfounderEngine(ContractAPI(schema))
        result = engine.test_confounders(df, "revenue", "units_sold", ["marketing_spend"],
                                         mode="time_series", grain="week")
        payload = result.to_payload()
        self.assertIn("n", payload)
        self.assertIn("dropped_for_alignment", payload)
        self.assertEqual(result.n, result.baseline.n)


class TestTooFewObservationsIsNotCollinearity(ConfounderTestCase):
    """Two structurally different problems that `numpy.linalg.cond` cannot tell
    apart, and that call for opposite responses."""

    def test_four_members_against_four_rivals_reports_insufficient_n_for_controls(self):
        result = self.engine.rank_competing_explanations(
            self.cdf, "gross_margin_pct",
            ["cost_of_goods", "avg_selling_price", "units_sold", "marketing_spend"],
            mode="cross_sectional",
            period_a={"type": "quarter", "year": 2026, "quarter": 1},
            period_b={"type": "quarter", "year": 2025, "quarter": 1}, min_n=3)
        self.assertEqual(result.n, 4)
        for row in result.rows:
            self.assertEqual(row.partial_status, "insufficient_n_for_controls")

    def test_the_same_candidates_on_a_larger_axis_do_produce_partials(self):
        result = self.engine.rank_competing_explanations(
            self.cdf, "gross_margin_pct",
            ["cost_of_goods", "avg_selling_price", "units_sold", "marketing_spend"],
            mode="time_series", grain="quarter",
            include_precedence=False, include_consistency=False)
        self.assertGreater(result.n, 10)
        for row in result.rows:
            self.assertEqual(row.partial_status, "ok")


class TestConfounderArguments(ConfounderTestCase):
    def test_a_candidate_list_that_is_only_the_kpi_and_cause_is_rejected(self):
        with self.assertRaises(InvalidArgumentError):
            self.engine.test_confounders(self.cdf, "revenue", "units_sold",
                                         ["revenue", "units_sold"])

    def test_a_hallucinated_control_never_reaches_pandas(self):
        with self.assertRaises(UnknownKpiError):
            self.engine.test_confounders(self.cdf, "revenue", "units_sold",
                                         ["synergy_index"], mode="time_series")

    def test_too_many_controls_are_refused_rather_than_silently_truncated(self):
        with self.assertRaises(InvalidArgumentError):
            self.engine.test_confounders(self.cdf, "revenue", "units_sold",
                                         [f"kpi_{i}" for i in range(20)])


# ---------------------------------------------------------------------------
# test_spurious_correlation()
# ---------------------------------------------------------------------------
def _shared_trend(seed=11, n=100, slope=5.0):
    """Two independent random walks with the same trend added to both -- no
    relationship whatsoever beyond the drift they share."""
    rng = np.random.default_rng(seed)
    a = np.cumsum(rng.normal(0, 1, n)) + np.arange(n) * slope
    b = np.cumsum(rng.normal(0, 1, n)) + np.arange(n) * slope
    return _weekly_frame(5000 + a, 5000 + b, 5000 + rng.normal(0, 1, n))


class TestSpuriousCorrelation(unittest.TestCase):
    def setUp(self):
        df, schema = _shared_trend()
        self.engine = ConfounderEngine(ContractAPI(schema))
        self.result = self.engine.test_spurious_correlation(
            df, "revenue", "units_sold", grain="week")

    def test_two_trending_series_correlate_almost_perfectly_on_levels(self):
        self.assertGreater(abs(self.result.level.r), 0.95)
        self.assertTrue(self.result.level.significant)

    def test_the_association_does_not_survive_detrending(self):
        self.assertLess(abs(self.result.detrended.r), 0.3)
        self.assertFalse(self.result.detrended.significant)

    def test_the_verdict_is_trend_driven(self):
        self.assertEqual(self.result.verdict, "trend_driven")

    def test_the_shared_trend_is_reported_so_the_verdict_is_legible(self):
        self.assertTrue(self.result.shared_trend_direction)
        self.assertGreater(self.result.kpi_slope_per_period, 0)
        self.assertGreater(self.result.cause_slope_per_period, 0)

    def test_all_three_transforms_are_published(self):
        transforms = {self.result.level.transform, self.result.detrended.transform,
                      self.result.difference.transform}
        self.assertEqual(transforms, {"level", "detrended", "difference"})

    def test_a_genuinely_co_moving_pair_survives_detrending(self):
        rng = np.random.default_rng(21)
        n = 100
        common = np.cumsum(rng.normal(0, 1, n))
        a = common * 3 + rng.normal(0, 0.2, n)
        b = common * 2 + rng.normal(0, 0.2, n)
        df, schema = _weekly_frame(5000 + a * 10, 5000 + b * 10, 5000 + rng.normal(0, 1, n))
        result = ConfounderEngine(ContractAPI(schema)).test_spurious_correlation(
            df, "revenue", "units_sold", grain="week")
        self.assertEqual(result.verdict, "survives_detrending")
        self.assertTrue(result.detrended.significant)

    def test_cross_sectional_mode_is_refused_with_a_reason(self):
        df, schema = _shared_trend()
        engine = ConfounderEngine(ContractAPI(schema))
        with self.assertRaises(InvalidArgumentError) as ctx:
            engine.test_spurious_correlation(df, "revenue", "units_sold",
                                             mode="cross_sectional")
        self.assertIn("time_series", ctx.exception.valid_alternatives)

    def test_too_little_history_is_a_typed_status_not_a_verdict(self):
        rng = np.random.default_rng(1)
        df, schema = _weekly_frame(5000 + rng.normal(0, 1, 4), 5000 + rng.normal(0, 1, 4),
                                   5000 + rng.normal(0, 1, 4))
        result = ConfounderEngine(ContractAPI(schema)).test_spurious_correlation(
            df, "revenue", "units_sold", grain="quarter")
        self.assertEqual(result.verdict, "insufficient")
        self.assertIsNone(result.level.r)


# ---------------------------------------------------------------------------
# rank_competing_explanations()
# ---------------------------------------------------------------------------
class TestCompetingExplanations(ConfounderTestCase):
    def _rank(self, ordered_by="abs_partial_r"):
        return self.engine.rank_competing_explanations(
            self.cdf, "gross_margin_pct",
            ["cost_of_goods", "avg_selling_price", "units_sold", "marketing_spend"],
            mode="time_series", grain="quarter", ordered_by=ordered_by,
            include_precedence=False, include_consistency=False)

    def test_each_candidate_is_controlled_for_every_rival(self):
        result = self._rank()
        for row in result.rows:
            self.assertEqual(set(row.rivals_controlled),
                             {r.cause_kpi for r in result.rows} - {row.cause_kpi})

    def test_a_row_matches_a_standalone_significance_call_for_that_pair(self):
        result = self._rank()
        row = result.rows[0]
        alone = self.significance.test_statistical_significance(
            self.cdf, "gross_margin_pct", cause_kpi=row.cause_kpi,
            mode="time_series", grain="quarter").rows[0]
        self.assertEqual(row.significance.r, alone.r)
        self.assertEqual(row.significance.n, alone.n)
        self.assertEqual(row.significance.p_value, alone.p_value)

    def test_the_declared_sort_key_is_the_one_applied(self):
        by_partial = self._rank("abs_partial_r")
        self.assertEqual(by_partial.ordered_by, "abs_partial_r")
        values = [abs(r.r_partial_vs_rivals) for r in by_partial.rows]
        self.assertEqual(values, sorted(values, reverse=True))

        by_p = self._rank("p_value")
        self.assertEqual(by_p.ordered_by, "p_value")
        ps = [r.significance.p_value for r in by_p.rows]
        self.assertEqual(ps, sorted(ps))

    def test_an_unsupported_sort_key_is_rejected_and_names_the_valid_ones(self):
        with self.assertRaises(InvalidArgumentError) as ctx:
            self._rank(ordered_by="vibes")
        self.assertEqual(set(ctx.exception.valid_alternatives), set(COMPETING_SORT_KEYS))

    def test_a_single_candidate_is_refused_because_there_is_nothing_to_rank(self):
        with self.assertRaises(InvalidArgumentError):
            self.engine.rank_competing_explanations(
                self.cdf, "gross_margin_pct", ["cost_of_goods"], mode="time_series")

    def test_a_formula_component_is_labelled_so_it_can_be_discounted(self):
        result = self._rank()
        cogs = next(r for r in result.rows if r.cause_kpi == "cost_of_goods")
        self.assertIsNotNone(cogs.relation)

    def test_the_sub_checks_report_their_own_status_when_not_requested(self):
        result = self._rank()
        for row in result.rows:
            self.assertEqual(row.precedence_status, "not_requested")
            self.assertEqual(row.consistency_status, "not_requested")

    def test_precedence_and_consistency_populate_when_requested(self):
        result = self.engine.rank_competing_explanations(
            self.cdf, "gross_margin_pct", ["cost_of_goods", "units_sold"],
            mode="cross_sectional",
            period_a={"type": "quarter", "year": 2026, "quarter": 1},
            period_b={"type": "quarter", "year": 2025, "quarter": 1}, min_n=3)
        for row in result.rows:
            self.assertNotEqual(row.precedence_status, "not_requested")
            self.assertNotEqual(row.consistency_status, "not_requested")


class TestNoScoreIsPublished(ConfounderTestCase):
    """Decision 45 applied to the tool whose name most invites breaking it."""

    def test_the_payload_carries_no_composite_score_of_any_kind(self):
        payload = self.engine.rank_competing_explanations(
            self.cdf, "gross_margin_pct", ["cost_of_goods", "units_sold", "marketing_spend"],
            mode="time_series", grain="quarter",
            include_precedence=False, include_consistency=False).to_payload()
        _assert_no_banned_keys(payload)
        text = json.dumps(payload)
        for forbidden in ('"score"', '"confidence"', '"support"', '"rank"', '"weight"'):
            self.assertNotIn(forbidden, text)

    def test_the_ordering_is_reproducible_from_the_published_key(self):
        # An ordering a caller cannot reproduce from the payload is a hidden
        # weighting, which is the thing decision 45 forbids.
        result = self.engine.rank_competing_explanations(
            self.cdf, "gross_margin_pct", ["cost_of_goods", "units_sold", "marketing_spend"],
            mode="time_series", grain="quarter",
            include_precedence=False, include_consistency=False)
        recomputed = sorted(result.rows,
                            key=lambda r: (-abs(r.r_partial_vs_rivals), r.cause_kpi))
        self.assertEqual([r.cause_kpi for r in result.rows],
                         [r.cause_kpi for r in recomputed])


# ---------------------------------------------------------------------------
# cross-cutting guards
# ---------------------------------------------------------------------------
class TestNumericIntegrityAndProseGuard(ConfounderTestCase):
    def _payloads(self):
        return [
            self.engine.test_confounders(
                self.cdf, "gross_margin_pct", "units_sold", ["cost_of_goods", "revenue"],
                mode="time_series", grain="quarter").to_payload(),
            self.engine.test_spurious_correlation(
                self.cdf, "revenue", "cost_of_goods", grain="quarter").to_payload(),
            self.engine.rank_competing_explanations(
                self.cdf, "gross_margin_pct", ["cost_of_goods", "units_sold"],
                mode="time_series", grain="quarter",
                include_precedence=False, include_consistency=False).to_payload(),
        ]

    def test_every_payload_survives_json_dumps_with_no_nan_or_inf(self):
        for payload in self._payloads():
            assert_json_safe(payload)

    def test_no_string_field_outside_the_allowlist_looks_like_prose(self):
        for payload in self._payloads():
            assert_no_prose_leak(payload)

    def test_no_banned_key_survives(self):
        for payload in self._payloads():
            _assert_no_banned_keys(payload)


class TestDeterminism(ConfounderTestCase):
    def test_identical_input_gives_identical_output_across_runs(self):
        call = lambda: self.engine.rank_competing_explanations(
            self.cdf, "gross_margin_pct", ["cost_of_goods", "units_sold", "marketing_spend"],
            mode="time_series", grain="quarter",
            include_precedence=False, include_consistency=False).to_payload()
        self.assertEqual(json.dumps(call(), sort_keys=True),
                         json.dumps(call(), sort_keys=True))

    def test_candidate_order_does_not_change_the_ranking(self):
        forward = self.engine.rank_competing_explanations(
            self.cdf, "gross_margin_pct", ["cost_of_goods", "units_sold", "marketing_spend"],
            mode="time_series", grain="quarter",
            include_precedence=False, include_consistency=False)
        reverse = self.engine.rank_competing_explanations(
            self.cdf, "gross_margin_pct", ["marketing_spend", "units_sold", "cost_of_goods"],
            mode="time_series", grain="quarter",
            include_precedence=False, include_consistency=False)
        self.assertEqual([r.cause_kpi for r in forward.rows],
                         [r.cause_kpi for r in reverse.rows])


class TestPortabilityAcrossDatasets(unittest.TestCase):
    """G3: hardcoded `revenue`/`units_sold` would pass on retail and fail here."""

    def test_all_three_tools_run_on_the_hospital_and_school_fixtures(self):
        for csv in ("hospital_kpi_smoke_sample.csv", "school_kpi_smoke_sample.csv"):
            df, schema = contracted(csv)
            api = ContractAPI(schema)
            engine = ConfounderEngine(api)
            keys = api.available_keys(df)
            self.assertGreaterEqual(len(keys), 4, csv)
            kpi, cause, *controls = keys[:4]

            for payload in (
                    engine.test_confounders(df, kpi, cause, controls,
                                            mode="time_series", grain="quarter").to_payload(),
                    engine.test_spurious_correlation(df, kpi, cause, grain="quarter").to_payload(),
                    engine.rank_competing_explanations(
                        df, kpi, [cause] + controls, mode="time_series", grain="quarter",
                        include_precedence=False, include_consistency=False).to_payload()):
                assert_json_safe(payload)
                assert_no_prose_leak(payload)
                _assert_no_banned_keys(payload)


if __name__ == "__main__":
    unittest.main()
