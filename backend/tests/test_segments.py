"""
`compare_segments` / `compare_cohorts` / `segment_by_behavior` -- `agent/segments.py`.

The property that matters most for `compare_segments` is antisymmetry, and it
is a deliberately partial property: `gap_abs` and `relative_gap_pct` must
negate exactly under swapping `member_a`/`member_b`, and `a_vs_b_pct` must
NOT -- both halves are pinned here (`TestGapAntisymmetry`), so the asymmetric
field cannot silently start looking symmetric (or vice versa) without a test
failing.

For `compare_cohorts`, what matters is that overlap is rejected on the actual
row-set intersection -- two filters that share no key can still share rows,
and that must be caught (`TestCohortOverlapRejection`).

For `segment_by_behavior`, what matters is that band *classification* never
uses a fabricated zero baseline (an `appeared` member is never `grew`), while
the reconciliation invariant -- summed `change_abs` across every band equals
the KPI's real total change for an additive KPI -- holds anyway, because the
dollar delta (not the band) is what is computed against an implicit zero
(`TestReconciliationInvariant`).
"""
from __future__ import annotations

import unittest

import pandas as pd

from app.agent.contract_api import ContractAPI
from app.agent.dimensions import MemberCatalogue
from app.agent.errors import InvalidArgumentError, UnknownDimensionError, UnknownMemberError
from app.agent.segments import SegmentEngine
from app.engines import metrics

from .base import (EngineTestCase, FIXTURES, assert_json_safe,
                   assert_no_prose_leak, contracted)

BANNED = {"interpretation", "detail", "narrative", "note"}


def _assert_no_banned_keys(payload, path: str = "$") -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            assert key not in BANNED, f"{path}.{key} is a banned prose field"
            _assert_no_banned_keys(value, f"{path}.{key}")
    elif isinstance(payload, list):
        for i, item in enumerate(payload):
            _assert_no_banned_keys(item, f"{path}[{i}]")


def _prepared_with_catalogue(raw: pd.DataFrame):
    """
    `detect_schema`/`prepare` alone (what every bare synthetic fixture in
    this suite otherwise uses) leaves `schema.member_catalogue` unset, so
    `ContractAPI.resolve_member` falls back to indexing whatever frame it was
    handed -- correct for `compare_segments`/`compare_cohorts`'s own filter
    resolution against a *time-narrowed* frame, but wrong for a test that
    wants "member absent from this window" to mean a real empty result
    rather than `UnknownMemberError`. `dataset_service.load`
    (`services/dataset_service.py:396`) attaches a dataset-wide catalogue at
    load time in production; this mirrors that exactly, off the full frame,
    so a member's canonical spelling resolves the same regardless of which
    period a later call narrows to.
    """
    schema = metrics.detect_schema(raw)
    df = metrics.prepare(raw, schema)
    schema.member_catalogue = MemberCatalogue(df, schema.dimensions)
    return df, schema


class SegmentsTestCase(EngineTestCase):
    """The shared retail fixture, with a compiled contract."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.rdf, cls.rschema = contracted("business_metrics_sample.csv")
        cls.api = ContractAPI(cls.rschema)
        cls.se = SegmentEngine(cls.api)
        cls.period_a = {"type": "quarter", "year": 2026, "quarter": 2}
        cls.period_b = {"type": "quarter", "year": 2026, "quarter": 1}


# ---------------------------------------------------------------------------
# compare_segments() -- antisymmetry
# ---------------------------------------------------------------------------
class TestGapAntisymmetry(SegmentsTestCase):
    def test_gap_abs_and_relative_gap_negate_exactly_under_swap(self):
        forward = self.se.compare_segments(self.rdf, "revenue", "region", "North", "South",
                                           {"type": "all"})
        backward = self.se.compare_segments(self.rdf, "revenue", "region", "South", "North",
                                            {"type": "all"})
        self.assertAlmostEqual(forward.gap.gap_abs, -backward.gap.gap_abs, places=6)
        self.assertAlmostEqual(forward.gap.relative_gap_pct, -backward.gap.relative_gap_pct, places=6)

    def test_direction_flips_under_swap(self):
        forward = self.se.compare_segments(self.rdf, "revenue", "region", "North", "South",
                                           {"type": "all"})
        backward = self.se.compare_segments(self.rdf, "revenue", "region", "South", "North",
                                            {"type": "all"})
        self.assertEqual(forward.gap.direction, "a_higher")
        self.assertEqual(backward.gap.direction, "b_higher")

    def test_a_vs_b_pct_is_deliberately_not_antisymmetric(self):
        forward = self.se.compare_segments(self.rdf, "revenue", "region", "North", "South",
                                           {"type": "all"})
        backward = self.se.compare_segments(self.rdf, "revenue", "region", "South", "North",
                                            {"type": "all"})
        # If it happened to be antisymmetric these would be equal; assert they
        # are not, so the asymmetry is pinned rather than assumed.
        self.assertNotAlmostEqual(forward.gap.a_vs_b_pct, -backward.gap.a_vs_b_pct, places=6)


class TestGapUndeterminedWhenASideIsMissing(unittest.TestCase):
    def test_both_present_but_one_side_has_no_rows_this_window_is_member_absent(self):
        # South is a real, known region -- it simply has no rows in Q2. North
        # anchors Q2 so the window itself is non-empty.
        raw = pd.DataFrame({
            "region": ["North", "North", "South"],
            "revenue": [100.0, 50.0, 200.0],
            "date": ["2026-05-01", "2026-02-01", "2026-02-01"],
        })
        df, schema = _prepared_with_catalogue(raw)
        result = SegmentEngine(ContractAPI(schema)).compare_segments(
            df, "revenue", "region", "North", "South", {"type": "quarter", "year": 2026, "quarter": 2})
        self.assertEqual(result.status, "member_absent_b")
        # `value` carries a raw NaN at the dataclass level (the same
        # Optional[float]-plus-`safe()` convention `decompose.py` and
        # `breakdown.py` use); `to_payload()` is the boundary where it
        # becomes a real `null`.
        self.assertIsNone(result.b.to_payload()["value"])
        self.assertEqual(result.gap.direction, "undetermined")


class TestUnknownMemberIsRejected(SegmentsTestCase):
    def test_a_member_not_in_the_dimension_at_all_raises_unknown_member(self):
        with self.assertRaises(UnknownMemberError):
            self.se.compare_segments(self.rdf, "revenue", "region", "Mars", "South", {"type": "all"})

    def test_the_same_member_twice_is_rejected(self):
        with self.assertRaises(InvalidArgumentError):
            self.se.compare_segments(self.rdf, "revenue", "region", "North", "North", {"type": "all"})

    def test_a_time_grain_column_as_the_dimension_is_rejected(self):
        with self.assertRaises(InvalidArgumentError):
            self.se.compare_segments(self.rdf, "revenue", "_period", "North", "South", {"type": "all"})


class TestTwoPeriodDivergence(SegmentsTestCase):
    def test_divergence_abs_equals_the_independently_computed_delta_of_deltas(self):
        result = self.se.compare_segments(self.rdf, "revenue", "region", "North", "South",
                                          self.period_a, self.period_b)
        self.assertAlmostEqual(result.divergence_abs,
                               result.a.change_abs - result.b.change_abs, places=6)


# ---------------------------------------------------------------------------
# compare_cohorts()
# ---------------------------------------------------------------------------
class TestCohortOverlapRejection(SegmentsTestCase):
    def test_cohorts_sharing_rows_via_different_dimensions_are_rejected(self):
        with self.assertRaises(InvalidArgumentError) as ctx:
            self.se.compare_cohorts(self.rdf, "revenue",
                                    {"region": ["North"]}, {"segment": ["Enterprise"]},
                                    {"type": "all"})
        self.assertIn("overlap", ctx.exception.to_payload()["reason"])

    def test_identical_filters_are_rejected(self):
        with self.assertRaises(InvalidArgumentError):
            self.se.compare_cohorts(self.rdf, "revenue",
                                    {"region": ["North"]}, {"region": ["North"]}, {"type": "all"})

    def test_disjoint_cohorts_succeed(self):
        result = self.se.compare_cohorts(self.rdf, "revenue",
                                         {"region": ["North"]}, {"region": ["South"]},
                                         {"type": "all"})
        self.assertEqual(result.status, "ok")
        self.assertGreater(result.a.value, 0)
        self.assertGreater(result.b.value, 0)

    def test_an_empty_but_valid_cohort_is_a_result_not_an_error(self):
        raw = pd.DataFrame({
            "region": ["North", "South"],
            "revenue": [100.0, 200.0],
            "date": ["2026-05-01", "2026-02-01"],
        })
        # South is a real, known region -- it simply has no rows in Q2, the
        # requested window. A dataset-wide catalogue (as `dataset_service`
        # attaches in production) is needed for that to resolve as a valid,
        # empty cohort rather than `UnknownMemberError` against the
        # Q2-narrowed frame alone.
        df, schema = _prepared_with_catalogue(raw)
        result = SegmentEngine(ContractAPI(schema)).compare_cohorts(
            df, "revenue", {"region": ["North"]}, {"region": ["South"]},
            {"type": "quarter", "year": 2026, "quarter": 2})
        self.assertEqual(result.status, "cohort_b_empty")
        self.assertFalse(result.b.present)
        self.assertIsNone(result.b.to_payload()["value"])


class TestCohortGapMatchesSegmentGapShape(SegmentsTestCase):
    def test_cohort_gap_abs_is_antisymmetric_too(self):
        forward = self.se.compare_cohorts(self.rdf, "revenue",
                                          {"region": ["North"]}, {"region": ["South"]}, {"type": "all"})
        backward = self.se.compare_cohorts(self.rdf, "revenue",
                                           {"region": ["South"]}, {"region": ["North"]}, {"type": "all"})
        self.assertAlmostEqual(forward.gap.gap_abs, -backward.gap.gap_abs, places=6)


# ---------------------------------------------------------------------------
# segment_by_behavior()
# ---------------------------------------------------------------------------
class TestBoundaryClassification(unittest.TestCase):
    """A member at exactly +t, -t, and just under +t must land in the right
    band -- the closed-on-the-threshold rule stated in the module docstring."""

    def _result(self, cur_val, base_val, threshold=10.0):
        raw = pd.DataFrame({
            "region": ["Fixed", "Fixed", "Test", "Test"],
            "revenue": [1000.0, 1000.0, cur_val, base_val],
            "date": ["2026-05-01", "2026-02-01", "2026-05-01", "2026-02-01"],
        })
        schema = metrics.detect_schema(raw)
        df = metrics.prepare(raw, schema)
        result = SegmentEngine(ContractAPI(schema)).segment_by_behavior(
            df, "revenue", "region",
            {"type": "quarter", "year": 2026, "quarter": 2},
            {"type": "quarter", "year": 2026, "quarter": 1}, threshold_pct=threshold)
        return next(m for m in result.members if m.member == "Test")

    def test_exactly_plus_threshold_is_grew(self):
        member = self._result(110.0, 100.0, threshold=10.0)
        self.assertEqual(member.band, "grew")

    def test_exactly_minus_threshold_is_declined(self):
        member = self._result(90.0, 100.0, threshold=10.0)
        self.assertEqual(member.band, "declined")

    def test_just_under_threshold_is_flat(self):
        member = self._result(109.9, 100.0, threshold=10.0)
        self.assertEqual(member.band, "flat")


class TestAppearedAndDisappearedNeverFabricatePercent(unittest.TestCase):
    def test_a_member_present_only_in_current_is_appeared_never_grew(self):
        # "Fixed" anchors both quarters so neither time window is empty;
        # "New" has rows only in the current quarter.
        raw = pd.DataFrame({
            "region": ["Fixed", "Fixed", "New"],
            "revenue": [1000.0, 1000.0, 500.0],
            "date": ["2026-05-01", "2026-02-01", "2026-05-01"],
        })
        schema = metrics.detect_schema(raw)
        df = metrics.prepare(raw, schema)
        result = SegmentEngine(ContractAPI(schema)).segment_by_behavior(
            df, "revenue", "region",
            {"type": "quarter", "year": 2026, "quarter": 2},
            {"type": "quarter", "year": 2026, "quarter": 1})
        new_member = next(m for m in result.members if m.member == "New")
        self.assertEqual(new_member.band, "appeared")
        self.assertIsNone(new_member.change_pct)
        self.assertEqual(new_member.change_abs, 500.0)   # dollar delta IS well-defined

    def test_a_member_present_only_in_baseline_is_disappeared(self):
        raw = pd.DataFrame({
            "region": ["Fixed", "Fixed", "Gone"],
            "revenue": [1000.0, 1000.0, 300.0],
            "date": ["2026-05-01", "2026-02-01", "2026-02-01"],
        })
        schema = metrics.detect_schema(raw)
        df = metrics.prepare(raw, schema)
        result = SegmentEngine(ContractAPI(schema)).segment_by_behavior(
            df, "revenue", "region",
            {"type": "quarter", "year": 2026, "quarter": 2},
            {"type": "quarter", "year": 2026, "quarter": 1})
        gone_member = next(m for m in result.members if m.member == "Gone")
        self.assertEqual(gone_member.band, "disappeared")
        self.assertIsNone(gone_member.change_pct)
        self.assertEqual(gone_member.change_abs, -300.0)


class TestZeroBaselineIsUndetermined(unittest.TestCase):
    def test_a_real_zero_baseline_with_nonzero_current_is_undetermined(self):
        raw = pd.DataFrame({
            "region": ["North", "North"],
            "revenue": [500.0, 0.0],
            "date": ["2026-05-01", "2026-02-01"],
        })
        schema = metrics.detect_schema(raw)
        df = metrics.prepare(raw, schema)
        result = SegmentEngine(ContractAPI(schema)).segment_by_behavior(
            df, "revenue", "region",
            {"type": "quarter", "year": 2026, "quarter": 2},
            {"type": "quarter", "year": 2026, "quarter": 1})
        member = next(m for m in result.members if m.member == "North")
        self.assertEqual(member.band, "undetermined")
        self.assertIsNone(member.change_pct)


class TestReconciliationInvariant(SegmentsTestCase):
    def test_summed_change_abs_across_every_band_equals_the_total_change_for_an_additive_kpi(self):
        result = self.se.segment_by_behavior(self.rdf, "revenue", "product",
                                             self.period_a, self.period_b)
        self.assertTrue(result.additive)
        band_sum = sum(m.change_abs for m in result.members if m.change_abs is not None)
        self.assertAlmostEqual(band_sum, result.total_change_abs, places=2)

    def test_change_abs_is_none_for_a_non_additive_kpi(self):
        result = self.se.segment_by_behavior(self.rdf, "gross_margin_pct", "product",
                                             self.period_a, self.period_b)
        self.assertFalse(result.additive)
        self.assertIsNone(result.total_change_abs)
        for m in result.members:
            if m.band in ("appeared", "disappeared"):
                self.assertIsNone(m.change_abs)
        for band in result.bands.values():
            self.assertIsNone(band.current)
            self.assertIsNone(band.change_abs)


class TestPolarityAwareFavourability(unittest.TestCase):
    def test_growth_in_a_lower_is_better_kpi_counts_as_unfavourable(self):
        raw = pd.DataFrame({
            "region": ["North", "North"],
            "marketing_spend": [200.0, 100.0],
            "date": ["2026-05-01", "2026-02-01"],
        })
        schema = metrics.detect_schema(raw)
        df = metrics.prepare(raw, schema)
        result = SegmentEngine(ContractAPI(schema)).segment_by_behavior(
            df, "marketing_spend", "region",
            {"type": "quarter", "year": 2026, "quarter": 2},
            {"type": "quarter", "year": 2026, "quarter": 1})
        member = next(m for m in result.members if m.member == "North")
        self.assertEqual(member.band, "grew")
        self.assertTrue(member.is_unfavourable)
        self.assertEqual(result.bands["grew"].unfavourable_count, 1)
        self.assertEqual(result.bands["grew"].favourable_count, 0)


class TestBehaviorAirlock(SegmentsTestCase):
    def test_a_time_grain_column_as_the_dimension_is_rejected(self):
        with self.assertRaises(InvalidArgumentError):
            self.se.segment_by_behavior(self.rdf, "revenue", "_period",
                                        self.period_a, self.period_b)

    def test_a_negative_threshold_is_rejected(self):
        with self.assertRaises(InvalidArgumentError):
            self.se.segment_by_behavior(self.rdf, "revenue", "region",
                                        self.period_a, self.period_b, threshold_pct=-1.0)

    def test_default_threshold_matches_min_material_change_pct(self):
        from app.config import get_settings
        result = self.se.segment_by_behavior(self.rdf, "revenue", "region",
                                             self.period_a, self.period_b)
        self.assertEqual(result.threshold_pct, get_settings().min_material_change_pct)


# ---------------------------------------------------------------------------
# integrity / prose guard / determinism / portability
# ---------------------------------------------------------------------------
class TestNumericIntegrityAndProseGuard(SegmentsTestCase):
    def test_the_payload_survives_json_dumps_with_no_nan_or_inf(self):
        seg = self.se.compare_segments(self.rdf, "revenue", "region", "North", "South",
                                       self.period_a, self.period_b)
        cohort = self.se.compare_cohorts(self.rdf, "revenue",
                                         {"region": ["North"]}, {"region": ["South"]},
                                         self.period_a, self.period_b)
        behavior = self.se.segment_by_behavior(self.rdf, "revenue", "product",
                                               self.period_a, self.period_b)
        assert_json_safe(seg.to_payload())
        assert_json_safe(cohort.to_payload())
        assert_json_safe(behavior.to_payload())

    def test_no_string_field_outside_the_allowlist_looks_like_prose(self):
        seg = self.se.compare_segments(self.rdf, "revenue", "region", "North", "South",
                                       self.period_a, self.period_b)
        behavior = self.se.segment_by_behavior(self.rdf, "revenue", "product",
                                               self.period_a, self.period_b)
        assert_no_prose_leak(seg.to_payload())
        assert_no_prose_leak(behavior.to_payload())

    def test_no_banned_prose_key_survives(self):
        seg = self.se.compare_segments(self.rdf, "revenue", "region", "North", "South",
                                       self.period_a, self.period_b)
        behavior = self.se.segment_by_behavior(self.rdf, "revenue", "product",
                                               self.period_a, self.period_b)
        _assert_no_banned_keys(seg.to_payload())
        _assert_no_banned_keys(behavior.to_payload())


class TestDeterminism(SegmentsTestCase):
    def test_identical_input_gives_identical_output_across_runs(self):
        first = self.se.segment_by_behavior(self.rdf, "revenue", "product",
                                            self.period_a, self.period_b).to_payload()
        second = self.se.segment_by_behavior(self.rdf, "revenue", "product",
                                             self.period_a, self.period_b).to_payload()
        self.assertEqual(first, second)


class TestPortabilityAcrossDatasets(unittest.TestCase):
    def test_hospital_department_comparison_works_on_a_contract_only_ratio_kpi(self):
        df, schema = contracted("hospital_sample.csv", FIXTURES)
        periods = sorted(df["_period"].dropna().unique().tolist())
        pa = {"type": "quarter", "year": int(periods[-1].split("-Q")[0]),
              "quarter": int(periods[-1][-1])}
        pb = {"type": "quarter", "year": int(periods[0].split("-Q")[0]),
              "quarter": int(periods[0][-1])}
        se = SegmentEngine(ContractAPI(schema))
        depts = sorted(df["department"].dropna().unique().tolist())
        result = se.compare_segments(df, "recovery_rate", "department", depts[0], depts[1], pa, pb)
        self.assertIn(result.status, ("ok", "member_absent_a", "member_absent_b", "both_absent"))

    def test_school_campus_behavior_segmentation_does_not_crash(self):
        df, schema = contracted("school_kpi_smoke_sample.csv")
        se = SegmentEngine(ContractAPI(schema))
        result = se.segment_by_behavior(df, "students_enrolled", "campus",
                                        {"type": "quarter", "year": 2026, "quarter": 2},
                                        {"type": "quarter", "year": 2025, "quarter": 2})
        self.assertTrue(result.bands)


if __name__ == "__main__":
    unittest.main()
