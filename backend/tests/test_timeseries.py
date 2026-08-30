"""
`get_timeseries` -- a KPI's history at year / quarter / month / week grain.

`observe.quarterly_series` and `observe.weekly_series` (`engines/observe.py:91,104`)
are the only series builders that have ever existed; neither has a monthly
grain, neither accepts a dimension filter, and neither reports a hole in the
middle of a series as anything other than an absent row. `SeriesEngine.series`
(`agent/series.py`) is a thin wrapper around `query.QueryEngine` -- the same
call `query_kpi` makes for "revenue by quarter" -- so the tests that matter
most here are not about arithmetic (that is `test_query_kpi.py`'s job and this
suite leans on it directly) but about the two things genuinely new: a real
monthly grain, and gaps reported honestly rather than zero-filled or silently
dropped.
"""
from __future__ import annotations

import unittest

import pandas as pd

from app.agent.contract_api import ContractAPI
from app.agent.errors import InvalidArgumentError, UnknownMemberError
from app.agent.query import QueryEngine
from app.agent.series import SERIES_GRAINS, SeriesEngine
from app.engines import metrics
from app.engines.metrics import compute as legacy_compute

from .base import (EngineTestCase, FIXTURES, assert_json_safe,
                   assert_no_prose_leak, contracted)


class SeriesTestCase(EngineTestCase):
    """The shared retail fixture, with a compiled contract so ratio KPIs
    (`gross_margin_pct`, ...) are seriesable."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.rdf, cls.rschema = contracted("business_metrics_sample.csv")
        cls.api = ContractAPI(cls.rschema)
        cls.se = SeriesEngine(cls.api)
        cls.qe = QueryEngine(cls.api)


class TestEachPointMatchesAHandBuiltSlice(SeriesTestCase):
    def test_quarterly_points_match_the_frames_own_year_quarter_mask(self):
        result = self.se.series(self.rdf, "revenue", grain="quarter")
        self.assertTrue(result.points)
        for point in result.points:
            year_str, q_str = point.period.split("-Q")
            mask = ((self.rdf["_year"] == int(year_str))
                    & (self.rdf["_quarter"] == int(q_str)))
            expected = legacy_compute(self.rdf[mask], "revenue", self.rschema.contract_resolver)
            self.assertAlmostEqual(point.value, expected, places=6)
            self.assertEqual(point.rows, int(mask.sum()))


class TestGrainConsistency(SeriesTestCase):
    """For an additive KPI, points at any grain must reconcile to the same
    ungrouped total -- a week can straddle a quarter boundary, so the
    invariant is asserted over the whole series, not period by period."""

    def test_weekly_and_quarterly_sums_reconcile_to_the_ungrouped_total(self):
        weekly = self.se.series(self.rdf, "revenue", grain="week")
        quarterly = self.se.series(self.rdf, "revenue", grain="quarter")
        total = self.qe.query(self.rdf, "revenue", {"type": "all"}).cells[0].values["revenue"]
        self.assertClose(sum(p.value for p in weekly.points), total)
        self.assertClose(sum(p.value for p in quarterly.points), total)

    def test_an_additive_kpi_reports_additive_true(self):
        result = self.se.series(self.rdf, "revenue", grain="quarter")
        self.assertTrue(result.additive)
        self.assertEqual(result.kind, "sum")

    def test_ratio_kpis_are_not_summable_across_grain(self):
        quarterly = self.se.series(self.rdf, "gross_margin_pct", grain="quarter")
        ungrouped = self.qe.query(
            self.rdf, "gross_margin_pct", {"type": "all"}).cells[0].values["gross_margin_pct"]
        summed = sum(p.value for p in quarterly.points)
        self.assertGreater(abs(summed - ungrouped), 1.0)
        self.assertFalse(quarterly.additive)
        self.assertEqual(quarterly.kind, "ratio")


class TestMonthlyGrainIsNowReadable(SeriesTestCase):
    """The first read of `_month` outside `timefilter.MonthRange` -- `prepare`
    has written this column since `_year`/`_quarter` were added and nothing
    has ever consulted it."""

    def test_monthly_points_are_real_values_summing_to_the_quarterly_total(self):
        monthly = self.se.series(self.rdf, "revenue", grain="month")
        quarterly = self.se.series(self.rdf, "revenue", grain="quarter")
        self.assertGreater(len(monthly.points), len(quarterly.points))
        self.assertClose(sum(p.value for p in monthly.points),
                         sum(p.value for p in quarterly.points))
        for point in monthly.points:
            self.assertRegex(point.period, r"^\d{4}-\d{2}$")


class TestChronologicalOrdering(SeriesTestCase):
    def test_every_grain_is_sorted_chronologically(self):
        for grain in SERIES_GRAINS:
            with self.subTest(grain=grain):
                result = self.se.series(self.rdf, "revenue", grain=grain)
                periods = [p.period for p in result.points]
                self.assertEqual(periods, sorted(periods))


class TestGaps(unittest.TestCase):
    """A hole in the middle of the series must be reported, never invented as
    a zero -- the `metrics.prepare` `fillna(0.0)` ambiguity this batch must
    not reintroduce one layer up."""

    @classmethod
    def setUpClass(cls):
        raw = pd.DataFrame({
            "date": ["2024-02-01", "2024-08-01"],   # Q1 present, Q2 absent, Q3 present
            "revenue": [100.0, 140.0],
        })
        cls.schema = metrics.detect_schema(raw)
        cls.df = metrics.prepare(raw, cls.schema)
        cls.se = SeriesEngine(ContractAPI(cls.schema))

    def test_a_quarter_with_no_rows_is_reported_as_a_gap_not_a_zero_point(self):
        result = self.se.series(self.df, "revenue", grain="quarter")
        self.assertEqual([p.period for p in result.points], ["2024-Q1", "2024-Q3"])
        self.assertEqual(result.gaps, ("2024-Q2",))
        for point in result.points:
            self.assertNotEqual(point.value, 0.0)

    def test_a_period_outside_the_covered_span_is_not_a_series_gap(self):
        """The wider request itself being only partly covered is
        `TimeSelection.missing`'s job, not `gaps`'s -- the two must stay
        distinct concepts."""
        result = self.se.series(
            self.df, "revenue", grain="quarter",
            time_filter={"type": "range", "start": "2023-01-01", "end": "2025-12-31"})
        self.assertEqual(result.gaps, ("2024-Q2",))
        self.assertTrue(result.selection.missing)


class TestFiltersAndAirlock(SeriesTestCase):
    def test_filters_narrow_the_series_the_same_way_query_kpi_does(self):
        filtered = self.se.series(self.rdf, "revenue", grain="quarter",
                                  filters={"region": "North"})
        expected = self.qe.query(self.rdf, ["revenue"], {"type": "all"},
                                 filters={"region": "North"}, group_by=["_period"])
        self.assertEqual(len(filtered.points), len(expected.cells))
        for point, cell in zip(filtered.points, expected.cells):
            self.assertAlmostEqual(point.value, cell.values["revenue"], places=6)

    def test_an_unknown_member_raises_the_same_error_query_kpi_does(self):
        with self.assertRaises(UnknownMemberError):
            self.se.series(self.rdf, "revenue", grain="quarter", filters={"region": "Nowhere"})

    def test_an_unknown_grain_is_rejected_naming_the_valid_ones(self):
        with self.assertRaises(InvalidArgumentError) as ctx:
            self.se.series(self.rdf, "revenue", grain="fortnight")
        self.assertEqual(ctx.exception.to_payload()["valid_alternatives"], list(SERIES_GRAINS))


class TestNumericIntegrityAndProseGuard(SeriesTestCase):
    def test_the_payload_survives_json_dumps_with_no_nan_or_inf(self):
        result = self.se.series(self.rdf, "gross_margin_pct", grain="quarter")
        assert_json_safe(result.to_payload())

    def test_no_string_field_outside_the_allowlist_looks_like_prose(self):
        result = self.se.series(self.rdf, "revenue", grain="quarter")
        assert_no_prose_leak(result.to_payload())


class TestDeterminism(SeriesTestCase):
    def test_identical_input_gives_identical_output_across_runs(self):
        first = self.se.series(self.rdf, "revenue", grain="quarter").to_payload()
        second = self.se.series(self.rdf, "revenue", grain="quarter").to_payload()
        self.assertEqual(first, second)


class TestPortabilityAcrossDatasets(unittest.TestCase):
    """Catches hardcoded retail vocabulary the way the hospital fixture caught
    it for the resolver-propagation bug (G3)."""

    @classmethod
    def setUpClass(cls):
        cls.df, cls.schema = contracted("hospital_sample.csv", FIXTURES)
        cls.se = SeriesEngine(ContractAPI(cls.schema))
        assert "recovery_rate" in cls.schema.available_kpis

    def test_a_contract_only_ratio_kpi_is_seriesable_at_every_grain(self):
        for grain in SERIES_GRAINS:
            with self.subTest(grain=grain):
                result = self.se.series(self.df, "recovery_rate", grain=grain)
                self.assertTrue(result.points)
                self.assertEqual(result.kind, "ratio")


if __name__ == "__main__":
    unittest.main()
