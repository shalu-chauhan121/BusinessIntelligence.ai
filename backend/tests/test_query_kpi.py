"""
`query_kpi` -- the generic retrieval engine that answers Tier 1 and Tier 2 of
the question taxonomy: a scalar, a range, a predicate set, a group-by
breakdown, a top-N ranking, all through one call shape built on `TimeFilter`
(`agent/timefilter.py`) and `ContractAPI` (`agent/contract_api.py`).

The flagship assertion is `TestTheFlagshipQuestion`: "total revenue in all
odd-numbered years" had no code path before this batch. The ratio tests are
the ones that matter most for correctness -- `query_kpi` is the easiest place
in the whole board to reintroduce the C2 bug (a grouped ratio silently
averaging member ratios instead of summing components and dividing once),
and it is asserted directly against that failure mode, not just against a
passing case.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from app.agent.contract_api import ContractAPI
from app.agent.errors import (EmptyPeriodError, InvalidArgumentError,
                              MalformedTimeFilterError, UnknownDimensionError,
                              UnknownKpiError, UnknownMemberError)
from app.agent.query import QueryEngine
from app.engines import metrics
from app.engines.metrics import compute as legacy_compute

from .base import (EngineTestCase,
                   assert_json_safe as _assert_json_safe,
                   assert_no_prose_leak as _assert_no_prose_leak,
                   contracted as _contracted)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SAMPLES = Path(__file__).resolve().parents[2] / "sample_data"


class RetailQueryTestCase(EngineTestCase):
    """The shared retail fixture, with a compiled contract so ratio KPIs
    (`gross_margin_pct`, `avg_selling_price`, ...) are queryable."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.rdf, cls.rschema = _contracted("business_metrics_sample.csv")
        cls.api = ContractAPI(cls.rschema)
        cls.qe = QueryEngine(cls.api)


class TestTheScalarCaseMatchesLegacyCompute(RetailQueryTestCase):
    def test_group_by_empty_is_the_scalar_getter(self):
        result = self.qe.query(self.rdf, "revenue", {"type": "year", "year": 2024})
        expected = legacy_compute(
            self.rdf[self.rdf["_year"] == 2024], "revenue", self.rschema.contract_resolver)
        self.assertEqual(len(result.cells), 1)
        self.assertEqual(result.cells[0].group, ())
        self.assertAlmostEqual(result.cells[0].values["revenue"], expected, places=6)


class TestTheFlagshipQuestion(RetailQueryTestCase):
    """'Total revenue in all odd-numbered years' -- no code path existed for
    this before `TimeFilter` + `query_kpi`."""

    def test_odd_years_grouped_by_year_matches_hand_built_slices_and_sums_to_the_total(self):
        result = self.qe.query(
            self.rdf, ["revenue"], {"type": "years", "values": [2023, 2025]},
            group_by=["_year"])
        self.assertEqual(result.total_groups, 2)
        self.assertFalse(result.truncated)
        by_year = {dict(c.group)["_year"]: c.values["revenue"] for c in result.cells}
        for year in ("2023", "2025"):
            expected = legacy_compute(
                self.rdf[self.rdf["_year"] == int(year)], "revenue",
                self.rschema.contract_resolver)
            self.assertAlmostEqual(by_year[year], expected, places=6)

        ungrouped = self.qe.query(
            self.rdf, ["revenue"], {"type": "years", "values": [2023, 2025]})
        self.assertAlmostEqual(
            sum(by_year.values()), ungrouped.cells[0].values["revenue"], places=6)


class TestRatioKpisAggregateCorrectlyPerGroup(RetailQueryTestCase):
    """The C2 invariant: sum numerator, sum denominator, divide once -- per
    group. `query_kpi` must not regress into averaging member ratios."""

    def test_each_groups_ratio_is_computed_from_that_groups_own_rows(self):
        result = self.qe.query(
            self.rdf, ["gross_margin_pct"], {"type": "all"}, group_by=["region"])
        for cell in result.cells:
            region = dict(cell.group)["region"]
            expected = legacy_compute(
                self.rdf[self.rdf["region"] == region], "gross_margin_pct",
                self.rschema.contract_resolver)
            self.assertAlmostEqual(cell.values["gross_margin_pct"], expected, places=6)

    def test_the_ungrouped_ratio_is_not_the_mean_of_the_grouped_ratios(self):
        """A regression to mean-of-member-ratios would pass every other test
        in this file while being quietly wrong -- this is the one that catches it."""
        grouped = self.qe.query(
            self.rdf, ["gross_margin_pct"], {"type": "all"}, group_by=["region"])
        ungrouped = self.qe.query(self.rdf, ["gross_margin_pct"], {"type": "all"})
        mean_of_cells = (sum(c.values["gross_margin_pct"] for c in grouped.cells)
                         / len(grouped.cells))
        self.assertGreater(abs(ungrouped.cells[0].values["gross_margin_pct"] - mean_of_cells),
                           1e-4)


class TestFilters(RetailQueryTestCase):
    def test_a_multi_value_filter_equals_the_union_of_single_value_queries(self):
        combined = self.qe.query(
            self.rdf, "revenue", {"type": "all"}, filters={"region": ["north", "south"]})
        north = self.qe.query(self.rdf, "revenue", {"type": "all"}, filters={"region": "North"})
        south = self.qe.query(self.rdf, "revenue", {"type": "all"}, filters={"region": "South"})
        self.assertAlmostEqual(
            combined.cells[0].values["revenue"],
            north.cells[0].values["revenue"] + south.cells[0].values["revenue"], places=6)

    def test_a_member_named_in_any_case_resolves_to_the_frames_own_spelling(self):
        result = self.qe.query(self.rdf, "revenue", {"type": "all"}, filters={"region": "north"})
        self.assertEqual(result.filters["region"], ("North",))

    def test_an_unknown_member_raises_naming_real_alternatives(self):
        with self.assertRaises(UnknownMemberError) as ctx:
            self.qe.query(self.rdf, "revenue", {"type": "all"}, filters={"region": "Nowhere"})
        payload = ctx.exception.to_payload()
        self.assertEqual(payload["requested"], "Nowhere")
        self.assertTrue(payload["valid_alternatives"])


class TestReconciliation(RetailQueryTestCase):
    def test_n_way_groupby_totals_reconcile_to_the_ungrouped_total(self):
        grouped = self.qe.query(
            self.rdf, ["revenue"], {"type": "all"}, group_by=["region", "product"])
        ungrouped = self.qe.query(self.rdf, ["revenue"], {"type": "all"})
        total = sum(c.values["revenue"] for c in grouped.cells)
        self.assertAlmostEqual(total, ungrouped.cells[0].values["revenue"], places=6)


class TestDeterminism(RetailQueryTestCase):
    def test_identical_input_gives_identical_output_across_runs(self):
        first = self.qe.query(self.rdf, ["revenue"], {"type": "all"}, group_by=["region"])
        second = self.qe.query(self.rdf, ["revenue"], {"type": "all"}, group_by=["region"])
        self.assertEqual(
            [c.to_payload() for c in first.cells], [c.to_payload() for c in second.cells])

    def test_a_row_shuffled_frame_gives_the_same_groups_in_the_same_order(self):
        """Row order must not affect which groups appear, their order, or
        their rows -- floating-point summation order can move the last digit
        of a sum (`11454963.530000001` vs `11454963.53`), which is a pandas
        property, not a determinism bug, so values are compared with tolerance."""
        shuffled = self.rdf.sample(frac=1, random_state=7)
        ordered = self.qe.query(self.rdf, ["revenue"], {"type": "all"}, group_by=["region"])
        reshuffled = self.qe.query(shuffled, ["revenue"], {"type": "all"}, group_by=["region"])
        self.assertEqual([c.group for c in ordered.cells], [c.group for c in reshuffled.cells])
        self.assertEqual([c.rows for c in ordered.cells], [c.rows for c in reshuffled.cells])
        for a, b in zip(ordered.cells, reshuffled.cells):
            self.assertAlmostEqual(a.values["revenue"], b.values["revenue"], places=6)


class TestSortingAndLimiting(RetailQueryTestCase):
    def test_sort_by_kpi_descending_is_a_top_n_ranking(self):
        result = self.qe.query(
            self.rdf, ["revenue"], {"type": "all"}, group_by=["product"],
            sort_by="revenue", order="desc")
        values = [c.values["revenue"] for c in result.cells]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_sort_by_a_group_by_dimension_orders_alphabetically(self):
        result = self.qe.query(
            self.rdf, ["revenue"], {"type": "all"}, group_by=["product"],
            sort_by="product", order="asc")
        members = [dict(c.group)["product"] for c in result.cells]
        self.assertEqual(members, sorted(members))

    def test_limit_caps_output_but_reports_the_true_total(self):
        result = self.qe.query(
            self.rdf, ["revenue"], {"type": "all"}, group_by=["region", "product"], limit=2)
        self.assertEqual(len(result.cells), 2)
        self.assertTrue(result.truncated)
        self.assertGreater(result.total_groups, 2)


class TestTheAirlockRejectsBeforeAnyAggregation(RetailQueryTestCase):
    def test_a_hallucinated_kpi_key_is_rejected(self):
        with self.assertRaises(UnknownKpiError) as ctx:
            self.qe.query(self.rdf, "not_a_real_kpi", {"type": "all"})
        self.assertTrue(ctx.exception.to_payload()["valid_alternatives"])

    def test_an_unknown_dimension_in_group_by_is_rejected(self):
        with self.assertRaises(UnknownDimensionError):
            self.qe.query(self.rdf, "revenue", {"type": "all"}, group_by=["not_a_dimension"])

    def test_an_unknown_dimension_in_filters_is_rejected(self):
        with self.assertRaises(UnknownDimensionError):
            self.qe.query(self.rdf, "revenue", {"type": "all"},
                          filters={"not_a_dimension": "x"})

    def test_a_malformed_time_filter_is_rejected(self):
        with self.assertRaises(MalformedTimeFilterError):
            self.qe.query(self.rdf, "revenue", {"type": "quarter", "year": 2024, "quarter": 9})

    def test_a_sort_by_naming_neither_a_kpi_nor_a_group_by_column_is_rejected(self):
        with self.assertRaises(InvalidArgumentError):
            self.qe.query(self.rdf, "revenue", {"type": "all"}, sort_by="not_requested")

    def test_a_period_the_dataset_holds_none_of_is_rejected(self):
        with self.assertRaises(EmptyPeriodError):
            self.qe.query(self.rdf, "revenue", {"type": "year", "year": 2099})


class TestZeroVersusMissingAreDistinguishable(unittest.TestCase):
    """A genuinely zero metric must not be confused with a query that matched
    no rows at all -- the `fillna(0.0)` artefact this whole batch must not
    reintroduce at a new boundary."""

    @classmethod
    def setUpClass(cls):
        raw = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "region": ["North", "South"],
            "product": ["A", "B"],
            "revenue": [0.0, 500.0],
        })
        cls.schema = metrics.detect_schema(raw)
        cls.df = metrics.prepare(raw, cls.schema)
        cls.qe = QueryEngine(ContractAPI(cls.schema))

    def test_a_row_whose_revenue_is_genuinely_zero_reports_a_real_zero(self):
        result = self.qe.query(self.df, "revenue", {"type": "all"}, filters={"region": "North"})
        self.assertEqual(result.cells[0].rows, 1)
        self.assertEqual(result.cells[0].values["revenue"], 0.0)

    def test_a_filter_combination_matching_no_rows_reports_none_not_zero(self):
        """'North' and 'B' each exist in the frame individually -- neither
        filter alone can empty it -- but no row holds both, so the *combined*
        filter is the only way to reach a genuine zero-row, non-error result."""
        result = self.qe.query(
            self.df, "revenue", {"type": "all"},
            filters={"region": "North", "product": "B"})
        self.assertEqual(result.cells[0].rows, 0)
        self.assertIsNone(result.cells[0].to_payload()["values"]["revenue"])


class TestNumericIntegrityAndProseGuard(RetailQueryTestCase):
    def test_the_payload_survives_json_dumps_with_no_nan_or_inf(self):
        result = self.qe.query(
            self.rdf, ["gross_margin_pct"], {"type": "all"}, group_by=["region"])
        _assert_json_safe(result.to_payload())

    def test_no_string_field_outside_the_allowlist_looks_like_prose(self):
        result = self.qe.query(
            self.rdf, ["revenue", "gross_margin_pct"], {"type": "all"}, group_by=["region"])
        _assert_no_prose_leak(result.to_payload())


class TestPortabilityAcrossDatasets(unittest.TestCase):
    """Catches hardcoded retail vocabulary the way the hospital fixture caught
    it for the resolver-propagation bug (G3)."""

    @classmethod
    def setUpClass(cls):
        cls.df, cls.schema = _contracted("hospital_sample.csv", FIXTURES)
        cls.qe = QueryEngine(ContractAPI(cls.schema))
        assert "recovery_rate" in cls.schema.available_kpis
        assert "recovery_rate" not in cls._seed_registry_keys()

    @staticmethod
    def _seed_registry_keys():
        return set(metrics.METRICS.keys())

    def test_a_contract_only_ratio_kpi_absent_from_the_seed_registry_is_queryable(self):
        result = self.qe.query(
            self.df, ["recovery_rate"], {"type": "all"}, group_by=["department"])
        self.assertTrue(result.cells)
        for cell in result.cells:
            department = dict(cell.group)["department"]
            expected = legacy_compute(
                self.df[self.df["department"] == department], "recovery_rate",
                self.schema.contract_resolver)
            self.assertAlmostEqual(cell.values["recovery_rate"], expected, places=6)


if __name__ == "__main__":
    unittest.main()
