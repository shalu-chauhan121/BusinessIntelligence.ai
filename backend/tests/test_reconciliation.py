"""
Heterogeneous-source reconciliation.

The property this module exists to guarantee: several sources that already share
field names are stitched into one canonical view without ever inventing a value.
Nothing is mapped, nothing is summed across sources, and missing coverage stays
missing rather than quietly becoming a number.
"""
from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import pandas as pd

from app.services import dataset_service
from app.services.reconciliation import (ReconciliationError, build_source, kpi_coverage,
                                         reconcile, refresh_sources, resolve_target_grain)

from .base import SAMPLE_CSV, setup_environment

MULTISOURCE = Path(__file__).resolve().parents[2] / "sample_data" / "multisource"
AS_OF = "2026-06-10T10:15:00Z"
DAYS = pd.date_range("2026-05-01", "2026-06-09", freq="D").strftime("%Y-%m-%d").tolist()


def src(name, measures, *, regions=None, dates=None, refresh="", basis="", cadence="unknown"):
    dd = dates or DAYS
    regions = regions or ["North"] * len(dd)
    frame = pd.DataFrame({"date": dd, "region": regions,
                          "product_id": ["P1"] * len(dd), **measures})
    return build_source(name, frame, refresh, basis, cadence)


class _Compiled:
    def __init__(self, fields):
        self.source_fields = fields


def contract_with(grain="day", rollups=None, kpi_id="revenue"):
    kpi = NS(kpi_id=kpi_id,
             granularity=NS(time_grain=grain,
                            valid_rollups=rollups or ["day", "week", "month", "quarter", "year"]))
    return NS(approved_kpis=[kpi], kpis=[kpi])


class TestReconciliationModes(unittest.TestCase):
    """Partition, complementary and overlap are decided per cell and reported."""

    def test_partitioned_sources_are_unioned_not_summed(self):
        n = len(DAYS)
        a = src("Sales.csv", {"revenue": [100.0] * n})
        b = src("Finance.csv", {"revenue": [200.0] * n}, regions=["South"] * n)
        df, rep = reconcile([a, b], AS_OF)
        self.assertEqual(float(df[df.region == "North"].revenue.sum()), 100.0 * n)
        self.assertEqual(float(df[df.region == "South"].revenue.sum()), 200.0 * n)
        self.assertEqual(rep.measure_modes["revenue"]["cells_overlapping"], 0)

    def test_complementary_attributes_join_on_the_business_grain(self):
        n = len(DAYS)
        a = src("Sales.csv", {"revenue": [100.0] * n})
        b = src("Finance.csv", {"cost_of_goods": [60.0] * n})
        df, _ = reconcile([a, b], AS_OF)
        self.assertIn("revenue", df.columns)
        self.assertIn("cost_of_goods", df.columns)
        self.assertEqual(len(df), n)

    def test_overlapping_agreeing_sources_are_not_double_counted(self):
        n = len(DAYS)
        a = src("Sales.csv", {"revenue": [100.0] * n})
        b = src("Finance.csv", {"revenue": [100.0] * n})
        df, rep = reconcile([a, b], AS_OF)
        self.assertEqual(set(df.revenue.dropna().unique()), {100.0})
        self.assertEqual(rep.measure_modes["revenue"]["cells_corroborated"], n)
        self.assertEqual(rep.cells_disputed, 0)

    def test_partition_and_overlap_can_coexist_in_one_measure(self):
        n = len(DAYS)
        a = src("Sales.csv", {"revenue": [100.0] * n})
        overlap = pd.DataFrame({"date": DAYS, "region": ["North"] * n,
                                "product_id": ["P1"] * n, "revenue": [100.0] * n})
        partition = pd.DataFrame({"date": DAYS, "region": ["South"] * n,
                                  "product_id": ["P1"] * n, "revenue": [200.0] * n})
        b = build_source("Finance.csv", pd.concat([overlap, partition], ignore_index=True))
        df, rep = reconcile([a, b], AS_OF)
        self.assertEqual(rep.measure_modes["revenue"]["mode"], "mixed")
        self.assertEqual(float(df[df.region == "North"].revenue.sum()), 100.0 * n)


class TestDisagreement(unittest.TestCase):
    """A contradiction is a decision for a human, not an ingest failure."""

    def setUp(self):
        n = len(DAYS)
        self.a = src("Sales.csv", {"revenue": [100.0] * n})
        self.b = src("Finance.csv", {"revenue": [190.0] * n})

    def test_disagreement_creates_a_blocking_conflict_and_ingest_succeeds(self):
        df, rep = reconcile([self.a, self.b], AS_OF)
        flags = [i for i in rep.issues if i["kind"] == "source_disagreement"]
        self.assertTrue(flags)
        self.assertEqual(flags[0]["severity"], "blocking")
        self.assertIsNotNone(df)

    def test_disputed_cells_are_left_missing_never_summed_or_picked(self):
        df, _ = reconcile([self.a, self.b], AS_OF)
        self.assertTrue(df.revenue.isna().all())
        self.assertNotIn(290.0, set(df.revenue.dropna()))

    def test_conflict_offers_one_option_per_source(self):
        _, rep = reconcile([self.a, self.b], AS_OF)
        flag = next(i for i in rep.issues if i["kind"] == "source_disagreement")
        options = {o["option_id"] for o in flag["resolution_options"]}
        # The option_id/effect must carry the source_id, not the display label —
        # that is what `authoritative=` (and a resolved conflict on rebuild) is
        # keyed on. A label there would make every resolution silently a no-op.
        self.assertEqual(options, {"authoritative:sales", "authoritative:finance"})
        for opt in flag["resolution_options"]:
            self.assertEqual(opt["effect"]["measure"], "revenue")
            self.assertIn(opt["effect"]["authoritative_source"], {"sales", "finance"})
            self.assertIn(opt["label"], {"Treat Sales as authoritative for 'revenue'",
                                         "Treat Finance as authoritative for 'revenue'"})

    def test_authoritative_ruling_resolves_the_cells(self):
        df, rep = reconcile([self.a, self.b], AS_OF, authoritative={"revenue": "finance"})
        self.assertEqual(set(df.revenue.dropna().unique()), {190.0})
        self.assertTrue(rep.disagreements)


class TestRefreshHandling(unittest.TestCase):
    """Cadence, grain and refresh provenance are three separate things."""

    def test_cadence_is_never_derived_from_data_grain(self):
        s = src("Sales.csv", {"revenue": [1.0] * len(DAYS)})
        self.assertEqual(s.data_grain, "day")
        self.assertEqual(s.refresh_cadence, "unknown")

    def test_refresh_basis_is_recorded(self):
        declared = src("A.csv", {"revenue": [1.0] * len(DAYS)},
                       refresh="2026-06-09T08:00:00Z", basis="source_reported")
        inferred = src("B.csv", {"revenue": [1.0] * len(DAYS)})
        self.assertEqual(declared.last_refresh_basis, "source_reported")
        self.assertEqual(inferred.last_refresh_basis, "data_max")

    def test_stale_is_only_asserted_with_a_known_cadence(self):
        n = len(DAYS)
        fresh = src("Sales.csv", {"revenue": [1.0] * n},
                    refresh="2026-06-09T10:00:00Z", basis="source_reported", cadence="daily")
        lagging = src("Finance.csv", {"cost_of_goods": [1.0] * n},
                      refresh="2026-06-01T08:00:00Z", basis="source_reported", cadence="daily")
        no_cadence = src("Ops.csv", {"units_sold": [1.0] * n},
                         refresh="2026-06-09T09:00:00Z", basis="source_reported")
        _, rep = reconcile([fresh, lagging, no_cadence], AS_OF)
        status = {s.label: s.freshness for s in rep.sources}
        self.assertEqual(status["Sales"], "current")
        self.assertEqual(status["Finance"], "stale")
        self.assertEqual(status["Ops"], "freshness_unknown")

    def test_each_source_keeps_its_own_refresh_timestamp(self):
        n = len(DAYS)
        a = src("Sales.csv", {"revenue": [1.0] * n}, refresh="2026-06-09T10:00:00Z",
                basis="source_reported")
        b = src("Finance.csv", {"cost_of_goods": [1.0] * n}, refresh="2026-06-09T08:00:00Z",
                basis="source_reported")
        _, rep = reconcile([a, b], AS_OF)
        stamps = {s.label: s.last_refresh_at for s in rep.sources}
        self.assertNotEqual(stamps["Sales"], stamps["Finance"])

    def test_freshness_is_re_evaluated_against_a_later_as_of(self):
        n = len(DAYS)
        a = src("Sales.csv", {"revenue": [1.0] * n}, refresh="2026-06-09T10:00:00Z",
                basis="source_reported", cadence="daily")
        _, rep = reconcile([a, src("Finance.csv", {"cost_of_goods": [1.0] * n})], AS_OF)
        self.assertEqual(rep.source("sales").freshness, "current")
        later = refresh_sources(rep.sources, "2026-06-20T00:00:00Z")
        self.assertEqual({s.source_id: s.freshness for s in later}["sales"], "stale")

    def test_future_rows_are_dropped_against_last_refresh(self):
        n = len(DAYS)
        a = src("Sales.csv", {"revenue": [1.0] * n})
        b = src("Finance.csv", {"cost_of_goods": [1.0] * n},
                refresh="2026-06-05T00:00:00Z", basis="source_reported")
        _, rep = reconcile([a, b], AS_OF)
        self.assertGreater(rep.source("finance").rows_dropped_as_future, 0)


class TestTargetGrain(unittest.TestCase):
    """The KPI Contract sets the target grain; sources only constrain it."""

    def setUp(self):
        weeks = pd.date_range("2026-01-05", periods=8, freq="7D").strftime("%Y-%m-%d").tolist()
        self.daily = src("Sales.csv", {"revenue": [10.0] * len(DAYS)})
        self.weekly = src("Finance.csv", {"cost_of_goods": [5.0] * len(weeks)}, dates=weeks)

    def test_target_grain_comes_from_the_contract(self):
        target, basis, flag = resolve_target_grain([self.daily, self.weekly], contract_with("week"))
        self.assertEqual(target, "week")
        self.assertEqual(basis, "contract")
        self.assertIsNone(flag)

    def test_a_coarser_source_is_never_rolled_down(self):
        target, basis, flag = resolve_target_grain([self.daily, self.weekly], contract_with("day"))
        self.assertEqual(target, "week")            # NOT day — that would fabricate detail
        self.assertEqual(basis, "contract_floored")
        self.assertIsNotNone(flag)
        self.assertIn("never rolled down", flag.detail)

    def test_first_ingest_without_a_contract_uses_the_source_floor(self):
        target, basis, _ = resolve_target_grain([self.daily, self.weekly], None)
        self.assertEqual(target, "week")
        self.assertEqual(basis, "bootstrap_floor")

    def test_coarser_kpi_is_served_from_a_finer_frame(self):
        """A KPI declaring 'month' is trivially computable from a day-grain
        frame — rolling up. The gate must not withhold it just because the
        frame happens to be finer than what the KPI declares."""
        n = len(DAYS)
        df, rep = reconcile([src("Sales.csv", {"revenue": [1.0] * n}),
                             src("Finance.csv", {"cost_of_goods": [1.0] * n})], AS_OF)
        self.assertEqual(rep.canonical_grain, "day")
        coverage = kpi_coverage({"revenue": _Compiled(["revenue"])}, df, rep,
                                contract=contract_with("month"))
        self.assertFalse(coverage["revenue"]["withheld"])
        self.assertNotIn("revenue", rep.withheld_kpis)

    def test_finer_kpi_is_withheld_when_sources_cannot_serve_it(self):
        """A KPI declaring 'day' cannot be served from a week-grain frame —
        that would mean fabricating daily detail the sources never had."""
        weeks = pd.date_range("2026-01-05", periods=8, freq="7D").strftime("%Y-%m-%d").tolist()
        df, rep = reconcile([src("Sales.csv", {"revenue": [1.0] * len(weeks)}, dates=weeks),
                             src("Finance.csv", {"cost_of_goods": [1.0] * len(weeks)}, dates=weeks)],
                            AS_OF)
        self.assertEqual(rep.canonical_grain, "week")
        coverage = kpi_coverage({"revenue": _Compiled(["revenue"])}, df, rep,
                                contract=contract_with("day"))
        self.assertTrue(coverage["revenue"]["withheld"])
        self.assertIn("never rolled down", rep.withheld_kpis["revenue"])


class TestMissingCoverageNeverBecomesZero(unittest.TestCase):
    """
    The critical guarantee. `CompiledKpi.compute` uses a bare `.sum()`, so a gap
    that reaches it either under-reports invisibly or returns 0.0.
    """

    def setUp(self):
        short = pd.date_range("2026-05-01", "2026-06-01", freq="D").strftime("%Y-%m-%d").tolist()
        full = src("Sales.csv", {"revenue": [100.0] * len(DAYS)})
        lagging = src("Finance.csv", {"cost_of_goods": [60.0] * len(short)}, dates=short)
        self.df, self.rep = reconcile([full, lagging], AS_OF)
        self.coverage = kpi_coverage(
            {"revenue": _Compiled(["revenue"]),
             "gross_profit": _Compiled(["revenue", "cost_of_goods"])},
            self.df, self.rep)

    def test_uncovered_cells_stay_missing_and_never_become_zero(self):
        tail = self.df[self.df.date > "2026-06-01"]
        self.assertTrue(len(tail))
        self.assertTrue(tail.cost_of_goods.isna().all())
        self.assertFalse((tail.cost_of_goods == 0).any())

    def test_a_kpi_needing_an_incomplete_field_is_withheld(self):
        self.assertTrue(self.coverage["gross_profit"]["withheld"])
        self.assertIn("gross_profit", self.rep.withheld_kpis)

    def test_an_unaffected_kpi_is_untouched(self):
        self.assertFalse(self.coverage["revenue"]["withheld"])
        self.assertEqual(self.coverage["revenue"]["excluded_periods"], [])

    def test_coverage_does_not_claim_a_source_is_required(self):
        entry = self.coverage["gross_profit"]
        self.assertIn("supplied_by", entry)
        self.assertNotIn("required_sources", entry)


class TestIncompatibleSources(unittest.TestCase):
    """A source that does not already speak the vocabulary is refused, not mapped."""

    def test_different_date_columns_are_refused(self):
        n = len(DAYS)
        good = src("Sales.csv", {"revenue": [1.0] * n})
        other = build_source("Ops.csv", pd.DataFrame({
            "order_date": DAYS, "region": ["North"] * n,
            "product_id": ["P1"] * n, "revenue": [1.0] * n}))
        with self.assertRaises(ReconciliationError) as ctx:
            reconcile([good, other], AS_OF)
        self.assertIn("different date columns", str(ctx.exception))

    def test_same_field_name_with_a_different_role_is_refused(self):
        n = len(DAYS)
        good = src("Sales.csv", {"revenue": [1.0] * n})
        other = build_source("Ops.csv", pd.DataFrame({
            "date": DAYS, "region": ["North"] * n,
            "product_id": ["P1"] * n, "revenue": ["tier-a"] * n}))
        with self.assertRaises(ReconciliationError) as ctx:
            reconcile([good, other], AS_OF)
        self.assertIn("must mean the same thing", str(ctx.exception))


class TestDeterminism(unittest.TestCase):
    def test_reconciliation_is_deterministic(self):
        n = len(DAYS)
        pair = [src("Sales.csv", {"revenue": [100.0] * n}),
                src("Finance.csv", {"cost_of_goods": [60.0] * n})]
        a_df, a_rep = reconcile(pair, AS_OF)
        b_df, b_rep = reconcile(pair, AS_OF)
        self.assertTrue(a_df.equals(b_df))
        self.assertEqual(a_rep.to_dict(), b_rep.to_dict())


class TestPendingPeriods(unittest.TestCase):
    """A lagging feed's tail is reported as pending — distinct from a genuine
    interior gap, since it is the normal, expected shape of a live feed."""

    def test_tail_periods_are_reported_as_pending(self):
        short = pd.date_range("2026-05-01", "2026-06-01", freq="D").strftime("%Y-%m-%d").tolist()
        full = src("Sales.csv", {"revenue": [100.0] * len(DAYS)})
        lagging = src("Finance.csv", {"cost_of_goods": [60.0] * len(short)}, dates=short)
        _, rep = reconcile([full, lagging], AS_OF)
        self.assertIn("cost_of_goods", rep.periods_pending)
        pending = set(rep.periods_pending["cost_of_goods"])
        self.assertTrue(pending)
        self.assertTrue(all(p > short[-1] for p in pending))
        self.assertNotIn("revenue", rep.periods_pending)


class TestCadenceAndProvenanceThroughStoreMultisourceUpload(unittest.TestCase):
    """`source_meta` is how cadence and a source-reported timestamp actually
    reach the reconciliation layer through the real ingest function — not just
    through `build_source` called directly in a test."""

    @classmethod
    def setUpClass(cls):
        setup_environment()

    def test_declared_cadence_makes_staleness_assertable(self):
        n = len(DAYS)
        sales = ("sales.csv", ("date,region,product_id,revenue\n" + "\n".join(
            f"{d},North,P1,100.0" for d in DAYS)).encode())
        finance = ("finance.csv", ("date,region,product_id,cost_of_goods\n" + "\n".join(
            f"{d},North,P1,60.0" for d in DAYS)).encode())
        ds = dataset_service.store_multisource_upload(
            "cadence_meta_user", [sales, finance], as_of=AS_OF,
            source_meta={"finance.csv": {"cadence": "daily",
                                         "last_refresh_at": "2026-06-01T08:00:00Z"}})
        by_label = {s["label"]: s for s in ds["sources"]["sources"]}
        self.assertEqual(by_label["Finance"]["refresh_cadence"], "daily")
        self.assertEqual(by_label["Finance"]["last_refresh_basis"], "source_reported")
        self.assertEqual(by_label["Finance"]["freshness"], "stale")

    def test_source_reported_timestamp_is_not_relabelled(self):
        n = len(DAYS)
        sales = ("sales.csv", ("date,region,product_id,revenue\n" + "\n".join(
            f"{d},North,P1,100.0" for d in DAYS)).encode())
        finance = ("finance.csv", ("date,region,product_id,cost_of_goods\n" + "\n".join(
            f"{d},North,P1,60.0" for d in DAYS)).encode())
        ds = dataset_service.store_multisource_upload(
            "cadence_meta_user2", [sales, finance], as_of=AS_OF, source_meta={})
        by_label = {s["label"]: s for s in ds["sources"]["sources"]}
        # Nothing was declared: both fall back to the honest inferred basis,
        # never presented as if the source itself had reported a timestamp.
        self.assertEqual(by_label["Sales"]["last_refresh_basis"], "data_max")
        self.assertEqual(by_label["Finance"]["last_refresh_basis"], "data_max")


class TestSingleSourceRegression(unittest.TestCase):
    """An ordinary one-file upload must be entirely unaffected by any of this."""

    @classmethod
    def setUpClass(cls):
        state = setup_environment()
        cls.uid, cls.dataset = state["uid"], state["dataset"]

    def test_single_source_dataset_carries_no_source_report(self):
        _, schema = dataset_service.load(self.dataset, self.uid)
        self.assertIsNone(schema.sources)

    def test_single_source_schema_still_zero_fills(self):
        df, schema = dataset_service.load(self.dataset, self.uid)
        for column in schema.base_metrics:
            self.assertFalse(df[column].isna().any(), column)


@unittest.skipUnless(MULTISOURCE.exists() and any(MULTISOURCE.glob("*.csv")),
                     "demo sources not generated")
class TestBundledDemo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_environment()
        files = [(p.name, p.read_bytes()) for p in sorted(MULTISOURCE.glob("*.csv"))]
        cls.ds = dataset_service.store_multisource_upload(
            "demo_multisource_user", files, as_of="2026-06-30T10:15:00Z")
        cls.report = cls.ds["sources"]

    def test_no_mapping_file_is_needed(self):
        self.assertFalse((MULTISOURCE / "bindings.json").exists())

    def test_every_demo_mode_is_exercised(self):
        modes = {m: i["mode"] for m, i in self.report["measure_modes"].items()}
        self.assertEqual(modes["revenue"], "mixed")
        self.assertIn("partition_or_complementary", set(modes.values()))

    def test_reconciled_revenue_matches_the_unsplit_source_of_truth(self):
        truth = pd.read_csv(SAMPLE_CSV, parse_dates=["date"])
        week = truth[truth["date"] == "2023-01-02"]
        df, _ = dataset_service.load(self.ds, "demo_multisource_user")
        got = df[df["_date"] == pd.Timestamp("2023-01-02")]["revenue"].sum()
        self.assertAlmostEqual(float(got), float(week["revenue"].sum()), places=1)

    def test_lagging_source_withholds_only_the_kpis_that_need_it(self):
        withheld = self.report["withheld_kpis"]
        self.assertTrue(withheld)
        self.assertNotIn("revenue", withheld)


if __name__ == "__main__":
    unittest.main()
