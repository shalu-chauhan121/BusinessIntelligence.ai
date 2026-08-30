"""
ContractAPI -- the one sanctioned entry point onto a dataset's KPI Contract.

The facade must be behaviourally identical to the legacy `app.engines.metrics`
functions for everything they already got right (equivalence tests below), and
must additionally guarantee what the legacy free-function API could not: that
an unknown KPI key is a typed, recoverable error rather than a silent NaN, and
that a caller can never forget to pass the resolver because there is no
per-call resolver argument to forget.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from app.agent.contract_api import ContractAPI, DimensionInfo, KpiInfo
from app.agent.errors import (UnknownDimensionError, UnknownKpiError,
                              UnknownMemberError)
from app.engines.metrics import compute as legacy_compute
from app.engines.metrics import detect_schema, higher_is_better, metric_label, metric_unit, prepare
from app.kpi import service as kpi_service
from app.kpi.resolver import compile_contract

from .base import EngineTestCase

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class TestContractApiOnContractBackedDataset(EngineTestCase):
    """The shared sample-company dataset, which does have a compiled contract."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # `EngineTestCase.schema` is loaded via `dataset_service.load`, which
        # only attaches a contract when a uid is supplied -- rebuild the
        # schema the way the real request path does, with a contract.
        from app.services import dataset_service
        cls.df, cls.schema_with_contract = dataset_service.load(cls.dataset, cls.uid)
        cls.api = ContractAPI(cls.schema_with_contract)

    def test_has_and_require_agree(self):
        for key in self.schema_with_contract.available_kpis:
            self.assertTrue(self.api.has(key))
            self.api.require(key)  # must not raise

    def test_require_unknown_key_raises_typed_error_with_alternatives(self):
        with self.assertRaises(UnknownKpiError) as ctx:
            self.api.require("not_a_real_kpi_xyz")
        self.assertEqual(ctx.exception.key, "not_a_real_kpi_xyz")
        self.assertGreater(len(ctx.exception.valid_alternatives), 0)
        payload = ctx.exception.to_payload()
        self.assertEqual(payload["error"], "unknown_kpi")
        self.assertIn("not_a_real_kpi_xyz", payload["requested"])

    def test_value_matches_legacy_compute_for_every_kpi(self):
        """Facade and legacy free function must agree, value for value."""
        resolver = self.schema_with_contract.contract_resolver
        for key in self.schema_with_contract.available_kpis:
            expected = legacy_compute(self.df, key, resolver)
            actual = self.api.value(self.df, key)
            if expected != expected and actual != actual:
                continue  # both NaN
            self.assertAlmostEqual(actual, expected, places=6, msg=f"{key} diverged")

    def test_value_on_unknown_key_raises_rather_than_returning_nan(self):
        with self.assertRaises(UnknownKpiError):
            self.api.value(self.df, "not_a_real_kpi_xyz")

    def test_list_kpis_matches_available_kpis(self):
        info = self.api.list_kpis()
        self.assertEqual({k.key for k in info}, set(self.schema_with_contract.available_kpis))
        for k in info:
            self.assertIsInstance(k, KpiInfo)
            self.assertIn(k.source, ("contract", "seed"))

    def test_list_kpis_labels_and_units_match_legacy_lookups(self):
        resolver = self.schema_with_contract.contract_resolver
        by_key = {k.key: k for k in self.api.list_kpis()}
        for key in self.schema_with_contract.available_kpis:
            self.assertEqual(by_key[key].label, metric_label(key, resolver))
            self.assertEqual(by_key[key].unit, metric_unit(key, resolver))
            self.assertEqual(by_key[key].higher_is_better, higher_is_better(key, resolver))

    def test_list_dimensions_names_match_schema(self):
        dims = self.api.list_dimensions()
        self.assertEqual({d.name for d in dims}, set(self.schema_with_contract.dimensions))
        for d in dims:
            self.assertIsInstance(d, DimensionInfo)

    def test_list_dimensions_with_df_computes_distinct_count(self):
        dims = self.api.list_dimensions(self.df)
        for d in dims:
            self.assertIsNotNone(d.distinct_count)
            self.assertEqual(d.distinct_count, int(self.df[d.name].nunique()))

    def test_require_dimension_unknown_raises_typed_error(self):
        with self.assertRaises(UnknownDimensionError) as ctx:
            self.api.require_dimension("not_a_real_dimension_xyz")
        self.assertEqual(ctx.exception.name, "not_a_real_dimension_xyz")
        self.assertEqual(set(ctx.exception.valid_alternatives),
                         set(self.schema_with_contract.dimensions))

    def test_require_dimension_known_returns_the_name(self):
        for dim in self.schema_with_contract.dimensions:
            self.assertEqual(self.api.require_dimension(dim), dim)


class TestContractApiFiltersAreHonoured(EngineTestCase):
    """
    A contract KPI's declared row filters must be applied when the facade
    evaluates it -- something the legacy seed `MetricSpec` path has no concept
    of at all (`metrics._col_sum` just sums a column; it cannot filter rows
    first). This is not hypothetical: `FilterSpec`/`apply_filters` exist in
    `kpi/resolver.py` specifically so a KPI Studio user can define "revenue
    excluding returns", and a facade that silently used the seed definition
    instead would silently drop that filter.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        raw = pd.read_csv(FIXTURES / "hospital_sample.csv")
        cls.h_schema = detect_schema(raw)
        cls.h_df = prepare(raw, cls.h_schema)
        contract = kpi_service.bootstrap(
            "u", {"_id": "hosp1", "filename": "hospital.csv"}, cls.h_df, cls.h_schema)
        resolver = compile_contract(contract)

        # No auto-bootstrapped KPI on this fixture happens to declare a
        # filter, so add one by hand: admissions, excluding Oncology.
        from app.kpi.contract import FilterSpec
        from app.kpi.resolver import CompiledKpi, parse_expression

        resolver["admissions_excl_oncology"] = CompiledKpi(
            key="admissions_excl_oncology",
            label="Admissions excluding Oncology",
            unit="count",
            kind="sum",
            expression_ast=parse_expression("{admissions}"),
            filters=[FilterSpec(field="department", op="ne", value="Oncology")],
            source_fields=["admissions", "department"],
        )
        cls.h_schema.contract_resolver = resolver
        cls.api = ContractAPI(cls.h_schema)

    def test_declared_filter_excludes_the_matching_rows(self):
        filtered = self.api.value(self.h_df, "admissions_excl_oncology")
        expected = self.h_df.loc[self.h_df["department"] != "Oncology", "admissions"].sum()
        unfiltered = self.h_df["admissions"].sum()

        self.assertEqual(filtered, expected)
        self.assertLess(filtered, unfiltered,
                        "the filter must actually remove rows, or this test proves nothing")

    def test_contract_only_ratio_resolves_to_a_real_number(self):
        value = self.api.value(self.h_df, "recovery_rate")
        self.assertEqual(value, value, "must not be NaN")
        self.assertGreater(value, 0)

    def test_contract_only_ratio_is_absent_from_seed_registry(self):
        from app.engines.metrics import METRICS
        self.assertNotIn("recovery_rate", METRICS)

    def test_shadowed_key_prefers_the_contract_definition(self):
        """cost_of_goods exists in both the seed registry and this hospital
        contract (bound to `treatment_cost`); the contract must win."""
        from app.engines.metrics import METRICS
        self.assertIn("cost_of_goods", METRICS)
        self.assertIn("cost_of_goods", self.h_schema.contract_resolver)
        value = self.api.value(self.h_df, "cost_of_goods")
        self.assertEqual(value, self.h_df["treatment_cost"].sum())


class TestContractApiOnContractlessDataset(unittest.TestCase):
    """A dataset that never went through `dataset_service.attach_contract` --
    `contract_resolver` is `None`. The facade must degrade to the seed
    registry exactly as the legacy free functions do, and say so."""

    @classmethod
    def setUpClass(cls):
        raw = pd.DataFrame({
            "date": pd.date_range("2023-01-02", periods=60, freq="W-MON"),
            "revenue": range(100, 160),
            "orders": range(50, 110),
            "region": (["North", "South"] * 30),
        })
        cls.schema = detect_schema(raw)
        cls.df = prepare(raw, cls.schema)
        assert cls.schema.contract_resolver is None
        cls.api = ContractAPI(cls.schema)

    def test_lists_kpis_from_the_seed_registry(self):
        info = self.api.list_kpis()
        self.assertGreater(len(info), 0)
        for k in info:
            self.assertEqual(k.source, "seed")

    def test_value_still_computes_via_seed_registry(self):
        value = self.api.value(self.df, "revenue")
        self.assertEqual(value, self.df["revenue"].sum())

    def test_unknown_key_still_raises_typed_error(self):
        with self.assertRaises(UnknownKpiError):
            self.api.value(self.df, "not_a_real_kpi_xyz")


class TestContractApiExposesDimensionMembers(EngineTestCase):
    """
    C4. `grounding.py` has always read `schema.dimension_members`, which never
    existed, so a question naming a region bound no filter. The facade is where
    the agent layer reaches that index, and an invented member has to be a
    recoverable error rather than an empty slice -- an empty slice is
    indistinguishable from a real zero once it becomes a number.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from app.services import dataset_service

        cls.df, cls.loaded_schema = dataset_service.load(cls.dataset, cls.uid)
        cls.api = ContractAPI(cls.loaded_schema)

    def test_list_members_returns_a_page_not_the_whole_column(self):
        page = self.api.list_members("region", limit=2)
        self.assertEqual(len(page.members), 2)
        self.assertEqual(page.total, int(self.df["region"].nunique()))
        self.assertTrue(page.indexed)

    def test_list_members_on_an_unknown_dimension_raises_the_typed_error(self):
        with self.assertRaises(UnknownDimensionError) as ctx:
            self.api.list_members("not_a_dimension_xyz")
        self.assertEqual(ctx.exception.to_payload()["error"], "unknown_dimension")
        self.assertIn("region", ctx.exception.valid_alternatives)

    def test_resolve_member_is_case_insensitive(self):
        for spelling in ("North", "north", "NORTH", "  north  "):
            self.assertEqual(self.api.resolve_member("region", spelling), "North")

    def test_resolve_member_on_an_unknown_value_raises_with_valid_alternatives(self):
        with self.assertRaises(UnknownMemberError) as ctx:
            self.api.resolve_member("region", "Atlantis")
        payload = ctx.exception.to_payload()
        self.assertEqual(payload["error"], "unknown_member")
        self.assertEqual(payload["dimension"], "region")
        self.assertEqual(payload["requested"], "Atlantis")
        self.assertIn("North", payload["valid_alternatives"])

    def test_a_schema_with_no_attached_catalogue_still_works_from_a_supplied_frame(self):
        """
        A schema built by a bare detect_schema/prepare -- how the KPI bootstrap
        path and much of this suite build theirs -- carries no catalogue. It
        must not silently look like a dataset with no members.
        """
        bare = detect_schema(pd.read_csv(FIXTURES / "hospital_sample.csv"))
        frame = prepare(pd.read_csv(FIXTURES / "hospital_sample.csv"), bare)
        api = ContractAPI(bare)
        self.assertIsNone(getattr(bare, "member_catalogue", None))
        page = api.list_members("department", df=frame)
        self.assertEqual(list(page.members), sorted(frame["department"].astype(str).unique()))

    def test_list_dimensions_reports_distinct_count_without_a_frame_when_a_catalogue_is_attached(self):
        by_name = {d.name: d for d in self.api.list_dimensions()}
        self.assertEqual(by_name["region"].distinct_count, int(self.df["region"].nunique()))
        self.assertTrue(by_name["region"].indexed)

    def test_find_members_never_chooses_between_two_columns(self):
        matches = self.api.find_members("Product A")
        self.assertEqual([(m.dimension, m.member) for m in matches],
                         [("product", "Product A")])
        self.assertEqual(self.api.find_members("Atlantis"), [])


if __name__ == "__main__":
    unittest.main()
