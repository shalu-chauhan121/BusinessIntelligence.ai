"""
`agent.registry` -- the tool table, its generated JSON Schemas, and dispatch.

The registry is what turns twenty engine classes into a model-callable tool
surface: this pins the mechanical guarantees that surface depends on (every
schema is legal and matches its underlying signature, `df`/`self`/`api`
never leak into a schema, dispatch never raises past its boundary) and the
two cross-cutting properties the master plan calls out explicitly:

- **G1, prose-leak guard**: no tool result carries a string field that reads
  like a sentence rather than a fact.
- **G3, multi-dataset portability**: the registry -- and, on a representative
  subset of tools, real dispatch -- works unchanged against the retail,
  hospital and school fixtures, which is what catches a hardcoded column
  name or a whitelisted dimension list before it ships.
"""
from __future__ import annotations

import inspect
import json
import unittest
from pathlib import Path

from app.agent import registry
from app.agent.contract_api import ContractAPI
from app.agent.errors import AgentToolError

from .base import (FIXTURES, SAMPLES, EngineTestCase, assert_json_safe,
                  assert_no_prose_leak, contracted)


def _build_registry(df, schema):
    api = ContractAPI(schema)
    return api, registry.build(api, df)


class RetailRegistryTestCase(EngineTestCase):
    """The shared retail fixture, contract-backed, with a registry built once
    per test class -- every structural and execution test below runs
    against this one registry instance."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.rdf, cls.rschema = contracted("business_metrics_sample.csv")
        cls.api, cls.reg = _build_registry(cls.rdf, cls.rschema)

        cls.kpi = "revenue"
        cls.ratio_kpi = "gross_margin_pct"
        cls.dim = "region"
        cls.dim2 = "product"
        cls.tf_year = {"type": "year", "year": 2024}
        cls.tf_year2 = {"type": "year", "year": 2025}
        cls.tf_quarter = {"type": "quarter", "year": 2024, "quarter": 2}

        # One example call per registered tool, exercised against a real
        # fixture. This is both the execution test and the material for
        # the prose-leak sweep -- if a tool gains a parameter with no
        # sensible default here, this dict is the place to extend.
        cls.examples = {
            "describe_dataset": {},
            "list_kpis": {},
            "get_kpi_definition": {"key": cls.kpi},
            "list_dimensions": {},
            "list_dimension_members": {"dimension": cls.dim},
            "get_kpi_inputs": {"key": cls.kpi},
            "get_formula_structure": {"key": cls.ratio_kpi},
            "search_kpis": {"text": "revenue"},
            "find_related_kpis": {"key": cls.kpi},
            "get_formula_components": {"key": cls.ratio_kpi},
            "get_kpi_neighbours": {"key": cls.kpi},
            "query_kpi": {"kpi_keys": cls.kpi, "time_filter": cls.tf_year},
            "get_timeseries": {"kpi_key": cls.kpi, "time_filter": cls.tf_year},
            "get_timeseries_multi": {"kpi_keys": [cls.kpi, cls.ratio_kpi], "time_filter": cls.tf_year},
            "rank_entities": {"kpi_key": cls.kpi, "dimension": cls.dim, "time_filter": cls.tf_year},
            "get_distribution": {"kpi_key": cls.kpi, "dimension": cls.dim, "time_filter": cls.tf_year},
            "cross_tabulate": {"kpi_key": cls.kpi, "dim_a": cls.dim, "dim_b": cls.dim2, "time_filter": cls.tf_year},
            "compare_periods": {"kpi_key": cls.kpi, "period_a": cls.tf_year, "period_b": cls.tf_year2},
            "assess_significance": {"kpi_key": cls.kpi, "period_a": cls.tf_quarter},
            "get_normal_range": {"kpi_key": cls.kpi},
            "filter_material_changes": {"candidates": [{"kpi": cls.kpi, "change_pct": 12.0}]},
            "check_data_quality": {"time_filter": cls.tf_year},
            "detect_trend": {"kpi_key": cls.kpi},
            "detect_changepoint": {"kpi_key": cls.kpi},
            "scan_kpis": {"time_filter": cls.tf_quarter},
            "rank_kpis_by_movement": {"time_filter": cls.tf_quarter},
            "scan_anomalies": {"time_filter": cls.tf_quarter},
            "scan_dimension_outliers": {"kpi_key": cls.kpi, "dimension": cls.dim, "time_filter": cls.tf_year},
            "decompose_by_dimension": {"kpi_key": cls.kpi, "dimension": cls.dim,
                                      "period_a": cls.tf_year, "period_b": cls.tf_year2},
            "decompose_rate_mix": {"kpi_key": cls.ratio_kpi, "dimension": cls.dim,
                                   "period_a": cls.tf_year, "period_b": cls.tf_year2},
            "decompose_nested": {"kpi_key": cls.kpi, "outer_dimension": cls.dim, "outer_member": "East",
                                "inner_dimension": cls.dim2, "period_a": cls.tf_year, "period_b": cls.tf_year2},
            "decompose_formula": {"kpi_key": cls.ratio_kpi, "period_a": cls.tf_year, "period_b": cls.tf_year2},
            "attribute_dimensions": {"kpi_key": cls.kpi, "period_a": cls.tf_year, "period_b": cls.tf_year2},
            "compute_over_index": {"kpi_key": cls.kpi, "dimension": cls.dim,
                                   "period_a": cls.tf_year, "period_b": cls.tf_year2},
            "rank_drivers": {"kpi_key": cls.kpi, "period_a": cls.tf_year, "period_b": cls.tf_year2},
            "bridge_periods": {"kpi_key": cls.kpi, "period_a": cls.tf_year, "period_b": cls.tf_year2},
            "measure_concentration": {"kpi_key": cls.kpi, "dimension": cls.dim, "time_filter": cls.tf_year},
            "find_outlier_contributors": {"kpi_key": cls.kpi, "dimension": cls.dim,
                                         "period_a": cls.tf_year, "period_b": cls.tf_year2},
            "compare_segments": {"kpi_key": cls.kpi, "dimension": cls.dim,
                                "member_a": "East", "member_b": "West", "period_a": cls.tf_year},
            "compare_cohorts": {"kpi_key": cls.kpi, "filter_a": {"region": "East"},
                               "filter_b": {"region": "West"}, "period_a": cls.tf_year},
            "segment_by_behavior": {"kpi_key": cls.kpi, "dimension": cls.dim,
                                   "period_a": cls.tf_year, "period_b": cls.tf_year2},
            "detect_seasonality": {"kpi_key": cls.kpi},
            "compare_to_seasonal_norm": {"kpi_key": cls.kpi, "time_filter": cls.tf_quarter},
            "correlate_kpis": {"kpi_a": cls.kpi, "kpi_b": cls.ratio_kpi,
                              "period_a": cls.tf_year, "period_b": cls.tf_year2},
            "correlate_kpi_matrix": {"kpi_key": cls.kpi, "period_a": cls.tf_year, "period_b": cls.tf_year2},
            "check_temporal_precedence": {"kpi_key": cls.kpi, "cause_kpi": cls.ratio_kpi},
            "cross_correlate_lagged": {"kpi_key": cls.kpi, "cause_kpi": cls.ratio_kpi},
            "test_reverse_causation": {"kpi_key": cls.kpi, "cause_kpi": cls.ratio_kpi},
            "find_counterexamples": {"kpi_key": cls.kpi, "cause_kpi": cls.ratio_kpi, "dimension": cls.dim,
                                    "period_a": cls.tf_year, "period_b": cls.tf_year2},
            "test_consistency_across_dimension": {"kpi_key": cls.kpi, "cause_kpi": cls.ratio_kpi,
                                                 "dimension": cls.dim, "period_a": cls.tf_year,
                                                 "period_b": cls.tf_year2},
            "test_holdout_segments": {"kpi_key": cls.kpi, "cause_kpi": cls.ratio_kpi, "dimension": cls.dim,
                                     "period_a": cls.tf_year, "period_b": cls.tf_year2},
            "test_confounders": {"kpi_key": cls.kpi, "cause_kpi": cls.ratio_kpi,
                                "candidates": ["cost_of_goods"],
                                "period_a": cls.tf_year, "period_b": cls.tf_year2},
            "test_spurious_correlation": {"kpi_key": cls.kpi, "cause_kpi": cls.ratio_kpi},
            "rank_competing_explanations": {"kpi_key": cls.kpi,
                                           "candidate_causes": [cls.ratio_kpi, "cost_of_goods"],
                                           "period_a": cls.tf_year, "period_b": cls.tf_year2},
            "test_statistical_significance": {"kpi_key": cls.kpi, "cause_kpi": cls.ratio_kpi,
                                             "period_a": cls.tf_year, "period_b": cls.tf_year2},
            "check_sample_adequacy": {"kpi_key": cls.kpi, "dimension": cls.dim},
            "test_sensitivity_to_outliers": {"kpi_key": cls.kpi, "cause_kpi": cls.ratio_kpi,
                                            "period_a": cls.tf_year, "period_b": cls.tf_year2},
            "estimate_effect_size": {"kpi_key": cls.kpi, "cause_kpi": cls.ratio_kpi,
                                     "period_a": cls.tf_year, "period_b": cls.tf_year2},
        }


class TestEveryToolProducesALegalSchema(RetailRegistryTestCase):
    def test_every_registered_tool_has_a_matching_example(self):
        self.assertEqual(set(self.reg.tool_names()), set(self.examples.keys()))

    def test_no_duplicate_tool_names(self):
        names = self.reg.tool_names()
        self.assertEqual(len(names), len(set(names)))

    def test_required_is_always_a_subset_of_properties(self):
        for tool in self.reg.tools:
            props = set(tool["input_schema"].get("properties", {}))
            required = set(tool["input_schema"].get("required", []))
            self.assertTrue(required <= props, f"{tool['name']}: {required - props}")

    def test_context_parameters_never_leak_into_a_schema(self):
        for tool in self.reg.tools:
            props = set(tool["input_schema"].get("properties", {}))
            self.assertNotIn("df", props, tool["name"])
            self.assertNotIn("api", props, tool["name"])
            self.assertNotIn("self", props, tool["name"])

    def test_every_schema_is_json_serialisable(self):
        json.dumps(self.reg.tools)

    def test_only_the_last_tool_carries_the_cache_breakpoint(self):
        self.assertIn("cache_control", self.reg.tools[-1])
        self.assertFalse(any("cache_control" in t for t in self.reg.tools[:-1]))


class TestTheSchemaCannotDriftFromItsSignature(RetailRegistryTestCase):
    """The guard that fails when someone changes an engine signature and
    forgets the registry: every schema property must exist in the bound
    callable's own parameters, and every parameter without a default
    (excluding context params and the omitted-knob set) must be required."""

    def test_every_property_exists_on_the_underlying_callable(self):
        for spec in registry.TOOL_SPECS:
            fn = registry._bound_callable(spec, self.reg._engine_instances)
            sig_params = set(inspect.signature(fn).parameters)
            tool = next(t for t in self.reg.tools if t["name"] == spec.name)
            for prop in tool["input_schema"].get("properties", {}):
                self.assertIn(prop, sig_params, f"{spec.name}.{prop} has no matching parameter")

    def test_every_required_signature_parameter_is_required_in_the_schema(self):
        for spec in registry.TOOL_SPECS:
            fn = registry._bound_callable(spec, self.reg._engine_instances)
            _, model_params = registry._context_and_model_params(fn)
            tool = next(t for t in self.reg.tools if t["name"] == spec.name)
            required = set(tool["input_schema"].get("required", []))
            for param in model_params:
                if param.default is inspect.Parameter.empty:
                    self.assertIn(param.name, required,
                                 f"{spec.name}.{param.name} has no default but is not required")


class TestGroupGatingReturnsTheRightSubset(RetailRegistryTestCase):
    def test_every_tool_appears_in_exactly_one_group_listing(self):
        groups = {registry._SPECS_BY_NAME[name].group for name in self.reg.tool_names()}
        seen = set()
        for group in groups:
            names = {t["name"] for t in self.reg.by_group(group)}
            self.assertTrue(names, group)
            self.assertTrue(seen.isdisjoint(names), f"{group} overlaps a prior group")
            seen |= names
        self.assertEqual(seen, set(self.reg.tool_names()))


class TestEveryToolExecutesOnARealFixture(RetailRegistryTestCase):
    def test_every_example_call_succeeds_and_round_trips_through_json(self):
        failures = []
        for name, args in self.examples.items():
            payload, is_error = self.reg.dispatch(name, args)
            assert_json_safe(payload)
            if is_error:
                failures.append((name, payload))
        self.assertEqual(failures, [])

    def test_g1_no_tool_result_leaks_a_sentence(self):
        for name, args in self.examples.items():
            payload, is_error = self.reg.dispatch(name, args)
            self.assertFalse(is_error, name)
            assert_no_prose_leak(payload, f"${name}")


class TestAHallucinatedArgumentIsARecoverableTurnNeverAnException(RetailRegistryTestCase):
    """
    A KPI key, a dimension, and a time-filter `type` are all schema `enum`s
    (built from `api.available_keys(df)` / `api.list_dimensions(df)` / the
    fixed eight `TimeFilter` shapes), so the airlock (A3) now catches a bad
    value for any of them *before* the engine runs at all -- earlier than
    the engine's own `UnknownKpiError` / `UnknownDimensionError` /
    `MalformedTimeFilterError`, which is why every one of these reports
    `invalid_argument` rather than the engine's own error code. Equally
    recoverable either way: `valid_alternatives` still names the real
    choices. The engine's own checks are not dead code -- see
    `TestTheEnginesOwnDomainChecksStillFireWithinALegalShape` below for a
    case the airlock's deliberately permissive `TimeFilter` schema
    (decision 55) does not catch.
    """

    def test_an_unknown_kpi_key_is_caught_by_the_airlock_with_alternatives(self):
        payload, is_error = self.reg.dispatch(
            "query_kpi", {"kpi_keys": "not_a_real_kpi", "time_filter": {"type": "all"}})
        self.assertTrue(is_error)
        self.assertEqual(payload["error"], "invalid_argument")
        self.assertTrue(payload["valid_alternatives"])

    def test_an_unknown_dimension_is_caught_by_the_airlock_with_alternatives(self):
        payload, is_error = self.reg.dispatch(
            "rank_entities", {"kpi_key": self.kpi, "dimension": "not_a_real_dimension",
                             "time_filter": self.tf_year})
        self.assertTrue(is_error)
        self.assertEqual(payload["error"], "invalid_argument")
        self.assertTrue(payload["valid_alternatives"])

    def test_a_bogus_time_filter_type_is_caught_by_the_airlock_naming_valid_types(self):
        payload, is_error = self.reg.dispatch(
            "query_kpi", {"kpi_keys": self.kpi, "time_filter": {"type": "not_a_real_type"}})
        self.assertTrue(is_error)
        self.assertEqual(payload["error"], "invalid_argument")
        self.assertTrue(payload["valid_alternatives"])

    def test_an_unknown_tool_name_returns_invalid_argument_naming_valid_tools(self):
        payload, is_error = self.reg.dispatch("not_a_real_tool", {})
        self.assertTrue(is_error)
        self.assertEqual(payload["error"], "invalid_argument")
        self.assertIn("query_kpi", payload["valid_alternatives"])

    def test_a_tool_that_raises_unexpectedly_is_still_a_typed_error_not_a_crash(self):
        # No schema here sets `additionalProperties: false`, so an unknown
        # extra key is legal JSON Schema and passes the airlock -- this is
        # the one shape of bad input that genuinely reaches `fn(**kwargs)`
        # and raises a Python `TypeError`, which `dispatch`'s own boundary
        # must still convert rather than let escape.
        payload, is_error = self.reg.dispatch(
            "query_kpi", {"kpi_keys": self.kpi, "time_filter": self.tf_year, "bogus_arg": 1})
        self.assertTrue(is_error)
        self.assertEqual(payload["error"], "invalid_argument")


class TestTheEnginesOwnDomainChecksStillFireWithinALegalShape(RetailRegistryTestCase):
    """
    The airlock's `TimeFilter` schema is deliberately permissive (decision
    55): it enforces the outer object shape and the `type` enum, but not
    which other keys a given `type` requires. That is `timefilter.parse`'s
    job, and it must still be reachable through `dispatch` for a value that
    is a legal `type` but an incomplete shape for it.
    """

    def test_a_quarter_time_filter_missing_its_quarter_key_still_reaches_malformed_time_filter(self):
        payload, is_error = self.reg.dispatch(
            "query_kpi", {"kpi_keys": self.kpi, "time_filter": {"type": "quarter", "year": 2024}})
        self.assertTrue(is_error)
        self.assertEqual(payload["error"], "malformed_time_filter")
        self.assertTrue(payload["valid_types"])


class TestMultiDatasetPortabilityG3(unittest.TestCase):
    """The registry -- schema generation and dispatch on a representative
    subset of tools -- works unchanged against fixtures with different KPI
    names, different dimensions, and (school) a dimension at cardinality 1.
    Catches a hardcoded 'revenue'/'units_sold' or a whitelisted dimension
    list before it ships."""

    FIXTURES = (
        ("hospital", "hospital_sample.csv", FIXTURES),
        ("hospital_smoke", "hospital_kpi_smoke_sample.csv", SAMPLES),
        ("school", "school_kpi_smoke_sample.csv", SAMPLES),
    )

    def test_registry_builds_and_a_representative_tool_set_executes(self):
        for label, csv_name, samples_dir in self.FIXTURES:
            with self.subTest(fixture=label):
                df, schema = contracted(csv_name, samples_dir)
                api, reg = _build_registry(df, schema)
                self.assertEqual(len(reg.tools), len(registry.TOOL_SPECS))

                kpi = api.list_kpi_keys()[0]
                ratio_candidates = [k for k in api.list_kpi_keys() if api.kind(k) == "ratio"]
                ratio_kpi = ratio_candidates[0] if ratio_candidates else kpi
                dim = api.list_dimensions(df)[0].name
                tf_a = {"type": "year", "year": 2024}
                tf_b = {"type": "year", "year": 2025}

                calls = [
                    ("describe_dataset", {}),
                    ("query_kpi", {"kpi_keys": kpi, "time_filter": tf_a}),
                    ("rank_entities", {"kpi_key": kpi, "dimension": dim, "time_filter": tf_a}),
                    ("compare_periods", {"kpi_key": kpi, "period_a": tf_a, "period_b": tf_b}),
                    ("decompose_by_dimension", {"kpi_key": kpi, "dimension": dim,
                                               "period_a": tf_a, "period_b": tf_b}),
                    ("rank_drivers", {"kpi_key": kpi, "period_a": tf_a, "period_b": tf_b}),
                    ("test_consistency_across_dimension",
                    {"kpi_key": kpi, "cause_kpi": ratio_kpi, "dimension": dim,
                     "period_a": tf_a, "period_b": tf_b}),
                ]
                for name, args in calls:
                    payload, is_error = reg.dispatch(name, args)
                    assert_json_safe(payload)
                    self.assertFalse(is_error, f"{label}.{name}: {payload}")
                    assert_no_prose_leak(payload, f"${label}.{name}")


if __name__ == "__main__":
    unittest.main()
