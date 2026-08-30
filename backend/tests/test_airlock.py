"""
`agent.airlock` -- type/shape validation in front of the tool registry.

This pins three failure modes found by fuzzing the live registry before A3
existed, each a genuine gap the semantic checks already inside every engine
(`query.py:118-141` is the canonical example -- KPI key, dimension, enum and
range checks, all before `apply_time_filter` touches `df`) do not cover:

  * a bool for a free-text search phrase **executed** and returned a
    plausible empty answer (`search_kpis(text=True)` -> `{"matches": []}`,
    `is_error=False`) -- the airlock's job is exactly to stop this class of
    silent, wrong-but-plausible success;
  * a list for a dict-shaped `filters` argument reached pandas and leaked a
    raw internal message ("dictionary update sequence element #0 has length
    6; 2 is required");
  * a string for a list-shaped `candidates` argument was iterated character
    by character and reported `unknown_kpi: 'n'` -- actively misdirecting
    the model about *what* was wrong.

Also pinned: the interaction with A2's schemas this module validates against
-- a bad enum value (a hallucinated KPI key, dimension, or `TimeFilter`
`type`) is now caught by the airlock *before* the engine's own typed error
fires, and the engine's own domain checks (`timefilter.parse`'s required-key
enforcement for a legal `type`) remain reachable for what the airlock's
deliberately permissive schema does not encode (decision 55).
"""
from __future__ import annotations

import json
import unittest
from typing import Any, Dict

from app.agent import airlock, registry
from app.agent.errors import InvalidArgumentError

from .base import assert_json_safe
from .test_registry import RetailRegistryTestCase


class TestTheThreeMeasuredGapsAreClosed(RetailRegistryTestCase):
    def test_a_bool_for_a_free_text_argument_never_executes(self):
        payload, is_error = self.reg.dispatch("search_kpis", {"text": True})
        self.assertTrue(is_error)
        self.assertEqual(payload["error"], "invalid_argument")
        self.assertEqual(payload["argument"], "text")

    def test_a_list_for_a_dict_shaped_filters_argument_is_a_typed_error_not_a_pandas_message(self):
        payload, is_error = self.reg.dispatch(
            "query_kpi", {"kpi_keys": self.kpi, "time_filter": self.tf_year, "filters": ["region"]})
        self.assertTrue(is_error)
        self.assertEqual(payload["error"], "invalid_argument")
        self.assertEqual(payload["argument"], "filters")
        self.assertNotIn("dictionary update sequence", payload["message"])

    def test_a_bare_string_for_a_list_shaped_candidates_argument_names_candidates_not_unknown_kpi(self):
        payload, is_error = self.reg.dispatch(
            "test_confounders", {"kpi_key": self.kpi, "cause_kpi": self.ratio_kpi,
                                "candidates": "not_a_list",
                                "period_a": self.tf_year, "period_b": self.tf_year2})
        self.assertTrue(is_error)
        self.assertEqual(payload["error"], "invalid_argument")
        self.assertEqual(payload["argument"], "candidates")


class TestValidateRaisesAMachineReadableInvalidArgumentError(RetailRegistryTestCase):
    def test_the_error_names_the_offending_argument_and_the_reason(self):
        schema = self.reg._schemas["query_kpi"]
        with self.assertRaises(InvalidArgumentError) as ctx:
            airlock.validate("query_kpi", schema, {"kpi_keys": self.kpi, "time_filter": self.tf_year,
                                                   "limit": "five"})
        payload = ctx.exception.to_payload()
        self.assertEqual(payload["argument"], "limit")
        self.assertIn("five", payload["message"])

    def test_a_bad_enum_value_carries_the_schemas_own_valid_alternatives(self):
        schema = self.reg._schemas["rank_entities"]
        with self.assertRaises(InvalidArgumentError) as ctx:
            airlock.validate("rank_entities", schema,
                            {"kpi_key": self.kpi, "dimension": "not_a_real_dimension",
                             "time_filter": self.tf_year})
        payload = ctx.exception.to_payload()
        self.assertEqual(payload["argument"], "dimension")
        self.assertIn(self.dim, payload["valid_alternatives"])

    def test_a_missing_required_argument_names_the_missing_key(self):
        schema = self.reg._schemas["query_kpi"]
        with self.assertRaises(InvalidArgumentError) as ctx:
            airlock.validate("query_kpi", schema, {"time_filter": self.tf_year})
        self.assertEqual(ctx.exception.to_payload()["argument"], "kpi_keys")

    def test_a_valid_call_raises_nothing(self):
        schema = self.reg._schemas["query_kpi"]
        airlock.validate("query_kpi", schema, {"kpi_keys": self.kpi, "time_filter": self.tf_year})


class TestFuzzingEveryRegisteredToolNeverRaisesAndAlwaysReturnsJsonSafeData(RetailRegistryTestCase):
    """
    A deterministic battery (not `random`, so a failure is reproducible),
    applied to one parameter at a time of every tool's own known-good
    example from `RetailRegistryTestCase.examples`. `None` is asserted
    strictly: no schema in this registry declares `"null"` as a legal type
    for any parameter, so every `None` substitution is a guaranteed schema
    violation and must be rejected -- if this ever stops being true for a
    real parameter, that parameter's schema is the thing to fix, not this
    test. The remaining values (`True`, `[]`, `{}`, a float, a long bogus
    string) can legitimately be valid for some parameters (an empty
    candidate list, a free-text search phrase); those are only required to
    never crash and never leak something that fails to round-trip as JSON.
    """

    BATTERY = (True, [], {}, 3.14, "zzz_not_a_real_value_zzz")

    def test_every_none_substitution_is_rejected_by_the_airlock(self):
        for name, args in self.examples.items():
            schema = self.reg._schemas[name]
            for key in args:
                with self.subTest(tool=name, param=key):
                    mutated = dict(args)
                    mutated[key] = None
                    payload, is_error = self.reg.dispatch(name, mutated)
                    assert_json_safe(payload)
                    self.assertTrue(is_error, f"{name}.{key}=None executed instead of being rejected")
                    self.assertIn("error", payload)

    def test_the_rest_of_the_battery_never_crashes_and_stays_json_safe(self):
        for name, args in self.examples.items():
            for key in args:
                for value in self.BATTERY:
                    with self.subTest(tool=name, param=key, value=repr(value)):
                        mutated = dict(args)
                        mutated[key] = value
                        payload, is_error = self.reg.dispatch(name, mutated)
                        assert_json_safe(payload)
                        if is_error:
                            self.assertIn("error", payload)
                            self.assertIsInstance(payload["error"], str)


class TestTheAirlockNeverRejectsAValidCall(RetailRegistryTestCase):
    """The guard in the other direction: A3 must narrow what reaches the
    engines, never reject a call `test_registry.py`'s own examples already
    prove is correct."""

    def test_every_known_good_example_still_passes(self):
        for name, args in self.examples.items():
            with self.subTest(tool=name):
                payload, is_error = self.reg.dispatch(name, args)
                self.assertFalse(is_error, f"{name}: {payload}")


if __name__ == "__main__":
    unittest.main()
