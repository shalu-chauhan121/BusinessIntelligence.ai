"""
`agent.context.AgentContext` -- the per-request handle the agent loop
threads everywhere.

Pins the three things nothing owned before this module existed: a dataset
loads exactly once per context even across a multi-tool trace (reusing
`dataset_service.load`'s own (path, mtime) cache, not re-deriving it), the
returned `df` is shared by reference rather than copied (so building a
context must never itself mutate it), and a turn budget is recorded and
reported rather than enforced a second time -- `LLMClient._call_with_tools`
already enforces `max_turns` itself.
"""
from __future__ import annotations

import unittest
from unittest import mock

import pandas as pd

from app.agent.context import AgentContext, Budget
from app.services import dataset_service

from .base import EngineTestCase


def _counting_patch(module, func_name: str):
    """The same call-counting idiom `test_scan.py` / `test_correlation.py`
    already use -- wraps the real function so it still runs, and counts."""
    original = getattr(module, func_name)
    calls = {"n": 0}

    def wrapper(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    return mock.patch.object(module, func_name, wrapper), calls


class TestBuildingAContextLoadsTheDatasetExactlyOnce(EngineTestCase):
    def test_a_ten_tool_trace_makes_one_dataset_service_load_call(self):
        patch, calls = _counting_patch(dataset_service, "load")
        with patch:
            ctx = AgentContext.build(self.uid, self.dataset)
            kpi = ctx.api.list_kpi_keys()[0]
            tf = {"type": "year", "year": 2024}
            for _ in range(10):
                payload, is_error = ctx.registry.dispatch("query_kpi", {"kpi_keys": kpi, "time_filter": tf})
                self.assertFalse(is_error)
        self.assertEqual(calls["n"], 1)


class TestTheFrameIsSharedNotCopied(EngineTestCase):
    def test_two_contexts_on_the_same_dataset_share_the_same_dataframe_object(self):
        ctx_a = AgentContext.build(self.uid, self.dataset)
        ctx_b = AgentContext.build(self.uid, self.dataset)
        self.assertIs(ctx_a.df, ctx_b.df)

    def test_building_a_context_does_not_mutate_the_underlying_frame(self):
        before = self.df.copy(deep=True)
        AgentContext.build(self.uid, self.dataset)
        pd.testing.assert_frame_equal(self.df, before)


class TestTwoContextsShareNoMutableState(EngineTestCase):
    def test_schemas_are_independent_dataclass_copies(self):
        ctx_a = AgentContext.build(self.uid, self.dataset)
        ctx_b = AgentContext.build(self.uid, self.dataset)
        self.assertIsNot(ctx_a.schema, ctx_b.schema)

    def test_each_context_gets_its_own_registry_and_contract_api(self):
        ctx_a = AgentContext.build(self.uid, self.dataset)
        ctx_b = AgentContext.build(self.uid, self.dataset)
        self.assertIsNot(ctx_a.registry, ctx_b.registry)
        self.assertIsNot(ctx_a.api, ctx_b.api)
        # Independent objects, but built from the same underlying contract,
        # so they agree on what this dataset actually measures.
        self.assertEqual(ctx_a.api.list_kpi_keys(), ctx_b.api.list_kpi_keys())


class TestBudgetTracksTurnsWithoutEnforcingThem(unittest.TestCase):
    def test_a_fresh_budget_has_its_full_allowance_remaining(self):
        b = Budget(max_turns=12)
        self.assertEqual(b.remaining_turns, 12)
        self.assertFalse(b.exhausted)

    def test_recording_turns_reduces_the_remaining_allowance(self):
        b = Budget(max_turns=12)
        b.record(5)
        self.assertEqual(b.remaining_turns, 7)
        self.assertFalse(b.exhausted)

    def test_exhaustion_is_a_reportable_property_never_an_exception(self):
        b = Budget(max_turns=3)
        b.record(3)
        self.assertTrue(b.exhausted)
        self.assertEqual(b.remaining_turns, 0)

    def test_recording_more_turns_than_the_allowance_never_goes_negative(self):
        b = Budget(max_turns=3)
        b.record(10)
        self.assertEqual(b.remaining_turns, 0)
        self.assertTrue(b.exhausted)


class TestAContextDefaultsItsBudgetFromSettings(EngineTestCase):
    def test_no_max_turns_override_uses_the_configured_default(self):
        from app.config import get_settings
        ctx = AgentContext.build(self.uid, self.dataset)
        self.assertEqual(ctx.budget.max_turns, get_settings().llm_max_turns)

    def test_an_explicit_max_turns_override_wins(self):
        ctx = AgentContext.build(self.uid, self.dataset, max_turns=3)
        self.assertEqual(ctx.budget.max_turns, 3)


if __name__ == "__main__":
    unittest.main()
