"""
Resolver propagation -- the KPI Contract must be honoured everywhere `compute`
is called from an engine, not only at the one or two call sites someone
remembered to thread it through.

Batch 1 (C2) exists because six call sites had `resolver` in scope and simply
did not pass it: `contest.py` (temporal_check x2, consistency_check),
`act.py` (monitoring_threshold), `drivers.py` (member_persistence x2). Each of
those falls back to the seed `METRICS` registry, which is entirely retail
vocabulary (revenue, orders, gross_margin_pct, ...) and contains none of a
hospital's clinical KPIs. Since those KPIs are ratios, not raw columns, the
silent fallback does not even find a column to sum -- it returns NaN.

The hospital fixture is the reproduction: `recovery_rate`, `readmission_rate`,
`mortality_rate` and `avg_length_of_stay` are contract-only ratios absent from
the seed registry entirely.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from app.engines.analysis import correlate, detect_onset, member_change_table, weekly_frame
from app.engines.metrics import compute, detect_schema, prepare
from app.kpi import service as kpi_service
from app.kpi.resolver import compile_contract

from .base import EngineTestCase

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class HospitalResolverTestCase(EngineTestCase):
    """A dataset whose KPIs are entirely absent from the seed METRICS registry."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        raw = pd.read_csv(FIXTURES / "hospital_sample.csv")
        cls.h_schema = detect_schema(raw)
        cls.h_df = prepare(raw, cls.h_schema)
        contract = kpi_service.bootstrap(
            "u", {"_id": "hosp1", "filename": "hospital.csv"}, cls.h_df, cls.h_schema)
        cls.h_resolver = compile_contract(contract)
        # Sanity: the fixture must actually exercise the bug, or this whole
        # suite is testing nothing.
        assert "recovery_rate" in cls.h_resolver
        assert "recovery_rate" not in cls._seed_registry_keys()

    @staticmethod
    def _seed_registry_keys():
        from app.engines.metrics import METRICS
        return set(METRICS.keys())


class TestComputeRequiresResolver(HospitalResolverTestCase):
    def test_contract_only_ratio_is_nan_without_the_resolver(self):
        """
        Documents the failure mode being fixed: `compute` with no resolver
        cannot see a contract-only KPI at all. This is the "before" behaviour
        that every call site must stop exhibiting.
        """
        without = compute(self.h_df, "recovery_rate")
        self.assertNotEqual(without, without, "expected NaN when the resolver is omitted")

    def test_contract_only_ratio_resolves_with_the_resolver(self):
        with_resolver = compute(self.h_df, "recovery_rate", self.h_resolver)
        self.assertEqual(with_resolver, with_resolver, "must not be NaN")
        self.assertGreater(with_resolver, 0)
        self.assertLess(with_resolver, 100)

    def test_contract_kpi_that_shadows_a_seed_key_prefers_the_contract(self):
        """cost_of_goods exists in both the seed registry and this contract;
        the contract's own source-field binding must win."""
        self.assertIn("cost_of_goods", self._seed_registry_keys())
        self.assertIn("cost_of_goods", self.h_resolver)
        contract_value = compute(self.h_df, "cost_of_goods", self.h_resolver)
        self.assertEqual(contract_value, self.h_df["treatment_cost"].sum())


class TestWeeklyFrameResolverPropagation(HospitalResolverTestCase):
    def test_weekly_frame_is_all_nan_without_a_resolver_today(self):
        """
        THE RED TEST. This documents current (pre-fix) behaviour: without a
        resolver argument, `weekly_frame` cannot compute a contract-only ratio
        at all, and returns a value column that is NaN in every row.
        """
        weeks = weekly_frame(self.h_df, "recovery_rate")
        self.assertTrue(weeks["value"].isna().all(),
                        "if this now fails, weekly_frame already threads the "
                        "resolver through -- update this test to describe the "
                        "fixed behaviour instead")

    def test_weekly_frame_resolves_a_contract_only_ratio_with_resolver(self):
        weeks = weekly_frame(self.h_df, "recovery_rate", self.h_resolver)
        self.assertFalse(weeks["value"].isna().all(), "expected real per-week ratios")
        self.assertGreater(len(weeks.dropna(subset=["value"])), 10)
        for v in weeks["value"].dropna():
            self.assertGreaterEqual(v, 0)
            self.assertLessEqual(v, 100)

    def test_detect_onset_finds_a_real_onset_once_resolver_is_threaded(self):
        """
        Downstream proof: with no resolver, weekly_frame is all-NaN, detect_onset
        cannot find >=3 non-NaN baseline points and returns None -- the temporal
        check silently no-ops. With the resolver, there is a real series to test.
        """
        broken = weekly_frame(self.h_df, "recovery_rate")
        self.assertIsNone(detect_onset(broken, baseline_weeks=8))

        fixed = weekly_frame(self.h_df, "recovery_rate", self.h_resolver)
        # Not asserting an onset necessarily exists (the data may not have one),
        # only that detect_onset can now see enough real values to try.
        base_values = fixed["value"].astype(float).values[:8]
        non_nan = base_values[base_values == base_values]
        self.assertGreaterEqual(len(non_nan), 3)


class TestMemberChangeTableResolverPropagation(HospitalResolverTestCase):
    def test_member_change_table_is_nan_without_a_resolver_today(self):
        """THE RED TEST for the cross-sectional path."""
        cur = self.h_df[self.h_df["_quarter"] == self.h_df["_quarter"].max()]
        base = self.h_df[self.h_df["_quarter"] != self.h_df["_quarter"].max()]
        table = member_change_table(cur, base, "department", ["recovery_rate"])
        self.assertTrue(table["recovery_rate__cur"].isna().all())
        self.assertTrue(table["recovery_rate__base"].isna().all())

    def test_member_change_table_resolves_with_resolver(self):
        cur = self.h_df[self.h_df["_quarter"] == self.h_df["_quarter"].max()]
        base = self.h_df[self.h_df["_quarter"] != self.h_df["_quarter"].max()]
        table = member_change_table(cur, base, "department", ["recovery_rate"],
                                    resolver=self.h_resolver)
        self.assertFalse(table["recovery_rate__cur"].isna().all())
        self.assertFalse(table["recovery_rate__base"].isna().all())

    def test_correlate_gets_real_n_once_resolver_is_threaded(self):
        """
        Downstream proof for the second silent no-op: with no resolver,
        `correlate` sees NaN in every member, masks them all out, and reports
        n=0 -- which score_hypothesis's `if corr is not None` guard then
        silently skips. With the resolver there is real cross-sectional signal.
        """
        cur = self.h_df[self.h_df["_quarter"] == self.h_df["_quarter"].max()]
        base = self.h_df[self.h_df["_quarter"] != self.h_df["_quarter"].max()]

        broken_table = member_change_table(cur, base, "department",
                                           ["recovery_rate", "admissions"])
        broken_corr = correlate(broken_table, "recovery_rate", "admissions")
        self.assertIsNone(broken_corr["r"])
        self.assertEqual(broken_corr["n"], 0)

        fixed_table = member_change_table(cur, base, "department",
                                          ["recovery_rate", "admissions"],
                                          resolver=self.h_resolver)
        fixed_corr = correlate(fixed_table, "recovery_rate", "admissions")
        self.assertGreater(fixed_corr["n"], 0)
        self.assertIsNotNone(fixed_corr["r"])


if __name__ == "__main__":
    unittest.main()
