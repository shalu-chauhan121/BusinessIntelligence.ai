"""Stage 1 — OBSERVE. Numbers must be reproducible and independently checkable."""
from __future__ import annotations

import unittest

import pandas as pd

from app.engines.metrics import compute, detect_schema, prepare
from app.engines.observe import Timeframe, decompose_dimension, observe, slice_period
from app.engines.investigate import investigate
from app.engines.contest import contest
from app.engines.act import act

from .base import EngineTestCase


class TestSchema(EngineTestCase):
    def test_schema_detection(self):
        self.assertEqual(self.schema.date_column, "date")
        self.assertEqual(set(self.schema.dimensions), {"region", "product", "channel", "segment"})
        self.assertIn("revenue", self.schema.available_kpis)
        self.assertIn("gross_margin_pct", self.schema.available_kpis)   # derived, never uploaded
        self.assertEqual(self.schema.grain, "weekly")
        self.assertGreaterEqual(len(self.schema.quarters), 12)


class TestKpiArithmetic(EngineTestCase):
    def test_kpi_matches_raw_pandas(self):
        result = observe(self.df, self.schema, "revenue", Timeframe(2026, 2))
        expected = self.df[(self.df["_year"] == 2026) & (self.df["_quarter"] == 2)]["revenue"].sum()
        self.assertClose(result["current_value"], expected)

    def test_ratio_kpi_is_not_an_average_of_averages(self):
        cur = slice_period(self.df, Timeframe(2026, 2))
        engine = compute(cur, "gross_margin_pct")
        manual = (cur["revenue"].sum() - cur["cost_of_goods"].sum()) / cur["revenue"].sum() * 100
        naive_row_mean = ((cur["revenue"] - cur["cost_of_goods"]) / cur["revenue"] * 100).mean()
        self.assertClose(engine, manual)
        self.assertGreater(abs(engine - naive_row_mean), 1e-6)

    def test_inventory_is_aggregated_as_a_level_not_a_sum(self):
        cur = slice_period(self.df, Timeframe(2026, 2))
        self.assertClose(compute(cur, "inventory_units"), cur["inventory_units"].mean())


class TestDriverDecomposition(EngineTestCase):
    def test_contributions_sum_to_100(self):
        cur = slice_period(self.df, Timeframe(2026, 2))
        base = slice_period(self.df, Timeframe(2026, 1))
        for dim in self.schema.dimensions:
            rows = decompose_dimension(cur, base, dim, "revenue", max_items=10)
            total = sum(r["contribution_pct"] for r in rows if r["contribution_pct"] is not None)
            self.assertAlmostEqual(total, 100.0, places=2, msg=dim)

    def test_deltas_sum_to_total_delta(self):
        cur = slice_period(self.df, Timeframe(2026, 2))
        base = slice_period(self.df, Timeframe(2026, 1))
        rows = decompose_dimension(cur, base, "region", "revenue", max_items=10)
        self.assertClose(sum(r["change_abs"] for r in rows),
                         cur["revenue"].sum() - base["revenue"].sum())

    def test_ratio_decomposition_splits_rate_and_mix(self):
        cur = slice_period(self.df, Timeframe(2026, 2))
        base = slice_period(self.df, Timeframe(2026, 1))
        rows = decompose_dimension(cur, base, "product", "gross_margin_pct", max_items=10)
        for r in rows:
            if r.get("is_aggregate"):
                continue
            self.assertIsNotNone(r["effects"])
            self.assertAlmostEqual(
                r["effects"]["rate_effect"] + r["effects"]["mix_effect"],
                (r["contribution_pct"] / 100.0) * (compute(cur, "gross_margin_pct") - compute(base, "gross_margin_pct")),
                places=6,
            )

    def test_over_index_flags_real_drivers_only(self):
        """A segment that moves exactly in line with its size is not a driver."""
        result = observe(self.df, self.schema, "revenue", Timeframe(2026, 2))
        flagged = {d["name"] for d in result["top_drivers"] if d.get("is_disproportionate")}
        self.assertIn("North", flagged)
        self.assertIn("Product A", flagged)
        self.assertNotIn("Enterprise", flagged)     # 60% of the change but also 60% of the business


class TestSignificance(EngineTestCase):
    def test_planted_anomaly_is_detected(self):
        result = observe(self.df, self.schema, "revenue", Timeframe(2026, 2))
        self.assertEqual(result["history_status"], "sufficient_history")
        self.assertTrue(result["anomaly"])
        self.assertEqual(result["verdict"], "meaningful_signal")
        self.assertLess(result["change_pct"], -15)

    def test_quiet_quarter_is_not_called_an_anomaly(self):
        """The whole point of Observe: ordinary movement must not be treated as signal."""
        result = observe(self.df, self.schema, "revenue", Timeframe(2026, 1))
        self.assertFalse(result["anomaly"])
        self.assertEqual(result["verdict"], "within_normal_variation")

    def test_robust_z_stays_plausible(self):
        for year, quarter in [(2025, 2), (2025, 4), (2026, 1), (2026, 2)]:
            z = observe(self.df, self.schema, "revenue", Timeframe(year, quarter))["significance"]["robust_z"]
            self.assertTrue(z is None or abs(z) < 25, f"{year}Q{quarter}: z={z}")

    def test_year_over_year_comparison(self):
        result = observe(self.df, self.schema, "revenue", Timeframe(2026, 2), comparison="year_over_year")
        self.assertEqual(result["baseline_timeframe"]["label"], "2025-Q2")
        expected = self.df[(self.df["_year"] == 2025) & (self.df["_quarter"] == 2)]["revenue"].sum()
        self.assertClose(result["baseline_value"], expected)

    def test_full_year_timeframe(self):
        result = observe(self.df, self.schema, "revenue", Timeframe(2025, None))
        self.assertEqual(result["timeframe"]["label"], "FY2025")
        self.assertClose(result["current_value"], self.df[self.df["_year"] == 2025]["revenue"].sum())


class TestRobustness(EngineTestCase):
    def test_every_available_kpi_runs(self):
        for kpi in self.schema.available_kpis:
            result = observe(self.df, self.schema, kpi, Timeframe(2026, 2))
            self.assertEqual(result["kpi"], kpi)
            self.assertIsNotNone(result["current_value"])

    def test_minimal_dataset_without_dimensions(self):
        raw = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=130, freq="W").astype(str),
            "revenue": [1000 + (i % 7) * 13 for i in range(130)],
        })
        schema = detect_schema(raw)
        df = prepare(raw, schema)
        self.assertEqual(schema.dimensions, [])
        self.assertTrue(any("dimension" in w.lower() for w in schema.warnings))
        result = observe(df, schema, "revenue", Timeframe(2025, 2))
        self.assertEqual(result["drivers"], {})

    def test_missing_date_column_is_rejected_clearly(self):
        with self.assertRaises(ValueError):
            detect_schema(pd.DataFrame({"revenue": [1, 2, 3], "region": ["a", "b", "c"]}))


class TestSparseHistory(EngineTestCase):
    def _observe(self, dates):
        raw = pd.DataFrame({"date": dates, "revenue": [100] * len(dates)})
        schema = detect_schema(raw)
        return prepare(raw, schema), schema

    def test_sparse_history_has_current_value_without_trend_conclusions(self):
        df, schema = self._observe(["2025-10-01", "2026-01-01", "2026-04-01"])
        result = observe(df, schema, "revenue", Timeframe(2026, 2))
        self.assertEqual(result["history_status"], "sparse_history")
        self.assertIsNotNone(result["current_value"])
        self.assertIsNone(result["significance"]["robust_z"])
        self.assertIsNone(result["significance"]["normal_range"])

    def test_new_kpi_has_no_fabricated_trend_or_confidence(self):
        df, schema = self._observe(["2026-04-01"])
        observation = observe(df, schema, "revenue", Timeframe(2026, 2))
        self.assertEqual(observation["history_status"], "newly_launched")
        investigation = investigate(df, schema, observation, self.uid, llm=None)
        contested = contest(df, schema, observation, investigation, self.uid, llm=None)
        action = act(df, observation, investigation, contested, llm=None)
        self.assertEqual(contested["ranking"], [])
        self.assertIsNone(action["recommendations"][0]["based_on"]["confidence"])


if __name__ == "__main__":
    unittest.main()
