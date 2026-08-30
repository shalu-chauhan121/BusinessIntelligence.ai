"""
Noise filtering and data quality -- `agent/quality.py`.

`QualityEngine.filter_material` re-expresses the materiality gate that today
lives only inside `signals.material_signals` (`engines/signals.py:62`), which
is too coupled to the whole `observe()` dict to wrap. The tests that matter
most here are the composition test -- `compare.Comparison`/`Significance`
payloads must feed straight into the gate with no adapter -- and the
distinction between a candidate that failed a significance test and one that
was never given one.

`QualityEngine.check_data_quality` wires up `analysis.coverage_report`
(`engines/analysis.py:228`), which has zero callers anywhere in the repo, and
adds the one fact it could never state: `prepare`'s `fillna(0.0)`
(`engines/metrics.py:344`) makes a fabricated zero and a real zero
byte-identical in the frame -- `schema.imputed_cells` is the one place that
difference survives, and the tests here are built to prove it does.
"""
from __future__ import annotations

import unittest

import pandas as pd

from app.agent.compare import ComparisonEngine
from app.agent.contract_api import ContractAPI
from app.agent.quality import QualityEngine
from app.config import get_settings
from app.engines import metrics

from .base import (EngineTestCase, FIXTURES, assert_json_safe,
                   assert_no_prose_leak, contracted)


class MaterialityTestCase(EngineTestCase):
    """The shared retail fixture, with a compiled contract."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.rdf, cls.rschema = contracted("business_metrics_sample.csv")
        cls.api = ContractAPI(cls.rschema)
        cls.qe = QualityEngine(cls.api)


# ---------------------------------------------------------------------------
# filter_material()
# ---------------------------------------------------------------------------
class TestMaterialityThresholds(MaterialityTestCase):
    def test_default_thresholds_come_from_settings(self):
        result = self.qe.filter_material([{"kpi": "revenue", "change_pct": 10.0}])
        s = get_settings()
        self.assertEqual(result.min_pct, s.min_material_change_pct)
        self.assertEqual(result.min_z, s.anomaly_z_threshold)

    def test_below_threshold_is_immaterial_above_is_material(self):
        result = self.qe.filter_material(
            [{"kpi": "revenue", "change_pct": 2.9}, {"kpi": "revenue", "change_pct": 3.1}],
            min_pct=3.0)
        self.assertFalse(result.candidates[0].is_material)
        self.assertEqual(result.candidates[0].reason, "below_pct")
        self.assertTrue(result.candidates[1].is_material)
        self.assertIsNone(result.candidates[1].reason)

    def test_the_boundary_value_itself_is_material(self):
        result = self.qe.filter_material([{"kpi": "revenue", "change_pct": 3.0}], min_pct=3.0)
        self.assertTrue(result.candidates[0].is_material)

    def test_overriding_min_pct_moves_the_boundary(self):
        result = self.qe.filter_material([{"kpi": "revenue", "change_pct": 5.0}], min_pct=10.0)
        self.assertFalse(result.candidates[0].is_material)


class TestMissingRobustZIsUntestedNotFailed(MaterialityTestCase):
    """A candidate with no `robust_z` (a plain `Comparison`, not a
    `Significance`) must never be reported as having failed a significance
    test it was never given -- the distinction the old `5.0` literal
    (`signals.py:106`) could not express."""

    def test_no_z_is_magnitude_only_and_can_still_be_material(self):
        result = self.qe.filter_material([{"kpi": "revenue", "change_pct": 10.0}])
        self.assertEqual(result.candidates[0].basis, "magnitude_only")
        self.assertTrue(result.candidates[0].is_material)
        self.assertIsNone(result.candidates[0].reason)

    def test_a_z_below_threshold_is_reason_below_z_not_below_pct(self):
        result = self.qe.filter_material(
            [{"kpi": "revenue", "change_pct": 10.0, "robust_z": 0.5}], min_z=2.0)
        self.assertEqual(result.candidates[0].basis, "magnitude_and_significance")
        self.assertFalse(result.candidates[0].is_material)
        self.assertEqual(result.candidates[0].reason, "below_z")

    def test_no_change_at_all_is_reason_no_change(self):
        result = self.qe.filter_material([{"kpi": "revenue", "change_pct": 0.0}])
        self.assertFalse(result.candidates[0].is_material)
        self.assertEqual(result.candidates[0].reason, "no_change")


class TestComposesDirectlyWithO4Payloads(MaterialityTestCase):
    def test_comparison_and_significance_payloads_feed_in_with_no_adapter(self):
        ce = ComparisonEngine(self.api)
        cmp = ce.compare(
            self.rdf, "revenue",
            {"type": "quarter", "year": 2026, "quarter": 1},
            {"type": "quarter", "year": 2024, "quarter": 1}).to_payload()
        sig = ce.significance(
            self.rdf, "revenue", {"type": "quarter", "year": 2024, "quarter": 2}).to_payload()

        result = self.qe.filter_material([cmp, sig])
        self.assertEqual(result.considered, 2)
        self.assertEqual(result.candidates[0].basis, "magnitude_only")
        self.assertEqual(result.candidates[1].basis, "magnitude_and_significance")


class TestNoHardcodedThresholdSurvives(unittest.TestCase):
    def test_the_stray_5_0_literal_is_not_in_executable_code(self):
        import re
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        source = (root / "app/agent/quality.py").read_text(encoding="utf-8")
        code = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
        code = re.sub(r"#.*", "", code)
        self.assertNotIn("5.0", code)


# ---------------------------------------------------------------------------
# check_data_quality() -- zero-fill artefact
# ---------------------------------------------------------------------------
class TestZeroFillArtefactIsExactlyKnowable(unittest.TestCase):
    """The core reason `check_data_quality` exists: after `prepare` a
    fabricated zero and a real zero are byte-identical in the frame -- this
    fixture has one of each in the same column, so the report must
    distinguish them."""

    @classmethod
    def setUpClass(cls):
        raw = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "region": ["North", "South", "North", "South"],
            "revenue": [100.0, None, 0.0, 50.0],   # one genuine null, one genuine zero
        })
        cls.schema = metrics.detect_schema(raw)
        cls.df = metrics.prepare(raw, cls.schema)
        cls.api = ContractAPI(cls.schema)
        cls.qe = QualityEngine(cls.api)

    def test_the_injected_null_reports_an_exact_imputed_count(self):
        result = self.qe.check_data_quality(self.df, {"type": "all"})
        col = next(c for c in result.columns if c.column == "revenue")
        self.assertEqual(col.imputed_rows, 1)

    def test_the_real_zero_is_not_counted_as_imputed(self):
        result = self.qe.check_data_quality(self.df, {"type": "all"})
        col = next(c for c in result.columns if c.column == "revenue")
        self.assertEqual(col.zero_rows, 2)          # the fabricated zero + the real one
        self.assertLess(col.imputed_rows, col.zero_rows)
        self.assertTrue(result.zero_fill_applied)


class TestPreserveMissingLeavesNoImputationRecord(unittest.TestCase):
    def test_imputed_cells_is_none_and_the_report_says_so(self):
        raw = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "revenue": [100.0, None],
        })
        schema = metrics.detect_schema(raw)
        df = metrics.prepare(raw, schema, preserve_missing=True)
        api = ContractAPI(schema)
        qe = QualityEngine(api)

        result = qe.check_data_quality(df, {"type": "all"})
        self.assertFalse(result.zero_fill_applied)
        col = next(c for c in result.columns if c.column == "revenue")
        self.assertIsNone(col.imputed_rows)
        self.assertEqual(col.null_rows, 1)


# ---------------------------------------------------------------------------
# check_data_quality() -- dimension coverage across two periods
# ---------------------------------------------------------------------------
class TestDimensionCoverageAcrossTwoPeriods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = pd.DataFrame({
            "date": ["2024-01-15", "2024-01-16", "2024-04-15", "2024-04-16", "2024-04-17"],
            "region": ["North", "South", "North", "East", "East"],
            "revenue": [10.0, 20.0, 15.0, 5.0, 5.0],
        })
        cls.schema = metrics.detect_schema(raw)
        cls.df = metrics.prepare(raw, cls.schema)
        cls.api = ContractAPI(cls.schema)
        cls.qe = QualityEngine(cls.api)

    def test_appeared_and_disappeared_members_are_both_reported(self):
        result = self.qe.check_data_quality(
            self.df, {"type": "quarter", "year": 2024, "quarter": 2},
            baseline={"type": "quarter", "year": 2024, "quarter": 1})
        dim = next(d for d in result.dimensions if d.dimension == "region")
        self.assertEqual(set(dim.appeared), {"East"})
        self.assertEqual(set(dim.disappeared), {"South"})

    def test_no_baseline_means_no_appeared_disappeared_block(self):
        result = self.qe.check_data_quality(self.df, {"type": "quarter", "year": 2024, "quarter": 2})
        dim = next(d for d in result.dimensions if d.dimension == "region")
        self.assertIsNone(dim.baseline_members)
        self.assertEqual(dim.appeared, ())
        self.assertEqual(dim.disappeared, ())
        self.assertIsNone(result.baseline_rows)
        self.assertIsNone(result.row_change_pct)


# ---------------------------------------------------------------------------
# integrity / prose guard / determinism / portability
# ---------------------------------------------------------------------------
class TestProseGuard(MaterialityTestCase):
    BANNED = {"issues", "filter_note", "note", "significance_note"}

    def _assert_no_banned_keys(self, payload, path: str = "$") -> None:
        if isinstance(payload, dict):
            for key, value in payload.items():
                self.assertNotIn(key, self.BANNED, f"{path}.{key} is a banned prose field")
                self._assert_no_banned_keys(value, f"{path}.{key}")
        elif isinstance(payload, list):
            for i, item in enumerate(payload):
                self._assert_no_banned_keys(item, f"{path}[{i}]")

    def test_no_banned_prose_key_survives_in_a_data_quality_payload(self):
        result = self.qe.check_data_quality(self.rdf, {"type": "quarter", "year": 2026, "quarter": 1})
        self._assert_no_banned_keys(result.to_payload())

    def test_no_string_field_outside_the_allowlist_looks_like_prose(self):
        result = self.qe.check_data_quality(self.rdf, {"type": "quarter", "year": 2026, "quarter": 1})
        assert_no_prose_leak(result.to_payload())

    def test_the_payload_survives_json_dumps_with_no_nan_or_inf(self):
        result = self.qe.check_data_quality(
            self.rdf, {"type": "quarter", "year": 2026, "quarter": 1},
            baseline={"type": "quarter", "year": 2025, "quarter": 4})
        assert_json_safe(result.to_payload())
        assert_json_safe(self.qe.filter_material([{"kpi": "revenue", "change_pct": None}]).to_payload())


class TestDeterminism(MaterialityTestCase):
    def test_identical_input_gives_identical_output_across_runs(self):
        first = self.qe.check_data_quality(self.rdf, {"type": "quarter", "year": 2026, "quarter": 1}).to_payload()
        second = self.qe.check_data_quality(self.rdf, {"type": "quarter", "year": 2026, "quarter": 1}).to_payload()
        self.assertEqual(first, second)


class TestPortabilityAcrossDatasets(unittest.TestCase):
    """The hospital fixture has a single dimension (`department`) -- a real
    edge case a retail-only assumption would miss."""

    @classmethod
    def setUpClass(cls):
        cls.df, cls.schema = contracted("hospital_sample.csv", FIXTURES)
        cls.qe = QualityEngine(ContractAPI(cls.schema))

    def test_check_data_quality_works_on_a_single_dimension_dataset(self):
        result = self.qe.check_data_quality(
            self.df, {"type": "quarter", "year": 2025, "quarter": 2},
            baseline={"type": "quarter", "year": 2025, "quarter": 1})
        self.assertEqual(len(result.dimensions), 1)
        self.assertEqual(result.dimensions[0].dimension, "department")


if __name__ == "__main__":
    unittest.main()
