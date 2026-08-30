"""
`correlate_kpis` / `correlate_kpi_matrix` -- `agent/correlate.py`.

This module does not wrap `analysis.correlate` (`engines/analysis.py:139`);
it recomputes the same arithmetic -- a finite-value mask, then
`numpy.corrcoef` -- through `QueryEngine`/`SeriesEngine`, so the tests that
matter most are: the recomputed r matches `numpy.corrcoef` directly
(`TestPearsonMatchesNumpy`); the three failure modes `analysis.correlate`
collapses into one identical sentence are told apart here
(`TestFailureModesAreDistinguishable`); differencing a trending pair actually
changes the answer, proving the transform choice rather than merely
exercising it (`TestDifferencingActuallyMatters`); and a candidate matrix
produces byte-identical rows to calling `correlate_kpis` on each pair alone,
at a bounded query cost (`TestMatrixEqualsIndividualCalls`).

Auto dimension selection is checked against the hospital and school fixtures
specifically, because the equivalent legacy rule
(`contest._cross_sectional_dimension`, `engines/contest.py:130`) is gated
behind a hardcoded `("region","product","channel","segment")` whitelist that
silently disables it on exactly those two fixtures.
"""
from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import pandas as pd

from app.agent.contract_api import ContractAPI
from app.agent.correlate import CorrelationEngine, pearson_correlate
from app.agent.errors import InvalidArgumentError, UnknownDimensionError, UnknownKpiError
from app.agent.query import QueryEngine
from app.agent.series import SeriesEngine
from app.engines import metrics

from .base import (EngineTestCase, FIXTURES, assert_json_safe,
                   assert_no_prose_leak, contracted)

BANNED = {"interpretation", "detail", "narrative", "note"}


def _assert_no_banned_keys(payload, path: str = "$") -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            assert key not in BANNED, f"{path}.{key} is a banned prose field"
            _assert_no_banned_keys(value, f"{path}.{key}")
    elif isinstance(payload, list):
        for i, item in enumerate(payload):
            _assert_no_banned_keys(item, f"{path}[{i}]")


def _counting_patch(cls, method_name: str):
    """Patches `cls.method_name` with a wrapper that counts calls while still
    executing the real implementation. Mirrors `test_scan.py`'s helper."""
    original = getattr(cls, method_name)
    calls = {"n": 0}

    def wrapper(self, *args, **kwargs):
        calls["n"] += 1
        return original(self, *args, **kwargs)

    return mock.patch.object(cls, method_name, wrapper), calls


class CorrelationTestCase(EngineTestCase):
    """The shared retail fixture, with a compiled contract."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.rdf, cls.rschema = contracted("business_metrics_sample.csv")
        cls.api = ContractAPI(cls.rschema)
        cls.ce = CorrelationEngine(cls.api)
        cls.period_a = {"type": "quarter", "year": 2026, "quarter": 2}
        cls.period_b = {"type": "quarter", "year": 2026, "quarter": 1}


def _member_frame(a_values, b_values, n_dates_per_member=1):
    """A synthetic frame with `len(a_values)` members of a `region` dimension,
    two periods, and two seed KPIs (`revenue`, `marketing_spend`) whose
    per-member % changes are exactly the ones the caller wants correlated."""
    rows = []
    for i, (base_a, cur_a, base_b, cur_b) in enumerate(zip(*a_values, *b_values)):
        member = f"R{i}"
        rows.append({"region": member, "revenue": base_a, "marketing_spend": base_b,
                    "date": "2026-02-01"})
        rows.append({"region": member, "revenue": cur_a, "marketing_spend": cur_b,
                    "date": "2026-05-01"})
    raw = pd.DataFrame(rows)
    schema = metrics.detect_schema(raw)
    df = metrics.prepare(raw, schema)
    return df, schema


def _weekly_pair_frame(a_values, b_values, start="2024-01-01"):
    dates = pd.date_range(start, periods=len(a_values), freq="7D").astype(str)
    raw = pd.DataFrame({"date": dates, "revenue": a_values, "marketing_spend": b_values})
    schema = metrics.detect_schema(raw)
    df = metrics.prepare(raw, schema)
    return df, schema


# ---------------------------------------------------------------------------
# correlate_kpis() -- cross_sectional
# ---------------------------------------------------------------------------
class TestPearsonMatchesNumpy(unittest.TestCase):
    def test_r_matches_numpy_corrcoef_on_a_hand_built_cross_section(self):
        # 8 members, base->current changes engineered so a and b move together
        # but not perfectly, with real numeric noise.
        base_a = [100, 100, 100, 100, 100, 100, 100, 100]
        cur_a = [110, 95, 130, 90, 120, 105, 140, 80]
        base_b = [50, 50, 50, 50, 50, 50, 50, 50]
        cur_b = [55, 47, 60, 46, 58, 52, 65, 44]
        df, schema = _member_frame((base_a, cur_a), (base_b, cur_b))
        result = CorrelationEngine(ContractAPI(schema)).correlate_kpis(
            df, "revenue", "marketing_spend", mode="cross_sectional",
            period_a={"type": "quarter", "year": 2026, "quarter": 2},
            period_b={"type": "quarter", "year": 2026, "quarter": 1},
            dimension="region", min_n=3)
        self.assertEqual(result.status, "ok")

        chg_a = [(c - b) / b * 100 for b, c in zip(base_a, cur_a)]
        chg_b = [(c - b) / b * 100 for b, c in zip(base_b, cur_b)]
        expected_r = float(np.corrcoef(chg_a, chg_b)[0, 1])
        self.assertAlmostEqual(result.r, round(expected_r, 3), places=3)

    def test_perfectly_correlated_series_is_r_equals_one(self):
        base_a = [100] * 6
        cur_a = [110, 120, 90, 130, 80, 140]
        base_b = [200] * 6
        cur_b = [220, 240, 180, 260, 160, 280]   # exactly 2x the % change of a
        df, schema = _member_frame((base_a, cur_a), (base_b, cur_b))
        result = CorrelationEngine(ContractAPI(schema)).correlate_kpis(
            df, "revenue", "marketing_spend", mode="cross_sectional",
            period_a={"type": "quarter", "year": 2026, "quarter": 2},
            period_b={"type": "quarter", "year": 2026, "quarter": 1},
            dimension="region", min_n=3)
        self.assertEqual(result.status, "ok")
        self.assertAlmostEqual(result.r, 1.0, places=6)

    def test_perfectly_anti_correlated_series_is_r_equals_minus_one(self):
        base_a = [100] * 6
        cur_a = [110, 120, 90, 130, 80, 140]
        base_b = [200] * 6
        cur_b = [180, 160, 220, 140, 240, 120]   # exactly -2x the % change of a
        df, schema = _member_frame((base_a, cur_a), (base_b, cur_b))
        result = CorrelationEngine(ContractAPI(schema)).correlate_kpis(
            df, "revenue", "marketing_spend", mode="cross_sectional",
            period_a={"type": "quarter", "year": 2026, "quarter": 2},
            period_b={"type": "quarter", "year": 2026, "quarter": 1},
            dimension="region", min_n=3)
        self.assertEqual(result.status, "ok")
        self.assertAlmostEqual(result.r, -1.0, places=6)


class TestFailureModesAreDistinguishable(unittest.TestCase):
    """`analysis.correlate` collapses n<3, a constant, and b constant into the
    identical sentence. Here they are three different statuses."""

    def test_fewer_pairs_than_min_n_is_insufficient_n(self):
        base_a, cur_a = [100, 100, 100], [110, 90, 105]
        base_b, cur_b = [50, 50, 50], [55, 45, 52]
        df, schema = _member_frame((base_a, cur_a), (base_b, cur_b))
        result = CorrelationEngine(ContractAPI(schema)).correlate_kpis(
            df, "revenue", "marketing_spend", mode="cross_sectional",
            period_a={"type": "quarter", "year": 2026, "quarter": 2},
            period_b={"type": "quarter", "year": 2026, "quarter": 1},
            dimension="region", min_n=6)
        self.assertEqual(result.status, "insufficient_n")
        self.assertIsNone(result.r)

    def test_a_constant_series_is_constant_a(self):
        base_a, cur_a = [100] * 6, [50] * 6   # revenue: -50% everywhere, no variance
        base_b, cur_b = [50, 50, 50, 50, 50, 50], [55, 45, 60, 40, 65, 35]
        df, schema = _member_frame((base_a, cur_a), (base_b, cur_b))
        result = CorrelationEngine(ContractAPI(schema)).correlate_kpis(
            df, "revenue", "marketing_spend", mode="cross_sectional",
            period_a={"type": "quarter", "year": 2026, "quarter": 2},
            period_b={"type": "quarter", "year": 2026, "quarter": 1},
            dimension="region", min_n=3)
        self.assertEqual(result.status, "constant_a")
        self.assertIsNone(result.r)

    def test_a_constant_series_is_constant_b(self):
        base_a, cur_a = [100, 100, 100, 100, 100, 100], [110, 90, 120, 80, 130, 70]
        base_b, cur_b = [50] * 6, [40] * 6   # marketing_spend: -20% everywhere
        df, schema = _member_frame((base_a, cur_a), (base_b, cur_b))
        result = CorrelationEngine(ContractAPI(schema)).correlate_kpis(
            df, "revenue", "marketing_spend", mode="cross_sectional",
            period_a={"type": "quarter", "year": 2026, "quarter": 2},
            period_b={"type": "quarter", "year": 2026, "quarter": 1},
            dimension="region", min_n=3)
        self.assertEqual(result.status, "constant_b")
        self.assertIsNone(result.r)


class TestAutoDimensionSelectionHasNoWhitelist(CorrelationTestCase):
    def test_hospital_department_is_selected_automatically(self):
        df, schema = contracted("hospital_sample.csv", FIXTURES)
        ce = CorrelationEngine(ContractAPI(schema))
        periods = sorted(df["_period"].dropna().unique().tolist())
        result = ce.correlate_kpis(
            df, "recovery_rate", "cost_per_admission", mode="cross_sectional",
            period_a={"type": "quarter", "year": int(periods[-1].split("-Q")[0]),
                     "quarter": int(periods[-1][-1])},
            period_b={"type": "quarter", "year": int(periods[0].split("-Q")[0]),
                     "quarter": int(periods[0][-1])},
            min_n=2)
        self.assertEqual(result.dimension, "department")

    def test_school_single_member_dimension_does_not_crash(self):
        df, schema = contracted("school_kpi_smoke_sample.csv")
        ce = CorrelationEngine(ContractAPI(schema))
        result = ce.correlate_kpis(
            df, "students_enrolled", "avg_test_score", mode="cross_sectional",
            period_a={"type": "quarter", "year": 2026, "quarter": 2},
            period_b={"type": "quarter", "year": 2025, "quarter": 2}, min_n=2)
        self.assertIn(result.status, ("ok", "insufficient_n", "constant_a", "constant_b"))


class TestRatioKpiCorrectness(unittest.TestCase):
    """A contract-only ratio KPI (`recovery_rate`, absent from seed `METRICS`)
    must produce real member values, not NaN -- the C2 bug class, checked
    from the aggregation side."""

    def test_recovery_rate_produces_real_values_not_nan(self):
        df, schema = contracted("hospital_sample.csv", FIXTURES)
        ce = CorrelationEngine(ContractAPI(schema))
        periods = sorted(df["_period"].dropna().unique().tolist())
        result = ce.correlate_kpis(
            df, "recovery_rate", "readmissions", mode="cross_sectional",
            period_a={"type": "quarter", "year": int(periods[-1].split("-Q")[0]),
                     "quarter": int(periods[-1][-1])},
            period_b={"type": "quarter", "year": int(periods[0].split("-Q")[0]),
                     "quarter": int(periods[0][-1])},
            min_n=2)
        self.assertIn(result.status, ("ok", "insufficient_n", "constant_a", "constant_b"))
        self.assertNotEqual(result.n, 0)


# ---------------------------------------------------------------------------
# correlate_kpis() -- time_series
# ---------------------------------------------------------------------------
class TestDifferencingActuallyMatters(unittest.TestCase):
    """Two independently-trending series correlate near 1.0 on levels and
    near 0 on changes -- proving the transform choice, not just exercising
    an estimator that always differences."""

    def test_two_unrelated_trends_correlate_on_levels_but_not_on_changes(self):
        rng = np.random.RandomState(7)
        a = [100 + 5 * i + n for i, n in enumerate(rng.normal(0, 1, 60))]
        b = [50 - 3 * i + n for i, n in enumerate(rng.normal(0, 1, 60))]
        df, schema = _weekly_pair_frame(a, b)

        ce = CorrelationEngine(ContractAPI(schema))
        result = ce.correlate_kpis(df, "revenue", "marketing_spend", mode="time_series",
                                   grain="week", min_n=6)
        self.assertEqual(result.status, "ok")

        level_r = float(np.corrcoef(a, b)[0, 1])
        self.assertGreater(abs(level_r), 0.9)          # levels: strongly (spuriously) correlated
        self.assertLess(abs(result.r), 0.4)             # changes: the shared trend is gone


class TestGapsAreNotDifferencedAcross(unittest.TestCase):
    def test_a_hole_in_the_middle_drops_the_spanning_pair(self):
        weekly_a = [100.0 + i for i in range(20)]
        weekly_b = [50.0 + i * 0.5 for i in range(20)]
        dates = pd.date_range("2024-01-01", periods=20, freq="7D").astype(str)
        # Remove week index 10 entirely -- a genuine hole in the series.
        keep = [i for i in range(20) if i != 10]
        raw = pd.DataFrame({"date": [dates[i] for i in keep],
                            "revenue": [weekly_a[i] for i in keep],
                            "marketing_spend": [weekly_b[i] for i in keep]})
        schema = metrics.detect_schema(raw)
        df = metrics.prepare(raw, schema)

        ce = CorrelationEngine(ContractAPI(schema))
        result = ce.correlate_kpis(df, "revenue", "marketing_spend", mode="time_series",
                                   grain="week", min_n=3)
        self.assertGreater(result.pairs_dropped_at_gaps, 0)
        # 19 held weeks -> 18 naive consecutive diffs; one pair (9->11) spans
        # the gap and must be excluded, leaving one fewer usable pair.
        self.assertEqual(result.n, 18 - result.pairs_dropped_at_gaps)


class TestTimeSeriesAirlock(CorrelationTestCase):
    def test_an_unknown_grain_is_rejected(self):
        with self.assertRaises(InvalidArgumentError):
            self.ce.correlate_kpis(self.rdf, "revenue", "units_sold", mode="time_series",
                                   grain="fortnight")

    def test_an_unknown_mode_is_rejected(self):
        with self.assertRaises(InvalidArgumentError):
            self.ce.correlate_kpis(self.rdf, "revenue", "units_sold", mode="sideways")

    def test_cross_sectional_without_both_periods_is_rejected(self):
        with self.assertRaises(InvalidArgumentError):
            self.ce.correlate_kpis(self.rdf, "revenue", "units_sold", mode="cross_sectional",
                                   period_a=self.period_a)

    def test_an_unknown_kpi_is_rejected(self):
        with self.assertRaises(UnknownKpiError):
            self.ce.correlate_kpis(self.rdf, "not_a_real_kpi", "units_sold", mode="time_series")

    def test_an_unknown_dimension_is_rejected(self):
        with self.assertRaises(UnknownDimensionError):
            self.ce.correlate_kpis(self.rdf, "revenue", "units_sold", mode="cross_sectional",
                                   period_a=self.period_a, period_b=self.period_b,
                                   dimension="not_a_real_dimension")


# ---------------------------------------------------------------------------
# correlate_kpi_matrix()
# ---------------------------------------------------------------------------
class TestMatrixEqualsIndividualCalls(CorrelationTestCase):
    def test_every_cross_sectional_row_matches_an_independent_pairwise_call(self):
        candidates = ["units_sold", "gross_profit", "cost_of_goods"]
        matrix = self.ce.correlate_kpi_matrix(
            self.rdf, "revenue", candidates=candidates, mode="cross_sectional",
            period_a=self.period_a, period_b=self.period_b, dimension="region", min_n=3)
        by_kpi = {row.correlation.kpi_b: row.correlation for row in matrix.rows}
        for cand in candidates:
            pairwise = self.ce.correlate_kpis(
                self.rdf, "revenue", cand, mode="cross_sectional",
                period_a=self.period_a, period_b=self.period_b, dimension="region", min_n=3)
            self.assertEqual(by_kpi[cand].to_payload(), pairwise.to_payload())

    def test_every_time_series_row_matches_an_independent_pairwise_call(self):
        candidates = ["units_sold", "gross_profit", "orders"]
        matrix = self.ce.correlate_kpi_matrix(
            self.rdf, "revenue", candidates=candidates, mode="time_series",
            grain="week", min_n=6)
        by_kpi = {row.correlation.kpi_b: row.correlation for row in matrix.rows}
        for cand in candidates:
            pairwise = self.ce.correlate_kpis(
                self.rdf, "revenue", cand, mode="time_series", grain="week", min_n=6)
            self.assertEqual(by_kpi[cand].to_payload(), pairwise.to_payload())


class TestMatrixCostsABoundedNumberOfQueries(CorrelationTestCase):
    def test_cross_sectional_matrix_issues_two_grouped_queries_regardless_of_candidate_count(self):
        candidates = ["units_sold", "gross_profit", "cost_of_goods", "orders", "customers"]
        patch, calls = _counting_patch(QueryEngine, "query")
        with patch:
            self.ce.correlate_kpi_matrix(self.rdf, "revenue", candidates=candidates,
                                        mode="cross_sectional", period_a=self.period_a,
                                        period_b=self.period_b, dimension="region")
        self.assertLessEqual(calls["n"], 2)

    def test_time_series_matrix_issues_one_grouped_query_regardless_of_candidate_count(self):
        candidates = ["units_sold", "gross_profit", "cost_of_goods", "orders", "customers"]
        patch, calls = _counting_patch(QueryEngine, "query")
        with patch:
            self.ce.correlate_kpi_matrix(self.rdf, "revenue", candidates=candidates,
                                        mode="time_series", grain="week")
        self.assertLessEqual(calls["n"], 1)


class TestMatrixRelationLabelling(CorrelationTestCase):
    def test_a_kpi_correlated_against_its_own_formula_component_is_labelled(self):
        matrix = self.ce.correlate_kpi_matrix(
            self.rdf, "gross_profit", candidates=["revenue", "cost_of_goods", "units_sold"],
            mode="time_series", grain="week", min_n=6)
        by_kpi = {row.correlation.kpi_b: row.relation for row in matrix.rows}
        self.assertIsNotNone(by_kpi.get("revenue"))
        self.assertIsNotNone(by_kpi.get("cost_of_goods"))


class TestMatrixDefaultCandidates(CorrelationTestCase):
    def test_default_candidates_are_seeded_from_the_relation_graph(self):
        matrix = self.ce.correlate_kpi_matrix(self.rdf, "gross_profit", mode="time_series",
                                             grain="week", min_n=6)
        self.assertGreater(matrix.considered, 0)
        self.assertNotIn("gross_profit", {row.correlation.kpi_b for row in matrix.rows})


class TestMatrixAirlock(CorrelationTestCase):
    def test_an_unknown_kpi_target_is_rejected(self):
        with self.assertRaises(UnknownKpiError):
            self.ce.correlate_kpi_matrix(self.rdf, "not_a_real_kpi", mode="time_series")

    def test_an_unknown_candidate_kpi_is_rejected(self):
        with self.assertRaises(UnknownKpiError):
            self.ce.correlate_kpi_matrix(self.rdf, "revenue", candidates=["not_a_real_kpi"],
                                        mode="time_series")


# ---------------------------------------------------------------------------
# integrity / prose guard / determinism / portability
# ---------------------------------------------------------------------------
class TestNumericIntegrityAndProseGuard(CorrelationTestCase):
    def test_the_payload_survives_json_dumps_with_no_nan_or_inf(self):
        pair = self.ce.correlate_kpis(self.rdf, "revenue", "units_sold", mode="time_series",
                                      grain="week")
        matrix = self.ce.correlate_kpi_matrix(self.rdf, "revenue", mode="time_series", grain="week")
        assert_json_safe(pair.to_payload())
        assert_json_safe(matrix.to_payload())

    def test_no_string_field_outside_the_allowlist_looks_like_prose(self):
        pair = self.ce.correlate_kpis(self.rdf, "revenue", "units_sold", mode="time_series",
                                      grain="week")
        matrix = self.ce.correlate_kpi_matrix(self.rdf, "revenue", mode="time_series", grain="week")
        assert_no_prose_leak(pair.to_payload())
        assert_no_prose_leak(matrix.to_payload())

    def test_no_banned_prose_key_survives(self):
        pair = self.ce.correlate_kpis(self.rdf, "revenue", "units_sold", mode="time_series",
                                      grain="week")
        matrix = self.ce.correlate_kpi_matrix(self.rdf, "revenue", mode="time_series", grain="week")
        _assert_no_banned_keys(pair.to_payload())
        _assert_no_banned_keys(matrix.to_payload())


class TestDeterminism(CorrelationTestCase):
    def test_identical_input_gives_identical_output_across_runs(self):
        first = self.ce.correlate_kpis(self.rdf, "revenue", "units_sold",
                                       mode="time_series", grain="week").to_payload()
        second = self.ce.correlate_kpis(self.rdf, "revenue", "units_sold",
                                        mode="time_series", grain="week").to_payload()
        self.assertEqual(first, second)


class TestPortabilityAcrossDatasets(unittest.TestCase):
    def test_time_series_mode_works_on_the_school_fixture(self):
        df, schema = contracted("school_kpi_smoke_sample.csv")
        ce = CorrelationEngine(ContractAPI(schema))
        result = ce.correlate_kpis(df, "students_enrolled", "avg_test_score",
                                   mode="time_series", grain="quarter", min_n=3)
        self.assertIn(result.status, ("ok", "insufficient_n", "constant_a", "constant_b", "no_overlap"))

    def test_cross_sectional_mode_works_on_the_hospital_fixture(self):
        df, schema = contracted("hospital_sample.csv", FIXTURES)
        ce = CorrelationEngine(ContractAPI(schema))
        periods = sorted(df["_period"].dropna().unique().tolist())
        result = ce.correlate_kpis(
            df, "admissions", "discharges", mode="cross_sectional",
            period_a={"type": "quarter", "year": int(periods[-1].split("-Q")[0]),
                     "quarter": int(periods[-1][-1])},
            period_b={"type": "quarter", "year": int(periods[0].split("-Q")[0]),
                     "quarter": int(periods[0][-1])}, min_n=2)
        self.assertIn(result.status, ("ok", "insufficient_n", "constant_a", "constant_b"))


class TestPairedSampleIsTheSeamNotAChange(CorrelationTestCase):
    """
    X3/X4 need the aligned vectors a correlation is taken over, so
    `correlate_kpis` was repointed at a published `paired_sample` builder. The
    whole point of that refactor is that it changed nothing: these pin the
    vectors against the arithmetic they replaced, and the payloads against a
    correlation recomputed from the sample by hand.
    """

    def test_the_published_vectors_reproduce_the_reported_r(self):
        for mode, kwargs in (("cross_sectional",
                              dict(period_a=self.period_a, period_b=self.period_b)),
                             ("time_series", dict(grain="quarter"))):
            with self.subTest(mode=mode):
                sample = self.ce.paired_sample(self.rdf, ["revenue", "units_sold"],
                                               mode=mode, **kwargs)
                result = self.ce.correlate_kpis(self.rdf, "revenue", "units_sold",
                                                mode=mode, **kwargs)
                status, r, _r2, n = pearson_correlate(sample.vectors["revenue"],
                                                      sample.vectors["units_sold"],
                                                      result.min_n)
                self.assertEqual((status, r, n), (result.status, result.r, result.n))

    def test_a_sample_over_many_keys_gives_each_pair_its_own_untouched_vectors(self):
        pair = self.ce.paired_sample(self.rdf, ["revenue", "units_sold"],
                                     period_a=self.period_a, period_b=self.period_b)
        wide = self.ce.paired_sample(
            self.rdf, ["revenue", "units_sold", "marketing_spend", "orders"],
            period_a=self.period_a, period_b=self.period_b)
        # Adding keys must not disturb the vectors already there: pairwise
        # masking happens in `pearson_correlate`, never in the sample.
        self.assertEqual(pair.index, wide.index)
        for key in ("revenue", "units_sold"):
            self.assertEqual(pair.vectors[key], wide.vectors[key])

    def test_listwise_deletion_is_opt_in_and_counts_what_it_dropped(self):
        sample = self.ce.paired_sample(self.rdf, ["revenue", "units_sold"],
                                       mode="time_series", grain="quarter")
        index, arrays, dropped = sample.listwise(["revenue", "units_sold"])
        self.assertEqual(len(index) + dropped, len(sample.index))
        for array in arrays.values():
            self.assertEqual(len(array), len(index))
            self.assertTrue(all(v == v for v in array))

    def test_a_time_series_sample_records_the_shared_periods_it_started_from(self):
        # `source_entries` is what tells "the two scopes never overlapped" apart
        # from "they overlapped but yielded too few adjacent pairs".
        sample = self.ce.paired_sample(self.rdf, ["revenue", "units_sold"],
                                       mode="time_series", grain="quarter")
        self.assertEqual(sample.source_entries, len(sample.index) + 1
                         + sample.pairs_dropped_at_gaps)

    def test_paired_sample_rejects_what_correlate_kpis_rejects(self):
        with self.assertRaises(InvalidArgumentError):
            self.ce.paired_sample(self.rdf, ["revenue"], mode="cross_sectional")
        with self.assertRaises(InvalidArgumentError):
            self.ce.paired_sample(self.rdf, [], mode="time_series")
        with self.assertRaises(UnknownKpiError):
            self.ce.paired_sample(self.rdf, ["revenue", "synergy_index"],
                                  mode="time_series")


if __name__ == "__main__":
    unittest.main()
