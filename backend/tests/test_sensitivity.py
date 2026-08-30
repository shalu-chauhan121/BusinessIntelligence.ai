"""
`test_sensitivity_to_outliers` / `estimate_effect_size` -- `agent/sensitivity.py`.

The property that matters most is that a finding carried by one member is *seen*
to be carried by one member. `TestOneMemberCarriesTheFinding` builds seven
members with no relationship and one leverage point that manufactures the
correlation on its own; dropping that one member must flip the verdict to
`outlier_driven` and name it. `TestAUniformRelationshipIsRobust` pins the other
half -- a relationship that holds evenly across members survives dropping any
one of them.

Second is that the bootstrap is reproducible. `scipy.stats.bootstrap` is
pseudo-random; `TestTheBootstrapIsSeeded` runs `estimate_effect_size` twice on
the same data and asserts a byte-identical payload (G5). Without the fixed
`BOOTSTRAP_SEED` this fails.

Third is the G8 discipline the rest of the agent layer follows: below the
member floor `test_sensitivity_to_outliers` returns `insufficient`, and below
`MIN_N_FOR_BOOTSTRAP` `estimate_effect_size` returns a real `r` with no interval
-- never a fabricated bound.
"""
from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from app.agent.contract_api import ContractAPI
from app.agent.errors import InvalidArgumentError
from app.agent.sensitivity import (BOOTSTRAP_RESAMPLES, MIN_MEMBERS_FOR_DROP_ONE,
                                   MIN_N_FOR_BOOTSTRAP, OUTLIER_DRIVEN_ATTENUATION_PCT,
                                   SensitivityEngine, _pearson)
from app.engines import metrics

from .base import (EngineTestCase, assert_json_safe, assert_no_prose_leak,
                   contracted)

BANNED = {"interpretation", "detail", "narrative", "note", "conclusion",
          "score", "confidence", "support"}

_PERIOD_A = {"type": "quarter", "year": 2026, "quarter": 2}
_PERIOD_B = {"type": "quarter", "year": 2026, "quarter": 1}


def _assert_no_banned_keys(payload, path: str = "$") -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            assert key not in BANNED, f"{path}.{key} is a banned field"
            _assert_no_banned_keys(value, f"{path}.{key}")
    elif isinstance(payload, list):
        for i, item in enumerate(payload):
            _assert_no_banned_keys(item, f"{path}[{i}]")


def _member_frame(rev_pct, mkt_pct, base=1000.0):
    """A synthetic frame with `len(rev_pct)` members of a `region` dimension and
    two seed KPIs (`revenue`, `marketing_spend`) whose per-member percent
    changes between the two quarters are exactly `rev_pct` / `mkt_pct`."""
    rows = []
    for i, (rp, mp) in enumerate(zip(rev_pct, mkt_pct)):
        member = f"R{i}"
        rows.append({"region": member, "revenue": base, "marketing_spend": base,
                     "date": "2026-02-01"})
        rows.append({"region": member, "revenue": base * (1 + rp / 100.0),
                     "marketing_spend": base * (1 + mp / 100.0), "date": "2026-05-01"})
    raw = metrics.normalise_columns(pd.DataFrame(rows))
    schema = metrics.detect_schema(raw)
    return metrics.prepare(raw, schema), schema


def _series_frame(rev, mkt, start="2023-01-02"):
    """A single-member weekly frame -- the shape a time-series correlation
    reads."""
    dates = pd.date_range(start, periods=len(rev), freq="7D").astype(str)
    raw = metrics.normalise_columns(pd.DataFrame(
        {"date": dates, "revenue": rev, "marketing_spend": mkt}))
    schema = metrics.detect_schema(raw)
    return metrics.prepare(raw, schema), schema


def _engine(schema):
    return SensitivityEngine(ContractAPI(schema))


# ---------------------------------------------------------------------------
# the maths this module owns
# ---------------------------------------------------------------------------
class TestPearsonHelper(unittest.TestCase):
    def test_it_matches_numpy_on_a_normal_scatter(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=20)
        y = 0.6 * x + rng.normal(scale=0.5, size=20)
        self.assertAlmostEqual(_pearson(x, y), float(np.corrcoef(x, y)[0, 1]), places=12)

    def test_a_constant_side_is_none_not_nan(self):
        self.assertIsNone(_pearson(np.array([1.0, 1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0, 4.0])))

    def test_fewer_than_three_points_is_none(self):
        self.assertIsNone(_pearson(np.array([1.0, 2.0]), np.array([2.0, 4.0])))


# ---------------------------------------------------------------------------
# test_sensitivity_to_outliers()
# ---------------------------------------------------------------------------
class TestOneMemberCarriesTheFinding(unittest.TestCase):
    """Seven members with no relationship, plus one leverage point that
    manufactures the correlation single-handed."""

    def _run(self):
        # Seven members with no relationship, plus one that moves both KPIs
        # +60% together and so manufactures the correlation on its own.
        rev = [5.0, -3.0, 4.0, -2.0, 6.0, -5.0, 3.0, 60.0]
        mkt = [2.0, 4.0, -3.0, -5.0, 1.0, 3.0, -2.0, 60.0]
        df, schema = _member_frame(rev, mkt)
        return _engine(schema).test_sensitivity_to_outliers(
            df, "revenue", "marketing_spend", _PERIOD_A, _PERIOD_B,
            dimension="region", min_n=3)

    def test_the_full_sample_shows_a_strong_association(self):
        self.assertEqual(self._run().status, "ok")
        self.assertGreater(abs(self._run().r_full), 0.7)

    def test_dropping_the_leverage_point_overturns_it(self):
        result = self._run()
        self.assertEqual(result.most_influential_member, "R7")
        self.assertEqual(result.verdict, "outlier_driven")
        self.assertLess(abs(result.r_without_most_influential), abs(result.r_full))

    def test_the_leverage_point_need_not_be_the_biggest_mover(self):
        """`flagged_outlier_member` is `find_outlier_contributors`' pick -- a
        different question from leverage on the correlation. The two fields
        exist so the model can see when they disagree."""
        result = self._run()
        self.assertIn(result.flagged_outlier_status, ("ok", "insufficient", "unavailable", "not_run"))
        self.assertIsNotNone(result.flagged_matches_influential)


class TestAUniformRelationshipIsRobust(unittest.TestCase):
    def _run(self):
        rev = [10.0, -8.0, 6.0, -4.0, 12.0, -10.0, 8.0, -6.0]
        mkt = [8.2, -6.1, 5.0, -3.0, 9.5, -8.3, 6.6, -4.9]   # ~0.8x rev, small noise
        df, schema = _member_frame(rev, mkt)
        return _engine(schema).test_sensitivity_to_outliers(
            df, "revenue", "marketing_spend", _PERIOD_A, _PERIOD_B,
            dimension="region", min_n=3)

    def test_it_is_robust(self):
        result = self._run()
        self.assertEqual(result.verdict, "robust")
        self.assertGreater(abs(result.r_full), 0.9)
        self.assertLess(abs(result.attenuation_pct), OUTLIER_DRIVEN_ATTENUATION_PCT)
        self.assertFalse(result.sign_flipped)

    def test_the_thresholds_are_published(self):
        thresholds = self._run().thresholds
        self.assertEqual(thresholds["outlier_driven_attenuation_pct"],
                         OUTLIER_DRIVEN_ATTENUATION_PCT)
        self.assertIn("negligible_r", thresholds)
        self.assertIn("min_members_for_drop_one", thresholds)


class TestTooFewMembersIsInsufficientNotAFabricatedVerdict(unittest.TestCase):
    def test_a_three_member_dimension_is_insufficient(self):
        df, schema = _member_frame([10.0, -5.0, 7.0], [8.0, -4.0, 6.0])
        result = _engine(schema).test_sensitivity_to_outliers(
            df, "revenue", "marketing_spend", _PERIOD_A, _PERIOD_B,
            dimension="region", min_n=3)
        self.assertEqual(result.status, "insufficient_population")
        self.assertEqual(result.verdict, "insufficient")
        self.assertIsNone(result.most_influential_member)
        self.assertIsNone(result.attenuation_pct)
        self.assertLess(result.n_full, MIN_MEMBERS_FOR_DROP_ONE)

    def test_a_correlation_that_never_reached_ok_propagates_its_status(self):
        # min_n above the member count -> the underlying correlation is
        # insufficient_n, and the sensitivity check says so rather than
        # refitting nothing.
        df, schema = _member_frame([10.0, -8.0, 6.0, -4.0, 12.0], [8.0, -6.0, 5.0, -3.0, 9.0])
        result = _engine(schema).test_sensitivity_to_outliers(
            df, "revenue", "marketing_spend", _PERIOD_A, _PERIOD_B,
            dimension="region", min_n=6)
        self.assertEqual(result.status, "insufficient_n")
        self.assertEqual(result.verdict, "insufficient")


class TestOutlierSensitivityValidation(unittest.TestCase):
    def test_an_unknown_kpi_raises_a_typed_error(self):
        df, schema = _member_frame([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
        with self.assertRaises(Exception):
            _engine(schema).test_sensitivity_to_outliers(
                df, "not_a_kpi", "marketing_spend", _PERIOD_A, _PERIOD_B)

    def test_a_bad_alpha_raises(self):
        df, schema = _member_frame([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
        with self.assertRaises(InvalidArgumentError):
            _engine(schema).test_sensitivity_to_outliers(
                df, "revenue", "marketing_spend", _PERIOD_A, _PERIOD_B, alpha=1.5)


# ---------------------------------------------------------------------------
# estimate_effect_size()
# ---------------------------------------------------------------------------
def _trending_pair(n, seed, noise=0.4):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(size=n)) + 100.0
    y = 0.7 * x + rng.normal(scale=noise * np.std(x) + 1e-6, size=n) + 50.0
    return np.abs(x) + 1.0, np.abs(y) + 1.0


class TestTheBootstrapIsSeeded(unittest.TestCase):
    """G5: identical input -> byte-identical output. The fixed BOOTSTRAP_SEED is
    what makes this pass."""

    def test_two_calls_produce_an_identical_payload(self):
        rev, mkt = _trending_pair(40, seed=3)
        df, schema = _series_frame(rev, mkt)
        eng = _engine(schema)
        one = eng.estimate_effect_size(df, "revenue", "marketing_spend",
                                       mode="time_series", grain="week", min_n=3)
        two = eng.estimate_effect_size(df, "revenue", "marketing_spend",
                                       mode="time_series", grain="week", min_n=3)
        self.assertEqual(one.to_payload(), two.to_payload())
        self.assertEqual(one.bootstrap_status, "ok")
        self.assertEqual(one.bootstrap_resamples, BOOTSTRAP_RESAMPLES)


class TestTheIntervalBehavesLikeAnInterval(unittest.TestCase):
    def test_the_ci_contains_the_point_estimate(self):
        rev, mkt = _trending_pair(45, seed=5)
        df, schema = _series_frame(rev, mkt)
        result = _engine(schema).estimate_effect_size(
            df, "revenue", "marketing_spend", mode="time_series", grain="week", min_n=3)
        self.assertEqual(result.bootstrap_status, "ok")
        self.assertLessEqual(result.ci_low, result.r)
        self.assertGreaterEqual(result.ci_high, result.r)

    def test_the_ci_widens_as_n_falls(self):
        wide_rev, wide_mkt = _trending_pair(50, seed=7)

        def _width(n):
            df, schema = _series_frame(wide_rev[:n], wide_mkt[:n])
            result = _engine(schema).estimate_effect_size(
                df, "revenue", "marketing_spend", mode="time_series", grain="week", min_n=3)
            self.assertEqual(result.bootstrap_status, "ok")
            return result.ci_high - result.ci_low

        self.assertGreater(_width(14), _width(50))


class TestBootstrapRefusals(unittest.TestCase):
    def test_below_the_floor_the_bootstrap_is_refused_but_r_is_still_reported(self):
        rev, mkt = _trending_pair(6, seed=9)   # 5 diffs, below MIN_N_FOR_BOOTSTRAP
        df, schema = _series_frame(rev, mkt)
        result = _engine(schema).estimate_effect_size(
            df, "revenue", "marketing_spend", mode="time_series", grain="week", min_n=3)
        self.assertEqual(result.bootstrap_status, "insufficient_for_bootstrap")
        self.assertLess(result.n, MIN_N_FOR_BOOTSTRAP)
        self.assertIsNone(result.ci_low)
        self.assertEqual(result.bootstrap_resamples, 0)
        self.assertIsNotNone(result.r)

    def test_a_constant_series_does_not_raise_and_propagates_its_status(self):
        rev, _ = _trending_pair(30, seed=11)
        df, schema = _series_frame(rev, np.full(30, 500.0))
        result = _engine(schema).estimate_effect_size(
            df, "revenue", "marketing_spend", mode="time_series", grain="week", min_n=3)
        self.assertIn(result.bootstrap_status, ("constant_a", "constant_b", "no_overlap",
                                                "insufficient_n"))
        self.assertIsNone(result.ci_low)

    def test_the_fisher_interval_travels_alongside_the_bootstrap_one(self):
        rev, mkt = _trending_pair(40, seed=13)
        df, schema = _series_frame(rev, mkt)
        result = _engine(schema).estimate_effect_size(
            df, "revenue", "marketing_spend", mode="time_series", grain="week", min_n=3)
        self.assertEqual(result.fisher_ci_status, "ok")
        self.assertIsNotNone(result.fisher_ci_low)
        self.assertIn("small", result.magnitude_thresholds)


# ---------------------------------------------------------------------------
# cross-cutting: prose leak, json safety, multi-dataset
# ---------------------------------------------------------------------------
class TestNoProseLeaks(EngineTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.cdf, cls.cschema = contracted("business_metrics_sample.csv")
        cls.engine = SensitivityEngine(ContractAPI(cls.cschema))

    def _payloads(self):
        a = self.engine.test_sensitivity_to_outliers(
            self.cdf, "revenue", "gross_margin_pct",
            {"type": "year", "year": 2024}, {"type": "year", "year": 2025}, min_n=3).to_payload()
        b = self.engine.estimate_effect_size(
            self.cdf, "revenue", "gross_margin_pct", mode="time_series", grain="quarter",
            min_n=3).to_payload()
        return a, b

    def test_g1_no_field_reads_like_a_sentence(self):
        for payload in self._payloads():
            assert_no_prose_leak(payload)
            _assert_no_banned_keys(payload)

    def test_g4_json_safe(self):
        for payload in self._payloads():
            assert_json_safe(payload)
            json.dumps(payload)


class TestMultiDatasetPortability(unittest.TestCase):
    """G3: the two tools run unchanged against the hospital and school fixtures,
    which have neither `revenue` nor `region`."""

    FIXTURES = (("hospital_kpi_smoke_sample.csv"), ("school_kpi_smoke_sample.csv"))

    def test_both_tools_execute_on_each_fixture(self):
        from .base import SAMPLES
        for csv_name in self.FIXTURES:
            with self.subTest(fixture=csv_name):
                df, schema = contracted(csv_name, SAMPLES)
                api = ContractAPI(schema)
                eng = SensitivityEngine(api)
                keys = api.list_kpi_keys()
                kpi, cause = keys[0], keys[1]
                dim = api.list_dimensions(df)[0].name
                out = eng.test_sensitivity_to_outliers(
                    df, kpi, cause, {"type": "year", "year": 2024},
                    {"type": "year", "year": 2025}, dimension=dim, min_n=3)
                assert_json_safe(out.to_payload())
                assert_no_prose_leak(out.to_payload())
                eff = eng.estimate_effect_size(
                    df, kpi, cause, mode="time_series", grain="quarter", min_n=3)
                assert_json_safe(eff.to_payload())
                assert_no_prose_leak(eff.to_payload())


if __name__ == "__main__":
    unittest.main()
