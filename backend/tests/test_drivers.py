"""
`attribute_dimensions` / `rank_drivers` / `compute_over_index` -- `agent/drivers.py`.

The Shapley arithmetic under `attribute_dimensions` is already covered by
`test_shapley.py`; what matters here is that the wrapper carries the same
invariants through -- efficiency, non-negativity, order-independence -- once
the cells are built through `QueryEngine` instead of `drivers.cell_delta_grid`.

The test that matters most in this file is `TestRankDriversMatchesLegacy`:
`rank_drivers` deliberately does not wrap `drivers.rank_drivers`
(`engines/drivers.py:513`), it recomposes `score_components` /
`composite_score` / `axis_weights` from batched primitives, so the one thing
that must be proven, not assumed, is that the recomposition produces the
identical ranking and the identical scores `drivers.rank_drivers` does on the
same input. Everything else about the rewrite -- arbitrary A/B periods,
batched history, no prose -- is only safe to trust once that parity holds.

`rank_drivers` also recovers the retail sample's planted scenario
(`sample_data/ground_truth.json`): Product A (supply disruption, expected
rank 1) and North (demand erosion, expected rank 2) are known, real drivers
of the Q2-2026 revenue decline.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

from app.agent.contract_api import ContractAPI
from app.agent.drivers import DriverEngine
from app.agent.errors import InvalidArgumentError, UnknownDimensionError, UnknownKpiError
from app.engines import drivers as legacy_drivers
from app.engines import metrics
from app.engines import observe as observe_engine

from .base import (EngineTestCase, FIXTURES, assert_json_safe,
                   assert_no_prose_leak, contracted)

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
TRUTH_PATH = ROOT / "sample_data" / "ground_truth.json"

BANNED = {"note", "skipped", "member_significance_note", "persistence_note", "interpretation"}


def _assert_no_banned_keys(payload, path: str = "$") -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            assert key not in BANNED, f"{path}.{key} is a banned prose field"
            _assert_no_banned_keys(value, f"{path}.{key}")
    elif isinstance(payload, list):
        for i, item in enumerate(payload):
            _assert_no_banned_keys(item, f"{path}[{i}]")


def _planted_factor(factor_id: str):
    truth = json.loads(TRUTH_PATH.read_text())
    return next(f for f in truth["planted_factors"] if f["id"] == factor_id)


class DriverTestCase(EngineTestCase):
    """The shared retail fixture, with a compiled contract."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.rdf, cls.rschema = contracted("business_metrics_sample.csv")
        cls.api = ContractAPI(cls.rschema)
        cls.de = DriverEngine(cls.api)
        cls.period_a = {"type": "quarter", "year": 2026, "quarter": 2}
        cls.period_b = {"type": "quarter", "year": 2026, "quarter": 1}


def _cross_frame(cells, dims):
    """A minimal synthetic cell grid as a DataFrame, prepared through the real
    pipeline, for building `cur`/`base` frames `attribute_dimensions` can
    query. `cells` is `[{**coords, "current": c, "baseline": b}]`. Uses the
    seed KPI name `revenue` so `ContractAPI.require` recognises the column --
    an arbitrary column name is not itself a KPI."""
    rows = []
    for cell in cells:
        row = {d: cell[d] for d in dims}
        row["revenue"] = cell["current"]
        row["date"] = "2026-05-15"
        rows.append(row)
        base_row = {d: cell[d] for d in dims}
        base_row["revenue"] = cell["baseline"]
        base_row["date"] = "2026-02-15"
        rows.append(base_row)
    raw = pd.DataFrame(rows)
    schema = metrics.detect_schema(raw)
    df = metrics.prepare(raw, schema)
    return df, schema


# ---------------------------------------------------------------------------
# attribute_dimensions()
# ---------------------------------------------------------------------------
class TestAttributionInvariantsHoldThroughTheWrapper(DriverTestCase):
    def test_phi_sums_to_total_variation(self):
        result = self.de.attribute_dimensions(self.rdf, "revenue", self.period_a, self.period_b)
        self.assertEqual(result.status, "ok")
        self.assertAlmostEqual(sum(d.phi for d in result.dimensions), result.total_variation, places=2)

    def test_shares_sum_to_one(self):
        result = self.de.attribute_dimensions(self.rdf, "revenue", self.period_a, self.period_b)
        self.assertAlmostEqual(sum(d.share for d in result.dimensions), 1.0, places=6)

    def test_every_phi_is_non_negative(self):
        result = self.de.attribute_dimensions(self.rdf, "revenue", self.period_a, self.period_b)
        for d in result.dimensions:
            self.assertGreaterEqual(d.phi, -1e-6)

    def test_dimension_order_does_not_change_the_answer(self):
        forward = self.de.attribute_dimensions(self.rdf, "revenue", self.period_a, self.period_b,
                                               dimensions=["region", "product", "channel", "segment"])
        backward = self.de.attribute_dimensions(self.rdf, "revenue", self.period_a, self.period_b,
                                                dimensions=["segment", "channel", "product", "region"])
        forward_phi = {d.dimension: d.phi for d in forward.dimensions}
        backward_phi = {d.dimension: d.phi for d in backward.dimensions}
        self.assertEqual(set(forward_phi), set(backward_phi))
        for dim in forward_phi:
            self.assertClose(forward_phi[dim], backward_phi[dim], rel=1e-6)


class TestAttributionDegenerateStatuses(unittest.TestCase):
    def test_no_dimensions_is_reported_not_raised(self):
        raw = pd.DataFrame({"revenue": [1.0, 2.0], "date": ["2026-01-01", "2026-04-01"]})
        schema = metrics.detect_schema(raw)
        df = metrics.prepare(raw, schema)
        result = DriverEngine(ContractAPI(schema)).attribute_dimensions(
            df, "revenue", {"type": "quarter", "year": 2026, "quarter": 2},
            {"type": "quarter", "year": 2026, "quarter": 1})
        self.assertEqual(result.status, "no_dimensions")
        self.assertEqual(result.dimensions, ())

    def test_a_flat_grid_is_no_variation_with_real_phi_and_no_shares(self):
        """Every cell moves by the identical amount -- real change, zero
        between-group variance to attribute."""
        cells = [
            {"region": "North", "product": "A", "current": 110.0, "baseline": 100.0},
            {"region": "North", "product": "B", "current": 110.0, "baseline": 100.0},
            {"region": "South", "product": "A", "current": 110.0, "baseline": 100.0},
            {"region": "South", "product": "B", "current": 110.0, "baseline": 100.0},
        ]
        df, schema = _cross_frame(cells, ["region", "product"])
        result = DriverEngine(ContractAPI(schema)).attribute_dimensions(
            df, "revenue", {"type": "quarter", "year": 2026, "quarter": 2},
            {"type": "quarter", "year": 2026, "quarter": 1}, dimensions=["region", "product"])
        self.assertEqual(result.status, "no_variation")
        self.assertEqual(result.total_variation, 0.0)
        for d in result.dimensions:
            self.assertIsNone(d.share)

    def test_more_than_eight_dimensions_is_reported_not_hung(self):
        dims = [f"dim{i}" for i in range(9)]
        raw = pd.DataFrame({**{d: ["x"] * 4 for d in dims},
                            "revenue": [1.0, 2.0, 3.0, 4.0],
                            "date": ["2026-01-01", "2026-01-02", "2026-04-01", "2026-04-02"]})
        schema = metrics.detect_schema(raw)
        df = metrics.prepare(raw, schema)
        result = DriverEngine(ContractAPI(schema)).attribute_dimensions(
            df, "revenue", {"type": "quarter", "year": 2026, "quarter": 2},
            {"type": "quarter", "year": 2026, "quarter": 1}, dimensions=dims)
        self.assertEqual(result.status, "too_many_dimensions")
        self.assertEqual(result.max_dimensions, 8)


# ---------------------------------------------------------------------------
# compute_over_index()
# ---------------------------------------------------------------------------
class TestOverIndex(DriverTestCase):
    def test_a_member_moving_exactly_proportionally_to_its_size_is_one(self):
        """Every region doubles -- each one's contribution share equals its
        baseline size share exactly, so over-index is 1.0 everywhere."""
        raw = pd.DataFrame({
            "region": ["North", "North", "South", "South"],
            "revenue": [200.0, 100.0, 400.0, 200.0],
            "date": ["2026-05-01", "2026-02-01", "2026-05-01", "2026-02-01"],
        })
        schema = metrics.detect_schema(raw)
        df = metrics.prepare(raw, schema)
        result = DriverEngine(ContractAPI(schema)).compute_over_index(
            df, "revenue", "region", {"type": "quarter", "year": 2026, "quarter": 2},
            {"type": "quarter", "year": 2026, "quarter": 1})
        for row in result.rows:
            self.assertAlmostEqual(row.over_index, 1.0, places=6)

    def test_a_zero_baseline_share_is_none_not_a_division_error(self):
        raw = pd.DataFrame({
            "region": ["North", "North", "South"],
            "revenue": [50.0, 0.0, 100.0],
            "date": ["2026-05-01", "2026-02-01", "2026-02-01"],
        })
        schema = metrics.detect_schema(raw)
        df = metrics.prepare(raw, schema)
        result = DriverEngine(ContractAPI(schema)).compute_over_index(
            df, "revenue", "region", {"type": "quarter", "year": 2026, "quarter": 2},
            {"type": "quarter", "year": 2026, "quarter": 1})
        north = next(r for r in result.rows if r.member == "North")
        self.assertIsNone(north.over_index)

    def test_rows_are_sorted_by_over_index_descending_with_none_last(self):
        result = self.de.compute_over_index(self.rdf, "revenue", "region",
                                            self.period_a, self.period_b)
        values = [r.over_index for r in result.rows]
        finite = [v for v in values if v is not None]
        self.assertEqual(finite, sorted(finite, reverse=True))
        none_positions = [i for i, v in enumerate(values) if v is None]
        self.assertTrue(all(p == len(values) - 1 - i for i, p in enumerate(reversed(none_positions))))

    def test_disproportionate_at_matches_the_house_threshold(self):
        result = self.de.compute_over_index(self.rdf, "revenue", "region",
                                            self.period_a, self.period_b)
        self.assertEqual(result.disproportionate_at, legacy_drivers.DISPROPORTIONATE_AT)


# ---------------------------------------------------------------------------
# rank_drivers()
# ---------------------------------------------------------------------------
class TestRankDriversMatchesLegacy(DriverTestCase):
    """`rank_drivers` recomposes `score_components`/`composite_score`/
    `axis_weights` from batched primitives instead of wrapping
    `drivers.rank_drivers`. This is the test that makes that recomposition
    safe: same members, same order, same scores, on the exact input
    `observe.observe` builds."""

    def test_same_members_same_order_same_scores_as_the_legacy_ranking(self):
        tf = observe_engine.Timeframe(2026, 2)
        legacy_observation = observe_engine.observe(self.rdf, self.rschema, "revenue", tf)
        legacy_ranked = legacy_drivers.rank_drivers(
            legacy_observation["drivers"], self.rdf, "revenue", tf,
            self.rschema.contract_resolver, limit=6,
            axis=legacy_drivers.axis_weights(legacy_observation["dimension_shapley"]))

        result = self.de.rank_drivers(self.rdf, "revenue", self.period_a, self.period_b, limit=6)

        self.assertEqual(len(result.rows), len(legacy_ranked))
        for new_row, old_row in zip(result.rows, legacy_ranked):
            self.assertEqual(new_row.member, old_row["name"])
            self.assertEqual(new_row.dimension, old_row["dimension"])
            self.assertAlmostEqual(new_row.driver_score, old_row["driver_score"], places=3)
            self.assertAlmostEqual(new_row.member_score, old_row["member_score"], places=3)
            self.assertAlmostEqual(new_row.axis_weight, old_row["axis_weight"], places=3)
            for component in ("magnitude", "disproportion", "significance", "persistence"):
                new_value = getattr(new_row.components, component)
                old_value = old_row["score_components"][component]
                if old_value is None:
                    self.assertIsNone(new_value)
                else:
                    self.assertAlmostEqual(new_value, old_value, places=3)


class TestRankDriversRecoversThePlantedFactors(DriverTestCase):
    def test_product_a_and_north_are_the_top_two_drivers(self):
        result = self.de.rank_drivers(self.rdf, "revenue", self.period_a, self.period_b, limit=6)
        top_two = {(r.member, r.dimension) for r in result.rows[:2]}
        self.assertIn(("Product A", "product"), top_two)
        self.assertIn(("North", "region"), top_two)


class TestMitigationRule(unittest.TestCase):
    def test_a_member_that_grew_during_a_decline_is_excluded_and_counted(self):
        raw = pd.DataFrame({
            "region": ["North", "North", "South", "South"],
            # Total declines (250 -> 180); North drives the decline, South
            # partially offsets it -- a mitigation, not a driver.
            "revenue": [50.0, 150.0, 130.0, 100.0],
            "date": ["2026-05-01", "2026-02-01", "2026-05-01", "2026-02-01"],
        })
        schema = metrics.detect_schema(raw)
        df = metrics.prepare(raw, schema)
        result = DriverEngine(ContractAPI(schema)).rank_drivers(
            df, "revenue", {"type": "quarter", "year": 2026, "quarter": 2},
            {"type": "quarter", "year": 2026, "quarter": 1}, dimensions=["region"])
        members = {r.member for r in result.rows}
        self.assertNotIn("South", members)
        self.assertIn("North", members)
        self.assertEqual(result.mitigations_excluded, 1)


class TestMissingComponentsAreDroppedNotZeroed(DriverTestCase):
    def test_a_member_with_insufficient_history_drops_significance_from_the_weight(self):
        raw = pd.DataFrame({
            "region": ["North"] * 4 + ["South"] * 4,
            "revenue": [100.0, 90.0, 80.0, 40.0, 200.0, 190.0, 180.0, 90.0],
            "date": ["2025-08-01", "2025-11-01", "2026-02-01", "2026-05-01"] * 2,
        })
        schema = metrics.detect_schema(raw)
        df = metrics.prepare(raw, schema)
        result = DriverEngine(ContractAPI(schema)).rank_drivers(
            df, "revenue", {"type": "quarter", "year": 2026, "quarter": 2},
            {"type": "quarter", "year": 2026, "quarter": 1}, dimensions=["region"])
        for row in result.rows:
            self.assertIsNone(row.robust_z)
            self.assertNotIn("significance", row.components_used)
            self.assertLess(row.weight_used, 1.0)


class TestArbitraryPeriodsWork(DriverTestCase):
    """The case `drivers.rank_drivers` cannot express at all: an A/B pair that
    is not a single `Timeframe`."""

    def test_year_sets_rank_successfully_with_persistence_unavailable(self):
        result = self.de.rank_drivers(self.rdf, "revenue",
                                      {"type": "years", "values": [2025]},
                                      {"type": "years", "values": [2024]}, limit=6)
        self.assertFalse(result.persistence_available)
        for row in result.rows:
            self.assertIsNone(row.onset_period)
            self.assertNotIn("persistence", row.components_used)
        self.assertGreater(len(result.rows), 0)


class TestRankDriversAirlock(DriverTestCase):
    def test_an_unknown_kpi_is_rejected(self):
        with self.assertRaises(UnknownKpiError):
            self.de.rank_drivers(self.rdf, "not_a_real_kpi", self.period_a, self.period_b)

    def test_an_unknown_dimension_is_rejected(self):
        with self.assertRaises(UnknownDimensionError):
            self.de.rank_drivers(self.rdf, "revenue", self.period_a, self.period_b,
                                 dimensions=["not_a_real_dimension"])

    def test_a_negative_limit_is_rejected(self):
        with self.assertRaises(InvalidArgumentError):
            self.de.rank_drivers(self.rdf, "revenue", self.period_a, self.period_b, limit=-1)


# ---------------------------------------------------------------------------
# integrity / prose guard / determinism / portability
# ---------------------------------------------------------------------------
class TestNumericIntegrityAndProseGuard(DriverTestCase):
    def test_the_payload_survives_json_dumps_with_no_nan_or_inf(self):
        attr = self.de.attribute_dimensions(self.rdf, "revenue", self.period_a, self.period_b)
        rank = self.de.rank_drivers(self.rdf, "revenue", self.period_a, self.period_b)
        over = self.de.compute_over_index(self.rdf, "revenue", "region", self.period_a, self.period_b)
        assert_json_safe(attr.to_payload())
        assert_json_safe(rank.to_payload())
        assert_json_safe(over.to_payload())

    def test_no_string_field_outside_the_allowlist_looks_like_prose(self):
        attr = self.de.attribute_dimensions(self.rdf, "revenue", self.period_a, self.period_b)
        rank = self.de.rank_drivers(self.rdf, "revenue", self.period_a, self.period_b)
        over = self.de.compute_over_index(self.rdf, "revenue", "region", self.period_a, self.period_b)
        assert_no_prose_leak(attr.to_payload())
        assert_no_prose_leak(rank.to_payload())
        assert_no_prose_leak(over.to_payload())

    def test_no_banned_prose_key_survives_in_any_payload(self):
        attr = self.de.attribute_dimensions(self.rdf, "revenue", self.period_a, self.period_b)
        rank = self.de.rank_drivers(self.rdf, "revenue", self.period_a, self.period_b)
        _assert_no_banned_keys(attr.to_payload())
        _assert_no_banned_keys(rank.to_payload())

    def test_no_templated_other_bucket_name_survives(self):
        rank = self.de.rank_drivers(self.rdf, "revenue", self.period_a, self.period_b)
        self.assertNotIn("Other (", str(rank.to_payload()))


class TestDeterminism(DriverTestCase):
    def test_identical_input_gives_identical_output_across_runs(self):
        first = self.de.rank_drivers(self.rdf, "revenue", self.period_a, self.period_b).to_payload()
        second = self.de.rank_drivers(self.rdf, "revenue", self.period_a, self.period_b).to_payload()
        self.assertEqual(first, second)


class TestPortabilityAcrossDatasets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hdf, cls.hschema = contracted("hospital_sample.csv", FIXTURES)
        cls.hapi = ContractAPI(cls.hschema)
        cls.hde = DriverEngine(cls.hapi)
        assert "recovery_rate" in cls.hschema.available_kpis

        cls.sdf, cls.sschema = contracted("school_kpi_smoke_sample.csv")
        cls.sapi = ContractAPI(cls.sschema)
        cls.sde = DriverEngine(cls.sapi)
        assert "school" in cls.sschema.dimensions
        assert cls.sdf["school"].nunique() == 1

    def test_attribute_dimensions_works_on_a_contract_only_ratio_kpi(self):
        periods = sorted(self.hdf["_period"].dropna().unique().tolist())
        result = self.hde.attribute_dimensions(
            self.hdf, "recovery_rate",
            {"type": "quarter", "year": int(periods[-1].split("-Q")[0]), "quarter": int(periods[-1][-1])},
            {"type": "quarter", "year": int(periods[0].split("-Q")[0]), "quarter": int(periods[0][-1])})
        self.assertIn(result.status, ("ok", "no_variation", "no_dimensions", "no_usable_cells"))

    def test_rank_drivers_does_not_crash_on_a_single_member_dimension(self):
        periods = sorted(self.sdf["_period"].dropna().unique().tolist())
        if len(periods) < 2:
            self.skipTest("school fixture has fewer than two periods")
        kpi = self.sschema.available_kpis[0]
        result = self.sde.rank_drivers(
            self.sdf, kpi,
            {"type": "quarter", "year": int(periods[-1].split("-Q")[0]), "quarter": int(periods[-1][-1])},
            {"type": "quarter", "year": int(periods[0].split("-Q")[0]), "quarter": int(periods[0][-1])})
        self.assertIsInstance(result.rows, tuple)


if __name__ == "__main__":
    unittest.main()
