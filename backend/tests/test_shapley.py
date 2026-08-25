"""
Shapley dimension attribution -- the axioms, then two grids computed by hand.

This suite deliberately needs no dataset, no fixture and no settings: the
attribution is pure arithmetic over a list of cells, so it is checked against
examples a reader can verify with a pencil before it is ever pointed at real
data. If the efficiency axiom does not hold here, every driver share the product
reports downstream is wrong.
"""
from __future__ import annotations

import random
import sys
import unittest
from itertools import combinations
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.engines.drivers import (                      # noqa: E402
    MAX_SHAPLEY_DIMENSIONS,
    explained_sum_of_squares,
    rank_dimensions,
    shapley_dimension_attribution,
)


def grid(values):
    """A 2x2 cell grid: values is ((a1b1, a1b2), (a2b1, a2b2))."""
    cells = []
    for i, row in enumerate(values):
        for j, delta in enumerate(row):
            cells.append({"A": f"a{i + 1}", "B": f"b{j + 1}", "delta": delta})
    return cells


# Case A -- the whole movement is on one dimension; B is a null player.
CASE_A = grid(((-10, -10),
               (2, 2)))

# Case B -- the movement lives only in the a1/b1 intersection. Neither dimension
# explains it alone; the interaction is real and must be split, not dropped.
CASE_B = grid(((-12, 0),
               (0, 0)))


def _total_sum_of_squares(cells):
    deltas = [c["delta"] for c in cells]
    mean = sum(deltas) / len(deltas)
    return sum((d - mean) ** 2 for d in deltas)


def _random_cells(rng, dims, levels=3):
    names = [f"d{i}" for i in range(dims)]
    cells = []

    def walk(prefix):
        if len(prefix) == dims:
            cells.append({**{names[i]: prefix[i] for i in range(dims)},
                          "delta": rng.gauss(0, 10)})
            return
        for level in range(levels):
            walk(prefix + [f"L{level}"])

    walk([])
    return names, cells


class TestValueFunction(unittest.TestCase):
    """V itself, before any Shapley weighting."""

    def test_empty_subset_explains_nothing(self):
        coords = [("a1", "b1"), ("a1", "b2"), ("a2", "b1"), ("a2", "b2")]
        deltas = [-10.0, -10.0, 2.0, 2.0]
        mean = sum(deltas) / len(deltas)
        self.assertEqual(explained_sum_of_squares(coords, deltas, (), mean), 0.0)

    def test_full_subset_is_the_total_sum_of_squares(self):
        coords = [("a1", "b1"), ("a1", "b2"), ("a2", "b1"), ("a2", "b2")]
        deltas = [-12.0, 0.0, 0.0, 0.0]
        mean = sum(deltas) / len(deltas)
        self.assertAlmostEqual(explained_sum_of_squares(coords, deltas, (0, 1), mean),
                               sum((d - mean) ** 2 for d in deltas), places=9)

    def test_hand_computed_values_for_case_a(self):
        coords = [(c["A"], c["B"]) for c in CASE_A]
        deltas = [float(c["delta"]) for c in CASE_A]
        mean = sum(deltas) / len(deltas)
        self.assertEqual(mean, -4.0)
        self.assertAlmostEqual(explained_sum_of_squares(coords, deltas, (0,), mean), 144.0)
        self.assertAlmostEqual(explained_sum_of_squares(coords, deltas, (1,), mean), 0.0)
        self.assertAlmostEqual(explained_sum_of_squares(coords, deltas, (0, 1), mean), 144.0)

    def test_hand_computed_values_for_case_b(self):
        coords = [(c["A"], c["B"]) for c in CASE_B]
        deltas = [float(c["delta"]) for c in CASE_B]
        mean = sum(deltas) / len(deltas)
        self.assertEqual(mean, -3.0)
        self.assertAlmostEqual(explained_sum_of_squares(coords, deltas, (0,), mean), 36.0)
        self.assertAlmostEqual(explained_sum_of_squares(coords, deltas, (1,), mean), 36.0)
        self.assertAlmostEqual(explained_sum_of_squares(coords, deltas, (0, 1), mean), 108.0)

    def test_monotonicity_under_refinement(self):
        """V(S) <= V(S + {d}). phi_d >= 0 rests entirely on this."""
        rng = random.Random(7)
        names, cells = _random_cells(rng, 3, levels=3)
        coords = [tuple(c[n] for n in names) for c in cells]
        deltas = [c["delta"] for c in cells]
        mean = sum(deltas) / len(deltas)
        idx = range(len(names))
        for size in range(len(names)):
            for subset in combinations(idx, size):
                base = explained_sum_of_squares(coords, deltas, subset, mean)
                for d in idx:
                    if d in subset:
                        continue
                    bigger = tuple(sorted(subset + (d,)))
                    self.assertLessEqual(
                        base, explained_sum_of_squares(coords, deltas, bigger, mean) + 1e-9,
                        msg=f"V({subset}) > V({bigger})")


class TestHandComputedCases(unittest.TestCase):
    """The two grids worked out by hand in docs/RANKING.md."""

    def test_case_a_null_player(self):
        """B never changes any coalition's value, so it must receive exactly zero."""
        out = shapley_dimension_attribution(CASE_A, ["A", "B"])
        self.assertAlmostEqual(out["phi"]["A"], 144.0, places=9)
        self.assertAlmostEqual(out["phi"]["B"], 0.0, places=9)
        self.assertAlmostEqual(out["shares"]["A"], 1.0, places=9)
        self.assertAlmostEqual(out["shares"]["B"], 0.0, places=9)

    def test_case_b_interaction_is_split_symmetrically(self):
        """
        Neither dimension explains the cell on its own (V = 36 each) but together
        they explain 108. The 36 of interaction is split 18/18, giving 54 each --
        not dropped, which is what reading the two dimensions independently does.
        """
        out = shapley_dimension_attribution(CASE_B, ["A", "B"])
        self.assertAlmostEqual(out["phi"]["A"], 54.0, places=9)
        self.assertAlmostEqual(out["phi"]["B"], 54.0, places=9)
        self.assertAlmostEqual(out["total_variation"], 108.0, places=9)
        self.assertAlmostEqual(out["shares"]["A"], 0.5, places=9)
        self.assertAlmostEqual(out["shares"]["B"], 0.5, places=9)

    def test_independent_reading_loses_the_interaction(self):
        """The failure this method exists to fix, stated as an assertion."""
        coords = [(c["A"], c["B"]) for c in CASE_B]
        deltas = [float(c["delta"]) for c in CASE_B]
        mean = sum(deltas) / len(deltas)
        marginal_only = (explained_sum_of_squares(coords, deltas, (0,), mean)
                         + explained_sum_of_squares(coords, deltas, (1,), mean))
        out = shapley_dimension_attribution(CASE_B, ["A", "B"])
        self.assertAlmostEqual(marginal_only, 72.0, places=9)
        self.assertAlmostEqual(out["total_variation"], 108.0, places=9)
        self.assertAlmostEqual(sum(out["phi"].values()), 108.0, places=9)


class TestAxioms(unittest.TestCase):

    def test_efficiency_on_the_hand_computed_cases(self):
        for name, cells in (("A", CASE_A), ("B", CASE_B)):
            out = shapley_dimension_attribution(cells, ["A", "B"])
            self.assertAlmostEqual(sum(out["phi"].values()), out["total_variation"],
                                   places=9, msg=f"case {name}")

    def test_efficiency_on_random_grids(self):
        """sum(phi) == V(D) for 3- and 4-dimension grids, at several seeds."""
        for dims in (2, 3, 4):
            for seed in range(5):
                rng = random.Random(1000 * dims + seed)
                names, cells = _random_cells(rng, dims, levels=3)
                out = shapley_dimension_attribution(cells, names)
                total = out["total_variation"]
                self.assertAlmostEqual(
                    sum(out["phi"].values()) / total, 1.0, places=9,
                    msg=f"efficiency violated at dims={dims} seed={seed}")

    def test_efficiency_matches_the_grids_own_total_sum_of_squares(self):
        """V(D) is not an internal convention -- it is the grid's TSS."""
        rng = random.Random(99)
        names, cells = _random_cells(rng, 3, levels=4)
        out = shapley_dimension_attribution(cells, names)
        self.assertAlmostEqual(out["total_variation"] / _total_sum_of_squares(cells),
                               1.0, places=9)

    def test_shares_sum_to_one(self):
        rng = random.Random(11)
        names, cells = _random_cells(rng, 4, levels=2)
        out = shapley_dimension_attribution(cells, names)
        self.assertAlmostEqual(sum(out["shares"].values()), 1.0, places=9)

    def test_phi_is_never_negative(self):
        rng = random.Random(23)
        for seed in range(5):
            names, cells = _random_cells(random.Random(rng.random() * 1e6), 3, levels=3)
            out = shapley_dimension_attribution(cells, names)
            for dim, value in out["phi"].items():
                self.assertGreaterEqual(value, -1e-9, msg=f"{dim} got phi={value}")

    def test_symmetry_of_interchangeable_dimensions(self):
        """Case B's two dimensions are interchangeable, so their phi must be equal."""
        out = shapley_dimension_attribution(CASE_B, ["A", "B"])
        self.assertAlmostEqual(out["phi"]["A"], out["phi"]["B"], places=9)

    def test_dimension_order_does_not_change_the_answer(self):
        forward = shapley_dimension_attribution(CASE_B, ["A", "B"])
        reverse = shapley_dimension_attribution(CASE_B, ["B", "A"])
        self.assertAlmostEqual(forward["phi"]["A"], reverse["phi"]["A"], places=9)
        self.assertAlmostEqual(forward["phi"]["B"], reverse["phi"]["B"], places=9)


class TestDegenerateInputs(unittest.TestCase):

    def test_single_dimension_takes_everything(self):
        cells = [{"A": "a1", "delta": -10.0}, {"A": "a2", "delta": 2.0}]
        out = shapley_dimension_attribution(cells, ["A"])
        self.assertAlmostEqual(out["phi"]["A"], out["total_variation"], places=9)
        self.assertAlmostEqual(out["shares"]["A"], 1.0, places=9)

    def test_flat_grid_reports_no_share_rather_than_dividing_by_zero(self):
        out = shapley_dimension_attribution(grid(((0, 0), (0, 0))), ["A", "B"])
        self.assertEqual(out["total_variation"], 0.0)
        self.assertIsNone(out["shares"]["A"])
        self.assertIsNone(out["shares"]["B"])

    def test_uniform_non_zero_grid_has_no_variation_to_attribute(self):
        """Every cell moved by the same amount: real change, but no driver."""
        out = shapley_dimension_attribution(grid(((-5, -5), (-5, -5))), ["A", "B"])
        self.assertAlmostEqual(out["total_variation"], 0.0, places=9)
        self.assertIsNone(out["shares"]["A"])

    def test_nan_cells_are_skipped_not_counted(self):
        cells = CASE_A + [{"A": "a3", "B": "b1", "delta": float("nan")}]
        out = shapley_dimension_attribution(cells, ["A", "B"])
        self.assertEqual(out["n_cells"], 4)
        self.assertAlmostEqual(out["phi"]["A"], 144.0, places=9)

    def test_no_dimensions_is_reported_not_raised(self):
        out = shapley_dimension_attribution(CASE_A, [])
        self.assertEqual(out["phi"], {})
        self.assertIn("skipped", out)

    def test_too_many_dimensions_degrades_instead_of_hanging(self):
        names = [f"d{i}" for i in range(MAX_SHAPLEY_DIMENSIONS + 1)]
        cells = [{**{n: "x" for n in names}, "delta": 1.0}]
        out = shapley_dimension_attribution(cells, names)
        self.assertIn("skipped", out)
        self.assertEqual(out["phi"], {})

    def test_empty_cells(self):
        out = shapley_dimension_attribution([], ["A", "B"])
        self.assertEqual(out["n_cells"], 0)
        self.assertEqual(out["phi"]["A"], 0.0)


class TestRanking(unittest.TestCase):

    def test_dimensions_are_ranked_by_phi(self):
        rows = rank_dimensions(shapley_dimension_attribution(CASE_A, ["A", "B"]))
        self.assertEqual([r["dimension"] for r in rows], ["A", "B"])
        self.assertEqual([r["rank"] for r in rows], [1, 2])

    def test_ranking_an_empty_attribution_is_empty(self):
        self.assertEqual(rank_dimensions({"phi": {}, "shares": {}}), [])


if __name__ == "__main__":
    unittest.main()
