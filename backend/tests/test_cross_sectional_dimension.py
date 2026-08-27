"""
CONTEST's cross-sectional check must work on datasets that are not retail.

`consistency_check` chose its dimension from a hardcoded
`("region", "product", "channel", "segment")` tuple. Those are retail words.
The hospital dataset is dimensioned by `hospital` / `department` and the school
dataset by `school` / `campus` / `grade`, so nothing matched, `dim` came back
`None`, and the function returned `not_applicable` for **every** hypothesis --
one of CONTEST's four adversarial checks silently disabled on exactly the
datasets whose KPIs the retail seed registry cannot describe either.

The bug survived because every test that exercised CONTEST used the retail
sample, where the whitelist happens to match. This file is the guard: it runs
the same assertion across three unrelated domains, so a retail-only assumption
fails here rather than in front of a user.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from app.engines.contest import (
    MIN_CROSS_SECTION_MEMBERS,
    _cross_sectional_dimension,
    consistency_check,
)
from app.engines.metrics import detect_schema, prepare

SAMPLES = Path(__file__).resolve().parents[2] / "sample_data"


def _frame(name: str):
    raw = pd.read_csv(SAMPLES / name)
    schema = detect_schema(raw)
    return prepare(raw, schema), schema


class TestDimensionChoiceIsDomainAgnostic(unittest.TestCase):
    """
    One implementation, three vocabularies with nothing in common.

    Retail is included precisely because it used to be the *only* case that
    worked -- it pins the fix as a widening rather than a change.
    """

    def test_retail_still_chooses_region(self):
        """The whitelist's own answer. If this moves, the fix changed retail
        behaviour instead of merely extending it to other domains."""
        df, schema = _frame("business_metrics_sample.csv")
        self.assertEqual(_cross_sectional_dimension(df, df, schema), "region")

    def test_hospital_chooses_department(self):
        """Was `None` under the whitelist -- the whole check was dead here."""
        df, schema = _frame("hospital_kpi_smoke_sample.csv")
        self.assertEqual(_cross_sectional_dimension(df, df, schema), "department")

    def test_school_chooses_grade(self):
        """Also `None` under the whitelist. `grade` has 4 members; `school` has
        1 and `campus` has 2, so both are correctly passed over."""
        df, schema = _frame("school_kpi_smoke_sample.csv")
        self.assertEqual(_cross_sectional_dimension(df, df, schema), "grade")

    def test_no_retail_vocabulary_decides_the_choice(self):
        """The chooser must not name a business term at all."""
        import inspect

        source = inspect.getsource(_cross_sectional_dimension)
        body = source.split('"""')[-1]           # drop the docstring, which cites the bug
        for word in ("region", "product", "channel", "segment",
                     "department", "campus", "grade"):
            self.assertNotIn(f'"{word}"', body)
            self.assertNotIn(f"'{word}'", body)


class TestDimensionChoiceRules(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            "solo": ["only"] * 12,                                  # 1 member
            "pair": ["a", "b"] * 6,                                 # 2 members
            "several": ["p", "q", "r", "s"] * 3,                    # 4 members
            "value": range(12),
        })

        class _Schema:
            dimensions = ["solo", "pair", "several"]
        self.schema = _Schema()

    def test_a_dimension_with_too_few_members_is_never_chosen(self):
        """Pearson r across two points is always exactly +/-1, and `correlate`
        refuses below three members anyway (`analysis.py:146`)."""
        self.assertGreaterEqual(MIN_CROSS_SECTION_MEMBERS, 3)
        self.assertEqual(_cross_sectional_dimension(self.df, self.df, self.schema),
                         "several")

    def test_the_most_comparable_dimension_wins(self):
        df = self.df.copy()
        df["many"] = [f"m{i}" for i in range(12)]                   # 12 members
        self.schema.dimensions = self.schema.dimensions + ["many"]
        self.assertEqual(_cross_sectional_dimension(df, df, self.schema), "many")

    def test_only_members_present_in_both_periods_count(self):
        """A member missing from one side cannot be compared, so it must not
        prop a dimension over the threshold."""
        base = self.df[self.df["several"].isin(["p", "q"])]
        self.assertIsNone(_cross_sectional_dimension(self.df, base, self.schema))

    def test_a_dimension_absent_from_the_frame_is_skipped(self):
        self.schema.dimensions = ["not_a_column", "several"]
        self.assertEqual(_cross_sectional_dimension(self.df, self.df, self.schema),
                         "several")

    def test_no_usable_dimension_returns_none_rather_than_guessing(self):
        self.schema.dimensions = ["solo", "pair"]
        self.assertIsNone(_cross_sectional_dimension(self.df, self.df, self.schema))

    def test_the_choice_is_deterministic(self):
        """Ties break on name, so repeated calls cannot disagree (G5)."""
        df = self.df.copy()
        df["alt"] = ["w", "x", "y", "z"] * 3                        # also 4 members
        self.schema.dimensions = ["several", "alt"]
        picks = {_cross_sectional_dimension(df, df, self.schema) for _ in range(5)}
        self.assertEqual(picks, {"alt"})                            # 'alt' sorts first


class TestConsistencyCheckRunsOnNonRetailData(unittest.TestCase):
    """The end-to-end consequence: the check actually produces a verdict."""

    def test_hospital_gets_a_real_cross_sectional_verdict(self):
        df, schema = _frame("hospital_kpi_smoke_sample.csv")
        half = len(df) // 2
        base, cur = df.iloc[:half], df.iloc[half:]
        result = consistency_check(cur, base, schema, "readmissions",
                                   {"cause_metric": "discharges",
                                    "cause_direction": "up"})
        self.assertEqual(result["status"], "checked",
                         "the hospital dataset must reach the cross-sectional test at all")
        self.assertEqual(result["dimension"], "department")
        self.assertGreaterEqual(result["correlation"]["n"], MIN_CROSS_SECTION_MEMBERS)

    def test_a_hypothesis_without_a_driver_series_is_still_not_applicable(self):
        """Not every `not_applicable` was the bug -- a dimension-concentration
        hypothesis genuinely has no series to compare across members."""
        df, schema = _frame("hospital_kpi_smoke_sample.csv")
        result = consistency_check(df, df, schema, "readmissions", {"cause_metric": None})
        self.assertEqual(result["status"], "not_applicable")


if __name__ == "__main__":
    unittest.main()
