"""
The `TimeFilter` union -- the general time scope `observe.slice_period`
(`engines/observe.py:68`) never had. `slice_period` filters on `_year` and, at
most, `_quarter`; nothing in the product could express "2022 to 2026" or
"every odd year" because no shape existed for it. These tests pin the eight
cases of that shape, and the two properties that matter most: a period the
dataset does not hold at all is a typed, recoverable error, while a period the
dataset holds *some* of is a real answer that says exactly what was missing.
"""
from __future__ import annotations

import unittest

import pandas as pd

from app.agent.errors import EmptyPeriodError, MalformedTimeFilterError
from app.agent.timefilter import apply, parse, resolve
from app.engines.observe import slice_period

from .base import EngineTestCase


class TestEachTypeSelectsExactlyItsRows(EngineTestCase):
    """Matrix across all eight `TimeFilter` types on the shared retail fixture."""

    def test_quarter_matches_year_and_quarter_columns(self):
        tf = parse({"type": "quarter", "year": 2024, "quarter": 2})
        frame, _ = apply(self.df, tf)
        expected = self.df[(self.df["_year"] == 2024) & (self.df["_quarter"] == 2)]
        self.assertEqual(set(frame.index), set(expected.index))

    def test_year_matches_the_whole_calendar_year(self):
        tf = parse({"type": "year", "year": 2024})
        frame, _ = apply(self.df, tf)
        expected = self.df[self.df["_year"] == 2024]
        self.assertEqual(set(frame.index), set(expected.index))

    def test_years_selects_exactly_the_odd_years(self):
        """The flagship case: 'revenue in all odd-numbered years' has no code
        path without this. Compared directly against a hand-built mask."""
        tf = parse({"type": "years", "values": [2023, 2025]})
        frame, selection = apply(self.df, tf)
        expected = self.df[self.df["_year"].isin([2023, 2025])]
        self.assertEqual(set(frame.index), set(expected.index))
        self.assertEqual(selection.missing, ())

    def test_quarters_selects_exactly_the_named_pairs(self):
        tf = parse({"type": "quarters", "values": [
            {"year": 2023, "quarter": 4}, {"year": 2025, "quarter": 1}]})
        frame, _ = apply(self.df, tf)
        expected = self.df[
            ((self.df["_year"] == 2023) & (self.df["_quarter"] == 4)) |
            ((self.df["_year"] == 2025) & (self.df["_quarter"] == 1))]
        self.assertEqual(set(frame.index), set(expected.index))

    def test_range_is_inclusive_at_both_ends(self):
        first, last = self.df["_date"].min(), self.df["_date"].max()
        tf = parse({"type": "range", "start": str(first.date()), "end": str(last.date())})
        frame, selection = apply(self.df, tf)
        self.assertEqual(len(frame), len(self.df["_date"].dropna()))
        self.assertEqual(selection.missing, ())

    def test_months_wires_the_month_column_prepare_writes_and_nothing_else_reads(self):
        """`_month` ('2024-01') is written by `prepare` and, before this
        module, read by nothing in the codebase."""
        months = parse({"type": "months", "start": "2024-01", "end": "2024-03"})
        quarter = parse({"type": "quarter", "year": 2024, "quarter": 1})
        month_frame, _ = apply(self.df, months)
        quarter_frame, _ = apply(self.df, quarter)
        self.assertEqual(set(month_frame.index), set(quarter_frame.index))

    def test_latest_counts_held_periods_not_calendar_ones(self):
        """A dataset missing a quarter must return the last N it actually
        holds, not the last N the calendar would predict."""
        dropped = self.df[~((self.df["_year"] == 2025) & (self.df["_quarter"] == 2))]
        tf = parse({"type": "latest", "grain": "quarter", "n": 4})
        _, selection = apply(dropped, tf)
        self.assertNotIn("2025-Q2", selection.periods)
        self.assertEqual(len(selection.periods), 4)

    def test_all_selects_every_row(self):
        tf = parse({"type": "all"})
        frame, _ = apply(self.df, tf)
        self.assertEqual(len(frame), len(self.df))


class TestPartialCoverageIsReportedNotSilentlyClamped(EngineTestCase):
    """Some but not all of a request being held is a real answer, not a
    failure -- clamping it silently is exactly the over-claim this module
    exists to prevent."""

    def test_a_range_wider_than_the_dataset_reports_what_it_could_not_cover(self):
        tf = parse({"type": "range", "start": "2022-01-01", "end": "2026-12-31"})
        frame, selection = apply(self.df, tf)
        # every row the dataset holds is still returned...
        self.assertEqual(len(frame), len(self.df["_date"].dropna()))
        # ...and the gap on both ends is named, not silently absorbed.
        self.assertTrue(selection.missing)
        self.assertEqual(selection.covered_start, str(self.df["_date"].min().date()))
        self.assertEqual(selection.covered_end, str(self.df["_date"].max().date()))

    def test_a_years_set_partly_outside_the_dataset_names_only_the_missing_ones(self):
        held_years = sorted(self.df["_year"].dropna().unique().tolist())
        tf = parse({"type": "years", "values": [held_years[0], 2099]})
        frame, selection = apply(self.df, tf)
        self.assertTrue(len(frame) > 0)
        self.assertEqual(selection.missing, ("2099",))


class TestAPeriodTheDatasetDoesNotHoldAtAllIsATypedError(EngineTestCase):
    def test_a_year_entirely_outside_the_dataset_raises_empty_period(self):
        with self.assertRaises(EmptyPeriodError) as ctx:
            resolve(self.df, parse({"type": "year", "year": 2099}))
        payload = ctx.exception.to_payload()
        self.assertEqual(payload["error"], "empty_period")
        self.assertTrue(payload["available_periods"])

    def test_an_empty_mask_never_silently_becomes_all_rows(self):
        """The classic pandas mask-inversion bug: an always-false condition
        must never be read back as an always-true one."""
        with self.assertRaises(EmptyPeriodError):
            resolve(self.df, parse({"type": "quarters", "values": [{"year": 1900, "quarter": 1}]}))


class TestMalformedSpecsAreRecoverableErrors(EngineTestCase):
    def test_an_out_of_range_quarter_is_rejected(self):
        with self.assertRaises(MalformedTimeFilterError) as ctx:
            parse({"type": "quarter", "year": 2024, "quarter": 7})
        self.assertIn("valid_types", ctx.exception.to_payload())

    def test_an_unknown_type_is_rejected_and_names_the_valid_ones(self):
        with self.assertRaises(MalformedTimeFilterError) as ctx:
            parse({"type": "decade"})
        self.assertTrue(ctx.exception.to_payload()["valid_types"])

    def test_an_unparseable_date_is_rejected(self):
        with self.assertRaises(MalformedTimeFilterError):
            parse({"type": "range", "start": "not-a-date", "end": "2024-12-31"})

    def test_a_missing_required_key_is_rejected(self):
        with self.assertRaises(MalformedTimeFilterError):
            parse({"type": "year"})

    def test_a_non_positive_n_is_rejected(self):
        with self.assertRaises(MalformedTimeFilterError):
            parse({"type": "latest", "grain": "quarter", "n": 0})

    def test_a_non_mapping_spec_is_rejected(self):
        with self.assertRaises(MalformedTimeFilterError):
            parse("2024-Q1")


class TestTheLegacyBridge(EngineTestCase):
    """`to_timeframe()` lets O4 hand legacy engine functions a period without
    duplicating their slicing logic."""

    def test_quarter_bridges_to_the_same_rows_as_slice_period(self):
        tf = parse({"type": "quarter", "year": 2024, "quarter": 3})
        frame, _ = apply(self.df, tf)
        legacy = slice_period(self.df, tf.to_timeframe())
        self.assertEqual(set(frame.index), set(legacy.index))

    def test_year_bridges_to_the_same_rows_as_slice_period(self):
        tf = parse({"type": "year", "year": 2023})
        frame, _ = apply(self.df, tf)
        legacy = slice_period(self.df, tf.to_timeframe())
        self.assertEqual(set(frame.index), set(legacy.index))

    def test_a_set_valued_filter_has_no_single_timeframe_equivalent(self):
        tf = parse({"type": "years", "values": [2023, 2024]})
        self.assertIsNone(tf.to_timeframe())


if __name__ == "__main__":
    unittest.main()
