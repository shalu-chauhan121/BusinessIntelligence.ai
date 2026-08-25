"""
Reading a period out of a question.

Everything here is deterministic. A question that says "Q4 2026" means that
quarter; a question that says "last quarter" means the most recent quarter the
*dataset* holds, not the one the wall clock is in — a dataset that ends in 2026
should answer "last quarter" the same way in 2027 as it did the day it was
uploaded.

Fiscal calendars are honoured through `to_timeframe`. The contract has carried a
`fiscal_year_start_month` since it was written but nothing ever read it, so
every period in the product is a calendar quarter. Routing the conversion
through one function does not change that today — the default start month is
January, so the arithmetic is identical — but it puts the offset in exactly one
place for the day a dataset needs it.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from ..engines.observe import Timeframe
from ..models.investigation import Ambiguity, PeriodSpec

# "Q4 2026", "2026 Q4", "q4/2026", "fourth quarter of 2026"
_Q_YEAR = re.compile(r"\bq([1-4])\s*(?:of\s+)?[,/ -]?\s*((?:19|20)\d{2})\b", re.I)
_YEAR_Q = re.compile(r"\b((?:19|20)\d{2})\s*[,/ -]?\s*q([1-4])\b", re.I)
_QUARTER_WORD = re.compile(
    r"\b(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter\b", re.I)
_BARE_QUARTER = re.compile(r"\bq([1-4])\b", re.I)
_BARE_YEAR = re.compile(r"\b((?:19|20)\d{2})\b")

_WORD_TO_QUARTER = {
    "first": 1, "1st": 1, "second": 2, "2nd": 2,
    "third": 3, "3rd": 3, "fourth": 4, "4th": 4,
}

# Phrases that pick a period relative to what the dataset holds.
_RELATIVE_CURRENT = ("this quarter", "current quarter", "latest quarter",
                     "most recent quarter", "the last quarter", "last quarter",
                     "recently", "lately", "right now", "at the moment")
_RELATIVE_PRIOR = ("previous quarter", "quarter before", "prior quarter")

# Phrases that pick which baseline to compare against.
_YOY_MARKERS = ("year over year", "year-over-year", "yoy", "same quarter last year",
                "versus last year", "vs last year", "compared to last year",
                "against last year", "a year ago", "year on year", "year-on-year")
_PRIOR_MARKERS = ("previous period", "prior period", "quarter over quarter",
                  "quarter-over-quarter", "qoq", "previous quarter",
                  "compared to last quarter", "vs last quarter", "sequentially")


def fiscal_offset(calendar: Any) -> int:
    """
    How many months this calendar's year is shifted from the calendar year.

    Zero for a January start, which is every contract in the product today.
    """
    start = int(getattr(calendar, "fiscal_year_start_month", 1) or 1)
    return (start - 1) % 12


def to_timeframe(year: int, quarter: Optional[int], calendar: Any = None) -> Timeframe:
    """
    A `Timeframe` for a year and quarter under a given calendar.

    With the default January calendar this is the identity, which is why
    behaviour is unchanged. The parameter exists so that fiscal-year support is
    a change to one function rather than a hunt through the engines.
    """
    if calendar is None or fiscal_offset(calendar) == 0:
        return Timeframe(year, quarter)
    # A fiscal year starting in month M labels the quarter containing M as Q1.
    # The dataset's own `_quarter` column is still calendar-based, so a future
    # implementation must shift the frame as well; refuse rather than silently
    # mislabel until that exists.
    raise NotImplementedError(
        "Fiscal calendars with a non-January start are declared in the contract "
        "but not yet applied to period slicing. Set fiscal_year_start_month=1 or "
        "extend metrics.prepare to emit fiscal period columns."
    )


def _latest(timeframes: List[Dict[str, Any]]) -> Optional[Tuple[int, int]]:
    if not timeframes:
        return None
    last = timeframes[-1]
    return int(last["year"]), int(last["quarter"])


def _detect_comparison(text: str) -> Tuple[str, bool]:
    """The baseline the question asked for, and whether it said so explicitly."""
    if any(m in text for m in _YOY_MARKERS):
        return "year_over_year", True
    if any(m in text for m in _PRIOR_MARKERS):
        return "previous_period", True
    return "previous_period", False


def resolve_period(question: str, timeframes: List[Dict[str, Any]],
                   calendar: Any = None) -> Tuple[Optional[PeriodSpec], List[Ambiguity]]:
    """
    The period a question is about.

    Returns the period plus any ambiguities. A missing or vague period never
    blocks: the investigation runs on the dataset's latest quarter and says so.
    A period the dataset does not hold *does* block, because answering about a
    different quarter than the one asked for is worse than not answering.
    """
    text = question.lower()
    ambiguities: List[Ambiguity] = []
    comparison, comparison_explicit = _detect_comparison(text)
    latest = _latest(timeframes)

    year: Optional[int] = None
    quarter: Optional[int] = None
    source = "inferred_default"

    m = _Q_YEAR.search(text) or None
    if m:
        quarter, year = int(m.group(1)), int(m.group(2))
        source = "explicit"
    else:
        m = _YEAR_Q.search(text)
        if m:
            year, quarter = int(m.group(1)), int(m.group(2))
            source = "explicit"

    if year is None:
        word = _QUARTER_WORD.search(text)
        bare_q = _BARE_QUARTER.search(text)
        bare_y = _BARE_YEAR.search(text)
        if word:
            quarter = _WORD_TO_QUARTER[word.group(1).lower()]
        elif bare_q:
            quarter = int(bare_q.group(1))
        if bare_y:
            year = int(bare_y.group(1))
        if year is not None or quarter is not None:
            source = "explicit"

    # Fill whatever the question left out from the dataset's own extent.
    if latest:
        latest_year, latest_quarter = latest
        if year is None and quarter is not None:
            year = latest_year
            ambiguities.append(Ambiguity(
                kind="period_vague", blocking=False,
                message=f"The question named Q{quarter} but not a year; "
                        f"read as Q{quarter} {latest_year}, the most recent one in this dataset.",
                assumed=f"{latest_year}-Q{quarter}"))
        elif year is not None and quarter is None:
            # A year with no quarter is a legitimate full-year request.
            source = "explicit" if source == "explicit" else source
        elif year is None and quarter is None:
            if any(p in text for p in _RELATIVE_PRIOR):
                prior = Timeframe(latest_year, latest_quarter).previous()
                year, quarter = prior.year, prior.quarter
            else:
                year, quarter = latest_year, latest_quarter
            source = "dataset_latest"

    if year is None:
        return None, [Ambiguity(
            kind="period_vague", blocking=True,
            message="This dataset has no usable time periods, so a change over time "
                    "cannot be investigated.")]

    # A period the dataset does not hold must not be silently substituted.
    if timeframes and quarter is not None:
        held = {(int(t["year"]), int(t["quarter"])) for t in timeframes}
        if (year, quarter) not in held:
            available = ", ".join(t["label"] for t in timeframes[-6:])
            return None, [Ambiguity(
                kind="unsupported_by_dataset", blocking=True,
                message=f"This dataset has no data for Q{quarter} {year}. "
                        f"Available periods include: {available}.")]

    note = ""
    if source == "dataset_latest":
        label = f"Q{quarter} {year}" if quarter else f"{year}"
        note = (f"The question did not name a period, so it was read as {label} — "
                f"the most recent period in this dataset.")
        ambiguities.append(Ambiguity(kind="period_vague", blocking=False,
                                     message=note, assumed=label))
    if not comparison_explicit:
        ambiguities.append(Ambiguity(
            kind="comparison_assumed", blocking=False,
            message="No baseline was named, so the change is measured against the "
                    "previous period.",
            assumed="previous_period"))

    return PeriodSpec(year=year, quarter=quarter, comparison=comparison,
                      source=source, assumption_note=note), ambiguities
