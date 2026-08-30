"""
`get_timeseries` -- a KPI's history at year / quarter / month / week grain.

`observe.quarterly_series` (`engines/observe.py:91`) and `weekly_series`
(`engines/observe.py:104`) are the only series builders that have ever
existed, and neither accepts a dimension filter, neither has a monthly grain,
and neither reports that a period is missing -- a gap in the middle of the
series is silently absent from the list, indistinguishable from "this KPI
never had a quarter there" without inspecting the surrounding periods by eye.
`metrics.prepare` has written `_month` since `_year`/`_quarter`/`_period`/
`_week` were added (`metrics.py:351`); nothing has ever read it.

This module does not reimplement any of that arithmetic. `series()` is a thin
wrapper around `query.QueryEngine.query` with `group_by=[<grain column>]` --
the same call `query_kpi` makes for "revenue by quarter" -- so every point is
computed straight from that period's own rows through `ContractAPI.value`,
inheriting the ratio-correctness invariant (a group's ratio is never the mean
of member ratios, it is that group's own numerator over its own denominator)
and the full validation airlock for free. The only genuinely new code here is
enumerating the calendar labels a grain *should* hold between the first and
last period actually present, so a hole in the middle of a series is reported
as a `gap`, not silently omitted.

Gaps are never zero-filled. `metrics.prepare`'s `fillna(0.0)` on missing
metric values (`metrics.py:344`) already makes a genuinely absent measurement
indistinguishable from a real zero within one row; doing the same thing here,
at the period level, would recreate that ambiguity one layer up. A missing
period is reported in `gaps`, a covered period with a real zero value is a
point with `value: 0.0` -- the two must never be confused.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

from ..engines.metrics import safe
from .contract_api import ContractAPI
from .errors import InvalidArgumentError
from .query import QueryEngine
from .timefilter import GRAIN_COLUMNS, TimeFilter, TimeSelection

# The grains a caller may request. Each maps onto the pseudo-dimension column
# `query.QueryEngine` already accepts in `group_by` -- the same mapping
# `timefilter.Latest` uses for "trailing N periods", kept as one shared table
# so the two cannot drift on what a grain name means.
SERIES_GRAINS = ("year", "quarter", "month", "week")

_FREQ = {"year": "Y", "quarter": "Q", "month": "M", "week": "W"}


@dataclass(frozen=True)
class SeriesPoint:
    period: str
    value: Optional[float]
    rows: int

    def to_payload(self) -> Dict[str, Any]:
        return {"period": self.period, "value": safe(self.value), "rows": self.rows}


@dataclass(frozen=True)
class Series:
    kpi: str
    label: str
    unit: str
    grain: str
    kind: str            # "sum" | "mean" | "ratio"
    additive: bool        # whether points may be summed across grain for this KPI
    points: Tuple[SeriesPoint, ...]
    gaps: Tuple[str, ...]
    selection: TimeSelection
    filters: Mapping[str, Tuple[str, ...]]
    truncated: bool

    def to_payload(self) -> Dict[str, Any]:
        return {
            "kpi": self.kpi,
            "label": self.label,
            "unit": self.unit,
            "grain": self.grain,
            "kind": self.kind,
            "additive": self.additive,
            "points": [p.to_payload() for p in self.points],
            "gaps": list(self.gaps),
            "selection": self.selection.to_payload(),
            "filters": {k: list(v) for k, v in self.filters.items()},
            "truncated": self.truncated,
        }


def _quarter_label_to_period(label: str) -> pd.Period:
    return pd.Period(label, freq="Q")


def _quarter_period_to_label(p: pd.Period) -> str:
    return f"{p.year}-Q{p.quarter}"


def _week_label_to_period(label: str) -> pd.Period:
    return pd.Timestamp(label).to_period("W")


def _week_period_to_label(p: pd.Period) -> str:
    return p.start_time.date().isoformat()


# Per-grain (label -> Period, Period -> label) pair. `year` and `month` parse
# and format directly through `pd.Period`; `quarter` and `week` need the same
# translation `metrics.prepare` used to write the column in the first place.
_LABEL_PERIOD = {
    "year": (lambda label: pd.Period(label, freq="Y"), lambda p: str(p.year)),
    "quarter": (_quarter_label_to_period, _quarter_period_to_label),
    "month": (lambda label: pd.Period(label, freq="M"), str),
    "week": (_week_label_to_period, _week_period_to_label),
}


def period_ordinal(grain: str, label: str) -> int:
    """The calendar ordinal of one grain label -- a monotonic integer x-axis
    for trend fitting (`agent/trend.py`) that reflects a real gap as a real
    gap (two periods apart, not one), rather than compressing a hole in the
    series into a single step the way indexing `points` positionally would.
    Reuses the same label<->`Period` translation `_calendar_labels` already
    performs, so the two cannot drift on what a label means."""
    to_period, _ = _LABEL_PERIOD[grain]
    return to_period(label).ordinal


def label_to_period(grain: str, label: str) -> pd.Period:
    """The `pd.Period` one grain label parses to -- the same translation
    `period_ordinal` and `_calendar_labels` use, exposed directly so a caller
    (`agent/seasonality.py`, deriving a period's place in its yearly cycle)
    reads calendar structure off the label without a second, private copy of
    this table."""
    to_period, _ = _LABEL_PERIOD[grain]
    return to_period(label)


def _calendar_labels(grain: str, first: str, last: str) -> List[str]:
    """Every label `grain` should hold between `first` and `last`, inclusive.

    Only ever enumerates *within* the span the series already covers -- a
    period requested but outside that span is a coverage gap the caller
    already sees via `TimeSelection.missing`, not a series gap."""
    to_period, to_label = _LABEL_PERIOD[grain]
    start, end = to_period(first), to_period(last)
    if start > end:
        return []
    return [to_label(p) for p in pd.period_range(start, end, freq=_FREQ[grain])]


class SeriesEngine:
    """`get_timeseries` bound to one dataset's `ContractAPI`."""

    def __init__(self, api: ContractAPI):
        self._api = api
        self._query = QueryEngine(api)

    def series(self, df: pd.DataFrame, kpi_key: str, grain: str = "quarter",
              time_filter: Optional[Union[Mapping[str, Any], TimeFilter]] = None,
              filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None
              ) -> Series:
        return self.series_multi(df, [kpi_key], grain=grain, time_filter=time_filter,
                                 filters=filters)[kpi_key]

    def series_multi(self, df: pd.DataFrame, kpi_keys: Sequence[str], grain: str = "quarter",
                     time_filter: Optional[Union[Mapping[str, Any], TimeFilter]] = None,
                     filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None
                     ) -> Dict[str, Series]:
        """
        Every key in `kpi_keys`, at the same grain and time scope, in one
        grouped query -- what a scan (`agent/scan.py`) needs instead of one
        `series()` round-trip per KPI. Building 25 KPIs' history this way
        costs roughly one groupby total rather than 25, because
        `QueryEngine._compute_cells` groups once and then evaluates each KPI
        per cell (`agent/query.py:189-204`). `series()` is this with a
        one-element `kpi_keys`, kept as the single implementation so the two
        call shapes cannot drift apart.
        """
        if grain not in SERIES_GRAINS:
            raise InvalidArgumentError("grain", grain,
                                       "must be one of the supported grains.",
                                       list(SERIES_GRAINS))
        tf = time_filter if time_filter is not None else {"type": "all"}
        column = GRAIN_COLUMNS[grain]
        kpi_keys = list(kpi_keys)

        result = self._query.query(df, kpi_keys, tf, filters=filters, group_by=[column])

        out: Dict[str, Series] = {}
        for kpi_key in kpi_keys:
            points = tuple(
                SeriesPoint(period=dict(cell.group)[column],
                           value=cell.values.get(kpi_key),
                           rows=cell.rows)
                for cell in result.cells
            )

            gaps: Tuple[str, ...] = ()
            if len(points) >= 2:
                held = {p.period for p in points}
                calendar = _calendar_labels(grain, points[0].period, points[-1].period)
                gaps = tuple(label for label in calendar if label not in held)

            out[kpi_key] = Series(
                kpi=kpi_key,
                label=self._api.label(kpi_key),
                unit=self._api.unit(kpi_key),
                grain=grain,
                kind=self._api.kind(kpi_key),
                additive=self._api.is_additive(kpi_key),
                points=points,
                gaps=gaps,
                selection=result.selection,
                filters=result.filters,
                truncated=result.truncated,
            )
        return out
