"""
The `TimeFilter` union -- a general time scope, closed over a boolean row mask.

`observe.slice_period` (`engines/observe.py:68`) filters on `_year` and, at
most, `_quarter`. That is the entire vocabulary the product has ever had for
"when": a question like *"total revenue from 2022 to 2026"* or *"revenue in
all odd-numbered years"* has no code path at all, because there is no shape
that can express it. This module is that shape.

A `TimeFilter` is a discriminated union over eight cases -- `quarter`, `year`,
`years`, `quarters`, `range`, `months`, `latest`, `all` -- each of which knows
how to turn itself into a boolean mask over a prepared frame (one that has been
through `engines.metrics.prepare`, so `_date`, `_year`, `_quarter`, `_period`,
`_month` and `_week` all exist). `months` is the first thing in the codebase to
read `_month`, which `prepare` has written since it was added and nothing has
ever consulted.

`parse()` is the only constructor the tool boundary uses; it turns a plain
`dict` (what a model's tool call actually sends) into a typed filter or raises
`MalformedTimeFilterError` naming the valid shapes. `resolve()` applies a
filter to a frame and reports what was actually covered: `TimeSelection`
carries the requested spec, the rows matched, the real date extent covered,
and a `missing` list of anything requested but not held. A selection matching
zero rows is a different failure from one that is merely incomplete -- the
former raises `EmptyPeriodError`, the latter is a real, honest answer.

This module is left standing next to `observe.slice_period` and `Timeframe`
rather than replacing them, matching how `contract_api.py` left `metrics.compute`
itself unchanged during its migration: existing engine code keeps working, and
`to_timeframe()` is the one bridge back for the one case (`quarter`/`year`)
where the two overlap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from ..engines.observe import Timeframe
from .errors import EmptyPeriodError, MalformedTimeFilterError

VALID_TYPES = ("quarter", "year", "years", "quarters", "range", "months", "latest", "all")

# Grain columns `group_by` and `latest` may address -- all written by
# `engines.metrics.prepare`.
GRAIN_COLUMNS = {
    "year": "_year",
    "quarter": "_period",   # a quarter is identified by (_year, _quarter); _period is the label
    "period": "_period",
    "month": "_month",
    "week": "_week",
}


def _fail(spec: Any, reason: str) -> "MalformedTimeFilterError":
    return MalformedTimeFilterError(spec, reason, list(VALID_TYPES))


def _require_keys(spec: Mapping[str, Any], keys: Sequence[str]) -> None:
    missing = [k for k in keys if k not in spec or spec[k] is None]
    if missing:
        raise _fail(spec, f"missing required key(s): {', '.join(missing)}")


def _as_int(spec: Any, value: Any, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise _fail(spec, f"'{field_name}' must be an integer, got {value!r}")


def _as_quarter(spec: Any, value: Any) -> int:
    q = _as_int(spec, value, "quarter")
    if q not in (1, 2, 3, 4):
        raise _fail(spec, f"'quarter' must be 1-4, got {q}")
    return q


def _as_date(spec: Any, value: Any, field_name: str) -> pd.Timestamp:
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError):
        raise _fail(spec, f"'{field_name}' could not be parsed as a date: {value!r}")
    if pd.isna(ts):
        raise _fail(spec, f"'{field_name}' could not be parsed as a date: {value!r}")
    return ts


def _as_month(spec: Any, value: Any, field_name: str) -> pd.Period:
    try:
        return pd.Period(str(value), freq="M")
    except (TypeError, ValueError):
        raise _fail(spec, f"'{field_name}' could not be parsed as a year-month "
                          f"(expected 'YYYY-MM'): {value!r}")


def _period_labels(df: pd.DataFrame, limit: int = EmptyPeriodError.MAX_AVAILABLE) -> List[str]:
    """The `_period` labels this frame actually holds, most recent last."""
    if "_period" not in df.columns:
        return []
    labels = sorted(df["_period"].dropna().unique().tolist())
    return labels[-limit:]


# ---------------------------------------------------------------------------
# the union
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TimeFilter:
    """Base class. Every case implements `mask` and `missing`."""

    raw: Any = field(repr=False, compare=False)

    def mask(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def missing(self, df: pd.DataFrame) -> Tuple[str, ...]:
        """Labels requested but not present in `df`. Empty when fully covered."""
        return ()

    def to_timeframe(self) -> Optional[Timeframe]:
        """A legacy `observe.Timeframe`, for the one case this maps onto. None
        for every case with no single-timeframe equivalent."""
        return None

    def to_payload(self) -> Any:
        return self.raw


@dataclass(frozen=True)
class Quarter(TimeFilter):
    year: int = 0
    quarter: int = 0

    def mask(self, df: pd.DataFrame) -> pd.Series:
        return (df["_year"] == self.year) & (df["_quarter"] == self.quarter)

    def missing(self, df: pd.DataFrame) -> Tuple[str, ...]:
        label = f"{self.year}-Q{self.quarter}"
        return () if bool(self.mask(df).any()) else (label,)

    def to_timeframe(self) -> Optional[Timeframe]:
        return Timeframe(self.year, self.quarter)


@dataclass(frozen=True)
class Year(TimeFilter):
    year: int = 0

    def mask(self, df: pd.DataFrame) -> pd.Series:
        return df["_year"] == self.year

    def missing(self, df: pd.DataFrame) -> Tuple[str, ...]:
        return () if bool(self.mask(df).any()) else (str(self.year),)

    def to_timeframe(self) -> Optional[Timeframe]:
        return Timeframe(self.year, None)


@dataclass(frozen=True)
class YearSet(TimeFilter):
    values: Tuple[int, ...] = ()

    def mask(self, df: pd.DataFrame) -> pd.Series:
        return df["_year"].isin(self.values)

    def missing(self, df: pd.DataFrame) -> Tuple[str, ...]:
        held = set(df["_year"].dropna().astype(int).unique().tolist())
        return tuple(str(y) for y in self.values if y not in held)


@dataclass(frozen=True)
class QuarterSet(TimeFilter):
    values: Tuple[Tuple[int, int], ...] = ()   # (year, quarter) pairs

    def mask(self, df: pd.DataFrame) -> pd.Series:
        mask = pd.Series(False, index=df.index)
        for year, quarter in self.values:
            mask = mask | ((df["_year"] == year) & (df["_quarter"] == quarter))
        return mask

    def missing(self, df: pd.DataFrame) -> Tuple[str, ...]:
        held = set(zip(df["_year"].dropna().astype(int), df["_quarter"].dropna().astype(int)))
        return tuple(f"{y}-Q{q}" for (y, q) in self.values if (y, q) not in held)


@dataclass(frozen=True)
class DateRange(TimeFilter):
    start: pd.Timestamp = None
    end: pd.Timestamp = None

    def mask(self, df: pd.DataFrame) -> pd.Series:
        return (df["_date"] >= self.start) & (df["_date"] <= self.end)

    def missing(self, df: pd.DataFrame) -> Tuple[str, ...]:
        dates = df["_date"].dropna()
        if dates.empty:
            return (f"{self.start.date()}..{self.end.date()}",)
        held_start, held_end = dates.min(), dates.max()
        missing: List[str] = []
        if self.start < held_start:
            missing.append(f"before {held_start.date()}")
        if self.end > held_end:
            missing.append(f"after {held_end.date()}")
        return tuple(missing)


@dataclass(frozen=True)
class MonthRange(TimeFilter):
    start: pd.Period = None
    end: pd.Period = None

    def mask(self, df: pd.DataFrame) -> pd.Series:
        months = pd.PeriodIndex(df["_month"], freq="M")
        return (months >= self.start) & (months <= self.end)

    def missing(self, df: pd.DataFrame) -> Tuple[str, ...]:
        months = pd.PeriodIndex(df["_month"].dropna(), freq="M")
        if len(months) == 0:
            return (f"{self.start}..{self.end}",)
        held_start, held_end = months.min(), months.max()
        missing: List[str] = []
        if self.start < held_start:
            missing.append(f"before {held_start}")
        if self.end > held_end:
            missing.append(f"after {held_end}")
        return tuple(missing)


@dataclass(frozen=True)
class Latest(TimeFilter):
    grain: str = "quarter"
    n: int = 1

    def _held_labels(self, df: pd.DataFrame) -> List[Any]:
        col = GRAIN_COLUMNS.get(self.grain)
        if col is None or col not in df.columns:
            return []
        return sorted(df[col].dropna().unique().tolist())

    def mask(self, df: pd.DataFrame) -> pd.Series:
        labels = self._held_labels(df)
        keep = set(labels[-self.n:])
        col = GRAIN_COLUMNS.get(self.grain)
        if col is None or col not in df.columns:
            return pd.Series(False, index=df.index)
        return df[col].isin(keep)

    def missing(self, df: pd.DataFrame) -> Tuple[str, ...]:
        labels = self._held_labels(df)
        if len(labels) < self.n:
            return (f"only {len(labels)} of {self.n} requested {self.grain}(s) held",)
        return ()


@dataclass(frozen=True)
class AllTime(TimeFilter):
    def mask(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(True, index=df.index)


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------
def parse(spec: Mapping[str, Any]) -> TimeFilter:
    """
    The only constructor a tool boundary should use.

    Raises `MalformedTimeFilterError` -- never a bare `KeyError`/`ValueError`
    -- so a model's malformed tool call becomes a recoverable turn.
    """
    if not isinstance(spec, Mapping):
        raise _fail(spec, "a time filter must be an object")
    kind = spec.get("type")
    if kind not in VALID_TYPES:
        raise _fail(spec, f"unknown type {kind!r}")

    if kind == "quarter":
        _require_keys(spec, ("year", "quarter"))
        return Quarter(raw=dict(spec), year=_as_int(spec, spec["year"], "year"),
                      quarter=_as_quarter(spec, spec["quarter"]))

    if kind == "year":
        _require_keys(spec, ("year",))
        return Year(raw=dict(spec), year=_as_int(spec, spec["year"], "year"))

    if kind == "years":
        _require_keys(spec, ("values",))
        values = spec["values"]
        if not isinstance(values, (list, tuple)) or not values:
            raise _fail(spec, "'values' must be a non-empty list of years")
        years = tuple(_as_int(spec, v, "values") for v in values)
        return YearSet(raw=dict(spec), values=years)

    if kind == "quarters":
        _require_keys(spec, ("values",))
        values = spec["values"]
        if not isinstance(values, (list, tuple)) or not values:
            raise _fail(spec, "'values' must be a non-empty list of {year, quarter} objects")
        pairs: List[Tuple[int, int]] = []
        for item in values:
            if not isinstance(item, Mapping):
                raise _fail(spec, f"each 'quarters' entry must be an object, got {item!r}")
            _require_keys(item, ("year", "quarter"))
            pairs.append((_as_int(spec, item["year"], "year"),
                         _as_quarter(spec, item["quarter"])))
        return QuarterSet(raw=dict(spec), values=tuple(pairs))

    if kind == "range":
        _require_keys(spec, ("start", "end"))
        start = _as_date(spec, spec["start"], "start")
        end = _as_date(spec, spec["end"], "end")
        if start > end:
            raise _fail(spec, f"'start' ({start.date()}) is after 'end' ({end.date()})")
        return DateRange(raw=dict(spec), start=start, end=end)

    if kind == "months":
        _require_keys(spec, ("start", "end"))
        start = _as_month(spec, spec["start"], "start")
        end = _as_month(spec, spec["end"], "end")
        if start > end:
            raise _fail(spec, f"'start' ({start}) is after 'end' ({end})")
        return MonthRange(raw=dict(spec), start=start, end=end)

    if kind == "latest":
        _require_keys(spec, ("grain", "n"))
        grain = spec["grain"]
        if grain not in GRAIN_COLUMNS:
            raise _fail(spec, f"'grain' must be one of {sorted(GRAIN_COLUMNS)}, got {grain!r}")
        n = _as_int(spec, spec["n"], "n")
        if n < 1:
            raise _fail(spec, f"'n' must be >= 1, got {n}")
        return Latest(raw=dict(spec), grain=grain, n=n)

    # kind == "all"
    return AllTime(raw=dict(spec))


# ---------------------------------------------------------------------------
# applying a filter
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TimeSelection:
    """What applying a `TimeFilter` to a frame actually covered."""

    requested: Any
    rows: int
    covered_start: Optional[str]
    covered_end: Optional[str]
    periods: Tuple[str, ...]
    missing: Tuple[str, ...]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "requested": self.requested,
            "rows": self.rows,
            "covered_start": self.covered_start,
            "covered_end": self.covered_end,
            "periods": list(self.periods),
            "missing": list(self.missing),
        }


def resolve(df: pd.DataFrame, tf: TimeFilter) -> TimeSelection:
    """
    Apply `tf` to `df` and describe what was covered.

    Raises `EmptyPeriodError` when the mask matches zero rows -- an empty
    selection must never be mistaken for a real, computed zero. Partial
    coverage (some but not all of what was requested) is not an error: it is
    reported through `missing` alongside a genuine answer over what was held.
    """
    mask = tf.mask(df)
    matched = df[mask]
    if len(matched) == 0:
        raise EmptyPeriodError(tf.to_payload(), _period_labels(df))

    dates = matched["_date"].dropna()
    return TimeSelection(
        requested=tf.to_payload(),
        rows=int(len(matched)),
        covered_start=str(dates.min().date()) if len(dates) else None,
        covered_end=str(dates.max().date()) if len(dates) else None,
        periods=tuple(sorted(matched["_period"].dropna().unique().tolist()))
                if "_period" in matched.columns else (),
        missing=tf.missing(df),
    )


def apply(df: pd.DataFrame, tf: TimeFilter) -> Tuple[pd.DataFrame, TimeSelection]:
    """The filtered frame, plus the `TimeSelection` describing it."""
    selection = resolve(df, tf)
    return df[tf.mask(df)], selection
