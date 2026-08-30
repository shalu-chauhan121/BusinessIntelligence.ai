"""
The values each dimension column actually holds.

`query/grounding.py` has always read `schema.dimension_members` to turn a named
entity in a question into a filter. That attribute has never existed on
`DatasetSchema`, so the read has always returned `{}` and `entity_filters` has
always been empty -- a question naming a member of a column bound nothing, and
the words were reported as unmapped instead. This module is the index that read
was written against.

Two policies shape it, and they are separate on purpose:

    the index cap    a column with more distinct values than `MAX_INDEXED_MEMBERS`
                     is never consulted by free-text lookup. `detect_schema`
                     admits a column as a dimension at up to 20% of row count,
                     which on a large frame is a great many values, and scanning
                     all of them against every question is both slow and a
                     generator of false matches.

    pagination       `members()` always returns a page and always reports the
                     true, uncapped `total`. Nothing here ever dumps a whole
                     column, and nothing ever silently truncates without saying
                     that it did.

Everything is computed lazily and memoised. Construction touches no column: the
catalogue is built on the dataset load path, where paying `nunique()` on every
dimension of a million-row frame is not acceptable, and most requests never look
at most columns.

The instance is shared by reference -- `dataset_service.load` hands the same
catalogue to every `dataclasses.replace(schema)` copy -- so every accessor
returns a tuple or a frozen dataclass and the memo tables are private. Nothing a
caller receives can be mutated back into another caller's view.

Like the rest of `app.agent`, this module contains no business vocabulary. It
reasons over strings and counts; deciding that a value is *worth* binding is a
linguistic judgement and belongs to the layer above.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .errors import UnknownDimensionError
from .text import normalise

# Above this many distinct values a column is not offered to free-text lookup.
# Its values stay fully reachable through explicit, paginated `members()` calls.
MAX_INDEXED_MEMBERS = 500

DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 500

# A one- or two-character value carries no evidence that a question meant it.
MIN_MEMBER_LENGTH = 3

# The widest phrase free-text lookup will ever be asked to consider. Members
# longer than this are still listed and still resolvable by name; they simply
# cannot be found by scanning a question for them.
MAX_MEMBER_TOKENS = 6

# The index is searched with n-grams the layer above generates, so both sides
# have to normalise identically or a value is filed under a key no question can
# ever produce -- and nothing reports that the lookup never fired. One shared
# implementation is what makes that impossible.


@dataclass(frozen=True)
class MemberMatch:
    """One value a phrase resolved to, and the column it belongs to."""

    dimension: str
    member: str          # the frame's own spelling, verbatim
    normalised: str


@dataclass(frozen=True)
class MemberPage:
    """One page of a column's values, with the true size of the whole."""

    dimension: str
    members: Tuple[str, ...]
    offset: int
    total: int           # the real distinct count, never the capped index size
    indexed: bool        # False when this column is too wide for free-text lookup


class MemberCatalogue:
    """
    The distinct values of each dimension column, indexed for lookup.

    Nothing is computed until it is asked for, and each answer is computed at
    most once.
    """

    def __init__(self, df: pd.DataFrame, dimensions: Sequence[str], *,
                 max_indexed: int = MAX_INDEXED_MEMBERS) -> None:
        self._df = df
        self._max_indexed = int(max_indexed)
        # `or []` would be an ambiguous truth test on a pandas Index.
        raw_columns = getattr(df, "columns", None)
        columns = set(raw_columns) if raw_columns is not None else set()
        self._dimensions: Tuple[str, ...] = tuple(
            d for d in (dimensions or []) if d in columns)
        self._values: Dict[str, Tuple[str, ...]] = {}
        self._by_normalised: Dict[str, Dict[str, str]] = {}
        self._index: Optional[Dict[str, Tuple[MemberMatch, ...]]] = None
        self._max_tokens: Optional[int] = None

    # -- catalogue -----------------------------------------------------------
    def dimensions(self) -> Tuple[str, ...]:
        """Every dimension this frame actually carries a column for."""
        return self._dimensions

    def _require(self, dimension: str) -> str:
        if dimension not in self._dimensions:
            raise UnknownDimensionError(dimension, list(self._dimensions))
        return dimension

    def _distinct(self, dimension: str) -> Tuple[str, ...]:
        """Sorted distinct values, computed once. Sorted so paging is stable."""
        cached = self._values.get(dimension)
        if cached is None:
            series = self._df[dimension].astype(str)
            cached = tuple(sorted(series.dropna().unique()))
            self._values[dimension] = cached
        return cached

    def _normalised_map(self, dimension: str) -> Dict[str, str]:
        """
        Normalised form -> the frame's own spelling, for one column.

        Built for any column on request, cap or no cap: resolving a value the
        caller named explicitly is a different question from scanning free text
        for one, and a wide column must still answer it.
        """
        cached = self._by_normalised.get(dimension)
        if cached is None:
            cached = {}
            for value in self._distinct(dimension):
                cached.setdefault(normalise(value), value)
            self._by_normalised[dimension] = cached
        return cached

    def count(self, dimension: str) -> int:
        """The true distinct cardinality of a column."""
        self._require(dimension)
        return len(self._distinct(dimension))

    def is_indexed(self, dimension: str) -> bool:
        """Whether this column is narrow enough to be searched by free text."""
        self._require(dimension)
        return len(self._distinct(dimension)) <= self._max_indexed

    def members(self, dimension: str, offset: int = 0,
                limit: int = DEFAULT_PAGE_SIZE) -> MemberPage:
        """
        One page of a column's values.

        An `offset` past the end is an empty page rather than an error -- a
        caller walking a column should stop, not fail.
        """
        self._require(dimension)
        values = self._distinct(dimension)
        offset = max(0, int(offset))
        limit = max(0, min(int(limit), MAX_PAGE_SIZE))
        return MemberPage(
            dimension=dimension,
            members=values[offset:offset + limit],
            offset=offset,
            total=len(values),
            indexed=len(values) <= self._max_indexed,
        )

    # -- lookup --------------------------------------------------------------
    def _indexable(self, value: str) -> Optional[str]:
        """
        The key a value is searchable under, or None when it must not be.

        Both exclusions are language-neutral. A purely numeric value collides
        with the years and quantities every question contains, and a value of
        one or two characters collides with everything.
        """
        key = normalise(value)
        if not key or key.replace("_", "").isdigit():
            return None
        if len(key) < MIN_MEMBER_LENGTH:
            return None
        if key.count("_") + 1 > MAX_MEMBER_TOKENS:
            return None
        return key

    def _reverse_index(self) -> Dict[str, Tuple[MemberMatch, ...]]:
        """Normalised value -> every column that holds it. Built once."""
        if self._index is not None:
            return self._index

        staged: Dict[str, List[MemberMatch]] = {}
        widest = 0
        for dimension in self._dimensions:
            if not self.is_indexed(dimension):
                continue
            for value in self._distinct(dimension):
                key = self._indexable(value)
                if key is None:
                    continue
                match = MemberMatch(dimension=dimension, member=value, normalised=key)
                bucket = staged.setdefault(key, [])
                if not any(m.dimension == match.dimension for m in bucket):
                    bucket.append(match)
                widest = max(widest, key.count("_") + 1)

        self._index = {k: tuple(v) for k, v in staged.items()}
        self._max_tokens = widest
        return self._index

    @property
    def max_member_tokens(self) -> int:
        """
        Tokens in the longest searchable value.

        The layer above widens its n-gram window to this, so a value of more
        words than the default window is still reachable -- and no wider, so a
        dataset of short values pays nothing.
        """
        if self._max_tokens is None:
            self._reverse_index()
        return int(self._max_tokens or 0)

    def lookup(self, text: str) -> Tuple[MemberMatch, ...]:
        """
        Every column whose values include this phrase.

        Returns all of them. Two columns holding the same value is a real
        ambiguity, and choosing one silently is the failure this catalogue
        exists to make impossible; the caller decides what to do about it.
        An unknown phrase returns an empty tuple rather than raising.
        """
        key = normalise(text)
        if not key:
            return ()
        return self._reverse_index().get(key, ())

    def resolve(self, dimension: str, text: str) -> Optional[str]:
        """
        The frame's own spelling of a value named in any case or spacing, or
        None when this column does not hold it.
        """
        self._require(dimension)
        key = normalise(text)
        if not key:
            return None
        return self._normalised_map(dimension).get(key)

    def as_dict(self) -> Dict[str, Tuple[str, ...]]:
        """
        Every searchable column mapped to its values.

        The shape `grounding` was originally written against. Kept for any
        caller that wants the plain form; the lookup path does not use it.
        """
        return {d: self._distinct(d) for d in self._dimensions if self.is_indexed(d)}
