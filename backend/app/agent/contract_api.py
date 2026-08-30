"""
The one sanctioned entry point onto a dataset's KPI Contract.

Every engine call that needs a KPI's meaning or value threads a `resolver`
argument by hand, and six call sites across `contest.py`, `act.py`, and
`drivers.py` had that argument in scope and simply did not pass it -- see
`test_resolver_propagation.py` for the reproduction. The bug was not
carelessness at one site; it was the shape of the API. Passing the same
value through forty call sites is a matter of time before one of them
forgets it.

`ContractAPI` closes that class of bug by binding the resolver once, at
construction, as instance state. There is no `compute(df, key)` call left
anywhere in this module that *could* omit it -- the resolver simply is not a
parameter any caller supplies per call. New code built on top of the agent
tool layer is required to go through this facade rather than
`app.engines.metrics` directly; `metrics.compute` itself is left unchanged
and stays lenient, since existing engine code depends on that leniency during
the migration described in the task board.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import pandas as pd

from ..engines.metrics import (
    DatasetSchema,
    Resolver,
    compute,
    higher_is_better,
    metric_components,
    metric_description,
    metric_label,
    metric_spec,
    metric_unit,
)
from ..engines.driver_graph import _tags
from ..kpi.resolver import available_keys as _resolver_available_keys
from .dimensions import (DEFAULT_PAGE_SIZE, MemberCatalogue, MemberMatch,
                         MemberPage)
from .errors import UnknownDimensionError, UnknownKpiError, UnknownMemberError


@dataclass(frozen=True)
class KpiInfo:
    key: str
    label: str
    unit: str
    higher_is_better: bool
    description: str
    status: str
    granularity: str
    source: str  # "contract" | "seed" -- provenance, visible rather than silent


@dataclass(frozen=True)
class DimensionInfo:
    name: str
    distinct_count: Optional[int] = None
    # False when the column is too wide to be searched by free text. Its values
    # stay listable and resolvable; only the scan is withheld.
    indexed: bool = True


class ContractAPI:
    """
    A read-only facade over one dataset's `DatasetSchema` and its compiled KPI
    Contract (`schema.contract_resolver`).

    Works identically for a contract-backed dataset and a contract-less one:
    with no contract, every lookup falls back to the seed `METRICS` registry,
    exactly as `app.engines.metrics` already does, and `KpiInfo.source`
    reports `"seed"` so the provenance is visible rather than silent.
    """

    def __init__(self, schema: DatasetSchema):
        self._schema = schema
        self._resolver: Optional[Resolver] = schema.contract_resolver

    # -- identity / catalogue ------------------------------------------------
    @property
    def resolver(self) -> Optional[Resolver]:
        """The bound resolver, for the rare caller that must hand it onward
        to a legacy function not yet wrapped by this facade."""
        return self._resolver

    def list_kpi_keys(self) -> List[str]:
        return list(self._schema.available_kpis)

    def has(self, key: str) -> bool:
        return metric_spec(key, self._resolver) is not None

    def require(self, key: str) -> Any:
        """The KPI spec for `key`, or a typed, recoverable error naming valid
        alternatives -- never `None`, never a silent NaN downstream."""
        spec = metric_spec(key, self._resolver)
        if spec is None:
            raise UnknownKpiError(key, self.list_kpi_keys())
        return spec

    def list_kpis(self) -> List[KpiInfo]:
        out: List[KpiInfo] = []
        for key in self._schema.available_kpis:
            spec = metric_spec(key, self._resolver)
            source = "contract" if (self._resolver is not None and key in self._resolver) else "seed"
            out.append(KpiInfo(
                key=key,
                label=metric_label(key, self._resolver),
                unit=metric_unit(key, self._resolver),
                higher_is_better=higher_is_better(key, self._resolver),
                description=metric_description(key, self._resolver),
                status=getattr(spec, "status", "") if spec is not None else "",
                granularity=getattr(spec, "granularity_label", "") if spec is not None else "",
                source=source,
            ))
        return out

    def available_keys(self, df: pd.DataFrame) -> List[str]:
        """KPI keys this specific dataframe can actually answer -- source
        columns present, not merely declared in the contract."""
        if self._resolver is not None:
            return _resolver_available_keys(self._resolver, df)
        return [k for k in self._schema.available_kpis if k in df.columns]

    def metric_columns(self) -> List[str]:
        """The raw source columns `prepare` treats as measures -- what
        `check_data_quality` (`agent/quality.py`) reports zero-fill and null
        counts over. Distinct from a KPI key: a KPI can be a formula over
        several of these."""
        return list(self._schema.base_metrics) + list(self._schema.extra_metrics)

    def imputed_cells(self) -> Optional[Any]:
        """Per-column count of cells `prepare`'s `fillna(0.0)` fabricated
        (`metrics.py:344`), or `None` when `preserve_missing` was on and
        nothing was filled. This is the one place a fabricated zero is still
        distinguishable from a real one after `prepare` has already made them
        byte-identical in the frame itself."""
        return self._schema.imputed_cells

    def quarters_held(self) -> int:
        """How many distinct quarters this dataset holds, dataset-wide --
        what `detect_schema`'s 'fewer than 5 quarters of history' warning
        (`metrics.py:296-301`) is about, as a typed fact rather than a
        sentence."""
        return len(self._schema.quarters)

    # -- semantics -------------------------------------------------------------
    def get_definition(self, key: str) -> Optional[Any]:
        """The full `KpiDefinition` behind `key`, or `None` for a contract-less
        dataset or a seed-only key. Callers that need the definition to exist
        should call `require` first."""
        return self._schema.kpi_definition(key)

    def label(self, key: str) -> str:
        return metric_label(key, self._resolver)

    def unit(self, key: str) -> str:
        return metric_unit(key, self._resolver)

    def polarity(self, key: str) -> bool:
        """True when a higher value of this KPI is better."""
        return higher_is_better(key, self._resolver)

    def description(self, key: str) -> str:
        return metric_description(key, self._resolver)

    def source_fields(self, key: str) -> List[str]:
        """The raw columns this KPI reads, from the compiled spec or its
        definition. What binds a question's vocabulary to a KPI through the
        concept library, so the search layer must not reach past the facade
        into `contract_resolver` for it."""
        spec = metric_spec(key, self._resolver)
        fields = getattr(spec, "source_fields", None) if spec is not None else None
        if not fields:
            definition = self.get_definition(key)
            fields = getattr(definition, "source_fields", None) or []
        return [str(f) for f in fields]

    def tags(self, key: str) -> List[str]:
        """The semantic tags discovery assigned this KPI."""
        return sorted(_tags(metric_spec(key, self._resolver)))

    def relevance(self, key: str) -> str:
        """Why this KPI matters to this business, when the contract says so."""
        definition = self.get_definition(key)
        return getattr(definition, "relevance", "") or ""

    def business_definition(self, key: str) -> str:
        """The contract's prose definition of what this KPI means."""
        definition = self.get_definition(key)
        return getattr(definition, "business_definition", "") or ""

    def is_additive(self, key: str) -> bool:
        spec = self.require(key)
        return bool(getattr(spec, "additive", False))

    def kind(self, key: str) -> str:
        """'sum' | 'mean' | 'ratio'."""
        spec = self.require(key)
        return getattr(spec, "kind", "sum")

    # -- evaluation — the only sanctioned compute path ------------------------
    def value(self, df: pd.DataFrame, key: str) -> float:
        """
        Aggregate `key` over `df`.

        `require`d first so an unknown key raises `UnknownKpiError` here,
        rather than falling through to `metrics.compute`'s lenient NaN, which
        is indistinguishable from "the KPI is genuinely zero" downstream.
        """
        self.require(key)
        return compute(df, key, self._resolver)

    def components(self, df: pd.DataFrame, key: str) -> Tuple[Optional[float], Optional[float]]:
        """(numerator, denominator) for a ratio KPI; `(None, None)` otherwise."""
        self.require(key)
        return metric_components(df, key, self._resolver)

    # -- dimensions ------------------------------------------------------------
    def list_dimensions(self, df: Optional[pd.DataFrame] = None) -> List[DimensionInfo]:
        """
        Every dimension column on this dataset, with its cardinality.

        `distinct_count` is the true, uncapped count -- never the size of the
        lookup index, which is capped. It comes from the attached catalogue, or
        from `df` when one is supplied, and is None only when neither is.
        """
        catalogue = self._catalogue(df, required=False)
        out: List[DimensionInfo] = []
        for name in self._schema.dimensions:
            count: Optional[int] = None
            indexed = True
            if df is not None and name in df.columns:
                count = int(df[name].nunique())
            if catalogue is not None and name in catalogue.dimensions():
                count = catalogue.count(name)
                indexed = catalogue.is_indexed(name)
            out.append(DimensionInfo(name=name, distinct_count=count, indexed=indexed))
        return out

    def require_dimension(self, name: str) -> str:
        if name not in self._schema.dimensions:
            raise UnknownDimensionError(name, list(self._schema.dimensions))
        return name

    # -- dimension members -----------------------------------------------------
    def _catalogue(self, df: Optional[pd.DataFrame] = None, *,
                   required: bool = True) -> Optional[MemberCatalogue]:
        """
        The attached catalogue, or one built from a supplied frame.

        `dataset_service.load` attaches a catalogue, but a schema assembled by a
        bare `detect_schema`/`prepare` -- which is how much of the test suite and
        the KPI bootstrap path build theirs -- has none. Accepting `df` mirrors
        `list_dimensions` and keeps those callers working, rather than making
        them silently see a dataset with no members.
        """
        attached = getattr(self._schema, "member_catalogue", None)
        if attached is not None:
            return attached
        if df is not None:
            return MemberCatalogue(df, self._schema.dimensions)
        if required:
            raise UnknownDimensionError(
                "<no member catalogue>", list(self._schema.dimensions))
        return None

    def member_catalogue(self, df: Optional[pd.DataFrame] = None) -> Optional[MemberCatalogue]:
        """The member index for this dataset, when one is reachable."""
        return self._catalogue(df, required=False)

    def list_members(self, dimension: str, offset: int = 0,
                     limit: int = DEFAULT_PAGE_SIZE,
                     df: Optional[pd.DataFrame] = None) -> MemberPage:
        """
        One page of the values a dimension holds.

        `require_dimension` first, so a dimension the model invented is an
        `UnknownDimensionError` before any pandas work happens.
        """
        self.require_dimension(dimension)
        return self._catalogue(df).members(dimension, offset=offset, limit=limit)

    def resolve_member(self, dimension: str, text: str,
                       df: Optional[pd.DataFrame] = None) -> str:
        """
        The frame's own spelling of a value named in any case or spacing.

        Raises rather than returning None: a filter on a member that does not
        exist yields an empty slice, and an empty slice is indistinguishable
        from a real zero once it reaches a number.
        """
        self.require_dimension(dimension)
        catalogue = self._catalogue(df)
        resolved = catalogue.resolve(dimension, text)
        if resolved is None:
            page = catalogue.members(dimension, limit=UnknownMemberError.MAX_ALTERNATIVES)
            raise UnknownMemberError(dimension, text, list(page.members))
        return resolved

    def find_members(self, text: str,
                     df: Optional[pd.DataFrame] = None) -> List[MemberMatch]:
        """
        Every dimension whose values include this phrase.

        All of them, never one: a value held by two columns is a real ambiguity
        and the caller decides what to do about it.
        """
        catalogue = self._catalogue(df, required=False)
        return list(catalogue.lookup(text)) if catalogue is not None else []
