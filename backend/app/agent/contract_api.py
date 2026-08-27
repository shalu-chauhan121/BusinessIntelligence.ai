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
from ..kpi.resolver import available_keys as _resolver_available_keys
from .errors import UnknownDimensionError, UnknownKpiError


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
        Every dimension column on this dataset.

        `distinct_count` is only computed when a dataframe is supplied — the
        schema alone does not carry member cardinality (see the C4 gap this
        facade does not yet close: `schema.dimension_members` is read at
        `grounding.py:320` but has never existed).
        """
        out: List[DimensionInfo] = []
        for name in self._schema.dimensions:
            count = int(df[name].nunique()) if (df is not None and name in df.columns) else None
            out.append(DimensionInfo(name=name, distinct_count=count))
        return out

    def require_dimension(self, name: str) -> str:
        if name not in self._schema.dimensions:
            raise UnknownDimensionError(name, list(self._schema.dimensions))
        return name
