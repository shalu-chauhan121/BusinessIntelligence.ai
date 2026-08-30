"""
Tool functions with no engine class behind them.

Every other tool in this package is a method on an `*Engine` built the same
way -- `__init__(self, api: ContractAPI)`, `df` as the method's first
positional argument. `ContractAPI` itself and `formula.structure` do not fit
that shape (`ContractAPI` takes a `DatasetSchema`, not itself; `structure` is
a module-level function taking `api` as its own first argument), so the
handful of orientation and formula-introspection tools built on them live
here instead, kept out of `registry.py` so that file stays about schemas and
dispatch only.

Every function below follows the same binding convention `registry.dispatch`
expects of any tool callable: an `api: ContractAPI` parameter and, where the
underlying primitive needs the frame, a `df: pd.DataFrame` parameter -- both
always bound from the request context, never supplied by the model. Every
other parameter is model-visible.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from ..engines.observe import available_timeframes
from . import formula as _formula
from .contract_api import ContractAPI


def describe_dataset(api: ContractAPI, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Orientation: what this dataset holds -- row count, date extent, every
    quarter held, every KPI and dimension, and whether any metric column
    was zero-filled by `prepare` rather than genuinely observed as zero.

    Seeded into the system prompt at loop start (per the master plan's
    prompt-seeding decision) so a Tier-1 question never wastes a round-trip
    discovering that `revenue` exists.
    """
    kpis = api.list_kpis()
    dims = api.list_dimensions(df)
    imputed = api.imputed_cells()
    return {
        "rows": int(len(df)),
        "quarters_held": api.quarters_held(),
        "timeframes": available_timeframes(df),
        "kpis": [
            {"key": k.key, "label": k.label, "unit": k.unit,
             "higher_is_better": k.higher_is_better, "source": k.source}
            for k in kpis
        ],
        "dimensions": [
            {"name": d.name, "distinct_count": d.distinct_count, "indexed": d.indexed}
            for d in dims
        ],
        # `prepare` fills a genuinely missing metric with 0.0 (metrics.py:301-303),
        # making it byte-identical to a real zero everywhere except here.
        "has_imputed_cells": bool(imputed),
    }


def list_kpis(api: ContractAPI) -> Dict[str, Any]:
    """The full KPI catalogue this dataset measures -- key, label, unit and
    direction for each. Call this for a fresh listing; `describe_dataset`
    already includes it once, at loop start."""
    kpis = api.list_kpis()
    return {
        "kpis": [
            {"key": k.key, "label": k.label, "unit": k.unit,
             "higher_is_better": k.higher_is_better, "source": k.source}
            for k in kpis
        ],
    }


def list_dimensions(api: ContractAPI, df: pd.DataFrame) -> Dict[str, Any]:
    """Every dimension this dataset can be sliced or grouped by, with its
    cardinality. Call this before `list_dimension_members` if you are not
    sure which dimension name to page through."""
    dims = api.list_dimensions(df)
    return {
        "dimensions": [
            {"name": d.name, "distinct_count": d.distinct_count, "indexed": d.indexed}
            for d in dims
        ],
    }


def get_kpi_definition(api: ContractAPI, key: str) -> Dict[str, Any]:
    """
    The structured semantic record for one KPI: its formula's meaning as
    facts (unit, direction, aggregation kind), its tags, and the raw fields
    it reads -- never a rephrased sentence. `ContractAPI` also carries
    human-authored `business_definition`/`relevance` prose for KPI Studio
    (`kpi/explanation.py`); that stays out of the tool boundary on purpose,
    per the one rule every tool result here follows: facts only, nothing a
    caller would just relay verbatim.

    Call this when a question needs to understand what a KPI *means* --
    "what does gross margin mean here" -- not its value.
    """
    api.require(key)
    return {
        "key": key,
        "label": api.label(key),
        "unit": api.unit(key),
        "higher_is_better": api.polarity(key),
        "kind": api.kind(key),
        "is_additive": api.is_additive(key),
        "source_fields": api.source_fields(key),
        "tags": api.tags(key),
    }


def get_kpi_inputs(api: ContractAPI, key: str) -> Dict[str, Any]:
    """The raw source columns a KPI's formula reads. Call this to see what
    feeds a KPI without pulling its full semantic record."""
    api.require(key)
    return {"key": key, "source_fields": api.source_fields(key)}


def list_dimension_members(api: ContractAPI, df: pd.DataFrame, dimension: str,
                           offset: int = 0, limit: int = 50) -> Dict[str, Any]:
    """
    One page of the values a dimension column actually holds -- e.g. every
    region name. Call this before filtering by a member you are not
    certain is spelled the way this dataset spells it.
    """
    page = api.list_members(dimension, offset=offset, limit=limit, df=df)
    return {
        "dimension": page.dimension,
        "members": list(page.members),
        "offset": page.offset,
        "total": page.total,
        "indexed": page.indexed,
    }


def get_formula_structure(api: ContractAPI, key: str, expand: bool = True,
                          max_depth: int = _formula.DEFAULT_MAX_DEPTH) -> Dict[str, Any]:
    """
    A KPI's formula decomposed into its components -- every field it reads,
    with its sign (additive vs. subtractive) and position (numerator vs.
    denominator), including fields reached through another KPI's formula.

    Call this before `decompose_formula` when the question is about *how* a
    KPI is built (e.g. "is margin a ratio, and what's in the denominator")
    rather than how it changed between two periods.
    """
    struct = _formula.structure(api, key, expand=expand, max_depth=max_depth)
    return {
        "key": struct.key,
        "kind": struct.kind,
        "scale": struct.scale,
        "terms": [
            {"field": t.field, "sign": t.sign, "position": t.position,
             "role": t.role, "depth": t.depth, "via": list(t.via)}
            for t in struct.terms
        ],
        "constants": list(struct.constants),
        "leaf_fields": list(struct.leaf_fields),
        "component_kpis": list(struct.component_kpis),
        "source": struct.source,
        "truncated": struct.truncated,
    }
