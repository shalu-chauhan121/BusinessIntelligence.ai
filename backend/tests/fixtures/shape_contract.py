"""
Formula shapes that no discovered contract produces.

Discovery binds KPIs from the library or from measure columns, so every formula
it emits is one level deep over raw columns: `{readmissions}/{discharges}`.
Three shapes therefore cannot be obtained from real data at all, and each one is
exactly where the set-containment heuristic this group replaces goes wrong:

    nested     a KPI whose formula references another KPI, so the component two
               levels down is not in this KPI's own field set
    cyclic     two KPIs referencing each other -- traversal must terminate
    malformed  a formula `compile_contract` silently drops (`resolver.py:397`)

These are built as `KpiDefinition` objects directly rather than as a CSV
fixture, because the point is the formula shape, not the data. Column names are
deliberately meaningless: a test written over `alpha`/`beta` cannot pass by
recognising a business term, which is the property the retail and healthcare
suites cannot check about themselves.
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd

from app.kpi.contract import (
    FormulaSpec,
    KpiContract,
    KpiDefinition,
    Provenance,
)


def _kpi(kpi_id: str, formula: FormulaSpec, source_fields: List[str],
         semantic_tags: Optional[List[str]] = None) -> KpiDefinition:
    return KpiDefinition(
        kpi_id=kpi_id,
        name=kpi_id.replace("_", " ").title(),
        formula=formula,
        source_fields=list(source_fields),
        semantic_tags=list(semantic_tags or []),
        provenance=Provenance(origin="user_defined"),
        status="approved",
    )


def _contract(kpis: List[KpiDefinition]) -> KpiContract:
    return KpiContract(
        contract_id="shape_test",
        uid="shape_uid",
        dataset_id="shape_dataset",
        status="approved",
        kpis=kpis,
    )


# ---------------------------------------------------------------------------
# nested: chi = gamma / alpha, and gamma = alpha - beta
#
# `chi`'s own referenced fields are {gamma, alpha}. `beta` is NOT among them, so
# set containment (`driver_graph.py:162-179`) cannot reach it -- while the AST,
# expanded one level, finds it at depth 1 with sign -1.
# ---------------------------------------------------------------------------
def nested_contract() -> KpiContract:
    return _contract([
        _kpi("alpha", FormulaSpec(expression="{alpha}", kind="sum"), ["alpha"]),
        _kpi("beta", FormulaSpec(expression="{beta}", kind="sum"), ["beta"]),
        _kpi("gamma", FormulaSpec(expression="{alpha} - {beta}", kind="sum"),
             ["alpha", "beta"]),
        _kpi("chi", FormulaSpec(expression="", kind="ratio",
                                numerator_expression="{gamma}",
                                denominator_expression="{alpha}"),
             ["gamma", "alpha"]),
    ])


def nested_frame() -> pd.DataFrame:
    """A frame carrying every leaf column, plus `gamma` as a materialised column.

    `gamma` has to exist as a column for `chi` to be computable at all, which is
    what makes the nesting real rather than a formula that cannot run.
    """
    rows = []
    for i in range(24):
        alpha = 100.0 + i * 5
        beta = 40.0 + (i % 6) * 3
        rows.append({
            "date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=7 * i),
            "segment": ["one", "two", "three"][i % 3],
            "alpha": alpha,
            "beta": beta,
            "gamma": alpha - beta,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# cyclic: each references the other. Nothing legitimate produces this; a
# traversal that does not guard against it hangs.
# ---------------------------------------------------------------------------
def cyclic_contract() -> KpiContract:
    return _contract([
        _kpi("ping", FormulaSpec(expression="{pong} + 1", kind="sum"), ["pong"]),
        _kpi("pong", FormulaSpec(expression="{ping} + 1", kind="sum"), ["ping"]),
    ])


# ---------------------------------------------------------------------------
# malformed: `compile_contract` drops this silently, so the KPI simply is not in
# the resolver and nothing records why.
# ---------------------------------------------------------------------------
def malformed_contract() -> KpiContract:
    return _contract([
        _kpi("sound", FormulaSpec(expression="{alpha} +", kind="sum"), ["alpha"]),
        _kpi("broken", FormulaSpec(expression="{alpha} @@ {beta}", kind="sum"),
             ["alpha", "beta"]),
    ])


# ---------------------------------------------------------------------------
# dual role: alpha appears in BOTH halves of the ratio.
# `driver_graph.py:162` tests the numerator first and `continue`s, so the
# denominator role is lost. Mirrors retail's `gross_margin_pct` with no
# business vocabulary attached.
# ---------------------------------------------------------------------------
def dual_role_contract() -> KpiContract:
    return _contract([
        _kpi("alpha", FormulaSpec(expression="{alpha}", kind="sum"), ["alpha"]),
        _kpi("beta", FormulaSpec(expression="{beta}", kind="sum"), ["beta"]),
        _kpi("delta", FormulaSpec(expression="", kind="ratio",
                                  numerator_expression="{alpha} - {beta}",
                                  denominator_expression="{alpha}", scale=100.0),
             ["alpha", "beta"]),
    ])
