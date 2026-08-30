"""
What could plausibly have moved a KPI, derived from the contract rather than guessed.

The old investigation asked a fixed library of retail explanations which of them
happened to fit. This module replaces the fixed part: it reads the KPI Contract
and works out, for a particular outcome KPI, which other KPIs stand in a real
relationship to it — because they appear in its formula, because they are built
from the same source fields, or because discovery recorded a derivation between
them.

Those relationships are structural facts, not opinions, which is what makes them
safe to reason from. A language model may propose additional edges the contract
does not encode, but a proposed edge carries `weight = 0.0` until a correlation
and lead/lag check on the actual data supports it. An unverified edge can suggest
a hypothesis; it can never lend that hypothesis weight.

One deliberate departure from the contract: `KpiDefinition.dimensions` lists every
dimension in the dataset for every KPI, because that is how discovery populates
it. Dimension relevance is therefore established here, empirically, from whether a
dimension actually explains any of the movement.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Set

import pandas as pd

from ..kpi.resolver import BinOp, FieldRef, referenced_fields
from ..models.investigation import DriverEdge, DriverGraph

log = logging.getLogger(__name__)

# Edge weights by relation. A KPI inside another's formula is a structural
# driver and is trusted completely; a shared source field is suggestive; a
# shared semantic tag is the weakest signal that still means anything.
WEIGHT_FORMULA = 1.0
WEIGHT_DERIVATION = 0.9
WEIGHT_SHARED_FIELDS = 0.6
WEIGHT_SEMANTIC_TAG = 0.35

# Below this overlap, two KPIs sharing a field is coincidence rather than kinship.
MIN_FIELD_OVERLAP = 0.34

# Verification thresholds for an LLM-proposed edge. Deliberately the same
# correlation bar `contest.causally_consistent` already uses, so a proposed edge
# is held to the standard the rest of the system applies to a causal claim.
MIN_VERIFY_CORRELATION = 0.5
MIN_VERIFY_PERIODS = 6


def _expr_fields(expr: Optional[str]) -> Set[str]:
    """Field names in a seed registry expression such as 'revenue-cost_of_goods'."""
    if not expr:
        return set()
    return {part.strip().lower() for part in re.split(r"[^A-Za-z0-9_]+", expr) if part.strip()}


def _numerator_denominator_fields(spec: Any) -> tuple:
    """
    The two halves of a ratio, whichever kind of spec describes it.

    A contract entry carries parsed ASTs; a seed-registry entry carries plain
    field names. Both are read here so a dataset with no contract — which still
    resolves KPIs through the seed registry — gets the same decomposition.
    """
    num: Set[str] = set()
    den: Set[str] = set()

    if getattr(spec, "numerator_ast", None) is not None:
        num = {f.lower() for f in referenced_fields(spec.numerator_ast)}
    if getattr(spec, "denominator_ast", None) is not None:
        den = {f.lower() for f in referenced_fields(spec.denominator_ast)}

    if not num:
        num = _expr_fields(getattr(spec, "numerator", None)) \
            or _expr_fields(getattr(spec, "numerator_expr", None))
    if not den:
        den = _expr_fields(getattr(spec, "denominator", None))
    return num, den


def _all_fields(spec: Any) -> Set[str]:
    fields: Set[str] = set(getattr(spec, "source_fields", None) or [])
    fields |= set(getattr(spec, "requires", None) or [])       # seed registry
    for attr in ("expression_ast", "numerator_ast", "denominator_ast"):
        node = getattr(spec, attr, None)
        if node is not None:
            try:
                fields |= referenced_fields(node)
            except Exception:                      # pragma: no cover - defensive
                continue
    num, den = set(), set()
    if not fields:
        num, den = _numerator_denominator_fields(spec)
    return {f.lower() for f in (fields | num | den)}


def _spec_map(schema: Any) -> Dict[str, Any]:
    """
    Every KPI definition available for this dataset.

    The contract is authoritative when there is one. Without it the seed
    registry answers, exactly as `metrics.metric_spec` does, so relationships
    are still derived rather than the graph simply coming back empty.
    """
    resolver = getattr(schema, "contract_resolver", None)
    if resolver:
        return dict(resolver)
    from .metrics import METRICS
    available = set(getattr(schema, "available_kpis", None) or [])
    return {k: v for k, v in METRICS.items() if not available or k in available}


def _tags(spec: Any) -> Set[str]:
    definition = getattr(spec, "definition", None)
    return {t.lower() for t in (getattr(definition, "semantic_tags", None) or [])}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def build_driver_graph(schema: Any, outcome_kpi: str,
                       observation: Optional[Dict[str, Any]] = None,
                       dimension_hints: Optional[Sequence[str]] = None) -> DriverGraph:
    """
    The KPIs and dimensions that could account for a movement in `outcome_kpi`.

    Deterministic throughout. Every edge records why it exists, so a hypothesis
    built on one can explain its own grounding rather than asserting a
    relationship the reader has to take on trust.
    """
    resolver = _spec_map(schema)
    outcome = resolver.get(outcome_kpi)
    edges: List[DriverEdge] = []

    if outcome is None:
        return DriverGraph(outcome_kpi=outcome_kpi,
                           note=f"'{outcome_kpi}' has no definition in this dataset's KPI "
                                "contract or the seed registry, so its relationships could "
                                "not be derived.")

    outcome_fields = _all_fields(outcome)
    num_fields, den_fields = _numerator_denominator_fields(outcome)
    outcome_tags = _tags(outcome)
    available = set(getattr(schema, "available_kpis", None) or resolver.keys())

    for key, spec in resolver.items():
        if key == outcome_kpi or key not in available:
            continue
        other_fields = _all_fields(spec)
        if not other_fields:
            continue

        # 1. The other KPI measures exactly what this one's numerator or
        #    denominator is built from. This is the strongest possible driver:
        #    a ratio cannot move unless one of its two halves did.
        if other_fields and other_fields <= num_fields:
            edges.append(DriverEdge(
                source_kpi=key, target_kpi=outcome_kpi, relation="formula_numerator",
                weight=WEIGHT_FORMULA, evidence_fields=sorted(other_fields),
                note=f"'{key}' measures the numerator of '{outcome_kpi}'."))
            continue
        if other_fields and other_fields <= den_fields:
            edges.append(DriverEdge(
                source_kpi=key, target_kpi=outcome_kpi, relation="formula_denominator",
                weight=WEIGHT_FORMULA, evidence_fields=sorted(other_fields),
                note=f"'{key}' measures the denominator of '{outcome_kpi}'."))
            continue
        if other_fields and other_fields <= outcome_fields:
            edges.append(DriverEdge(
                source_kpi=key, target_kpi=outcome_kpi, relation="formula_term",
                weight=WEIGHT_FORMULA, evidence_fields=sorted(other_fields),
                note=f"'{key}' is a term in the formula for '{outcome_kpi}'."))
            continue

        # 2. Discovery recorded that one was derived from the other's inputs.
        if _derivation_related(outcome, spec):
            edges.append(DriverEdge(
                source_kpi=key, target_kpi=outcome_kpi, relation="derivation",
                weight=WEIGHT_DERIVATION,
                evidence_fields=sorted(other_fields & outcome_fields),
                note=f"'{outcome_kpi}' was derived from fields '{key}' also measures."))
            continue

        # 3. Substantial overlap in source fields — related, but not structural.
        overlap = _jaccard(outcome_fields, other_fields)
        if overlap >= MIN_FIELD_OVERLAP:
            shared = sorted(outcome_fields & other_fields)
            edges.append(DriverEdge(
                source_kpi=key, target_kpi=outcome_kpi, relation="shared_source_field",
                weight=round(WEIGHT_SHARED_FIELDS * overlap, 3),
                evidence_fields=shared,
                note=f"Shares the source field(s) {', '.join(shared)} with '{outcome_kpi}'."))
            continue

        # 4. The weakest admissible signal: discovery gave them the same
        #    business meaning even though they read different columns.
        shared_tags = outcome_tags & _tags(spec)
        if shared_tags:
            edges.append(DriverEdge(
                source_kpi=key, target_kpi=outcome_kpi, relation="semantic_tag",
                weight=WEIGHT_SEMANTIC_TAG,
                note=f"Shares the business theme(s) {', '.join(sorted(shared_tags))}."))

    edges.extend(_not_comparable_edges(outcome, outcome_kpi, resolver, available))
    edges.sort(key=lambda e: (-e.weight, e.source_kpi))

    dimension_drivers, relevant = _dimension_drivers(observation, dimension_hints)

    return DriverGraph(
        outcome_kpi=outcome_kpi,
        edges=edges,
        dimension_drivers=dimension_drivers,
        relevant_dimensions=relevant,
        note=("Relationships read from the KPI contract: formula components, recorded "
              "derivations, shared source fields and shared business themes."),
    )


def _derivation_related(outcome: Any, other: Any) -> bool:
    """Whether discovery recorded the outcome as derived from the other's fields."""
    definition = getattr(outcome, "definition", None)
    provenance = getattr(definition, "provenance", None)
    derived_from = {f.lower() for f in (getattr(provenance, "derived_from", None) or [])}
    if not derived_from:
        return False
    return bool(derived_from & _all_fields(other))


def _not_comparable_edges(outcome: Any, outcome_kpi: str, resolver: Dict[str, Any],
                          available: Set[str]) -> List[DriverEdge]:
    """
    Pairs the contract explicitly says must not be compared.

    Recorded with zero weight as a contradiction hint: if a hypothesis leans on
    comparing these two, that is itself evidence against it.
    """
    definition = getattr(outcome, "definition", None)
    out: List[DriverEdge] = []
    for entry in (getattr(definition, "comparability", None) or []):
        other = getattr(entry, "other_kpi_id", None) or getattr(entry, "kpi_id", None)
        comparable = getattr(entry, "comparable", True)
        if not other or comparable or other not in available:
            continue
        reasons = getattr(entry, "reasons", None) or []
        out.append(DriverEdge(
            source_kpi=other, target_kpi=outcome_kpi, relation="not_comparable",
            weight=0.0,
            note="The contract records these as not directly comparable"
                 + (f": {'; '.join(reasons[:2])}" if reasons else ".")))
    return out


def _dimension_drivers(observation: Optional[Dict[str, Any]],
                       hints: Optional[Sequence[str]]) -> tuple:
    """
    Which dimension members actually moved the KPI.

    Reuses `determine_focus`'s test rather than inventing another: a member
    counts only if it carries a large share of the change *and* over-contributes
    relative to its own size, so the "driver" of a decline is never simply
    whichever segment happens to be biggest.
    """
    if not observation:
        return [], list(hints or [])

    from .observe import MIN_FOCUS_CONTRIBUTION_PCT, MIN_FOCUS_OVER_INDEX

    rows: List[Dict[str, Any]] = []
    relevant: List[str] = []
    for dim, members in (observation.get("drivers") or {}).items():
        material = False
        for r in members or []:
            if r.get("is_aggregate"):
                continue
            contribution, over_index = r.get("contribution_pct"), r.get("over_index")
            if contribution is None or contribution < MIN_FOCUS_CONTRIBUTION_PCT:
                continue
            if over_index is not None and over_index < MIN_FOCUS_OVER_INDEX:
                continue
            rows.append({"dimension": dim, "member": r.get("name"),
                         "contribution_pct": contribution, "over_index": over_index,
                         "change_pct": r.get("change_pct")})
            material = True
        if material:
            relevant.append(dim)

    for hint in (hints or []):
        if hint not in relevant:
            relevant.append(hint)

    rows.sort(key=lambda r: -(r.get("contribution_pct") or 0))
    return rows, relevant


# ---------------------------------------------------------------------------
# verification of proposed edges
# ---------------------------------------------------------------------------
def period_change_table(df: pd.DataFrame, metrics: Sequence[str],
                        resolver: Any) -> pd.DataFrame:
    """
    Period-over-period percentage change for several KPIs, one row per period.

    Shaped exactly like `analysis.member_change_table` — a `{metric}__chg`
    column per metric — so `analysis.correlate` and `analysis.counterexamples`
    work on it unchanged. The unit of comparison is a period rather than a
    dimension member, which is what makes it usable for testing whether two
    KPIs move together over time.

    Changes, not levels, deliberately: two series that both trend upward
    correlate almost perfectly at any lag, which is the spurious result this
    check exists to avoid.
    """
    from .observe import quarterly_series

    series = {m: quarterly_series(df, m, resolver) for m in metrics}
    periods = sorted({r["period"] for rows in series.values() for r in rows})

    rows: List[Dict[str, Any]] = []
    by_metric = {m: {r["period"]: r["value"] for r in rows_} for m, rows_ in series.items()}
    for i in range(1, len(periods)):
        prev, cur = periods[i - 1], periods[i]
        row: Dict[str, Any] = {"member": cur}
        for m in metrics:
            a, b = by_metric[m].get(prev), by_metric[m].get(cur)
            if a in (None, 0) or b is None:
                row[f"{m}__chg"] = float("nan")
            else:
                row[f"{m}__chg"] = (b - a) / abs(a) * 100.0
        rows.append(row)
    return pd.DataFrame(rows)


def _lagged_r(table: pd.DataFrame, effect: str, cause: str, lag: int) -> Optional[float]:
    """Correlation with the cause shifted `lag` periods earlier than the effect."""
    import numpy as np

    a = table[f"{effect}__chg"].astype(float)
    b = table[f"{cause}__chg"].astype(float).shift(lag)
    mask = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def verify_edge(edge: DriverEdge, df: pd.DataFrame, schema: Any) -> DriverEdge:
    """
    Test a proposed relationship against the data before it counts for anything.

    A model can propose that staffing cost drives profitability; whether the two
    actually move together in this dataset is a question for the data alone. An
    edge that fails leaves with `verified=False` and `weight=0.0`, which lets it
    inspire a hypothesis while contributing nothing to that hypothesis's score.
    """
    from .analysis import correlate

    resolver = getattr(schema, "contract_resolver", None) or {}
    if edge.source_kpi not in resolver or edge.target_kpi not in resolver:
        edge.verification_note = "One of the two KPIs is not available in this dataset."
        edge.weight = 0.0
        return edge

    try:
        table = period_change_table(df, [edge.target_kpi, edge.source_kpi], resolver)
    except Exception as exc:                       # pragma: no cover - defensive
        edge.verification_note = f"Could not build a comparable series ({exc})."
        edge.weight = 0.0
        return edge

    if len(table) < MIN_VERIFY_PERIODS:
        edge.verification_note = (
            f"Only {len(table)} comparable period-over-period changes are available; at "
            f"least {MIN_VERIFY_PERIODS} are needed before a correlation means anything.")
        edge.weight = 0.0
        return edge

    corr = correlate(table, edge.target_kpi, edge.source_kpi)
    r = corr.get("r")
    edge.correlation = r
    if r is None or abs(float(r)) < MIN_VERIFY_CORRELATION:
        edge.verification_note = (
            f"'{edge.source_kpi}' and '{edge.target_kpi}' do not move together in this "
            f"dataset (r={r if r is not None else 'n/a'}), so the proposed relationship "
            "is not supported by the data.")
        edge.weight = 0.0
        return edge

    # Ordering: does the proposed cause lead the outcome, or trail it? A driver
    # that consistently moves after what it supposedly drives is not a driver.
    same = abs(float(r))
    leads = _lagged_r(table, edge.target_kpi, edge.source_kpi, 1)
    trails = _lagged_r(table, edge.source_kpi, edge.target_kpi, 1)
    edge.lead_lag_periods = 1 if (leads is not None and abs(leads) > same) else 0
    if trails is not None and abs(trails) > same and abs(trails) > abs(leads or 0):
        edge.verification_note = (
            f"'{edge.source_kpi}' moves after '{edge.target_kpi}' rather than before it "
            "(r is strongest with the outcome leading), which is the wrong order for a driver.")
        edge.weight = 0.0
        edge.lead_lag_periods = -1
        return edge

    edge.verified = True
    edge.weight = round(min(0.8, same), 3)
    edge.verification_note = (
        f"Verified against the data: r={edge.correlation} across {len(table)} "
        f"period-over-period changes"
        + (", with the driver moving one period ahead."
           if edge.lead_lag_periods else ", moving in step."))
    return edge


def attach_proposed_edges(graph: DriverGraph, proposed: Sequence[DriverEdge],
                          df: pd.DataFrame, schema: Any) -> DriverGraph:
    """Verify each proposed edge and file it as accepted or unverified."""
    known = {(e.source_kpi, e.target_kpi) for e in graph.edges}
    for edge in proposed:
        if (edge.source_kpi, edge.target_kpi) in known:
            continue
        edge.origin = "llm"
        edge.relation = "llm_proposed"
        checked = verify_edge(edge, df, schema)
        if checked.verified:
            graph.edges.append(checked)
        else:
            graph.unverified_llm_edges.append(checked)
    graph.edges.sort(key=lambda e: (-e.weight, e.source_kpi))
    return graph
