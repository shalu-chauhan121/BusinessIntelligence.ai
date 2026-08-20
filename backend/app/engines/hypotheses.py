"""
The competing-hypothesis library.

Each template asks one business question and answers it with *numbers computed
from the user's rows*. A template only enters the investigation when the dataset
actually contains the columns needed to test it — the system never proposes an
explanation it cannot examine.

Every template also declares:
  * `cause_metric`  — the series CONTEST will date against the KPI to check
                      temporal precedence, and use for counterexample search;
  * `rag_queries`   — what to look for in the user's documents;
  * `missing`       — evidence that would be needed but is not in the dataset.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from .analysis import price_volume_decomposition
from .metrics import DatasetSchema, compute, metric_label, metric_unit, pct_change, safe


# ---------------------------------------------------------------------------
# investigation context
# ---------------------------------------------------------------------------
@dataclass
class Context:
    df: pd.DataFrame
    schema: DatasetSchema
    metric: str
    cur: pd.DataFrame
    base: pd.DataFrame
    observation: Dict[str, Any]
    focus: Dict[str, str] = field(default_factory=dict)

    # -- availability ------------------------------------------------------
    def has(self, *metrics: str) -> bool:
        return all(m in self.schema.available_kpis for m in metrics)

    # -- measurement -------------------------------------------------------
    def value(self, metric: str, scope: Optional[Dict[str, str]] = None):
        cur, base = self.scoped(scope)
        return compute(cur, metric), compute(base, metric)

    def scoped(self, scope: Optional[Dict[str, str]] = None):
        cur, base = self.cur, self.base
        if scope:
            for dim, member in scope.items():
                if dim in cur.columns:
                    cur = cur[cur[dim] == member]
                    base = base[base[dim] == member]
        return cur, base

    @property
    def kpi_change_pct(self) -> float:
        return self.observation.get("change_pct") or 0.0

    def focus_scope(self, dimension: str) -> Optional[Dict[str, str]]:
        member = self.focus.get(dimension)
        return {dimension: member} if member else None

    def focus_label(self) -> str:
        parts = [f"{v}" for v in self.focus.values()]
        return " / ".join(parts) if parts else "the business"


# ---------------------------------------------------------------------------
# evidence helpers
# ---------------------------------------------------------------------------
def _fmt(value: Optional[float], unit: str) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "n/a"
    if unit == "currency":
        return f"{value:,.0f}"
    if unit == "percent":
        return f"{value:.1f}%"
    if unit == "ratio":
        return f"{value:,.2f}"
    return f"{value:,.0f}"


def strength_of(change_pct: Optional[float], reference: float = 12.0) -> float:
    if change_pct is None or change_pct != change_pct:
        return 0.0
    return float(min(1.0, abs(change_pct) / reference))


def evidence(ctx: Context, metric: str, stance: str, scope: Optional[Dict[str, str]] = None,
             note: str = "", weight: float = 1.0, reference: float = 12.0,
             expect: str = "any") -> Optional[Dict[str, Any]]:
    """
    Build one structured-evidence item by measuring `metric` over the current
    and baseline periods (optionally within a dimension slice).

    `expect` states the direction the hypothesis predicts. If the metric moves
    the other way, the item is automatically re-labelled as contradicting — a
    hypothesis is not allowed to claim evidence that points against it.
    """
    cur_v, base_v = ctx.value(metric, scope)
    if cur_v != cur_v or base_v != base_v:
        return None
    chg = pct_change(cur_v, base_v)
    unit = metric_unit(metric)
    scope_label = " / ".join(f"{v}" for v in scope.values()) if scope else "whole business"

    actual = "up" if (cur_v - base_v) > 0 else ("down" if (cur_v - base_v) < 0 else "flat")
    resolved_stance = stance
    if expect in ("up", "down"):
        if actual == "flat" or (abs(chg) < 2.0 if chg == chg else True):
            resolved_stance = "contradicting" if stance == "supporting" else "neutral"
            note = note or f"Predicted to move {expect}, but it is essentially unchanged."
        elif actual != expect:
            resolved_stance = "contradicting" if stance == "supporting" else "supporting"
            note = note or f"Predicted to move {expect}, but it moved {actual}."

    delta_txt = f"{chg:+.1f}%" if chg == chg else "n/a"
    if unit == "percent":
        delta_txt = f"{(cur_v - base_v):+.1f} pts ({delta_txt})"

    return {
        "type": "structured",
        "stance": resolved_stance,
        "metric": metric,
        "label": metric_label(metric),
        "scope": scope_label,
        "baseline": safe(base_v),
        "current": safe(cur_v),
        "change_abs": safe(cur_v - base_v),
        "change_pct": safe(chg),
        "unit": unit,
        "strength": round(strength_of(chg, reference), 3),
        "weight": weight,
        "detail": (f"{metric_label(metric)} in {scope_label}: "
                   f"{_fmt(base_v, unit)} → {_fmt(cur_v, unit)} ({delta_txt})."),
        "note": note,
        "source": "Structured data analysis (uploaded dataset)",
    }


def custom_evidence(label: str, detail: str, stance: str, strength: float,
                    weight: float = 1.0, metric: Optional[str] = None,
                    extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    item = {
        "type": "structured",
        "stance": stance,
        "metric": metric,
        "label": label,
        "scope": "whole business",
        "detail": detail,
        "strength": round(float(min(1.0, max(0.0, strength))), 3),
        "weight": weight,
        "note": "",
        "source": "Structured data analysis (uploaded dataset)",
    }
    if extra:
        item.update(extra)
    return item


def _compact(items: List[Optional[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    return [i for i in items if i]


# ---------------------------------------------------------------------------
# hypothesis templates
# ---------------------------------------------------------------------------
@dataclass
class Template:
    key: str
    title: str
    family: str
    applies: Callable[[Context], bool]
    build: Callable[[Context], Dict[str, Any]]


def _region_scope(ctx: Context) -> Optional[Dict[str, str]]:
    for dim in ("region", "product", "channel", "segment"):
        sc = ctx.focus_scope(dim)
        if sc:
            return sc
    return None


# --- 1. supply / availability ---------------------------------------------
def _supply(ctx: Context) -> Dict[str, Any]:
    scope = _region_scope(ctx)
    ev = _compact([
        evidence(ctx, "fulfillment_rate", "supporting", expect="down", weight=1.3, reference=6.0)
        if ctx.has("fulfillment_rate") else None,
        evidence(ctx, "stockout_rate", "supporting", expect="up", weight=1.2, reference=40.0)
        if ctx.has("stockout_rate") else None,
        evidence(ctx, "inventory_units", "supporting", expect="down", weight=1.0, reference=20.0)
        if ctx.has("inventory_units") else None,
        evidence(ctx, "stockout_rate", "supporting", scope=scope, expect="up", weight=0.8, reference=40.0)
        if (ctx.has("stockout_rate") and scope) else None,
        evidence(ctx, "avg_selling_price", "neutral", expect="any", weight=0.4)
        if ctx.has("avg_selling_price") else None,
    ])
    missing = []
    if not ctx.has("fulfillment_rate"):
        missing.append("Order fulfilment rate (needs `fulfilled_orders` and `orders` columns).")
    if not ctx.has("inventory_units"):
        missing.append("Inventory position (needs an `inventory_units` column).")
    missing.append("Supplier lead-time and on-time-in-full data is not present in the uploaded dataset.")
    return {
        "key": "supply_constraint",
        "title": "Supply constraint limited what could be sold",
        "family": "supply",
        "statement": (
            f"{ctx.observation['kpi_label']} fell because the business could not fulfil the demand it had — "
            "availability, not demand, was the binding constraint."
        ),
        "evidence": ev,
        "cause_metric": "stockout_rate" if ctx.has("stockout_rate") else (
            "fulfillment_rate" if ctx.has("fulfillment_rate") else None),
        "cause_direction": "up" if ctx.has("stockout_rate") else "down",
        "rag_queries": [
            "supply disruption stockout inventory shortage allocation supplier incident",
            "fulfilment delay backorder lead time shipment delayed",
        ],
        "missing": missing,
    }


# --- 2. demand contraction --------------------------------------------------
def _demand(ctx: Context) -> Dict[str, Any]:
    scope = _region_scope(ctx)
    ev = _compact([
        evidence(ctx, "orders", "supporting", expect="down", weight=1.2, reference=10.0)
        if ctx.has("orders") else None,
        evidence(ctx, "customers", "supporting", expect="down", weight=1.2, reference=10.0)
        if ctx.has("customers") else None,
        evidence(ctx, "orders", "supporting", scope=scope, expect="down", weight=0.9, reference=10.0)
        if (ctx.has("orders") and scope) else None,
        evidence(ctx, "avg_order_value", "neutral", weight=0.5)
        if ctx.has("avg_order_value") else None,
        evidence(ctx, "fulfillment_rate", "neutral", weight=0.6,
                 note="If fulfilment held up while orders fell, the shortfall is demand-side, not supply-side.")
        if ctx.has("fulfillment_rate") else None,
    ])
    return {
        "key": "demand_contraction",
        "title": "Underlying customer demand contracted",
        "family": "demand",
        "statement": (
            "Fewer customers placed fewer orders. The business sold less because less was asked for, "
            "not because it could not supply."
        ),
        "evidence": ev,
        "cause_metric": "orders" if ctx.has("orders") else ("customers" if ctx.has("customers") else None),
        "cause_direction": "down",
        "rag_queries": [
            "demand weakened slowdown fewer orders pipeline deferred budget freeze procurement",
            "customer churn non-renewal lost account cancelled contract",
        ],
        "missing": [
            "Enquiry / lead / web-traffic volume is not in the dataset, so upper-funnel demand cannot be observed directly.",
            "Win-loss reasons are not in the dataset, so lost demand cannot be attributed to a cause.",
        ],
    }


# --- 3. competitive pressure ------------------------------------------------
def _competition(ctx: Context) -> Dict[str, Any]:
    scope = _region_scope(ctx)
    ev = _compact([
        evidence(ctx, "customers", "supporting", scope=scope, expect="down", weight=1.1, reference=10.0)
        if (ctx.has("customers") and scope) else None,
        evidence(ctx, "avg_selling_price", "supporting", scope=scope, expect="down", weight=1.0, reference=4.0)
        if (ctx.has("avg_selling_price") and scope) else None,
        evidence(ctx, "avg_selling_price", "neutral", weight=0.6)
        if ctx.has("avg_selling_price") else None,
    ])
    conc = ctx.observation.get("driver_concentration_pct")
    if conc:
        ev.append(custom_evidence(
            "Geographic concentration of the change",
            f"The single largest driver accounts for {conc:.0f}% of the total change. A change concentrated "
            "in one part of the business is more consistent with a localised competitive event than with a "
            "broad market shift.",
            "supporting" if conc >= 45 else "neutral",
            strength=min(1.0, conc / 70.0), weight=0.9,
        ))
    return {
        "key": "competitive_pressure",
        "title": "Competitive pressure eroded share",
        "family": "competitive",
        "statement": (
            "A competitor took volume — the decline is concentrated where a competitor is active, and "
            "realised price moved without recovering volume."
        ),
        "evidence": ev,
        "cause_metric": "customers" if ctx.has("customers") else None,
        "cause_direction": "down",
        "rag_queries": [
            "competitor campaign discount aggressive pricing market share switch rival",
            "lost to competitor better commercial terms re-tender renewal",
        ],
        "missing": [
            "Market share and competitor pricing are not in the uploaded dataset; any competitive claim rests "
            "on documents rather than on measured share.",
            "Win-loss reason codes are not available in the structured data.",
        ],
    }


# --- 4. price / discounting -------------------------------------------------
def _pricing(ctx: Context) -> Dict[str, Any]:
    ev = _compact([
        evidence(ctx, "avg_selling_price", "supporting", expect="down", weight=1.2, reference=5.0)
        if ctx.has("avg_selling_price") else None,
        evidence(ctx, "gross_margin_pct", "supporting", expect="down", weight=1.0, reference=4.0)
        if ctx.has("gross_margin_pct") else None,
        evidence(ctx, "units_sold", "neutral", weight=0.7) if ctx.has("units_sold") else None,
    ])
    pv = price_volume_decomposition(ctx.cur, ctx.base)
    if pv:
        share = pv.get("price_share_pct")
        ev.append(custom_evidence(
            "Price vs volume decomposition",
            (f"Of the total revenue change, {abs(share or 0):.0f}% is attributable to realised price per unit "
             f"and {abs(pv.get('volume_share_pct') or 0):.0f}% to unit volume. {pv['narrative']}"),
            "supporting" if (share or 0) > 40 else "contradicting",
            strength=min(1.0, abs(share or 0) / 60.0), weight=1.3,
            extra={"decomposition": pv},
        ))
    return {
        "key": "price_and_discounting",
        "title": "Realised price / discounting moved the KPI",
        "family": "pricing",
        "statement": "The change is driven by what was charged per unit rather than by how many units moved.",
        "evidence": ev,
        "cause_metric": "avg_selling_price" if ctx.has("avg_selling_price") else None,
        "cause_direction": "down",
        "reverse_causation_risk": (
            "Realised price per unit is computed from the same revenue figure being explained, so part of any "
            "association between them is arithmetic rather than behavioural."
        ),
        "rag_queries": ["pricing discount promotion price cut list price rebate margin decision"],
        "missing": ["Contracted / negotiated price terms are not in the dataset, only realised revenue per unit."],
    }


# --- 5. mix shift -----------------------------------------------------------
def _mix(ctx: Context) -> Dict[str, Any]:
    ev: List[Dict[str, Any]] = []
    for dim, rows in (ctx.observation.get("drivers") or {}).items():
        shifts = []
        for r in rows:
            if r.get("is_aggregate"):
                continue
            eff = r.get("effects") or {}
            if eff.get("mix_effect") is not None:
                shifts.append((r["name"], eff["mix_effect"], eff.get("rate_effect")))
        if shifts:
            worst = max(shifts, key=lambda t: abs(t[1] or 0))
            ev.append(custom_evidence(
                f"Mix effect within {dim}",
                (f"'{worst[0]}' contributes a mix effect of {worst[1]:+.2f} versus a rate effect of "
                 f"{(worst[2] or 0):+.2f}. A mix effect means the KPI moved because the composition of the "
                 "business changed, not because any individual part got worse."),
                "supporting" if abs(worst[1] or 0) > abs(worst[2] or 0) else "contradicting",
                strength=0.6, weight=1.0,
            ))
    for dim in ctx.schema.dimensions[:2]:
        if dim not in ctx.cur.columns or "revenue" not in ctx.cur.columns:
            continue
        cur_share = ctx.cur.groupby(dim)["revenue"].sum()
        base_share = ctx.base.groupby(dim)["revenue"].sum()
        if len(cur_share) and len(base_share) and cur_share.sum() and base_share.sum():
            cs, bs = cur_share / cur_share.sum() * 100, base_share / base_share.sum() * 100
            diff = (cs - bs).dropna()
            if len(diff):
                mover = diff.abs().idxmax()
                ev.append(custom_evidence(
                    f"Composition shift in {dim}",
                    (f"'{mover}' moved from {bs.get(mover, 0):.1f}% to {cs.get(mover, 0):.1f}% of revenue "
                     f"({diff[mover]:+.1f} pts of mix)."),
                    "supporting" if abs(diff[mover]) >= 3 else "neutral",
                    strength=min(1.0, abs(diff[mover]) / 8.0), weight=0.8,
                ))
    return {
        "key": "mix_shift",
        "title": "Business mix shifted toward lower-value activity",
        "family": "mix",
        "statement": ("Individual parts of the business did not deteriorate much; the blend of what was sold, "
                      "to whom or through which channel changed."),
        "evidence": ev,
        "cause_metric": None,
        "cause_direction": "down",
        "rag_queries": ["product mix segment mix channel shift portfolio change assortment"],
        "missing": ["Planned mix / budget by dimension is not in the dataset, so actual-vs-plan mix cannot be tested."],
    }


# --- 6. channel underperformance -------------------------------------------
def _channel(ctx: Context) -> Optional[Dict[str, Any]]:
    dim = next((d for d in ("channel", "store", "sales_rep") if d in ctx.schema.dimensions), None)
    if not dim:
        return None
    rows = [r for r in (ctx.observation.get("drivers") or {}).get(dim, []) if not r.get("is_aggregate")]
    if not rows:
        return None
    worst = max(rows, key=lambda r: abs(r.get("contribution_pct") or 0))
    ev = [custom_evidence(
        f"{dim.title()} concentration",
        (f"'{worst['name']}' accounts for {worst.get('contribution_pct', 0):.0f}% of the total change "
         f"({worst.get('change_pct', 0):+.1f}% within that {dim})."),
        "supporting" if abs(worst.get("contribution_pct") or 0) >= 50 else "neutral",
        strength=min(1.0, abs(worst.get("contribution_pct") or 0) / 70.0), weight=1.0,
    )]
    scope = {dim: worst["name"]}
    for m in ("orders", "customers", "avg_order_value"):
        if ctx.has(m):
            item = evidence(ctx, m, "neutral", scope=scope, weight=0.7)
            if item:
                ev.append(item)
    return {
        "key": "channel_execution",
        "title": f"Execution problem concentrated in one {dim}",
        "family": "operational",
        "statement": f"The change is concentrated in a single {dim} rather than being a market-wide movement.",
        "evidence": ev,
        "cause_metric": None,
        "cause_direction": "down",
        "rag_queries": [f"{dim} performance partner distributor coverage execution issue"],
        "missing": [f"Activity data per {dim} (visits, campaigns, headcount) is not in the dataset."],
    }


# --- 7. customer experience / quality ---------------------------------------
def _quality(ctx: Context) -> Optional[Dict[str, Any]]:
    if not (ctx.has("return_rate") or ctx.has("tickets_per_1k_orders")):
        return None
    ev = _compact([
        evidence(ctx, "return_rate", "supporting", expect="up", weight=1.0, reference=20.0)
        if ctx.has("return_rate") else None,
        evidence(ctx, "tickets_per_1k_orders", "supporting", expect="up", weight=1.0, reference=20.0)
        if ctx.has("tickets_per_1k_orders") else None,
    ])
    return {
        "key": "service_quality",
        "title": "Product or service quality damaged demand",
        "family": "quality",
        "statement": "Customers experienced worse quality or service and bought less as a result.",
        "evidence": ev,
        "cause_metric": "return_rate" if ctx.has("return_rate") else "tickets_per_1k_orders",
        "cause_direction": "up",
        "reverse_causation_risk": (
            "Service load per order rises mechanically when order volume falls, so an increase can be a "
            "consequence of the decline rather than a cause of it."
        ),
        "rag_queries": ["quality defect complaint return faulty service issue escalation satisfaction"],
        "missing": ["NPS / CSAT scores are not in the structured dataset."],
    }


# --- 8. marketing pullback ---------------------------------------------------
def _marketing(ctx: Context) -> Optional[Dict[str, Any]]:
    if not ctx.has("marketing_spend"):
        return None
    ev = _compact([
        evidence(ctx, "marketing_spend", "supporting", expect="down", weight=1.1, reference=15.0),
        evidence(ctx, "customer_acquisition_cost", "neutral", weight=0.8)
        if ctx.has("customer_acquisition_cost") else None,
    ])
    return {
        "key": "marketing_pullback",
        "title": "Reduced marketing investment reduced demand",
        "family": "marketing",
        "statement": "Lower marketing investment fed through to lower customer acquisition and volume.",
        "evidence": ev,
        "cause_metric": "marketing_spend",
        "cause_direction": "down",
        "reverse_causation_risk": (
            "Marketing spend is frequently budgeted as a percentage of revenue. In that case spend FALLS "
            "BECAUSE revenue fell, so a tight association between the two is expected even when marketing "
            "had no causal role at all."
        ),
        "rag_queries": ["marketing budget campaign spend reduction media plan acquisition"],
        "missing": [
            "Marketing spend in this dataset moves with revenue, so it may be an EFFECT of lower volume "
            "(variable spend) rather than a cause. Campaign-level and timing data would be needed to separate them.",
        ],
    }


# --- 9. seasonality / calendar ----------------------------------------------
def _seasonality(ctx: Context) -> Dict[str, Any]:
    sig = ctx.observation.get("significance", {})
    same_q = sig.get("same_quarter_points", 0)
    med = sig.get("median_historical_change_pct")
    z = sig.get("robust_z")
    ev = []
    if med is not None:
        ev.append(custom_evidence(
            "Seasonal norm for this quarter transition",
            (f"In prior years this same quarter transition moved the KPI by a median of {med:+.1f}% "
             f"(from {same_q} comparable transition(s)). The current period moved "
             f"{ctx.kpi_change_pct:+.1f}%."),
            "supporting" if (z is not None and abs(z) < 2) else "contradicting",
            strength=0.9 if (z is not None and abs(z) < 2) else min(1.0, abs(z or 0) / 4.0),
            weight=1.4,
        ))
    ev.append(custom_evidence(
        "Comparison-window equality",
        (f"The current period contains {ctx.cur['_date'].nunique()} distinct dates versus "
         f"{ctx.base['_date'].nunique()} in the baseline."),
        "supporting" if abs(ctx.cur["_date"].nunique() - ctx.base["_date"].nunique()) > 2 else "contradicting",
        strength=0.5, weight=0.8,
    ))
    return {
        "key": "seasonality_calendar",
        "title": "The movement is seasonal or calendar-driven, not a business problem",
        "family": "seasonality",
        "statement": "The change is what this KPI normally does at this point in the year.",
        "evidence": ev,
        "cause_metric": None,
        "cause_direction": "down",
        "rag_queries": ["seasonal seasonality holiday quarter-end phasing calendar"],
        "missing": ["Trading-day calendars and promotional calendars are not in the dataset."],
    }


# --- 10. data quality artefact ----------------------------------------------
def _data_quality(ctx: Context) -> Dict[str, Any]:
    from .analysis import coverage_report

    cov = coverage_report(ctx.cur, ctx.base, ctx.schema.dimensions)
    ev = [custom_evidence(
        "Row and date coverage",
        (f"{cov['rows_current']:,} rows over {cov['days_current']} dates in the current period versus "
         f"{cov['rows_baseline']:,} rows over {cov['days_baseline']} dates in the baseline "
         f"({cov['row_change_pct']:+.1f}% rows)." if cov["row_change_pct"] is not None else
         f"{cov['rows_current']:,} rows in the current period versus {cov['rows_baseline']:,} in the baseline."),
        "supporting" if cov["issues"] else "contradicting",
        strength=0.85 if cov["issues"] else 0.15, weight=1.2,
        extra={"coverage": cov},
    )]
    for issue in cov["issues"][:3]:
        ev.append(custom_evidence("Coverage issue", issue, "supporting", strength=0.8, weight=1.0))
    return {
        "key": "data_artefact",
        "title": "Part of the change is a data artefact, not business performance",
        "family": "data_quality",
        "statement": "Missing rows, missing dimension members or an unequal comparison window explain part of the move.",
        "evidence": ev,
        "cause_metric": None,
        "cause_direction": "down",
        "rag_queries": ["data quality migration reporting change restatement system cutover"],
        "missing": ["Source-system extract logs are not available to confirm completeness."],
    }


TEMPLATES: List[Template] = [
    Template("supply_constraint", "Supply constraint", "supply",
             lambda c: c.has("fulfillment_rate") or c.has("stockout_rate") or c.has("inventory_units"), _supply),
    Template("demand_contraction", "Demand contraction", "demand",
             lambda c: c.has("orders") or c.has("customers"), _demand),
    Template("competitive_pressure", "Competitive pressure", "competitive",
             lambda c: bool(c.schema.dimensions), _competition),
    Template("price_and_discounting", "Price / discounting", "pricing",
             lambda c: c.has("avg_selling_price") or c.has("gross_margin_pct"), _pricing),
    Template("mix_shift", "Mix shift", "mix", lambda c: len(c.schema.dimensions) >= 1, _mix),
    Template("channel_execution", "Channel execution", "operational",
             lambda c: any(d in c.schema.dimensions for d in ("channel", "store", "sales_rep")), _channel),
    Template("service_quality", "Service quality", "quality",
             lambda c: c.has("return_rate") or c.has("tickets_per_1k_orders"), _quality),
    Template("marketing_pullback", "Marketing pullback", "marketing",
             lambda c: c.has("marketing_spend"), _marketing),
    Template("seasonality_calendar", "Seasonality", "seasonality", lambda c: True, _seasonality),
    Template("data_artefact", "Data artefact", "data_quality", lambda c: True, _data_quality),
]


def build_candidates(ctx: Context) -> List[Dict[str, Any]]:
    """Run every applicable template and return candidate hypotheses with evidence."""
    out: List[Dict[str, Any]] = []
    for tpl in TEMPLATES:
        try:
            if not tpl.applies(ctx):
                continue
            built = tpl.build(ctx)
            if not built:
                continue
            ev = built.get("evidence") or []
            support = sum((e.get("strength", 0) * e.get("weight", 1)) for e in ev if e["stance"] == "supporting")
            against = sum((e.get("strength", 0) * e.get("weight", 1)) for e in ev if e["stance"] == "contradicting")
            built["prior_support"] = round(float(support), 3)
            built["prior_against"] = round(float(against), 3)
            built["testable"] = True
            out.append(built)
        except Exception as exc:                       # a broken probe must never kill the run
            out.append({
                "key": tpl.key, "title": tpl.title, "family": tpl.family,
                "statement": "Could not be tested against this dataset.",
                "evidence": [], "rag_queries": [], "missing": [f"Probe failed: {exc}"],
                "prior_support": 0.0, "prior_against": 0.0, "testable": False,
            })
    out.sort(key=lambda h: h["prior_support"] - 0.5 * h["prior_against"], reverse=True)
    return out
