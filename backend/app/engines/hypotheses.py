"""
Competing explanations, and the machinery that measures them.

A hypothesis here is never an assertion. It is a set of *predictions* — this
metric should have moved this way, if the mechanism I am proposing is what
happened — and `evidence()` below measures each one against the user's rows and
decides for itself whether the prediction held. A hypothesis that predicted a
rise in a metric that fell leaves with contradicting evidence attached to it,
regardless of how plausible its wording was.

That property is what makes it safe to let a language model propose mechanisms.
It supplies the business reasoning; it never supplies a number, a direction that
was observed, or a verdict. Two sources feed the same measuring machinery:

  * `formula_decomposition_candidates` — deterministic, read from the KPI
    Contract's own formula, always available, needs no model;
  * `llm_hypotheses.generate` — mechanisms specific to the detected business
    domain, proposed by a model and validated before they get here.

Every hypothesis, from either source, declares:
  * `cause_metric`            — the series CONTEST dates against the KPI to check
                                temporal precedence and search counterexamples;
  * `rag_queries`             — what to look for in the user's documents;
  * `contradiction_queries`   — what would show the hypothesis to be wrong;
  * `missing`                 — evidence needed but absent from the dataset.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from .metrics import (DatasetSchema, Resolver, compute, metric_label, metric_unit,
                      pct_change, safe)


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

    @property
    def resolver(self) -> Optional[Resolver]:
        """The compiled KPI Contract for this dataset, if it has one."""
        return self.schema.contract_resolver

    # -- availability ------------------------------------------------------
    def has(self, *metrics: str) -> bool:
        return all(m in self.schema.available_kpis for m in metrics)

    # -- measurement -------------------------------------------------------
    def value(self, metric: str, scope: Optional[Dict[str, str]] = None):
        cur, base = self.scoped(scope)
        return compute(cur, metric, self.resolver), compute(base, metric, self.resolver)

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
    unit = metric_unit(metric, ctx.resolver)
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
        "label": metric_label(metric, ctx.resolver),
        "scope": scope_label,
        "baseline": safe(base_v),
        "current": safe(cur_v),
        "change_abs": safe(cur_v - base_v),
        "change_pct": safe(chg),
        "unit": unit,
        "strength": round(strength_of(chg, reference), 3),
        "weight": weight,
        "detail": (f"{metric_label(metric, ctx.resolver)} in {scope_label}: "
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
# contract-derived decomposition
# ---------------------------------------------------------------------------
# The library of retail explanations that used to live here has been removed. It
# could not do the job asked of it: a template written around orders, stockouts
# and discounting fires on any dataset that has a dimension column, so a
# hospital's declining margin was explained as a competitor taking volume.
#
# What replaces it comes in two halves. This half is deterministic and reads the
# KPI Contract's own formula: a ratio cannot move unless its numerator or its
# denominator moved, and that is true of every business in every industry. It
# needs no language model and is always available. The other half — mechanisms
# specific to how this kind of business actually operates — is proposed by a
# model in `llm_hypotheses.py` and tested here by exactly the same machinery.
MAX_DECOMPOSITION_HYPOTHESES = 3


def _direction_of(ctx: Context) -> str:
    return "down" if (ctx.kpi_change_pct or 0) < 0 else "up"


def _component_metrics(ctx: Context, graph: Any) -> Dict[str, List[str]]:
    """The KPIs that make up the outcome, grouped by the part they play."""
    groups: Dict[str, List[str]] = {"numerator": [], "denominator": [], "term": []}
    for edge in (getattr(graph, "edges", None) or []):
        if edge.relation == "formula_numerator":
            groups["numerator"].append(edge.source_kpi)
        elif edge.relation == "formula_denominator":
            groups["denominator"].append(edge.source_kpi)
        elif edge.relation == "formula_term":
            groups["term"].append(edge.source_kpi)
    return groups


def formula_decomposition_candidates(ctx: Context, graph: Any) -> List[Dict[str, Any]]:
    """
    Hypotheses read directly off the outcome KPI's formula.

    These are arithmetic rather than commercial: they say which half of a ratio
    moved, not why it moved. That is deliberately modest, and it is the floor the
    product stands on when no language model is available — a real, testable,
    contract-grounded explanation rather than nothing at all.
    """
    groups = _component_metrics(ctx, graph)
    outcome_dir = _direction_of(ctx)
    outcome_label = metric_label(ctx.metric, ctx.resolver)
    out: List[Dict[str, Any]] = []

    # The two halves of a ratio move the result in opposite directions.
    pairs = [
        ("numerator", outcome_dir,
         "rose faster than" if outcome_dir == "up" else "fell faster than"),
        ("denominator", "down" if outcome_dir == "up" else "up",
         "shrank against" if outcome_dir == "up" else "grew against"),
    ]

    for part, expected, phrase in pairs:
        for metric in groups.get(part, [])[:2]:
            if not ctx.has(metric):
                continue
            label = metric_label(metric, ctx.resolver)
            other = "denominator" if part == "numerator" else "numerator"
            other_metrics = [m for m in groups.get(other, []) if ctx.has(m)]
            ev = _compact([
                evidence(ctx, metric, "supporting", expect=expected, weight=1.4,
                         reference=8.0),
                evidence(ctx, other_metrics[0], "neutral", weight=0.7)
                if other_metrics else None,
            ])
            if not ev:
                continue
            other_label = (", ".join(metric_label(m, ctx.resolver) for m in other_metrics)
                           or "its other inputs")
            out.append({
                "key": f"component_{part}_{metric}",
                "title": f"{label} moved the {part}",
                "family": "other",
                "statement": (
                    f"{label} {phrase} the other half of the calculation, which is "
                    f"arithmetically sufficient to move {outcome_label} in the direction "
                    f"observed. This identifies which component moved, not why it moved."),
                "mechanism": (f"{outcome_label} is computed from {label} and {other_label}."),
                "evidence": ev,
                "cause_metric": metric,
                "cause_direction": expected,
                "rag_queries": [f"{label} change reason explanation"],
                "contradiction_queries": [
                    f"{label} stable unchanged no change",
                    f"{outcome_label} driven by something other than {label}",
                ],
                "missing": [
                    f"The dataset shows that {label} moved, but contains nothing explaining "
                    "why it moved — that needs operational context this data does not carry."],
                "domain_specific": False,
                "source": "contract_formula",
            })

    # An additive KPI has no two halves to play against each other; each term is
    # a driver in its own right, and the one that moved most is the account.
    for metric in groups.get("term", [])[:3]:
        if not ctx.has(metric):
            continue
        label = metric_label(metric, ctx.resolver)
        item = evidence(ctx, metric, "supporting", expect=outcome_dir, weight=1.3,
                        reference=8.0)
        if not item:
            continue
        out.append({
            "key": f"component_term_{metric}",
            "title": f"{label} moved with it",
            "family": "other",
            "statement": (
                f"{label} is one of the components {outcome_label} is built from, and it moved "
                f"in the same direction. That is arithmetically part of the movement, though it "
                f"does not by itself say why {label.lower()} moved."),
            "mechanism": f"{outcome_label} is computed from {label} among its inputs.",
            "evidence": _compact([item]),
            "cause_metric": metric,
            "cause_direction": outcome_dir,
            "rag_queries": [f"{label} change reason"],
            "contradiction_queries": [f"{label} unchanged stable",
                                      f"{outcome_label} moved without {label}"],
            "missing": [f"Why {label.lower()} itself moved is not recorded in this dataset."],
            "domain_specific": False,
            "source": "contract_formula",
        })

    return out[:MAX_DECOMPOSITION_HYPOTHESES]


MAX_RELATED_HYPOTHESES = 3

# Relations that mean "these two measure overlapping things" rather than "one is
# arithmetically part of the other". Weaker, but still contract-derived.
_ASSOCIATION_RELATIONS = ("derivation", "shared_source_field", "semantic_tag")


def related_driver_candidates(ctx: Context, graph: Any) -> List[Dict[str, Any]]:
    """
    Explanations built from measures the contract says are related to the outcome.

    Weaker than a formula decomposition and honest about it: sharing source
    fields makes two KPIs related, not one the cause of the other. Each of these
    is a falsifiable prediction — if this measure drove the movement, it should
    have moved the same way — and the data decides. A related measure that moved
    the other way ends up as evidence against its own hypothesis.
    """
    outcome_dir = _direction_of(ctx)
    outcome_label = metric_label(ctx.metric, ctx.resolver)
    structural = {e.source_kpi for e in (getattr(graph, "edges", None) or [])
                  if e.relation.startswith("formula_")}
    out: List[Dict[str, Any]] = []

    for edge in (getattr(graph, "edges", None) or []):
        if len(out) >= MAX_RELATED_HYPOTHESES:
            break
        if edge.relation not in _ASSOCIATION_RELATIONS:
            continue
        metric = edge.source_kpi
        # A measure already covered by the formula decomposition needs no second,
        # weaker hypothesis saying the same thing.
        if metric in structural or metric == ctx.metric or not ctx.has(metric):
            continue

        item = evidence(ctx, metric, "supporting", expect=outcome_dir, weight=1.0,
                        reference=10.0)
        if not item:
            continue
        # Only worth proposing when the measure actually moved. A related metric
        # sitting still explains nothing and would only pad the list.
        if abs(item.get("change_pct") or 0) < 2.0 and item["stance"] != "supporting":
            continue

        label = metric_label(metric, ctx.resolver)
        out.append({
            "key": f"related_{metric}",
            "title": f"{label} moved with {outcome_label.lower()}",
            "family": "other",
            "statement": (
                f"{label} is built from data that overlaps with {outcome_label.lower()}, and it "
                f"moved in the same period. If it is part of the account, it should have moved "
                f"in the same direction — which is what this test checks. Shared inputs make the "
                f"two related; they do not by themselves make one the cause of the other."),
            "mechanism": edge.note or f"{label} and {outcome_label} share source data.",
            "evidence": _compact([item]),
            "cause_metric": metric,
            "cause_direction": outcome_dir,
            "rag_queries": [f"{label} change reason {outcome_label}"],
            "contradiction_queries": [
                f"{label} unchanged while {outcome_label} moved",
                f"{outcome_label} explained by something other than {label}",
            ],
            "missing": [
                f"The two share source data, so their movements are not independent. Nothing in "
                f"this dataset separates {label.lower()} driving {outcome_label.lower()} from "
                f"both responding to the same underlying cause."],
            "domain_specific": False,
            "source": "contract_association",
        })

    return out


def concentration_candidate(ctx: Context) -> Optional[Dict[str, Any]]:
    """
    The movement is concentrated in one part of the business.

    Domain-neutral and available to any dataset with a dimension, so it is the
    floor beneath the floor: even a KPI with no decomposable formula and no
    model available still gets one real, testable explanation. It reuses the
    over-index test, so a segment qualifies only by moving more than its own
    size implies — not merely by being large.
    """
    if not ctx.focus:
        return None
    dimension, member = next(iter(ctx.focus.items()))
    scope = {dimension: member}
    outcome_dir = _direction_of(ctx)
    outcome_label = metric_label(ctx.metric, ctx.resolver)

    items = _compact([
        evidence(ctx, ctx.metric, "supporting", scope=scope, expect=outcome_dir,
                 weight=1.2, reference=8.0),
        evidence(ctx, ctx.metric, "neutral", weight=0.6),
    ])
    if not items:
        return None

    concentration = ctx.observation.get("driver_concentration_pct")
    if concentration:
        items.append(custom_evidence(
            "Concentration of the change",
            f"The largest single contributor accounts for {concentration:.0f}% of the total "
            f"movement. A change this concentrated points at something specific to that part "
            f"of the business rather than at a condition affecting all of it.",
            "supporting" if concentration >= 45 else "neutral",
            strength=min(1.0, concentration / 70.0), weight=0.9,
        ))

    return {
        "key": f"concentrated_in_{dimension}_{member}".lower().replace(" ", "_"),
        "title": f"The change is concentrated in {member}",
        "family": "operational",
        "statement": (
            f"{outcome_label} moved further in {member} than that {dimension}'s share of the "
            f"business would imply, so the movement is localised rather than general. What is "
            f"specific to {member} is not something this dataset records."),
        "mechanism": (f"A condition affecting {member} specifically, rather than the whole "
                      f"business, would produce exactly this pattern."),
        "evidence": items,
        "cause_metric": None,
        "cause_direction": outcome_dir,
        "rag_queries": [f"{member} {dimension} issue change incident"],
        "contradiction_queries": [f"{member} performing normally no change",
                                  "change affected all areas equally company-wide"],
        "missing": [f"Nothing in this dataset explains what is different about {member}; "
                    f"that needs operational context from the people who run it."],
        "domain_specific": False,
        "source": "dimension_concentration",
    }


# ---------------------------------------------------------------------------
# assembling candidates
# ---------------------------------------------------------------------------
def score_priors(built: Dict[str, Any]) -> Dict[str, Any]:
    """
    Attach the prior weights the shortlist is ordered by.

    Unchanged in spirit from the template era: support and opposition are summed
    from the evidence items themselves, so a hypothesis whose predictions the
    data contradicted sorts below one whose predictions held.
    """
    ev = built.get("evidence") or []
    support = sum((e.get("strength", 0) * e.get("weight", 1))
                  for e in ev if e.get("stance") == "supporting")
    against = sum((e.get("strength", 0) * e.get("weight", 1))
                  for e in ev if e.get("stance") == "contradicting")
    built["prior_support"] = round(float(support), 3)
    built["prior_against"] = round(float(against), 3)
    built.setdefault("testable", True)
    built.setdefault("family", "other")
    built.setdefault("missing", [])
    built.setdefault("rag_queries", [])
    built.setdefault("contradiction_queries", [])
    built.setdefault("domain_specific", False)
    return built


def build_candidates(ctx: Context, graph: Any = None,
                     llm_candidates: Optional[List[Dict[str, Any]]] = None
                     ) -> List[Dict[str, Any]]:
    """
    Every hypothesis worth testing for this observation.

    Two sources, measured the same way: the deterministic decomposition of the
    outcome KPI's own formula, and whatever mechanisms a language model proposed
    for this business. Neither is trusted on assertion — both arrive as
    predictions and leave carrying evidence whose stance the data decided.
    """
    out: List[Dict[str, Any]] = []

    if graph is not None:
        try:
            out.extend(formula_decomposition_candidates(ctx, graph))
            out.extend(related_driver_candidates(ctx, graph))
        except Exception as exc:               # a broken probe must never kill the run
            out.append({
                "key": "component_decomposition", "title": "Formula decomposition",
                "family": "other", "statement": "Could not be tested against this dataset.",
                "evidence": [], "rag_queries": [], "missing": [f"Probe failed: {exc}"],
                "prior_support": 0.0, "prior_against": 0.0, "testable": False,
            })

    # Available to any dataset with a dimension, including KPIs whose formula
    # has nothing to decompose. Without it, a base metric with no components
    # would produce no hypotheses at all when no model is available.
    try:
        concentrated = concentration_candidate(ctx)
        if concentrated:
            out.append(concentrated)
    except Exception:                          # pragma: no cover - defensive
        pass

    out.extend(llm_candidates or [])

    seen: set = set()
    deduped: List[Dict[str, Any]] = []
    for h in out:
        h = score_priors(h)
        signature = (h.get("cause_metric"), h.get("family"), h.get("cause_direction"))
        # Two hypotheses claiming the same driver moving the same way within the
        # same family are the same hypothesis, however differently they are worded.
        if h.get("cause_metric") and signature in seen:
            continue
        seen.add(signature)
        deduped.append(h)

    deduped.sort(key=lambda h: h["prior_support"] - 0.5 * h["prior_against"], reverse=True)
    return deduped
