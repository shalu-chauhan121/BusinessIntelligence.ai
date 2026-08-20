"""
STAGE 4 — ACT

Purpose: turn validated findings into clear business actions, written for a
business leader rather than an analyst, and keeping the uncertainty that the
Contest stage found rather than hiding it.

Every recommendation carries: the hypothesis it rests on, the evidence behind
it, the confidence in that evidence, a monitoring threshold computed from the
user's own history, and an explicit statement of what would change the advice.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .analysis import weekly_frame
from .metrics import metric_label, metric_unit, safe

PLAYBOOK: Dict[str, Dict[str, Any]] = {
    "supply": {
        "title": "Protect availability where the shortfall is concentrated",
        "actions": [
            "Prioritise replenishment of {focus} until inventory cover returns to policy level.",
            "Introduce an automated stockout early-warning threshold so availability problems are detected in-week rather than at the period review.",
            "Qualify a second source for the constrained item and hold safety stock against the lead time.",
        ],
        "owner": "Supply chain / operations",
        "horizon": "Immediate (0–4 weeks)",
        "monitor": ["stockout_rate", "fulfillment_rate", "inventory_units"],
    },
    "demand": {
        "title": "Address the demand shortfall directly",
        "actions": [
            "Run a win-back and retention programme against the accounts in {focus} that stopped ordering.",
            "Make win/loss reason capture mandatory so future demand loss can be attributed rather than inferred.",
            "Re-base the forecast for {focus} to the new run-rate rather than the pre-decline plan.",
        ],
        "owner": "Sales / commercial leadership",
        "horizon": "This quarter",
        "monitor": ["orders", "customers"],
    },
    "competitive": {
        "title": "Mount a targeted competitive response",
        "actions": [
            "Build a value-based counter-offer for {focus} rather than an across-the-board discount, which erodes margin everywhere to defend one area.",
            "Stand up weekly share and win/loss tracking in the affected area, and in the areas the competitor has not yet entered.",
            "Pre-brief retention offers for the accounts with renewals falling in the next two quarters.",
        ],
        "owner": "Commercial strategy / pricing",
        "horizon": "This quarter",
        "monitor": ["customers", "avg_selling_price"],
    },
    "pricing": {
        "title": "Re-examine the pricing position",
        "actions": [
            "Stop extending the tactical discount until an elasticity read is available — the data shows realised price fell without a volume recovery.",
            "Quantify margin given away per unit of volume defended, by segment.",
        ],
        "owner": "Pricing committee",
        "horizon": "Immediate (0–4 weeks)",
        "monitor": ["avg_selling_price", "gross_margin_pct"],
    },
    "mix": {
        "title": "Manage the mix, not just the total",
        "actions": [
            "Set mix targets by dimension so a shift toward lower-value activity is visible before it shows up in the headline number.",
            "Review incentives that may be steering volume toward lower-value parts of the portfolio.",
        ],
        "owner": "Commercial finance",
        "horizon": "This quarter",
        "monitor": ["avg_order_value", "gross_margin_pct"],
    },
    "operational": {
        "title": "Remediate the concentrated execution gap",
        "actions": [
            "Run an operational review of {focus} against the parts of the business that held up.",
            "Compare process, coverage and staffing between the affected and unaffected areas.",
        ],
        "owner": "Regional / channel leadership",
        "horizon": "This quarter",
        "monitor": ["orders", "fulfillment_rate"],
    },
    "quality": {
        "title": "Close the quality / service gap",
        "actions": [
            "Run a root-cause review on the returns and support volume increase before it converts into churn.",
            "Track repeat-contact rate for the affected cohort.",
        ],
        "owner": "Quality / customer operations",
        "horizon": "Immediate (0–4 weeks)",
        "monitor": ["return_rate", "tickets_per_1k_orders"],
    },
    "marketing": {
        "title": "Test whether the investment cut is causal",
        "actions": [
            "Run a controlled reinstatement of spend in part of the affected area and hold the rest, so the effect can be measured rather than assumed.",
        ],
        "owner": "Marketing",
        "horizon": "This quarter",
        "monitor": ["marketing_spend", "customer_acquisition_cost"],
    },
    "seasonality": {
        "title": "Adjust expectations rather than operations",
        "actions": [
            "Re-baseline the period target against the seasonal norm before committing to a corrective programme.",
        ],
        "owner": "Finance / planning",
        "horizon": "Next planning cycle",
        "monitor": [],
    },
    "data_quality": {
        "title": "Fix the data before acting on it",
        "actions": [
            "Reconcile the current period extract against source systems before any operational decision is taken on this number.",
        ],
        "owner": "Data engineering",
        "horizon": "Immediate (0–4 weeks)",
        "monitor": [],
    },
}


def monitoring_threshold(df: pd.DataFrame, metric: str) -> Optional[Dict[str, Any]]:
    """A control threshold derived from the KPI's own weekly history."""
    try:
        weeks = weekly_frame(df, metric)
    except Exception:
        return None
    values = weeks["value"].astype(float).dropna().values
    if len(values) < 8:
        return None
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    sigma = 1.4826 * mad or float(np.std(values, ddof=1))
    if not sigma or sigma != sigma:
        return None
    return {
        "metric": metric,
        "label": metric_label(metric),
        "unit": metric_unit(metric),
        "weekly_median": safe(med),
        "robust_sigma": safe(sigma),
        "upper_alert": safe(med + 2 * sigma),
        "lower_alert": safe(med - 2 * sigma),
        "rule": (f"Alert when the weekly {metric_label(metric).lower()} moves outside "
                 f"{med - 2 * sigma:,.2f} – {med + 2 * sigma:,.2f} for two consecutive weeks "
                 f"(median ± 2 robust sigma over {len(values)} weeks of your own history)."),
    }


def _fmt_change(observation: Dict[str, Any]) -> str:
    pct = observation.get("change_pct")
    return f"{pct:+.1f}%" if pct is not None else "n/a"


def build_recommendations(df: pd.DataFrame, observation: Dict[str, Any],
                          contested: Dict[str, Any], focus_label: str) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    for h in contested.get("hypotheses", [])[:3]:
        scoring = h["scoring"]
        if scoring["confidence"] < 20:
            continue
        play = PLAYBOOK.get(h.get("family"), None)
        if not play:
            continue
        target = focus_label or "the affected part of the business"
        priority = ("high" if scoring["confidence"] >= 60 else
                    "medium" if scoring["confidence"] >= 40 else "low")

        supporting = [e["detail"] for e in h.get("evidence", []) if e["stance"] == "supporting"][:3]
        quotes = [f"{d['source']}: \"{d['quote'][:180]}...\"" for d in h.get("documentary_evidence", [])[:2]]
        contra = [d["quote"][:180] for d in h["contest"]["contradictory_evidence"][:1]]

        monitors = [m for m in (monitoring_threshold(df, metric)
                                for metric in play["monitor"]) if m]

        change_mind = []
        temporal = h["contest"]["temporal"]
        if temporal.get("status") == "kpi_precedes_cause":
            change_mind.append(
                "If the KPI's decline is re-dated (for example because the weekly data is revised), the "
                "ordering that limits this explanation would change."
            )
        if h["contest"]["consistency"].get("counterexamples"):
            change_mind.append(
                "If the members that fell without this driver moving are explained by something else, this "
                "explanation would strengthen."
            )
        for m in (h.get("missing") or [])[:2]:
            change_mind.append(f"Obtaining this would sharpen the conclusion: {m}")

        recs.append({
            "id": f"rec_{h['key']}",
            "title": play["title"],
            "based_on": {"hypothesis": h["title"], "key": h["key"],
                         "confidence": scoring["confidence"], "band": scoring["confidence_band"],
                         "status": scoring["status"]},
            "priority": priority,
            "horizon": play["horizon"],
            "owner": play["owner"],
            "actions": [a.format(focus=target) for a in play["actions"]],
            "rationale": (
                f"{h['statement']} Evidence strength for this explanation is {scoring['confidence']}% "
                f"({scoring['confidence_band']}). {scoring['causal_claim']}"
            ),
            "supporting_evidence": supporting,
            "documentary_evidence": quotes,
            "counter_evidence": contra,
            "monitoring": monitors,
            "what_would_change_this": change_mind,
        })
    return recs


def act(df: pd.DataFrame, observation: Dict[str, Any], investigation: Dict[str, Any],
        contested: Dict[str, Any], llm=None) -> Dict[str, Any]:
    focus_label = investigation.get("focus_label", "")
    ranking = contested.get("ranking", [])
    top = contested["hypotheses"][0] if contested.get("hypotheses") else None

    kpi_label = observation["kpi_label"]
    tf = observation["timeframe"]["pretty"]
    base_tf = observation["baseline_timeframe"]["pretty"]
    change = _fmt_change(observation)
    verdict = observation["verdict"]

    q_now = observation["timeframe"].get("quarter")
    q_base = observation["baseline_timeframe"].get("quarter")
    transition = f"Q{q_base}→Q{q_now}" if (q_now and q_base) else "period-on-period"
    significance_sentence = {
        "meaningful_signal": (
            f"This is outside {kpi_label.lower()}'s normal variation: the move is "
            f"{abs(observation['significance'].get('robust_z') or 0):.1f} robust standard deviations away from "
            f"the {transition} change this business normally sees."),
        "within_normal_variation": (
            f"This is within {kpi_label.lower()}'s normal variation for this comparison — the same size of "
            "move has happened before without a specific cause."),
        "statistically_unusual_but_immaterial": (
            "The move is statistically unusual but too small to be commercially material."),
    }.get(verdict, "")

    all_drivers = observation.get("top_drivers", [])
    # A part of the business that moved exactly in line with its own size is arithmetic,
    # not a driver — only name the ones that moved disproportionately.
    drivers = [d for d in all_drivers if d.get("is_disproportionate")][:3] or all_drivers[:2]
    driver_sentence = ""
    if drivers:
        parts = []
        for d in drivers:
            oi = d.get("over_index")
            idx = f", {oi:.1f}x its size" if oi else ""
            parts.append(f"{d['name']} ({d['dimension']}, {d['contribution_pct']:.0f}% of the change{idx})")
        driver_sentence = "The change is concentrated in " + "; ".join(parts) + "."

    if top:
        headline_expl = (
            f"The strongest-evidenced explanation is “{top['title']}” at {top['scoring']['confidence']}% "
            f"evidence-based confidence ({top['scoring']['confidence_band']}). "
            f"{top['scoring']['causal_claim']}"
        )
    else:
        headline_expl = "No hypothesis could be tested against the available data."

    uncertainty = []
    if contested.get("ambiguity_note"):
        uncertainty.append(contested["ambiguity_note"])
    for h in contested.get("hypotheses", [])[:2]:
        if h["contest"]["temporal"].get("status") == "kpi_precedes_cause":
            uncertainty.append(h["contest"]["temporal"]["detail"])
    uncertainty.extend(contested.get("unresolved_questions", [])[:3])

    recommendations = build_recommendations(df, observation, contested, focus_label)

    narrative = {
        "headline": f"{kpi_label} {('fell' if (observation.get('change_pct') or 0) < 0 else 'rose')} "
                    f"{change} in {tf} versus {base_tf}.",
        "what_changed": (
            f"{kpi_label} moved from {observation['baseline_value']:,.0f} in {base_tf} to "
            f"{observation['current_value']:,.0f} in {tf} ({change})."
            if observation.get("baseline_value") is not None else
            f"{kpi_label} was {observation['current_value']:,.0f} in {tf}."
        ),
        "how_significant": significance_sentence,
        "what_drove_it": driver_sentence,
        "leading_explanation": headline_expl,
        "what_we_are_not_sure_about": uncertainty,
    }

    llm_story = None
    if llm is not None and llm.enabled:
        try:
            llm_story = llm.write_story(observation, investigation, contested, recommendations)
        except Exception as exc:
            llm_story = {"error": f"LLM narrative unavailable ({exc}). The deterministic summary is shown instead."}

    return {
        "narrative": narrative,
        "recommendations": recommendations,
        "ranking": ranking,
        "llm_story": llm_story,
        "generated_from": {
            "kpi": observation["kpi"],
            "timeframe": observation["timeframe"]["label"],
            "hypotheses_tested": len(contested.get("hypotheses", [])),
        },
        "limits": [
            "This analysis explains the data that was uploaded. Causes that leave no trace in that data "
            "cannot be found by it.",
            "Confidence scores measure evidence strength, not probability, and do not establish causation.",
            "Recommendations are decision support for a human owner, not automated decisions.",
        ],
    }
