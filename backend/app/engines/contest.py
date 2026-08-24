"""
STAGE 3 — CONTEST

Purpose: actively try to break each hypothesis instead of accepting the first
convincing one. This stage is the project's differentiator, so it is deliberate
about the difference between the two questions:

    INVESTIGATE asks : "could this be the cause?"
    CONTEST asks     : "what would disprove it?"
                       "did it even start before the KPI moved?"
                       "is there a part of the business where the KPI fell
                        without this cause being present?"
                       "is the evidence merely correlated with the change?"

Four adversarial checks run against every hypothesis, all deterministic:

  1. TEMPORAL PRECEDENCE  — date the onset of the KPI and of the proposed cause
     from weekly data. A cause that starts after the KPI moved cannot be the
     whole explanation, however strong its supporting evidence looks.
  2. CROSS-SECTIONAL CONSISTENCY — across dimension members, does the proposed
     cause move with the KPI? Reported as an association, never as causation.
  3. COUNTEREXAMPLE SEARCH — members where the KPI fell but the cause did not move.
  4. CONTRADICTORY RETRIEVAL — pull document passages that argue against the
     hypothesis, not only ones that argue for it.

Confidence is then computed from evidence for, evidence against, and evidence
missing. It is an evidence-strength score, NOT a probability.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from ..rag.retriever import Retriever
from .analysis import (
    compare_onsets,
    correlate,
    counterexamples,
    detect_onset,
    lead_lag,
    member_change_table,
    weekly_frame,
)
from .metrics import DatasetSchema, metric_label

CONTRADICTION_QUERIES = {
    "supply_constraint": [
        "does not explain unaffected no supply issue before the disruption began adequate stock",
        "decline started before allocation began demand weakness earlier",
    ],
    "demand_contraction": [
        "demand held up orders unaffected availability delay could not ship stockout",
        "customers wanted to buy but we could not supply",
    ],
    "competitive_pressure": [
        "no competitor activity share stable not price related unaffected regions",
        "we do not have win loss data competitor claim unverified",
    ],
    "price_and_discounting": [
        "price unchanged discount did not recover volume list price stable",
    ],
    "mix_shift": ["mix unchanged composition stable same product split"],
    "channel_execution": ["channel performing well partner unaffected coverage stable"],
    "service_quality": ["quality stable no defects complaints unchanged satisfaction stable"],
    "marketing_pullback": ["marketing spend follows revenue variable budget effect not cause"],
    "seasonality_calendar": ["not seasonal unusual for this quarter unprecedented"],
    "data_artefact": ["data complete reconciled no system change extract validated"],
}

# A passage only counts as counter-evidence if it actually argues against something.
# Strong markers are explicit denials, limitations or orderings; weak markers are
# generic contrast words that need corroboration before they mean anything.
STRONG_MARKERS = (
    "does not explain", "not explain", "does not", "did not", "was not", "were not",
    "cannot", "is not", "no supply", "not a sufficient", "insufficient", "unaffected",
    "unchanged", "not the same", "unverified", "before any", "already", "not settle",
    "no carrier", "not implemented", "did not recover", "not available", "largely unpopulated",
    "caveat", "not sufficient", "rather than",
)
WEAK_MARKERS = ("however", "but ", "although", "contrary", "gap", "risk", "open question")
NEGATION_MARKERS = STRONG_MARKERS + WEAK_MARKERS


# ---------------------------------------------------------------------------
def _scoped(df: pd.DataFrame, focus: Dict[str, str]) -> pd.DataFrame:
    out = df
    for dim, member in focus.items():
        if dim in out.columns:
            out = out[out[dim] == member]
    return out if len(out) else df


def temporal_check(df: pd.DataFrame, kpi: str, hypothesis: Dict[str, Any],
                   focus: Dict[str, str], window: pd.DataFrame) -> Dict[str, Any]:
    cause_metric = hypothesis.get("cause_metric")
    if not cause_metric:
        return {
            "status": "not_applicable",
            "detail": "This hypothesis has no single measurable driver series to date against the KPI.",
            "lag_weeks": None, "cause_metric": None,
        }
    scoped = _scoped(window, focus)
    kpi_weeks = weekly_frame(scoped, kpi)
    cause_weeks = weekly_frame(scoped, cause_metric)
    baseline_weeks = max(6, min(10, len(kpi_weeks) // 3))
    kpi_dir = "down" if (hypothesis.get("kpi_direction") or "down") == "down" else "up"
    kpi_onset = detect_onset(kpi_weeks, baseline_weeks=baseline_weeks, direction=kpi_dir)
    cause_onset = detect_onset(cause_weeks, baseline_weeks=baseline_weeks,
                               direction=hypothesis.get("cause_direction", "down"))
    result = compare_onsets(kpi_onset, cause_onset)
    result["lead_lag"] = lead_lag(kpi_weeks, cause_weeks)
    result["cause_metric"] = cause_metric
    result["cause_metric_label"] = metric_label(cause_metric)
    result["scope"] = " / ".join(focus.values()) if focus else "whole business"
    result["kpi_series"] = kpi_weeks.to_dict("records")
    result["cause_series"] = cause_weeks.to_dict("records")
    return result


def consistency_check(cur: pd.DataFrame, base: pd.DataFrame, schema: DatasetSchema,
                      kpi: str, hypothesis: Dict[str, Any]) -> Dict[str, Any]:
    cause_metric = hypothesis.get("cause_metric")
    dim = next((d for d in ("region", "product", "channel", "segment") if d in schema.dimensions), None)
    if not dim or not cause_metric:
        return {"status": "not_applicable",
                "detail": "No dimension or no single driver series available for a cross-sectional test."}
    table = member_change_table(cur, base, dim, [kpi, cause_metric])
    corr = correlate(table, kpi, cause_metric)
    counter = counterexamples(
        table, kpi, cause_metric,
        kpi_drop_pct=-5.0, cause_move_pct=5.0,
        cause_direction=hypothesis.get("cause_direction", "down"),
    )
    return {
        "status": "checked",
        "dimension": dim,
        "cause_metric": cause_metric,
        "cause_metric_label": metric_label(cause_metric),
        "correlation": corr,
        "counterexamples": counter,
        "members": [
            {
                "member": r["member"],
                "kpi_change_pct": None if pd.isna(r.get(f"{kpi}__chg")) else round(float(r[f"{kpi}__chg"]), 2),
                "cause_change_pct": None if pd.isna(r.get(f"{cause_metric}__chg")) else round(float(r[f"{cause_metric}__chg"]), 2),
            }
            for _, r in table.iterrows()
        ],
        "detail": (
            f"{len(counter)} of {len(table)} '{dim}' members show the KPI falling without the proposed "
            f"driver moving." if counter else
            f"No '{dim}' member shows the KPI falling while the proposed driver stayed put."
        ),
    }


def contradictory_retrieval(hypothesis: Dict[str, Any], retriever: Retriever,
                            terms: List[str], llm=None) -> List[Dict[str, Any]]:
    """Retrieve passages that argue AGAINST the hypothesis."""
    if not retriever.available:
        return []
    supporting_ids = [e.get("chunk_id") for e in hypothesis.get("documentary_evidence", [])]
    queries = CONTRADICTION_QUERIES.get(hypothesis["key"], [])
    found: List[Dict[str, Any]] = []
    seen = set()
    for q in queries:
        for item in retriever.retrieve_evidence(" ".join([q] + terms), top_k=4,
                                                stance="contradicting", min_score=3.5):
            if item["chunk_id"] in seen or item["chunk_id"] in supporting_ids:
                continue
            seen.add(item["chunk_id"])
            blob = item["quote"].lower()
            strong = [m.strip() for m in STRONG_MARKERS if m in blob]
            weak = [m.strip() for m in WEAK_MARKERS if m in blob]
            item["contrast_markers"] = (strong + weak)[:4]
            item["_strong_hits"] = len(strong)
            item["_weak_hits"] = len(weak)
            item["classified_by"] = "lexical_contrast_heuristic"
            found.append(item)

    # Only keep passages that actually read as counter-evidence.
    filtered = [
        f for f in found
        if f["_strong_hits"] >= 1 or (f["_weak_hits"] >= 2 and f.get("strength", 0) >= 0.5)
    ]
    for f in filtered:
        f.pop("_strong_hits", None)
        f.pop("_weak_hits", None)

    if llm is not None and llm.enabled and filtered:
        try:
            verdicts = llm.classify_stance(hypothesis, filtered)
            keep = []
            for item in filtered:
                v = verdicts.get(item["chunk_id"]) or {}
                stance = v.get("stance")
                if stance in ("contradicting", "supporting", "neutral"):
                    item["stance"] = stance
                    item["classified_by"] = "llm_stance_classifier"
                    item["classification_reason"] = v.get("reason", "")
                if item["stance"] == "contradicting":
                    keep.append(item)
            filtered = keep
        except Exception:
            pass                                        # keep the heuristic result
    return filtered[:4]


def mechanism_check(hypothesis: Dict[str, Any], temporal: Dict[str, Any]) -> Dict[str, Any]:
    """
    Screen for reverse causation and mechanical dependency.

    Some "causes" are arithmetically downstream of the KPI being explained
    (marketing budgeted as a share of revenue; realised price computed from
    revenue; tickets per order rising because orders fell). Such a hypothesis
    correlates beautifully and explains nothing. This check states the risk
    explicitly and tests the timing that would distinguish the two directions.
    """
    declared = hypothesis.get("reverse_causation_risk")
    ll = temporal.get("lead_lag")
    verdict = (ll or {}).get("verdict")

    suspected = bool(declared) and verdict in ("kpi_leads", "simultaneous", None)
    if declared and verdict == "cause_leads":
        conclusion = (
            "A reverse-causation risk was flagged for this explanation, but the timing test does not support "
            "it: the proposed cause moves first. The hypothesis survives this check."
        )
    elif suspected and verdict == "kpi_leads":
        conclusion = (
            "Reverse causation is likely. The declared risk is confirmed by the timing: the KPI moves first."
        )
    elif suspected:
        conclusion = (
            "Reverse causation cannot be ruled out. The two quantities may be mechanically linked, and the "
            "timing test does not separate the directions."
        )
    else:
        conclusion = "No mechanical dependency on the KPI was declared for this explanation."

    return {
        "declared_risk": declared,
        "lead_lag": ll,
        "reverse_causation_suspected": suspected,
        "conclusion": conclusion,
    }


# ---------------------------------------------------------------------------
def score_hypothesis(hypothesis: Dict[str, Any], temporal: Dict[str, Any],
                     consistency: Dict[str, Any], contra_docs: List[Dict[str, Any]],
                     mechanism: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Evidence-based confidence. NOT a probability.

        confidence = support / (support + against + missing_penalty + prior)

    where support and against are weighted sums of measured evidence strengths.
    The `prior` term (a constant) keeps a hypothesis with only one weak piece of
    evidence from scoring highly just because nothing contradicts it yet.
    """
    ledger: List[Dict[str, Any]] = []
    support = against = 0.0

    for e in hypothesis.get("evidence", []):
        contribution = e.get("strength", 0) * e.get("weight", 1)
        if e["stance"] == "supporting":
            support += contribution
            ledger.append({"side": "for", "source": "structured", "label": e["label"], "value": round(contribution, 3)})
        elif e["stance"] == "contradicting":
            against += contribution
            ledger.append({"side": "against", "source": "structured", "label": e["label"], "value": round(contribution, 3)})

    for d in hypothesis.get("documentary_evidence", []):
        contribution = d.get("strength", d.get("relevance", 0)) * 0.9
        support += contribution
        ledger.append({"side": "for", "source": "document", "label": d["source"], "value": round(contribution, 3)})

    for d in contra_docs:
        contribution = d.get("strength", d.get("relevance", 0)) * 1.0
        against += contribution
        ledger.append({"side": "against", "source": "document", "label": d["source"], "value": round(contribution, 3)})

    temporal_status = temporal.get("status")
    if temporal_status == "cause_precedes_kpi":
        support += 1.6
        ledger.append({"side": "for", "source": "temporal", "label": "Cause precedes the KPI move", "value": 1.6})
    elif temporal_status == "kpi_precedes_cause":
        against += 2.4
        ledger.append({"side": "against", "source": "temporal", "label": "KPI moved before the proposed cause", "value": 2.4})
    elif temporal_status == "simultaneous":
        ledger.append({"side": "neutral", "source": "temporal", "label": "Simultaneous onset", "value": 0})

    corr = (consistency.get("correlation") or {}).get("r")
    if corr is not None:
        if abs(corr) >= 0.5:
            support += 1.1 * abs(corr)
            ledger.append({"side": "for", "source": "cross_section",
                           "label": f"Consistent across {consistency.get('dimension')} members (r={corr})",
                           "value": round(1.1 * abs(corr), 3)})
        else:
            against += 0.8
            ledger.append({"side": "against", "source": "cross_section",
                           "label": f"Inconsistent across {consistency.get('dimension')} members (r={corr})",
                           "value": 0.8})

    n_counter = len(consistency.get("counterexamples") or [])
    if n_counter:
        penalty = min(2.0, 0.6 * n_counter)
        against += penalty
        ledger.append({"side": "against", "source": "counterexample",
                       "label": f"{n_counter} member(s) fell without the proposed cause", "value": round(penalty, 3)})

    mechanism = mechanism or {}
    if mechanism.get("reverse_causation_suspected"):
        against += 1.8
        ledger.append({"side": "against", "source": "mechanism",
                       "label": "Reverse causation / mechanical dependency suspected", "value": 1.8})
    elif (mechanism.get("lead_lag") or {}).get("verdict") == "cause_leads":
        support += 0.8
        ledger.append({"side": "for", "source": "mechanism",
                       "label": "Weekly movements of the cause lead the KPI", "value": 0.8})

    missing = hypothesis.get("missing", [])
    missing_penalty = min(1.5, 0.3 * len(missing))

    raw = support / (support + against + missing_penalty + 1.2) if (support + against) > 0 else 0.0
    confidence = round(raw * 100)

    capped_reason = None
    if temporal_status == "kpi_precedes_cause" and confidence > 55:
        confidence, capped_reason = 55, (
            "Capped: the KPI began moving before this cause did, so it cannot be the whole explanation "
            "however strong its supporting evidence is."
        )
    if mechanism.get("reverse_causation_suspected") and confidence > 45:
        confidence = 45
        capped_reason = (capped_reason or "") + (
            " Capped: this explanation may be mechanically downstream of the KPI it is meant to explain, "
            "so a strong association between them is expected even with no causal role."
        ).strip()
    if support == 0:
        confidence = min(confidence, 10)

    if confidence >= 65:
        band, status = "strong", "well_supported"
    elif confidence >= 45:
        band, status = "moderate", "partially_supported"
    elif confidence >= 25:
        band, status = "weak", "weakly_supported"
    else:
        band, status = "insufficient", "not_supported"

    causal_ok = (
        temporal_status in ("cause_precedes_kpi", "simultaneous")
        and corr is not None and abs(corr) >= 0.5
        and n_counter == 0
        and confidence >= 45
        and not mechanism.get("reverse_causation_suspected")
    )
    causal_claim = (
        "Evidence is consistent with a causal contribution (temporal ordering holds, the pattern is "
        "consistent across the business, and no counterexample was found). This is still not proof of causation."
        if causal_ok else
        "Association only. The evidence links this explanation to the change but does not establish that it "
        "caused it."
    )

    return {
        "confidence": confidence,
        "confidence_band": band,
        "confidence_label": "Evidence-based confidence (not a probability)",
        "status": status,
        "support_score": round(support, 3),
        "against_score": round(against, 3),
        "missing_penalty": round(missing_penalty, 3),
        "score_ledger": ledger,
        "cap_reason": capped_reason,
        "causal_claim": causal_claim,
        "causally_consistent": bool(causal_ok),
    }


def contest(df: pd.DataFrame, schema: DatasetSchema, observation: Dict[str, Any],
            investigation: Dict[str, Any], uid: str, llm=None) -> Dict[str, Any]:
    from .observe import Timeframe, slice_period

    tf = Timeframe(observation["timeframe"]["year"], observation["timeframe"]["quarter"])
    base_tf = Timeframe(observation["baseline_timeframe"]["year"],
                        observation["baseline_timeframe"]["quarter"])
    cur, base = slice_period(df, tf), slice_period(df, base_tf)

    lo = base["_date"].min() if len(base) else df["_date"].min()
    hi = cur["_date"].max()
    window = df[(df["_date"] >= lo) & (df["_date"] <= hi)]

    kpi = observation["kpi"]
    focus = investigation.get("focus", {})
    retriever = Retriever(uid)
    terms = list(focus.values())[:3]

    kpi_direction = "down" if (observation.get("change_pct") or 0) < 0 else "up"

    contested: List[Dict[str, Any]] = []
    for h in investigation.get("hypotheses", []):
        h = dict(h)
        h["kpi_direction"] = kpi_direction
        temporal = temporal_check(df, kpi, h, focus, window)
        consistency = consistency_check(cur, base, schema, kpi, h)
        contra_docs = contradictory_retrieval(h, retriever, terms, llm=llm)
        mechanism = mechanism_check(h, temporal)
        scoring = score_hypothesis(h, temporal, consistency, contra_docs, mechanism)

        trail = [
            {"step": "Hypothesis proposed",
             "detail": h["statement"]},
            {"step": "Structured tests run",
             "detail": f"{len(h.get('evidence', []))} metric test(s) computed from the uploaded dataset over "
                       f"{observation['timeframe']['pretty']} versus {observation['baseline_timeframe']['pretty']}."},
            {"step": "Documentary evidence retrieved",
             "detail": f"{len(h.get('documentary_evidence', []))} passage(s) retrieved from uploaded documents."},
            {"step": "Temporal precedence checked", "detail": temporal.get("detail", "")},
            {"step": "Cross-sectional consistency checked",
             "detail": (consistency.get("correlation") or {}).get("interpretation") or consistency.get("detail", "")},
            {"step": "Counterexample search",
             "detail": consistency.get("detail", "Not applicable.")},
            {"step": "Reverse-causation screen", "detail": mechanism["conclusion"]},
            {"step": "Lead/lag timing test",
             "detail": (mechanism.get("lead_lag") or {}).get("detail",
                        "Not enough weekly history to run a lead/lag test.")},
            {"step": "Contradictory evidence search",
             "detail": (f"{len(contra_docs)} passage(s) retrieved that argue against this explanation."
                        if contra_docs else "No contradicting passage found in the uploaded documents.")},
            {"step": "Confidence computed",
             "detail": (f"Support {scoring['support_score']} vs against {scoring['against_score']} "
                        f"(missing-evidence penalty {scoring['missing_penalty']}) → "
                        f"{scoring['confidence']}% {scoring['confidence_band']}.")},
        ]
        if scoring.get("cap_reason"):
            trail.append({"step": "Confidence capped", "detail": scoring["cap_reason"]})

        contested.append({
            **h,
            "contest": {
                "temporal": {k: v for k, v in temporal.items() if k not in ("kpi_series", "cause_series")},
                "temporal_series": {"kpi": temporal.get("kpi_series", []), "cause": temporal.get("cause_series", [])},
                "consistency": consistency,
                "mechanism": mechanism,
                "contradictory_evidence": contra_docs,
            },
            "scoring": scoring,
            "reasoning_trail": trail,
        })

    contested.sort(key=lambda h: h["scoring"]["confidence"], reverse=True)

    top = contested[0] if contested else None
    second = contested[1] if len(contested) > 1 else None
    ambiguity = None
    if top and second and (top["scoring"]["confidence"] - second["scoring"]["confidence"]) <= 12:
        ambiguity = (
            f"The evidence does not separate '{top['title']}' from '{second['title']}' "
            f"({top['scoring']['confidence']}% vs {second['scoring']['confidence']}%). Treat both as live "
            "explanations rather than picking the higher one."
        )

    unresolved: List[str] = []
    for h in contested[:3]:
        for m in (h.get("missing") or [])[:2]:
            if m not in unresolved:
                unresolved.append(m)

    clarification_request = None
    if unresolved and (not top or top["scoring"]["confidence"] < 45 or ambiguity):
        reason = ambiguity or (
            "No explanation has sufficient evidence to support a recommendation."
        )
        clarification_request = {
            "status": "clarification_required",
            "reason": reason,
            "missing_evidence": unresolved,
            "questions": [f"Can you provide: {item}" for item in unresolved],
        }

    return {
        "ranking": [
            {
                "rank": i + 1,
                "key": h["key"],
                "title": h["title"],
                "confidence": h["scoring"]["confidence"],
                "band": h["scoring"]["confidence_band"],
                "status": h["scoring"]["status"],
                "causally_consistent": h["scoring"]["causally_consistent"],
            }
            for i, h in enumerate(contested)
        ],
        "hypotheses": contested,
        "ambiguity_note": ambiguity,
        "unresolved_questions": unresolved,
        "clarification_request": clarification_request,
        "confidence_disclaimer": (
            "Confidence scores are evidence-strength scores computed from how much measured evidence "
            "supports each explanation, how much contradicts it, and how much is missing. They are not "
            "probabilities and they do not establish causation."
        ),
    }
