"""
STAGE 2 — INVESTIGATE

Purpose: generate COMPETING explanations for the change Observe found, and test
each one against real evidence — structured (the user's rows) and unstructured
(the user's documents) — rather than producing one explanation that merely sounds
convincing.

Division of labour, per the project's design philosophy:
  * the data-analysis layer computes every number,
  * RAG retrieves every quotation,
  * the LLM (optional) proposes framings and writes prose over those facts.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from ..rag.retriever import Retriever
from .hypotheses import Context, build_candidates
from .metrics import DatasetSchema

MAX_HYPOTHESES = 5


MIN_CONTRIBUTION_PCT = 25.0
MIN_OVER_INDEX = 1.2


def determine_focus(observation: Dict[str, Any]) -> Dict[str, str]:
    """
    The dimension members genuinely responsible for the change.

    A member only qualifies if it both (a) accounts for a large share of the
    change and (b) over-contributes relative to its own size. Without the second
    test the "driver" of any decline is simply whichever segment is biggest,
    which tells a business nothing.
    """
    scored = []
    for dim, rows in (observation.get("drivers") or {}).items():
        for r in rows:
            if r.get("is_aggregate"):
                continue
            contribution, over_index = r.get("contribution_pct"), r.get("over_index")
            if contribution is None or contribution < MIN_CONTRIBUTION_PCT:
                continue
            if over_index is not None and over_index < MIN_OVER_INDEX:
                continue
            scored.append((over_index or 1.0, contribution, dim, r["name"]))
    scored.sort(reverse=True)

    focus: Dict[str, str] = {}
    for _, _, dim, name in scored:
        focus.setdefault(dim, name)
    return dict(list(focus.items())[:2])       # at most two dimensions keeps slices meaningful


def _rag_terms(ctx: Context) -> List[str]:
    """Entity terms that make retrieval specific to the drivers Observe found."""
    return [v for v in ctx.focus.values()][:3]


def attach_documentary_evidence(hypothesis: Dict[str, Any], retriever: Retriever,
                                terms: List[str]) -> Dict[str, Any]:
    hits: List[Dict[str, Any]] = []
    seen = set()
    for query in hypothesis.get("rag_queries", []):
        scoped_query = " ".join([query] + terms)
        for item in retriever.retrieve_evidence(scoped_query, top_k=3, stance="supporting"):
            if item["chunk_id"] in seen:
                continue
            seen.add(item["chunk_id"])
            hits.append(item)
    hits.sort(key=lambda h: h.get("strength", h["relevance"]), reverse=True)
    hypothesis["documentary_evidence"] = hits[:4]
    if not retriever.available:
        hypothesis.setdefault("missing", []).append(
            "No business documents have been uploaded, so this hypothesis is tested against "
            "structured data only. Upload operations reports, customer feedback or market notes "
            "to let the system corroborate or contradict it with written evidence."
        )
    elif not hits:
        hypothesis.setdefault("missing", []).append(
            "Nothing in the uploaded documents refers to this explanation."
        )
    return hypothesis


def investigate(df: pd.DataFrame, schema: DatasetSchema, observation: Dict[str, Any],
                uid: str, llm=None) -> Dict[str, Any]:
    from .observe import Timeframe, slice_period

    tf = Timeframe(observation["timeframe"]["year"], observation["timeframe"]["quarter"])
    base_tf = Timeframe(observation["baseline_timeframe"]["year"],
                        observation["baseline_timeframe"]["quarter"])

    ctx = Context(
        df=df, schema=schema, metric=observation["kpi"],
        cur=slice_period(df, tf), base=slice_period(df, base_tf),
        observation=observation, focus=determine_focus(observation),
    )

    if observation.get("history_status") != "sufficient_history":
        return {
            "focus": {}, "focus_label": "", "hypotheses": [], "considered_count": 0,
            "not_carried_forward": [], "documents_indexed": 0, "rag_available": False,
            "llm_used": False, "llm_note": None,
            "method_note": observation.get("history_note") + " No causal hypotheses or confidence scores were generated.",
        }

    candidates = build_candidates(ctx)
    retriever = Retriever(uid)
    terms = _rag_terms(ctx)

    shortlist = [h for h in candidates if h.get("testable")][:MAX_HYPOTHESES]
    for h in shortlist:
        attach_documentary_evidence(h, retriever, terms)

    llm_note = None
    if llm is not None and llm.enabled:
        try:
            enriched = llm.frame_hypotheses(observation, shortlist, ctx.focus)
            by_key = {e.get("key"): e for e in enriched.get("hypotheses", [])}
            for h in shortlist:
                patch = by_key.get(h["key"])
                if patch:
                    h["statement"] = patch.get("statement") or h["statement"]
                    h["title"] = patch.get("title") or h["title"]
                    h["analyst_note"] = patch.get("analyst_note", "")
                    extra_q = patch.get("additional_evidence_to_seek") or []
                    if extra_q:
                        h.setdefault("missing", []).extend(
                            [q for q in extra_q if q not in h.get("missing", [])][:2]
                        )
            llm_note = enriched.get("framing_note")
        except Exception as exc:                        # never let the LLM break the pipeline
            llm_note = f"LLM framing unavailable ({exc}); deterministic hypothesis statements used."

    rejected = [
        {"key": h["key"], "title": h["title"],
         "reason": "Not carried forward: weaker prior evidence than the shortlisted explanations."}
        for h in candidates if h not in shortlist and h.get("testable")
    ]

    return {
        "focus": ctx.focus,
        "focus_label": ctx.focus_label(),
        "hypotheses": shortlist,
        "considered_count": len(candidates),
        "not_carried_forward": rejected,
        "documents_indexed": len(retriever.chunks),
        "rag_available": retriever.available,
        "llm_used": bool(llm is not None and llm.enabled),
        "llm_note": llm_note,
        "method_note": (
            "Hypotheses are generated from a library of business explanations, filtered to those the "
            "uploaded dataset can actually test. Every number attached to a hypothesis is computed by the "
            "data-analysis layer; every quotation is retrieved verbatim from an uploaded document."
        ),
    }
