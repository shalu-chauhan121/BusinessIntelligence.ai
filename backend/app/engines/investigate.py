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
    """
    Search the user's documents both for and against this hypothesis.

    Searching only for support is how an investigation talks itself into a
    conclusion, so the disconfirmation pass is not optional: every hypothesis
    carries `contradiction_queries` describing what would show it to be wrong,
    and those are run with the same weight as the supporting ones.
    """
    hits: List[Dict[str, Any]] = []
    seen = set()

    def search(queries: List[str], stance: str) -> None:
        for query in queries or []:
            scoped_query = " ".join([query] + terms)
            for item in retriever.retrieve_evidence(scoped_query, top_k=3, stance=stance):
                if item["chunk_id"] in seen:
                    continue
                seen.add(item["chunk_id"])
                item.setdefault("stance", stance)
                hits.append(item)

    search(hypothesis.get("rag_queries", []), "supporting")
    search(hypothesis.get("contradiction_queries", []), "contradicting")

    hits.sort(key=lambda h: h.get("strength", h["relevance"]), reverse=True)
    hypothesis["documentary_evidence"] = hits[:6]
    if not retriever.available:
        hypothesis.setdefault("missing", []).append(
            "No business documents have been uploaded, so this hypothesis is tested against "
            "structured data only. Upload operations reports, customer feedback or market notes "
            "to let the system corroborate or contradict it with written evidence."
        )
    elif not hits:
        hypothesis.setdefault("missing", []).append(
            "Nothing in the uploaded documents refers to this explanation, either to support "
            "it or to rule it out."
        )
    return hypothesis


def _kpi_semantics(schema: DatasetSchema, metric: str) -> Dict[str, Any]:
    """What the contract knows about the KPI, for the generation prompt."""
    definition = schema.kpi_definition(metric) if hasattr(schema, "kpi_definition") else None
    spec = (schema.contract_resolver or {}).get(metric)
    formula = getattr(definition, "formula", None)
    return {
        "key": metric,
        "label": getattr(definition, "name", "") or getattr(spec, "label", "") or metric,
        "business_definition": getattr(definition, "business_definition", "") or "",
        "why_it_matters": getattr(definition, "relevance", "") or "",
        "semantic_tags": list(getattr(definition, "semantic_tags", None) or []),
        "unit": getattr(spec, "unit", ""),
        "higher_is_better": getattr(spec, "higher_is_better", None),
        "formula": getattr(formula, "expression", "") or "",
        "numerator": getattr(formula, "numerator_expression", "") or "",
        "denominator": getattr(formula, "denominator_expression", "") or "",
        "granularity": getattr(spec, "granularity_label", ""),
        "source_fields": list(getattr(spec, "source_fields", None) or []),
    }


def _domain_facts(schema: DatasetSchema) -> Dict[str, Any]:
    """
    The kind of business this is, in the words its explanations should use.

    Without this the generation prompt has never been told it is looking at a
    hospital, which is the whole reason explanations used to come back reading
    like they were written for a retailer.
    """
    domain = getattr(schema, "domain", None)
    vocab = getattr(domain, "vocab", None)
    if vocab is None:
        return {"domain": "uncertain",
                "note": "The industry could not be determined from the available fields; "
                        "reason from the dataset's own measures rather than assuming one."}
    facts = {
        "domain": getattr(domain, "domain", "uncertain"),
        "description": vocab.label,
        "serves": vocab.entity,
        "core_activity": vocab.activity,
        "unit_of_work": vocab.unit_of_work,
        "capacity_means": vocab.capacity_note,
        "confidence": getattr(domain, "confidence", 0.0),
        "is_uncertain": getattr(domain, "is_uncertain", True),
        "dimensions": list(getattr(schema, "dimensions", None) or []),
    }
    report = getattr(schema, "sources", None)
    if report is not None:
        # Read-only grounding. The model may reason about what a stale or
        # withheld source means for an explanation; it has no authority over
        # any number, any freshness verdict or any withholding decision.
        facts["source_freshness"] = [
            {"source": s.label or s.source_id, "status": s.freshness,
             "last_refresh_at": s.last_refresh_at, "basis": s.last_refresh_basis,
             "note": s.note}
            for s in report.sources
        ]
        facts["withheld_kpis"] = dict(report.withheld_kpis)
    return facts


def _generate_llm_candidates(ctx: Context, schema: DatasetSchema, signals: Any,
                             graph: Any, llm) -> tuple:
    """Ask the model for mechanisms, and validate whatever comes back."""
    from .llm_hypotheses import build_llm_candidates

    allowed = sorted(set(schema.available_kpis or []))
    try:
        raw = llm.generate_hypotheses(
            signals=signals.model_dump() if hasattr(signals, "model_dump") else signals,
            kpi_semantics=_kpi_semantics(schema, ctx.metric),
            domain=_domain_facts(schema),
            driver_graph=graph.model_dump() if hasattr(graph, "model_dump") else {},
            allowed_metrics=allowed,
        )
    except Exception as exc:                        # never let the LLM break the pipeline
        return [], f"Domain hypothesis generation unavailable ({exc}); the contract-derived " \
                   f"decomposition was used on its own."
    candidates = build_llm_candidates(ctx, raw, allowed)
    note = (raw or {}).get("generation_note") or ""
    if not candidates:
        note = ("No proposed mechanism could be tested with the measures this dataset "
                "contains; the contract-derived decomposition was used on its own.")
    return candidates, note


def investigate(df: pd.DataFrame, schema: DatasetSchema, observation: Dict[str, Any],
                uid: str, llm=None, signals: Any = None,
                graph: Any = None) -> Dict[str, Any]:
    """
    Generate competing explanations and test each against the data.

    Two sources of hypotheses, one measuring machinery. The contract's own
    formula yields a decomposition that always works and needs no model; the
    model proposes mechanisms specific to this kind of business. Both arrive as
    predictions, and `hypotheses.evidence` decides from the data whether each
    prediction held — so neither source can assert its way to a conclusion.
    """
    from .driver_graph import build_driver_graph
    from .llm_hypotheses import category_summary, shortlist as pick_shortlist
    from .observe import Timeframe, slice_period
    from .signals import material_signals

    tf = Timeframe(observation["timeframe"]["year"], observation["timeframe"]["quarter"])
    base_tf = Timeframe(observation["baseline_timeframe"]["year"],
                        observation["baseline_timeframe"]["quarter"])

    ctx = Context(
        df=df, schema=schema, metric=observation["kpi"],
        cur=slice_period(df, tf), base=slice_period(df, base_tf),
        observation=observation, focus=determine_focus(observation),
    )

    # Not enough history for the significance test to qualify the movement in the
    # first place. Proposing mechanisms here would be explaining a number the
    # engine has already said it cannot stand behind.
    if observation.get("history_status") != "sufficient_history":
        return {
            "focus": {}, "focus_label": "", "hypotheses": [], "considered_count": 0,
            "not_carried_forward": [], "documents_indexed": 0, "rag_available": False,
            "llm_used": False, "llm_note": None,
            "nothing_to_explain": True,
            "signal_filter": {},
            "method_note": ((observation.get("history_note") or "")
                            + " No causal hypotheses or confidence scores were generated.").strip(),
        }

    if graph is None:
        graph = build_driver_graph(schema, ctx.metric, observation)
    if signals is None:
        signals = material_signals(observation, {}, ctx.focus)

    # Nothing moved beyond normal variation. Explaining noise on request is the
    # failure this boundary exists to prevent, so the honest answer is no
    # hypotheses at all.
    if getattr(signals, "nothing_material", False):
        return {
            "focus": ctx.focus, "focus_label": ctx.focus_label(),
            "hypotheses": [], "considered_count": 0, "not_carried_forward": [],
            "documents_indexed": 0, "rag_available": False,
            "llm_used": False, "llm_note": None,
            "nothing_to_explain": True,
            "signal_filter": signals.model_dump() if hasattr(signals, "model_dump") else {},
            "method_note": signals.filter_note,
        }

    llm_candidates: List[Dict[str, Any]] = []
    llm_note = None
    if llm is not None and llm.enabled:
        llm_candidates, llm_note = _generate_llm_candidates(ctx, schema, signals, graph, llm)

    candidates = build_candidates(ctx, graph, llm_candidates)
    retriever = Retriever(uid)
    terms = _rag_terms(ctx)

    shortlist = pick_shortlist(candidates, MAX_HYPOTHESES)
    for h in shortlist:
        attach_documentary_evidence(h, retriever, terms)

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
        "nothing_to_explain": False,
        "driver_graph": graph.model_dump() if hasattr(graph, "model_dump") else {},
        "signal_filter": signals.model_dump() if hasattr(signals, "model_dump") else {},
        "category_coverage": category_summary(shortlist),
        "method_note": (
            "Hypotheses come from two sources: a decomposition of the KPI's own formula in the "
            "contract, and mechanisms proposed for this kind of business and then filtered to "
            "those this dataset can test. Both are stated as predictions; every number attached "
            "to them is computed by the data-analysis layer, which decides whether each "
            "prediction held. Every quotation is retrieved verbatim from an uploaded document."
        ),
    }
