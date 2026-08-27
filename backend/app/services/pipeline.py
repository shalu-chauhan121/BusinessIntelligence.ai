"""
Orchestration of the investigation.

    UNDERSTAND -> OBSERVE -> INVESTIGATE -> CONTEST -> ACT

Deliberately a plain deterministic sequence, not an autonomous agent loop: the
order of the stages is the product's core idea, so it is expressed in code rather
than left to a model to decide.

`run_question` is the entry point a business question takes: the question is
resolved against the KPI contract first, and only then does the existing
four-stage sequence run on the KPI and period that resolution produced.
`run_full` remains the KPI-and-period entry point, used by the dashboard and by
the question flow once interpretation has finished.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from ..db.repositories import InvestigationRepository
from ..engines.act import act
from ..engines.contest import contest
from ..engines.driver_graph import build_driver_graph
from ..engines.investigate import determine_focus, investigate
from ..engines.observe import Timeframe, available_timeframes, observe
from ..engines.signals import material_signals
from ..llm.client import get_llm
from ..query import interpret_question_cached
from . import dataset_service
from .telemetry import track_processing_step


# The headline KPI a business would lead with, in order of preference. Revenue
# when there is one; otherwise the measure that best represents what the
# organisation *does* — admissions for a hospital, shipments for a carrier —
# rather than whichever KPI happens to sort first. A cost is never the headline.
HEADLINE_TAGS = ["topline", "demand_value", "demand_volume", "activity", "throughput"]


def default_kpi(schema) -> str:
    """Pick the KPI to lead with when the caller did not name one."""
    available = list(schema.available_kpis)
    if not available:
        raise ValueError("This dataset has no approved KPIs to analyse.")
    if "revenue" in available:
        return "revenue"

    resolver = schema.contract_resolver or {}

    def tags(key: str) -> List[str]:
        definition = getattr(resolver.get(key), "definition", None)
        return list(getattr(definition, "semantic_tags", []) or [])

    for tag in HEADLINE_TAGS:
        for key in available:
            if tag in tags(key):
                return key
    # Nothing declared a headline role: fall back to the first measure that is a
    # plain additive quantity and is not a cost.
    for key in available:
        spec = resolver.get(key)
        if spec is not None and spec.kind == "sum" and spec.higher_is_better:
            return key
    return available[0]


def resolve_timeframe(df, year: Optional[int], quarter: Optional[int]) -> Timeframe:
    frames = available_timeframes(df)
    if not frames:
        raise ValueError("The dataset contains no usable dates.")
    if year is None:
        latest = frames[-1]
        return Timeframe(latest["year"], latest["quarter"])
    return Timeframe(int(year), int(quarter) if quarter else None)


def run_observe(dataset: Dict[str, Any], metric: Optional[str], year: Optional[int],
                quarter: Optional[int], comparison: str = "previous_period",
                uid: Optional[str] = None, as_of: Optional[str] = None) -> Dict[str, Any]:
    with track_processing_step("Observe", "Non-LLM Processing"):
        df, schema = dataset_service.load(dataset, uid, as_of=as_of)
        kpi = metric or default_kpi(schema)
        if kpi not in schema.available_kpis:
            raise ValueError(
                f"'{kpi}' is not an approved KPI for this dataset. Available: "
                f"{', '.join(schema.available_kpis)}. Define or approve it in the KPI contract."
            )
        tf = resolve_timeframe(df, year, quarter)
        observation = observe(df, schema, kpi, tf, comparison)
        _attach_sources(observation, schema)
        return observation


def _attach_sources(observation: Dict[str, Any], schema) -> None:
    """
    Surface per-source freshness on the observation, deterministically. A no-op
    for an ordinary single-file dataset. The LLM may later explain what a stale
    or withheld source implies; it never computes or overrides any of it.
    """
    report = getattr(schema, "sources", None)
    if report is None:
        return
    observation["sources"] = report.to_dict()

    warnings = list(observation.get("data_warnings") or [])
    for src in report.sources:
        who = src.label or src.source_id
        if src.freshness == "stale":
            warnings.append(f"{who} was last refreshed at {src.last_refresh_at} "
                            f"({src.last_refresh_basis}) and is stale for its "
                            f"{src.refresh_cadence} cadence.")
        elif src.freshness == "missing":
            warnings.append(f"{who} contributed no data for this view.")
        elif src.freshness == "freshness_unknown":
            warnings.append(f"Freshness of {who} cannot be asserted: {src.note}")
    for measure, periods in sorted(report.periods_pending.items()):
        warnings.append(f"'{measure}' has not caught up for the most recent "
                        f"{len(periods)} period(s); those are pending, not missing data.")
    for kpi_id, reason in sorted(report.withheld_kpis.items()):
        warnings.append(f"KPI withheld — {reason}")
    observation["data_warnings"] = warnings


def prepare_stage_context(df, schema, observation: Dict[str, Any],
                          intent: Optional[Any] = None,
                          comparison: str = "previous_period") -> Dict[str, Any]:
    """
    The comparisons, focus, driver graph and material-signal filter one
    observation needs before INVESTIGATE can run.

    Shared between `run_full` and the single-stage endpoints so that calling
    `/api/investigate` for one KPI and calling the full pipeline for the same
    KPI can never disagree about what counts as a material signal or a driver —
    there is exactly one place this is computed.
    """
    comparisons = _observe_comparisons(df, schema, intent, observation, comparison)
    focus = determine_focus(observation)
    graph = build_driver_graph(schema, observation["kpi"], observation,
                               getattr(intent, "dimension_hints", None))
    signals = material_signals(observation, comparisons, focus)
    return {"comparisons": comparisons, "focus": focus, "graph": graph, "signals": signals}


def prepare_stage_context(df, schema, observation: Dict[str, Any],
                          intent: Optional[Any] = None,
                          comparison: str = "previous_period") -> Dict[str, Any]:
    """
    The comparisons, focus, driver graph and material-signal filter one
    observation needs before INVESTIGATE can run.

    Shared between `run_full` and the single-stage endpoints so that calling
    `/api/investigate` for one KPI and calling the full pipeline for the same
    KPI can never disagree about what counts as a material signal or a driver —
    there is exactly one place this is computed.
    """
    comparisons = _observe_comparisons(df, schema, intent, observation, comparison)
    focus = determine_focus(observation)
    graph = build_driver_graph(schema, observation["kpi"], observation,
                               getattr(intent, "dimension_hints", None))
    signals = material_signals(observation, comparisons, focus)
    return {"comparisons": comparisons, "focus": focus, "graph": graph, "signals": signals}


def run_full(uid: str, dataset: Dict[str, Any], metric: Optional[str], year: Optional[int],
             quarter: Optional[int], comparison: str = "previous_period",
             persist: bool = True, use_llm: bool = True,
             persona: Optional[str] = None, question: str = "",
             intent: Optional[Any] = None, as_of: Optional[str] = None) -> Dict[str, Any]:
    started = time.time()
    with track_processing_step("Load dataset", "Non-LLM Processing"):
        df, schema = dataset_service.load(dataset, uid, as_of=as_of)
    llm = get_llm() if use_llm else None

    observation = run_observe(dataset, metric, year, quarter, comparison, uid, as_of=as_of)
    stage_times = {"observe": round(time.time() - started, 3)}

    # Everything the later stages reason about passes through here, so a
    # question-driven run and a KPI-driven one share exactly the same machinery.
    ctx = prepare_stage_context(df, schema, observation, intent, comparison)
    comparisons, graph, signals = ctx["comparisons"], ctx["graph"], ctx["signals"]

    t = time.time()
    with track_processing_step("Investigate", "Non-LLM Processing"):
        investigation = investigate(df, schema, observation, uid, llm=llm,
                                    signals=signals, graph=graph)
    stage_times["investigate"] = round(time.time() - t, 3)

    t = time.time()
    with track_processing_step("Contest", "Non-LLM Processing"):
        contested = contest(df, schema, observation, investigation, uid, llm=llm)
    stage_times["contest"] = round(time.time() - t, 3)

    t = time.time()
    with track_processing_step("Act", "Non-LLM Processing"):
        action = act(df, observation, investigation, contested, llm=llm,
                     resolver=schema.contract_resolver, persona=persona)
    stage_times["act"] = round(time.time() - t, 3)

    result = {
        "dataset": {
            "id": dataset["_id"],
            "filename": dataset.get("filename"),
            "rows": dataset.get("schema", {}).get("row_count"),
        },
        "question": question,
        "intent": intent.model_dump() if hasattr(intent, "model_dump") else None,
        "comparisons": comparisons,
        "observe": observation,
        "investigate": investigation,
        "contest": contested,
        "act": action,
        "sources": observation.get("sources"),
        "engine": {
            "llm": (llm.status if llm else {"enabled": False, "mode": "disabled"}),
            "persona": persona,
            "stage_seconds": stage_times,
            "total_seconds": round(time.time() - started, 3),
            "pipeline": ["understand", "observe", "investigate", "contest", "act"]
                        if question else ["observe", "investigate", "contest", "act"],
        },
    }

    if persist:
        top = (contested.get("ranking") or [{}])[0]
        saved = InvestigationRepository().create(uid, {
            "question": question,
            "kpi": observation["kpi"],
            "kpi_label": observation["kpi_label"],
            "timeframe": observation["timeframe"],
            "baseline_timeframe": observation["baseline_timeframe"],
            "comparison": comparison,
            "change_pct": observation.get("change_pct"),
            "verdict": observation.get("verdict"),
            "headline": action["narrative"]["headline"],
            "leading_hypothesis": top.get("title"),
            "leading_confidence": top.get("confidence"),
            "persona": persona,
            "dataset_id": dataset["_id"],
            "result": result,
        })
        result["investigation_id"] = saved["_id"]
    return result


def _observe_comparisons(df, schema, intent, observation: Dict[str, Any],
                         comparison: str) -> Dict[str, Any]:
    """
    Observe the KPIs the question contrasted against, alongside the outcome.

    A question like "why did occupancy fall even though admissions rose" is
    unanswerable unless admissions is measured too — including when it turns out
    not to have moved, which is frequently the whole point.
    """
    out: Dict[str, Any] = {}
    if intent is None:
        return out
    tf = Timeframe(observation["timeframe"]["year"], observation["timeframe"]["quarter"])
    for ref in getattr(intent, "comparison_kpis", None) or []:
        if ref.kpi_key == observation["kpi"] or ref.kpi_key not in schema.available_kpis:
            continue
        try:
            out[ref.kpi_key] = observe(df, schema, ref.kpi_key, tf, comparison)
        except Exception:                          # pragma: no cover - defensive
            continue
    return out


def run_question(uid: str, dataset: Dict[str, Any], question: str,
                 persona: Optional[str] = None, persist: bool = True,
                 use_llm: bool = True,
                 confirmed_intent: Optional[Any] = None,
                 as_of: Optional[str] = None) -> Dict[str, Any]:
    """
    Answer a business question against a dataset.

    Interpretation comes first and may refuse: a question whose outcome KPI
    cannot be resolved, or that names a period this dataset does not hold,
    returns a clarification rather than an investigation of something adjacent.
    Guessing produces a confident answer to a question nobody asked.
    """
    df, schema = dataset_service.load(dataset, uid, as_of=as_of)
    llm = get_llm() if use_llm else None

    intent = confirmed_intent or interpret_question_cached(
        question, schema, df, dataset["_id"], llm=llm,
        contract=getattr(schema, "contract", None))

    if intent.blocked or not intent.outcome:
        return {
            "status": "needs_clarification",
            "question": question,
            "intent": intent.model_dump(),
            "ambiguities": [a.model_dump() for a in intent.ambiguities if a.blocking],
            "dataset": {"id": dataset["_id"], "filename": dataset.get("filename")},
        }

    period = intent.period
    result = run_full(
        uid, dataset, intent.outcome.kpi_key,
        period.year if period else None,
        period.quarter if period else None,
        comparison=period.comparison if period else "previous_period",
        persist=persist, use_llm=use_llm, persona=persona,
        question=question, intent=intent, as_of=as_of,
    )
    result["status"] = "ok"
    result["assumptions"] = intent.assumptions
    return result
