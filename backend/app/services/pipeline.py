"""
Orchestration of the four-stage investigation.

    OBSERVE -> INVESTIGATE -> CONTEST -> ACT

Deliberately a plain deterministic sequence, not an autonomous agent loop: the
order of the stages is the product's core idea, so it is expressed in code rather
than left to a model to decide.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from ..db.repositories import InvestigationRepository
from ..engines.act import act
from ..engines.contest import contest
from ..engines.investigate import investigate
from ..engines.observe import Timeframe, available_timeframes, observe
from ..llm.client import get_llm
from . import dataset_service


def resolve_timeframe(df, year: Optional[int], quarter: Optional[int]) -> Timeframe:
    frames = available_timeframes(df)
    if not frames:
        raise ValueError("The dataset contains no usable dates.")
    if year is None:
        latest = frames[-1]
        return Timeframe(latest["year"], latest["quarter"])
    return Timeframe(int(year), int(quarter) if quarter else None)


def run_observe(dataset: Dict[str, Any], metric: Optional[str], year: Optional[int],
                quarter: Optional[int], comparison: str = "previous_period") -> Dict[str, Any]:
    df, schema = dataset_service.load(dataset)
    kpi = metric or ("revenue" if "revenue" in schema.available_kpis else schema.available_kpis[0])
    if kpi not in schema.available_kpis:
        raise ValueError(f"'{kpi}' is not available in this dataset. Available: {', '.join(schema.available_kpis)}")
    tf = resolve_timeframe(df, year, quarter)
    return observe(df, schema, kpi, tf, comparison)


def run_full(uid: str, dataset: Dict[str, Any], metric: Optional[str], year: Optional[int],
             quarter: Optional[int], comparison: str = "previous_period",
             persist: bool = True, use_llm: bool = True) -> Dict[str, Any]:
    started = time.time()
    df, schema = dataset_service.load(dataset)
    llm = get_llm() if use_llm else None

    observation = run_observe(dataset, metric, year, quarter, comparison)
    stage_times = {"observe": round(time.time() - started, 3)}

    t = time.time()
    investigation = investigate(df, schema, observation, uid, llm=llm)
    stage_times["investigate"] = round(time.time() - t, 3)

    t = time.time()
    contested = contest(df, schema, observation, investigation, uid, llm=llm)
    stage_times["contest"] = round(time.time() - t, 3)

    t = time.time()
    action = act(df, observation, investigation, contested, llm=llm)
    stage_times["act"] = round(time.time() - t, 3)

    result = {
        "dataset": {
            "id": dataset["_id"],
            "filename": dataset.get("filename"),
            "rows": dataset.get("schema", {}).get("row_count"),
        },
        "observe": observation,
        "investigate": investigation,
        "contest": contested,
        "act": action,
        "engine": {
            "llm": (llm.status if llm else {"enabled": False, "mode": "disabled"}),
            "stage_seconds": stage_times,
            "total_seconds": round(time.time() - started, 3),
            "pipeline": ["observe", "investigate", "contest", "act"],
        },
    }

    if persist:
        top = (contested.get("ranking") or [{}])[0]
        saved = InvestigationRepository().create(uid, {
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
            "dataset_id": dataset["_id"],
            "result": result,
        })
        result["investigation_id"] = saved["_id"]
    return result
