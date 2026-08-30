"""
The `observe` call `/api/dashboard` needs.

Extracted from the retired `services/pipeline.py` at A9 — this is the one
sliver of the old orchestrator any surviving endpoint still calls.
`/api/dashboard` never ran investigate/contest/act, so nothing else from that
module belongs here.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..engines.observe import Timeframe, available_timeframes, observe
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
    for an ordinary single-file dataset.
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
