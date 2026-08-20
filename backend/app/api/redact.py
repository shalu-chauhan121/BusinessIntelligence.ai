"""
Role-based response shaping (authorisation).

Both roles see the same investigation. The Data Analyst additionally sees how it
was produced: the statistical method and its parameters, the full driver tables,
the evidence ledger behind every confidence score, the per-member consistency
tables and the weekly series used for the temporal check.

Removing these on the server (rather than only hiding them in the UI) is what
makes it authorisation rather than presentation.
"""
from __future__ import annotations

import copy
from typing import Any, Dict

ANALYST_ONLY_SIGNIFICANCE = [
    "robust_z", "robust_sigma_pct", "median_all_history_pct", "sigma_all_history_pct",
    "historical_changes", "method", "statistical_power", "history_points", "same_quarter_points",
    "z_threshold", "material_threshold_pct",
]
ANALYST_ONLY_SCORING = ["score_ledger", "support_score", "against_score", "missing_penalty"]


def is_analyst(user: Dict[str, Any]) -> bool:
    return user.get("role") == "data_analyst"


def redact_observation(observation: Dict[str, Any], analyst: bool) -> Dict[str, Any]:
    if analyst:
        return observation
    out = copy.deepcopy(observation)
    sig = out.get("significance", {})
    for key in ANALYST_ONLY_SIGNIFICANCE:
        sig.pop(key, None)
    sig["explanation"] = _plain_significance(observation)
    out["drivers"] = {}                     # leaders get `top_drivers` only
    out["analyst_view_available"] = True
    return out


def _plain_significance(observation: Dict[str, Any]) -> str:
    sig = observation.get("significance", {})
    verdict = observation.get("verdict")
    normal = sig.get("median_historical_change_pct")
    if verdict == "meaningful_signal":
        base = "This is a meaningful change, not normal fluctuation."
    elif verdict == "within_normal_variation":
        base = "This is inside the range this measure normally moves in."
    else:
        base = "This is an unusual movement, but too small to be commercially material."
    if normal is not None:
        base += f" Typically this comparison moves by about {normal:+.1f}%."
    return base


def redact_investigation(investigation: Dict[str, Any], analyst: bool) -> Dict[str, Any]:
    if analyst:
        return investigation
    out = copy.deepcopy(investigation)
    out.pop("not_carried_forward", None)
    out.pop("considered_count", None)
    return out


def redact_contest(contested: Dict[str, Any], analyst: bool) -> Dict[str, Any]:
    if analyst:
        return contested
    out = copy.deepcopy(contested)
    for h in out.get("hypotheses", []):
        for key in ANALYST_ONLY_SCORING:
            h.get("scoring", {}).pop(key, None)
        contest_block = h.get("contest", {})
        contest_block.pop("temporal_series", None)
        consistency = contest_block.get("consistency", {})
        consistency.pop("members", None)
        corr = consistency.get("correlation")
        if isinstance(corr, dict):
            consistency["correlation"] = {"interpretation": corr.get("interpretation")}
        for e in h.get("evidence", []):
            e.pop("strength", None)
            e.pop("weight", None)
    return out


def redact_result(result: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
    analyst = is_analyst(user)
    return {
        **result,
        "observe": redact_observation(result["observe"], analyst),
        "investigate": redact_investigation(result["investigate"], analyst),
        "contest": redact_contest(result["contest"], analyst),
        "view": {"role": user.get("role"), "analyst_detail_included": analyst},
    }
