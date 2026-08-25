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

# How a driver was RANKED is analyst detail: the score components, the member's
# robust z against its own history, the Shapley axis weighting. A leader still
# sees which drivers came out on top, their rank, and their contribution -- the
# finding -- without the arithmetic that produced the ordering.
ANALYST_ONLY_DRIVER = [
    "score_components", "member_score", "axis_weight", "score_weight_used",
    "member_robust_z", "member_significance_note", "weeks_outside_band",
    "weeks_in_period", "persistence_note", "method",
]

# A leader reads the KPI Contract to understand what a number means; the machinery
# that produced the proposal — the row-level checks, the derivation rule, the
# confidence internals — is analyst detail.
ANALYST_ONLY_KPI = ["provenance", "confidence", "comparability", "validation_rules"]
ANALYST_ONLY_CONTRACT = ["field_profiles", "rejected_candidates", "screened_by"]


def is_analyst(user: Dict[str, Any]) -> bool:
    return user.get("role") == "data_analyst"


def redact_contract(contract: Dict[str, Any], analyst: bool) -> Dict[str, Any]:
    """
    Shape a KPI Contract for the reader's role.

    What a KPI *means* — its definition, formula, unit, grain, business rules and
    approval state — is never hidden: that is the whole point of the contract.
    Only the derivation machinery is analyst-only.

    Fields are emptied rather than removed, so the response still satisfies the
    published `KpiContract` schema. A consumer sees the same shape whoever asks;
    the detail is what changes.
    """
    if analyst:
        return contract
    out = copy.deepcopy(contract)
    out["field_profiles"] = []
    out["rejected_candidates"] = []
    for kpi in out.get("kpis", []) or []:
        provenance = kpi.get("provenance") or {}
        # Where a definition came from is part of trusting it, so the origin and
        # the fact of review survive even though the evidence behind them does not.
        kpi["provenance"] = {
            "origin": provenance.get("origin", "general_library"),
            "derived_from": [],
            "derivation_rule": None,
            "computability_evidence": {},
            "screened_by": provenance.get("screened_by"),
            "screening_verdict": None,
            "created_at": provenance.get("created_at", ""),
            "edited_by": [],
            "notes": [],
        }
        kpi["comparability"] = []
        kpi["validation_rules"] = []
    out["analyst_view_available"] = True
    return out


def redact_observation(observation: Dict[str, Any], analyst: bool) -> Dict[str, Any]:
    if analyst:
        return observation
    out = copy.deepcopy(observation)
    sig = out.get("significance", {})
    for key in ANALYST_ONLY_SIGNIFICANCE:
        sig.pop(key, None)
    sig["explanation"] = _plain_significance(observation)
    out["drivers"] = {}                     # leaders get `top_drivers` only

    # The ranking survives; the machinery behind it does not.
    for driver in out.get("top_drivers") or []:
        for key in ANALYST_ONLY_DRIVER:
            driver.pop(key, None)
    out.pop("dimension_shapley", None)
    out.pop("dimension_ranking", None)

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
