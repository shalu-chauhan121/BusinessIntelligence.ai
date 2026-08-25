"""
The boundary between what the data says and what the language model is told.

Everything a model learns about this dataset's numbers passes through here.
That is the point: a model handed the full observation would see every KPI that
wobbled by a percent and would, reliably, produce a confident explanation for
each. Filtering first means the only movements it can reason about are the ones
a deterministic significance test already called real.

Three rules, all of them reusing tests the engines already apply rather than
inventing new thresholds:

  * A KPI is a *finding* only if `observe` gave it the verdict
    `meaningful_signal`. Anything else is carried as context, flagged as not
    having moved.
  * A dimension member is a *driver* only if it passes `determine_focus` —
    a large share of the change, and more of it than its own size implies.
  * A KPI the question explicitly named is always carried, even when flat,
    because "profit fell even though admissions rose" cannot be answered
    without the metric that did not move.

When nothing is material, that is the answer. The pipeline reports that there is
no meaningful change to explain rather than asking anything to explain noise.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..models.investigation import MaterialSignals

# The one verdict that means "this really moved". The other two — within normal
# variation, and statistically unusual but immaterial — are explicitly not
# findings, and `observe` has already done the work of telling them apart.
MATERIAL_VERDICT = "meaningful_signal"


def _summarise(observation: Dict[str, Any], *, moved: bool,
               named_in_question: bool = False) -> Dict[str, Any]:
    """One KPI's movement, reduced to what a hypothesis could reason about."""
    significance = observation.get("significance") or {}
    return {
        "kpi": observation.get("kpi"),
        "label": observation.get("kpi_label"),
        "unit": observation.get("unit"),
        "higher_is_better": observation.get("higher_is_better"),
        "current": observation.get("current_value"),
        "baseline": observation.get("baseline_value"),
        "change_pct": observation.get("change_pct"),
        "direction": observation.get("direction"),
        "is_unfavourable": observation.get("is_unfavourable"),
        "verdict": observation.get("verdict"),
        "period": (observation.get("timeframe") or {}).get("pretty"),
        "baseline_period": (observation.get("baseline_timeframe") or {}).get("pretty"),
        "moved": moved,
        "named_in_question": named_in_question,
        # Plain-language significance only. The z-score itself is an analyst
        # detail and has no bearing on which mechanism is plausible.
        "significance_note": significance.get("summary") or significance.get("explanation") or "",
    }


def material_signals(outcome: Dict[str, Any],
                     comparisons: Optional[Dict[str, Dict[str, Any]]] = None,
                     focus: Optional[Dict[str, str]] = None,
                     scoreboard_limit: int = 6) -> MaterialSignals:
    """
    Filter an observation down to the signals worth explaining.

    `outcome` is the observation for the KPI the question asked about;
    `comparisons` holds observations for any KPI the question named alongside
    it. Both are produced by `observe`, so every number here has already been
    computed deterministically.
    """
    comparisons = comparisons or {}
    considered = 0
    material: List[Dict[str, Any]] = []
    context: List[Dict[str, Any]] = []

    # 1. The outcome itself. Its verdict decides whether there is anything to
    #    explain at all.
    considered += 1
    outcome_moved = outcome.get("verdict") == MATERIAL_VERDICT
    entry = _summarise(outcome, moved=outcome_moved, named_in_question=True)
    (material if outcome_moved else context).append(entry)

    # 2. KPIs the question named. Always carried, flagged honestly. A flat
    #    comparison metric is frequently the whole point of the question.
    for key, obs in comparisons.items():
        considered += 1
        moved = obs.get("verdict") == MATERIAL_VERDICT
        summary = _summarise(obs, moved=moved, named_in_question=True)
        (material if moved else context).append(summary)

    # 3. Everything else the scoreboard measured, but only where it moved
    #    enough to be worth a mechanism. The scoreboard has no per-KPI verdict —
    #    it is a two-period comparison — so materiality here is a magnitude
    #    test, deliberately stricter than the outcome's own significance test
    #    because there is no baseline distribution behind it.
    named = {outcome.get("kpi"), *comparisons.keys()}
    others = []
    for row in (outcome.get("kpi_scoreboard") or []):
        if row.get("key") in named:
            continue
        considered += 1
        change = row.get("change_pct")
        if change is None or abs(change) < 5.0:
            continue
        others.append({
            "kpi": row.get("key"),
            "label": row.get("label"),
            "unit": row.get("unit"),
            "higher_is_better": row.get("higher_is_better"),
            "current": row.get("current"),
            "baseline": row.get("baseline"),
            "change_pct": change,
            "direction": "up" if change > 0 else "down",
            "moved": True,
            "named_in_question": False,
            "significance_note": "Two-period comparison; not independently significance-tested.",
        })
    others.sort(key=lambda r: -abs(r.get("change_pct") or 0))
    material.extend(others[:scoreboard_limit])

    # 4. Dimension members that genuinely drove the outcome.
    focus_members = _focus_members(outcome, focus)

    nothing_material = not outcome_moved and not any(m.get("moved") for m in material)
    note = _filter_note(considered, len(material), outcome_moved, nothing_material)

    return MaterialSignals(
        outcome_kpi=outcome.get("kpi") or "",
        material=material,
        context=context,
        focus_members=focus_members,
        signals_considered=considered,
        signals_retained=len(material),
        nothing_material=nothing_material,
        filter_note=note,
    )


def _focus_members(observation: Dict[str, Any],
                   focus: Optional[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    The dimension members responsible for the change.

    Reuses `determine_focus`'s thresholds rather than a second set: a member
    must carry a large share of the movement *and* over-contribute relative to
    its own size, so the "driver" of a decline is never just the biggest segment.
    """
    from .investigate import MIN_CONTRIBUTION_PCT, MIN_OVER_INDEX

    rows: List[Dict[str, Any]] = []
    for dim, members in (observation.get("drivers") or {}).items():
        for r in members or []:
            if r.get("is_aggregate"):
                continue
            contribution, over_index = r.get("contribution_pct"), r.get("over_index")
            if contribution is None or contribution < MIN_CONTRIBUTION_PCT:
                continue
            if over_index is not None and over_index < MIN_OVER_INDEX:
                continue
            rows.append({
                "dimension": dim,
                "member": r.get("name"),
                "contribution_pct": contribution,
                "over_index": over_index,
                "change_pct": r.get("change_pct"),
                "effects": r.get("effects"),
            })
    rows.sort(key=lambda r: -(r.get("contribution_pct") or 0))

    # A member the caller already identified as the focus leads the list.
    if focus:
        preferred = set(focus.values())
        rows.sort(key=lambda r: (r.get("member") not in preferred,
                                 -(r.get("contribution_pct") or 0)))
    return rows[:6]


def _filter_note(considered: int, retained: int, outcome_moved: bool,
                 nothing_material: bool) -> str:
    if nothing_material:
        return (f"{considered} measure(s) were checked and none moved beyond normal "
                "variation for this dataset, so there is no material change to explain.")
    lead = (f"{retained} of {considered} measure(s) moved materially and were carried "
            "into the explanation; the rest are held as context.")
    if not outcome_moved:
        lead += (" The KPI asked about did not itself move beyond normal variation — "
                 "any explanation below concerns the measures that did.")
    return lead
