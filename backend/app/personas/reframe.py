"""
Saying the same findings differently, and advising different actions on them.

An analyst and a floor manager reading one investigation must not be able to
reach different conclusions about what happened — but they should absolutely
reach different conclusions about what to do next. An analyst's next step is
another analysis; a floor manager's is something they can start on the next
shift. Handing both the same recommendation serves neither.

The mechanism keeps those two things apart:

  * `recommendation_cores` extracts the persona-invariant substrate from the
    contested hypotheses — which driver, how confident, what evidence for and
    against, what would change the conclusion. This is computed once and is
    identical for every reader.
  * `reframe` then produces one persona's explanation and actions *from* that
    substrate. It may change emphasis, depth, vocabulary and the action itself;
    it may not introduce a driver the substrate does not contain.

That last constraint is what stops "reframing" from becoming "inventing". Every
recommendation is checked against the cause metrics the evidence actually
implicated, and one that reaches outside them is dropped rather than shown.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..models.investigation import RecommendationCore
from .profiles import PersonaProfile, persona_for

log = logging.getLogger(__name__)

# Below this, the evidence does not support recommending an intervention to
# anybody. Matches the threshold `act.build_recommendations` already applies.
MIN_ACTIONABLE_CONFIDENCE = 20


def recommendation_cores(contested: Dict[str, Any],
                         recommendations: List[Dict[str, Any]],
                         focus: Optional[Dict[str, str]] = None) -> List[RecommendationCore]:
    """
    The facts every persona's advice must rest on.

    Assembled from the contested hypotheses and the deterministic
    recommendations, before any persona is consulted. Two readers given
    different actions are still given them for the same reasons.
    """
    by_key = {r.get("based_on", {}).get("key"): r for r in recommendations}
    cores: List[RecommendationCore] = []

    for h in (contested.get("hypotheses") or [])[:4]:
        scoring = h.get("scoring") or {}
        if scoring.get("confidence", 0) < MIN_ACTIONABLE_CONFIDENCE:
            continue
        rec = by_key.get(h.get("key")) or {}
        evidence = h.get("evidence") or []
        contest_block = h.get("contest") or {}

        cores.append(RecommendationCore(
            hypothesis_key=h.get("key", ""),
            hypothesis_title=h.get("title", ""),
            cause_metric=h.get("cause_metric"),
            cause_direction=h.get("cause_direction"),
            confidence=float(scoring.get("confidence") or 0),
            confidence_label=scoring.get("confidence_band", ""),
            causal_claim=scoring.get("causal_claim", ""),
            supporting_evidence=[e.get("detail", "") for e in evidence
                                 if e.get("stance") == "supporting"][:4],
            contradicting_evidence=[e.get("detail", "") for e in evidence
                                    if e.get("stance") == "contradicting"][:4],
            missing_evidence=list(h.get("missing") or [])[:3],
            affected_areas=_affected_areas(contest_block, rec, focus),
            monitoring_threshold=(rec.get("monitoring") or [None])[0],
            what_would_change_this=list(rec.get("what_would_change_this") or [])[:3],
        ))
    return cores


def _affected_areas(contest_block: Dict[str, Any], rec: Dict[str, Any],
                    focus: Optional[Dict[str, str]] = None) -> List[str]:
    """
    Where in the business this shows up.

    The investigation's own focus comes first — it is the dimension member that
    over-contributed to the movement, which is exactly the area a manager or
    shift lead needs named.
    """
    areas: List[str] = [str(v) for v in (focus or {}).values() if v]
    scope = (contest_block.get("temporal") or {}).get("scope")
    if scope and scope != "whole business" and scope not in areas:
        areas.append(scope)
    consistency = contest_block.get("consistency") or {}
    for member in (consistency.get("members") or [])[:3]:
        name = member.get("member")
        if name and str(name) not in areas:
            areas.append(str(name))
    return areas[:4]


def reframe(profile: PersonaProfile, observation: Dict[str, Any],
            investigation: Dict[str, Any], contested: Dict[str, Any],
            recommendations: List[Dict[str, Any]], llm=None) -> Dict[str, Any]:
    """
    One persona's view of a finished investigation.

    Returns the explanation and the recommendations for this reader, plus the
    invariant substrate they were built from so a caller can verify that nothing
    was invented. Degrades to a deterministic reframing when no model is
    available — the persona still changes what is emphasised and what is
    advised, just without generated prose.
    """
    cores = recommendation_cores(contested, recommendations,
                                 focus=investigation.get("focus"))
    allowed_causes = {c.cause_metric for c in cores if c.cause_metric}

    generated: Optional[Dict[str, Any]] = None
    note = ""
    if llm is not None and getattr(llm, "enabled", False) and cores:
        try:
            generated = llm.write_for_persona(
                profile, observation, investigation, contested,
                [c.model_dump() for c in cores])
        except Exception as exc:                   # never let the LLM break the pipeline
            note = f"Persona narrative unavailable ({exc}); a deterministic framing was used."
            log.warning("Persona reframing failed for %s: %s", profile.key, exc)

    if generated:
        actions, dropped = _validate_actions(generated.get("recommendations") or [],
                                             allowed_causes, cores)
        if dropped:
            note = (f"{dropped} recommendation(s) were withheld because they rested on a "
                    f"driver the evidence did not establish.")
        return {
            "persona": profile.key,
            "persona_label": profile.label,
            "summary": generated.get("summary", ""),
            "what_changed": generated.get("what_changed", ""),
            "why_it_likely_happened": generated.get("why_it_likely_happened", ""),
            "what_we_cannot_yet_say": generated.get("what_we_cannot_yet_say", ""),
            "recommendations": actions,
            "question_to_ask": generated.get("question_to_ask", ""),
            "recommendation_basis": [c.model_dump() for c in cores],
            "generated": True,
            "note": note,
        }

    deterministic = _deterministic_reframe(profile, observation, contested, cores)
    deterministic["note"] = note
    return deterministic


def _validate_actions(actions: List[Dict[str, Any]], allowed_causes: set,
                      cores: List[RecommendationCore]) -> tuple:
    """
    Keep only advice that traces back to a driver the evidence implicated.

    A persona may say something in its own way; it may not point at a cause the
    investigation never established. Anything unattributable is dropped rather
    than shown with borrowed authority.
    """
    keys = {c.hypothesis_key for c in cores}
    kept: List[Dict[str, Any]] = []
    dropped = 0
    for action in actions:
        basis = (action.get("based_on") or "").strip()
        if basis and basis not in allowed_causes and basis not in keys:
            dropped += 1
            continue
        kept.append({
            "action": action.get("action", ""),
            "why": action.get("why", ""),
            "owner": action.get("owner", ""),
            "timeframe": action.get("timeframe", ""),
            "based_on": basis,
        })
    return kept, dropped


def _deterministic_reframe(profile: PersonaProfile, observation: Dict[str, Any],
                           contested: Dict[str, Any],
                           cores: List[RecommendationCore]) -> Dict[str, Any]:
    """
    A persona's view assembled without a model.

    Still genuinely persona-aware — depth and advice both change — but built by
    selection rather than generation, so the product behaves sensibly with no
    API key configured.
    """
    kpi = observation.get("kpi_label", "the KPI")
    change = observation.get("change_pct")
    period = (observation.get("timeframe") or {}).get("pretty", "the period")
    change_txt = f"{change:+.1f}%" if isinstance(change, (int, float)) else "n/a"
    leader = cores[0] if cores else None

    what_changed = f"{kpi} moved {change_txt} in {period}."
    if leader:
        why = (f"The best-evidenced account is {leader.hypothesis_title.lower()} "
               f"({leader.confidence:.0f}%, {leader.confidence_label}). {leader.causal_claim}")
        uncertain = "; ".join(leader.contradicting_evidence[:1] + leader.missing_evidence[:1]) \
            or "The evidence does not yet separate this from the alternatives."
    else:
        why = "No explanation reached the confidence needed to be worth acting on."
        uncertain = "The evidence available cannot yet account for this movement."

    if profile.detail_level == "full":
        summary = f"{what_changed} {why} Open questions: {uncertain}"
    elif profile.detail_level == "standard":
        summary = f"{what_changed} {why}"
    else:
        summary = f"{what_changed} {why.split('.')[0]}."

    return {
        "persona": profile.key,
        "persona_label": profile.label,
        "summary": summary,
        "what_changed": what_changed,
        "why_it_likely_happened": why,
        "what_we_cannot_yet_say": uncertain,
        "recommendations": _deterministic_actions(profile, cores),
        "question_to_ask": (f"What changed in {cores[0].affected_areas[0]} this period?"
                            if cores and cores[0].affected_areas else
                            "What else changed in this period that the data does not record?"),
        "recommendation_basis": [c.model_dump() for c in cores],
        "generated": False,
    }


# What each persona is told to do about an established driver, in their own
# frame of reference. The driver and the evidence are identical across rows;
# only the action changes.
_ACTION_TEMPLATES = {
    "business_analyst": (
        "Decompose {driver} across the dimensions that moved and test whether the "
        "relationship holds within each.",
        "Analysis backlog", "next analysis cycle"),
    "business_manager": (
        "Review {driver} with the team accountable for {area} and agree what changes "
        "this quarter.",
        "Area manager", "this quarter"),
    "business_leader": (
        "Decide whether {driver} warrants reprioritising resource toward {area}, or "
        "whether to accept the current trajectory.",
        "Executive sponsor", "this planning cycle"),
    "domain_specialist": (
        "Assess whether {driver} is consistent with what you would expect "
        "operationally in {area}, and identify the practice behind it.",
        "Domain lead", "next 2-4 weeks"),
    "operational_user": (
        "Watch {driver} in {area} at the start of each shift and raise it if it moves "
        "further.",
        "Shift lead", "immediately"),
}


def _driver_phrase(core: RecommendationCore) -> str:
    """
    How to refer to this hypothesis's driver inside a sentence.

    A hypothesis with a single cause metric reads naturally as that measure. One
    without — a concentration finding, say — has to be referred to by what it
    claims, so the title is lowercased and stripped of its leading article to
    sit inside a sentence rather than starting one.
    """
    if core.cause_metric:
        return core.cause_metric.replace("_", " ")
    # No single measurable driver — a concentration finding, for instance. The
    # hypothesis title is a whole clause and reads badly mid-sentence, and the
    # area it concerns is already carried separately, so refer to the movement
    # itself and let the template place it.
    return "the movement"


def _deterministic_actions(profile: PersonaProfile,
                           cores: List[RecommendationCore]) -> List[Dict[str, Any]]:
    template, owner, timeframe = _ACTION_TEMPLATES.get(
        profile.key, _ACTION_TEMPLATES["business_analyst"])
    out: List[Dict[str, Any]] = []

    for core in cores[:3]:
        driver = _driver_phrase(core)
        area = core.affected_areas[0] if core.affected_areas else "the affected area"
        out.append({
            "action": template.replace("{driver}", driver).replace("{area}", area),
            "why": (f"{core.hypothesis_title} is the {core.confidence_label or 'current'} "
                    f"account at {core.confidence:.0f}% confidence. {core.causal_claim}"),
            "owner": owner,
            "timeframe": timeframe,
            "based_on": core.cause_metric or core.hypothesis_key,
        })

    if not out:
        out.append({
            "action": ("Treat this as unexplained for now and gather the missing evidence "
                       "before acting."),
            "why": "No explanation reached a confidence worth acting on.",
            "owner": "Analysis backlog",
            "timeframe": "before the next review",
            "based_on": "",
        })
    return out


def reframe_for(persona_key: Optional[str], observation: Dict[str, Any],
                investigation: Dict[str, Any], contested: Dict[str, Any],
                recommendations: List[Dict[str, Any]], llm=None) -> Dict[str, Any]:
    """Convenience wrapper resolving a persona key to its profile first."""
    return reframe(persona_for(persona_key), observation, investigation,
                   contested, recommendations, llm=llm)
