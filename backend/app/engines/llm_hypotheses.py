"""
Turning proposed mechanisms into tested hypotheses.

This is the airlock between the language model and the analysis. A model
proposes that longer stays raised cost without matching revenue; nothing about
that proposal is treated as true here. Its predictions are extracted, checked
against the metrics this dataset actually has, and then measured — and the
measurement decides what the evidence says.

Three guarantees are enforced in this module:

  * A prediction naming a metric outside `allowed_metrics` is dropped. The model
    cannot investigate something the dataset does not measure, however confident
    it sounds.
  * A hypothesis left with no measurable prediction is discarded entirely, so an
    untestable claim never reaches the reader wearing the same clothes as a
    tested one.
  * Stance is never taken from the model. Every prediction goes through
    `hypotheses.evidence(..., expect=...)`, which flips a prediction that did
    not hold into evidence *against* the hypothesis that made it.

Whatever survives is an ordinary hypothesis dict, indistinguishable to
`contest.py` from a deterministically generated one, and scored by exactly the
same machinery.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Set

from ..models.investigation import LlmHypothesis, LlmHypothesisSet
from .hypotheses import Context, _compact, evidence

log = logging.getLogger(__name__)

# The families `act.PLAYBOOK` knows how to recommend against. Anything else is
# normalised to "other", which now has a generic play rather than being dropped.
KNOWN_FAMILIES = {"supply", "demand", "competitive", "pricing", "mix", "operational",
                  "quality", "marketing", "seasonality", "data_quality", "other"}

MAX_HYPOTHESES = 8
_SLUG = re.compile(r"[^a-z0-9_]+")


def _slug(text: str, fallback: str) -> str:
    out = _SLUG.sub("_", (text or "").strip().lower()).strip("_")
    return out[:48] or fallback


def build_llm_candidates(ctx: Context, raw: Dict[str, Any],
                         allowed_metrics: Sequence[str]) -> List[Dict[str, Any]]:
    """
    Validate a model's proposed mechanisms and measure each one.

    Returns hypothesis dicts in the shape the rest of the pipeline consumes.
    Anything malformed is dropped quietly rather than raising: a bad generation
    should cost the investigation some breadth, never the whole run.
    """
    allowed: Set[str] = set(allowed_metrics)
    try:
        parsed = LlmHypothesisSet.model_validate(raw or {})
    except Exception as exc:
        log.warning("Hypothesis generation returned an unusable shape: %s", exc)
        return []

    out: List[Dict[str, Any]] = []
    seen_keys: Set[str] = set()

    for index, proposal in enumerate(parsed.hypotheses[:MAX_HYPOTHESES]):
        built = _build_one(ctx, proposal, allowed, index, seen_keys)
        if built:
            out.append(built)
    return out


def _build_one(ctx: Context, proposal: LlmHypothesis, allowed: Set[str],
               index: int, seen_keys: Set[str]) -> Optional[Dict[str, Any]]:
    key = _slug(proposal.key or proposal.title, f"proposed_{index}")
    while key in seen_keys:
        key = f"{key}_{index}"
    seen_keys.add(key)

    dropped: List[str] = []
    items: List[Dict[str, Any]] = []

    for prediction in proposal.predictions:
        metric = (prediction.metric or "").strip()
        if metric not in allowed:
            # The model reached for something this dataset cannot measure. Record
            # it as a known gap rather than silently pretending it was tested.
            if metric:
                dropped.append(metric)
            continue
        if not ctx.has(metric):
            dropped.append(metric)
            continue
        items.append(evidence(
            ctx, metric, "supporting",
            scope=prediction.scope or None,
            weight=_clamp(prediction.weight, 0.5, 1.5),
            reference=_clamp(prediction.reference_pct, 1.0, 100.0),
            expect=prediction.expected_direction,
            note=prediction.rationale or "",
        ))

    items = _compact(items)
    if not items:
        # Nothing about this proposal could be tested. A hypothesis that cannot
        # be checked is not a finding, and presenting it beside checked ones
        # would give it borrowed credibility.
        log.debug("Dropping untestable hypothesis %r (no measurable predictions)", key)
        return None

    cause_metric = proposal.cause_metric if proposal.cause_metric in allowed else None
    if cause_metric is None:
        # Fall back to the strongest prediction, so temporal and consistency
        # checks in `contest` still have a series to work with.
        strongest = max(items, key=lambda e: e.get("strength", 0) * e.get("weight", 1))
        cause_metric = strongest.get("metric")

    missing = list(proposal.missing)
    if dropped:
        missing.append(
            "This explanation also implies a change in "
            + ", ".join(sorted(set(dropped))[:3])
            + ", which this dataset does not measure, so that part is untested.")

    family = proposal.family if proposal.family in KNOWN_FAMILIES else "other"

    return {
        "key": key,
        "title": proposal.title or key.replace("_", " ").capitalize(),
        "family": family,
        "statement": proposal.statement,
        "mechanism": proposal.mechanism,
        "evidence": items,
        "cause_metric": cause_metric,
        "cause_direction": proposal.cause_direction or _direction_of(items, cause_metric),
        "reverse_causation_risk": proposal.reverse_causation_risk or "",
        "rag_queries": list(proposal.rag_queries),
        "contradiction_queries": list(proposal.contradiction_queries),
        "missing": missing,
        "domain_specific": bool(proposal.domain_specific),
        "source": "llm_domain_reasoning",
    }


def _direction_of(items: List[Dict[str, Any]], metric: Optional[str]) -> str:
    for e in items:
        if e.get("metric") == metric:
            change = e.get("change_pct")
            if change is not None:
                return "up" if change > 0 else "down"
    return "down"


def _clamp(value: Optional[float], low: float, high: float) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return low


# ---------------------------------------------------------------------------
# category coverage
# ---------------------------------------------------------------------------
def shortlist(candidates: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    """
    Choose which hypotheses to carry forward, preserving both categories.

    Ranking alone can crowd one category out — several domain-specific
    mechanisms with strong priors would push every general one off a top-five
    list, and the reader would never see that a simple cost or volume account
    was also considered. So the strongest of each category is reserved a place
    before the remaining slots are filled on merit.

    Neither category is invented to fill its slot: if only one produced testable
    candidates, the shortlist is simply drawn from that one.
    """
    testable = [c for c in candidates if c.get("testable", True)]
    if not testable:
        return []

    ordered = sorted(testable,
                     key=lambda h: h.get("prior_support", 0) - 0.5 * h.get("prior_against", 0),
                     reverse=True)

    picked: List[Dict[str, Any]] = []
    for domain_flag in (True, False):
        best = next((h for h in ordered if bool(h.get("domain_specific")) is domain_flag), None)
        if best is not None:
            picked.append(best)

    for h in ordered:
        if len(picked) >= limit:
            break
        if h not in picked:
            picked.append(h)

    # Restore evidence order so the reader still sees the best-supported first.
    picked.sort(key=lambda h: h.get("prior_support", 0) - 0.5 * h.get("prior_against", 0),
                reverse=True)
    return picked[:limit]


def category_summary(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """What each category actually yielded, for honest reporting."""
    domain = [c for c in candidates if c.get("domain_specific")]
    general = [c for c in candidates if not c.get("domain_specific")]
    note = ""
    if not domain:
        note = ("No domain-specific mechanism could be tested against this dataset; "
                "the explanations below are general business ones.")
    elif not general:
        note = ("No general business mechanism added anything testable beyond the "
                "domain-specific explanations below.")
    return {"domain_specific_count": len(domain), "general_count": len(general),
            "coverage_note": note}
