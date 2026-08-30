"""
Assembling a question into an `InvestigationIntent`.

Deterministic grounding runs first and usually finishes the job. The language
model is called only when the grounding is genuinely uncertain — no outcome
found, or several equally plausible ones — and its answer is then validated
against the contract before it is believed. The model chooses among KPIs that
exist; it cannot introduce one. A key it returns that is not in the resolver is
discarded, and the question falls through to a clarification rather than to a
confident answer about the wrong measure.

The ambiguity policy is deliberate, and it turns on a distinction. An outcome
that resolves to *nothing* blocks: there is no answer to give, and inventing a
KPI answers a question nobody asked. An outcome that resolves to several things
does not block. Every one of those candidates is a KPI this dataset genuinely
measures, so the best-scoring one is used, the others are named, and `assumed`
records which was taken -- the reader sees the choice and can restate it. A
vague period behaves the same way, on the same reasoning.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from ..models.investigation import (Ambiguity, InvestigationIntent, KpiRef,
                                    PeriodSpec)
from .grounding import GroundingResult, ground_question
from .periods import resolve_period

log = logging.getLogger(__name__)

# Two outcome candidates this close together are not a ranking, they are a tie,
# and the reader is the only one who can break it.
TIE_MARGIN = 0.15
# Below this, a "match" is really a guess.
MIN_OUTCOME_CONFIDENCE = 0.5


def interpret_question(question: str, schema: Any, df: Any = None,
                       llm: Any = None, contract: Any = None) -> InvestigationIntent:
    """
    Turn a business question into a resolved, contract-grounded intent.

    Never raises on an unanswerable question: an intent whose `blocked` property
    is true carries the reason, and the caller turns that into a clarification.
    """
    question = (question or "").strip()
    if not question:
        return InvestigationIntent(
            question="", intent_type="unsupported",
            ambiguities=[Ambiguity(kind="outcome_unresolved", blocking=True,
                                   message="Ask a question about this dataset to start "
                                           "an investigation.")])

    grounded = ground_question(question, schema)
    ambiguities: List[Ambiguity] = []

    timeframes = _timeframes(df, schema)
    calendar = _calendar(contract)
    period, period_ambiguities = resolve_period(question, timeframes, calendar)
    ambiguities.extend(period_ambiguities)

    outcome, outcome_ambiguities, llm_used, note = _resolve_outcome(
        question, grounded, schema, llm)
    ambiguities.extend(outcome_ambiguities)

    comparisons = _dedupe(grounded.comparison_candidates, outcome)

    intent = InvestigationIntent(
        question=question,
        intent_type="change_explanation" if outcome else "unsupported",
        outcome=outcome,
        comparison_kpis=comparisons,
        period=period,
        dimension_hints=grounded.dimension_hints,
        entity_filters=grounded.entity_filters,
        ambiguities=ambiguities,
        unmapped_terms=grounded.unmapped_terms,
        llm_used=llm_used,
        grounding_note=note or grounded.note,
    )
    return intent


def _resolve_outcome(question: str, grounded: GroundingResult, schema: Any,
                     llm: Any) -> tuple:
    """The KPI the question is asking about, or the reason it could not be found."""
    candidates = [c for c in grounded.outcome_candidates
                  if c.confidence >= MIN_OUTCOME_CONFIDENCE]

    if len(candidates) == 1:
        return candidates[0], [], False, ""

    if len(candidates) > 1:
        top, second = candidates[0], candidates[1]
        if top.confidence - second.confidence >= TIE_MARGIN:
            return top, [], False, ""

        # A tie is not an impasse. Every candidate is a KPI this dataset really
        # measures, so answering about the best-scoring one and saying which
        # others were in play is strictly more useful than refusing -- and it is
        # the case a model is best placed to settle, which it was never asked to
        # do before: the fallback below only ever ran on an EMPTY candidate list.
        near = [c for c in candidates if top.confidence - c.confidence < TIE_MARGIN][:4]
        chosen, note, llm_used = top, "", False
        if llm is not None and getattr(llm, "enabled", False):
            picked, llm_note = _ask_llm(question, schema, llm)
            llm_used = True
            # The model may break the tie; it may not step outside it. An
            # unconstrained pick would let it choose a KPI that scored nothing,
            # which is a different failure from the one being fixed.
            if picked and picked.kpi_key in {c.kpi_key for c in near}:
                chosen, note = picked, llm_note
        others = ", ".join(f"'{c.label}'" for c in near if c.kpi_key != chosen.kpi_key)
        message = (f"Read as '{chosen.label}'. It scored level with {others}, so ask "
                   f"again naming the measure if that is not what you meant."
                   if others else f"Read as '{chosen.label}'.")
        return chosen, [Ambiguity(
            kind="outcome_multiple", blocking=False, message=message,
            candidates=near, assumed=chosen.kpi_key)], llm_used, note

    # Nothing bound deterministically — this is where the model earns its place.
    if llm is not None and getattr(llm, "enabled", False):
        chosen, note = _ask_llm(question, schema, llm)
        if chosen:
            return chosen, [], True, note

    available = [c.label for c in _all_kpi_refs(schema)[:6]]
    hint = f" This dataset measures: {', '.join(available)}." if available else ""
    return None, [Ambiguity(
        kind="outcome_unresolved", blocking=True,
        message=f"No KPI in this dataset matches what the question is asking about.{hint}",
        candidates=_all_kpi_refs(schema)[:6])], bool(llm and getattr(llm, "enabled", False)), ""


def _ask_llm(question: str, schema: Any, llm: Any):
    """
    Let the model pick among the KPIs that exist.

    The returned key is checked against the resolver before it is used, so a
    hallucinated KPI degrades into "could not resolve" rather than into an
    investigation of something the dataset does not measure.
    """
    catalogue = [
        {"key": e.kpi_key, "label": e.label} | (
            {"definition": d} if (d := _definition_of(schema, e.kpi_key)) else {})
        for e in _all_kpi_refs(schema)
    ]
    try:
        raw = llm.understand_question(question, catalogue,
                                      list(getattr(schema, "dimensions", []) or []))
    except Exception as exc:                       # never let the LLM break the flow
        log.warning("Query understanding unavailable: %s", exc)
        return None, f"Query understanding unavailable ({exc}); deterministic grounding used."

    key = (raw or {}).get("outcome_kpi_key")
    valid = {e.kpi_key for e in _all_kpi_refs(schema)}
    if not key or key not in valid:
        return None, ("The question was not resolvable to a KPI this dataset "
                      "measures.")
    label = next((e.label for e in _all_kpi_refs(schema) if e.kpi_key == key), key)
    return (KpiRef(kpi_key=key, label=label, role="outcome",
                   match_basis="llm_proposed_verified",
                   confidence=float((raw or {}).get("confidence") or 0.6)),
            "Resolved by interpreting the question against the KPI contract.")


def _definition_of(schema: Any, key: str) -> str:
    if not hasattr(schema, "kpi_definition"):
        return ""
    d = schema.kpi_definition(key)
    return (getattr(d, "business_definition", "") or "")[:240] if d else ""


def _all_kpi_refs(schema: Any) -> List[KpiRef]:
    refs: List[KpiRef] = []
    for key in getattr(schema, "available_kpis", []) or []:
        d = schema.kpi_definition(key) if hasattr(schema, "kpi_definition") else None
        spec = (getattr(schema, "contract_resolver", None) or {}).get(key)
        label = getattr(d, "name", "") or getattr(spec, "label", "") or key
        refs.append(KpiRef(kpi_key=key, label=label, role="outcome",
                           match_basis="exact_name", confidence=0.0))
    return refs


def _dedupe(comparisons: List[KpiRef], outcome: Optional[KpiRef]) -> List[KpiRef]:
    seen = {outcome.kpi_key} if outcome else set()
    out: List[KpiRef] = []
    for c in comparisons:
        if c.kpi_key in seen:
            continue
        seen.add(c.kpi_key)
        out.append(c)
    return out[:4]


def _timeframes(df: Any, schema: Any) -> List[Dict[str, Any]]:
    if df is None:
        return []
    try:
        from ..engines.observe import available_timeframes
        return available_timeframes(df)
    except Exception:                              # pragma: no cover - defensive
        return []


def _calendar(contract: Any):
    if contract is None:
        return None
    try:
        return contract.calendar("default")
    except Exception:                              # pragma: no cover - defensive
        return None


# ---------------------------------------------------------------------------
# caching
# ---------------------------------------------------------------------------
# Interpretation is the one part of a question-driven investigation that can be
# genuinely expensive relative to what it produces: the LLM fallback is a full
# round-trip for something that, asked twice in a row against an unchanged
# contract, must resolve to the same answer every time. An in-process cache,
# keyed the same way `dataset_service._CACHE` already is (module-level dict,
# explicit invalidation), removes that repeat cost without needing a store.
_INTENT_CACHE: Dict[Tuple[str, str, int], InvestigationIntent] = {}
_INTENT_CACHE_MAX = 256


def _normalise_question(question: str) -> str:
    """Two questions that differ only in spacing or case are the same question."""
    return " ".join((question or "").strip().lower().split())


def _cache_key(question: str, dataset_id: str, schema: Any) -> Tuple[str, str, int]:
    version = int(getattr(schema, "contract_version", 0) or 0)
    return (_normalise_question(question), dataset_id or "", version)


def interpret_question_cached(question: str, schema: Any, df: Any, dataset_id: str,
                              llm: Any = None, contract: Any = None) -> InvestigationIntent:
    """
    `interpret_question`, remembered per (question, dataset, contract version).

    Keyed on the contract's version rather than only its id, so approving a KPI
    contract edit in the KPI Studio — which can change what a question resolves
    to — invalidates every cached reading for that dataset automatically, by
    virtue of the version no longer matching. No explicit cache-clearing call is
    needed for the common case.

    A cache hit is returned as a deep copy: the caller may read freely, but must
    never be able to corrupt what a later caller sees by mutating a shared object.
    """
    key = _cache_key(question, dataset_id, schema)
    cached = _INTENT_CACHE.get(key)
    if cached is not None:
        return cached.model_copy(deep=True)

    intent = interpret_question(question, schema, df, llm=llm, contract=contract)
    if len(_INTENT_CACHE) >= _INTENT_CACHE_MAX:
        _INTENT_CACHE.pop(next(iter(_INTENT_CACHE)), None)      # crude FIFO eviction
    _INTENT_CACHE[key] = intent
    return intent.model_copy(deep=True)


def clear_intent_cache(dataset_id: Optional[str] = None) -> None:
    """
    Drop cached interpretations.

    Not required for a contract edit — the version key already handles that —
    but kept as an explicit escape hatch, mirroring `dataset_service.clear_cache`.
    """
    if dataset_id is None:
        _INTENT_CACHE.clear()
        return
    for key in [k for k in _INTENT_CACHE if k[1] == dataset_id]:
        _INTENT_CACHE.pop(key, None)
