"""
Matching the words in a question to the KPIs a dataset actually has.

This runs before any language model and resolves most questions on its own. The
contract already carries everything needed: each KPI's name, the semantic tags
discovery assigned it, the source fields its formula reads, and a prose
definition of what it means for this business. The concept library adds the
missing piece — that "readmitted", "readmission_count" and "readmissions" are
the same idea — so a question phrased in the reader's words still finds the
field the dataset happens to use.

Matching is ordered by how much the evidence is worth. An exact KPI name is
near-certain; a semantic tag is strong; a concept alias reaching a KPI through
one of its source fields is good; overlap with the prose definition is weak and
scored accordingly. Nothing here guesses: when two KPIs match a phrase equally
well, both are returned and the caller decides whether that is an ambiguity
worth asking about.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from ..models.investigation import KpiRef

# Words that carry no signal about which KPI is meant. Deliberately small: the
# risk of stripping a real term is worse than the cost of a stopword surviving.
_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "did",
    "do", "does", "why", "what", "how", "when", "where", "which", "who", "in",
    "on", "at", "of", "for", "to", "from", "by", "with", "and", "or", "but",
    "even", "though", "although", "despite", "while", "whereas", "if", "so",
    "our", "we", "us", "my", "their", "its", "this", "that", "these", "those",
    "it", "there", "here", "not", "no", "any", "all", "some", "much", "many",
    "more", "less", "than", "as", "up", "down", "over", "under", "about",
    "q1", "q2", "q3", "q4", "quarter", "year", "last", "previous", "prior",
}

# Words that describe a movement rather than a measure. Used to read intent and
# direction, and excluded from KPI matching so "drop" never binds to a KPI.
_DECLINE_WORDS = {"fall", "fell", "falling", "drop", "dropped", "drops", "decline",
                  "declined", "declining", "plunge", "plunged", "decrease",
                  "decreased", "down", "lower", "worse", "worsened", "shrank",
                  "contracted", "slump", "slumped", "deteriorated", "weak", "weakened"}
_RISE_WORDS = {"rise", "rose", "rising", "increase", "increased", "increasing",
               "grow", "grew", "growing", "up", "higher", "better", "improved",
               "jump", "jumped", "surge", "surged", "climb", "climbed", "spike"}
_FLAT_WORDS = {"flat", "stable", "unchanged", "steady", "same", "constant",
               "remained", "held"}

# Phrases that introduce a KPI the question is contrasting against, rather than
# the one it is asking about. "profit fell even though CAC was up" — the outcome
# is profit; CAC is the contrast.
_CONTRAST_MARKERS = ("even though", "even if", "although", "though", "despite",
                     "in spite of", "while", "whereas", "but", "yet",
                     "without a corresponding", "without corresponding")

_TOKEN = re.compile(r"[a-z0-9_]+")


@dataclass
class GroundingResult:
    outcome_candidates: List[KpiRef]
    comparison_candidates: List[KpiRef]
    dimension_hints: List[str]
    entity_filters: Dict[str, str]
    unmapped_terms: List[str]
    direction: str = ""            # "down" | "up" | ""
    note: str = ""


def _tokens(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


def _ngrams(tokens: List[str], max_n: int = 4) -> List[str]:
    """Longest first, so 'bed occupancy rate' wins over 'bed'."""
    out: List[str] = []
    for n in range(min(max_n, len(tokens)), 0, -1):
        for i in range(len(tokens) - n + 1):
            out.append("_".join(tokens[i:i + n]))
    return out


def _normalise(text: str) -> str:
    return "_".join(_tokens(text))


def _contains_phrase(haystack: str, needle: str) -> bool:
    """
    Whether `needle` appears in `haystack` as whole words.

    A plain substring test is not good enough here: "admissions" is a substring
    of "readmissions", and letting that count would quietly make every question
    about admissions also a question about readmissions — two clinically
    different measures.
    """
    # Split on underscores as well as whitespace: "bed_occupancy_rate" and
    # "Bed occupancy rate" have to compare equal, and the tokeniser keeps
    # underscores inside a token.
    hay = [w for w in _normalise(haystack).split("_") if w]
    need = [w for w in _normalise(needle).split("_") if w]
    if not need or len(need) > len(hay):
        return False
    return any(hay[i:i + len(need)] == need for i in range(len(hay) - len(need) + 1))


def _concept_alias_index() -> Dict[str, str]:
    """
    Every concept alias in the library, mapped to the field-ish name it implies.

    Built once per call from the library's own concept refs so the two can never
    drift apart. The value is the alias itself — matching a KPI happens through
    its `source_fields`, which is what makes this dataset-specific rather than a
    second vocabulary to maintain.
    """
    from ..kpi import library as lib

    index: Dict[str, str] = {}
    for name in dir(lib):
        if not name.startswith("C_"):
            continue
        ref = getattr(lib, name)
        aliases = getattr(ref, "aliases", None)
        concept = getattr(ref, "concept", None)
        if not aliases or not concept:
            continue
        for alias in aliases:
            index.setdefault(_normalise(alias), concept)
            index.setdefault(alias.lower(), concept)
    return index


def _split_contrast(question: str) -> Tuple[str, str]:
    """
    The clause asking the question, and the clause contrasting with it.

    Splitting on the first contrast marker is crude but reliable for the shapes
    these questions take, and it is what stops "profit fell even though CAC rose"
    from treating CAC as the thing to explain.
    """
    low = question.lower()
    best = len(question)
    marker_len = 0
    for marker in _CONTRAST_MARKERS:
        idx = low.find(marker)
        if idx != -1 and idx < best:
            best, marker_len = idx, len(marker)
    if best == len(question):
        return question, ""
    return question[:best], question[best + marker_len:]


def _direction(text: str) -> str:
    toks = set(_tokens(text))
    if toks & _DECLINE_WORDS:
        return "down"
    if toks & _RISE_WORDS:
        return "up"
    return ""


def _kpi_search_space(schema: Any) -> List[Dict[str, Any]]:
    """Everything known about each available KPI, flattened for matching."""
    space: List[Dict[str, Any]] = []
    for key in getattr(schema, "available_kpis", []) or []:
        definition = None
        if hasattr(schema, "kpi_definition"):
            definition = schema.kpi_definition(key)
        spec = (getattr(schema, "contract_resolver", None) or {}).get(key)
        space.append({
            "key": key,
            "label": getattr(definition, "name", "") or getattr(spec, "label", "") or key,
            "tags": [t.lower() for t in (getattr(definition, "semantic_tags", None) or [])],
            "source_fields": [f.lower() for f in (getattr(spec, "source_fields", None)
                                                  or getattr(definition, "source_fields", None) or [])],
            "definition": (getattr(definition, "business_definition", "") or "").lower(),
            "relevance": (getattr(definition, "relevance", "") or "").lower(),
        })
    return space


def _match_clause(clause: str, space: List[Dict[str, Any]],
                  aliases: Dict[str, str], role: str) -> Tuple[List[KpiRef], Set[str]]:
    """
    Every KPI this clause could be referring to, best evidence first.

    Returns the candidates and the set of n-grams that matched something, so the
    caller can report the words that bound to nothing.
    """
    grams = _ngrams(_tokens(clause))
    matched_text: Set[str] = set()
    scored: Dict[str, Tuple[float, str, str]] = {}     # key -> (score, basis, text)

    def offer(key: str, score: float, basis: str, text: str) -> None:
        prev = scored.get(key)
        if prev is None or score > prev[0]:
            scored[key] = (score, basis, text)

    for gram in grams:
        pretty = gram.replace("_", " ")
        for entry in space:
            key_n = _normalise(entry["key"])
            label_n = _normalise(entry["label"])

            if gram == key_n or gram == label_n:
                offer(entry["key"], 1.0, "exact_name", pretty)
                matched_text.add(gram)
                continue
            # A multi-word gram contained in the label is still strong evidence
            # ("bed occupancy" for "Bed occupancy rate"), a single word much less
            # so ("rate" must not claim every rate KPI). Whole words only —
            # "admissions" must not match "readmissions".
            if len(gram) > 3 and (_contains_phrase(label_n, gram)
                                  or _contains_phrase(key_n, gram)):
                weight = 0.85 if "_" in gram else 0.55
                offer(entry["key"], weight, "exact_name", pretty)
                matched_text.add(gram)
                continue
            if gram in entry["tags"]:
                offer(entry["key"], 0.8, "semantic_tag", pretty)
                matched_text.add(gram)
                continue

            concept = aliases.get(gram)
            if concept:
                # The alias names a concept; bind it to a KPI only when one of
                # that KPI's own source fields carries the same alias or concept.
                for fld in entry["source_fields"]:
                    fld_n = _normalise(fld)
                    if fld_n == gram or aliases.get(fld_n) == concept:
                        offer(entry["key"], 0.7, "concept_alias", pretty)
                        matched_text.add(gram)
                        break
                else:
                    if _normalise(concept) in (key_n, label_n):
                        offer(entry["key"], 0.65, "concept_alias", pretty)
                        matched_text.add(gram)
                continue

            if "_" in gram and len(gram) > 6:
                if _contains_phrase(entry["definition"], gram):
                    offer(entry["key"], 0.35, "business_definition", pretty)
                    matched_text.add(gram)

    refs = [
        KpiRef(kpi_key=k, label=next(e["label"] for e in space if e["key"] == k),
               role=role, match_basis=basis, confidence=round(score, 2),
               matched_text=text)
        for k, (score, basis, text) in scored.items()
    ]
    refs.sort(key=lambda r: (-r.confidence, r.kpi_key))
    return refs, matched_text


def ground_question(question: str, schema: Any) -> GroundingResult:
    """
    Resolve a question against the contract, without a language model.

    The outcome is whatever the main clause is about; anything named in a
    contrast clause ("even though CAC was up") becomes a comparison rather than
    the thing to explain. Dimension words are matched against the dataset's own
    dimensions, and their members against the values those columns hold.
    """
    space = _kpi_search_space(schema)
    aliases = _concept_alias_index()
    main, contrast = _split_contrast(question)

    outcome, matched_main = _match_clause(main, space, aliases, "outcome")
    comparison, matched_contrast = _match_clause(contrast, space, aliases, "comparison") \
        if contrast.strip() else ([], set())

    # A KPI named in both clauses belongs to the outcome; drop the duplicate.
    outcome_keys = {r.kpi_key for r in outcome}
    comparison = [r for r in comparison if r.kpi_key not in outcome_keys]
    # A contrast clause names one measure, not a family of them. "even though
    # admissions rose" is about admissions — cost-per-admission also matching,
    # through a shared source field, is noise rather than a second contrast.
    if comparison:
        ceiling = comparison[0].confidence
        comparison = [r for r in comparison if ceiling - r.confidence < 0.25]

    dimension_hints, entity_filters = _match_dimensions(question, schema)

    all_tokens = set(_tokens(question))
    consumed = {t for gram in (matched_main | matched_contrast) for t in gram.split("_")}
    unmapped = sorted(
        t for t in all_tokens
        if t not in _STOP and t not in consumed and t not in _DECLINE_WORDS
        and t not in _RISE_WORDS and t not in _FLAT_WORDS
        and not t.isdigit() and len(t) > 2
        and t not in {d.lower() for d in dimension_hints}
        and t not in {v.lower() for v in entity_filters.values()}
    )

    return GroundingResult(
        outcome_candidates=outcome,
        comparison_candidates=comparison,
        dimension_hints=dimension_hints,
        entity_filters=entity_filters,
        unmapped_terms=unmapped,
        direction=_direction(main),
        note=("Resolved from the KPI contract: KPI names, semantic tags and the "
              "concept library's field aliases."),
    )


def _match_dimensions(question: str, schema: Any) -> Tuple[List[str], Dict[str, str]]:
    """Dimension columns the question names, and the members it filters to."""
    grams = set(_ngrams(_tokens(question)))
    dims: List[str] = []
    filters: Dict[str, str] = {}

    for dim in getattr(schema, "dimensions", []) or []:
        dim_n = _normalise(dim)
        if dim_n in grams or dim_n.rstrip("s") in grams:
            dims.append(dim)

    members = getattr(schema, "dimension_members", None) or {}
    for dim, values in members.items():
        for value in values or []:
            if _normalise(str(value)) in grams:
                filters[dim] = str(value)
                if dim not in dims:
                    dims.append(dim)
                break

    return dims, filters
