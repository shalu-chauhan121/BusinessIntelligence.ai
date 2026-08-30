"""
Matching the words in a question to the KPIs a dataset actually has.

This runs before any language model and resolves most questions on its own. The
contract already carries everything needed: each KPI's name, the semantic tags
discovery assigned it, the source fields its formula reads, and a prose
definition of what it means for this business. The concept library adds the
missing piece — that "readmitted", "readmission_count" and "readmissions" are
the same idea — so a question phrased in the reader's words still finds the
field the dataset happens to use.

The scoring itself lives in `agent/kpi_search.py` -- this module shapes a
question into clauses, delegates each one, and turns the ranked candidates into
the `KpiRef`s the intent models speak. Nothing here guesses: when two KPIs match
a phrase equally well both are returned, and the caller decides what to do about
it.

Dimension members are resolved the same way, against the catalogue in
`agent/dimensions.py`. That half of the function never worked until the
catalogue existed: it read an attribute nothing ever set.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple

from ..agent.contract_api import ContractAPI
from ..agent.dimensions import MAX_MEMBER_TOKENS
from ..agent.kpi_search import KpiSearch
from ..agent.text import contains_phrase, ngrams, normalise, tokens
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


@dataclass
class GroundingResult:
    outcome_candidates: List[KpiRef]
    comparison_candidates: List[KpiRef]
    dimension_hints: List[str]
    entity_filters: Dict[str, str]
    unmapped_terms: List[str]
    direction: str = ""            # "down" | "up" | ""
    note: str = ""


# The text primitives live in `app.agent.text`: the member index files values
# under a normalised key and the KPI search scores phrases against KPI names,
# and all three have to agree on every input or a match silently never fires.
_tokens = tokens
_ngrams = ngrams
_normalise = normalise
_contains_phrase = contains_phrase


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


def _match_clause(clause: str, search: KpiSearch, role: str) -> Tuple[List[KpiRef], Set[str]]:
    """
    Every KPI this clause could be referring to, best evidence first.

    The scoring lives in `agent/kpi_search.py`; this only re-shapes it into the
    `KpiRef` the intent models speak. Returns the candidates and the set of
    n-grams that matched something, so the caller can report the words that
    bound to nothing.
    """
    result = search.search(clause, limit=None)
    refs = [KpiRef(kpi_key=m.kpi_key, label=m.label, role=role,
                   match_basis=m.basis, confidence=m.score,
                   matched_text=m.matched_gram.replace("_", " "))
            for m in result.matches]
    return refs, set(result.matched_grams)


def ground_question(question: str, schema: Any) -> GroundingResult:
    """
    Resolve a question against the contract, without a language model.

    The outcome is whatever the main clause is about; anything named in a
    contrast clause ("even though CAC was up") becomes a comparison rather than
    the thing to explain. Dimension words are matched against the dataset's own
    dimensions, and their members against the values those columns hold.
    """
    search = KpiSearch(ContractAPI(schema))
    main, contrast = _split_contrast(question)

    outcome, matched_main = _match_clause(main, search, "outcome")
    if contrast.strip():
        comparison, matched_contrast = _match_clause(contrast, search, "comparison")
    else:
        comparison, matched_contrast = [], set()

    # A KPI named in both clauses belongs to the outcome; drop the duplicate.
    outcome_keys = {r.kpi_key for r in outcome}
    comparison = [r for r in comparison if r.kpi_key not in outcome_keys]
    # A contrast clause names one measure, not a family of them. "even though
    # admissions rose" is about admissions — cost-per-admission also matching,
    # through a shared source field, is noise rather than a second contrast.
    if comparison:
        ceiling = comparison[0].confidence
        comparison = [r for r in comparison if ceiling - r.confidence < 0.25]

    # KPI matching runs first and its grams are handed on: a word already spent
    # naming the measure must not also be read as a value to filter by.
    matched_kpi_grams = matched_main | matched_contrast
    dimension_hints, entity_filters, bound_grams = _match_dimensions(
        question, schema, matched_kpi_grams)

    all_tokens = set(_tokens(question))
    consumed = {t for gram in matched_kpi_grams for t in gram.split("_")}
    consumed |= {t for gram in bound_grams for t in gram.split("_")}
    unmapped = sorted(
        t for t in all_tokens
        if t not in _STOP and t not in consumed and t not in _DECLINE_WORDS
        and t not in _RISE_WORDS and t not in _FLAT_WORDS
        and not t.isdigit() and len(t) > 2
        and t not in {d.lower() for d in dimension_hints}
        and t not in {t2 for v in entity_filters.values() for t2 in _tokens(v)}
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


# Words that name a movement, a period or nothing at all. A dimension member
# that happens to be spelled like one of them is not evidence the question meant
# that member -- a segment called "Other" must not filter every question that
# says "other".
_UNBINDABLE = _STOP | _DECLINE_WORDS | _RISE_WORDS | _FLAT_WORDS


def _match_dimensions(question: str, schema: Any,
                      consumed_grams: Set[str] = frozenset()
                      ) -> Tuple[List[str], Dict[str, str], Set[str]]:
    """
    Dimension columns the question names, and the members it filters to.

    Until the member catalogue existed this read `schema.dimension_members`, an
    attribute nothing ever set, so the filter half never fired. It now walks the
    catalogue's index, longest phrase first, under three rules:

      * a phrase already spent on a KPI is not reconsidered as a member. The
        KPI is what the question is *about*, and a column whose values collide
        with a measure's name must not steal it.
      * a phrase that is a stopword or a movement word never binds.
      * a phrase held by two different columns binds neither. That is a real
        ambiguity, and picking one silently is the failure this closes.

    Returns the dimensions, the filters, and the grams that resolved to a
    member, so the caller does not report them as unmapped.
    """
    catalogue = getattr(schema, "member_catalogue", None)
    dims: List[str] = []
    filters: Dict[str, str] = {}
    bound_grams: Set[str] = set()

    tokens = _tokens(question)
    # Widen the window only as far as the widest value actually indexed, so a
    # dataset of one-word members pays nothing for one that has five.
    width = 4
    if catalogue is not None:
        width = max(4, min(catalogue.max_member_tokens, MAX_MEMBER_TOKENS))

    if catalogue is not None:
        # Longest first so "north campus" is considered before "north"; within
        # one width, lexicographic, so the reading never depends on gram order.
        for gram in sorted(_ngrams(tokens, max_n=width),
                           key=lambda g: (-(g.count("_") + 1), g)):
            if gram in consumed_grams or gram in _UNBINDABLE:
                continue
            matches = catalogue.lookup(gram)
            if len(matches) != 1:
                # Nothing, or an ambiguity no evidence here can settle. Either
                # way the phrase is accounted for rather than merely dropped.
                if matches:
                    bound_grams.add(gram)
                continue
            match = matches[0]
            if match.dimension in filters:
                continue
            filters[match.dimension] = match.member
            bound_grams.add(gram)
            if match.dimension not in dims:
                dims.append(match.dimension)

    grams = set(_ngrams(tokens, max_n=width))
    for dim in getattr(schema, "dimensions", []) or []:
        if dim in dims:
            continue
        dim_n = _normalise(dim)
        if dim_n in bound_grams or dim_n.rstrip("s") in bound_grams:
            continue
        if dim_n in grams or dim_n.rstrip("s") in grams:
            dims.append(dim)

    return dims, filters, bound_grams
