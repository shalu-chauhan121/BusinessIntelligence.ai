"""
Which KPI a phrase is referring to, ranked by how good the evidence is.

This is `query/grounding.py`'s matching ladder, moved down a layer and turned
from a step inside one function into a service anything can call. The scoring is
unchanged: `grounding._match_clause` now delegates here, so there is exactly one
implementation and the two cannot drift.

What changes is what a tie *means*. `understanding._resolve_outcome` treated two
KPIs scoring within `TIE_MARGIN` as an impasse and refused to answer, which is
the "please pick a KPI" wall. Nothing here refuses: `search` returns the ranked
list with each candidate's score and the basis it was scored on, and the caller
decides. A tie is two rows with close scores, not an exception.

The ladder is ordered by how much each kind of evidence is worth:

    1.00  the phrase is exactly the KPI's key or label
    0.85  a multi-word phrase appears whole-word inside the key or label
    0.80  the phrase is one of the KPI's semantic tags
    0.70  the phrase is a concept alias, and the KPI reads a field carrying it
    0.65  the phrase is a concept alias naming the KPI itself
    0.55  a single word appears whole-word inside the key or label
    0.35  a multi-word phrase appears in the prose definition
    0.30  a multi-word phrase appears in the relevance text

A single word scores far below a phrase on purpose: "rate" must not claim every
rate KPI. The bottom two tiers sit below the caller's confidence floor, so prose
overlap alone can never resolve a question -- it only separates near-ties and
records that the words were understood.

Everything reaches the contract through `ContractAPI`. Nothing here touches
`schema.contract_resolver`.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional, Set, Tuple

from .contract_api import ContractAPI
from .text import contains_phrase, ngrams, normalise, tokens

# Score, and the name the caller reports it under. `MatchBasis` in
# `models/investigation.py` is the same vocabulary.
SCORE_EXACT_NAME = 1.0
SCORE_PHRASE_IN_NAME = 0.85
SCORE_SEMANTIC_TAG = 0.8
SCORE_ALIAS_VIA_FIELD = 0.7
SCORE_ALIAS_VIA_NAME = 0.65
SCORE_WORD_IN_NAME = 0.55
SCORE_DEFINITION = 0.35
SCORE_RELEVANCE = 0.3

# Below this many characters a fragment is noise rather than a name.
MIN_NAME_FRAGMENT = 3
MIN_PROSE_FRAGMENT = 6

DEFAULT_LIMIT = 8


@dataclass(frozen=True)
class SearchEntry:
    """Everything known about one KPI, flattened and normalised once."""

    key: str
    label: str
    key_normalised: str
    label_normalised: str
    tags: Tuple[str, ...]
    source_fields: Tuple[str, ...]
    definition: str
    relevance: str


@dataclass(frozen=True)
class KpiMatch:
    """One KPI a phrase could mean, and the evidence that says so."""

    kpi_key: str
    label: str
    score: float
    basis: str
    matched_gram: str    # underscore-joined; a space here would mean a sentence


@dataclass(frozen=True)
class SearchResult:
    matches: Tuple[KpiMatch, ...]
    matched_grams: Tuple[str, ...]


@lru_cache(maxsize=1)
def concept_alias_index() -> Dict[str, str]:
    """
    Every concept alias in the library, mapped to the concept it implies.

    Built from the library's own concept refs so the two can never drift apart.
    Cached: the library is a static module, and this was previously rebuilt by
    walking `dir()` over it on every single question.
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
            index.setdefault(normalise(alias), concept)
            index.setdefault(alias.lower(), concept)
    return index


class KpiSearch:
    """
    Ranked KPI candidates for a phrase. Never blocks, never chooses.

    Built once per schema and reused across clauses: normalising each KPI's key
    and label happens here, at construction, rather than once per phrase per KPI
    inside the scoring loop.
    """

    def __init__(self, api: ContractAPI) -> None:
        self._api = api
        self._entries: Optional[Tuple[SearchEntry, ...]] = None

    def entries(self) -> Tuple[SearchEntry, ...]:
        if self._entries is None:
            self._entries = tuple(self._build(key) for key in self._api.list_kpi_keys())
        return self._entries

    def _build(self, key: str) -> SearchEntry:
        definition = self._api.get_definition(key)
        # The contract's own name wins over the compiled spec's label, which is
        # the order `grounding` has always used to pick what a KPI is called.
        label = (getattr(definition, "name", "") or self._api.label(key) or key)
        return SearchEntry(
            key=key,
            label=label,
            key_normalised=normalise(key),
            label_normalised=normalise(label),
            tags=tuple(t.lower() for t in self._api.tags(key)),
            source_fields=tuple(f.lower() for f in self._api.source_fields(key)),
            definition=self._api.business_definition(key).lower(),
            relevance=self._api.relevance(key).lower(),
        )

    def search(self, text: str, *, limit: int = DEFAULT_LIMIT,
               min_score: float = 0.0) -> SearchResult:
        """
        Every KPI this text could be referring to, best evidence first.

        Also returns the phrases that matched something, so the caller can
        report the words that bound to nothing.
        """
        space = self.entries()
        aliases = concept_alias_index()
        grams = ngrams(tokens(text))
        matched_text: Set[str] = set()
        scored: Dict[str, Tuple[float, str, str]] = {}   # key -> (score, basis, gram)

        def offer(key: str, score: float, basis: str, gram: str) -> None:
            previous = scored.get(key)
            if previous is None or score > previous[0]:
                scored[key] = (score, basis, gram)

        for gram in grams:
            for entry in space:
                if gram == entry.key_normalised or gram == entry.label_normalised:
                    offer(entry.key, SCORE_EXACT_NAME, "exact_name", gram)
                    matched_text.add(gram)
                    continue
                # A multi-word gram contained in the label is still strong
                # evidence ("bed occupancy" for "Bed occupancy rate"), a single
                # word much less so ("rate" must not claim every rate KPI).
                if len(gram) > MIN_NAME_FRAGMENT and (
                        contains_phrase(entry.label_normalised, gram)
                        or contains_phrase(entry.key_normalised, gram)):
                    weight = SCORE_PHRASE_IN_NAME if "_" in gram else SCORE_WORD_IN_NAME
                    offer(entry.key, weight, "exact_name", gram)
                    matched_text.add(gram)
                    continue
                if gram in entry.tags:
                    offer(entry.key, SCORE_SEMANTIC_TAG, "semantic_tag", gram)
                    matched_text.add(gram)
                    continue

                concept = aliases.get(gram)
                if concept:
                    # The alias names a concept; bind it to a KPI only when one
                    # of that KPI's own source fields carries the same alias or
                    # concept.
                    for field in entry.source_fields:
                        field_n = normalise(field)
                        if field_n == gram or aliases.get(field_n) == concept:
                            offer(entry.key, SCORE_ALIAS_VIA_FIELD, "concept_alias", gram)
                            matched_text.add(gram)
                            break
                    else:
                        if normalise(concept) in (entry.key_normalised,
                                                  entry.label_normalised):
                            offer(entry.key, SCORE_ALIAS_VIA_NAME, "concept_alias", gram)
                            matched_text.add(gram)
                    continue

                if "_" in gram and len(gram) > MIN_PROSE_FRAGMENT:
                    if contains_phrase(entry.definition, gram):
                        offer(entry.key, SCORE_DEFINITION, "business_definition", gram)
                        matched_text.add(gram)
                    elif contains_phrase(entry.relevance, gram):
                        # Populated by the contract and, until now, never
                        # scored against. Weaker than the definition: relevance
                        # prose says why a KPI matters, which is poorer evidence
                        # about which KPI was named.
                        offer(entry.key, SCORE_RELEVANCE, "relevance_text", gram)
                        matched_text.add(gram)

        labels = {e.key: e.label for e in space}
        matches = [
            KpiMatch(kpi_key=key, label=labels[key], score=round(score, 2),
                     basis=basis, matched_gram=gram)
            for key, (score, basis, gram) in scored.items()
            if score >= min_score
        ]
        matches.sort(key=lambda m: (-m.score, m.kpi_key))
        if limit is not None and limit >= 0:
            matches = matches[:limit]
        return SearchResult(matches=tuple(matches),
                            matched_grams=tuple(sorted(matched_text)))
