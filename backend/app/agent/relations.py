"""
How one KPI relates to another, with the formula edges read from the AST.

This is the same question `engines/driver_graph.py` answers, and it reuses that
module's edge weights and its non-formula tests unchanged -- shared source
fields, recorded derivations, semantic tags, declared non-comparability. There
is exactly one difference, and it is the reason this module exists: **formula
edges come from `agent/formula.py`'s AST walk, never from field-set
containment.**

On the fixture `chi = {gamma}/{alpha}` where `gamma = {alpha}-{beta}`,
`build_driver_graph` returns a single edge -- `alpha -> chi`. It misses `beta`,
which is a real driver one level down, and it misses `gamma`, the *direct
numerator*, because `gamma`'s own fields `{alpha, beta}` are not a subset of
`chi`'s `{gamma, alpha}`. Containment cannot see either. The AST finds both, and
carries the sign that says `beta` moves `chi` in the opposite direction.

Two rules establish a formula relation, and both are exact rather than
heuristic:

    reference    the other KPI's key literally appears as a field reference in
                 this KPI's formula, at any depth after expansion

    equivalence  the other KPI's signed terms are exactly one half of this
                 KPI's formula. `gross_profit` is `revenue - cost_of_goods`;
                 `gross_margin_pct`'s numerator is `revenue - cost_of_goods`;
                 the two signed term-sets are equal, so `gross_profit` *is*
                 that numerator even though it is never named in the formula.

Deliberately absent: any `note` / `verification_note` field. `DriverEdge`
carries prose (`models/investigation.py:145-147`) that is assembled from
f-strings and then read back out by the narrative layer. Tools in the agent
layer return numbers and facts; the model writes the sentences. Keeping the
field off the dataclass is what stops it coming back.

Like `formula.py`, this module contains no business vocabulary. It reasons over
`{field}` names and arithmetic only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ..engines.driver_graph import (
    MIN_FIELD_OVERLAP,
    WEIGHT_DERIVATION,
    WEIGHT_FORMULA,
    WEIGHT_SEMANTIC_TAG,
    WEIGHT_SHARED_FIELDS,
    _derivation_related,
    _jaccard,
    _tags,
)
from . import formula as F
from .contract_api import ContractAPI

# A component reached through another KPI is a weaker claim than one written in
# the formula directly -- real, but one substitution further from the evidence.
DEPTH_DECAY = 0.9

FORMULA_RELATIONS = ("formula_numerator", "formula_denominator", "formula_term")


@dataclass(frozen=True)
class Relation:
    """One directed relationship, `source_kpi` -> `target_kpi`."""

    source_kpi: str
    target_kpi: str
    relation: str
    weight: float
    evidence_fields: Tuple[str, ...] = ()
    sign: int = 0        # +1 / -1 for formula relations; 0 where not meaningful
    depth: int = 0
    origin: str = "contract"


def _relation_name(kind: str, position: str) -> str:
    """A numerator only means something when the KPI is actually a ratio."""
    if kind != "ratio":
        return "formula_term"
    return ("formula_numerator" if position == F.NUMERATOR
            else "formula_denominator")


def _signed_terms(struct: F.FormulaStructure, position: Optional[str] = None
                  ) -> Set[Tuple[str, int]]:
    return {(t.field, t.sign) for t in struct.terms
            if position is None or t.position == position}


class RelationGraph:
    """
    Traversal over one dataset's KPI relationships.

    Built on `ContractAPI`, so the resolver is bound once and no method can
    compute against the wrong one. Works for contract-backed and contract-less
    datasets alike; `Relation.origin` reports which.
    """

    def __init__(self, api: ContractAPI):
        self._api = api
        self._keys = list(api.list_kpi_keys())
        self._structures: Dict[str, Optional[F.FormulaStructure]] = {}

    # -- structure cache -----------------------------------------------------
    def _structure(self, key: str) -> Optional[F.FormulaStructure]:
        """`None` for a KPI whose formula will not parse -- never an exception,
        because one unreadable KPI must not make the graph unusable."""
        if key not in self._structures:
            try:
                self._structures[key] = F.structure(self._api, key)
            except Exception:
                self._structures[key] = None
        return self._structures[key]

    def _spec(self, key: str) -> Any:
        try:
            return self._api.require(key)
        except Exception:
            return None

    # -- formula relations ---------------------------------------------------
    def components(self, key: str) -> List[Relation]:
        """
        The KPIs this KPI's formula is built from, signed and placed.

        Covers both the reference rule and the equivalence rule described in the
        module docstring; a KPI satisfying both is reported once per distinct
        position, so a field in numerator *and* denominator yields two relations
        rather than the first one found.
        """
        struct = self._structure(key)
        if struct is None:
            return []

        found: Dict[Tuple[str, str], Relation] = {}

        def offer(other: str, position: str, sign: int, depth: int,
                  fields: Sequence[str]) -> None:
            slot = (other, position)
            previous = found.get(slot)
            if previous is not None and previous.depth <= depth:
                return
            found[slot] = Relation(
                source_kpi=other, target_kpi=key,
                relation=_relation_name(struct.kind, position),
                weight=round(WEIGHT_FORMULA * (DEPTH_DECAY ** depth), 3),
                evidence_fields=tuple(sorted(set(fields))),
                sign=sign, depth=depth, origin=struct.source)

        # Rule 1 -- the other KPI is named in the formula. `component_terms`
        # records each one where it was encountered, so a KPI that expansion
        # replaced still reports the position and sign it held: `gamma` is
        # `chi`'s numerator even though only `alpha` and `beta` survive in
        # `terms`.
        for term in struct.component_terms:
            other_struct = self._structure(term.field)
            fields = other_struct.leaf_fields if other_struct else (term.field,)
            offer(term.field, term.position, term.sign, term.depth, fields)

        # Rule 2 -- the other KPI's signed terms are exactly one half of this one.
        for other in self._keys:
            if other == key:
                continue
            other_struct = self._structure(other)
            if other_struct is None or not other_struct.terms:
                continue
            theirs = _signed_terms(other_struct)
            for position in (F.NUMERATOR, F.DENOMINATOR):
                mine = _signed_terms(struct, position)
                if mine and mine == theirs and (other, position) not in found:
                    offer(other, position, 1, 0, other_struct.leaf_fields)

        return sorted(found.values(),
                      key=lambda r: (r.depth, -r.weight, r.source_kpi, r.relation))

    # -- everything else -----------------------------------------------------
    def _other_relations(self, key: str, already: Set[str]) -> List[Relation]:
        """Derivation, shared fields and semantic tags -- `driver_graph`'s own
        tests, applied to C3's expanded leaf fields."""
        out: List[Relation] = []
        struct = self._structure(key)
        spec = self._spec(key)
        if spec is None:
            return out
        mine = {f.lower() for f in (struct.leaf_fields if struct else ())}
        my_tags = _tags(spec)
        origin = struct.source if struct else "contract"

        for other in self._keys:
            if other == key or other in already:
                continue
            other_spec = self._spec(other)
            other_struct = self._structure(other)
            if other_spec is None:
                continue
            theirs = {f.lower() for f in (other_struct.leaf_fields if other_struct else ())}
            if not theirs:
                continue

            if _derivation_related(spec, other_spec):
                out.append(Relation(source_kpi=other, target_kpi=key,
                                    relation="derivation", weight=WEIGHT_DERIVATION,
                                    evidence_fields=tuple(sorted(theirs & mine)),
                                    origin=origin))
                continue

            overlap = _jaccard(mine, theirs)
            if overlap >= MIN_FIELD_OVERLAP:
                out.append(Relation(
                    source_kpi=other, target_kpi=key, relation="shared_source_field",
                    weight=round(WEIGHT_SHARED_FIELDS * overlap, 3),
                    evidence_fields=tuple(sorted(mine & theirs)), origin=origin))
                continue

            shared = my_tags & _tags(other_spec)
            if shared:
                out.append(Relation(source_kpi=other, target_kpi=key,
                                    relation="semantic_tag", weight=WEIGHT_SEMANTIC_TAG,
                                    origin=origin))

        out.extend(self._not_comparable(key, origin))
        return out

    def _not_comparable(self, key: str, origin: str) -> List[Relation]:
        """
        Pairs the contract says must not be compared, at zero weight.

        Same rule as `driver_graph._not_comparable_edges`, rebuilt here only
        because that helper returns a `DriverEdge` carrying prose.
        """
        definition = self._api.get_definition(key)
        available = set(self._keys)
        out: List[Relation] = []
        for entry in (getattr(definition, "comparability", None) or []):
            other = getattr(entry, "other_kpi_id", None) or getattr(entry, "kpi_id", None)
            if not other or getattr(entry, "comparable", True) or other not in available:
                continue
            out.append(Relation(source_kpi=other, target_kpi=key,
                                relation="not_comparable", weight=0.0, origin=origin))
        return out

    # -- public traversal ----------------------------------------------------
    def related(self, key: str, relations: Optional[Sequence[str]] = None,
                depth: int = 1) -> List[Relation]:
        """
        Everything related to `key`, best evidence first.

        `depth` walks the component graph: at 2, the components of this KPI's
        components are included, each carrying the depth at which it was
        reached. Traversal is guarded by a visited set, so a cyclic contract
        terminates instead of hanging.
        """
        self._api.require(key)
        wanted = set(relations) if relations else None

        collected: Dict[Tuple[str, str], Relation] = {}
        visited: Set[str] = set()
        frontier: List[Tuple[str, int]] = [(key, 0)]

        while frontier:
            current, hop = frontier.pop(0)
            if current in visited:
                continue
            visited.add(current)

            for relation in self.components(current):
                stepped = relation if current == key else Relation(
                    source_kpi=relation.source_kpi, target_kpi=key,
                    relation=relation.relation,
                    weight=round(relation.weight * (DEPTH_DECAY ** hop), 3),
                    evidence_fields=relation.evidence_fields, sign=relation.sign,
                    depth=relation.depth + hop, origin=relation.origin)
                slot = (stepped.source_kpi, stepped.relation)
                if slot not in collected and stepped.source_kpi != key:
                    collected[slot] = stepped
                if hop + 1 < depth:
                    frontier.append((relation.source_kpi, hop + 1))

        formula_sources = {s for s, _ in collected}
        for relation in self._other_relations(key, formula_sources):
            collected.setdefault((relation.source_kpi, relation.relation), relation)

        out = [r for r in collected.values() if wanted is None or r.relation in wanted]
        out.sort(key=lambda r: (-r.weight, r.depth, r.source_kpi, r.relation))
        return out

    def neighbours(self, key: str, depth: int = 1) -> List[str]:
        """The distinct KPIs related to `key`, in the order `related` ranks them."""
        seen: List[str] = []
        for relation in self.related(key, depth=depth):
            if relation.source_kpi not in seen:
                seen.append(relation.source_kpi)
        return seen
