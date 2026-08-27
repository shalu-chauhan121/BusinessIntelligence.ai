"""
What a KPI is actually built from, read off the formula rather than guessed.

The contract already parses every formula into a typed AST (`kpi/resolver.py:95`
`parse_expression`), but nothing has ever read it *structurally*. Every consumer
instead asks whether one KPI's flat set of source fields is a subset of
another's -- `driver_graph.py:162-179`, and again in
`_derive_structural_depends_on` (`resolver.py:347`). Set containment throws away
three things the AST already knows, and each one produces a wrong answer rather
than a vague one:

    sign        In `(revenue - cost_of_goods) / revenue`, `cost_of_goods` is a
                subtractive term. Containment reports it as a numerator driver
                with no sign, so an explanation built on that edge has the
                direction of the effect backwards.

    dual role   `revenue` is in *both* halves of that same ratio. The loop at
                `driver_graph.py:162` tests the numerator first and `continue`s,
                so the denominator role is silently lost.

    transitive  A ratio whose numerator is another KPI (`{gamma}/{alpha}`) has
                referenced fields `{gamma, alpha}`. Anything one level down
                inside `gamma` is not in that set, so containment cannot reach
                it -- and it does not even find `gamma` itself, whose own fields
                `{alpha, beta}` are not a subset of `{gamma, alpha}` either.

This module answers the same question from the AST: which fields, with what
sign, in which position, how far down.

It knows nothing about any business domain. It walks `{field}` names and
arithmetic; whether a field means "beds" or "revenue" is settled at discovery
time by `kpi/library.py`, long before anything here runs. No term from those
libraries belongs in this file, and a test enforces that.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Optional, Set, Tuple

from ..kpi.resolver import (
    BinOp,
    Const,
    FieldRef,
    KpiResolutionError,
    Node,
    parse_expression,
)
from .contract_api import ContractAPI
from .errors import MalformedFormulaError

# How far expansion follows KPI-on-KPI references before giving up. Four is well
# past anything a real contract produces; the bound exists so a pathological
# contract cannot make traversal expensive, not because depth 5 is meaningless.
DEFAULT_MAX_DEPTH = 4

NUMERATOR = "numerator"
DENOMINATOR = "denominator"
EXPRESSION = "expression"

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _flip(position: str) -> str:
    return DENOMINATOR if position == NUMERATOR else NUMERATOR


@dataclass(frozen=True)
class Term:
    """One field reference reached in a formula, with how it got there."""

    field: str
    sign: int              # +1 / -1, from additive context
    position: str          # numerator | denominator, after any division
    role: str              # which AST it came from: numerator | denominator | expression
    depth: int             # 0 == written in this KPI's own formula
    via: Tuple[str, ...]   # KPI keys traversed to reach it


@dataclass(frozen=True)
class FormulaStructure:
    key: str
    kind: str                          # sum | mean | ratio
    scale: float
    terms: Tuple[Term, ...]
    constants: Tuple[float, ...]
    leaf_fields: Tuple[str, ...]       # what nothing expanded further into
    component_kpis: Tuple[str, ...]    # AST-derived; replaces containment `depends_on`
    # Every term whose field is itself a KPI, captured where it was encountered
    # rather than after substitution -- so a component that expansion replaced
    # still reports the position and sign it held in the formula.
    component_terms: Tuple[Term, ...]
    source: str                        # contract | seed | column
    truncated: bool                    # expansion hit max_depth or a cycle

    def terms_in(self, position: str) -> Tuple[Term, ...]:
        return tuple(t for t in self.terms if t.position == position)


# ---------------------------------------------------------------------------
# the walk
# ---------------------------------------------------------------------------
def _is_unary_minus(node: Any) -> bool:
    """`parse_expression` desugars `-x` into `BinOp('-', Const(0.0), x)`."""
    return (isinstance(node, BinOp) and node.op == "-"
            and isinstance(node.left, Const) and node.left.value == 0.0)


def _walk(node: Node, sign: int, position: str, role: str, depth: int,
          via: Tuple[str, ...], terms: List[Term], constants: List[float]) -> None:
    if isinstance(node, Const):
        constants.append(float(node.value))
        return
    if isinstance(node, FieldRef):
        terms.append(Term(field=node.name, sign=sign, position=position,
                          role=role, depth=depth, via=via))
        return
    if isinstance(node, BinOp):
        if _is_unary_minus(node):
            # The synthetic 0.0 is punctuation, not a constant anyone wrote.
            _walk(node.right, -sign, position, role, depth, via, terms, constants)
            return
        _walk(node.left, sign, position, role, depth, via, terms, constants)
        if node.op == "-":
            _walk(node.right, -sign, position, role, depth, via, terms, constants)
        elif node.op == "/":
            _walk(node.right, sign, _flip(position), role, depth, via, terms, constants)
        else:                                    # + and * both preserve context
            _walk(node.right, sign, position, role, depth, via, terms, constants)
        return
    raise MalformedFormulaError("<expression>", repr(node),
                                f"unsupported node {type(node).__name__}")


# ---------------------------------------------------------------------------
# reading a spec's ASTs -- contract or seed, one shape out
# ---------------------------------------------------------------------------
def _brace(expression: str) -> str:
    """
    Wrap bare identifiers so a seed expression parses with the contract grammar.

    The seed registry stores formulas as plain text: `numerator_expr` is
    `"revenue-cost_of_goods"`, not `"{revenue}-{cost_of_goods}"`
    (`metrics.py:40-51`). Rather than write a second grammar for it, brace the
    identifiers and hand the result to the same `parse_expression` the contract
    uses, so both paths produce structurally identical ASTs.
    """
    return _IDENTIFIER.sub(lambda m: "{" + m.group(0) + "}", expression or "")


def _parse(key: str, expression: str) -> Node:
    try:
        return parse_expression(expression)
    except KpiResolutionError as exc:
        raise MalformedFormulaError(key, expression, str(exc)) from exc


def _require(api: ContractAPI, key: str) -> Any:
    """
    The spec for `key`, or `None` when it is a bare measure column.

    `schema.available_kpis` can offer a raw column that has no `MetricSpec` at
    all -- `fulfilled_orders` on the retail sample. `metrics.compute` handles
    that case by summing the column directly (`metrics.py:343`), so it is a
    real KPI shape rather than an error, and introspection has to describe it
    too. A key that is neither a spec nor an offered column still raises.
    """
    if api.has(key):
        return api.require(key)
    if key in set(api.list_kpi_keys()):
        return None
    return api.require(key)          # raises UnknownKpiError, naming alternatives


def _asts(api: ContractAPI, key: str) -> Tuple[List[Tuple[Node, str]], str, float, str]:
    """`([(ast, role)], kind, scale, source)` for a contract, seed or column KPI."""
    spec = _require(api, key)
    if spec is None:
        # A bare column is its own formula.
        return [(FieldRef(key), EXPRESSION)], "sum", 1.0, "column"
    kind = getattr(spec, "kind", "sum") or "sum"
    scale = float(getattr(spec, "scale", 1.0) or 1.0)

    # A contract KPI arrives already parsed; nothing is re-parsed here.
    if getattr(spec, "expression_ast", None) is not None:
        return [(spec.expression_ast, EXPRESSION)], kind, scale, "contract"
    if getattr(spec, "numerator_ast", None) is not None:
        pair = [(spec.numerator_ast, NUMERATOR)]
        if getattr(spec, "denominator_ast", None) is not None:
            pair.append((spec.denominator_ast, DENOMINATOR))
        return pair, kind, scale, "contract"

    # Seed registry: plain strings, normalised through the same grammar.
    if kind == "ratio":
        num = getattr(spec, "numerator_expr", None) or getattr(spec, "numerator", None)
        den = getattr(spec, "denominator", None)
        if not num or not den:
            raise MalformedFormulaError(key, f"{num} / {den}",
                                        "a ratio without both halves declared")
        return ([(_parse(key, _brace(num)), NUMERATOR),
                 (_parse(key, _brace(den)), DENOMINATOR)], kind, scale, "seed")

    expression = getattr(spec, "numerator_expr", None)
    if not expression:
        requires = list(getattr(spec, "requires", None) or [])
        if len(requires) != 1:
            raise MalformedFormulaError(key, ", ".join(requires),
                                        "no expression and not exactly one source field")
        expression = requires[0]
    return [(_parse(key, _brace(expression)), EXPRESSION)], kind, scale, "seed"


# ---------------------------------------------------------------------------
# expansion
# ---------------------------------------------------------------------------
def _is_identity(api: ContractAPI, key: str) -> bool:
    """
    A KPI that is simply its own column, such as `revenue = {revenue}`.

    Expanding one yields itself, so it is treated as a leaf. Without this the
    visited-set would flag every such KPI as a cycle and mark the whole
    structure truncated.
    """
    try:
        asts, _, _, _ = _asts(api, key)
    except Exception:
        return True
    if len(asts) != 1:
        return False
    node, _ = asts[0]
    return isinstance(node, FieldRef) and node.name == key


def structure(api: ContractAPI, key: str, expand: bool = True,
              max_depth: int = DEFAULT_MAX_DEPTH) -> FormulaStructure:
    """
    The structure of `key`'s formula: every field it reaches, signed and placed.

    With `expand=True` a term that is itself a KPI is replaced by that KPI's own
    terms, carrying sign and position through the substitution, so a component
    two levels down is reported at `depth=1` instead of being invisible. That
    substitution is the whole difference from field-set containment.
    """
    _require(api, key)
    known = set(api.list_kpi_keys())

    seed_terms: List[Term] = []
    constants: List[float] = []
    components: List[str] = []
    component_terms: List[Term] = []
    truncated = False

    asts, kind, scale, source = _asts(api, key)
    for node, role in asts:
        start = role if role in (NUMERATOR, DENOMINATOR) else NUMERATOR
        _walk(node, 1, start, role, 0, (), seed_terms, constants)

    if not expand:
        return FormulaStructure(
            key=key, kind=kind, scale=scale, terms=tuple(seed_terms),
            constants=tuple(constants),
            leaf_fields=tuple(dict.fromkeys(t.field for t in seed_terms)),
            component_kpis=tuple(dict.fromkeys(
                t.field for t in seed_terms if t.field in known and t.field != key)),
            component_terms=tuple(t for t in seed_terms
                                  if t.field in known and t.field != key),
            source=source, truncated=False)

    # Breadth-first substitution. `via` carries the path, so a branch that would
    # revisit a KPI already on its own path stops and flags truncation rather
    # than recursing forever.
    final: List[Term] = []
    frontier = list(seed_terms)

    while frontier:
        term = frontier.pop(0)
        field = term.field
        if field in known and field != key:
            components.append(field)
            component_terms.append(term)

        if field == key and term.depth > 0:
            # Expansion came back around to the KPI it started from: a cycle.
            truncated = True
            final.append(term)
            continue
        if field not in known or field == key or _is_identity(api, field):
            final.append(term)
            continue
        if term.depth >= max_depth or field in term.via:
            truncated = True
            final.append(term)
            continue

        try:
            inner_asts, _, _, _ = _asts(api, field)
        except MalformedFormulaError:
            truncated = True
            final.append(term)
            continue

        inner: List[Term] = []
        for node, role in inner_asts:
            start = role if role in (NUMERATOR, DENOMINATOR) else NUMERATOR
            _walk(node, 1, start, role, 0, (), inner, constants)

        for sub in inner:
            # Substituting under a denominator inverts: dividing by (a/b) is
            # multiplying by b, so the inner position flips with the outer one.
            position = sub.position if term.position == NUMERATOR else _flip(sub.position)
            frontier.append(Term(
                field=sub.field,
                sign=term.sign * sub.sign,
                position=position,
                role=term.role,
                depth=term.depth + 1,
                via=term.via + (field,),
            ))

    return FormulaStructure(
        key=key, kind=kind, scale=scale, terms=tuple(final),
        constants=tuple(constants),
        leaf_fields=tuple(dict.fromkeys(t.field for t in final)),
        component_kpis=tuple(dict.fromkeys(components)),
        component_terms=tuple(component_terms),
        source=source, truncated=truncated)


def roles_of(struct: FormulaStructure, field: str) -> Set[str]:
    """
    Every position `field` occupies in this formula.

    Returns both when a field sits in the numerator *and* the denominator --
    the case `driver_graph.py:162` loses by testing the numerator first and
    moving on.
    """
    return {t.position for t in struct.terms if t.field == field}


def signs_of(struct: FormulaStructure, field: str) -> Set[int]:
    """Every sign `field` carries. `{-1}` means a purely subtractive term."""
    return {t.sign for t in struct.terms if t.field == field}


# ---------------------------------------------------------------------------
# what compilation dropped
# ---------------------------------------------------------------------------
def unparsable_kpis(contract: Any) -> List[Tuple[str, str]]:
    """
    Every KPI whose formula `compile_contract` silently skipped, and why.

    `compile_contract` swallows `KpiResolutionError` per entry
    (`resolver.py:397-400`) so one bad formula cannot make a whole contract
    unusable. That is right for compilation and invisible to everyone
    downstream: the KPI simply is not there, and nothing records why. This
    re-parses each definition so the omission can be reported.
    """
    out: List[Tuple[str, str]] = []
    for definition in (getattr(contract, "kpis", None) or []):
        formula = getattr(definition, "formula", None)
        if formula is None:
            continue
        key = getattr(definition, "kpi_id", "")
        expressions = ([formula.numerator_expression, formula.denominator_expression]
                       if formula.kind == "ratio" else [formula.expression])
        for expression in expressions:
            if not expression or not str(expression).strip():
                out.append((key, "the formula declares no expression"))
                break
            try:
                parse_expression(expression)
            except KpiResolutionError as exc:
                out.append((key, str(exc)))
                break
    return out
