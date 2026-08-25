"""
Compiling a KPI Contract into something that can compute.

The contract stores formulas as text. Text is not executed here: it is parsed
into a small typed AST over an allowlist of column names and evaluated with
pandas. There is no `eval`, no `exec` and no name lookup that can escape the
frame, because a KPI definition is user-supplied data and the API accepts
arbitrary formula strings from the KPI Studio.

`CompiledKpi` deliberately exposes exactly the attribute surface the legacy
`MetricSpec` had — `key, label, unit, kind, scale, higher_is_better, additive` —
so the analysis engines swap `METRICS.get(k)` for `resolver.get(k)` and need no
other change.

Aggregation semantics, which are the whole point of getting this right:

    kind="sum"    the expression is evaluated PER ROW and summed
    kind="mean"   the expression is evaluated PER ROW and averaged (a level)
    kind="ratio"  numerator and denominator are each evaluated per row and
                  summed, and only then divided

The ratio case is what keeps a rate correct under aggregation. Averaging four
weekly margins is not the quarterly margin; summing the components and dividing
once is. The legacy code got this right by construction, and the contract now
states it as a declared rule instead of leaving it to be re-derived.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .contract import FilterSpec, KpiContract, KpiDefinition


class KpiResolutionError(ValueError):
    """A contract entry cannot be compiled or computed against this dataset."""


# ---------------------------------------------------------------------------
# expression AST
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FieldRef:
    name: str


@dataclass(frozen=True)
class Const:
    value: float


@dataclass(frozen=True)
class BinOp:
    op: str                    # + - * /
    left: Any
    right: Any


Node = Any

_TOKEN = re.compile(r"""
    \s*(?:
        (?P<field>\{[A-Za-z_][A-Za-z0-9_ .\-]*\})
      | (?P<number>\d+(?:\.\d+)?)
      | (?P<op>[+\-*/()])
    )
""", re.VERBOSE)


def _tokenise(expression: str) -> List[Tuple[str, str]]:
    tokens: List[Tuple[str, str]] = []
    pos = 0
    while pos < len(expression):
        if expression[pos].isspace():
            pos += 1
            continue
        m = _TOKEN.match(expression, pos)
        if not m or m.end() == pos:
            raise KpiResolutionError(
                f"Could not parse the formula at position {pos}: {expression!r}. "
                "Formulas may use {field_name}, numbers, + - * / and parentheses."
            )
        kind = m.lastgroup or ""
        value = (m.group(kind) or "").strip()
        tokens.append((kind, value))
        pos = m.end()
    return tokens


def parse_expression(expression: str) -> Node:
    """
    Parse a restricted arithmetic grammar into an AST.

        expr   := term (('+' | '-') term)*
        term   := factor (('*' | '/') factor)*
        factor := '{' name '}' | number | '(' expr ')' | '-' factor
    """
    if not expression or not expression.strip():
        raise KpiResolutionError("Formula is empty.")
    tokens = _tokenise(expression)
    pos = 0

    def peek() -> Optional[Tuple[str, str]]:
        return tokens[pos] if pos < len(tokens) else None

    def eat(value: Optional[str] = None) -> Tuple[str, str]:
        nonlocal pos
        tok = peek()
        if tok is None:
            raise KpiResolutionError(f"Formula ended unexpectedly: {expression!r}")
        if value is not None and tok[1] != value:
            raise KpiResolutionError(f"Expected {value!r} in formula {expression!r}")
        pos += 1
        return tok

    def parse_factor() -> Node:
        tok = peek()
        if tok is None:
            raise KpiResolutionError(f"Formula ended unexpectedly: {expression!r}")
        kind, value = tok
        if kind == "op" and value == "-":
            eat()
            return BinOp("-", Const(0.0), parse_factor())
        if kind == "op" and value == "(":
            eat()
            inner = parse_expr()
            eat(")")
            return inner
        if kind == "field":
            eat()
            return FieldRef(value[1:-1].strip())
        if kind == "number":
            eat()
            return Const(float(value))
        raise KpiResolutionError(f"Unexpected {value!r} in formula {expression!r}")

    def parse_term() -> Node:
        node = parse_factor()
        while True:
            tok = peek()
            if tok and tok[0] == "op" and tok[1] in ("*", "/"):
                eat()
                node = BinOp(tok[1], node, parse_factor())
            else:
                return node

    def parse_expr() -> Node:
        node = parse_term()
        while True:
            tok = peek()
            if tok and tok[0] == "op" and tok[1] in ("+", "-"):
                eat()
                node = BinOp(tok[1], node, parse_term())
            else:
                return node

    tree = parse_expr()
    if pos != len(tokens):
        raise KpiResolutionError(f"Trailing input in formula {expression!r}")
    return tree


def referenced_fields(node: Node) -> Set[str]:
    if isinstance(node, FieldRef):
        return {node.name}
    if isinstance(node, BinOp):
        return referenced_fields(node.left) | referenced_fields(node.right)
    return set()


def evaluate(node: Node, df: pd.DataFrame) -> pd.Series:
    """Evaluate the AST per row, returning a float Series aligned to `df`."""
    if isinstance(node, Const):
        return pd.Series(node.value, index=df.index, dtype="float64")
    if isinstance(node, FieldRef):
        if node.name not in df.columns:
            raise KpiResolutionError(
                f"The dataset has no column '{node.name}' required by this KPI."
            )
        return pd.to_numeric(df[node.name], errors="coerce").astype("float64")
    if isinstance(node, BinOp):
        left, right = evaluate(node.left, df), evaluate(node.right, df)
        if node.op == "+":
            return left + right
        if node.op == "-":
            return left - right
        if node.op == "*":
            return left * right
        if node.op == "/":
            return left.divide(right).replace([np.inf, -np.inf], np.nan)
    raise KpiResolutionError(f"Unsupported node in formula: {node!r}")


# ---------------------------------------------------------------------------
# filters
# ---------------------------------------------------------------------------
def apply_filters(df: pd.DataFrame, filters: Sequence[FilterSpec]) -> pd.DataFrame:
    """Row filters declared on the KPI, applied before any aggregation."""
    out = df
    for f in filters:
        if f.field not in out.columns:
            raise KpiResolutionError(
                f"Filter references column '{f.field}', which this dataset does not have."
            )
        col = out[f.field]
        if f.op == "eq":
            mask = col == f.value
        elif f.op == "ne":
            mask = col != f.value
        elif f.op == "in":
            mask = col.isin(list(f.value or []))
        elif f.op == "not_in":
            mask = ~col.isin(list(f.value or []))
        elif f.op == "gt":
            mask = pd.to_numeric(col, errors="coerce") > float(f.value)
        elif f.op == "gte":
            mask = pd.to_numeric(col, errors="coerce") >= float(f.value)
        elif f.op == "lt":
            mask = pd.to_numeric(col, errors="coerce") < float(f.value)
        elif f.op == "lte":
            mask = pd.to_numeric(col, errors="coerce") <= float(f.value)
        elif f.op == "is_null":
            mask = col.isna()
        elif f.op == "not_null":
            mask = col.notna()
        else:
            raise KpiResolutionError(f"Unknown filter operator '{f.op}'.")
        out = out[mask]
    return out


# ---------------------------------------------------------------------------
# the compiled KPI
# ---------------------------------------------------------------------------
@dataclass
class CompiledKpi:
    """
    An executable KPI. The first six attributes mirror the legacy `MetricSpec`
    so the analysis engines are indifferent to which one they were handed.
    """

    key: str
    label: str
    unit: str
    kind: str                       # sum | mean | ratio
    scale: float = 1.0
    higher_is_better: bool = True
    description: str = ""

    expression_ast: Optional[Node] = None
    numerator_ast: Optional[Node] = None
    denominator_ast: Optional[Node] = None
    filters: List[FilterSpec] = field(default_factory=list)

    granularity_label: str = ""
    rollup_policy: str = "sum"
    source_fields: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)
    status: str = "approved"
    definition: Optional[KpiDefinition] = None

    @property
    def additive(self) -> bool:
        return self.kind == "sum"

    # -- computation -------------------------------------------------------
    def _frame(self, df: pd.DataFrame) -> pd.DataFrame:
        return apply_filters(df, self.filters) if self.filters else df

    def compute(self, df: pd.DataFrame) -> float:
        if df is None or len(df) == 0:
            return float("nan")
        frame = self._frame(df)
        if len(frame) == 0:
            return float("nan")

        if self.kind == "sum":
            return float(evaluate(self.expression_ast, frame).sum())
        if self.kind == "mean":
            return float(evaluate(self.expression_ast, frame).mean())

        num, den = self.components(frame)
        if den in (None, 0) or den != den:
            return float("nan")
        return num / den * self.scale

    def components(self, df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
        """(numerator, denominator) for ratio KPIs — needed by rate/mix decomposition."""
        if self.kind != "ratio":
            return None, None
        frame = self._frame(df)
        if len(frame) == 0:
            return None, None
        num = float(evaluate(self.numerator_ast, frame).sum())
        den = float(evaluate(self.denominator_ast, frame).sum())
        return num, den


# ---------------------------------------------------------------------------
# compilation
# ---------------------------------------------------------------------------
def compile_kpi(definition: KpiDefinition) -> CompiledKpi:
    formula = definition.formula
    expression_ast = numerator_ast = denominator_ast = None

    if formula.kind == "ratio":
        if not formula.numerator_expression or not formula.denominator_expression:
            raise KpiResolutionError(
                f"KPI '{definition.kpi_id}' is a ratio but does not declare both a "
                "numerator and a denominator."
            )
        numerator_ast = parse_expression(formula.numerator_expression)
        denominator_ast = parse_expression(formula.denominator_expression)
        fields = referenced_fields(numerator_ast) | referenced_fields(denominator_ast)
    else:
        if not formula.expression:
            raise KpiResolutionError(f"KPI '{definition.kpi_id}' has no formula expression.")
        expression_ast = parse_expression(formula.expression)
        fields = referenced_fields(expression_ast)

    return CompiledKpi(
        key=definition.kpi_id,
        label=definition.name,
        unit=definition.unit,
        kind=formula.kind,
        scale=formula.scale,
        higher_is_better=definition.higher_is_better,
        description=definition.business_definition,
        expression_ast=expression_ast,
        numerator_ast=numerator_ast,
        denominator_ast=denominator_ast,
        filters=list(definition.filters),
        granularity_label=definition.granularity.label,
        rollup_policy=definition.aggregation.rollup_policy,
        source_fields=sorted(fields),
        depends_on=list(definition.depends_on),
        status=definition.status,
        definition=definition,
    )


def _derive_structural_depends_on(compiled: Dict[str, CompiledKpi]) -> None:
    """
    Fill in each KPI's structural dependencies, from its own formula.

    `KpiDefinition.depends_on` is hand-authored and, in practice, always empty —
    nothing in discovery populates it, so nothing downstream has ever been able
    to read a real KPI relationship off the contract. But the formula already
    says which other KPIs a ratio or additive KPI is built from: if another
    KPI's fields are entirely contained in this one's numerator, denominator or
    expression, that KPI is structurally a dependency whether or not anyone
    declared it by hand. Declared dependencies are kept; structural ones are
    added on top, never removed.
    """
    field_sets: Dict[str, Set[str]] = {}
    for key, compiled_kpi in compiled.items():
        fields: Set[str] = set(compiled_kpi.source_fields)
        for node in (compiled_kpi.expression_ast, compiled_kpi.numerator_ast,
                    compiled_kpi.denominator_ast):
            if node is not None:
                fields |= referenced_fields(node)
        field_sets[key] = fields

    for key, compiled_kpi in compiled.items():
        own_fields = field_sets[key]
        if not own_fields:
            continue
        structural = {
            other for other, other_fields in field_sets.items()
            if other != key and other_fields and other_fields <= own_fields
        }
        if structural:
            compiled_kpi.depends_on = sorted(set(compiled_kpi.depends_on) | structural)


def compile_contract(contract: KpiContract,
                     include_unapproved: bool = False) -> Dict[str, CompiledKpi]:
    """
    Compile a contract into the resolver the engines consume.

    Only approved KPIs are authoritative, which is what makes the review step
    meaningful. A provisional bootstrap contract marks its entries approved so
    that a user who has never opened the KPI Studio still gets a working
    product — the status is what carries the distinction, not the compilation.
    """
    out: Dict[str, CompiledKpi] = {}
    for definition in contract.kpis:
        if not include_unapproved and definition.status != "approved":
            continue
        try:
            out[definition.kpi_id] = compile_kpi(definition)
        except KpiResolutionError:
            # A single malformed entry must never make the whole contract
            # unusable; it simply does not resolve and the KPI is unavailable.
            continue
    _derive_structural_depends_on(out)
    return out


def available_keys(resolver: Dict[str, CompiledKpi], df: pd.DataFrame) -> List[str]:
    """Contract KPIs whose source columns are all present in this frame."""
    columns = set(df.columns)
    return [k for k, c in resolver.items() if set(c.source_fields) <= columns]
