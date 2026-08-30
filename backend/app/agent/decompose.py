"""
`decompose_by_dimension` / `decompose_rate_mix` / `decompose_nested` /
`decompose_formula` -- which part of a KPI's change came from where, along
both axes a KPI has: its dimension members, and its own formula.

The first three wrap `observe.decompose_dimension` (`engines/observe.py:292`)
rather than re-deriving it. That function is real, already-validated
arithmetic -- contribution and over-index for additive KPIs, a genuine
rate/mix split for ratio KPIs -- and it is the exact code
`test_rca_ground_truth.py` checks against the retail sample's planted
three-factor scenario. Re-expressing it through `QueryEngine` the way
`breakdown.py` and `scan.py` do would mean reimplementing that arithmetic with
real drift risk against the one suite that actually validates it. So here it
is wrapped: `TimeFilter` and dimension filters are resolved at the agent
boundary exactly as `quality.py` and `scan.py` already do, and the two
resulting frames are handed to `decompose_dimension` unchanged. What the
wrapper fixes is shape, not arithmetic: the tail row it folds unmatched
members into is literally named `f"Other ({len(tail)} members)"`
(`observe.py:371`), a templated sentence standing in for a member name, and
`is_aggregate` is present only on that one row so every caller has to
special-case it. Here the roll-up is a typed sibling, `others`, and never
appears in `members` at all.

`decompose_formula` is genuinely new maths. Today `formula_decomposition_candidates`
(`engines/hypotheses.py:218`) reads a KPI's formula and writes a sentence about
which half of a ratio moved -- computing **no attribution arithmetic at all**.
The only real formula arithmetic in the codebase,
`analysis.price_volume_decomposition` (`engines/analysis.py:192`), is hardcoded
to columns literally named `revenue` and `units_sold`, carries a prose
`narrative` field, and has no caller anywhere in `app/` -- only a test. This
module generalises that arithmetic through `agent/formula.py`'s AST structure,
for whatever a KPI's formula actually is.

Checked against every KPI in all four sample fixtures (retail, both
hospitals, school): the only operator any formula uses is `-`, twice. Every
ratio's denominator is a single bare field. There is no multiplicative KPI
anywhere in this repo -- the grammar and `resolver.evaluate` both support `*`
fine, so the multiplicative branch below is real and tested, just tested
synthetically rather than against a fixture that does not exist yet.

**The reconciliation invariant is unconditional, not case-by-case**:
`sum(effect.value) + interaction + unexplained == total_change`, exactly, for
every shape, because `unexplained` is defined as whatever is left over rather
than computed independently. For an additive formula that residual is
genuinely 0 -- `CompiledKpi.compute` for a `sum` KPI is a row-wise `evaluate`
then `.sum()`, and summation is linear, so the leaf-field effects add up
exactly. For a row-wise product it is not: `Σ(aᵢ·bᵢ) ≠ Σa · Σb`, so effects
built from aggregate factor totals cannot reconcile to the KPI's true,
row-wise value, and the gap is reported rather than hidden -- the opposite of
`price_volume_decomposition`'s one-sided convention, which makes its two
effects sum exactly to Δrevenue only by silently absorbing the cross-term into
one of them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

from ..config import get_settings
from ..engines import observe as observe_engine
from ..engines.metrics import safe
from ..kpi.resolver import BinOp, FieldRef, evaluate
from . import formula
from .contract_api import ContractAPI
from .errors import InvalidArgumentError
from .query import TIME_GRAIN_COLUMNS
from .timefilter import TimeFilter, TimeSelection
from .timefilter import apply as apply_time_filter
from .timefilter import parse as parse_time_filter


def _field_sum(frame: pd.DataFrame, field: str) -> float:
    """The raw total of one leaf field -- `evaluate(FieldRef(field), frame).sum()`,
    the same primitive `agent/formula.py` and `CompiledKpi.compute` are built
    on, so a leaf effect is computed the identical way the KPI's own value is."""
    return float(evaluate(FieldRef(field), frame).sum())


# ---------------------------------------------------------------------------
# decompose_by_dimension() / decompose_rate_mix() / decompose_nested()
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RateMixEffects:
    rate_effect: Optional[float]
    mix_effect: Optional[float]

    def to_payload(self) -> Dict[str, Any]:
        return {"rate_effect": safe(self.rate_effect), "mix_effect": safe(self.mix_effect)}


@dataclass(frozen=True)
class MemberContribution:
    name: str
    current: Optional[float]
    baseline: Optional[float]
    change_abs: Optional[float]
    change_pct: Optional[float]
    contribution_pct: Optional[float]
    share_of_current_pct: Optional[float]
    share_of_baseline_pct: Optional[float]
    over_index: Optional[float]
    effects: Optional[RateMixEffects]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "member": self.name, "current": safe(self.current), "baseline": safe(self.baseline),
            "change_abs": safe(self.change_abs), "change_pct": safe(self.change_pct),
            "contribution_pct": safe(self.contribution_pct),
            "share_of_current_pct": safe(self.share_of_current_pct),
            "share_of_baseline_pct": safe(self.share_of_baseline_pct),
            "over_index": safe(self.over_index),
            "effects": self.effects.to_payload() if self.effects else None,
        }


@dataclass(frozen=True)
class OthersRollup:
    """The tail `decompose_dimension` folds past `max_items`, as a typed
    sibling rather than a member row named `f"Other ({N} members)"`. Exact
    for the four fields it sums; the rest have no member-level meaning to
    aggregate and are honestly `None`, not recomputed."""
    member_count: int
    current: Optional[float]
    baseline: Optional[float]
    change_abs: Optional[float]
    contribution_pct: Optional[float]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "member_count": self.member_count, "current": safe(self.current),
            "baseline": safe(self.baseline), "change_abs": safe(self.change_abs),
            "contribution_pct": safe(self.contribution_pct),
        }


@dataclass(frozen=True)
class DimensionDecomposition:
    kpi: str
    label: str
    unit: str
    dimension: str
    kind: str
    current_total: Optional[float]
    baseline_total: Optional[float]
    change_abs: Optional[float]
    members: Tuple[MemberContribution, ...]
    others: Optional[OthersRollup]
    selection: TimeSelection
    baseline_selection: TimeSelection
    filters: Mapping[str, Tuple[str, ...]]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "kpi": self.kpi, "label": self.label, "unit": self.unit,
            "dimension": self.dimension, "kind": self.kind,
            "current_total": safe(self.current_total), "baseline_total": safe(self.baseline_total),
            "change_abs": safe(self.change_abs),
            "members": [m.to_payload() for m in self.members],
            "others": self.others.to_payload() if self.others else None,
            "selection": self.selection.to_payload(),
            "baseline_selection": self.baseline_selection.to_payload(),
            "filters": {k: list(v) for k, v in self.filters.items()},
        }


@dataclass(frozen=True)
class RateMixEntry:
    member: str
    rate_effect: Optional[float]
    mix_effect: Optional[float]
    change_abs: Optional[float]

    def to_payload(self) -> Dict[str, Any]:
        return {"member": self.member, "rate_effect": safe(self.rate_effect),
               "mix_effect": safe(self.mix_effect), "change_abs": safe(self.change_abs)}


@dataclass(frozen=True)
class RateMixDecomposition:
    status: str                       # "ok" | "not_a_ratio"
    kpi: str
    dimension: str
    members: Tuple[RateMixEntry, ...]
    total_rate_effect: Optional[float] = None
    total_mix_effect: Optional[float] = None
    selection: Optional[TimeSelection] = None
    baseline_selection: Optional[TimeSelection] = None
    # `()` cannot stand in for an empty mapping (`.items()` below would raise
    # `AttributeError`); `{}` cannot be the literal default (a dataclass
    # forbids a mutable default), so `field(default_factory=dict)` is used.
    filters: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        payload = {
            "status": self.status, "kpi": self.kpi, "dimension": self.dimension,
            "members": [m.to_payload() for m in self.members],
            "total_rate_effect": safe(self.total_rate_effect),
            "total_mix_effect": safe(self.total_mix_effect),
        }
        if self.selection is not None:
            payload["selection"] = self.selection.to_payload()
            payload["baseline_selection"] = self.baseline_selection.to_payload()
            payload["filters"] = {k: list(v) for k, v in self.filters.items()}
        return payload


# ---------------------------------------------------------------------------
# decompose_formula()
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Effect:
    name: str
    value: Optional[float]
    # The numerator's own leaf fields, when its own structure is a simple
    # additive combination (`agent/formula.py` `Term`s at depth 0) -- never
    # attempted for a denominator, since every ratio denominator in this
    # codebase is already a single bare field.
    components: Optional[Tuple["Effect", ...]] = None

    def to_payload(self) -> Dict[str, Any]:
        return {
            "name": self.name, "value": safe(self.value),
            "components": [c.to_payload() for c in self.components] if self.components else None,
        }


@dataclass(frozen=True)
class FormulaDecomposition:
    kpi: str
    label: str
    unit: str
    status: str                       # "ok" | "single_term" | "unsupported_formula" | "insufficient"
    shape: Optional[str]              # "ratio" | "additive" | "multiplicative" | None
    reason: Optional[str]
    kpi_value_current: Optional[float]
    kpi_value_baseline: Optional[float]
    total_change: Optional[float]
    effects: Tuple[Effect, ...]
    interaction: Optional[float]
    unexplained: Optional[float]
    selection: TimeSelection
    baseline_selection: TimeSelection
    filters: Mapping[str, Tuple[str, ...]]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "kpi": self.kpi, "label": self.label, "unit": self.unit,
            "status": self.status, "shape": self.shape, "reason": self.reason,
            "kpi_value_current": safe(self.kpi_value_current),
            "kpi_value_baseline": safe(self.kpi_value_baseline),
            "total_change": safe(self.total_change),
            "effects": [e.to_payload() for e in self.effects],
            "interaction": safe(self.interaction), "unexplained": safe(self.unexplained),
            "selection": self.selection.to_payload(),
            "baseline_selection": self.baseline_selection.to_payload(),
            "filters": {k: list(v) for k, v in self.filters.items()},
        }


def _classify(api: ContractAPI, key: str) -> Tuple[str, Any]:
    """
    `("ratio"|"single_term"|"multiplicative"|"additive"|"unsupported_formula", extra)`.

    `formula.structure()` deliberately does not distinguish `+` from `*` --
    both "preserve context" for its sign/position bookkeeping, because neither
    changes which position a divided-into field ends up in. That is exactly
    the distinction this function needs, so multiplicative shape is detected
    by inspecting the raw AST directly rather than through `Term`.

    Only a *contract* sum/mean KPI can be multiplicative: the seed registry's
    `numerator_expr` is evaluated by `_col_sum` (`metrics.py`), which only
    ever supports a single optional subtraction -- confirmed empirically,
    every seed and contract formula in every sample fixture uses no operator
    but `-`. So a seed KPI's `expression_ast` never needs inspecting for `*`.
    """
    struct = formula.structure(api, key, expand=True)
    if struct.kind == "ratio":
        return "ratio", struct
    if len(struct.leaf_fields) <= 1:
        return "single_term", struct
    if struct.kind == "mean":
        # `CompiledKpi.compute` takes `evaluate(ast, frame).mean()` for a mean
        # KPI. Mean is linear too, so a multi-field mean *could* be decomposed
        # the same way a sum is -- but every mean-kind KPI in every sample
        # fixture is a single bare column (caught by the check above), so
        # nothing here has ever needed it, and `_field_sum`'s `.sum()` would
        # silently be the wrong operator if it were used naively. Refuse
        # rather than guess.
        return "unsupported_formula", "a mean KPI over more than one field"

    spec = api.require(key)
    ast = getattr(spec, "expression_ast", None)
    if isinstance(ast, BinOp) and ast.op == "*":
        if isinstance(ast.left, FieldRef) and isinstance(ast.right, FieldRef):
            return "multiplicative", (ast.left.name, ast.right.name)
        return "unsupported_formula", "a product of more than two simple factors"
    return "additive", struct


def _numerator_components(struct: "formula.FormulaStructure", cur: pd.DataFrame,
                          base: pd.DataFrame, den_b: float, scale: float
                          ) -> Optional[Tuple[Effect, ...]]:
    """The numerator's own leaf-field effects, valid only because
    `numerator_effect = ΔN / D_b · scale` is linear in N -- so each field's
    share of ΔN carries the same `1/D_b·scale` factor. Only attempted when
    every numerator term is written directly in this KPI's own formula
    (depth 0), never through a nested KPI expansion, where the same linear
    argument would not hold."""
    terms = struct.terms_in(formula.NUMERATOR)
    if not terms or any(t.depth > 0 for t in terms):
        return None
    fields = list(dict.fromkeys(t.field for t in terms))
    if len(fields) <= 1:
        return None
    out = []
    for field in fields:
        weight = sum(t.sign for t in terms if t.field == field)
        delta = _field_sum(cur, field) - _field_sum(base, field)
        out.append(Effect(name=field, value=weight * delta / den_b * scale))
    return tuple(out)


class DecomposeEngine:
    """`decompose_by_dimension` / `decompose_rate_mix` / `decompose_nested` /
    `decompose_formula`, bound to one dataset's `ContractAPI`."""

    def __init__(self, api: ContractAPI):
        self._api = api

    def _select_frame(self, df: pd.DataFrame, tf: TimeFilter,
                      filters: Optional[Mapping[str, Union[str, Sequence[str]]]]
                      ) -> Tuple[pd.DataFrame, TimeSelection]:
        """The same time-then-filter resolution `QueryEngine.query` performs
        (`agent/query.py:143-156`), also duplicated in `quality.py` and
        `scan.py` for the same reason: this module needs the raw filtered
        frame itself, not an aggregated cell."""
        frame, selection = apply_time_filter(df, tf)
        resolved: Dict[str, Tuple[str, ...]] = {}
        for dim, raw_values in (filters or {}).items():
            self._api.require_dimension(dim)
            values = [raw_values] if isinstance(raw_values, str) else list(raw_values)
            resolved[dim] = tuple(self._api.resolve_member(dim, v, frame) for v in values)
        for dim, members in resolved.items():
            frame = frame[frame[dim].isin(members)]
        return frame, selection

    def _two_frames(self, df: pd.DataFrame, period_a, period_b,
                    filters) -> Tuple[pd.DataFrame, TimeSelection, pd.DataFrame, TimeSelection]:
        tf_a = period_a if isinstance(period_a, TimeFilter) else parse_time_filter(period_a)
        tf_b = period_b if isinstance(period_b, TimeFilter) else parse_time_filter(period_b)
        cur, sel_a = self._select_frame(df, tf_a, filters)
        base, sel_b = self._select_frame(df, tf_b, filters)
        return cur, sel_a, base, sel_b

    # -- decompose_by_dimension() -----------------------------------------------
    def decompose_by_dimension(self, df: pd.DataFrame, kpi_key: str, dimension: str,
                               period_a: Union[Mapping[str, Any], TimeFilter],
                               period_b: Union[Mapping[str, Any], TimeFilter],
                               filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
                               max_items: Optional[int] = None) -> DimensionDecomposition:
        if dimension in TIME_GRAIN_COLUMNS:
            raise InvalidArgumentError(
                "dimension", dimension,
                "is a time grain, not an entity dimension -- use get_timeseries "
                "for a KPI's history instead.",
                [d.name for d in self._api.list_dimensions()])
        self._api.require_dimension(dimension)

        cur, sel_a, base, sel_b = self._two_frames(df, period_a, period_b, filters)
        limit = max_items if max_items is not None else get_settings().max_drivers_per_dimension
        rows = observe_engine.decompose_dimension(cur, base, dimension, kpi_key, limit,
                                                   self._api.resolver)

        non_aggregate = [r for r in rows if not r.get("is_aggregate")]
        aggregate = next((r for r in rows if r.get("is_aggregate")), None)

        others: Optional[OthersRollup] = None
        if aggregate is not None:
            all_members = (set(cur[dimension].dropna().astype(str).unique())
                          | set(base[dimension].dropna().astype(str).unique()))
            others = OthersRollup(
                member_count=len(all_members) - len(non_aggregate),
                current=aggregate["current"], baseline=aggregate["baseline"],
                change_abs=aggregate["change_abs"], contribution_pct=aggregate["contribution_pct"])

        members = tuple(
            MemberContribution(
                name=str(r["name"]), current=r["current"], baseline=r["baseline"],
                change_abs=r["change_abs"], change_pct=r["change_pct"],
                contribution_pct=r["contribution_pct"],
                share_of_current_pct=r["share_of_current_pct"],
                share_of_baseline_pct=r["share_of_baseline_pct"], over_index=r["over_index"],
                effects=RateMixEffects(**r["effects"]) if r.get("effects") else None)
            for r in non_aggregate)

        return DimensionDecomposition(
            kpi=kpi_key, label=self._api.label(kpi_key), unit=self._api.unit(kpi_key),
            dimension=dimension, kind=self._api.kind(kpi_key),
            current_total=self._api.value(cur, kpi_key), baseline_total=self._api.value(base, kpi_key),
            change_abs=self._api.value(cur, kpi_key) - self._api.value(base, kpi_key),
            members=members, others=others, selection=sel_a, baseline_selection=sel_b,
            filters=dict(filters or {}))

    # -- decompose_rate_mix() ---------------------------------------------------
    def decompose_rate_mix(self, df: pd.DataFrame, kpi_key: str, dimension: str,
                           period_a: Union[Mapping[str, Any], TimeFilter],
                           period_b: Union[Mapping[str, Any], TimeFilter],
                           filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
                           max_items: Optional[int] = None) -> RateMixDecomposition:
        if self._api.kind(kpi_key) != "ratio":
            return RateMixDecomposition(status="not_a_ratio", kpi=kpi_key, dimension=dimension,
                                        members=())

        full = self.decompose_by_dimension(df, kpi_key, dimension, period_a, period_b,
                                           filters=filters, max_items=max_items)
        entries = tuple(
            RateMixEntry(member=m.name,
                        rate_effect=m.effects.rate_effect if m.effects else None,
                        mix_effect=m.effects.mix_effect if m.effects else None,
                        change_abs=m.change_abs)
            for m in full.members)
        rates = [e.rate_effect for e in entries if e.rate_effect is not None]
        mixes = [e.mix_effect for e in entries if e.mix_effect is not None]

        return RateMixDecomposition(
            status="ok", kpi=kpi_key, dimension=dimension, members=entries,
            total_rate_effect=sum(rates) if rates else None,
            total_mix_effect=sum(mixes) if mixes else None,
            selection=full.selection, baseline_selection=full.baseline_selection,
            filters=full.filters)

    # -- decompose_nested() -------------------------------------------------
    def decompose_nested(self, df: pd.DataFrame, kpi_key: str, outer_dimension: str,
                         outer_member: str, inner_dimension: str,
                         period_a: Union[Mapping[str, Any], TimeFilter],
                         period_b: Union[Mapping[str, Any], TimeFilter],
                         filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
                         max_items: Optional[int] = None) -> DimensionDecomposition:
        merged = dict(filters or {})
        merged[outer_dimension] = outer_member
        return self.decompose_by_dimension(df, kpi_key, inner_dimension, period_a, period_b,
                                           filters=merged, max_items=max_items)

    # -- decompose_formula() -----------------------------------------------
    def decompose_formula(self, df: pd.DataFrame, kpi_key: str,
                          period_a: Union[Mapping[str, Any], TimeFilter],
                          period_b: Union[Mapping[str, Any], TimeFilter],
                          filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None
                          ) -> FormulaDecomposition:
        cur, sel_a, base, sel_b = self._two_frames(df, period_a, period_b, filters)
        label, unit = self._api.label(kpi_key), self._api.unit(kpi_key)
        filters_out = dict(filters or {})

        kpi_c = self._api.value(cur, kpi_key)
        kpi_b = self._api.value(base, kpi_key)
        total_change = (kpi_c - kpi_b) if (kpi_c == kpi_c and kpi_b == kpi_b) else None

        def _done(status: str, shape: Optional[str] = None, reason: Optional[str] = None,
                  effects: Tuple[Effect, ...] = (), interaction: Optional[float] = None
                  ) -> FormulaDecomposition:
            unexplained = None
            if status == "ok" and total_change is not None:
                explained = sum(e.value for e in effects if e.value is not None) + (interaction or 0.0)
                unexplained = total_change - explained
            return FormulaDecomposition(
                kpi=kpi_key, label=label, unit=unit, status=status, shape=shape, reason=reason,
                kpi_value_current=kpi_c, kpi_value_baseline=kpi_b, total_change=total_change,
                effects=effects, interaction=interaction, unexplained=unexplained,
                selection=sel_a, baseline_selection=sel_b, filters=filters_out)

        shape, extra = _classify(self._api, kpi_key)

        if shape == "single_term":
            return _done("single_term", shape="single_term",
                        reason="this KPI is a single field; there is nothing to decompose")
        if shape == "unsupported_formula":
            return _done("unsupported_formula", shape=None, reason=extra)

        if shape == "ratio":
            struct = extra
            num_c, den_c = self._api.components(cur, kpi_key)
            num_b, den_b = self._api.components(base, kpi_key)
            if not den_b or not den_c or den_b != den_b or den_c != den_c or num_c is None:
                return _done("insufficient", shape="ratio",
                            reason="a zero or missing denominator in one of the two periods")
            scale = struct.scale
            numerator_effect = (num_c - num_b) / den_b * scale
            denominator_effect = num_b * (1.0 / den_c - 1.0 / den_b) * scale
            interaction = (num_c - num_b) * (1.0 / den_c - 1.0 / den_b) * scale
            effects = (
                Effect(name="numerator", value=numerator_effect,
                      components=_numerator_components(struct, cur, base, den_b, scale)),
                Effect(name="denominator", value=denominator_effect),
            )
            return _done("ok", shape="ratio", effects=effects, interaction=interaction)

        if shape == "multiplicative":
            field_a, field_b = extra
            a_c, a_b = _field_sum(cur, field_a), _field_sum(base, field_a)
            b_c, b_b = _field_sum(cur, field_b), _field_sum(base, field_b)
            effect_a = (a_c - a_b) * b_b
            effect_b = a_b * (b_c - b_b)
            interaction = (a_c - a_b) * (b_c - b_b)
            effects = (Effect(name=field_a, value=effect_a), Effect(name=field_b, value=effect_b))
            return _done("ok", shape="multiplicative", effects=effects, interaction=interaction)

        # additive: {a} - {b}, or any depth-0 sum/difference of leaf fields
        struct = extra
        effects = []
        for field in dict.fromkeys(t.field for t in struct.terms):
            weight = sum(t.sign for t in struct.terms if t.field == field)
            delta = _field_sum(cur, field) - _field_sum(base, field)
            effects.append(Effect(name=field, value=weight * delta))
        return _done("ok", shape="additive", effects=tuple(effects), interaction=0.0)
