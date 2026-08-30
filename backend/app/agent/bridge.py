"""
`bridge_periods` -- baseline -> current as a waterfall: named steps, an
"others" rollup, an explicit residual, and a running total, along either a
KPI's formula or one of its dimensions.

`decompose.py` already answers "which members moved" and "which formula
component moved", but neither output is ordered or cumulative -- there is no
step-by-step reconciliation a reader can follow from the baseline value to
the current one. This module adds no arithmetic of its own; it is a thin
layer over `DecomposeEngine`, the same way `concentration.py` layers over
`BreakdownEngine` and `drivers.py` layers over `DecomposeEngine` itself.

Three traps were measured against the real sample fixtures before writing
this module, because the field a waterfall would naively sum is wrong in two
different ways, and one KPI kind cannot be bridged by dimension at all:

1. **A ratio KPI's member step is `rate_effect + mix_effect`, never
   `change_abs`.** `MemberContribution.change_abs` is that member's own rate
   change, `(r_cur - r_base) * scale` -- it does not sum to the KPI's total
   change. What sums, exactly, is the Bennet-style split
   `w_cur*(r_cur - r_base) + r_base*(w_cur - w_base) = w_cur*r_cur -
   w_base*r_base`, whose members telescope to `R_cur - R_base` with no
   interaction term needed. Measured on the retail sample,
   `fulfillment_rate` by `product`, 2026-Q2 vs 2026-Q1: two of three members
   have `change_abs` of exactly `0.0` while their `rate_effect + mix_effect`
   is `+4.38` and `+3.12` -- a naive bridge renders two of three bars as
   zero, and on `gross_margin_pct` by `product` the naive sum is `+0.308`
   against a true change of `-0.342`: the waterfall would show the margin
   *rising* when it fell.
2. **The "others" step is the exact sum of the tail members' own step
   values, computed here -- never `OthersRollup.change_abs` and never a
   `contribution_pct` rescale.** `OthersRollup.change_abs` inherits trap 1
   (measured: `0.040` reported against a true `1.219` on the same KPI, a 30x
   error). Rescaling by `contribution_pct` is arithmetically exact but goes
   through a headcount that silently drops any member whose
   `contribution_pct` was `None`, and inherits a head/tail cut that ranks
   ratio members by rate change rather than waterfall contribution -- the
   two zero-`change_abs` members above would sort into the tail *first*.
   `decompose_by_dimension` already computes every member and only truncates
   afterwards, so this module asks for every member (`max_items=10**9`,
   never `None` -- see `observe.decompose_dimension`'s `max_items=None`
   duplication bug, fixed separately), ranks by each member's own true step
   value, and sums the tail exactly.
3. **A mean-kind KPI cannot be bridged by dimension at all**, and is refused
   with a typed status rather than rendering a wrong waterfall.
   `observe.decompose_dimension` routes `kind in ("sum", "mean")` down the
   same branch, but a mean KPI's members are group means and its total is
   the overall mean -- sum(group means) != overall mean. Measured, retail
   `inventory_units` (kind `mean`) by `region`: total change is `-12.48`
   while the naive member sum is `-49.92`, so the residual the un-refused
   tool would report is *three times the total change, with the opposite
   sign*. Gated on `kind() == "mean"`, not `is_additive()` -- a ratio KPI is
   also non-additive and reconciles exactly, so `is_additive()` would refuse
   the one axis this module gets most right.

The reconciliation invariant is unconditional and inherited from
`decompose_formula`'s own convention: `start_value + sum(step.value) ==
end_value`, where the last step is always `unexplained`, a residual defined
as whatever is left over and never computed independently. Publishing it as
a step, not only as a top-level field, is deliberate: trap 3's residual is
three times the total change with the opposite sign, and that has to be
visible in the artifact a reader looks at, not only in a sibling scalar a
renderer might not read. A waterfall that does not visually close has failed
at its one job.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

from ..config import get_settings
from ..engines.metrics import safe
from .contract_api import ContractAPI
from .decompose import DecomposeEngine
from .errors import InvalidArgumentError
from .query import TIME_GRAIN_COLUMNS
from .timefilter import TimeFilter, TimeSelection

STEP_KINDS = ("component", "interaction", "member", "others", "unexplained")


@dataclass(frozen=True)
class BridgeStep:
    label: str
    kind: str              # one of STEP_KINDS
    value: Optional[float]
    cumulative: Optional[float]

    def to_payload(self) -> Dict[str, Any]:
        return {"label": self.label, "kind": self.kind,
               "value": safe(self.value), "cumulative": safe(self.cumulative)}


@dataclass(frozen=True)
class Bridge:
    kpi: str
    label: str
    unit: str
    by: str
    axis: str                         # "formula" | "dimension"
    kind: str                         # the KPI's own kind: "sum" | "mean" | "ratio"
    status: str                       # "ok" | "single_term" | "unsupported_formula" | "insufficient" | "unsupported_kind"
    shape: Optional[str]
    reason: Optional[str]
    start_value: Optional[float]
    end_value: Optional[float]
    total_change: Optional[float]
    steps: Tuple[BridgeStep, ...]
    unexplained: Optional[float]
    selection: Optional[TimeSelection]
    baseline_selection: Optional[TimeSelection]
    filters: Mapping[str, Tuple[str, ...]]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "kpi": self.kpi, "label": self.label, "unit": self.unit,
            "by": self.by, "axis": self.axis, "kind": self.kind,
            "status": self.status, "shape": self.shape, "reason": self.reason,
            "start_value": safe(self.start_value), "end_value": safe(self.end_value),
            "total_change": safe(self.total_change),
            "steps": [s.to_payload() for s in self.steps],
            "unexplained": safe(self.unexplained),
            "selection": self.selection.to_payload() if self.selection else None,
            "baseline_selection": self.baseline_selection.to_payload() if self.baseline_selection else None,
            "filters": {k: list(v) for k, v in self.filters.items()},
        }


class BridgeEngine:
    """`bridge_periods`, bound to one dataset's `ContractAPI`. Computes no
    member or formula arithmetic of its own -- every number comes from
    `DecomposeEngine`."""

    def __init__(self, api: ContractAPI):
        self._api = api
        self._decompose = DecomposeEngine(api)

    def bridge_periods(self, df: pd.DataFrame, kpi_key: str,
                       period_a: Union[Mapping[str, Any], TimeFilter],
                       period_b: Union[Mapping[str, Any], TimeFilter],
                       by: str = "formula",
                       filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
                       max_items: Optional[int] = None) -> Bridge:
        label, unit = self._api.label(kpi_key), self._api.unit(kpi_key)
        kind = self._api.kind(kpi_key)
        filters_dict = dict(filters or {})

        if by == "formula":
            if any(d.name == "formula" for d in self._api.list_dimensions()):
                raise InvalidArgumentError(
                    "by", by,
                    "is ambiguous on this dataset -- 'formula' is both the reserved "
                    "axis keyword and a real dimension name here; use "
                    "decompose_by_dimension directly for that dimension.")
            return self._bridge_formula(df, kpi_key, period_a, period_b, filters_dict,
                                        label, unit, kind)

        if by in TIME_GRAIN_COLUMNS:
            raise InvalidArgumentError(
                "by", by,
                "is a time grain, not an entity dimension or 'formula' -- use "
                "get_timeseries for a KPI's history instead.",
                ["formula"] + [d.name for d in self._api.list_dimensions()])
        self._api.require_dimension(by)

        if kind == "mean":
            return Bridge(
                kpi=kpi_key, label=label, unit=unit, by=by, axis="dimension", kind=kind,
                status="unsupported_kind", shape=None,
                reason="a mean KPI's dimension members do not sum to its total "
                      "(sum of group means != overall mean); bridge by 'formula' instead",
                start_value=None, end_value=None, total_change=None,
                steps=(), unexplained=None,
                selection=None, baseline_selection=None, filters=filters_dict)

        return self._bridge_dimension(df, kpi_key, by, period_a, period_b, filters_dict,
                                      max_items, label, unit, kind)

    # -- formula axis -----------------------------------------------------------
    def _bridge_formula(self, df: pd.DataFrame, kpi_key: str, period_a, period_b,
                        filters: Mapping[str, Tuple[str, ...]], label: str, unit: str,
                        kind: str) -> Bridge:
        fd = self._decompose.decompose_formula(df, kpi_key, period_a, period_b, filters=filters)

        if fd.status != "ok":
            return Bridge(
                kpi=kpi_key, label=label, unit=unit, by="formula", axis="formula", kind=kind,
                status=fd.status, shape=fd.shape, reason=fd.reason,
                start_value=fd.kpi_value_baseline, end_value=fd.kpi_value_current,
                total_change=fd.total_change, steps=(), unexplained=None,
                selection=fd.selection, baseline_selection=fd.baseline_selection,
                filters=dict(fd.filters))

        steps: List[BridgeStep] = []
        for effect in fd.effects:
            if effect.value is not None:
                steps.append(BridgeStep(label=effect.name, kind="component",
                                        value=effect.value, cumulative=None))
        if fd.interaction is not None and fd.interaction != 0.0:
            steps.append(BridgeStep(label="interaction", kind="interaction",
                                    value=fd.interaction, cumulative=None))
        steps.sort(key=lambda s: (-abs(s.value), s.label))

        return self._finalise(kpi_key, label, unit, "formula", "formula", kind,
                              "ok", fd.shape, None,
                              fd.kpi_value_baseline, fd.kpi_value_current, fd.total_change,
                              steps, fd.unexplained,
                              fd.selection, fd.baseline_selection, dict(fd.filters))

    # -- dimension axis ---------------------------------------------------------
    def _bridge_dimension(self, df: pd.DataFrame, kpi_key: str, dimension: str,
                          period_a, period_b, filters: Mapping[str, Tuple[str, ...]],
                          max_items: Optional[int], label: str, unit: str,
                          kind: str) -> Bridge:
        # Every member, untruncated -- `decompose_by_dimension` only ever
        # truncates *after* computing every member, so asking for all of them
        # costs nothing extra and lets this module rank on the true waterfall
        # step value rather than inheriting a head/tail cut made on a
        # different key (see the module docstring, trap 2).
        full = self._decompose.decompose_by_dimension(df, kpi_key, dimension, period_a, period_b,
                                                       filters=filters, max_items=10**9)
        limit = max_items if max_items is not None else get_settings().max_drivers_per_dimension

        def _step_value(m) -> Optional[float]:
            if m.effects is not None:
                r, x = m.effects.rate_effect, m.effects.mix_effect
                return None if (r is None or x is None) else r + x
            return m.change_abs

        scored = [(m.name, _step_value(m)) for m in full.members]
        # Unresolved (`None`) members are never shown individually and never
        # folded into "others" either -- they simply are not subtracted from
        # `total_change`, so their contribution lands in `unexplained` rather
        # than being fabricated.
        resolved = [(name, value) for name, value in scored if value is not None]
        resolved.sort(key=lambda item: -abs(item[1]))

        head, tail = resolved[:limit], resolved[limit:]

        steps = [BridgeStep(label=name, kind="member", value=value, cumulative=None)
                for name, value in head]
        steps.sort(key=lambda s: (-abs(s.value), s.label))

        if tail:
            tail_sum = sum(value for _, value in tail)
            steps.append(BridgeStep(label=f"Other ({len(tail)} members)", kind="others",
                                    value=tail_sum, cumulative=None))

        total_change = full.change_abs
        explained = sum(s.value for s in steps)
        unexplained = (total_change - explained) if total_change is not None else None

        return self._finalise(kpi_key, label, unit, dimension, "dimension", kind,
                              "ok", None, None,
                              full.baseline_total, full.current_total, total_change,
                              steps, unexplained,
                              full.selection, full.baseline_selection, dict(full.filters))

    # -- shared finish: cumulative totals + the terminal residual step ----------
    def _finalise(self, kpi: str, label: str, unit: str, by: str, axis: str, kind: str,
                 status: str, shape: Optional[str], reason: Optional[str],
                 start_value: Optional[float], end_value: Optional[float],
                 total_change: Optional[float], steps: Sequence[BridgeStep],
                 unexplained: Optional[float],
                 selection: Optional[TimeSelection], baseline_selection: Optional[TimeSelection],
                 filters: Mapping[str, Tuple[str, ...]]) -> Bridge:
        running = start_value
        finalised: List[BridgeStep] = []
        for step in steps:
            running = running + step.value
            finalised.append(BridgeStep(label=step.label, kind=step.kind,
                                        value=step.value, cumulative=running))
        if unexplained is not None:
            running = running + unexplained
            finalised.append(BridgeStep(label="unexplained", kind="unexplained",
                                        value=unexplained, cumulative=running))

        return Bridge(kpi=kpi, label=label, unit=unit, by=by, axis=axis, kind=kind,
                      status=status, shape=shape, reason=reason,
                      start_value=start_value, end_value=end_value, total_change=total_change,
                      steps=tuple(finalised), unexplained=unexplained,
                      selection=selection, baseline_selection=baseline_selection,
                      filters=filters)
