"""
Noise filtering and data quality -- is a change worth explaining, and can this
number be trusted.

`filter_material_changes` re-expresses the materiality gate that today lives
only inside `signals.material_signals` (`engines/signals.py:62`). That
function is not wrappable: it takes a whole `observe()` output dict, reads
roughly sixteen keys off it, emits two *different* row shapes depending on
whether a KPI was named in the question or found on the scoreboard, carries a
prose `filter_note`, and gates the scoreboard sweep on a bare literal
(`abs(change) < 5.0`, `signals.py:106`) that quietly disagrees with
`Settings.min_material_change_pct = 3.0` (`config.py:41`). Dragging that dict
shape into the agent layer is exactly what this rewrite exists to delete, so
`QualityEngine.filter_material` is a pure function over a list of candidate
records instead -- the same shape `compare.Comparison.to_payload()` and
`compare.Significance.to_payload()` already emit, so O4's output composes
straight into this gate with no adapter in between. `signals.py` and its
callers (`pipeline.py`, `investigate.py`) are untouched; the legacy gate keeps
serving the legacy pipeline.

`check_data_quality` wires up `analysis.coverage_report`
(`engines/analysis.py:228`), which computes rows, day counts, and dimension
members appeared/disappeared between two periods and has **zero callers
anywhere in the repo** -- and strips its prose `issues[]`. It adds the one
fact `coverage_report` was never able to state at all: whether a zero in a
metric column is a real zero or one `prepare`'s `fillna(0.0)`
(`metrics.py:344`) fabricated. That distinction is recorded once, at load
time, on `schema.imputed_cells` (see `metrics.prepare`'s docstring) and
surfaced here through `ContractAPI.imputed_cells()` -- without it, a
fabricated zero and a real zero are byte-identical by the time any tool sees
the frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

from ..config import get_settings
from ..engines.metrics import pct_change, safe
from .contract_api import ContractAPI
from .timefilter import TimeFilter, TimeSelection
from .timefilter import apply as apply_time_filter
from .timefilter import parse as parse_time_filter


# ---------------------------------------------------------------------------
# filter_material()
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MaterialityCandidate:
    kpi: str
    change_pct: Optional[float]
    is_material: bool
    basis: str                        # "magnitude_and_significance" | "magnitude_only"
    reason: Optional[str]             # "below_pct" | "below_z" | "no_change" | None (material)
    is_unfavourable: Optional[bool]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "kpi": self.kpi,
            "change_pct": safe(self.change_pct),
            "is_material": self.is_material,
            "basis": self.basis,
            "reason": self.reason,
            "is_unfavourable": self.is_unfavourable,
        }


@dataclass(frozen=True)
class MaterialityResult:
    candidates: Tuple[MaterialityCandidate, ...]
    considered: int
    retained: int
    min_pct: float
    min_z: float

    def to_payload(self) -> Dict[str, Any]:
        return {
            "candidates": [c.to_payload() for c in self.candidates],
            "considered": self.considered,
            "retained": self.retained,
            "min_pct": self.min_pct,
            "min_z": self.min_z,
        }


def _reason(is_material: bool, change_pct: Optional[float], pct_ok: bool) -> Optional[str]:
    if is_material:
        return None
    if change_pct is None or change_pct == 0:
        return "no_change"
    if not pct_ok:
        return "below_pct"
    return "below_z"


# ---------------------------------------------------------------------------
# check_data_quality()
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DimensionCoverage:
    dimension: str
    current_members: int
    baseline_members: Optional[int]   # None when no baseline was requested
    appeared: Tuple[str, ...]
    disappeared: Tuple[str, ...]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "current_members": self.current_members,
            "baseline_members": self.baseline_members,
            "appeared": list(self.appeared),
            "disappeared": list(self.disappeared),
        }


@dataclass(frozen=True)
class ColumnImputation:
    column: str
    rows: int
    zero_rows: int
    null_rows: int
    # Dataset-wide count from `prepare` (not scoped to the current selection --
    # `prepare` counts once, over the whole frame, and never records *which*
    # rows it filled). `None` when `preserve_missing` was on for this dataset.
    imputed_rows: Optional[int]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "rows": self.rows,
            "zero_rows": self.zero_rows,
            "null_rows": self.null_rows,
            "imputed_rows": self.imputed_rows,
        }


@dataclass(frozen=True)
class DataQuality:
    rows: int
    days: int
    periods_held: int                 # dataset-wide, not scoped to this selection
    selection: TimeSelection
    baseline_rows: Optional[int]
    baseline_days: Optional[int]
    baseline_selection: Optional[TimeSelection]
    row_change_pct: Optional[float]
    window_ratio: Optional[float]
    window_balanced: Optional[bool]
    dimensions: Tuple[DimensionCoverage, ...]
    zero_fill_applied: bool
    columns: Tuple[ColumnImputation, ...]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "rows": self.rows,
            "days": self.days,
            "periods_held": self.periods_held,
            "selection": self.selection.to_payload(),
            "baseline_rows": self.baseline_rows,
            "baseline_days": self.baseline_days,
            "baseline_selection": self.baseline_selection.to_payload() if self.baseline_selection else None,
            "row_change_pct": safe(self.row_change_pct),
            "window_ratio": safe(self.window_ratio),
            "window_balanced": self.window_balanced,
            "dimensions": [d.to_payload() for d in self.dimensions],
            "zero_fill_applied": self.zero_fill_applied,
            "columns": [c.to_payload() for c in self.columns],
        }


class QualityEngine:
    """`filter_material_changes` / `check_data_quality`, bound to one
    dataset's `ContractAPI`."""

    def __init__(self, api: ContractAPI):
        self._api = api

    # -- filter_material() -----------------------------------------------------
    def filter_material(self, candidates: Sequence[Mapping[str, Any]],
                        min_pct: Optional[float] = None,
                        min_z: Optional[float] = None) -> MaterialityResult:
        """
        Partition `candidates` (records carrying `kpi` and `change_pct`,
        optionally `robust_z`) into material and immaterial, on the same two
        thresholds `observe.assess_significance` uses -- `min_pct` defaults to
        `Settings.min_material_change_pct`, `min_z` to
        `Settings.anomaly_z_threshold` -- so this gate and the z-test the
        candidates may already carry can never disagree.

        A candidate with no `robust_z` (a plain `Comparison`, not a
        `Significance`) is judged on magnitude alone and marked
        `basis="magnitude_only"` -- an untested candidate, never a candidate
        that *failed* a significance test it was never given.
        """
        s = get_settings()
        min_pct = s.min_material_change_pct if min_pct is None else min_pct
        min_z = s.anomaly_z_threshold if min_z is None else min_z

        out: List[MaterialityCandidate] = []
        for c in candidates:
            kpi = c.get("kpi") or ""
            change_pct = c.get("change_pct")
            robust_z = c.get("robust_z")

            pct_ok = change_pct is not None and abs(change_pct) >= min_pct
            if robust_z is not None:
                z_ok = abs(robust_z) >= min_z
                is_material = pct_ok and z_ok
                basis = "magnitude_and_significance"
            else:
                is_material = pct_ok
                basis = "magnitude_only"

            is_unfavourable: Optional[bool] = None
            if change_pct is not None:
                higher_better = self._api.polarity(kpi)
                is_unfavourable = (change_pct < 0) if higher_better else (change_pct > 0)

            out.append(MaterialityCandidate(
                kpi=kpi, change_pct=change_pct, is_material=is_material, basis=basis,
                reason=_reason(is_material, change_pct, pct_ok),
                is_unfavourable=is_unfavourable))

        retained = sum(1 for c in out if c.is_material)
        return MaterialityResult(candidates=tuple(out), considered=len(out),
                                 retained=retained, min_pct=min_pct, min_z=min_z)

    # -- check_data_quality() ---------------------------------------------------
    def _select_frame(self, df: pd.DataFrame, tf: TimeFilter,
                      filters: Optional[Mapping[str, Union[str, Sequence[str]]]]
                      ) -> Tuple[pd.DataFrame, TimeSelection]:
        """The same time-then-filter resolution `QueryEngine.query` performs
        (`agent/query.py:143-156`) -- every filter resolved against the
        time-selected frame as it stood before any filter narrowed it."""
        frame, selection = apply_time_filter(df, tf)
        resolved: Dict[str, Tuple[str, ...]] = {}
        for dim, raw_values in (filters or {}).items():
            self._api.require_dimension(dim)
            values = [raw_values] if isinstance(raw_values, str) else list(raw_values)
            resolved[dim] = tuple(self._api.resolve_member(dim, v, frame) for v in values)
        for dim, members in resolved.items():
            frame = frame[frame[dim].isin(members)]
        return frame, selection

    def check_data_quality(self, df: pd.DataFrame,
                           time_filter: Union[Mapping[str, Any], TimeFilter],
                           baseline: Optional[Union[Mapping[str, Any], TimeFilter]] = None,
                           filters: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
                           window_tolerance: float = 0.1) -> DataQuality:
        """
        `baseline` is optional because `coverage_report` needs two frames to
        compare members and window balance -- with one, those facts are
        reported; without one, only the single-period facts are, rather than
        fabricating a baseline to compare against. `window_tolerance` is the
        0.9-1.1 rule `coverage_report` hardcoded (`analysis.py:258`), named
        here as the deviation from a balanced 1.0 ratio it always was.
        """
        tf = time_filter if isinstance(time_filter, TimeFilter) else parse_time_filter(time_filter)
        frame, selection = self._select_frame(df, tf, filters)
        rows = int(len(frame))
        days = int(frame["_date"].nunique()) if rows else 0

        base_frame: Optional[pd.DataFrame] = None
        base_selection: Optional[TimeSelection] = None
        baseline_rows = baseline_days = None
        row_change_pct = window_ratio = window_balanced = None
        if baseline is not None:
            base_tf = baseline if isinstance(baseline, TimeFilter) else parse_time_filter(baseline)
            base_frame, base_selection = self._select_frame(df, base_tf, filters)
            baseline_rows = int(len(base_frame))
            baseline_days = int(base_frame["_date"].nunique()) if baseline_rows else 0
            row_change_pct = pct_change(rows, baseline_rows)
            if days and baseline_days:
                window_ratio = days / baseline_days
                window_balanced = (1 - window_tolerance) <= window_ratio <= (1 + window_tolerance)

        dimensions: List[DimensionCoverage] = []
        for dim_info in self._api.list_dimensions(df):
            name = dim_info.name
            if name not in frame.columns:
                continue
            current_members = set(frame[name].dropna().unique())
            if base_frame is not None and name in base_frame.columns:
                baseline_members = set(base_frame[name].dropna().unique())
                appeared = tuple(sorted(current_members - baseline_members))[:10]
                disappeared = tuple(sorted(baseline_members - current_members))[:10]
                baseline_count: Optional[int] = len(baseline_members)
            else:
                appeared, disappeared, baseline_count = (), (), None
            dimensions.append(DimensionCoverage(
                dimension=name, current_members=len(current_members),
                baseline_members=baseline_count, appeared=appeared, disappeared=disappeared))

        imputed = self._api.imputed_cells()
        zero_fill_applied = imputed is not None
        columns: List[ColumnImputation] = []
        for col in self._api.metric_columns():
            if col not in frame.columns:
                continue
            values = frame[col]
            zero_rows = int((values == 0.0).sum())
            null_rows = int(values.isna().sum())
            imputed_rows = (imputed or {}).get(col, 0) if zero_fill_applied else None
            columns.append(ColumnImputation(column=col, rows=rows, zero_rows=zero_rows,
                                           null_rows=null_rows, imputed_rows=imputed_rows))

        return DataQuality(
            rows=rows, days=days, periods_held=self._api.quarters_held(),
            selection=selection, baseline_rows=baseline_rows, baseline_days=baseline_days,
            baseline_selection=base_selection, row_change_pct=row_change_pct,
            window_ratio=window_ratio, window_balanced=window_balanced,
            dimensions=tuple(dimensions), zero_fill_applied=zero_fill_applied,
            columns=tuple(columns))
