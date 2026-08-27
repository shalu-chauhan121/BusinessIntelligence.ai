"""
Reconciling one business view out of several heterogeneous sources.

The assumption this layer is built on: **sources already share field names and
semantics.** `Sales.csv`, `Finance.csv` and `Ops.csv` all expose `date`,
`region`, `product_id`, `revenue`, and the KPI Contract already defines what
`revenue` means. There is therefore nothing to map, and this module deliberately
offers no way to declare a mapping — no field_map, no key_map, no member_map, no
per-upload configuration. A source that does not already speak the shared
vocabulary is refused, not translated.

What actually needs deciding, and what this module owns:

    * are the sources mutually compatible at all
    * how fresh is each one, independently
    * how do their periods align
    * which periods have enough coverage to compute a given KPI
    * where two sources describe the same thing, which value is canonical

The KPI Contract remains the authority on semantics. Everything this module
needs is *derived* from what the contract and profiler already produce:
required fields from `CompiledKpi.source_fields` (parsed from the formula),
business grain from `detect_row_grain`, rollup from `FieldProfile.additivity`.

The central idea is that reconciliation is decided **per cell** — one
(business key, period, measure) triple — not per column:

    1 source   take it
    N agree    corroborate
    N disagree record a blocking `source_disagreement` conflict and leave the
               cell MISSING. Choosing between two contradictory accounts of the
               same fact is a business decision, so it goes to the human through
               the contract's existing conflict-resolution flow.

Because the decision is per cell, one measure can be partitioned in some places
and overlapping in others; the report summarises such a measure as `mixed`.
Cross-source addition never happens. Summing occurs only *within* one source
when rolling its rows up to the target grain.

Two things are deliberately kept apart that are easy to conflate:

    data_grain       how finely a source's rows are spaced   -> time alignment
    refresh_cadence  how often the feed is republished       -> staleness

Neither is evidence for the other, so cadence is never inferred from grain. A
cadence that was not declared or observed stays `unknown`, and freshness is then
reported as `freshness_unknown` rather than a fabricated "current".

The **KPI Contract** sets the target grain. Source grains only constrain it:
each source may be rolled UP to the target, never down, because rolling down
would fabricate detail the source does not have.

Nothing here calls an LLM.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from ..config import get_settings
from ..engines.metrics import DatasetSchema, detect_schema, normalise_columns
from ..kpi.contract import ConflictFlag
from ..kpi.profiling import detect_row_grain, infer_additivity, tokenise

# How long after its declared refresh a source of each cadence is still
# considered current. A cadence with no entry here can never be called stale:
# the honest answer is that freshness cannot be asserted, not a fabricated
# "current".
_CADENCE_TOLERANCE_SECONDS = {
    "hourly": 2 * 3600,
    "daily": 36 * 3600,
    "weekly": 10 * 86400,
    "monthly": 40 * 86400,
}

# Coarseness order for choosing a shared grain. A source may always be rolled UP
# to a coarser grain; it can never be rolled down, because that would mean
# inventing detail the source does not have.
_GRAIN_ORDER = ["day", "week", "month", "quarter", "year"]

_PERIOD_FREQ = {"day": "D", "week": "W", "month": "M", "quarter": "Q", "year": "Y"}

# `DatasetSchema.grain` speaks in cadences ("daily", "14-day"); the aligner
# needs a period grain. Mirrors `kpi.service._time_grain_from_schema_grain`.
_CADENCE_TO_GRAIN = {"daily": "day", "weekly": "week", "monthly": "month",
                     "quarterly": "quarter", "yearly": "year"}


class ReconciliationError(ValueError):
    """The sources cannot be safely reconciled. Never raised speculatively."""

    def __init__(self, message: str, issues: Optional[List[ConflictFlag]] = None):
        super().__init__(message)
        self.issues = issues or []


# ---------------------------------------------------------------------------
# structures
# ---------------------------------------------------------------------------
@dataclass
class SourceFile:
    """One uploaded file, before reconciliation. Transient."""

    source_id: str
    label: str
    filename: str
    frame: pd.DataFrame
    schema: DatasetSchema
    last_refresh_at: str = ""
    last_refresh_basis: str = "unknown"     # source_reported | data_max | upload_time | unknown
    refresh_cadence: str = "unknown"        # declared | observed | unknown — NEVER from grain
    rows_in: int = 0

    @property
    def data_grain(self) -> str:
        """
        How finely this source's ROWS are spaced.

        Deliberately distinct from `refresh_cadence`: a feed can carry weekly
        rows and be republished hourly, or carry hourly rows and land once a
        day. Grain constrains time alignment; cadence governs staleness. Neither
        is evidence for the other.
        """
        return _grain_from_cadence(self.schema.grain or "")


@dataclass
class SourceReport:
    """Per-source provenance and freshness, persisted on the dataset document."""

    source_id: str
    label: str
    filename: str
    last_refresh_at: str = ""
    last_refresh_basis: str = "unknown"  # source_reported | data_max | upload_time | unknown
    refresh_cadence: str = "unknown"     # independent of data_grain
    data_grain: str = "day"
    rollup_applied: str = ""             # native grain -> target grain
    freshness: str = "freshness_unknown"  # current | stale | missing | freshness_unknown
    staleness_seconds: Optional[float] = None
    tolerance_seconds: int = 0
    coverage_min: str = ""
    coverage_max: str = ""
    periods_covered: List[str] = dc_field(default_factory=list)
    fields_contributed: List[str] = dc_field(default_factory=list)
    rows_in: int = 0
    rows_kept: int = 0
    rows_dropped_as_future: int = 0
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class ReconciliationReport:
    as_of: str = ""
    canonical_grain: str = "day"
    grain_basis: str = "bootstrap_floor"   # contract | contract_floored | bootstrap_floor
    join_keys: List[str] = dc_field(default_factory=list)
    measures: List[str] = dc_field(default_factory=list)
    periods: List[str] = dc_field(default_factory=list)

    # -- what the DATA covers (a pure field/period fact; says nothing about
    #    which physical source is "required" — the contract does not know that)
    field_coverage: Dict[str, List[str]] = dc_field(default_factory=dict)
    # measure -> periods after its own last-covered period. Distinct from an
    # interior gap: a pending tail means "not caught up yet", which is the
    # normal, expected shape of a lagging feed rather than a data problem.
    periods_pending: Dict[str, List[str]] = dc_field(default_factory=dict)
    kpi_coverage: Dict[str, Dict[str, Any]] = dc_field(default_factory=dict)
    withheld_kpis: Dict[str, str] = dc_field(default_factory=dict)   # kpi_id -> reason

    # -- what each SOURCE offered (availability/freshness, kept separate)
    sources: List[SourceReport] = dc_field(default_factory=list)

    # -- how each measure was reconciled
    measure_modes: Dict[str, Dict[str, Any]] = dc_field(default_factory=dict)
    cells_single_source: int = 0
    cells_corroborated: int = 0
    cells_disputed: int = 0
    disagreements: List[Dict[str, Any]] = dc_field(default_factory=list)
    authoritative: Dict[str, str] = dc_field(default_factory=dict)   # measure -> source_id
    issues: List[Dict[str, Any]] = dc_field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["sources"] = [s.to_dict() for s in self.sources]
        return d

    def source(self, source_id: str) -> Optional[SourceReport]:
        return next((s for s in self.sources if s.source_id == source_id), None)


def _flag(cid: str, kind: str, severity: str, detail: str,
          fields: Optional[List[str]] = None) -> ConflictFlag:
    return ConflictFlag(conflict_id=cid, kind=kind, severity=severity, detail=detail,
                        affected_fields=fields or [])


def source_id_for(filename: str) -> str:
    """A stable identifier from the filename — the only 'source identity' input."""
    stem = os.path.splitext(os.path.basename(filename or "source"))[0]
    slug = re.sub(r"[^a-z0-9]+", "_", stem.strip().lower()).strip("_")
    return slug or "source"


# ---------------------------------------------------------------------------
# ① ingestion
# ---------------------------------------------------------------------------
def build_source(filename: str, frame: pd.DataFrame,
                 last_refresh_at: str = "", last_refresh_basis: str = "",
                 refresh_cadence: str = "unknown",
                 upload_time: str = "") -> SourceFile:
    """
    Describe one uploaded file using the existing detection path.

    `normalise_columns` is the whole of the naming reconciliation: strip,
    lowercase, spaces and hyphens to underscores. That is deliberate — anything
    more would be the semantic matching this design refuses to do.

    Refresh provenance is explicit. A timestamp the source system actually
    reported is preferred; failing that the latest event in the data
    (`data_max`) is the honest measure of how current the data is; failing that
    the upload time is recorded as a clearly-labelled inferred fallback and is
    never presented as a source refresh time. `refresh_cadence` is only ever
    declared or observed — it is never inferred from the row grain.
    """
    normalised = normalise_columns(frame)
    schema = detect_schema(normalised)

    basis = last_refresh_basis or ("source_reported" if last_refresh_at else "")
    stamp = last_refresh_at
    if not stamp:
        data_max = (schema.date_max or "").strip()
        if data_max:
            stamp, basis = data_max, "data_max"
        elif upload_time:
            stamp, basis = upload_time, "upload_time"
        else:
            basis = "unknown"

    return SourceFile(
        source_id=source_id_for(filename),
        label=os.path.splitext(os.path.basename(filename or "source"))[0].replace("_", " ").title(),
        filename=os.path.basename(filename or "source.csv"),
        frame=normalised,
        schema=schema,
        last_refresh_at=stamp,
        last_refresh_basis=basis or "unknown",
        refresh_cadence=refresh_cadence or "unknown",
        rows_in=int(len(normalised)),
    )


# ---------------------------------------------------------------------------
# ② structural / contract check
# ---------------------------------------------------------------------------
def check_compatibility(sources: Sequence[SourceFile]) -> Tuple[List[str], List[str], List[ConflictFlag]]:
    """
    Are these sources describing the same business, in the same vocabulary?

    Returns `(join_keys, measures, issues)`. A blocking issue means the sources
    must not be reconciled — never that a mapping should be invented.
    """
    issues: List[ConflictFlag] = []
    if len(sources) < 2:
        raise ReconciliationError("Reconciliation needs at least two sources.")

    # every source must carry a usable date column (detect_schema already
    # refused the file otherwise, so this is about them AGREEING on one)
    date_cols = {s.schema.date_column for s in sources}
    if len(date_cols) > 1:
        issues.append(_flag(
            "cf_source_date_column", "grain_mismatch", "blocking",
            "The sources use different date columns ("
            + ", ".join(f"{s.label}: '{s.schema.date_column}'" for s in sources)
            + "). Reconciliation needs one shared time field; it will not guess which "
              "columns correspond.",
        ))

    # identically-named columns must mean the same KIND of thing
    role_by_field: Dict[str, Dict[str, str]] = {}
    for s in sources:
        for name in s.schema.base_metrics + s.schema.extra_metrics:
            role_by_field.setdefault(name, {})[s.label] = "measure"
        for name in s.schema.dimensions:
            role_by_field.setdefault(name, {})[s.label] = "dimension"
    for name, roles in sorted(role_by_field.items()):
        if len(set(roles.values())) > 1:
            issues.append(_flag(
                f"cf_source_role_{name}", "unit_mismatch", "blocking",
                f"'{name}' is a measure in some sources and a dimension in others ("
                + ", ".join(f"{lbl}: {role}" for lbl, role in sorted(roles.items()))
                + "). The same field name must mean the same thing in every source.",
                fields=[name],
            ))

    # the business grain the sources agree to be joined on
    key_sets = []
    for s in sources:
        grain, _, _ = detect_row_grain(s.frame, s.schema.date_column, s.schema.dimensions)
        key_sets.append({k for k in grain if k != s.schema.date_column})
    shared_dims = sorted(set.intersection(*key_sets)) if key_sets else []

    if not shared_dims:
        # Falling back to the plain dimension intersection is legitimate: a
        # source whose row grain happens to be unique on date alone still has
        # dimensions that identify the business slice.
        shared_dims = sorted(set.intersection(*[set(s.schema.dimensions) for s in sources]))

    measures = sorted(set().union(*[set(s.schema.base_metrics) | set(s.schema.extra_metrics)
                                    for s in sources]))
    if not measures:
        issues.append(_flag(
            "cf_source_no_measures", "subset_check_failed", "blocking",
            "None of the sources expose a numeric measure to reconcile.",
        ))

    return shared_dims, measures, issues


# ---------------------------------------------------------------------------
# ④ refresh evaluation
# ---------------------------------------------------------------------------
def _tolerance_for(cadence: str) -> int:
    return _CADENCE_TOLERANCE_SECONDS.get(cadence, 0)


def evaluate_freshness(report: SourceReport, as_of: str) -> SourceReport:
    """
    Freshness of one source against `as_of`.

    Pure timestamp arithmetic, so it can be recomputed at any time without
    touching reconciled values. Staleness is asserted **only** when the cadence
    is actually known and the timestamp has a trustworthy basis; an inferred
    upload time says nothing about the source system, so it yields
    `freshness_unknown` rather than a confident verdict.
    """
    report.tolerance_seconds = _tolerance_for(report.refresh_cadence)
    as_of_ts = pd.to_datetime(as_of, errors="coerce", utc=True)
    refresh_ts = pd.to_datetime(report.last_refresh_at, errors="coerce", utc=True)

    if report.rows_kept == 0:
        report.freshness = "missing"
        report.note = report.note or "This source contributed no rows for the analysed range."
        return report

    trustworthy = report.last_refresh_basis in ("source_reported", "data_max")
    if pd.isna(refresh_ts) or pd.isna(as_of_ts):
        report.freshness = "freshness_unknown"
        report.note = "No usable refresh timestamp; freshness cannot be asserted."
        return report

    report.staleness_seconds = (as_of_ts - refresh_ts).total_seconds()
    if not trustworthy:
        report.freshness = "freshness_unknown"
        report.note = (f"Only an inferred {report.last_refresh_basis} timestamp is available, "
                       "which does not describe the source system; freshness cannot be asserted.")
    elif report.tolerance_seconds <= 0:
        report.freshness = "freshness_unknown"
        report.note = (f"Refresh cadence is '{report.refresh_cadence}', so there is no window to "
                       "judge staleness against; freshness cannot be asserted.")
    elif report.staleness_seconds > report.tolerance_seconds:
        report.freshness = "stale"
        report.note = (f"Last refreshed {report.last_refresh_at} ({report.last_refresh_basis}), "
                       f"beyond its {report.refresh_cadence} refresh window.")
    else:
        report.freshness = "current"
        report.note = ""
    return report


def refresh_sources(reports: Sequence[SourceReport], as_of: str) -> List[SourceReport]:
    """Re-evaluate every stored source report against a new `as_of`."""
    return [evaluate_freshness(r, as_of) for r in reports]


# ---------------------------------------------------------------------------
# ⑤ time alignment
# ---------------------------------------------------------------------------
def _grain_from_cadence(cadence: str) -> str:
    """`DatasetSchema.grain` speaks in cadences ("daily", "14-day"); alignment
    needs a period grain. Mirrors `kpi.service._time_grain_from_schema_grain`."""
    if cadence in _CADENCE_TO_GRAIN:
        return _CADENCE_TO_GRAIN[cadence]
    m = re.match(r"^(\d+)-day$", cadence or "")
    if m:
        days = int(m.group(1))
        return "day" if days <= 3 else "week" if days <= 10 else "month" if days <= 45 else "quarter"
    return "day"


def _coarser(a: str, b: str) -> str:
    return max(a, b, key=_GRAIN_ORDER.index)


def contract_target_grain(contract: Any) -> str:
    """
    The grain the KPI Contract asks for: the finest `time_grain` any approved
    KPI declares. Finest, because a frame at that grain also serves every
    coarser KPI — `observe` already rolls up when it slices periods.
    """
    grains = [k.granularity.time_grain for k in getattr(contract, "approved_kpis", [])
              if getattr(getattr(k, "granularity", None), "time_grain", None) in _GRAIN_ORDER]
    return min(grains, key=_GRAIN_ORDER.index) if grains else ""


def resolve_target_grain(sources: Sequence[SourceFile], contract: Any = None
                         ) -> Tuple[str, str, Optional[ConflictFlag]]:
    """
    Decide the grain the canonical view is built at.

    The **contract** is authoritative for what is wanted; the sources only
    constrain what is reachable, because a source may be rolled UP but never
    down — rolling down fabricates detail it does not have.

        desired = finest time_grain the contract's approved KPIs declare
        floor   = coarsest source data_grain
        target  = coarser_of(desired, floor)

    Returns `(target, basis, mismatch_flag)`. With no contract yet (first
    ingest) there is no declared ask, so the target is the floor — the only
    grain every source reaches without fabrication — and the contract
    subsequently bootstraps from that frame, so the two agree by construction.
    """
    floor = "day"
    for src in sources:
        floor = _coarser(floor, src.data_grain)

    desired = contract_target_grain(contract) if contract is not None else ""
    if not desired:
        return floor, "bootstrap_floor", None

    target = _coarser(desired, floor)
    if target == desired:
        return target, "contract", None

    coarse = [f"{s.label} is {s.data_grain}" for s in sources
              if _GRAIN_ORDER.index(s.data_grain) > _GRAIN_ORDER.index(desired)]
    flag = _flag(
        "cf_source_grain_mismatch", "grain_mismatch", "warning",
        f"The KPI Contract asks for {desired}-grain analysis, but " + "; ".join(coarse)
        + f". Coarser data is never rolled down to reach a finer grain, so the reconciled view "
          f"is built at {target}. Re-declare the KPI grain, or supply a finer source.",
    )
    return target, "contract_floored", flag


def _period_series(dates: pd.Series, grain: str) -> pd.Series:
    naive = dates.dt.tz_localize(None) if dates.dt.tz is not None else dates
    return naive.dt.to_period(_PERIOD_FREQ.get(grain, "D")).apply(
        lambda p: p.start_time.date().isoformat())


def _rollup_method(name: str, frame: pd.DataFrame) -> str:
    """
    How to carry a raw field to a coarser period, from the contract's own
    additivity vocabulary rather than a second table of rules.
    """
    series = pd.to_numeric(frame[name], errors="coerce").dropna()
    integral = bool(len(series)) and bool((series % 1 == 0).all())
    semantic = "count" if integral else "money"
    additivity, _ = infer_additivity(semantic, tokenise(name))
    return {"flow": "sum", "stock": "mean", "rate": "mean", "non_additive": "mean"}.get(
        additivity, "sum")


def _sum_preserving_missing(series: pd.Series) -> float:
    """
    Sum that stays missing when there is nothing to add.

    `Series([NaN, NaN]).sum()` is `0.0` in pandas, so a plain groupby-sum would
    turn "this source covers none of this period" into a hard zero — the exact
    fabrication reconciliation exists to prevent, and it would then look like a
    real value that disagrees with another source.
    """
    return series.sum(min_count=1)


def _agg_for(method: str):
    return _sum_preserving_missing if method == "sum" else method


# ---------------------------------------------------------------------------
# ⑦ reconciliation
# ---------------------------------------------------------------------------
def reconcile(sources: Sequence[SourceFile], as_of: str, contract: Any = None,
              authoritative: Optional[Dict[str, str]] = None
              ) -> Tuple[pd.DataFrame, ReconciliationReport]:
    """
    Stitch several same-vocabulary sources into one canonical business view.

    Raises `ReconciliationError` only when the sources are structurally
    incompatible — reconciling them would mean inventing a mapping, which this
    design refuses to do. A *disagreement* between sources is not an error: the
    ingest succeeds, the disputed cells are left missing, and a blocking
    `source_disagreement` conflict is recorded for the contract's existing
    resolution flow. `authoritative` maps measure -> source_id once a human has
    decided, and is applied when the view is rebuilt.
    """
    settings = get_settings()
    authoritative = dict(authoritative or {})
    join_keys, measures, issues = check_compatibility(sources)
    if any(i.severity == "blocking" for i in issues):
        raise ReconciliationError(
            "; ".join(i.detail for i in issues if i.severity == "blocking"), issues)

    grain, grain_basis, grain_flag = resolve_target_grain(sources, contract)
    if grain_flag is not None:
        issues.append(grain_flag)

    report = ReconciliationReport(as_of=as_of, canonical_grain=grain, grain_basis=grain_basis,
                                  join_keys=list(join_keys), measures=list(measures),
                                  authoritative=authoritative,
                                  issues=[i.model_dump() for i in issues])
    as_of_ts = pd.to_datetime(as_of, errors="coerce", utc=True)

    # -- normalise + align, per source. Roll UP only; never down. ---------
    rolled: Dict[str, pd.DataFrame] = {}
    for src in sources:
        frame = src.frame.copy()
        dates = pd.to_datetime(frame[src.schema.date_column], errors="coerce", utc=True)
        frame = frame.assign(_ts=dates).dropna(subset=["_ts"])

        # future-data guard: a source cannot be believed about a moment it has
        # not refreshed for, and none may speak past the analysis instant.
        refresh_ts = pd.to_datetime(src.last_refresh_at, errors="coerce", utc=True)
        cutoff = min([t for t in (refresh_ts, as_of_ts) if pd.notna(t)], default=None)
        dropped_future = 0
        if cutoff is not None:
            future = frame["_ts"] > cutoff
            dropped_future = int(future.sum())
            frame = frame[~future]

        frame["_period"] = _period_series(frame["_ts"], grain)
        present_keys = [k for k in join_keys if k in frame.columns]
        present_measures = [m for m in measures if m in frame.columns]
        for m in present_measures:
            frame[m] = pd.to_numeric(frame[m], errors="coerce")
        for k in present_keys:
            frame[k] = frame[k].astype(str)

        agg = {m: _agg_for(_rollup_method(m, frame)) for m in present_measures}
        grouped = (frame[present_keys + present_measures + ["_period"]]
                   .groupby(present_keys + ["_period"], as_index=False, dropna=False)
                   .agg(agg)) if present_measures else frame[present_keys + ["_period"]].drop_duplicates()
        rolled[src.source_id] = grouped

        report.sources.append(evaluate_freshness(SourceReport(
            source_id=src.source_id, label=src.label, filename=src.filename,
            last_refresh_at=src.last_refresh_at, last_refresh_basis=src.last_refresh_basis,
            refresh_cadence=src.refresh_cadence, data_grain=src.data_grain,
            rollup_applied=(f"{src.data_grain} -> {grain}" if src.data_grain != grain
                            else f"{grain} (native)"),
            coverage_min=(frame["_ts"].min().date().isoformat() if len(frame) else ""),
            coverage_max=(frame["_ts"].max().date().isoformat() if len(frame) else ""),
            periods_covered=sorted(set(grouped["_period"])) if len(grouped) else [],
            fields_contributed=present_measures,
            rows_in=src.rows_in, rows_kept=int(len(frame)),
            rows_dropped_as_future=dropped_future,
        ), as_of))

    # -- cell-level reconciliation ---------------------------------------
    labels = {s.source_id: s.label for s in sources}
    canonical, disagreements, stats, modes = _reconcile_cells(
        rolled, join_keys, measures, settings.source_agreement_tolerance_pct,
        authoritative, labels)

    report.cells_single_source = stats["single"]
    report.cells_corroborated = stats["corroborated"]
    report.cells_disputed = stats["disputed"]
    report.measure_modes = modes
    report.disagreements = disagreements

    # A disagreement is a decision for a human, so it becomes a contract
    # conflict rather than an ingest failure. The disputed cells stay missing,
    # which withholds the KPIs that need them until it is resolved.
    for flag in _disagreement_flags(disagreements, labels):
        report.issues.append(flag.model_dump())

    canonical = canonical.rename(columns={"_period": "date"})
    value_cols = [m for m in measures if m in canonical.columns]
    if value_cols:
        # trim: a period row no source populated at all is not data
        canonical = canonical[canonical[value_cols].notna().any(axis=1)].reset_index(drop=True)
    report.periods = sorted(set(canonical["date"])) if len(canonical) else []
    report.field_coverage = {
        m: sorted(set(canonical.loc[canonical[m].notna(), "date"])) if m in canonical.columns else []
        for m in measures
    }
    for m, covered_periods in report.field_coverage.items():
        if not covered_periods:
            continue
        tail = sorted(p for p in report.periods if p > covered_periods[-1])
        if tail:
            report.periods_pending[m] = tail
    return canonical, report


def _disagreement_flags(disagreements: List[Dict[str, Any]],
                        labels: Dict[str, str]) -> List[ConflictFlag]:
    """
    One blocking conflict per disputed measure, with a resolution option per
    contributing source — exactly the shape `kpi_service.resolve_conflict`
    already consumes, so the existing ConflictPanel resolves these with no new UI.

    `disagreements[*]["values"]` is keyed by display LABEL for readability, but
    `effect["authoritative_source"]` must carry the `source_id` — that is what
    `reconcile()`'s own `authoritative` map is keyed on when a resolved conflict
    is applied on rebuild. Storing the label there instead would silently make
    every resolution a no-op, since no source's `source_id` reads as its label.
    """
    from ..kpi.contract import ResolutionOption

    id_by_label = {v: k for k, v in labels.items()}

    by_measure: Dict[str, List[Dict[str, Any]]] = {}
    for d in disagreements:
        by_measure.setdefault(d["measure"], []).append(d)

    out: List[ConflictFlag] = []
    for measure, items in sorted(by_measure.items()):
        contributor_labels = sorted({src for d in items for src in d["values"]})
        first = items[0]
        where = ", ".join(f"{k}={v}" for k, v in first["keys"].items()) or "the whole business"
        reported = "; ".join(f"{src} reports {val:,.2f}" for src, val in first["values"].items())
        more = f" and {len(items) - 1} other cell(s)" if len(items) > 1 else ""
        out.append(ConflictFlag(
            conflict_id=f"cf_source_disagreement_{measure}",
            kind="source_disagreement", severity="blocking",
            detail=(f"Sources disagree about '{measure}' for {first['period']} ({where}): "
                    f"{reported}{more}. These are observations of the same fact, so they are "
                    f"never added together. Until one source is chosen as authoritative for "
                    f"'{measure}', the disputed cells stay missing and any KPI needing "
                    f"'{measure}' is withheld."),
            affected_fields=[measure],
            resolution_options=[
                ResolutionOption(
                    option_id=f"authoritative:{id_by_label.get(label, label)}",
                    label=f"Treat {label} as authoritative for '{measure}'",
                    detail=f"Rebuild the reconciled view using {label}'s value wherever sources disagree.",
                    effect={"authoritative_source": id_by_label.get(label, label), "measure": measure},
                )
                for label in contributor_labels
            ],
        ))
    return out


def _reconcile_cells(rolled: Dict[str, pd.DataFrame], join_keys: List[str],
                     measures: List[str], tolerance_pct: float,
                     authoritative: Dict[str, str], labels: Dict[str, str]
                     ) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, int],
                                Dict[str, Dict[str, Any]]]:
    """
    One pass over every (business key, period, measure) cell.

    The decision is per cell, so a single measure can be partitioned in some
    places and overlapping in others; such a measure is summarised as `mixed`.

        one source     take it
        many, agree    corroborate
        many, disagree leave the cell MISSING and record it

    A disputed cell is never resolved by guesswork. It stays missing, which
    withholds the KPIs that need it, until a human picks an authoritative
    source through the contract's conflict flow.
    """
    spine_frames = [f[[c for c in join_keys if c in f.columns] + ["_period"]]
                    for f in rolled.values()]
    spine = pd.concat(spine_frames, ignore_index=True).drop_duplicates()
    key_cols = [c for c in join_keys if c in spine.columns] + ["_period"]
    spine = spine.sort_values(key_cols).reset_index(drop=True)

    canonical = spine.copy()
    disagreements: List[Dict[str, Any]] = []
    stats = {"single": 0, "corroborated": 0, "disputed": 0}
    modes: Dict[str, Dict[str, Any]] = {}

    for measure in measures:
        contributors = {sid: f for sid, f in rolled.items() if measure in f.columns}
        if not contributors:
            continue

        aligned: Dict[str, pd.Series] = {}
        for sid, frame in contributors.items():
            on = [c for c in key_cols if c in frame.columns]
            merged = spine.merge(frame[on + [measure]], on=on, how="left")
            aligned[sid] = merged[measure].reset_index(drop=True)

        stacked = pd.DataFrame(aligned)
        n_sources = stacked.notna().sum(axis=1)
        values = pd.Series(np.nan, index=stacked.index, dtype="float64")

        # exactly one source: take it. Covers a partitioned cell and a
        # complementary-attribute cell identically — neither needs declaring.
        only_one = n_sources == 1
        n_single = int(only_one.sum())
        if n_single:
            values[only_one] = stacked[only_one].bfill(axis=1).iloc[:, 0]
            stats["single"] += n_single

        n_overlap = n_agree = n_disputed = 0
        overlap = n_sources > 1
        if overlap.any():
            sub = stacked[overlap]
            n_overlap = int(len(sub))
            lo, hi = sub.min(axis=1), sub.max(axis=1)
            denom = sub.abs().max(axis=1).replace(0.0, np.nan)
            spread_pct = ((hi - lo) / denom * 100.0).fillna(0.0)
            agrees = spread_pct <= tolerance_pct

            # agreeing overlap: corroborated, take the (equal) value. Never summed.
            agreed_idx = sub.index[agrees]
            if len(agreed_idx):
                values[agreed_idx] = sub.loc[agreed_idx].bfill(axis=1).iloc[:, 0]
            n_agree = int(agrees.sum())
            stats["corroborated"] += n_agree

            winner = authoritative.get(measure, "")
            disputed_idx = sub.index[~agrees]
            for idx in disputed_idx:
                row = sub.loc[idx]
                reporting = {sid: float(v) for sid, v in row.items() if pd.notna(v)}
                if winner and winner in row.index and pd.notna(row.get(winner)):
                    # a human has ruled; apply it and keep the record
                    values[idx] = float(row[winner])
                else:
                    n_disputed += 1
                disagreements.append({
                    "measure": measure,
                    "period": spine.loc[idx, "_period"],
                    "keys": {k: spine.loc[idx, k] for k in key_cols if k != "_period"},
                    "values": {labels.get(sid, sid): v for sid, v in reporting.items()},
                    "spread_pct": round(float(spread_pct.loc[idx]), 4),
                    "resolved_to": labels.get(winner, winner) if winner else None,
                })
            stats["disputed"] += n_disputed

        modes[measure] = {
            "mode": _classify_mode(n_single, n_overlap),
            "sources": sorted(labels.get(sid, sid) for sid in contributors),
            "cells_single_source": n_single,
            "cells_overlapping": n_overlap,
            "cells_corroborated": n_agree,
            "cells_disputed": n_disputed,
        }
        canonical[measure] = values.values

    return canonical, disagreements, stats, modes


def _classify_mode(single: int, overlapping: int) -> str:
    """
    How a measure was reconciled, reported rather than declared.

    `partition` and `complementary` are the same mechanism (one source per
    cell) and are distinguished only for the reader: a measure carried by
    several sources over disjoint cells is a partition; one carried by a single
    source is complementary to whatever the others bring.
    """
    if overlapping and single:
        return "mixed"
    if overlapping:
        return "overlap"
    return "partition_or_complementary"


# ---------------------------------------------------------------------------
# coverage evaluation — a field/period fact, kept separate from source freshness
# ---------------------------------------------------------------------------
def kpi_coverage(resolver: Dict[str, Any], canonical: pd.DataFrame,
                 report: ReconciliationReport, contract: Any = None) -> Dict[str, Dict[str, Any]]:
    """
    Which KPIs can honestly be computed, and which must be withheld.

    This is the guarantee that missing coverage never becomes a number.
    `CompiledKpi.compute` uses a bare `.sum()`, so a gap that reaches it either
    under-reports invisibly (`[100, NaN, 50].sum() == 150`) or fabricates a zero
    (`[NaN, NaN].sum() == 0.0`). Preserving NaN is therefore not protection on
    its own — a KPI is instead **withheld** unless every field its formula
    references is present for every period in the view. No number beats a wrong
    number.

    Two reasons to withhold, both recorded with an explicit message:
      * a referenced field has gaps (a source lags, or its cells are disputed)
      * the target grain is not in the KPI's own `valid_rollups`, i.e. the
        contract forbids serving it at this grain

    Note what this deliberately does NOT do: it never calls a physical source
    "required". The contract has no concept of one. It reports which fields are
    thin and, descriptively, which sources supply them.
    """
    if canonical is None or not len(canonical):
        return {}

    periods = set(report.periods)
    supplier: Dict[str, List[str]] = {}
    for src in report.sources:
        for f in src.fields_contributed:
            supplier.setdefault(f, []).append(src.label)

    covered = {m: set(ps) for m, ps in report.field_coverage.items()}
    disputed_fields = {d["measure"] for d in report.disagreements if not d.get("resolved_to")}

    out: Dict[str, Dict[str, Any]] = {}
    for key, compiled in (resolver or {}).items():
        needed = list(getattr(compiled, "source_fields", []) or [])
        if not needed:
            continue

        absent = [f for f in needed if f not in canonical.columns]
        thin = sorted({f for f in needed
                       if f in canonical.columns and covered.get(f, set()) != periods})
        computable = sorted(set.intersection(*[covered.get(f, set()) for f in needed])
                            & periods) if needed and not absent else []

        reason = ""
        if absent:
            reason = (f"'{key}' needs {', '.join(sorted(absent))}, which no source supplied.")
        elif thin:
            disputed = [f for f in thin if f in disputed_fields]
            if disputed:
                reason = (f"'{key}' needs {', '.join(disputed)}, where sources disagree. "
                          f"The disputed periods are missing until an authoritative source is "
                          f"chosen, so this KPI is withheld rather than computed from a gap.")
            else:
                who = sorted({lbl for f in thin for lbl in supplier.get(f, [])})
                reason = (f"'{key}' needs {', '.join(thin)}, which is missing for "
                          f"{len(periods) - len(computable)} of {len(periods)} periods"
                          + (f" (supplied by {', '.join(who)})" if who else "")
                          + ". Computing it would silently under-report, so it is withheld.")
        else:
            declared = _declared_grain(contract, key)
            reachable = _reachable_grains(report.canonical_grain)
            # A source may only be rolled UP. The frame is at `canonical_grain`,
            # so a KPI is servable only if its own declared grain is that grain
            # or coarser — never finer, which would mean fabricating detail the
            # sources do not have.
            if declared in _GRAIN_ORDER and declared not in reachable:
                reason = (f"'{key}' declares grain '{declared}', but these sources can only be "
                          f"reconciled to '{report.canonical_grain}' grain. Data is never rolled "
                          f"down to fabricate '{declared}'-level detail, so this KPI is withheld.")

        if reason:
            report.withheld_kpis[key] = reason

        out[key] = {
            "computable_periods": computable,
            "excluded_periods": sorted(periods - set(computable)),
            "fields_incomplete": thin or sorted(absent),
            "supplied_by": sorted({lbl for f in (thin or absent) for lbl in supplier.get(f, [])}),
            "withheld": bool(reason),
        }
    return out


def _kpi_definition(contract: Any, kpi_id: str) -> Any:
    for k in getattr(contract, "kpis", []) or []:
        if k.kpi_id == kpi_id:
            return k
    return None


def _reachable_grains(canonical_grain: str) -> List[str]:
    """Grains a KPI may be served at from a frame built at `canonical_grain` —
    itself and everything coarser, since a source may only roll UP."""
    if canonical_grain not in _GRAIN_ORDER:
        return list(_GRAIN_ORDER)
    return _GRAIN_ORDER[_GRAIN_ORDER.index(canonical_grain):]


def _declared_grain(contract: Any, kpi_id: str) -> str:
    kpi = _kpi_definition(contract, kpi_id)
    return getattr(getattr(kpi, "granularity", None), "time_grain", "") or "unknown"
