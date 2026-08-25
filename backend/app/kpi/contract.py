"""
The KPI Contract — the artefact this whole layer exists to produce.

Everything else in `app.kpi` serves these models. A contract is one versioned,
user-approved document per (user, dataset) that makes every KPI reproducible,
auditable and unambiguous for downstream systems.

Design notes worth knowing before editing:

* `FormulaSpec.kind` is deliberately `sum | mean | ratio` — the same three
  aggregation shapes the legacy `MetricSpec` supported — so a compiled contract
  entry is a drop-in replacement for it and the analysis engines need no rewrite.
  Expressiveness beyond that is carried by the expression strings, which the
  resolver parses into a typed AST (never `eval`).
* Nothing here silently resolves an ambiguity. Where the system cannot know the
  answer it records a `ConflictFlag` and refuses approval until a human decides.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# vocabularies
# ---------------------------------------------------------------------------
KpiType = Literal["general", "atomic", "derived", "user_defined"]
Unit = Literal["currency", "count", "percent", "ratio", "duration", "score"]

SemanticType = Literal[
    "money", "count", "rate_pct", "ratio", "duration",
    "score", "flag", "id", "category", "date", "unknown",
]

# flow  -> summable across time (revenue, admissions)
# stock -> a level; mean/last across time, never summed (inventory on hand)
# rate  -> never summed and never averaged; recompute from its components
Additivity = Literal["flow", "stock", "rate", "non_additive"]

FieldRole = Literal["measure", "dimension", "time", "identifier", "unknown"]

AggregationMethod = Literal["sum", "mean", "median", "min", "max", "last", "count_distinct", "ratio"]

# How a value at one time grain may legally be rolled up to a coarser one.
RollupPolicy = Literal["sum", "mean", "last", "recompute_from_components", "not_aggregatable"]

TimeGrain = Literal["day", "week", "month", "quarter", "year"]
TIME_GRAIN_ORDER: List[str] = ["day", "week", "month", "quarter", "year"]

KpiStatus = Literal["proposed", "approved", "rejected", "needs_confirmation"]
ContractStatus = Literal["draft", "provisional", "proposed", "approved", "superseded"]

KpiOrigin = Literal[
    "general_library",     # matched a concept in the shipped library
    "dynamic_atomic",      # a measure column in this dataset, directly measurable
    "dynamic_derived",     # produced by a typed derivation rule
    "llm_suggested",       # proposed by the screening model, re-validated deterministically
    "user_defined",        # authored or overridden by a human
]

ConflictSeverity = Literal["blocking", "warning", "info"]
ConflictKind = Literal[
    "duplicate_definition",
    "formula_disagreement",
    "grain_mismatch",
    "calendar_mismatch",
    "hierarchy_violation",
    "aggregation_ambiguity",
    "unit_mismatch",
    "source_disagreement",
    "subset_check_failed",
    "insufficient_history",
]


# ---------------------------------------------------------------------------
# computation
# ---------------------------------------------------------------------------
class FilterSpec(BaseModel):
    """A row filter applied before the KPI is aggregated."""

    field: str
    op: Literal["eq", "ne", "in", "not_in", "gt", "gte", "lt", "lte", "is_null", "not_null"]
    value: Any = None
    description: str = ""


class FormulaSpec(BaseModel):
    """
    How the KPI is computed from its source fields.

    `kind` mirrors the three aggregation shapes the analysis engines understand:

      sum    -> `expression` is evaluated per row and summed over the slice
      mean   -> `expression` is evaluated per row and averaged (a stock/level)
      ratio  -> numerator and denominator are each summed over the slice, then
                divided and multiplied by `scale`

    The ratio form is what keeps a rate correct under aggregation: summing the
    components and dividing once is not the same as averaging per-period ratios,
    and only the former is right.
    """

    expression: str
    kind: Literal["sum", "mean", "ratio"] = "sum"
    numerator_expression: Optional[str] = None
    denominator_expression: Optional[str] = None
    scale: float = 1.0


class AggregationSpec(BaseModel):
    method: AggregationMethod = "sum"
    rollup_policy: RollupPolicy = "sum"
    # Aggregating across dimension members (as opposed to across time).
    cross_sectional_policy: RollupPolicy = "sum"
    note: str = ""


# ---------------------------------------------------------------------------
# granularity, time, hierarchy
# ---------------------------------------------------------------------------
class GranularitySpec(BaseModel):
    """
    The KPI's intended level of detail. A first-class concept: two KPIs at
    different grains must not be compared, joined or aggregated together
    without an explicit alignment step.
    """

    entity_grain: List[str] = Field(default_factory=list)   # [] == whole business
    time_grain: TimeGrain = "month"
    native_row_grain: List[str] = Field(default_factory=list)   # the floor, from profiling
    valid_rollups: List[TimeGrain] = Field(default_factory=list)
    declared_by: Literal["inferred", "user", "library"] = "inferred"
    requires_confirmation: bool = False
    confidence: float = 0.5
    note: str = ""

    @property
    def label(self) -> str:
        entity = "-".join(self.entity_grain) if self.entity_grain else "company"
        return f"{entity}-{self.time_grain}"


class CalendarSpec(BaseModel):
    """A named calendar. Fiscal year vs calendar year is a definitional choice."""

    calendar_id: str = "gregorian"
    label: str = "Calendar year"
    fiscal_year_start_month: int = 1        # 1 == January == calendar year
    week_start: Literal["monday", "sunday"] = "monday"
    period_alignment: Literal["calendar", "445", "custom"] = "calendar"
    timezone: str = "UTC"


class TimeSemantics(BaseModel):
    date_field: str = ""
    date_role: Literal["event_date", "period_start", "period_end", "posting_date"] = "period_start"
    calendar_id: str = "gregorian"
    comparison_default: Literal["previous_period", "year_over_year"] = "previous_period"
    restatement_window_days: int = 0        # data may be revised after it lands
    note: str = ""


class ContainmentEvidence(BaseModel):
    child: str
    parent: str
    violations: int = 0
    checked_members: int = 0
    clean: bool = True


class HierarchySpec(BaseModel):
    name: str
    levels: List[str] = Field(default_factory=list)          # leaf -> root
    level_fields: Dict[str, str] = Field(default_factory=dict)
    containment_evidence: List[ContainmentEvidence] = Field(default_factory=list)
    rollup_allowed: bool = True


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------
class SourceCompleteness(BaseModel):
    date_min: str = ""
    date_max: str = ""
    row_count: int = 0
    null_rates: Dict[str, float] = Field(default_factory=dict)


class SourceBinding(BaseModel):
    """
    Where a KPI's data comes from. Modelled as a list everywhere so that two or
    three sources with different grains, calendars and refresh cadences can be
    expressed; the service constrains it to one until multi-source ingestion
    exists.
    """

    dataset_id: str
    source_label: str = ""
    fields: List[str] = Field(default_factory=list)
    native_grain: List[str] = Field(default_factory=list)
    calendar_id: str = "gregorian"
    refresh_cadence: str = "unknown"        # daily | weekly | monthly | adhoc | unknown
    completeness: SourceCompleteness = Field(default_factory=SourceCompleteness)
    precedence: int = 0                     # which source wins on disagreement


class ComparabilityEntry(BaseModel):
    """Whether this KPI may be compared/joined with another, and why not."""

    kpi_id: str
    comparable: bool
    reasons: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# rules, provenance, approval
# ---------------------------------------------------------------------------
class BusinessRule(BaseModel):
    statement: str
    enforced: bool = False
    machine_check: Optional[str] = None     # a FilterSpec-like check, when expressible
    source: str = ""


class ValidationRule(BaseModel):
    kind: Literal["non_negative", "range", "not_null", "monotonic", "subset_of", "sums_to"]
    detail: str = ""
    field: Optional[str] = None
    min: Optional[float] = None
    max: Optional[float] = None
    severity: ConflictSeverity = "warning"


class Provenance(BaseModel):
    origin: KpiOrigin
    derived_from: List[str] = Field(default_factory=list)     # library id, or source field names
    derivation_rule: Optional[str] = None
    computability_evidence: Dict[str, Any] = Field(default_factory=dict)
    screened_by: Optional[str] = None       # "deterministic" | "llm:<model>"
    screening_verdict: Optional[str] = None
    created_at: str = ""
    edited_by: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class ApprovalState(BaseModel):
    approved: bool = False
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    rejected_reason: Optional[str] = None
    user_confirmed_granularity: bool = False
    user_confirmed_definition: bool = False


class ResolutionOption(BaseModel):
    option_id: str
    label: str
    detail: str = ""
    effect: Dict[str, Any] = Field(default_factory=dict)


class ConflictFlag(BaseModel):
    """
    An ambiguity or contradiction the system refuses to resolve on its own.

    A `blocking` conflict prevents approval of the KPI it names, and of the
    contract as a whole, until a human chooses a resolution.
    """

    conflict_id: str
    kind: ConflictKind
    severity: ConflictSeverity = "warning"
    detail: str = ""
    affected_kpis: List[str] = Field(default_factory=list)
    affected_fields: List[str] = Field(default_factory=list)
    resolution_options: List[ResolutionOption] = Field(default_factory=list)
    resolved: bool = False
    resolved_by: Optional[str] = None
    resolved_at: Optional[str] = None
    resolution_option_id: Optional[str] = None
    resolution_rationale: str = ""


# ---------------------------------------------------------------------------
# the KPI definition
# ---------------------------------------------------------------------------
class KpiDefinition(BaseModel):
    """One KPI, completely specified. This is what makes a KPI reproducible."""

    kpi_id: str
    name: str
    business_definition: str = ""
    kpi_type: KpiType = "atomic"

    formula: FormulaSpec
    computation_note: str = ""

    sources: List[SourceBinding] = Field(default_factory=list)
    source_fields: List[str] = Field(default_factory=list)

    dimensions: List[str] = Field(default_factory=list)
    filters: List[FilterSpec] = Field(default_factory=list)

    unit: Unit = "count"
    higher_is_better: bool = True
    aggregation: AggregationSpec = Field(default_factory=AggregationSpec)
    granularity: GranularitySpec = Field(default_factory=GranularitySpec)
    time_semantics: TimeSemantics = Field(default_factory=TimeSemantics)
    hierarchy_refs: List[str] = Field(default_factory=list)

    business_rules: List[BusinessRule] = Field(default_factory=list)
    depends_on: List[str] = Field(default_factory=list)
    validation_rules: List[ValidationRule] = Field(default_factory=list)

    relevance: str = ""
    semantic_tags: List[str] = Field(default_factory=list)
    comparability: List[ComparabilityEntry] = Field(default_factory=list)

    provenance: Provenance
    confidence: float = 0.5
    required: bool = False
    status: KpiStatus = "proposed"
    approval: ApprovalState = Field(default_factory=ApprovalState)

    @property
    def is_active(self) -> bool:
        """Only an approved KPI is authoritative for downstream analysis."""
        return self.status == "approved"


# ---------------------------------------------------------------------------
# the contract
# ---------------------------------------------------------------------------
class UnavailableKpi(BaseModel):
    """A library KPI this dataset cannot support — recorded, never fabricated."""

    library_id: str
    name: str
    domain: str = "general"
    why_unavailable: str = ""
    missing_concepts: List[str] = Field(default_factory=list)


class RejectedCandidate(BaseModel):
    """A combination the system considered and declined, with the reason."""

    label: str
    expression: str = ""
    rule: str = ""
    reason: str = ""


class KpiContract(BaseModel):
    """The authoritative KPI definition set for one dataset."""

    contract_id: str
    uid: str
    dataset_id: str
    version: int = 1
    is_current: bool = True
    status: ContractStatus = "draft"

    dataset_label: str = ""
    detected_domains: List[str] = Field(default_factory=list)
    # The business context every KPI's explanation was grounded in (see
    # `app.kpi.domain`). `domain_uncertain=True` means the evidence did not
    # clearly point to one industry, and every explanation says so rather than
    # asserting one.
    domain_confidence: float = 0.0
    domain_uncertain: bool = True
    domain_evidence: List[str] = Field(default_factory=list)

    calendars: List[CalendarSpec] = Field(default_factory=lambda: [CalendarSpec()])
    hierarchies: List[HierarchySpec] = Field(default_factory=list)
    sources: List[SourceBinding] = Field(default_factory=list)

    kpis: List[KpiDefinition] = Field(default_factory=list)
    conflicts: List[ConflictFlag] = Field(default_factory=list)

    unavailable: List[UnavailableKpi] = Field(default_factory=list)
    rejected_candidates: List[RejectedCandidate] = Field(default_factory=list)

    field_profiles: List[Dict[str, Any]] = Field(default_factory=list)
    discovery_note: str = ""
    screened_by: str = "deterministic"

    created_at: str = ""
    updated_at: str = ""
    approved_at: Optional[str] = None
    approved_by: Optional[str] = None

    # -- lookups -----------------------------------------------------------
    def kpi(self, kpi_id: str) -> Optional[KpiDefinition]:
        return next((k for k in self.kpis if k.kpi_id == kpi_id), None)

    @property
    def approved_kpis(self) -> List[KpiDefinition]:
        return [k for k in self.kpis if k.status == "approved"]

    @property
    def blocking_conflicts(self) -> List[ConflictFlag]:
        return [c for c in self.conflicts if c.severity == "blocking" and not c.resolved]

    def blocking_conflicts_for(self, kpi_id: str) -> List[ConflictFlag]:
        return [c for c in self.blocking_conflicts if kpi_id in c.affected_kpis]

    def calendar(self, calendar_id: str) -> CalendarSpec:
        found = next((c for c in self.calendars if c.calendar_id == calendar_id), None)
        return found or CalendarSpec()

    def summary(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for k in self.kpis:
            counts[k.status] = counts.get(k.status, 0) + 1
        return {
            "contract_id": self.contract_id,
            "version": self.version,
            "status": self.status,
            "dataset_id": self.dataset_id,
            "kpi_count": len(self.kpis),
            "counts_by_status": counts,
            "conflict_count": len(self.conflicts),
            "blocking_conflicts": len(self.blocking_conflicts),
            "approvable": len(self.blocking_conflicts) == 0,
            "screened_by": self.screened_by,
            "detected_domains": self.detected_domains,
            "domain_confidence": self.domain_confidence,
            "domain_uncertain": self.domain_uncertain,
        }
