"""
The tool registry: JSON Schemas for every agent-layer tool, plus name ->
callable dispatch.

Every stateful module in this package has the identical constructor shape
`__init__(self, api: ContractAPI)`, and every public method takes `df` as its
own parameter -- confirmed across all twenty engines. That uniformity is what
makes a tool mechanical: `ToolSpec(engine=SomeEngine, method="some_method")`,
instantiated once per request from one `ContractAPI`, with `df` (and `api`,
for the handful of `introspect.py` functions with no engine behind them)
bound from context and never sent by the model.

**The one rule this registry exists to enforce mechanically:** every tool
argument the model gets wrong -- a hallucinated KPI key, an unknown
dimension, a malformed time filter -- becomes one of `agent.errors`' typed,
recoverable errors, never an unhandled traceback and never a silent wrong
number. `dispatch()` is the only place that boundary is drawn.
"""
from __future__ import annotations

import collections.abc
import dataclasses
import inspect
import logging
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple, Union

import pandas as pd

from ..engines.metrics import safe
from . import airlock, introspect
from .breakdown import DEFAULT_PERCENTILES, _ORDERS, BreakdownEngine
from .bridge import BridgeEngine
from .compare import SUPPORTED_BASELINES, ComparisonEngine
from .concentration import DEFAULT_TOP_K, ConcentrationEngine
from .confounders import COMPETING_SORT_KEYS, ConfounderEngine
from .consistency import ConsistencyEngine
from .contract_api import ContractAPI
from .correlate import MODES, CorrelationEngine
from .decompose import DecomposeEngine
from .drivers import DriverEngine
from .errors import AgentToolError, InvalidArgumentError
from .kpi_search import KpiSearch
from .quality import QualityEngine
from .query import QueryEngine
from .relations import RelationGraph
from .scan import ORDER_BY_CHOICES, ScanEngine
from .seasonality import CYCLE_LENGTH, SeasonalityEngine
from .segments import SegmentEngine
from .series import SERIES_GRAINS, SeriesEngine
from .significance import SignificanceEngine
from .temporal import TemporalEngine
from .timefilter import VALID_TYPES as TIME_FILTER_TYPES
from .trend import TrendEngine

log = logging.getLogger(__name__)

# Parameters the registry binds from request context, never from the model.
# An engine method's first parameter is always `df`; the handful of
# `introspect.py` functions with no engine behind them also take `api`.
_CONTEXT_PARAMS = {"api", "df"}

# Knobs the model should never choose -- internal tuning constants an engine
# already defaults sensibly. Hidden from every schema; the engine call simply
# never receives a value for these, so its own default applies.
_OMITTED_PARAMS = {"max_kpis", "max_members", "limit_cells", "min_points",
                   "baseline_periods", "k_sigma", "weights",
                   # `compare.significance(series=...)`: an internal
                   # already-computed-`Series` object passthrough to skip
                   # recomputation, not something JSON-representable or ever
                   # model-supplied.
                   "series"}


# ---------------------------------------------------------------------------
# shared schema fragments
# ---------------------------------------------------------------------------
# Flat, not an 8-branch discriminated `anyOf`: the model sends one object and
# `timefilter.parse` raises a typed, recoverable `MalformedTimeFilterError`
# naming the valid shapes when it gets the combination wrong. That is cheaper
# in tokens, repeated across ~40 tool schemas, than carrying the full union.
TIME_FILTER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "description": (
        "A time scope. `type` selects the shape; only that shape's other "
        "keys matter. quarter: {type, year, quarter (1-4)}. year: {type, "
        "year}. years: {type, values: [year, ...]} -- an arbitrary set, e.g. "
        "odd years. quarters: {type, values: [{year, quarter}, ...]}. "
        "range: {type, start, end} -- ISO dates, inclusive both ends. "
        "months: {type, start, end} -- 'YYYY-MM' strings, inclusive. "
        "latest: {type, grain (year|quarter|month|week), n} -- the trailing "
        "n periods actually held, not n calendar periods. all: {type} only."
    ),
    "properties": {
        "type": {"type": "string", "enum": list(TIME_FILTER_TYPES)},
        "year": {"type": "integer"},
        "quarter": {"type": "integer", "enum": [1, 2, 3, 4]},
        "values": {"type": "array", "items": {}},
        "start": {"type": "string"},
        "end": {"type": "string"},
        "grain": {"type": "string", "enum": ["year", "quarter", "month", "week"]},
        "n": {"type": "integer"},
    },
    "required": ["type"],
}


def _filters_schema(dimensions: Sequence[str]) -> Dict[str, Any]:
    """
    `filters` / `filter_a` / `filter_b`: dimension -> one value or a list of
    values (an implicit OR). `propertyNames.enum` is injected per request
    from this dataset's actual dimensions, so a hallucinated dimension name
    fails schema validation before it ever reaches `resolve_member`.
    """
    schema: Dict[str, Any] = {
        "type": "object",
        "description": "Map of dimension name -> member value, or a list of "
                      "member values to match any of (e.g. {\"region\": [\"East\", \"West\"]}).",
        "additionalProperties": {
            "anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}],
        },
    }
    if dimensions:
        schema["propertyNames"] = {"enum": list(dimensions)}
    return schema


# Per-parameter-name enums shared across every tool where that name means the
# same thing everywhere it appears (confirmed against each module's own
# validation, not assumed from the name alone).
_SHARED_ENUMS: Dict[str, Sequence[str]] = {
    "mode": MODES,
    "order_by": ORDER_BY_CHOICES,
    "ordered_by": COMPETING_SORT_KEYS,
    "direction": ("up", "down", "any"),
    "cause_direction": ("up", "down", "any"),
    "kpi_direction": ("up", "down", "any"),
    "grain": SERIES_GRAINS,
}

# The rare cases where the same parameter name means a genuinely different
# thing in different tools -- confirmed by reading each tool's own
# validation rather than assumed from the shared table above.
_PER_TOOL_ENUM_OVERRIDES: Dict[str, Dict[str, Sequence[str]]] = {
    # query.QueryEngine.query only ever validates order in {"asc","desc"};
    # breakdown.rank_entities also accepts "best"/"worst" (polarity-aware).
    "query_kpi": {"order": ("asc", "desc")},
    "rank_entities": {"order": _ORDERS},
    # Seasonality only ever validates a cycle length for quarter/month/week --
    # there is no yearly cycle to detect.
    "detect_seasonality": {"grain": tuple(CYCLE_LENGTH)},
    "compare_to_seasonal_norm": {"grain": tuple(CYCLE_LENGTH)},
}

# Complete, hand-written schema fragments for the handful of parameters that
# are neither a shared enum nor a KPI/dimension enum -- keyed by
# (tool_name, param_name) so a name collision with another tool's same-named,
# differently-shaped parameter can never leak an override across tools.
_PARAM_SCHEMA_OVERRIDES: Dict[Tuple[str, str], Dict[str, Any]] = {
    # The one engine method with no `df` at all: it re-filters candidate
    # dicts another tool call already produced (e.g. a scan's rows), not a
    # list of KPI keys -- so it must not get the KPI-enum treatment
    # `candidates` gets everywhere else it appears.
    ("filter_material_changes", "candidates"): {
        "type": "array",
        "description": "Candidate dicts from a prior tool call's output, each "
                       "with at least a 'kpi' key and a 'change_pct' key.",
        "items": {"type": "object"},
    },
    # 'formula' or a dataset dimension name -- not a fixed enum; the tool
    # itself raises a typed, recoverable error naming the dimensions when
    # given anything else.
    ("bridge_periods", "by"): {
        "type": "string",
        "description": "'formula' to bridge by formula component, or the "
                       "name of a dimension to bridge by its members.",
    },
}


# ---------------------------------------------------------------------------
# tool table
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ToolSpec:
    name: str
    group: str          # orient | retrieve | noise | scan | investigate | contest
    engine: Optional[type]      # the *Engine class to instantiate; None -> introspect.*
    method: str
    description: str


TOOL_SPECS: Tuple[ToolSpec, ...] = (
    # -- orient --------------------------------------------------------------
    ToolSpec("describe_dataset", "orient", None, "describe_dataset",
            "Call this first, once, before anything else: row count, date "
            "extent, every KPI and dimension this dataset has, and whether "
            "any metric was zero-filled rather than genuinely observed. "
            "Answers 'what can you see' and 'what date range does my data "
            "cover' in a single call."),
    ToolSpec("list_kpis", "orient", None, "list_kpis",
            "A fresh listing of every KPI this dataset measures. Call this "
            "only if you need the catalogue again after `describe_dataset`; "
            "it is already included there once, at the start."),
    ToolSpec("get_kpi_definition", "orient", None, "get_kpi_definition",
            "What a KPI means: its formula's business meaning, unit, "
            "direction (higher_is_better), and the raw fields it reads. "
            "Call this when the question asks what a KPI *means* -- e.g. "
            "'what does fulfilment rate mean here' -- not its value."),
    ToolSpec("list_dimensions", "orient", None, "list_dimensions",
            "Every dimension this dataset can be sliced or grouped by, with "
            "cardinality. Call this when a question names a way to slice "
            "the data (e.g. 'by region') and you need to confirm the exact "
            "dimension name before filtering or grouping by it."),
    ToolSpec("list_dimension_members", "orient", None, "list_dimension_members",
            "The actual values a dimension column holds (e.g. every region "
            "name). Call this before filtering by a member name you are not "
            "certain is spelled the way this dataset spells it."),
    ToolSpec("get_kpi_inputs", "orient", None, "get_kpi_inputs",
            "The raw source columns a KPI's formula reads, with no other "
            "semantic detail. Prefer `get_kpi_definition` unless you "
            "specifically need just the inputs."),
    ToolSpec("get_formula_structure", "orient", None, "get_formula_structure",
            "A KPI's formula decomposed into its components -- every field "
            "it reads, signed and placed (numerator/denominator/additive "
            "term), including fields reached through another KPI. Call this "
            "before `decompose_formula` when the question is about *how* a "
            "KPI is built (e.g. 'is margin a ratio, and what's in the "
            "denominator') rather than how it changed."),
    ToolSpec("search_kpis", "orient", KpiSearch, "search",
            "Find which KPI a phrase refers to, ranked by match strength. "
            "Call this when a question names a KPI informally (e.g. "
            "'fulfilment rate', 'churn') and you are not certain of its "
            "exact key -- this never blocks, it returns a ranked list even "
            "when the match is ambiguous."),
    ToolSpec("find_related_kpis", "orient", RelationGraph, "related",
            "Other KPIs related to one KPI -- by shared formula component, "
            "shared source field, or semantic tag. Call this to seed "
            "candidate drivers for an open question like 'what factors "
            "affect profit', before testing any of them."),
    ToolSpec("get_formula_components", "orient", RelationGraph, "components",
            "The KPIs that appear as components inside one KPI's own "
            "formula (AST-derived, so a component two levels down is still "
            "found). Narrower than `find_related_kpis`, which also finds "
            "KPIs related by shared field or tag rather than formula."),
    ToolSpec("get_kpi_neighbours", "orient", RelationGraph, "neighbours",
            "Just the KPI keys related to one KPI, with no relation detail -- "
            "a cheap first pass before `find_related_kpis` when you only "
            "need to know what exists nearby, not why."),

    # -- retrieve --------------------------------------------------------------
    ToolSpec("query_kpi", "retrieve", QueryEngine, "query",
            "The core retrieval tool: one or more KPIs, over any time scope, "
            "optionally filtered and grouped by dimension, sorted and "
            "limited. With no `group_by` this is the scalar getter for 'what "
            "was X in period Y'. Call this for any question with a named KPI "
            "and a defined scope -- it covers direct lookups, ranges, "
            "predicate sets like odd years, and group-by breakdowns in one "
            "tool."),
    ToolSpec("get_timeseries", "retrieve", SeriesEngine, "series",
            "One KPI's history at year/quarter/month/week grain, with gaps "
            "reported explicitly. Call this for a trajectory question -- "
            "'is X getting better or worse' -- before `detect_trend`, which "
            "needs this shape."),
    ToolSpec("get_timeseries_multi", "retrieve", SeriesEngine, "series_multi",
            "The same history for several KPIs at once, grain-aligned. Call "
            "this instead of several `get_timeseries` calls when you need "
            "more than one KPI's trajectory over the same scope."),
    ToolSpec("rank_entities", "retrieve", BreakdownEngine, "rank_entities",
            "Top-N or bottom-N members of a dimension by a KPI, with each "
            "member's share of the total. Call this for 'top 5 products by "
            "revenue' or 'which region has the lowest margin'."),
    ToolSpec("get_distribution", "retrieve", BreakdownEngine, "get_distribution",
            "Percentiles, spread and a histogram of a KPI across a "
            "dimension's members. Call this for 'what's the spread of order "
            "values' or 'median deal size by segment'."),
    ToolSpec("cross_tabulate", "retrieve", BreakdownEngine, "cross_tabulate",
            "A two-way pivot of one KPI across two dimensions at once. Call "
            "this for a multi-dimension slice like 'revenue for enterprise "
            "customers in APAC'."),

    # -- noise (is this signal or nothing) --------------------------------------
    ToolSpec("compare_periods", "noise", ComparisonEngine, "compare",
            "Compare one KPI between two arbitrary periods -- not limited to "
            "adjacent quarters or year-over-year. Call this for any "
            "period-over-period or entity-vs-entity comparison question."),
    ToolSpec("assess_significance", "noise", ComparisonEngine, "significance",
            "Whether a KPI's move is unusual against its own history (robust-"
            "z), not just whether it moved. Call this for 'is this dip "
            "normal or something to worry about' -- it can answer 'that's "
            "normal, ignore it' without ever needing an explanation."),
    ToolSpec("get_normal_range", "noise", ComparisonEngine, "normal_range",
            "The control band a KPI normally moves within, with no "
            "comparison period needed. Call this for 'is a 3% move unusual "
            "for us' when there is no specific baseline to compare against."),
    ToolSpec("filter_material_changes", "noise", QualityEngine, "filter_material",
            "Strip candidate changes below a materiality threshold. Call "
            "this after a scan to discard noise before investigating "
            "further -- pass the candidate list a prior tool call returned."),
    ToolSpec("check_data_quality", "noise", QualityEngine, "check_data_quality",
            "Whether an apparent change is a data artefact -- members "
            "appearing or disappearing, an unbalanced comparison window, or "
            "a zero-filled value indistinguishable from a real zero. Call "
            "this before trusting a surprising number, especially one that "
            "moved a lot."),
    ToolSpec("detect_trend", "noise", TrendEngine, "detect_trend",
            "A robust slope, direction and persistence over a KPI's history. "
            "Call this for 'is churn getting better or worse' -- there is no "
            "other way to answer a trajectory question."),
    ToolSpec("detect_changepoint", "noise", TrendEngine, "detect_changepoint",
            "When a KPI's level actually shifted, not just that it has. Call "
            "this for 'when did this start' as an observation about the "
            "past, before treating it as something to explain."),

    # -- scan (Tier-5: nothing is named yet) ------------------------------------
    ToolSpec("scan_kpis", "scan", ScanEngine, "scan_kpis",
            "Every KPI's move over one comparison, in one call. Call this "
            "first when no KPI is named -- 'what should I be worried about', "
            "'is anything unusual' -- to see what moved before deciding what "
            "to investigate."),
    ToolSpec("rank_kpis_by_movement", "scan", ScanEngine, "rank_kpis_by_movement",
            "Every KPI ranked by how unfavourable its move is (aware of "
            "which direction is bad for each KPI), or by raw magnitude. Call "
            "this after `scan_kpis` to decide which KPI is worst first."),
    ToolSpec("scan_anomalies", "scan", ScanEngine, "scan_anomalies",
            "Every KPI checked for a pattern break, not just a large move. "
            "Call this for 'is anything unusual in the data' or 'show me "
            "anything that broke from its normal pattern'."),
    ToolSpec("scan_dimension_outliers", "scan", ScanEngine, "scan_dimension_outliers",
            "Which members of a dimension moved unusually for one KPI. Call "
            "this once a scan has narrowed to one KPI and you need to know "
            "which segment is behind it."),

    # -- investigate (produce explanations) -------------------------------------
    ToolSpec("decompose_by_dimension", "investigate", DecomposeEngine, "decompose_by_dimension",
            "Which members of a dimension drove a KPI's change between two "
            "periods, with each member's contribution. Call this for 'where "
            "did the change come from' once a dimension is in scope."),
    ToolSpec("decompose_rate_mix", "investigate", DecomposeEngine, "decompose_rate_mix",
            "Splits a ratio KPI's change into a rate effect and a mix "
            "effect across a dimension. Call this specifically for 'is it "
            "rate or mix' -- e.g. did margin move because each segment's "
            "own margin changed, or because the sales mix shifted toward "
            "lower-margin segments."),
    ToolSpec("decompose_nested", "investigate", DecomposeEngine, "decompose_nested",
            "Drill one level deeper: decompose a KPI's change within one "
            "member of an outer dimension, by an inner dimension. Call this "
            "for 'APAC is down -- which product inside APAC' after an outer "
            "decomposition has already named the member."),
    ToolSpec("decompose_formula", "investigate", DecomposeEngine, "decompose_formula",
            "Splits a KPI's change into the effect of each term in its own "
            "formula (e.g. price vs. volume for revenue), with an explicit "
            "interaction/residual term. Call this for 'is the margin "
            "decline coming from pricing, cost, or mix' when the cause is "
            "structural rather than driven by one dimension."),
    ToolSpec("attribute_dimensions", "investigate", DriverEngine, "attribute_dimensions",
            "Exact Shapley attribution of a KPI's change across every "
            "dimension at once, answering *which axis* matters most. Call "
            "this when several dimensions could plausibly explain a change "
            "and you need to know which one actually does."),
    ToolSpec("compute_over_index", "investigate", DriverEngine, "compute_over_index",
            "Whether one member of a dimension moved more than its size "
            "alone would imply. Call this to test a single named suspect -- "
            "narrower than `rank_drivers`, which ranks every member."),
    ToolSpec("rank_drivers", "investigate", DriverEngine, "rank_drivers",
            "The dimension members most responsible for a KPI's change, "
            "ranked by a composite of contribution, over-index, robust-z "
            "and persistence. Call this for open driver enumeration -- "
            "'what's driving churn' -- once a KPI and a period pair are "
            "named."),
    ToolSpec("bridge_periods", "investigate", BridgeEngine, "bridge_periods",
            "An additive waterfall reconciling a KPI from one period to "
            "another, by dimension or by formula component, with an "
            "explicit unexplained residual. Call this when the question "
            "wants a full reconciliation of a change, not just its biggest "
            "driver."),
    ToolSpec("measure_concentration", "investigate", ConcentrationEngine, "measure_concentration",
            "Real concentration (HHI, Gini, top-k share) of a KPI across a "
            "dimension's members. Call this for 'what share of revenue "
            "comes from our top 3 customers' or to check whether a "
            "dimension is dominated by a handful of members."),
    ToolSpec("find_outlier_contributors", "investigate", ConcentrationEngine, "find_outlier_contributors",
            "Members of a dimension that contributed disproportionately to "
            "a KPI's change relative to their own size. Call this to find "
            "which specific members are punching above or below their "
            "weight."),
    ToolSpec("compare_segments", "investigate", SegmentEngine, "compare_segments",
            "Why two named members of one dimension differ on a KPI (e.g. "
            "APAC vs. EMEA on margin). Call this for a named entity-vs-"
            "entity divergence question."),
    ToolSpec("compare_cohorts", "investigate", SegmentEngine, "compare_cohorts",
            "A generalised two-group comparison defined by arbitrary "
            "filters rather than one dimension's members. Call this when "
            "the two groups being compared cut across more than one "
            "dimension (e.g. enterprise-APAC vs. SMB-EMEA)."),
    ToolSpec("segment_by_behavior", "investigate", SegmentEngine, "segment_by_behavior",
            "Buckets a dimension's members into grew/flat/declined/"
            "appeared/disappeared between two periods. Call this for "
            "'which parts held up' when you need the whole population "
            "banded, not just the extremes a ranking would show."),
    ToolSpec("detect_seasonality", "investigate", SeasonalityEngine, "detect_seasonality",
            "Whether a KPI has a real seasonal pattern at all. Call this "
            "before treating a periodic-looking move as a problem -- "
            "confusing a seasonal dip for a real one is the most common "
            "false alarm in a multi-factor investigation."),
    ToolSpec("compare_to_seasonal_norm", "investigate", SeasonalityEngine, "compare_to_seasonal_norm",
            "Compares a period against the same season in prior years "
            "rather than the immediately preceding one. Call this once "
            "`detect_seasonality` confirms a real seasonal pattern, to see "
            "whether the current move is normal for this time of year."),
    ToolSpec("correlate_kpis", "investigate", CorrelationEngine, "correlate_kpis",
            "Whether two named KPIs move together, cross-sectionally across "
            "a dimension or over time. Call this for a single named-cause "
            "check -- 'did the price increase hurt volume'."),
    ToolSpec("correlate_kpi_matrix", "investigate", CorrelationEngine, "correlate_kpi_matrix",
            "One KPI correlated against many candidates in a single call. "
            "Call this for 'what factors affect profit' instead of calling "
            "`correlate_kpis` once per candidate -- it is the one-to-many "
            "sweep this kind of question needs."),

    # -- contest (verify explanations) -------------------------------------------
    ToolSpec("check_temporal_precedence", "contest", TemporalEngine, "check_temporal_precedence",
            "Whether a candidate cause's move actually preceded the KPI's "
            "own move, not just correlates with it. Call this to stress-"
            "test a finding before trusting its direction of causation."),
    ToolSpec("cross_correlate_lagged", "contest", TemporalEngine, "cross_correlate_lagged",
            "Correlation between two KPIs at a range of time lags, to find "
            "which one moves first. Call this for 'what's the earliest "
            "warning sign' or to locate the lag a precedence check should "
            "test at."),
    ToolSpec("test_reverse_causation", "contest", TemporalEngine, "test_reverse_causation",
            "Whether the reverse direction (KPI causing the candidate, not "
            "the other way round) fits the data at least as well. Call this "
            "before accepting a causal-sounding correlation at face value."),
    ToolSpec("find_counterexamples", "contest", ConsistencyEngine, "find_counterexamples",
            "Members of a dimension where a candidate cause's usual "
            "relationship to the KPI does not hold. Call this to check "
            "whether an explanation holds everywhere or only in the cases "
            "that first suggested it."),
    ToolSpec("test_consistency_across_dimension", "contest", ConsistencyEngine, "test_consistency_across_dimension",
            "How strongly a candidate cause's relationship to the KPI holds "
            "per member of a dimension, pooled into one verdict. Broader "
            "than `find_counterexamples`, which only lists violations."),
    ToolSpec("test_holdout_segments", "contest", ConsistencyEngine, "test_holdout_segments",
            "Splits a dimension into members the candidate cause affected "
            "and members it did not, and compares their behaviour. Call "
            "this as a difference-in-differences style check when a clean "
            "affected/unaffected split exists."),
    ToolSpec("test_confounders", "contest", ConfounderEngine, "test_confounders",
            "Whether a candidate cause's correlation with the KPI survives "
            "controlling for other candidate variables (partial "
            "correlation). Call this before trusting a bivariate "
            "correlation as causal -- a third variable may be driving both."),
    ToolSpec("test_spurious_correlation", "contest", ConfounderEngine, "test_spurious_correlation",
            "Whether a KPI/candidate relationship survives detrending, or "
            "is just two things that both happen to trend over time. "
            "Time-series only -- a cross-sectional request is refused with "
            "a typed error, since detrending a snapshot means nothing."),
    ToolSpec("rank_competing_explanations", "contest", ConfounderEngine, "rank_competing_explanations",
            "Several candidate causes tested on the same evidence basis and "
            "compared -- with the ordering key it used published alongside "
            "the results, never a hidden score. Call this once you have "
            "more than one plausible explanation and need to compare them "
            "fairly."),
    ToolSpec("test_statistical_significance", "contest", SignificanceEngine, "test_statistical_significance",
            "p-value and confidence interval for a correlation, refusing to "
            "report one below a minimum sample size. Call this before "
            "treating any correlation this loop found as more than "
            "suggestive."),
    ToolSpec("check_sample_adequacy", "contest", SignificanceEngine, "check_sample_adequacy",
            "Whether there is enough data to conclude anything at all about "
            "a KPI's relationship to a dimension, before running a test "
            "that would otherwise report a number nobody should trust."),
)

_SPECS_BY_NAME: Dict[str, ToolSpec] = {spec.name: spec for spec in TOOL_SPECS}
assert len(_SPECS_BY_NAME) == len(TOOL_SPECS), "duplicate tool name in TOOL_SPECS"


# ---------------------------------------------------------------------------
# schema generation
# ---------------------------------------------------------------------------
def _bound_callable(spec: ToolSpec, engine_instances: Mapping[type, Any]) -> Callable[..., Any]:
    if spec.engine is None:
        return getattr(introspect, spec.method)
    return getattr(engine_instances[spec.engine], spec.method)


def _resolved_hints(fn: Callable[..., Any]) -> Dict[str, Any]:
    """
    `typing.get_type_hints`, not `inspect.signature(...).annotation`.

    Every module in this package has `from __future__ import annotations`
    (PEP 563), so `inspect.signature` alone hands back the *string* source of
    each annotation (`'Optional[Sequence[str]]'`) rather than the resolved
    type. `get_type_hints` evaluates those strings against the function's own
    module globals, which is what actually lets `typing.get_origin`/
    `get_args` work on the result.
    """
    try:
        return typing.get_type_hints(fn)
    except Exception:                                   # pragma: no cover - defensive
        log.warning("registry: could not resolve type hints for %r", fn)
        return {}


def _context_and_model_params(fn: Callable[..., Any]) -> Tuple[Set[str], List[inspect.Parameter]]:
    """Split a tool callable's parameters into what the registry binds from
    context (`api`, `df`) and what the model actually supplies."""
    params = list(inspect.signature(fn).parameters.values())
    context = {p.name for p in params if p.name in _CONTEXT_PARAMS}
    model_params = [p for p in params if p.name not in _CONTEXT_PARAMS
                    and p.name not in _OMITTED_PARAMS]
    return context, model_params


def _unwrap_optional(annotation: Any) -> Any:
    """`Optional[X]` (i.e. `Union[X, None]`) -> `X`. Any other annotation,
    including a genuine multi-member `Union`, passes through unchanged."""
    origin = typing.get_origin(annotation)
    if origin is Union:
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _base_schema_for(annotation: Any) -> Dict[str, Any]:
    """
    The generic, type-hint-driven fallback -- used only once a parameter has
    already failed every name-based special case below. Every annotation in
    this package that reaches here is a plain `str`/`int`/`float`/`bool` or a
    `Sequence[...]` of one; anything stranger (a bare `Mapping[str, Any]`,
    a `TimeFilter`) is always caught by name first.
    """
    ann = _unwrap_optional(annotation)
    if ann is str:
        return {"type": "string"}
    if ann is int:
        return {"type": "integer"}
    if ann is float:
        return {"type": "number"}
    if ann is bool:
        return {"type": "boolean"}
    origin = typing.get_origin(ann)
    # `from __future__ import annotations` means these are resolved via
    # `typing.get_type_hints`, whose origin for a `Sequence[...]`/`List[...]`
    # annotation is the `collections.abc` ABC, not the bare `typing.Sequence`
    # alias -- both forms are checked so either spelling in source resolves.
    if origin in (list, tuple, collections.abc.Sequence, collections.abc.MutableSequence,
                 Sequence, typing.Sequence, typing.List, typing.Tuple):
        args = typing.get_args(ann)
        item_ann = args[0] if args else str
        return {"type": "array", "items": _base_schema_for(item_ann)}
    if origin is Union:
        return {"anyOf": [_base_schema_for(a) for a in typing.get_args(ann)
                          if a is not type(None)]}
    log.warning("registry: no schema rule for annotation %r -- defaulting to string", annotation)
    return {"type": "string"}


def _schema_for_param(tool_name: str, name: str, annotation: Any,
                      available_kpis: Sequence[str], dimensions: Sequence[str]) -> Dict[str, Any]:
    if name in ("time_filter", "period_a", "period_b", "baseline"):
        return TIME_FILTER_SCHEMA
    if name in ("filters", "filter_a", "filter_b"):
        return _filters_schema(dimensions)

    override = _PARAM_SCHEMA_OVERRIDES.get((tool_name, name))
    if override is not None:
        return override

    if name in ("kpi_key", "kpi_a", "kpi_b", "cause_kpi"):
        return {"type": "string", "enum": list(available_kpis)}
    if name == "kpi_keys":
        # `QueryEngine.query`'s own annotation is `Union[str, Sequence[str]]`
        # -- a single KPI key needs no wrapping array. `candidates` /
        # `candidate_causes` below are `Sequence[str]`-only everywhere they
        # appear, so they do not get this `anyOf`; A3's airlock is what
        # caught this schema being array-only for `kpi_keys` when the
        # engine itself always accepted a bare string.
        kpi_enum = {"type": "string", "enum": list(available_kpis)}
        return {"anyOf": [kpi_enum, {"type": "array", "items": kpi_enum}]}
    if name in ("candidates", "candidate_causes"):
        return {"type": "array", "items": {"type": "string", "enum": list(available_kpis)}}
    if name in ("dimension", "dim_a", "dim_b", "outer_dimension", "inner_dimension"):
        return {"type": "string", "enum": list(dimensions)}

    enum_values = _PER_TOOL_ENUM_OVERRIDES.get(tool_name, {}).get(name) or _SHARED_ENUMS.get(name)
    if enum_values is not None:
        return {"type": "string", "enum": list(enum_values)}

    return _base_schema_for(annotation)


def build_input_schema(spec: ToolSpec, fn: Callable[..., Any],
                       available_kpis: Sequence[str], dimensions: Sequence[str]) -> Dict[str, Any]:
    _, model_params = _context_and_model_params(fn)
    hints = _resolved_hints(fn)
    properties: Dict[str, Any] = {}
    required: List[str] = []
    for param in model_params:
        annotation = hints.get(param.name, param.annotation)
        properties[param.name] = _schema_for_param(spec.name, param.name, annotation,
                                                   available_kpis, dimensions)
        if param.default is inspect.Parameter.empty:
            required.append(param.name)
    schema: Dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


# ---------------------------------------------------------------------------
# result serialisation
# ---------------------------------------------------------------------------
def _deep_safe(obj: Any) -> Any:
    """Recursively apply `metrics.safe` (NaN/inf/numpy scalar -> JSON-safe)
    to every leaf of an arbitrarily nested tool result. `safe` itself only
    cleans one scalar; every tool result is a nested dict/list/tuple of
    dataclasses-turned-dicts, so this is the pass that actually reaches G4's
    "no NaN/inf reaches JSON" everywhere in the tree, not just at the top."""
    if isinstance(obj, dict):
        return {k: _deep_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_deep_safe(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_deep_safe(v) for v in obj)
    return safe(obj)


def _serialise(result: Any) -> Any:
    """
    A tool result to plain, JSON-safe data -- recursively, since a result is
    not always a single dataclass: `series_multi` returns `Dict[str, Series]`,
    a plain dict whose *values* are dataclasses, and a list of dataclasses
    (e.g. `RelationGraph.related -> List[Relation]`) is just as common.

    Most result dataclasses already carry `to_payload()`; the eleven that do
    not (`KpiInfo`, `DimensionInfo`, `MemberMatch`, `MemberPage`, `Term`,
    `FormulaStructure`, `Relation`, `SearchEntry`, `KpiMatch`, `SearchResult`,
    `PairedSample`) fall through to `dataclasses.asdict`. Plain dicts (every
    `introspect.py` function) recurse structurally instead.
    """
    to_payload = getattr(result, "to_payload", None)
    if callable(to_payload):
        return _deep_safe(to_payload())
    if dataclasses.is_dataclass(result) and not isinstance(result, type):
        return _deep_safe(dataclasses.asdict(result))
    if isinstance(result, dict):
        return {k: _serialise(v) for k, v in result.items()}
    if isinstance(result, (list, tuple)):
        return [_serialise(item) for item in result]
    return _deep_safe(result)


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------
# Every result dataclass has a `to_payload()`; a tool that returns a list of
# them (e.g. `RelationGraph.related` -> `List[Relation]`) needs the list
# branch of `_serialise` above, not a bare `to_payload()` call on the list.

@dataclass(frozen=True)
class ToolCallResult:
    payload: Any
    is_error: bool


class ToolRegistry:
    """
    Built once per request from one `ContractAPI` and its dataframe.

    `.tools` is the Anthropic `tools=` list (JSON Schemas, enums resolved
    against this specific dataset). `.dispatch(name, args)` is the
    `executor` `LLMClient._call_with_tools` expects: it never raises past
    this boundary -- a bad tool name, a bad argument, or an unexpected
    engine exception all become a typed, recoverable `(payload, is_error)`.
    """

    def __init__(self, api: ContractAPI, df: pd.DataFrame):
        self._api = api
        self._df = df
        self._engine_instances: Dict[type, Any] = {}
        engine_classes = {spec.engine for spec in TOOL_SPECS if spec.engine is not None}
        for cls in engine_classes:
            self._engine_instances[cls] = cls(api)

        available_kpis = sorted(api.available_keys(df))
        dimensions = sorted(d.name for d in api.list_dimensions(df))

        self._callables: Dict[str, Callable[..., Any]] = {}
        self._context_params: Dict[str, Set[str]] = {}
        self._schemas: Dict[str, Dict[str, Any]] = {}
        self.tools: List[Dict[str, Any]] = []
        for spec in TOOL_SPECS:
            fn = _bound_callable(spec, self._engine_instances)
            context, _ = _context_and_model_params(fn)
            schema = build_input_schema(spec, fn, available_kpis, dimensions)
            self._callables[spec.name] = fn
            self._context_params[spec.name] = context
            self._schemas[spec.name] = schema
            self.tools.append({
                "name": spec.name,
                "description": spec.description,
                "input_schema": schema,
            })
        # The tool array is the largest stable prefix of every request in the
        # loop -- one cache breakpoint on the last definition lets every turn
        # after the first read it from cache instead of re-paying for it.
        if self.tools:
            self.tools[-1] = {**self.tools[-1], "cache_control": {"type": "ephemeral"}}

    def by_group(self, group: str) -> List[Dict[str, Any]]:
        return [t for t in self.tools if _SPECS_BY_NAME[t["name"]].group == group]

    def tool_names(self) -> List[str]:
        return [t["name"] for t in self.tools]

    def dispatch(self, name: str, args: Mapping[str, Any]) -> Tuple[Any, bool]:
        """The `executor` `LLMClient._call_with_tools` calls. Never raises."""
        fn = self._callables.get(name)
        if fn is None:
            error = InvalidArgumentError("tool", name, "No tool with this name is registered.",
                                         valid_alternatives=self.tool_names())
            return error.to_payload(), True

        try:
            airlock.validate(name, self._schemas[name], args)
        except AgentToolError as exc:
            # A malformed argument must never reach the engine at all -- see
            # `agent/airlock.py` for the three measured failure modes this
            # guards (a wrong-typed argument executing into a plausible
            # empty answer, a list reaching pandas as a dict update, a
            # string iterated character by character).
            return exc.to_payload(), True

        context = self._context_params[name]
        kwargs: Dict[str, Any] = dict(args)
        if "api" in context:
            kwargs["api"] = self._api
        if "df" in context:
            kwargs["df"] = self._df

        try:
            result = fn(**kwargs)
        except AgentToolError as exc:
            return exc.to_payload(), True
        except TypeError as exc:
            # A malformed argument shape that got past JSON Schema (e.g. the
            # model omitted a required key `strict` validation would have
            # caught) surfaces here as a Python TypeError -- still a
            # recoverable turn, never a 500.
            error = InvalidArgumentError(name, args, f"Could not call this tool with these "
                                         f"arguments: {exc}")
            return error.to_payload(), True
        except Exception as exc:                      # pragma: no cover - defensive
            log.exception("Unexpected error executing tool %s", name)
            return {"error": "tool_error", "message": str(exc)}, True

        return _serialise(result), False


def build(api: ContractAPI, df: pd.DataFrame) -> ToolRegistry:
    return ToolRegistry(api, df)
