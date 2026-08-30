"""
Score the agent loop against the question taxonomy (task A8).

Runs the REAL loop -- `agent.loop.answer` over all 56 registered tools, with a
live model -- across seventeen questions spanning Tiers 1-5 of the master plan's
taxonomy, then scores each run on what it *did*, never on how it worded the
answer.

    python scripts/validate_agent.py         # print the scorecard, write the report
    make validate-agent                      # same

Two things this produces that nothing else can:

1. **A tool-usage report.** Which of the 56 tools actually fire, and which never
   do, grouped by registry group. `DECISIONS_TAKEN.md` decision 60 defers two
   open questions to exactly this evidence: whether the Contest robustness
   family (task X5) earns its place, and whether the loop needs progressive
   tool disclosure (`registry.by_group` is built and deliberately unused).
2. **Per-tier turn and cost budgets**, measured rather than guessed.

--------------------------------------------------------------------------
The inversion, versus `validate_rca.py`
--------------------------------------------------------------------------
The RCA triangle is `ground_truth.json` (input, emitted by the sample-data
generator from counterfactual re-simulations) -> `validate_rca.py`
(instrument) -> a scorecard. **This triangle runs the other way.** The cases
live in Python, right here in `CASES`, and the JSON is the *output*.

That is deliberate. Every Tier-1/2 expectation is a *computation over the
dataframe*, and it has to stay one: a hand-typed JSON number would be either
circular (someone ran `query_kpi` to get it) or stale the next time the sample
CSV is regenerated. Expectations are plain pandas over `raw_frame()`, which
reads the CSV and nothing else -- no `metrics.prepare`, no `ContractAPI`, no
`CompiledKpi.compute`. That shared-nothing property is what "independently
computed" means here, and `test_tier_eval.py` asserts it rather than trusting
this docstring.

--------------------------------------------------------------------------
What this measures, and what it does not
--------------------------------------------------------------------------
A Tier-1/2 value check asks: **did the agent retrieve the right number?** It
looks for the expected figure among the *tool results*, because that is a fact
about what ran. It does **not** verify that the prose quotes that number
correctly -- and it cannot, because any such check would be an assertion about
wording, which this harness makes structurally impossible (see `Observed`).
The live scorecard prints the prose for a human to read; the report stores only
its length. Do not mistake a green run for a claim about the writing.

`backend/tests/test_tier_eval.py` tests this module headless, with no API key.
It cannot assert the *reading* the way `test_rca_ground_truth.py` asserts
`validate_rca`'s -- that pipeline is deterministic and keyless, and this one is
neither -- so it asserts the *instrument*: that the manifest is sound, that the
expectations are genuinely independent, and that the scorer fails a wrong run.
`main()` is the only untested part, exactly as `validate_rca.main()` is.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
for path in (str(BACKEND), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from app.agent.registry import TOOL_SPECS                        # noqa: E402

SAMPLE_CSV = ROOT / "sample_data" / "business_metrics_sample.csv"
TRUTH_PATH = ROOT / "sample_data" / "ground_truth.json"
REPORT_PATH = ROOT / "docs" / "agent_eval.json"

# --- the bars ---------------------------------------------------------------
# `loop.answer` counts the final `end_turn` message as a turn, so one tool call
# plus its answer is TWO turns (pinned by test_agent_loop.py:47). A tier-1 bar
# of 1 would therefore fail always; 2 is "one retrieval, then answer".
MAX_TURNS_BY_TIER = {1: 2, 2: 3, 3: 6, 4: 12, 5: 12}
MAX_TOOL_CALLS_BY_TIER = {1: 2, 2: 3, 3: 6, 4: 14, 5: 16}
VALUE_TOLERANCE_PCT = 0.5
# Tier 5 names nothing, so the agent must at minimum scan and then look into
# something -- two registry groups. Hard here; only reported on tier 4, where a
# model converging in two groups instead of three may simply be more efficient.
MIN_GROUPS_TIER_5 = 2

# "Touched no data" is spelled as "called nothing outside the orient group",
# derived from the registry so it cannot go stale as tools are added or moved.
NON_ORIENT: FrozenSet[str] = frozenset(s.name for s in TOOL_SPECS if s.group != "orient")
GROUPS: Tuple[str, ...] = tuple(sorted({s.group for s in TOOL_SPECS}))


# ---------------------------------------------------------------------------
# independent expectations -- plain pandas, sharing no code with the tools
# ---------------------------------------------------------------------------
def raw_frame() -> pd.DataFrame:
    """
    The CSV, and nothing else.

    Deliberately NOT `metrics.normalise_columns` -> `detect_schema` ->
    `metrics.prepare`, and deliberately not through `ContractAPI` or
    `CompiledKpi.compute` -- that chain is the whole stack under test. Every
    expectation below is plain pandas over these raw columns, so a Tier-1/2
    check compares the agent's retrieval against arithmetic that shares no code
    with it. `_year`/`_quarter`/`_period` do not exist on this frame; that
    absence is asserted by the test suite, and it is what "independently
    computed" is enforced by.
    """
    return pd.read_csv(SAMPLE_CSV, parse_dates=["date"])


def _q2_2026_revenue(raw: pd.DataFrame) -> float:
    # Literal ISO bounds, not `Timeframe(2026, 2)` / `slice_period` -- using
    # `engines.observe` here would reintroduce exactly the shared code the
    # independence of this expectation depends on being absent.
    mask = (raw.date >= "2026-04-01") & (raw.date <= "2026-06-30")
    return float(raw.loc[mask, "revenue"].sum())


def _odd_year_revenue(raw: pd.DataFrame) -> float:
    """The flagship. Odd years *in extent* are 2023 and 2025; the model does
    the "odd" reasoning, the tool does the arithmetic."""
    return float(raw.loc[raw.date.dt.year.isin([2023, 2025]), "revenue"].sum())


def _all_revenue(raw: pd.DataFrame) -> float:
    return float(raw.revenue.sum())


def _orders_per_year(raw: pd.DataFrame) -> List[float]:
    return [float(v) for v in raw.groupby(raw.date.dt.year)["orders"].sum()]


def _fulfilment_rate_2026(raw: pd.DataFrame) -> float:
    """
    Sum-numerator / sum-denominator, divided once -- never the mean of member
    ratios. This independently re-tests the invariant the contract resolver
    exists to guarantee, which is the live bug (task C2) this whole board
    opened with.

    2026 on purpose: fulfilment is a flat 100% in 2023-2025, where sum/sum and
    mean-of-ratios agree trivially and the check would discriminate nothing.
    The planted supply disruption separates them -- 96.79 against 95.74.
    """
    year = raw[raw.date.dt.year == 2026]
    return 100.0 * float(year.fulfilled_orders.sum()) / float(year.orders.sum())


# ---------------------------------------------------------------------------
# the cases
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Case:
    id: str
    tier: int
    question: str
    # Tiers 1-2 only. Returns one number or a list of numbers, all of which
    # must appear among the tool results.
    expects_value: Optional[Callable[[pd.DataFrame], Any]] = None
    required_all: FrozenSet[str] = frozenset()
    # One satisfied member per set -- how "an explanatory step happened" is
    # expressed without naming a sequence the model has no obligation to follow.
    required_any: Tuple[FrozenSet[str], ...] = ()
    forbidden: FrozenSet[str] = frozenset()
    expects_kpis: FrozenSet[str] = frozenset()
    expects_periods: FrozenSet[str] = frozenset()
    expects_error_code: Optional[str] = None
    # Reported in the scorecard, never gated on.
    soft: FrozenSet[str] = frozenset()


CASES: Tuple[Case, ...] = (
    # -- Tier 1: one value, one scope ---------------------------------------
    Case("t1_scalar_in_extent", 1,
         "What was total revenue in Q2 2026?",
         expects_value=_q2_2026_revenue,
         required_all=frozenset({"query_kpi"}),
         expects_kpis=frozenset({"revenue"})),
    Case("t1_out_of_extent", 1,
         # The master plan's own example, kept verbatim. The data ends
         # 2026-06-29, so Q4 2026 does not exist -- which makes this the
         # honesty case: the right answer is "the data does not cover that",
         # and inventing a figure is the exact failure this rewrite targets.
         "What was total revenue in Q4 2026?",
         required_all=frozenset({"query_kpi"}),
         expects_error_code="empty_period"),
    # No `expects_kpis` here, deliberately: nothing was successfully queried, so
    # `kpis_used` is correctly empty. An answer resting on no data is exactly
    # what this case wants.
    Case("t1_definition_only", 1,
         "What does 'fulfilment rate' mean in our business?",
         required_all=frozenset({"get_kpi_definition"}),
         forbidden=NON_ORIENT,
         expects_kpis=frozenset({"fulfillment_rate"})),
    Case("t1_segment_lookup", 1,
         "What was revenue in the North region last quarter?",
         required_all=frozenset({"query_kpi"}),
         expects_kpis=frozenset({"revenue"})),

    # -- Tier 2: set-valued scope, still purely descriptive ------------------
    Case("t2_range_clipped", 2,
         "What was total revenue from 2022 to 2026?",
         expects_value=_all_revenue,
         required_all=frozenset({"query_kpi"}),
         expects_kpis=frozenset({"revenue"})),
    Case("t2_odd_years", 2,
         "What was revenue in all odd-numbered years?",
         expects_value=_odd_year_revenue,
         required_all=frozenset({"query_kpi"}),
         # `loop._format_period` renders {"type":"years","values":[2023,2025]}
         # exactly so -- this label IS the reasoning fingerprint. A model that
         # scanned every year and summed afterwards would not produce it.
         expects_periods=frozenset({"2023,2025"}),
         expects_kpis=frozenset({"revenue"})),
    Case("t2_per_year_orders", 2,
         "How many orders did we get in each year?",
         expects_value=_orders_per_year,
         required_all=frozenset({"query_kpi"}),
         expects_kpis=frozenset({"orders"})),
    Case("t2_ratio_kpi", 2,
         # The only case whose expected number distinguishes a correctly
         # aggregated ratio from a mean of member ratios.
         "What was our fulfillment rate in 2026?",
         expects_value=_fulfilment_rate_2026,
         required_all=frozenset({"query_kpi"}),
         expects_kpis=frozenset({"fulfillment_rate"})),

    # -- Tier 3: two scopes plus a judgment; KPI named -----------------------
    Case("t3_margin_dip", 3,
         "Is this quarter's gross margin dip normal, or something to worry about?",
         required_all=frozenset({"compare_periods"}),
         required_any=(frozenset({"assess_significance", "get_normal_range",
                                  "compare_to_seasonal_norm", "detect_seasonality"}),),
         expects_kpis=frozenset({"gross_margin_pct"}),
         soft=frozenset({"stayed_out_of_contest"})),
    Case("t3_seasonality", 3,
         "Is the Q1 drop in orders just seasonality?",
         required_any=(frozenset({"detect_seasonality", "compare_to_seasonal_norm"}),
                       frozenset({"get_timeseries", "compare_periods", "detect_trend"})),
         expects_kpis=frozenset({"orders"})),
    Case("t3_normal_range", 3,
         "Is a 3% move in fulfillment rate unusual for us?",
         required_any=(frozenset({"get_normal_range", "assess_significance"}),),
         expects_kpis=frozenset({"fulfillment_rate"})),

    # -- Tier 4: cause is plural and unknown; KPI still named ----------------
    Case("t4_profit_factors", 4,
         "What factors are affecting my gross profit?",
         required_any=(
             # candidate generation...
             frozenset({"find_related_kpis", "get_formula_components", "get_kpi_inputs",
                        "get_formula_structure", "correlate_kpi_matrix"}),
             # ...then attribution, in either order
             frozenset({"rank_drivers", "decompose_by_dimension", "attribute_dimensions",
                        "decompose_formula"}),
         ),
         expects_kpis=frozenset({"gross_profit"}),
         soft=frozenset({"crossed_three_groups"})),
    Case("t4_revenue_decline", 4,
         # The one eval question whose ANSWER has independent ground truth:
         # `ground_truth.json` records three planted factors with exact Shapley
         # impacts for exactly this comparison.
         "Why did revenue fall in Q2 2026 compared with Q1 2026?",
         required_all=frozenset({"compare_periods"}),
         required_any=(frozenset({"rank_drivers", "decompose_by_dimension",
                                  "attribute_dimensions"}),),
         expects_periods=frozenset({"2026-Q2", "2026-Q1"}),
         expects_kpis=frozenset({"revenue"}),
         soft=frozenset({"planted_locus_recall"})),
    Case("t4_rate_or_mix", 4,
         "Is the margin decline coming from pricing, from cost, or from mix?",
         required_any=(frozenset({"decompose_rate_mix", "decompose_formula",
                                  "decompose_by_dimension"}),),
         expects_kpis=frozenset({"gross_margin_pct"})),

    # -- Tier 5: nothing is named ------------------------------------------
    Case("t5_worry", 5,
         "What should I be worried about right now?",
         required_all=frozenset({"scan_kpis"}),
         required_any=(frozenset({"rank_kpis_by_movement", "filter_material_changes"}),),
         soft=frozenset({"touched_two_groups"})),
    Case("t5_anything_unusual", 5,
         "Is anything unusual in the data?",
         required_any=(frozenset({"scan_anomalies", "scan_kpis"}),)),
    Case("t5_biggest_change", 5,
         "What changed the most last quarter?",
         required_any=(frozenset({"scan_kpis", "rank_kpis_by_movement"}),)),
)


# ---------------------------------------------------------------------------
# what a scorer is allowed to see
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Observed:
    """
    Everything a scorer is allowed to see about one run.

    `AgentAnswer.answer` is deliberately absent, and there is no field of any
    kind that could carry it -- only `answer_chars`. A check that asserted on
    wording would have nothing to assert against. **That is the enforcement**,
    standing in for a comment asking nobody to write one, and it is what the
    task board means by "flaky-wording assertions are structurally impossible".
    """

    status: str
    turns: int
    tool_calls: int
    tools: Tuple[str, ...]            # call order, errored calls included
    tools_ok: Tuple[str, ...]         # only calls that actually ran
    groups: FrozenSet[str]
    kpis_used: Tuple[str, ...]        # as the endpoint would serve it
    kpis_used_ok: Tuple[str, ...]     # re-derived here, from non-errored steps only
    periods_used: Tuple[Any, ...]
    error_codes: Tuple[str, ...]
    # (json_path, value) rather than a bare float, so a coincidental match
    # against a row count is visible in the printed detail, not a silent pass.
    values: Tuple[Tuple[str, float], ...]
    drivers: Tuple[Dict[str, Any], ...]   # for the t4 planted-locus soft check
    answer_chars: int

    @classmethod
    def of(cls, ans) -> "Observed":
        from app.agent.registry import _SPECS_BY_NAME

        trace = list(ans.evidence)
        ok = [s for s in trace if not s.get("is_error")]
        tools = tuple(s.get("tool") for s in trace)
        codes = tuple(s["result"]["error"] for s in trace
                      if s.get("is_error") and isinstance(s.get("result"), dict)
                      and "error" in s["result"])
        values: List[Tuple[str, float]] = []
        drivers: List[Dict[str, Any]] = []
        for step in ok:
            _walk_numbers(step.get("result"), f"{step.get('tool')}", values)
            _collect_drivers(step.get("result"), drivers)
        return cls(
            status=ans.status,
            turns=int(ans.engine.get("turns") or 0),
            tool_calls=len(trace),
            tools=tools,
            tools_ok=tuple(s.get("tool") for s in ok),
            # Read off the registry, so the group taxonomy can never drift.
            groups=frozenset(_SPECS_BY_NAME[t].group for t in tools if t in _SPECS_BY_NAME),
            kpis_used=tuple(ans.kpis_used),
            kpis_used_ok=tuple(_kpis_from(ok)),
            periods_used=tuple(ans.periods_used),
            error_codes=codes,
            values=tuple(values),
            drivers=tuple(drivers),
            answer_chars=len(ans.answer or ""),
        )


def _kpis_from(steps: List[Dict[str, Any]]) -> List[str]:
    """`loop._derive_kpis_used`, re-derived here rather than imported.

    The harness must not inherit a bug from the thing it measures: if the loop's
    own derivation ever regressed to counting rejected calls again, importing it
    would hide that, and re-deriving surfaces it as a disagreement."""
    from app.agent.loop import _KPI_ARG_NAMES

    found: List[str] = []
    for step in steps:
        for name in _KPI_ARG_NAMES:
            value = step.get("args", {}).get(name)
            for item in ([value] if isinstance(value, str) else value or []):
                if isinstance(item, str) and item not in found:
                    found.append(item)
    return found


def _walk_numbers(obj: Any, path: str, out: List[Tuple[str, float]]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.append((path, float(obj)))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _walk_numbers(v, f"{path}.{k}", out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _walk_numbers(v, f"{path}[{i}]", out)


def _collect_drivers(obj: Any, out: List[Dict[str, Any]]) -> None:
    """Anything shaped like a ranked driver -- a dimension plus a member name."""
    if isinstance(obj, dict):
        if "dimension" in obj and ("member" in obj or "name" in obj):
            out.append(obj)
        for v in obj.values():
            _collect_drivers(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_drivers(v, out)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def _close(a: float, b: float) -> bool:
    if b == 0:
        return abs(a) < 1e-9
    return abs(a - b) / abs(b) * 100.0 <= VALUE_TOLERANCE_PCT


def score(case: Case, obs: Observed, raw: pd.DataFrame,
          available_kpis: FrozenSet[str], truth: Optional[Dict[str, Any]] = None
          ) -> List[Dict[str, Any]]:
    """Every check for one case, in the `validate_rca.py` scorecard shape."""
    checks: List[Dict[str, Any]] = []

    def check(name, passed, detail, value=None, bar=None, soft=False):
        checks.append({"check": name, "pass": bool(passed), "value": value,
                       "bar": bar, "detail": detail, "soft": soft})

    # -- universal ----------------------------------------------------------
    check("terminated", obs.status in {"ok", "max_turns_exhausted"},
          f"status={obs.status}", obs.status, "ok|max_turns_exhausted")

    if obs.status == "ok":
        check("wrote_an_answer", obs.answer_chars > 0,
              f"{obs.answer_chars} characters of prose", obs.answer_chars, "> 0")

    invented = sorted(set(obs.kpis_used_ok) - set(available_kpis))
    check("no_invented_kpi", not invented,
          "every KPI queried exists in this dataset" if not invented
          else f"not in this dataset: {invented}", invented, [])

    check("loop_derivation_agrees", set(obs.kpis_used) == set(obs.kpis_used_ok),
          "`kpis_used` counts only calls that ran"
          if set(obs.kpis_used) == set(obs.kpis_used_ok)
          else f"loop reported {sorted(set(obs.kpis_used) - set(obs.kpis_used_ok))} "
               "from a rejected call",
          sorted(obs.kpis_used), sorted(obs.kpis_used_ok))

    bar = MAX_TURNS_BY_TIER[case.tier]
    check("turn_budget", obs.turns <= bar, f"{obs.turns} turns", obs.turns, bar)
    call_bar = MAX_TOOL_CALLS_BY_TIER[case.tier]
    check("tool_call_budget", obs.tool_calls <= call_bar,
          f"{obs.tool_calls} tool calls", obs.tool_calls, call_bar)

    # -- tools --------------------------------------------------------------
    # Normally a required tool must have *run*: a call the airlock rejected did
    # not answer anything. A case that expects an error is the exception -- for
    # `t1_out_of_extent` the correct behaviour is precisely that `query_kpi` was
    # called and refused, so there "was called" is the right measure.
    called = set(obs.tools) if case.expects_error_code else set(obs.tools_ok)
    if case.required_all:
        missing = sorted(case.required_all - called)
        check("required_tools", not missing,
              "called " + ", ".join(sorted(case.required_all)) if not missing
              else f"never called: {missing}", sorted(called), sorted(case.required_all))

    for i, alternatives in enumerate(case.required_any):
        hit = sorted(alternatives & called)
        check(f"required_any_{i + 1}", bool(hit),
              f"satisfied by {hit}" if hit
              else f"none of {sorted(alternatives)} were called",
              hit, sorted(alternatives))

    if case.forbidden:
        touched = sorted(case.forbidden & set(obs.tools))
        check("forbidden_tools", not touched,
              "touched no data, as it should not have" if not touched
              else f"should not have called: {touched}", touched, [])

    # -- what it says it used ----------------------------------------------
    if case.expects_kpis:
        missing = sorted(case.expects_kpis - set(obs.kpis_used_ok))
        check("expected_kpis", not missing,
              "queried " + ", ".join(sorted(case.expects_kpis)) if not missing
              else f"never queried: {missing}", sorted(obs.kpis_used_ok),
              sorted(case.expects_kpis))

    if case.expects_periods:
        labels = {str(p) for p in obs.periods_used}
        missing = sorted(case.expects_periods - labels)
        check("expected_periods", not missing,
              f"periods {sorted(labels)}" if not missing
              else f"never scoped to: {missing}", sorted(labels),
              sorted(case.expects_periods))

    # -- the number ---------------------------------------------------------
    if case.expects_value is not None:
        expected = case.expects_value(raw)
        wanted = expected if isinstance(expected, list) else [expected]
        found, matches = [], []
        for target in wanted:
            hit = next(((p, v) for p, v in obs.values if _close(v, target)), None)
            found.append(hit is not None)
            if hit:
                matches.append(f"{target:,.2f} at {hit[0]}")
        check("retrieved_the_right_number", all(found),
              "; ".join(matches) if all(found)
              else f"expected {[round(float(w), 2) for w in wanted]}, "
                   f"not found among {len(obs.values)} returned numbers",
              matches, [round(float(w), 2) for w in wanted])

    if case.expects_error_code:
        check("reported_the_missing_period",
              case.expects_error_code in obs.error_codes,
              f"errors={list(obs.error_codes)}", list(obs.error_codes),
              case.expects_error_code)

    # -- soft: reported, never gated ---------------------------------------
    if "stayed_out_of_contest" in case.soft:
        check("stayed_out_of_contest", "contest" not in obs.groups,
              f"groups touched: {sorted(obs.groups)}", sorted(obs.groups), "no contest",
              soft=True)
    if "crossed_three_groups" in case.soft:
        check("crossed_three_groups", len(obs.groups) >= 3,
              f"{len(obs.groups)} groups: {sorted(obs.groups)}", len(obs.groups), 3, soft=True)
    if "touched_two_groups" in case.soft:
        check("touched_two_groups", len(obs.groups) >= MIN_GROUPS_TIER_5,
              f"{len(obs.groups)} groups: {sorted(obs.groups)}", len(obs.groups),
              MIN_GROUPS_TIER_5, soft=True)
    if "planted_locus_recall" in case.soft and truth:
        from scripts.validate_rca import matches_locus

        factors = truth["planted_factors"]
        hits = [f["id"] for f in factors
                if any(matches_locus(d, f["locus"]) for d in obs.drivers)]
        check("planted_locus_recall", len(hits) == len(factors),
              f"{len(hits)}/{len(factors)} planted loci surfaced: {hits}",
              len(hits), len(factors), soft=True)

    return checks


def passed(checks: List[Dict[str, Any]]) -> bool:
    """Soft checks are reported, never gated -- see the module docstring."""
    return all(c["pass"] for c in checks if not c["soft"])


# ---------------------------------------------------------------------------
# the tool-usage report
# ---------------------------------------------------------------------------
def coverage(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Which of the 56 tools fired, grouped.

    `never_fired` is **evidence, not a bar**. Seventeen questions cannot exercise
    56 tools, and a coverage percentage as a pass/fail check would be gamed
    within a week. It feeds exactly two decisions: whether task X5's robustness
    family is worth building, and whether decision 60's deferred progressive
    disclosure is needed. `by_group` is the more useful cut -- if all eleven
    `contest` tools fire zero times across three Tier-5 questions, that is a
    finding about how far the loop gets, not about eleven dead tools.
    """
    calls: Dict[str, int] = {}
    errors: Dict[str, int] = {}
    for run in runs:
        for name in run["tools"]:
            calls[name] = calls.get(name, 0) + 1
        for name in set(run["tools"]) - set(run["tools_ok"]):
            errors[name] = errors.get(name, 0) + 1

    by_group: Dict[str, Any] = {}
    for group in GROUPS:
        names = sorted(s.name for s in TOOL_SPECS if s.group == group)
        fired = [n for n in names if calls.get(n)]
        by_group[group] = {"total": len(names), "fired": len(fired),
                           "fired_names": fired,
                           "never_names": [n for n in names if n not in fired]}
    never = sorted(s.name for s in TOOL_SPECS if not calls.get(s.name))
    return {"tools_total": len(TOOL_SPECS), "tools_fired": len(TOOL_SPECS) - len(never),
            "tools_never_fired": len(never), "by_group": by_group,
            "tool_calls": dict(sorted(calls.items())),
            "tool_errors": dict(sorted(errors.items())), "never_fired": never}


# ---------------------------------------------------------------------------
# the live run
# ---------------------------------------------------------------------------
def bootstrap() -> Dict[str, Any]:
    """
    An isolated data dir with the retail sample uploaded.

    Deliberately NOT `tests.base.setup_environment`: that sets
    `ANTHROPIC_API_KEY=""`, which would silently disable the very thing this
    script exists to validate. It also ingests the RAG corpus, which the agent
    loop does not use.
    """
    os.environ.setdefault("DB_BACKEND", "json")
    os.environ.setdefault("AUTH_MODE", "demo")
    os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="agent_eval_")

    from app.config import get_settings

    get_settings.cache_clear()
    from app.db.repositories import UserRepository
    from app.services import dataset_service

    uid = "agent_eval"
    UserRepository().upsert_profile(uid, "eval@example.com", "Eval", "data_analyst")
    dataset = dataset_service.store_upload(uid, SAMPLE_CSV.name, SAMPLE_CSV.read_bytes())
    return {"uid": uid, "dataset": dataset}


def main() -> int:
    from app.config import get_settings

    if not (os.environ.get("ANTHROPIC_API_KEY") or "").strip():
        print("\n  validate_agent needs a live model: set ANTHROPIC_API_KEY and re-run.")
        print("  (No report written -- a scorecard that looks live but is not is worse "
              "than none.)\n")
        return 2

    state = bootstrap()
    from app.agent.context import AgentContext
    from app.agent.loop import answer
    from app.llm.client import get_llm
    from app.services.telemetry import request_telemetry

    truth = json.loads(TRUTH_PATH.read_text()) if TRUTH_PATH.exists() else None
    raw = raw_frame()
    llm = get_llm()

    runs: List[Dict[str, Any]] = []
    for case in CASES:
        # A fresh context per question so `Budget` is per question, not shared.
        ctx = AgentContext.build(state["uid"], state["dataset"],
                                 max_turns=MAX_TURNS_BY_TIER[case.tier])
        available = frozenset(ctx.api.available_keys(ctx.df))
        with request_telemetry(state["uid"], case.id) as telemetry:
            ans = answer(case.question, ctx, llm=llm,
                         max_turns=MAX_TURNS_BY_TIER[case.tier])
        obs = Observed.of(ans)
        checks = score(case, obs, raw, available, truth)
        runs.append({
            "id": case.id, "tier": case.tier, "question": case.question,
            "status": obs.status, "turns": obs.turns, "tool_calls": obs.tool_calls,
            "tools": list(obs.tools), "tools_ok": list(obs.tools_ok),
            "groups": sorted(obs.groups), "kpis_used": list(obs.kpis_used_ok),
            "periods_used": [str(p) for p in obs.periods_used],
            "error_codes": list(obs.error_codes), "answer_chars": obs.answer_chars,
            "tokens": (telemetry.saved or {}).get("total_tokens"),
            "cost": (telemetry.saved or {}).get("estimated_cost"),
            "checks": checks, "passed": passed(checks),
            "answer": ans.answer,        # printed for a human; not scored
        })
        mark = "PASS" if passed(checks) else "FAIL"
        print(f"  [{mark}] T{case.tier} {case.id:<24} "
              f"{obs.turns:>2} turns, {obs.tool_calls:>2} calls")

    return report(runs, get_settings().anthropic_model)


def report(runs: List[Dict[str, Any]], model: str) -> int:
    cover = coverage(runs)
    ok = sum(1 for r in runs if r["passed"])
    budget: Dict[str, Any] = {}
    for tier in sorted({r["tier"] for r in runs}):
        rows = [r for r in runs if r["tier"] == tier]
        budget[str(tier)] = {
            "cases": len(rows),
            "max_turns_seen": max(r["turns"] for r in rows),
            "bar": MAX_TURNS_BY_TIER[tier],
            "tokens_total": sum(r["tokens"] or 0 for r in rows),
            "cost": round(sum(r["cost"] or 0 for r in rows), 6),
        }

    doc = {
        "generated_from": "scripts/validate_agent.py",
        "dataset": SAMPLE_CSV.name,
        "model": model,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "summary": {"cases": len(runs), "passed": ok,
                    "all_passed": ok == len(runs),
                    "tools_total": cover["tools_total"],
                    "tools_fired": cover["tools_fired"],
                    "tools_never_fired": cover["tools_never_fired"]},
        "coverage_by_group": cover["by_group"],
        "tool_calls": cover["tool_calls"],
        "tool_errors": cover["tool_errors"],
        "never_fired": cover["never_fired"],
        "budget_by_tier": budget,
        "cases": runs,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(doc, indent=2, default=str) + "\n")

    print(f"\n  {ok}/{len(runs)} cases passed\n")
    print("  tool coverage")
    for group in GROUPS:
        g = cover["by_group"][group]
        print(f"    {group:<12} {g['fired']:>2}/{g['total']:<2} fired")
    if cover["never_fired"]:
        print(f"\n  never fired ({len(cover['never_fired'])}) -- evidence for the X5 "
              f"and progressive-disclosure decisions, not a failure:")
        for name in cover["never_fired"]:
            print(f"    {name}")
    print(f"\n  report -> {REPORT_PATH}\n")

    for run in runs:
        if not run["passed"]:
            print(f"  {run['id']} failed:")
            for c in run["checks"]:
                if not c["pass"] and not c["soft"]:
                    print(f"    {c['check']}: {c['detail']}")
    return 0 if ok == len(runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
