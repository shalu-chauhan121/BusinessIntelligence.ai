"""
The agent loop -- decides what to ask, with which tools, and when to stop.

`answer(question, ctx)` is the one entry point. It composes the two halves
built in earlier batches: `LLMClient._call_with_tools` (the multi-turn tool-
use transport) and `ctx.registry` (the 56 schema-described, name-dispatched
tools). Nothing here re-implements turn iteration or `stop_reason` handling
-- that stays exactly where A1 put it.

All 56 tools are exposed every turn, not a progressively-disclosed subset.
The tool array already carries the loop's one cache breakpoint
(`registry.py`), so it is the largest stable prefix of every request in the
run; swapping it mid-loop would forfeit that for the rest of the run.
`ctx.registry.by_group(...)` stays built and unused -- A8's eval harness is
what earns or retires progressive disclosure, not a guess made here.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..llm.client import LLMClient, ToolLoopResult
from ..llm.prompts import agent_system
from .context import AgentContext

# Tool-call argument names that name a KPI, gathered across the registry's
# 56 tools -- `key` covers the `introspect.py` orientation tools
# (`get_kpi_definition`, `get_kpi_inputs`, `get_formula_structure`), which
# use that name rather than `kpi_key`.
_KPI_ARG_NAMES = ("kpi_key", "kpi_a", "kpi_b", "cause_kpi", "key",
                  "kpi_keys", "candidates", "candidate_causes")
_PERIOD_ARG_NAMES = ("time_filter", "period_a", "period_b", "baseline")


@dataclass(frozen=True)
class AgentAnswer:
    """
    The response shape Part 3 of the master plan specifies: a final answer
    plus its evidence trail. A7 serialises this rather than reshaping it.

    `status` widens `ToolLoopResult.status` ("ok" | "max_turns_exhausted" |
    "truncated" | "refused") with one loop-level addition: `"llm_required"`,
    returned when there is no usable LLM at all -- the existing deterministic
    pipeline keeps its own no-key fallback untouched; this loop has none, by
    design (the master plan: "no deterministic fallback is required").
    """
    status: str
    question: str
    answer: str
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    kpis_used: List[str] = field(default_factory=list)
    periods_used: List[Any] = field(default_factory=list)
    engine: Dict[str, Any] = field(default_factory=dict)


def _format_period(spec: Any) -> Any:
    """A compact label for one `TimeFilter` spec, for `periods_used`. Falls
    back to the raw spec for a shape this does not specifically know --
    `periods_used` is a UI hint, not a validated result, so an unrecognised
    shape degrading to its raw form is fine."""
    if not isinstance(spec, dict):
        return spec
    kind = spec.get("type")
    if kind == "quarter":
        return f"{spec.get('year')}-Q{spec.get('quarter')}"
    if kind == "year":
        return str(spec.get("year"))
    if kind == "years":
        return ",".join(str(y) for y in spec.get("values", []))
    if kind == "quarters":
        return ",".join(f"{q.get('year')}-Q{q.get('quarter')}" for q in spec.get("values", []))
    if kind == "range":
        return f"{spec.get('start')}..{spec.get('end')}"
    if kind == "months":
        return f"{spec.get('start')}..{spec.get('end')}"
    if kind == "latest":
        return f"latest {spec.get('n')} {spec.get('grain')}"
    if kind == "all":
        return "all"
    return spec


def _succeeded(trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Only the tool calls that actually ran.

    An errored step's `args` are what the model *asked for*, not what the
    dataset answered: a KPI key the airlock rejected, a dimension that does
    not exist, a period the data does not hold. Deriving `kpis_used` from
    those would report a hallucinated key as one this answer rests on --
    which is exactly the claim the derivation exists to make impossible.
    """
    return [step for step in trace if not step.get("is_error")]


def _derive_kpis_used(trace: List[Dict[str, Any]]) -> List[str]:
    """
    KPI keys actually queried, read back from the trace's own recorded
    arguments -- never asserted by the model. A tool call's `args` is exactly
    what was sent to `registry.dispatch`, so this is a fact about what ran,
    not a claim the answer's prose makes about itself.
    """
    found: List[str] = []
    for step in _succeeded(trace):
        for name in _KPI_ARG_NAMES:
            value = step.get("args", {}).get(name)
            if isinstance(value, str) and value not in found:
                found.append(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item not in found:
                        found.append(item)
    return found


def _derive_periods_used(trace: List[Dict[str, Any]]) -> List[Any]:
    found: List[Any] = []
    for step in _succeeded(trace):
        for name in _PERIOD_ARG_NAMES:
            value = step.get("args", {}).get(name)
            if value is None:
                continue
            label = _format_period(value)
            if label not in found:
                found.append(label)
    return found


def _seed(ctx: AgentContext) -> Dict[str, Any]:
    """Orientation seeded into the system prompt -- dispatched through the
    same registry the loop itself calls, so it is byte-identical to what the
    model would receive had it asked (A6)."""
    described, _ = ctx.registry.dispatch("describe_dataset", {})
    listed, _ = ctx.registry.dispatch("list_kpis", {})
    return {**described, **listed}


def answer(question: str, ctx: AgentContext, *, llm: Optional[LLMClient] = None,
          max_turns: Optional[int] = None) -> AgentAnswer:
    """
    Answer one business question against `ctx`'s dataset.

    Returns a typed `AgentAnswer` in every case -- non-convergence
    (`max_turns_exhausted`, `truncated`), a safety `refused`, and no usable
    LLM at all (`llm_required`) are all normal endings a caller renders, not
    exceptions it must catch.
    """
    if llm is None:
        from ..llm.client import get_llm
        llm = get_llm()

    if not llm.enabled:
        # No dataset work at all -- not even the seeding tool calls -- when
        # there is nothing that could use their result.
        return AgentAnswer(status="llm_required", question=question, answer="",
                          engine={"turns": 0, "model": None, "seconds": 0.0})

    system = agent_system(_seed(ctx))
    messages = [{"role": "user", "content": question}]
    turn_budget = max_turns if max_turns is not None else ctx.budget.remaining_turns

    started = time.perf_counter()
    result: ToolLoopResult = llm._call_with_tools(
        system, messages, ctx.registry.tools, ctx.registry.dispatch, max_turns=turn_budget)
    elapsed = time.perf_counter() - started

    ctx.budget.record(result.turns)

    return AgentAnswer(
        status=result.status,
        question=question,
        answer=result.text,
        evidence=result.trace,
        kpis_used=_derive_kpis_used(result.trace),
        periods_used=_derive_periods_used(result.trace),
        engine={"turns": result.turns, "model": llm.model, "seconds": round(elapsed, 3)},
    )
