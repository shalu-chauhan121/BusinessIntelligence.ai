"""
LLM reasoning layer (Anthropic Claude), with a deterministic fallback.

The product is designed so that the four-stage investigation completes with or
without an API key: the analysis, retrieval, contest checks, confidence scoring
and recommendations are all deterministic. The model adds framing and narrative
on top of facts it is given. That is why `enabled=False` degrades the prose, not
the analysis.
"""
from __future__ import annotations

import json
import logging
import re
import time
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..config import get_settings
from .prompts import (ACT_SYSTEM, CONTEST_SYSTEM, HYPOTHESIS_SYSTEM,
                      INVESTIGATE_SYSTEM, KPI_DISCOVERY_SYSTEM, QUERY_SYSTEM,
                      persona_system)
from ..services.telemetry import track_llm_cache, track_llm_call

log = logging.getLogger(__name__)
JSON_BLOCK = re.compile(r"\{.*\}", re.S)
CACHE_VERSION = "llm-result-v1"
# The tool-loop transport (`_call_with_tools`) hashes the *whole* message
# history rather than a single (system, user) pair, so it cannot share a cache
# namespace with the single-turn `_call` roles above -- a v1 key collision
# would return turn 3 of one question as the answer to turn 3 of another.
TOOL_CACHE_VERSION = "llm-tool-result-v1"
# Executing a tool never raises past this boundary -- a bad name or bad
# arguments becomes a recoverable tool_result, per `agent/errors.py`.
ToolExecutor = Callable[[str, Dict[str, Any]], Tuple[Any, bool]]


class LLMTransportError(RuntimeError):
    """
    A provider-side failure of the tool loop's transport -- a timeout, a rate
    limit, an auth rejection, a dropped connection.

    Exists so callers outside `app.llm` can catch the failure without importing
    `anthropic`. This module is already the only place the SDK is imported (in
    `LLMClient.__init__`); this keeps it the only place the SDK's exception
    *types* are named too.

    Deliberately a `RuntimeError` and not a `ValueError`, so it cannot be
    swallowed by `routes_analysis._guard`'s 422 branch: a provider outage is
    not a malformed request, and the two must not map to the same status.

    Raised only from `_call_with_tools`. `_call`'s four single-turn roles keep
    their existing behaviour, since the deterministic pipeline already degrades
    around them and nothing there needs a new failure type.
    """


def _extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = JSON_BLOCK.search(text)
        if not m:
            raise ValueError("model did not return JSON")
        return json.loads(m.group(0))


def _round(x, nd=2):
    return round(x, nd) if isinstance(x, (int, float)) else x


def _blocks_to_dicts(content) -> List[Dict[str, Any]]:
    """
    Normalise a response's content blocks (SDK pydantic objects, SimpleNamespace
    test fakes, or already-plain dicts from a cache replay) to plain dicts.

    This is what lets the tool loop treat a live API response and a cached
    turn identically -- both become the same shape before anything downstream
    (history, trace, text extraction, caching) touches them. Server-side tool
    block kinds (this batch declares none) fall through to their raw `text`
    if any, since only `text`/`tool_use`/`thinking`/`redacted_thinking` are
    given full field-by-field handling.
    """
    blocks: List[Dict[str, Any]] = []
    for b in content or []:
        if isinstance(b, dict):
            blocks.append(dict(b))
            continue
        kind = getattr(b, "type", None)
        block: Dict[str, Any] = {"type": kind}
        if kind == "text":
            block["text"] = getattr(b, "text", "")
        elif kind == "tool_use":
            block["id"] = getattr(b, "id", None)
            block["name"] = getattr(b, "name", None)
            block["input"] = getattr(b, "input", None) or {}
        elif kind == "thinking":
            block["thinking"] = getattr(b, "thinking", "")
            if getattr(b, "signature", None) is not None:
                block["signature"] = b.signature
        elif kind == "redacted_thinking":
            block["data"] = getattr(b, "data", None)
        else:
            text = getattr(b, "text", None)
            if text is not None:
                block["text"] = text
        blocks.append(block)
    return blocks


def _extract_text(content: List[Dict[str, Any]]) -> str:
    return "\n".join(b.get("text", "") for b in content if b.get("type") == "text").strip()


@dataclass(frozen=True)
class ToolLoopResult:
    """
    The outcome of a multi-turn tool-use exchange.

    Non-convergence is a typed result, never an exception: `max_turns`
    exhaustion, a truncated final turn and a safety refusal are all normal
    endings a caller must be able to render, not failures that crash the
    request. `trace` is the raw evidence-panel material -- one entry per tool
    call, in call order, regardless of how the loop itself ended.
    """
    status: str                       # "ok" | "max_turns_exhausted" | "truncated" | "refused"
    text: str
    content: List[Any] = field(default_factory=list)      # final assistant turn's content blocks
    trace: List[Dict[str, Any]] = field(default_factory=list)
    turns: int = 0
    stop_reason: Optional[str] = None


class LLMClient:
    def __init__(self):
        s = get_settings()
        self.settings = s
        self.model = s.anthropic_model
        self._client = None
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.last_error: Optional[str] = None
        if s.llm_enabled:
            try:
                import anthropic

                self._client = anthropic.Anthropic(
                    api_key=s.anthropic_api_key, timeout=float(s.llm_timeout_seconds),
                    max_retries=3,
                )
            except Exception as exc:                      # pragma: no cover
                self.last_error = str(exc)
                log.warning("Anthropic client unavailable: %s", exc)

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @property
    def status(self) -> Dict[str, Any]:
        return {
            "provider": "anthropic",
            "model": self.model if self.enabled else None,
            "enabled": self.enabled,
            "mode": "llm_reasoning" if self.enabled else "deterministic_fallback",
            "note": (
                "Claude is framing hypotheses, classifying document stance and writing the leader "
                "narrative over facts computed by the analysis layer."
                if self.enabled else
                "No ANTHROPIC_API_KEY configured. The full four-stage investigation still runs; "
                "hypothesis statements and the narrative come from the deterministic reasoner."
            ),
            "error": self.last_error,
        }

    # -- transport ---------------------------------------------------------
    def _cache_key(self, system: str, user: str) -> str:
        payload = json.dumps({"version": CACHE_VERSION, "provider": "anthropic", "model": self.model,
                              "system": system, "user": user}, sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode()).hexdigest()

    def _call(self, system: str, user: str) -> Dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("LLM disabled")
        cache_key = self._cache_key(system, user)
        cached = self._cache.get(cache_key)
        if cached is not None:
            track_llm_cache(hit=True)
            return deepcopy(cached)
        track_llm_cache(hit=False)
        started = time.perf_counter()
        try:
            resp = self._client.messages.create(
                model=self.model, max_tokens=self.settings.llm_max_tokens, system=system,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as exc:
            track_llm_call(model=self.model, system=system, user=user, error=exc, started=started)
            raise
        track_llm_call(model=self.model, system=system, user=user, response=resp, started=started)
        text = "".join(getattr(b, "text", "") for b in resp.content)
        result = _extract_json(text)
        self._cache[cache_key] = deepcopy(result)
        return result

    # -- tool-use transport --------------------------------------------------
    # A sibling to `_call`, not a replacement: every existing role method above
    # keeps calling `_call` unchanged. This is the multi-turn loop the agent
    # tool registry (agent/registry.py) dispatches through.
    def _tool_cache_key(self, system: str, messages: List[Dict[str, Any]],
                        tools: List[Dict[str, Any]], tool_choice: Optional[Dict[str, Any]],
                        max_tokens: int) -> str:
        payload = json.dumps({
            "version": TOOL_CACHE_VERSION, "provider": "anthropic", "model": self.model,
            "system": system, "messages": messages, "tools": tools,
            "tool_choice": tool_choice, "max_tokens": max_tokens,
        }, sort_keys=True, separators=(",", ":"), default=str)
        return sha256(payload.encode()).hexdigest()

    def _call_with_tools(self, system: str, messages: List[Dict[str, Any]],
                         tools: List[Dict[str, Any]], executor: "ToolExecutor", *,
                         max_turns: Optional[int] = None, max_tokens: Optional[int] = None,
                         tool_choice: Optional[Dict[str, Any]] = None) -> "ToolLoopResult":
        """
        Run the agentic tool-use loop until the model stops calling tools.

        executor(name, args) -> (payload, is_error) is the only thing this
        method knows about tools -- it never inspects a schema or a KPI key.
        agent/registry.dispatch is the production executor; tests pass a
        plain function. A tool that raises inside executor is turned into an
        is_error tool_result here, never let past this boundary -- one bad
        call must not end the request.
        """
        if not self.enabled:
            raise RuntimeError("LLM disabled")
        max_turns = max_turns or self.settings.llm_max_turns
        max_tokens = max_tokens or self.settings.llm_tool_max_tokens

        history: List[Dict[str, Any]] = list(messages)
        trace: List[Dict[str, Any]] = []
        step = 0
        last_content: List[Dict[str, Any]] = []
        last_stop_reason: Optional[str] = None

        for turn in range(1, max_turns + 1):
            cache_key = self._tool_cache_key(system, history, tools, tool_choice, max_tokens)
            cached = self._cache.get(cache_key)
            refusal_category = None
            if cached is not None:
                track_llm_cache(hit=True)
                content, stop_reason = deepcopy(cached)
            else:
                track_llm_cache(hit=False)
                started = time.perf_counter()
                kwargs: Dict[str, Any] = dict(
                    model=self.model, max_tokens=max_tokens, system=system, messages=history,
                    tools=tools, thinking={"type": "adaptive"},
                    output_config={"effort": self.settings.llm_effort},
                )
                if tool_choice is not None:
                    kwargs["tool_choice"] = tool_choice
                log_user = json.dumps(history, default=str)
                try:
                    resp = self._client.messages.create(**kwargs)
                except Exception as exc:
                    track_llm_call(model=self.model, system=system, user=log_user, error=exc,
                                  started=started, step=f"Tool loop turn {turn}")
                    # Re-raised as our own type so the API layer can map it to a
                    # 503 without importing `anthropic` to name the SDK's
                    # exceptions. The original is chained, so the SDK message is
                    # still available to log -- it is just not the caller's to
                    # return to a user.
                    raise LLMTransportError(
                        f"Anthropic request failed on tool loop turn {turn}: {exc}") from exc
                track_llm_call(model=self.model, system=system, user=log_user, response=resp,
                              started=started, step=f"Tool loop turn {turn}")
                content = _blocks_to_dicts(resp.content)
                stop_reason = resp.stop_reason
                if stop_reason == "refusal":
                    # A refusal is a safety-classifier decision, not a stable
                    # answer to memoize -- never cached, always re-attempted.
                    details = getattr(resp, "stop_details", None)
                    refusal_category = getattr(details, "category", None) if details else None
                else:
                    self._cache[cache_key] = deepcopy((content, stop_reason))

            history.append({"role": "assistant", "content": content})
            last_content, last_stop_reason = content, stop_reason

            if stop_reason == "end_turn":
                return ToolLoopResult(status="ok", text=_extract_text(content), content=content,
                                      trace=trace, turns=turn, stop_reason=stop_reason)

            if stop_reason == "max_tokens":
                return ToolLoopResult(status="truncated", text=_extract_text(content), content=content,
                                      trace=trace, turns=turn, stop_reason=stop_reason)

            if stop_reason == "refusal":
                text = f"refused ({refusal_category})" if refusal_category else "refused"
                return ToolLoopResult(status="refused", text=text, content=content,
                                      trace=trace, turns=turn, stop_reason=stop_reason)

            if stop_reason == "pause_turn":
                # No server-side tools are declared in this batch, so a paused
                # turn should never occur in practice; resume defensively by
                # re-sending unchanged, per the documented recovery pattern.
                continue

            if stop_reason == "tool_use":
                tool_calls = [b for b in content if b.get("type") == "tool_use"]
                results = []
                for call in tool_calls:
                    step += 1
                    name = call.get("name")
                    args = call.get("input") or {}
                    tool_use_id = call.get("id")
                    try:
                        payload, is_error = executor(name, args)
                    except Exception as exc:      # a tool must never end the request
                        payload, is_error = {"error": "tool_error", "message": str(exc)}, True
                    trace.append({"step": step, "tool": name, "args": args,
                                 "result": payload, "is_error": is_error})
                    results.append({"type": "tool_result", "tool_use_id": tool_use_id,
                                   "content": json.dumps(payload, default=str), "is_error": is_error})
                history.append({"role": "user", "content": results})
                continue

            # An unrecognised stop_reason from a future SDK: stop rather than
            # loop forever on something this version does not understand.
            return ToolLoopResult(status="ok", text=_extract_text(content), content=content,
                                  trace=trace, turns=turn, stop_reason=stop_reason)

        return ToolLoopResult(status="max_turns_exhausted", text=_extract_text(last_content),
                              content=last_content, trace=trace, turns=max_turns,
                              stop_reason=last_stop_reason)

    # -- fact sheets -------------------------------------------------------
    @staticmethod
    def _observation_facts(observation: Dict[str, Any]) -> Dict[str, Any]:
        sig = observation.get("significance", {})
        return {
            "kpi": observation["kpi_label"],
            "period": observation["timeframe"]["pretty"],
            "baseline_period": observation["baseline_timeframe"]["pretty"],
            "current_value": _round(observation.get("current_value")),
            "baseline_value": _round(observation.get("baseline_value")),
            "change_pct": _round(observation.get("change_pct")),
            "verdict": observation.get("verdict"),
            "history_status": observation.get("history_status"),
            "history_context": observation.get("history_note"),
            "historical_trend_available": observation.get("history_status") == "sufficient_history",
            "robust_z": _round(sig.get("robust_z")),
            "normal_change_median_pct": _round(sig.get("median_historical_change_pct")),
            "method": sig.get("method"),
            "top_drivers": [
                {"dimension": d["dimension"], "name": d["name"],
                 "contribution_pct": _round(d.get("contribution_pct"), 1),
                 "change_pct": _round(d.get("change_pct"), 1)}
                for d in observation.get("top_drivers", [])[:5]
            ],
        }

    @staticmethod
    def _hypothesis_facts(h: Dict[str, Any], include_contest: bool = False) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "key": h["key"],
            "current_title": h["title"],
            "current_statement": h["statement"],
            "structured_evidence": [
                {"stance": e["stance"], "label": e["label"], "scope": e.get("scope"),
                 "detail": e["detail"]}
                for e in h.get("evidence", [])
            ],
            "document_evidence": [
                {"source": d["source"], "section": d.get("section"), "quote": d["quote"]}
                for d in h.get("documentary_evidence", [])
            ],
            "missing_evidence": h.get("missing", []),
        }
        if include_contest and h.get("contest"):
            out["contest"] = {
                "temporal": h["contest"]["temporal"].get("detail"),
                "temporal_status": h["contest"]["temporal"].get("status"),
                "consistency": (h["contest"]["consistency"].get("correlation") or {}).get("interpretation"),
                "counterexamples": h["contest"]["consistency"].get("counterexamples"),
                "contradicting_quotes": [d["quote"] for d in h["contest"].get("contradictory_evidence", [])],
            }
            out["confidence"] = h["scoring"]["confidence"]
            out["confidence_band"] = h["scoring"]["confidence_band"]
            out["causal_claim"] = h["scoring"]["causal_claim"]
        return out

    # -- roles -------------------------------------------------------------
    def understand_question(self, question: str, kpi_catalogue: List[Dict[str, Any]],
                            dimensions: List[str]) -> Dict[str, Any]:
        """
        Read a business question against the KPIs this dataset actually has.

        Called only when deterministic grounding could not settle it. The chosen
        key is validated by the caller against the resolver, so a KPI the model
        invents becomes "could not resolve" rather than a wrong investigation.
        """
        payload = {
            "question": question,
            "available_kpis": kpi_catalogue,
            "available_dimensions": dimensions,
        }
        return self._call(QUERY_SYSTEM, json.dumps(payload, indent=2, default=str))

    def generate_hypotheses(self, signals: Dict[str, Any], kpi_semantics: Dict[str, Any],
                            domain: Dict[str, Any], driver_graph: Dict[str, Any],
                            allowed_metrics: List[str]) -> Dict[str, Any]:
        """
        Propose mechanisms that could explain what the data shows.

        `signals` has already been filtered to material movements, so the model
        cannot be misled into explaining noise. It returns predictions rather
        than evidence; the engine measures them and decides what they support.
        """
        payload = {
            "business_context": domain,
            "kpi_under_investigation": kpi_semantics,
            "what_changed": signals,
            "related_measures": driver_graph,
            "allowed_metrics": allowed_metrics,
        }
        return self._call(HYPOTHESIS_SYSTEM, json.dumps(payload, indent=2, default=str))

    def write_for_persona(self, profile: Any, observation: Dict[str, Any],
                          investigation: Dict[str, Any], contested: Dict[str, Any],
                          cores: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Explain the investigation, and say what THIS reader should do about it.

        The findings arrive fixed. `cores` carries the persona-invariant
        substrate every recommendation must trace back to, so the reframing can
        change the advice without being able to change what the evidence says.
        """
        payload = {
            "observation": self._observation_facts(observation),
            "focus": investigation.get("focus", {}),
            "hypotheses": [self._hypothesis_facts(h, include_contest=True)
                           for h in (contested.get("ranking") or [])[:4]],
            "recommendation_basis": cores,
            "reader": {"persona": profile.key, "label": profile.label,
                       "action_horizon": profile.action_horizon},
        }
        system = persona_system(profile.explanation_brief, profile.recommendation_brief,
                                profile.vocabulary, profile.action_horizon)
        return self._call(system, json.dumps(payload, indent=2, default=str))

    def frame_hypotheses(self, observation: Dict[str, Any], hypotheses: List[Dict[str, Any]],
                         focus: Dict[str, str]) -> Dict[str, Any]:
        payload = {
            "observation": self._observation_facts(observation),
            "focus": focus,
            "hypotheses": [self._hypothesis_facts(h) for h in hypotheses],
        }
        return self._call(INVESTIGATE_SYSTEM, json.dumps(payload, indent=2, default=str))

    def classify_stance(self, hypothesis: Dict[str, Any],
                        chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
        payload = {
            "hypothesis": {"title": hypothesis["title"], "statement": hypothesis["statement"]},
            "passages": [
                {"chunk_id": c["chunk_id"], "source": c["source"], "section": c.get("section"),
                 "text": c["quote"]}
                for c in chunks
            ],
        }
        data = self._call(CONTEST_SYSTEM, json.dumps(payload, indent=2, default=str))
        return {v["chunk_id"]: v for v in data.get("verdicts", []) if v.get("chunk_id")}

    def screen_kpi_candidates(self, dataset_facts: Dict[str, Any],
                              candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Judge which computable metrics are semantically meaningful KPIs.

        The model is handed the dataset's SHAPE — column names, inferred semantic
        types, summary statistics — and never a single row, so it has nothing to
        compute a business figure from even if it tried. Its output is re-validated
        against the field list before anything reaches the contract.
        """
        payload = {"dataset": dataset_facts, "candidates": candidates}
        return self._call(KPI_DISCOVERY_SYSTEM, json.dumps(payload, indent=2, default=str))

    def write_story(self, observation: Dict[str, Any], investigation: Dict[str, Any],
                    contested: Dict[str, Any], recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        payload = {
            "observation": self._observation_facts(observation),
            "focus": investigation.get("focus", {}),
            "ranked_hypotheses": [
                self._hypothesis_facts(h, include_contest=True) for h in contested.get("hypotheses", [])[:4]
            ],
            "ambiguity_note": contested.get("ambiguity_note"),
            "unresolved_questions": contested.get("unresolved_questions", []),
            "prepared_recommendations": [
                {"title": r["title"], "actions": r["actions"], "priority": r["priority"],
                 "based_on": r["based_on"]["hypothesis"], "confidence": r["based_on"]["confidence"]}
                for r in recommendations
            ],
        }
        return self._call(ACT_SYSTEM, json.dumps(payload, indent=2, default=str))


_client: Optional[LLMClient] = None


def get_llm() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def reset_llm() -> None:
    global _client
    _client = None
