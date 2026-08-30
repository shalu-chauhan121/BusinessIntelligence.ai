"""
Test doubles for the Anthropic SDK.

There is no LLM test double anywhere else in this repo -- every existing LLM
path is tested by disabling the LLM entirely (`ANTHROPIC_API_KEY=""` in
`base.setup_environment`), and the one place that does fake a response
(`test_telemetry.py`) does it inline with a single `SimpleNamespace`. The
tool-use transport (`LLMClient._call_with_tools`) needs a scripted
*multi-turn* double instead: it has to exercise `stop_reason` handling,
`tool_use` blocks, and cache-key behaviour across several turns of one
conversation, which a single ad-hoc fake per test would make repetitive.

Assign a `ScriptedAnthropic` to `LLMClient._client` directly -- the same seam
`test_telemetry.py:74` already uses for the single-turn `_call` path.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def tool_use_block(tool_use_id: str, name: str, input: Dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=tool_use_id, name=name, input=input)


def thinking_block(thinking: str = "", signature: Optional[str] = None) -> SimpleNamespace:
    return SimpleNamespace(type="thinking", thinking=thinking, signature=signature)


def scripted_message(content: Sequence[SimpleNamespace], stop_reason: str, *,
                     input_tokens: int = 10, output_tokens: int = 10,
                     stop_details: Optional[SimpleNamespace] = None) -> SimpleNamespace:
    return SimpleNamespace(
        content=list(content), stop_reason=stop_reason, stop_details=stop_details,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


class _ScriptedMessages:
    def __init__(self, responses: List[SimpleNamespace], calls: List[Dict[str, Any]]):
        self._responses = list(responses)
        self._calls = calls

    def create(self, **kwargs):
        # `messages` is the loop's own growing history list, mutated in place
        # turn over turn -- snapshot it now so a later assertion on an
        # earlier call sees the request as it was sent, not as history looks
        # after every subsequent turn appended to the same list object.
        recorded = dict(kwargs)
        if "messages" in recorded:
            recorded["messages"] = list(recorded["messages"])
        self._calls.append(recorded)
        if not self._responses:
            raise AssertionError("ScriptedAnthropic ran out of scripted responses")
        return self._responses.pop(0)


class ScriptedAnthropic:
    """
    A minimal stand-in for `anthropic.Anthropic`, replaying a fixed list of
    responses in call order.

    `.calls` records every kwargs dict `messages.create` was invoked with,
    in order, so a test can assert on message-history echoing, the tools
    sent, or that a cached turn made no second call at all.
    """

    def __init__(self, responses: List[SimpleNamespace]):
        self.calls: List[Dict[str, Any]] = []
        self.messages = _ScriptedMessages(responses, self.calls)
