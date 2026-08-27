"""
Typed errors for the agent tool boundary.

A tool argument the model gets wrong -- a hallucinated KPI key, a dimension
that does not exist on this dataset -- must never reach pandas and must never
surface as an unhandled traceback. It becomes one of these, with a machine
readable `to_payload()` the loop hands back to the model as a tool result, so
the model can recover (pick a real key) instead of the request failing.

This is the mechanism, not just the convention: `ContractAPI.require` and
`require_dimension` raise these instead of returning None or NaN, which is
what makes "the model asked for something that does not exist" a normal,
recoverable turn of the loop rather than a 500.
"""
from __future__ import annotations

from typing import Any, Dict, List


class AgentToolError(ValueError):
    """Base class for every error a tool boundary can raise."""

    code = "tool_error"

    def to_payload(self) -> Dict[str, Any]:
        return {"error": self.code, "message": str(self)}


class UnknownKpiError(AgentToolError):
    """Raised when a requested KPI key is not in this dataset's contract."""

    code = "unknown_kpi"

    def __init__(self, key: str, valid_alternatives: List[str]):
        self.key = key
        self.valid_alternatives = list(valid_alternatives)
        super().__init__(
            f"Unknown KPI key '{key}'. This dataset does not measure that.")

    def to_payload(self) -> Dict[str, Any]:
        payload = super().to_payload()
        payload.update(requested=self.key, valid_alternatives=self.valid_alternatives)
        return payload


class UnknownDimensionError(AgentToolError):
    """Raised when a requested dimension does not exist on this dataset."""

    code = "unknown_dimension"

    def __init__(self, name: str, valid_alternatives: List[str]):
        self.name = name
        self.valid_alternatives = list(valid_alternatives)
        super().__init__(
            f"Unknown dimension '{name}'. This dataset cannot be sliced by that.")

    def to_payload(self) -> Dict[str, Any]:
        payload = super().to_payload()
        payload.update(requested=self.name, valid_alternatives=self.valid_alternatives)
        return payload


class MalformedFormulaError(AgentToolError):
    """
    Raised when a KPI's formula cannot be parsed.

    `compile_contract` (`resolver.py:397-400`) swallows exactly this case: a
    malformed entry is skipped so one bad formula cannot make a whole contract
    unusable, which is right for compilation and wrong for introspection --
    the KPI simply vanishes and nothing records why. Introspection raises
    instead, and `formula.unparsable_kpis` reports every entry compilation
    dropped.
    """

    code = "malformed_formula"

    def __init__(self, key: str, expression: str, reason: str):
        self.key = key
        self.expression = expression
        self.reason = reason
        super().__init__(f"The formula for '{key}' could not be parsed: {reason}")

    def to_payload(self) -> Dict[str, Any]:
        payload = super().to_payload()
        payload.update(requested=self.key, expression=self.expression, reason=self.reason)
        return payload
