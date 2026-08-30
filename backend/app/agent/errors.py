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

from typing import Any, Dict, List, Optional


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


class UnknownMemberError(AgentToolError):
    """
    Raised when a requested dimension member is not a value that column holds.

    The counterpart to `UnknownKpiError` for the other half of a filtered
    query. A model that asks for a region the dataset does not have must get a
    recoverable turn naming the regions it does have, not an empty slice that
    is indistinguishable from a real zero.
    """

    code = "unknown_member"

    # A wide column has thousands of values and none of them help the model if
    # the payload is too large to read.
    MAX_ALTERNATIVES = 20

    def __init__(self, dimension: str, value: str, valid_alternatives: List[str]):
        self.dimension = dimension
        self.value = value
        self.valid_alternatives = list(valid_alternatives)[:self.MAX_ALTERNATIVES]
        super().__init__(
            f"Unknown member '{value}' for dimension '{dimension}'. "
            f"That column does not hold that value.")

    def to_payload(self) -> Dict[str, Any]:
        payload = super().to_payload()
        payload.update(requested=self.value, dimension=self.dimension,
                       valid_alternatives=self.valid_alternatives)
        return payload


class MalformedTimeFilterError(AgentToolError):
    """
    Raised when a `TimeFilter` spec cannot be parsed.

    A model can invent a `type` that does not exist, omit a required key, or
    name a quarter outside 1-4. None of that should ever reach pandas as a
    KeyError or a silently-wrong mask; it becomes this, with the valid shapes
    named so the model can retry with a spec that parses.
    """

    code = "malformed_time_filter"

    def __init__(self, spec: Any, reason: str, valid_types: List[str]):
        self.spec = spec
        self.reason = reason
        self.valid_types = list(valid_types)
        super().__init__(f"Could not parse time filter {spec!r}: {reason}")

    def to_payload(self) -> Dict[str, Any]:
        payload = super().to_payload()
        payload.update(requested=self.spec, reason=self.reason,
                       valid_types=self.valid_types)
        return payload


class EmptyPeriodError(AgentToolError):
    """
    Raised when a `TimeFilter` selects zero rows.

    An empty selection must never be silently reported as a real zero -- it
    means the requested period does not exist in this dataset at all. Partial
    coverage (some but not all of the requested span is held) is not this
    error; it is reported instead as a `missing` list alongside a real answer.
    """

    code = "empty_period"

    # A wide dataset can hold hundreds of periods; only a handful help the
    # model see what it should have asked for instead.
    MAX_AVAILABLE = 20

    def __init__(self, requested: Any, available_periods: List[str]):
        self.requested = requested
        self.available_periods = list(available_periods)[:self.MAX_AVAILABLE]
        super().__init__(
            f"No rows match the requested time filter {requested!r}. "
            "This dataset does not hold that period.")

    def to_payload(self) -> Dict[str, Any]:
        payload = super().to_payload()
        payload.update(requested=self.requested,
                       available_periods=self.available_periods)
        return payload


class InvalidArgumentError(AgentToolError):
    """
    Generic typed error for a tool argument that fails validation but is not
    one of the more specific kinds above (an unknown KPI, dimension, member,
    or time filter). Reused by the validation airlock in front of the full
    tool registry.
    """

    code = "invalid_argument"

    def __init__(self, argument: str, value: Any, reason: str,
                valid_alternatives: Optional[List[Any]] = None):
        self.argument = argument
        self.value = value
        self.reason = reason
        self.valid_alternatives = list(valid_alternatives) if valid_alternatives else []
        super().__init__(f"Invalid value for '{argument}': {value!r}. {reason}")

    def to_payload(self) -> Dict[str, Any]:
        payload = super().to_payload()
        payload.update(argument=self.argument, value=self.value, reason=self.reason,
                       valid_alternatives=self.valid_alternatives)
        return payload
