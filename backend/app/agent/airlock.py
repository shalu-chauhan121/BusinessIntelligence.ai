"""
The validation airlock in front of the tool registry.

A tool argument's *semantics* -- is this a KPI this dataset measures, is this
dimension real, does this time filter parse -- is already validated inside
each engine, first thing, before any pandas work
(`query.QueryEngine.query`, `query.py:118-141`, is the canonical example).
That half of the airlock has existed since batch O, and this module does not
duplicate it.

What was missing was *type and shape*, measured by fuzzing the live registry
rather than assumed:

  * `search_kpis(text=True)` executed and returned a plausible empty answer
    (`{"matches": []}`, `is_error=False`) -- a malformed argument that looked
    like a real "no match" result, so the model never learns it sent a bool.
  * `query_kpi(filters=["region"])` reached pandas and leaked
    "dictionary update sequence element #0 has length 6; 2 is required" --
    the generic catch-all, carrying no `valid_alternatives`.
  * `test_confounders(candidates="not_a_list")` had its string iterated
    character by character and reported `unknown_kpi: 'n'` -- actively
    misdirecting: the model is told its KPI key is wrong when its argument
    *type* is wrong, and "recovers" by picking a different key.

`tests/test_airlock.py` pins each of these three by name. `validate()` checks
arguments against the *exact* JSON Schema `registry.py` already generated and
already sent to the model (`build_input_schema`) -- one source of truth, so a
schema the model was shown and a schema it is checked against cannot drift
apart the way two independently maintained ones could.

`TimeFilter`'s own schema (`registry.TIME_FILTER_SCHEMA`) is deliberately
permissive -- it checks the outer shape and the `type` enum only.
`timefilter.parse` remains the authority on which keys each of the eight
shapes needs, and its `MalformedTimeFilterError` already names the valid
shapes; restating that union here would be the discriminated schema decision
55 chose not to pay for, paid for anyway.
"""
from __future__ import annotations

import re
from typing import Any, List, Mapping, Optional, Sequence

import jsonschema

from .errors import InvalidArgumentError

_REQUIRED_PROPERTY = re.compile(r"'([^']+)' is a required property")


def _enum_at(schema: Mapping[str, Any], path: Sequence[Any]) -> Optional[List[Any]]:
    """
    Walk `schema` along a `jsonschema` error's `path` to the failing
    subschema, and return its `enum` if it has one -- so a bad enum value
    reports the same `valid_alternatives` the tool definition already showed
    the model, not nothing.
    """
    node: Any = schema
    for key in path:
        if not isinstance(node, Mapping):
            return None
        node = node.get("items", {}) if isinstance(key, int) else node.get("properties", {}).get(key, {})
    if not isinstance(node, Mapping):
        return None
    if "enum" in node:
        return list(node["enum"])
    if "anyOf" in node:
        # `kpi_keys` is `anyOf: [string-enum, array-of-string-enum]` (a
        # single key or a list of them) -- neither branch alone is "the"
        # schema for this parameter, so union whatever enum each carries.
        found: List[Any] = []
        for branch in node["anyOf"]:
            values = _enum_at(branch, ()) or _enum_at(branch.get("items", {}), ()) if isinstance(branch, Mapping) else None
            for v in values or []:
                if v not in found:
                    found.append(v)
        return found or None
    return None


def validate(tool_name: str, schema: Mapping[str, Any], args: Mapping[str, Any]) -> None:
    """
    Raise `InvalidArgumentError` if `args` does not match `schema`.

    Called by `registry.dispatch` before any engine method runs: a malformed
    argument must never reach pandas, and must never silently execute into a
    plausible-looking wrong answer.
    """
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(dict(args)), key=lambda e: (len(list(e.path)), list(e.path)))
    if not errors:
        return

    error = errors[0]
    path = list(error.path)
    if path:
        argument = ".".join(str(p) for p in path)
        value = args.get(str(path[0])) if not isinstance(path[0], int) else args
        alternatives = _enum_at(schema, path) or []
    else:
        # A `required` error has no `path` -- it names the missing key only
        # in its message ("'text' is a required property").
        match = _REQUIRED_PROPERTY.search(error.message)
        argument = match.group(1) if match else tool_name
        value = None
        alternatives = []

    raise InvalidArgumentError(argument, value, error.message, valid_alternatives=alternatives)
