"""
The per-request handle the agent loop threads everywhere.

`registry.build(api, df)` already needs one `ContractAPI` and one dataframe;
nothing before this module owned constructing that pair for a real request,
loading the dataset exactly once, or tracking a turn budget across a run.
`AgentContext.build(...)` is that: one call, built on the same
`dataset_service.load` every other route already uses, so a dataset loads
through the same cache-on-(path, mtime) path everywhere in this codebase.

**`df` is shared by reference, not copied.** `dataset_service.load` returns
the cached frame itself (`dataset_service.py:397`); only the schema is
shallow-copied per call, and `schema.member_catalogue` is shared across every
copy. Routes in this codebase are sync `def`, so FastAPI runs them in a
threadpool -- this is a real concurrency surface, not a theoretical one.
**Never mutate `ctx.df` in place.** Every registered tool already treats `df`
as read-only; this context does not change that contract, it just makes it
explicit at the one place a request's handle is built.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd

from ..config import get_settings
from ..engines.metrics import DatasetSchema
from ..services import dataset_service
from .contract_api import ContractAPI
from .registry import ToolRegistry
from .registry import build as build_registry


@dataclass
class Budget:
    """
    Turn accounting for one `AgentContext`.

    `LLMClient._call_with_tools` already enforces `max_turns` itself and
    returns how many it actually used (`ToolLoopResult.turns`) -- this is
    not a second enforcement path, it is what a caller records that result
    into so `remaining_turns` / `exhausted` are answerable without re-deriving
    them from a `ToolLoopResult` the caller may not still have at hand.
    """

    max_turns: int
    turns_used: int = 0

    @property
    def remaining_turns(self) -> int:
        return max(0, self.max_turns - self.turns_used)

    @property
    def exhausted(self) -> bool:
        return self.turns_used >= self.max_turns

    def record(self, turns: int) -> None:
        self.turns_used += max(0, turns)


@dataclass
class AgentContext:
    """
    One dataset, loaded once, with the facade and tool registry built on it.

    Construct via `AgentContext.build(...)`, not the constructor directly --
    building `api`/`registry` from a `df`/`schema` pair that were not loaded
    together (e.g. from two different calls to `dataset_service.load`) would
    silently mismatch a contract against a frame it was not compiled for.
    """

    uid: str
    dataset: Dict[str, Any]
    df: pd.DataFrame
    schema: DatasetSchema
    api: ContractAPI
    registry: ToolRegistry
    budget: Budget

    @classmethod
    def build(cls, uid: str, dataset: Dict[str, Any], *,
             as_of: Optional[str] = None, max_turns: Optional[int] = None) -> "AgentContext":
        df, schema = dataset_service.load(dataset, uid, as_of=as_of)
        api = ContractAPI(schema)
        registry = build_registry(api, df)
        settings = get_settings()
        budget = Budget(max_turns=max_turns if max_turns is not None else settings.llm_max_turns)
        return cls(uid=uid, dataset=dataset, df=df, schema=schema,
                   api=api, registry=registry, budget=budget)
