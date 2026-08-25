"""
Field profiling — working out what each column actually MEANS.

`detect_schema` in `app.engines.metrics` classifies a column as date, numeric or
categorical. That is enough to sum things; it is not enough to know whether a
KPI is meaningful. Knowing that `inventory_units` is a *level* and not a *flow*,
or that `recovered_patients` is a subset of `admissions` and therefore that
`recovered / admissions` is a genuine rate while `revenue / cost` is not, is
what separates a real KPI from an arithmetic accident.

Three kinds of signal are combined, in increasing order of authority:

  1. name tokens      — cheap, and wrong often enough that it never decides alone
  2. distribution     — integrality, sign, range, cardinality
  3. relational checks— row-level containment between columns; the only evidence
                        strong enough to justify proposing a rate

Every conclusion carries the checks that produced it in `evidence`, so a
proposal can be audited rather than trusted.

The frame passed in is expected to have normalised column names already (the
ingestion path does this in `metrics.normalise_columns`). This module is
deliberately free of imports from `app.engines` so the dependency runs one way.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# name-token vocabularies
#
# These are hints, never verdicts: a distribution check can and does override
# them. `DIMENSION_HINTS` in engines/metrics.py has been dead since it was
# written; the same idea earns its keep here.
# ---------------------------------------------------------------------------
MONEY_TOKENS = {
    "revenue", "sales", "cost", "costs", "price", "spend", "spending", "amount",
    "value", "profit", "income", "expense", "expenses", "fee", "fees", "charge",
    "charges", "billed", "billing", "payment", "payments", "budget", "gmv",
    "turnover", "cogs", "margin", "salary", "wage", "wages", "premium", "claim",
    "claims", "reimbursement", "usd", "eur", "gbp", "inr",
}
RATE_TOKENS = {
    "rate", "pct", "percent", "percentage", "share", "proportion", "ratio",
    "utilisation", "utilization", "occupancy", "penetration", "attach",
}
# A time UNIT in the name is not enough to make a column a duration:
# `bed_days` is a summable quantity of bed-days, while `length_of_stay_days` is
# a per-case duration that must be averaged. Only a duration CONCEPT decides.
DURATION_UNIT_TOKENS = {"days", "day", "hours", "hour", "minutes", "minute", "seconds", "secs"}
DURATION_CONCEPT_TOKENS = {
    "duration", "length", "los", "stay", "tenure", "age", "lead", "leadtime",
    "cycle", "elapsed", "wait", "waiting", "downtime", "uptime", "runtime", "time",
}
DURATION_TOKENS = DURATION_UNIT_TOKENS | DURATION_CONCEPT_TOKENS
SCORE_TOKENS = {"score", "rating", "nps", "csat", "ces", "index", "grade", "satisfaction"}
COUNT_TOKENS = {
    "count", "orders", "units", "customers", "users", "visits", "admissions",
    "discharges", "patients", "tickets", "returns", "events", "sessions",
    "transactions", "shipments", "claims", "cases", "incidents", "employees",
    "subscribers", "signups", "leads", "clicks", "impressions", "quantity", "qty",
}
# A stock is a level measured at a point in time. Summing one across periods is
# meaningless: twelve month-end headcounts do not add up to an annual headcount.
STOCK_TOKENS = {
    "inventory", "stock", "on_hand", "onhand", "balance", "headcount", "occupied",
    "level", "levels", "position", "capacity", "backlog", "outstanding", "open",
    "active", "beds", "seats", "available", "in_stock", "wip",
}
IDENTIFIER_TOKENS = {"id", "uuid", "guid", "key", "code", "number", "no", "ref", "reference"}

# Fields that read as an outflow, for the profit-shaped derivation rule.
OUTFLOW_TOKENS = {
    "cost", "costs", "cogs", "expense", "expenses", "spend", "spending", "fee",
    "fees", "charge", "charges", "salary", "wage", "wages", "budget", "payment",
    "payments", "refund", "refunds", "discount", "discounts", "tax", "taxes",
}
INFLOW_TOKENS = {
    "revenue", "sales", "income", "turnover", "gmv", "billed", "billing",
    "receipts", "collections", "premium", "reimbursement", "value", "amount",
}

_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")

# A containment relation is accepted only if it holds on nearly every row. The
# few percent of slack absorbs data-entry noise without letting a coincidental
# ordering masquerade as a real subset relation.
SUBSET_ROW_THRESHOLD = 0.95


def tokenise(name: str) -> List[str]:
    """Split a column name into lowercase tokens for vocabulary matching."""
    return [t for t in _TOKEN_SPLIT.split(str(name).strip().lower()) if t]


# ---------------------------------------------------------------------------
# the profile
# ---------------------------------------------------------------------------
@dataclass
class FieldProfile:
    name: str
    role: str = "unknown"               # measure | dimension | time | identifier | unknown
    semantic_type: str = "unknown"
    additivity: str = "flow"
    null_rate: float = 0.0
    distinct_count: int = 0
    min: Optional[float] = None
    max: Optional[float] = None
    median: Optional[float] = None
    is_integer: bool = False
    is_non_negative: bool = True
    name_tokens: List[str] = field(default_factory=list)
    subset_of: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_measure(self) -> bool:
        return self.role == "measure"

    @property
    def default_aggregation(self) -> str:
        return {
            "flow": "sum",
            "stock": "mean",
            "rate": "mean",
            "non_additive": "mean",
        }.get(self.additivity, "sum")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "semantic_type": self.semantic_type,
            "additivity": self.additivity,
            "null_rate": round(self.null_rate, 4),
            "distinct_count": self.distinct_count,
            "min": self.min,
            "max": self.max,
            "median": self.median,
            "is_integer": self.is_integer,
            "is_non_negative": self.is_non_negative,
            "subset_of": list(self.subset_of),
            "default_aggregation": self.default_aggregation,
            "evidence": self.evidence,
        }


@dataclass
class HierarchyCandidate:
    child: str
    parent: str
    violations: int
    checked_members: int
    child_cardinality: int
    parent_cardinality: int

    @property
    def clean(self) -> bool:
        return self.violations == 0


@dataclass
class DatasetProfile:
    fields: List[FieldProfile] = field(default_factory=list)
    row_grain: List[str] = field(default_factory=list)
    row_grain_is_unique: bool = False
    duplicate_row_groups: int = 0
    hierarchies: List[HierarchyCandidate] = field(default_factory=list)
    date_column: str = ""
    row_count: int = 0

    def get(self, name: str) -> Optional[FieldProfile]:
        return next((f for f in self.fields if f.name == name), None)

    @property
    def measures(self) -> List[FieldProfile]:
        return [f for f in self.fields if f.role == "measure"]

    @property
    def dimensions(self) -> List[FieldProfile]:
        return [f for f in self.fields if f.role == "dimension"]

    def to_dict_list(self) -> List[Dict[str, Any]]:
        return [f.to_dict() for f in self.fields]


# ---------------------------------------------------------------------------
# semantic inference
# ---------------------------------------------------------------------------
def _looks_like_flag(series: pd.Series) -> bool:
    vals = set(pd.unique(series.dropna()))
    return len(vals) > 0 and vals.issubset({0, 1, 0.0, 1.0, True, False})


def infer_semantic_type(name: str, tokens: Sequence[str], series: pd.Series,
                        is_integer: bool, vmin: Optional[float],
                        vmax: Optional[float]) -> Tuple[str, Dict[str, Any]]:
    """Semantic type for a numeric measure column, with the reasoning kept."""
    tok = set(tokens)
    why: Dict[str, Any] = {}

    # A 0/1 column is only a flag when the name does not say otherwise. A
    # `returns` column can be entirely 0s and 1s at a fine grain and still be a
    # count of returns, which is what it must aggregate as.
    named_measure = tok & (COUNT_TOKENS | MONEY_TOKENS | DURATION_TOKENS | SCORE_TOKENS)
    if _looks_like_flag(series) and not named_measure:
        why["flag"] = "values are only 0/1 and the name carries no measure vocabulary"
        return "flag", why

    if tok & RATE_TOKENS:
        # A name saying "rate" is only believed if the values could BE a rate.
        in_pct = vmax is not None and vmin is not None and -1.0 <= vmin and vmax <= 100.0
        if in_pct:
            why["rate_pct"] = f"name token {sorted(tok & RATE_TOKENS)} and range [{vmin}, {vmax}]"
            return "rate_pct", why
        why["rate_rejected"] = f"name suggests a rate but range [{vmin}, {vmax}] is not 0-100"

    if tok & DURATION_CONCEPT_TOKENS:
        why["duration"] = f"duration concept token {sorted(tok & DURATION_CONCEPT_TOKENS)}"
        return "duration", why
    if tok & DURATION_UNIT_TOKENS:
        # A time unit with no duration concept behind it — treat as a quantity.
        why["duration_rejected"] = (
            f"time unit {sorted(tok & DURATION_UNIT_TOKENS)} with no duration concept; "
            "read as a summable quantity, not a per-unit duration"
        )

    if tok & MONEY_TOKENS:
        why["money"] = f"name token {sorted(tok & MONEY_TOKENS)}"
        return "money", why

    if tok & SCORE_TOKENS:
        why["score"] = f"name token {sorted(tok & SCORE_TOKENS)}"
        return "score", why

    if tok & COUNT_TOKENS:
        why["count"] = f"name token {sorted(tok & COUNT_TOKENS)}"
        return "count", why

    # No vocabulary hit. Integers are counts; anything else is left as a count
    # too, which is the conservative choice — it aggregates by summing, exactly
    # what the product already did with unrecognised numeric columns.
    why["default"] = "no name-vocabulary match; defaulted to count"
    why["is_integer"] = is_integer
    return "count", why


def infer_additivity(semantic_type: str, tokens: Sequence[str]) -> Tuple[str, str]:
    tok = set(tokens)
    if semantic_type in ("rate_pct", "ratio"):
        return "rate", "a rate is never summed and never averaged across periods"
    if semantic_type in ("duration", "score"):
        return "non_additive", f"a {semantic_type} is averaged, not summed"
    if tok & STOCK_TOKENS:
        return "stock", f"name token {sorted(tok & STOCK_TOKENS)} indicates a level, not a flow"
    return "flow", "accumulates over the period"


def _cash_direction(tokens: Sequence[str]) -> Optional[str]:
    """'inflow' / 'outflow' / None — used by the profit-shaped derivation rule."""
    tok = set(tokens)
    if tok & OUTFLOW_TOKENS:
        return "outflow"
    if tok & INFLOW_TOKENS:
        return "inflow"
    return None


# ---------------------------------------------------------------------------
# relational checks
# ---------------------------------------------------------------------------
def detect_subset_relations(df: pd.DataFrame, measures: List[FieldProfile]) -> None:
    """
    Populate `subset_of` — the measures this one is contained by, row by row.

    This is the check that makes rate discovery honest. `recovered <= admissions`
    on essentially every row means recovery rate is a real rate; `revenue` and
    `cost_of_goods` have no such relation, so revenue/cost is not proposed as
    one however arithmetically valid it is.

    Containment is only tested **within a semantic family** — counts against
    counts, money against money. `orders <= revenue` holds on every row of a
    typical dataset and means precisely nothing: a subset relation is a claim
    that two columns measure the same kind of thing and one is part of the
    other, which a count and a currency amount can never satisfy.
    """
    # Stocks are excluded on both sides: a flow measured over a period cannot be
    # "part of" a level measured at an instant, whatever the arithmetic says.
    eligible = [m for m in measures
                if m.semantic_type in ("count", "money", "flag") and m.additivity == "flow"]

    def same_family(x: FieldProfile, y: FieldProfile) -> bool:
        countish = {"count", "flag"}
        return (x.semantic_type in countish and y.semantic_type in countish) or (
            x.semantic_type == "money" and y.semantic_type == "money"
        )

    for a in eligible:
        sa = pd.to_numeric(df[a.name], errors="coerce")
        checks: Dict[str, Any] = {}
        for b in eligible:
            if a.name == b.name or not same_family(a, b):
                continue
            sb = pd.to_numeric(df[b.name], errors="coerce")
            both = sa.notna() & sb.notna()
            n = int(both.sum())
            if n == 0:
                continue
            within = float((sa[both] <= sb[both]).mean())
            # Identical columns satisfy <= trivially; a real subset must also be
            # strictly smaller in total, or it carries no information.
            strictly_smaller = float(sa[both].sum()) < float(sb[both].sum())
            if within >= SUBSET_ROW_THRESHOLD and strictly_smaller:
                a.subset_of.append(b.name)
                checks[b.name] = {"rows_within": round(within, 4), "n": n, "passed": True}
            elif within >= 0.5:
                checks[b.name] = {"rows_within": round(within, 4), "n": n, "passed": False}
        if checks:
            a.evidence["subset_checks"] = checks


def detect_row_grain(df: pd.DataFrame, date_column: str,
                     dimensions: Sequence[str]) -> Tuple[List[str], bool, int]:
    """
    The smallest key that identifies a row — the floor for every KPI's grain.

    Greedy rather than exhaustive: start from the date column and add whichever
    dimension removes the most duplication, until the key is unique or the
    dimensions run out. Exhaustive search is exponential in the dimension count
    and buys nothing here.
    """
    if date_column not in df.columns:
        return [], False, len(df)

    key = [date_column]
    remaining = [d for d in dimensions if d in df.columns]
    total = len(df)

    def dup_groups(cols: List[str]) -> int:
        return int(total - df.groupby(cols, dropna=False).ngroups)

    dups = dup_groups(key)
    while dups > 0 and remaining:
        best, best_dups = None, dups
        for dim in remaining:
            d = dup_groups(key + [dim])
            if d < best_dups:
                best, best_dups = dim, d
        if best is None:
            break
        key.append(best)
        remaining.remove(best)
        dups = best_dups

    return key, dups == 0, dups


# A pair this broken is not a damaged hierarchy, it is two independent
# dimensions. Reporting those as violations would bury the real findings.
HIERARCHY_VIOLATION_TOLERANCE = 0.25


def detect_hierarchies(df: pd.DataFrame, dimensions: Sequence[str],
                       max_members: int = 5000) -> List[HierarchyCandidate]:
    """
    Find dimension pairs where each child value rolls up to exactly one parent.

    Returns clean containments (a usable hierarchy level) and *nearly* clean
    ones — a pair that looks intended but has exceptions, which is a real data
    problem the user must see. Orthogonal pairs, where most members map to many
    parents, are dropped: region and product are simply unrelated, not broken.

    Violations are reported, never repaired.
    """
    dims = [d for d in dimensions if d in df.columns]
    out: List[HierarchyCandidate] = []
    for child in dims:
        child_card = int(df[child].nunique(dropna=True))
        if child_card == 0 or child_card > max_members:
            continue
        for parent in dims:
            if child == parent:
                continue
            parent_card = int(df[parent].nunique(dropna=True))
            # A parent must be strictly coarser, or it is not a parent.
            if parent_card == 0 or parent_card >= child_card:
                continue
            per_child = df.groupby(child, dropna=True)[parent].nunique(dropna=True)
            checked = int(len(per_child))
            violations = int((per_child > 1).sum())
            # A handful of exceptions is a damaged hierarchy worth reporting; a
            # large fraction means the two dimensions are simply independent. The
            # absolute floor keeps the ratio from over-firing on a small dataset,
            # where one bad member is already a third of the population.
            if violations > max(2, HIERARCHY_VIOLATION_TOLERANCE * checked):
                continue
            out.append(HierarchyCandidate(
                child=child, parent=parent,
                violations=violations, checked_members=checked,
                child_cardinality=child_card, parent_cardinality=parent_card,
            ))
    return out


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def profile_dataset(df: pd.DataFrame, date_column: str, dimensions: Sequence[str],
                    measures: Sequence[str]) -> DatasetProfile:
    """
    Profile a prepared frame.

    `date_column`, `dimensions` and `measures` come from the existing
    `DatasetSchema`, so profiling adds a semantic layer on top of the detection
    the ingestion path already does rather than replacing it.
    """
    profiles: List[FieldProfile] = []
    row_count = int(len(df))

    if date_column and date_column in df.columns:
        profiles.append(FieldProfile(
            name=date_column, role="time", semantic_type="date", additivity="non_additive",
            distinct_count=int(df[date_column].nunique(dropna=True)),
            null_rate=float(df[date_column].isna().mean()),
            name_tokens=tokenise(date_column),
            evidence={"source": "schema date column"},
        ))

    for name in dimensions:
        if name not in df.columns:
            continue
        tokens = tokenise(name)
        distinct = int(df[name].nunique(dropna=True))
        role = "dimension"
        semantic = "category"
        # A near-unique text column is an identifier, not something to slice by.
        if set(tokens) & IDENTIFIER_TOKENS and row_count and distinct > 0.8 * row_count:
            role, semantic = "identifier", "id"
        profiles.append(FieldProfile(
            name=name, role=role, semantic_type=semantic, additivity="non_additive",
            distinct_count=distinct, null_rate=float(df[name].isna().mean()),
            name_tokens=tokens,
            evidence={"cardinality": distinct, "cardinality_ratio":
                      round(distinct / row_count, 4) if row_count else None},
        ))

    for name in measures:
        if name not in df.columns:
            continue
        tokens = tokenise(name)
        series = pd.to_numeric(df[name], errors="coerce")
        non_null = series.dropna()
        vmin = float(non_null.min()) if len(non_null) else None
        vmax = float(non_null.max()) if len(non_null) else None
        vmed = float(non_null.median()) if len(non_null) else None
        is_integer = bool(len(non_null)) and bool(np.allclose(non_null, non_null.round()))
        semantic, why = infer_semantic_type(name, tokens, non_null, is_integer, vmin, vmax)
        additivity, additivity_why = infer_additivity(semantic, tokens)
        direction = _cash_direction(tokens)

        evidence: Dict[str, Any] = {"semantic_type": why, "additivity": additivity_why}
        if direction:
            evidence["cash_direction"] = direction

        profiles.append(FieldProfile(
            name=name, role="measure", semantic_type=semantic, additivity=additivity,
            null_rate=float(series.isna().mean()),
            distinct_count=int(non_null.nunique()),
            min=vmin, max=vmax, median=vmed,
            is_integer=is_integer,
            is_non_negative=bool(vmin is not None and vmin >= 0),
            name_tokens=tokens, evidence=evidence,
        ))

    profile = DatasetProfile(
        fields=profiles, date_column=date_column, row_count=row_count,
    )
    detect_subset_relations(df, profile.measures)
    grain, unique, dups = detect_row_grain(df, date_column, dimensions)
    profile.row_grain = grain
    profile.row_grain_is_unique = unique
    profile.duplicate_row_groups = dups
    profile.hierarchies = detect_hierarchies(df, dimensions)
    return profile


def cash_direction(profile: FieldProfile) -> Optional[str]:
    """Public accessor for the inflow/outflow hint recorded during profiling."""
    return profile.evidence.get("cash_direction")
