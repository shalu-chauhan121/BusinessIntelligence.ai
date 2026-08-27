"""
Metric registry and dataset schema handling.

Everything numerical the product shows is produced here or in the other modules
of `app.engines`. The LLM never computes a business number.

**The KPI Contract is the source of truth for KPI definitions.** The `METRICS`
dict below is no longer that: it is a seed for the general KPI library in
`app.kpi.library`, kept here so the fallback path stays honest when a dataset
has no contract yet.

Every lookup helper takes an optional `resolver` — the compiled contract for the
dataset being analysed, produced by `app.kpi.resolver.compile_contract`. Because
`CompiledKpi` exposes the same attribute surface as `MetricSpec`
(`key, label, unit, kind, scale, higher_is_better, additive`), the engines are
indifferent to which they were handed, and a caller that passes nothing gets
exactly the behaviour this module had before the contract existed.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

# A resolver maps a KPI key to something MetricSpec-shaped. It is typed loosely
# on purpose: importing app.kpi here would make the dependency circular, and the
# only contract between the two is the attribute surface described above.
Resolver = Mapping[str, Any]

DIMENSION_HINTS = ["region", "product", "channel", "segment", "category", "sub_category",
                   "country", "store", "team", "sales_rep", "industry", "plan", "tier"]
DATE_COLUMNS = ["date", "period", "week", "day", "month", "timestamp", "order_date"]


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    kind: str                      # "sum" | "mean" | "ratio"
    unit: str                      # "currency" | "count" | "percent" | "ratio"
    requires: List[str]
    numerator: Optional[str] = None
    denominator: Optional[str] = None
    scale: float = 1.0
    higher_is_better: bool = True
    description: str = ""
    numerator_expr: Optional[str] = None   # "revenue-cost_of_goods"

    @property
    def additive(self) -> bool:
        return self.kind == "sum"


METRICS: Dict[str, MetricSpec] = {
    m.key: m
    for m in [
        MetricSpec("revenue", "Revenue", "sum", "currency", ["revenue"],
                   description="Total sales value in the period."),
        MetricSpec("gross_profit", "Gross profit", "sum", "currency",
                   ["revenue", "cost_of_goods"], numerator_expr="revenue-cost_of_goods",
                   description="Revenue less cost of goods sold."),
        MetricSpec("units_sold", "Units sold", "sum", "count", ["units_sold"],
                   description="Total unit volume shipped."),
        MetricSpec("orders", "Orders", "sum", "count", ["orders"],
                   description="Number of orders placed."),
        MetricSpec("customers", "Customers", "sum", "count", ["customers"],
                   description="Number of purchasing customers."),
        MetricSpec("marketing_spend", "Marketing spend", "sum", "currency", ["marketing_spend"],
                   higher_is_better=False, description="Marketing investment in the period."),
        MetricSpec("cost_of_goods", "Cost of goods", "sum", "currency", ["cost_of_goods"],
                   higher_is_better=False, description="Direct cost of the units sold."),
        MetricSpec("stockout_events", "Stockout events", "sum", "count", ["stockout_events"],
                   higher_is_better=False, description="Occasions demand could not be served from stock."),
        MetricSpec("inventory_units", "Inventory level", "mean", "count", ["inventory_units"],
                   description="Average on-hand inventory across the period (a stock, not a flow)."),
        MetricSpec("returns", "Returns", "sum", "count", ["returns"],
                   higher_is_better=False, description="Returned orders."),
        MetricSpec("support_tickets", "Support tickets", "sum", "count", ["support_tickets"],
                   higher_is_better=False, description="Inbound support contacts."),

        # --- ratio KPIs -----------------------------------------------------
        MetricSpec("gross_margin_pct", "Gross margin %", "ratio", "percent",
                   ["revenue", "cost_of_goods"], numerator_expr="revenue-cost_of_goods",
                   denominator="revenue", scale=100.0,
                   description="Gross profit as a share of revenue."),
        MetricSpec("avg_order_value", "Average order value", "ratio", "currency",
                   ["revenue", "orders"], numerator="revenue", denominator="orders",
                   description="Revenue per order."),
        MetricSpec("avg_selling_price", "Average selling price", "ratio", "currency",
                   ["revenue", "units_sold"], numerator="revenue", denominator="units_sold",
                   description="Realised price per unit — separates price effects from volume effects."),
        MetricSpec("fulfillment_rate", "Fulfilment rate", "ratio", "percent",
                   ["fulfilled_orders", "orders"], numerator="fulfilled_orders",
                   denominator="orders", scale=100.0,
                   description="Share of orders fulfilled — a supply-side signal."),
        MetricSpec("stockout_rate", "Stockout rate", "ratio", "percent",
                   ["stockout_events", "orders"], numerator="stockout_events",
                   denominator="orders", scale=100.0, higher_is_better=False,
                   description="Stockouts per order — a supply-side signal."),
        MetricSpec("return_rate", "Return rate", "ratio", "percent",
                   ["returns", "orders"], numerator="returns", denominator="orders",
                   scale=100.0, higher_is_better=False,
                   description="Returns per order — a quality signal."),
        MetricSpec("customer_acquisition_cost", "Customer acquisition cost", "ratio", "currency",
                   ["marketing_spend", "customers"], numerator="marketing_spend",
                   denominator="customers", higher_is_better=False,
                   description="Marketing spend per purchasing customer."),
        MetricSpec("units_per_order", "Units per order", "ratio", "ratio",
                   ["units_sold", "orders"], numerator="units_sold", denominator="orders",
                   description="Basket size."),
        MetricSpec("tickets_per_1k_orders", "Support tickets per 1k orders", "ratio", "ratio",
                   ["support_tickets", "orders"], numerator="support_tickets",
                   denominator="orders", scale=1000.0, higher_is_better=False,
                   description="Service load normalised by volume."),
    ]
}

DEFAULT_KPI_ORDER = [
    "revenue", "orders", "customers", "units_sold", "avg_order_value", "gross_margin_pct",
    "avg_selling_price", "fulfillment_rate", "stockout_rate", "inventory_units",
    "customer_acquisition_cost", "return_rate", "gross_profit", "marketing_spend",
    "units_per_order", "tickets_per_1k_orders", "support_tickets", "returns", "cost_of_goods",
]


# ---------------------------------------------------------------------------
# schema detection
# ---------------------------------------------------------------------------
@dataclass
class DatasetSchema:
    date_column: str
    dimensions: List[str] = field(default_factory=list)
    base_metrics: List[str] = field(default_factory=list)
    extra_metrics: List[str] = field(default_factory=list)
    available_kpis: List[str] = field(default_factory=list)
    row_count: int = 0
    date_min: str = ""
    date_max: str = ""
    quarters: List[str] = field(default_factory=list)
    years: List[int] = field(default_factory=list)
    grain: str = "unknown"
    warnings: List[str] = field(default_factory=list)
    # The compiled KPI Contract for this dataset, attached by `dataset_service.load`.
    # Not serialisable and deliberately excluded from `to_dict`.
    contract_resolver: Optional[Resolver] = field(default=None, repr=False, compare=False)
    contract_status: str = ""
    contract_version: int = 0
    # The detected business context, rebuilt from the contract by
    # `dataset_service.attach_contract`. This is what lets an explanation speak
    # about beds and admissions rather than about "units". Like the resolver it
    # is not serialisable and is excluded from `to_dict`.
    domain: Optional[Any] = field(default=None, repr=False, compare=False)
    # Per-source provenance and freshness when this dataset was reconciled from
    # several sources (see `app.services.reconciliation`), else None. Like the
    # resolver and domain it is not a plain value, so `to_dict` serialises it.
    sources: Optional[Any] = field(default=None, repr=False, compare=False)

    # -- contract semantics -------------------------------------------------
    def kpi_definition(self, key: str) -> Optional[Any]:
        """
        The full `KpiDefinition` behind a KPI key, when this dataset has a
        contract.

        The compiled resolver already carries the definition; the engines only
        ever received the label and unit, which is why a KPI's business meaning
        never reached an explanation. Returns None for a dataset with no
        contract, so every caller must treat the semantics as optional.
        """
        spec = (self.contract_resolver or {}).get(key)
        return getattr(spec, "definition", None) if spec else None

    @property
    def domain_key(self) -> str:
        """The detected domain key, or 'uncertain' when nothing was detected."""
        return getattr(self.domain, "domain", None) or "uncertain"

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()
             if k not in ("contract_resolver", "domain", "sources")}
        d["domain"] = self.domain_key
        d["sources"] = self.sources.to_dict() if self.sources is not None else None
        d["kpi_catalogue"] = [
            {
                "key": k,
                "label": metric_label(k, self.contract_resolver),
                "unit": metric_unit(k, self.contract_resolver),
                "higher_is_better": higher_is_better(k, self.contract_resolver),
                "description": metric_description(k, self.contract_resolver),
                "granularity": getattr(
                    metric_spec(k, self.contract_resolver), "granularity_label", ""),
                "kpi_status": getattr(metric_spec(k, self.contract_resolver), "status", ""),
            }
            for k in self.available_kpis
        ]
        return d


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    return df


def detect_schema(df: pd.DataFrame) -> DatasetSchema:
    df = normalise_columns(df)

    date_col = next((c for c in DATE_COLUMNS if c in df.columns), None)
    if date_col is None:
        for c in df.columns:
            # A numeric column must never be mistaken for a date: pandas will happily
            # read [1, 2, 3] as nanosecond timestamps in 1970.
            if pd.api.types.is_numeric_dtype(df[c]):
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    parsed = pd.to_datetime(df[c], errors="raise", format="mixed")
            except Exception:
                continue
            if parsed.notna().mean() <= 0.9:
                continue
            years = parsed.dt.year.dropna()
            if len(years) and years.between(1990, 2100).mean() > 0.9:
                date_col = c
                break
    if date_col is None:
        raise ValueError(
            "No date column found. Add a 'date' column formatted as YYYY-MM-DD."
        )

    dates = pd.to_datetime(df[date_col], errors="coerce")
    if dates.notna().sum() == 0:
        raise ValueError(f"Column '{date_col}' could not be parsed as dates.")

    numeric_cols, dim_cols = [], []
    for c in df.columns:
        if c == date_col:
            continue
        series = df[c]
        coerced = pd.to_numeric(series, errors="coerce")
        # Judge numeric-ness over the values that are actually PRESENT. A
        # reconciled frame legitimately carries blanks where no source covered a
        # cell, and those must not tip a measure into being read as a dimension.
        populated = int(series.notna().sum())
        numeric_share = (coerced.notna().sum() / populated) if populated else 0.0
        if numeric_share > 0.8:
            numeric_cols.append(c)
        else:
            if series.nunique(dropna=True) <= max(60, int(len(df) * 0.2)):
                dim_cols.append(c)

    base = [c for c in numeric_cols if c in METRICS and METRICS[c].kind in ("sum", "mean")]
    extra = [c for c in numeric_cols if c not in METRICS]

    available: List[str] = []
    for key in DEFAULT_KPI_ORDER:
        spec = METRICS[key]
        if all(r in numeric_cols for r in spec.requires):
            available.append(key)
    available += sorted(extra)

    uniq_dates = sorted(dates.dropna().dt.date.unique())
    grain = "unknown"
    if len(uniq_dates) > 1:
        gaps = np.diff(np.array(uniq_dates, dtype="datetime64[D]")).astype(int)
        med = int(np.median(gaps)) if len(gaps) else 0
        grain = {1: "daily", 7: "weekly"}.get(med, "monthly" if 28 <= med <= 31 else f"{med}-day")

    q = (dates.dt.year.astype("Int64").astype(str) + "-Q" + dates.dt.quarter.astype("Int64").astype(str))
    quarters = sorted(x for x in q.dropna().unique())

    warnings: List[str] = []
    if len(quarters) < 5:
        warnings.append(
            f"Only {len(quarters)} quarter(s) of history. The significance test needs at least "
            "5 quarters to separate a real signal from normal variation, and 8+ to model seasonality."
        )
    if not dim_cols:
        warnings.append(
            "No dimension columns detected (region / product / channel / segment). "
            "Driver decomposition and cross-dimension consistency checks will be skipped."
        )
    if "revenue" not in numeric_cols:
        warnings.append("No 'revenue' column found — the default headline KPI will be the first available metric.")

    return DatasetSchema(
        date_column=date_col,
        dimensions=[c for c in dim_cols],
        base_metrics=base,
        extra_metrics=extra,
        available_kpis=available,
        row_count=int(len(df)),
        date_min=str(dates.min().date()),
        date_max=str(dates.max().date()),
        quarters=quarters,
        years=sorted({int(y) for y in dates.dt.year.dropna().unique()}),
        grain=grain,
        warnings=warnings,
    )


def prepare(df: pd.DataFrame, schema: DatasetSchema,
            preserve_missing: bool = False) -> pd.DataFrame:
    """
    Normalise, type and enrich the frame with period keys.

    `preserve_missing` keeps genuinely absent measures as NaN instead of
    zero-filling them. Off by default, so an ordinary upload behaves exactly as
    before (a blank cell in a hand-maintained CSV is a data-entry gap, and 0 is
    as good a default as any). A reconciled multi-source frame turns it on,
    because there a blank means "no source covered this" — zero-filling it would
    fabricate the very value reconciliation refused to invent.
    """
    df = normalise_columns(df)
    df["_date"] = pd.to_datetime(df[schema.date_column], errors="coerce")
    df = df.dropna(subset=["_date"])
    for c in schema.base_metrics + schema.extra_metrics:
        if c in df.columns:
            numeric = pd.to_numeric(df[c], errors="coerce")
            df[c] = numeric if preserve_missing else numeric.fillna(0.0)
    for c in schema.dimensions:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("Unknown")
    df["_year"] = df["_date"].dt.year
    df["_quarter"] = df["_date"].dt.quarter
    df["_period"] = df["_year"].astype(str) + "-Q" + df["_quarter"].astype(str)
    df["_month"] = df["_date"].dt.to_period("M").astype(str)
    df["_week"] = df["_date"].dt.to_period("W").apply(lambda p: p.start_time.date().isoformat())
    return df


# ---------------------------------------------------------------------------
# metric evaluation
# ---------------------------------------------------------------------------
def _col_sum(df: pd.DataFrame, expr: str) -> float:
    if "-" in expr:
        a, b = expr.split("-", 1)
        return float(df[a.strip()].sum() - df[b.strip()].sum())
    return float(df[expr].sum())


def metric_spec(metric_key: str, resolver: Optional[Resolver] = None) -> Optional[Any]:
    """
    The definition for a KPI key.

    The contract wins when it has the key; otherwise the seed registry answers,
    which is what keeps datasets with no contract working unchanged.
    """
    if resolver is not None and metric_key in resolver:
        return resolver[metric_key]
    return METRICS.get(metric_key)


def compute(df: pd.DataFrame, metric_key: str, resolver: Optional[Resolver] = None) -> float:
    """Aggregate one KPI over an arbitrary slice of rows."""
    if df is None or len(df) == 0:
        return float("nan")
    spec = metric_spec(metric_key, resolver)
    if spec is None:                                  # uploaded numeric column
        return float(df[metric_key].sum()) if metric_key in df.columns else float("nan")
    # A contract entry carries its own compiled formula.
    if hasattr(spec, "compute"):
        return spec.compute(df)

    if spec.kind == "sum":
        return _col_sum(df, spec.numerator_expr or spec.key)
    if spec.kind == "mean":
        return float(df[spec.key].mean())
    num = _col_sum(df, spec.numerator_expr or spec.numerator)
    den = _col_sum(df, spec.denominator)
    if den == 0:
        return float("nan")
    return num / den * spec.scale


def metric_components(df: pd.DataFrame, metric_key: str,
                      resolver: Optional[Resolver] = None):
    """(numerator, denominator) for ratio KPIs — needed for mix/rate decomposition."""
    spec = metric_spec(metric_key, resolver)
    if spec is None:
        return None, None
    if hasattr(spec, "components"):
        return spec.components(df)
    if spec.kind != "ratio":
        return None, None
    return _col_sum(df, spec.numerator_expr or spec.numerator), _col_sum(df, spec.denominator)


def metric_label(metric_key: str, resolver: Optional[Resolver] = None) -> str:
    spec = metric_spec(metric_key, resolver)
    return spec.label if spec else metric_key.replace("_", " ").title()


def metric_unit(metric_key: str, resolver: Optional[Resolver] = None) -> str:
    spec = metric_spec(metric_key, resolver)
    return spec.unit if spec else "count"


def higher_is_better(metric_key: str, resolver: Optional[Resolver] = None) -> bool:
    spec = metric_spec(metric_key, resolver)
    return spec.higher_is_better if spec else True


def metric_description(metric_key: str, resolver: Optional[Resolver] = None) -> str:
    spec = metric_spec(metric_key, resolver)
    return getattr(spec, "description", "") if spec else "Uploaded numeric column."


def pct_change(current: float, previous: float) -> float:
    if previous in (0, None) or (isinstance(previous, float) and (math.isnan(previous) or previous == 0)):
        return float("nan")
    return (current - previous) / abs(previous) * 100.0


def safe(x: Any) -> Any:
    """JSON-safe numbers (NaN/inf are not valid JSON)."""
    if isinstance(x, (np.floating, np.integer)):
        x = x.item()
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x
