"""
STAGE 1 — OBSERVE

Purpose: identify meaningful KPI changes and separate them from normal business
variation, then decompose the change into the business dimensions that drove it.

This stage is deliberately 100% deterministic (pandas / NumPy). No language model
is involved. Its output is the factual base every later stage reasons over.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import get_settings
from .drivers import (
    axis_weights,
    cell_delta_grid,
    rank_dimensions,
    rank_drivers,
    robust_sigma as _robust_sigma_impl,
    shapley_dimension_attribution,
)
from .metrics import (
    DatasetSchema,
    Resolver,
    compute,
    higher_is_better,
    metric_components,
    metric_label,
    metric_spec,
    metric_unit,
    pct_change,
    safe,
)

QUARTER_MONTHS = {1: (1, 3), 2: (4, 6), 3: (7, 9), 4: (10, 12)}


# ---------------------------------------------------------------------------
# timeframe helpers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Timeframe:
    year: int
    quarter: Optional[int] = None      # None => full year

    @property
    def label(self) -> str:
        return f"{self.year}-Q{self.quarter}" if self.quarter else f"FY{self.year}"

    @property
    def pretty(self) -> str:
        return f"Q{self.quarter} {self.year}" if self.quarter else f"Full year {self.year}"

    def previous(self) -> "Timeframe":
        if self.quarter is None:
            return Timeframe(self.year - 1, None)
        return Timeframe(self.year - 1, 4) if self.quarter == 1 else Timeframe(self.year, self.quarter - 1)

    def year_ago(self) -> "Timeframe":
        return Timeframe(self.year - 1, self.quarter)


def slice_period(df: pd.DataFrame, tf: Timeframe) -> pd.DataFrame:
    if tf.quarter is None:
        return df[df["_year"] == tf.year]
    return df[(df["_year"] == tf.year) & (df["_quarter"] == tf.quarter)]


def available_timeframes(df: pd.DataFrame) -> List[Dict[str, Any]]:
    out = []
    for (y, q), grp in df.groupby(["_year", "_quarter"]):
        out.append({
            "year": int(y),
            "quarter": int(q),
            "label": f"{int(y)}-Q{int(q)}",
            "rows": int(len(grp)),
            "start": str(grp["_date"].min().date()),
            "end": str(grp["_date"].max().date()),
        })
    return sorted(out, key=lambda r: (r["year"], r["quarter"]))


# ---------------------------------------------------------------------------
# series
# ---------------------------------------------------------------------------
def quarterly_series(df: pd.DataFrame, metric: str,
                     resolver: Optional[Resolver] = None) -> List[Dict[str, Any]]:
    rows = []
    for (y, q), grp in df.groupby(["_year", "_quarter"]):
        rows.append({
            "year": int(y),
            "quarter": int(q),
            "period": f"{int(y)}-Q{int(q)}",
            "value": safe(compute(grp, metric, resolver)),
        })
    return sorted(rows, key=lambda r: (r["year"], r["quarter"]))


def weekly_series(df: pd.DataFrame, metric: str, tf: Optional[Timeframe] = None,
                  lookback_quarters: int = 2,
                  resolver: Optional[Resolver] = None) -> List[Dict[str, Any]]:
    """Weekly series for the selected period plus a short run-up, for temporal checks."""
    frame = df
    if tf is not None:
        start_tf = tf
        for _ in range(lookback_quarters):
            start_tf = start_tf.previous()
        lo = slice_period(df, start_tf)["_date"].min()
        hi = slice_period(df, tf)["_date"].max()
        if pd.notna(lo) and pd.notna(hi):
            frame = df[(df["_date"] >= lo) & (df["_date"] <= hi)]
    rows = []
    for week, grp in frame.groupby("_week"):
        rows.append({"week": str(week), "value": safe(compute(grp, metric, resolver))})
    return sorted(rows, key=lambda r: r["week"])


# ---------------------------------------------------------------------------
# significance
# ---------------------------------------------------------------------------
def _robust_sigma(values: np.ndarray) -> Tuple[float, float]:
    """
    (median, robust sigma) using the median absolute deviation.

    Defined once in `drivers.robust_sigma` so the KPI's significance test and the
    per-member significance test cannot drift apart.
    """
    return _robust_sigma_impl(values)


def assess_significance(series: List[Dict[str, Any]], tf: Timeframe,
                        comparison: str = "previous_period") -> Dict[str, Any]:
    """
    Is the observed change meaningful, or is it inside this KPI's normal variation?

    Method: robust z-score of the current period-over-period change against the
    distribution of the same change across the KPI's own history, using the
    median and the median absolute deviation (resistant to the outliers we are
    trying to detect). When enough history exists, the comparison is restricted
    to the *same quarter transition* in previous years, which absorbs seasonality.
    """
    s = get_settings()
    ordered = [r for r in series if r["value"] is not None]
    by_period = {r["period"]: r for r in ordered}
    idx = {r["period"]: i for i, r in enumerate(ordered)}

    cur_key = tf.label
    base_tf = tf.year_ago() if comparison == "year_over_year" else tf.previous()
    base_key = base_tf.label

    changes: List[Dict[str, Any]] = []
    for i, row in enumerate(ordered):
        if comparison == "year_over_year":
            prev = by_period.get(f"{row['year'] - 1}-Q{row['quarter']}")
        else:
            prev = ordered[i - 1] if i > 0 else None
        if prev and prev["value"] not in (None, 0):
            changes.append({
                "period": row["period"],
                "quarter": row["quarter"],
                "change_pct": pct_change(row["value"], prev["value"]),
                "from": prev["period"],
            })

    current = next((c for c in changes if c["period"] == cur_key), None)
    history = [c for c in changes if idx.get(c["period"], 1e9) < idx.get(cur_key, -1)]
    hist_all = np.array([c["change_pct"] for c in history if c["change_pct"] == c["change_pct"]])
    same_q = np.array([c["change_pct"] for c in history if c["quarter"] == (tf.quarter or 0)
                       and c["change_pct"] == c["change_pct"]])

    history_status = "newly_launched" if current is None else (
        "sparse_history" if len(hist_all) < s.min_history_comparisons else "sufficient_history"
    )
    history_note = (
        "This KPI is newly launched for the selected period. Its current value is available, but there is no prior "
        "period for historical trend analysis."
        if history_status == "newly_launched" else
        (f"Only {len(hist_all)} historical comparison period(s) are available; at least "
         f"{s.min_history_comparisons} are required for trend and normal-variation analysis."
         if history_status == "sparse_history" else None)
    )

    med_all, sig_all = _robust_sigma(hist_all)
    med_seasonal, sig_seasonal = _robust_sigma(same_q)

    seasonal_ok = len(same_q) >= 3 and sig_seasonal == sig_seasonal and sig_seasonal > 0
    method = "seasonal_robust_z" if seasonal_ok else "robust_z"
    med, sigma = (med_seasonal, sig_seasonal) if seasonal_ok else (med_all, sig_all)

    # Small samples make the dispersion estimate optimistic, which would inflate
    # every z-score into a spurious "extreme anomaly". Two corrections:
    #   1. widen sigma for sample size (a t-like finite-sample inflation);
    #   2. floor the seasonal sigma so it can never claim the KPI is far more
    #      stable than its own full history says it is.
    n_used = len(same_q) if seasonal_ok else len(hist_all)
    floor_note = None
    if sigma == sigma and sigma > 0:
        sigma *= (1.0 + 1.0 / max(n_used, 1)) ** 0.5
        floor = max(0.75, 0.5 * sig_all) if (sig_all == sig_all and sig_all > 0) else 0.75
        if seasonal_ok and sigma < floor:
            sigma = floor
            floor_note = (
                "The same-quarter sample was too small and too tight to estimate dispersion reliably, "
                "so the estimate was floored at half the KPI's all-history dispersion. Without this floor "
                "a very small sample produces an implausibly extreme z-score."
            )

    change_pct = current["change_pct"] if current else float("nan")
    z = (change_pct - med) / sigma if (sigma and sigma == sigma and sigma > 0 and change_pct == change_pct) else float("nan")

    baseline_value = by_period.get(base_key, {}).get("value")
    expected_value = baseline_value * (1 + med / 100.0) if (baseline_value and med == med) else None
    band = None
    if baseline_value and med == med and sigma == sigma:
        band = [
            baseline_value * (1 + (med - s.anomaly_z_threshold * sigma) / 100.0),
            baseline_value * (1 + (med + s.anomaly_z_threshold * sigma) / 100.0),
        ]

    material = abs(change_pct) >= s.min_material_change_pct if change_pct == change_pct else False
    statistically_unusual = abs(z) >= s.anomaly_z_threshold if z == z else False
    is_anomaly = bool(material and statistically_unusual)

    if history_status == "newly_launched":
        power = history_note
    elif history_status == "sparse_history":
        power = history_note
    elif len(hist_all) < 4:
        power = ("Weak: fewer than 4 historical comparison periods, so 'normal variation' is "
                 "estimated from very little data. Treat the verdict as indicative.")
    elif not seasonal_ok and (tf.quarter is not None):
        power = ("Moderate: not enough same-quarter history to model seasonality, so the "
                 "comparison uses all historical period-over-period changes.")
    else:
        power = "Good: the comparison is restricted to the same quarter transition in previous years."

    if history_status != "sufficient_history":
        verdict = history_status
        is_anomaly = material = statistically_unusual = False
        expected_value = band = None
        z = med = sigma = float("nan")
        method = "insufficient_history"
    elif is_anomaly:
        verdict = "meaningful_signal"
    elif material and not statistically_unusual:
        verdict = "within_normal_variation"
    elif statistically_unusual and not material:
        verdict = "statistically_unusual_but_immaterial"
    else:
        verdict = "within_normal_variation"

    return {
        "method": method,
        "history_status": history_status,
        "history_note": history_note,
        "comparison": comparison,
        "current_period": cur_key,
        "baseline_period": base_key,
        "change_pct": safe(change_pct),
        "robust_z": safe(z),
        "median_historical_change_pct": safe(med),
        "robust_sigma_pct": safe(sigma),
        "median_all_history_pct": safe(med_all),
        "sigma_all_history_pct": safe(sig_all),
        "history_points": int(len(hist_all)),
        "same_quarter_points": int(len(same_q)),
        "z_threshold": s.anomaly_z_threshold,
        "material_threshold_pct": s.min_material_change_pct,
        "is_material": bool(material),
        "is_statistically_unusual": bool(statistically_unusual),
        "is_anomaly": is_anomaly,
        "verdict": verdict,
        "expected_value": safe(expected_value),
        "normal_range": [safe(band[0]), safe(band[1])] if band else None,
        "historical_changes": [
            {"period": c["period"], "from": c["from"], "change_pct": safe(c["change_pct"])}
            for c in history
        ],
        "statistical_power": power,
        "dispersion_note": floor_note,
    }


# ---------------------------------------------------------------------------
# driver decomposition
# ---------------------------------------------------------------------------
def decompose_dimension(cur: pd.DataFrame, base: pd.DataFrame, dimension: str,
                        metric: str, max_items: int,
                        resolver: Optional[Resolver] = None) -> List[Dict[str, Any]]:
    """
    Contribution of every member of a dimension to the KPI's change.

    * additive KPIs (sums): contribution = member delta / total delta
    * ratio KPIs: a rate/mix decomposition —
        ΔR = Σ w_i,cur·(r_i,cur − r_i,base)   [rate effect]
           + Σ r_i,base·(w_i,cur − w_i,base)  [mix effect]
      so a KPI can move because members got worse, or because the *mix* shifted
      toward weaker members. Those are different business problems.
    """
    spec = metric_spec(metric, resolver)
    members = sorted(set(cur[dimension].dropna().unique()) | set(base[dimension].dropna().unique()))
    rows: List[Dict[str, Any]] = []

    if spec is None or spec.kind in ("sum", "mean"):
        total_cur, total_base = compute(cur, metric, resolver), compute(base, metric, resolver)
        total_delta = total_cur - total_base
        for m in members:
            c, b = cur[cur[dimension] == m], base[base[dimension] == m]
            cv = compute(c, metric, resolver) if len(c) else 0.0
            bv = compute(b, metric, resolver) if len(b) else 0.0
            cv = 0.0 if cv != cv else cv
            bv = 0.0 if bv != bv else bv
            delta = cv - bv
            share_base = (bv / total_base * 100.0) if total_base else None
            contribution = (delta / total_delta * 100.0) if total_delta else None
            rows.append({
                "name": str(m),
                "current": safe(cv),
                "baseline": safe(bv),
                "change_abs": safe(delta),
                "change_pct": safe(pct_change(cv, bv)),
                "contribution_pct": safe(contribution),
                "share_of_current_pct": safe(cv / total_cur * 100.0) if total_cur else None,
                "share_of_baseline_pct": safe(share_base),
                # >1 means this member moved the KPI more than its size alone would imply.
                # A member contributing exactly in line with its size is not a "driver".
                "over_index": safe(contribution / share_base) if (contribution is not None and share_base) else None,
                "effects": None,
            })
    else:
        num_c, den_c = metric_components(cur, metric, resolver)
        num_b, den_b = metric_components(base, metric, resolver)
        R_c = (num_c / den_c) if den_c else float("nan")
        R_b = (num_b / den_b) if den_b else float("nan")
        total_delta = (R_c - R_b) * spec.scale
        for m in members:
            c, b = cur[cur[dimension] == m], base[base[dimension] == m]
            n_c, d_c = metric_components(c, metric, resolver) if len(c) else (0.0, 0.0)
            n_b, d_b = metric_components(b, metric, resolver) if len(b) else (0.0, 0.0)
            r_c = (n_c / d_c) if d_c else 0.0
            r_b = (n_b / d_b) if d_b else 0.0
            w_c = (d_c / den_c) if den_c else 0.0
            w_b = (d_b / den_b) if den_b else 0.0
            rate_effect = w_c * (r_c - r_b) * spec.scale
            mix_effect = r_b * (w_c - w_b) * spec.scale
            delta = rate_effect + mix_effect
            contribution = (delta / total_delta * 100.0) if total_delta else None
            share_base = w_b * 100.0
            rows.append({
                "name": str(m),
                "current": safe(r_c * spec.scale),
                "baseline": safe(r_b * spec.scale),
                "change_abs": safe((r_c - r_b) * spec.scale),
                "change_pct": safe(pct_change(r_c, r_b)),
                "contribution_pct": safe(contribution),
                "share_of_current_pct": safe(w_c * 100.0),
                "share_of_baseline_pct": safe(share_base),
                "over_index": safe(contribution / share_base) if (contribution is not None and share_base) else None,
                "effects": {"rate_effect": safe(rate_effect), "mix_effect": safe(mix_effect)},
            })

    rows.sort(key=lambda r: abs(r["change_abs"] or 0), reverse=True)
    # `rows[:None]` and `rows[None:]` both return every row, so `max_items=None`
    # used to duplicate the whole member list into a same-named "Other" row on
    # top of the members it was meant to summarise. Callers that want every
    # member with no rollup pass `max_items=None` explicitly for that reason.
    head, tail = (rows, []) if max_items is None else (rows[:max_items], rows[max_items:])
    if tail:
        head.append({
            "name": f"Other ({len(tail)} members)",
            "current": safe(sum(r["current"] or 0 for r in tail)),
            "baseline": safe(sum(r["baseline"] or 0 for r in tail)),
            "change_abs": safe(sum(r["change_abs"] or 0 for r in tail)),
            "change_pct": None,
            "contribution_pct": safe(sum(r["contribution_pct"] or 0 for r in tail)),
            "share_of_current_pct": None,
            "share_of_baseline_pct": None,
            "over_index": None,
            "effects": None,
            "is_aggregate": True,
        })
    return head


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
def observe(df: pd.DataFrame, schema: DatasetSchema, metric: str, tf: Timeframe,
            comparison: str = "previous_period") -> Dict[str, Any]:
    s = get_settings()
    resolver = schema.contract_resolver
    cur = slice_period(df, tf)
    base_tf = tf.year_ago() if comparison == "year_over_year" else tf.previous()
    base = slice_period(df, base_tf)

    if len(cur) == 0:
        raise ValueError(f"No rows found for {tf.pretty} in this dataset.")

    cur_val = compute(cur, metric, resolver)
    base_val = compute(base, metric, resolver) if len(base) else float("nan")
    change_abs = cur_val - base_val if base_val == base_val else float("nan")
    change_pct = pct_change(cur_val, base_val)

    series = quarterly_series(df, metric, resolver)
    significance = assess_significance(series, tf, comparison)

    drivers: Dict[str, List[Dict[str, Any]]] = {}
    if len(base):
        for dim in schema.dimensions:
            if dim in df.columns:
                drivers[dim] = decompose_dimension(cur, base, dim, metric,
                                                  s.max_drivers_per_dimension, resolver)

    unfavourable = (change_abs or 0) < 0 if higher_is_better(metric, resolver) else (change_abs or 0) > 0

    # Which dimension the movement is actually shaped by. Exact Shapley over the
    # cell grid, so the interaction between two dimensions is split between them
    # rather than counted once for each -- which is what reading the per-dimension
    # decompositions side by side silently does. Computed BEFORE ranking, because
    # it decides how much each dimension's members are allowed to count.
    cells = cell_delta_grid(cur, base, schema.dimensions, metric, resolver) if len(base) else []
    dimension_shapley = shapley_dimension_attribution(cells, [
        d for d in schema.dimensions if d in cur.columns and d in base.columns
    ]) if cells else {}
    dimension_ranking = rank_dimensions(dimension_shapley) if dimension_shapley else []

    # Ranking a member on contribution alone makes the biggest segment the
    # "driver" of everything; ranking on over-index alone makes the noisiest
    # small one the driver. `rank_drivers` scores four things together --
    # contribution, over-index, the member's significance against its OWN
    # history, and whether it stayed moved -- and publishes the arithmetic on
    # every row. See docs/RANKING.md.
    top = rank_drivers(drivers, df, metric, tf, resolver, limit=6,
                       axis=axis_weights(dimension_shapley))

    concentration = None
    if drivers:
        first_dim = next(iter(drivers))
        contribs = [abs(r.get("contribution_pct") or 0) for r in drivers[first_dim] if not r.get("is_aggregate")]
        if contribs:
            concentration = safe(max(contribs))

    # a compact scoreboard of every other KPI over the same two periods,
    # so later stages have supporting/contradicting numbers available
    scoreboard = []
    for key in schema.available_kpis:
        c, b = compute(cur, key, resolver), compute(base, key, resolver) if len(base) else float("nan")
        spec = metric_spec(key, resolver)
        scoreboard.append({
            "key": key,
            "label": metric_label(key, resolver),
            "unit": metric_unit(key, resolver),
            "higher_is_better": higher_is_better(key, resolver),
            "granularity": getattr(spec, "granularity_label", ""),
            "kpi_status": getattr(spec, "status", ""),
            "current": safe(c),
            "baseline": safe(b),
            "change_pct": safe(pct_change(c, b)),
            "is_primary": key == metric,
        })

    return {
        "kpi": metric,
        "kpi_label": metric_label(metric, resolver),
        "unit": metric_unit(metric, resolver),
        "higher_is_better": higher_is_better(metric, resolver),
        "granularity": getattr(metric_spec(metric, resolver), "granularity_label", ""),
        "contract_status": schema.contract_status,
        "timeframe": {"year": tf.year, "quarter": tf.quarter, "label": tf.label, "pretty": tf.pretty},
        "baseline_timeframe": {"year": base_tf.year, "quarter": base_tf.quarter,
                               "label": base_tf.label, "pretty": base_tf.pretty},
        "comparison": comparison,
        "current_value": safe(cur_val),
        "baseline_value": safe(base_val),
        "change_abs": safe(change_abs),
        "change_pct": safe(change_pct),
        "direction": "up" if (change_abs or 0) > 0 else ("down" if (change_abs or 0) < 0 else "flat"),
        "is_unfavourable": bool(unfavourable),
        "anomaly": significance["is_anomaly"],
        "verdict": significance["verdict"],
        "history_status": significance["history_status"],
        "history_note": significance["history_note"],
        "significance": significance,
        "drivers": drivers,
        "top_drivers": top,
        "dimension_shapley": dimension_shapley,
        "dimension_ranking": dimension_ranking,
        "driver_concentration_pct": concentration,
        "series": {
            "quarterly": series,
            "weekly": weekly_series(df, metric, tf, lookback_quarters=2, resolver=resolver),
        },
        "kpi_scoreboard": scoreboard,
        "rows_analysed": int(len(cur)),
        "baseline_rows": int(len(base)),
        "data_warnings": schema.warnings,
    }
