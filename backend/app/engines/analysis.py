"""
Shared deterministic analysis primitives used by Investigate and Contest.

Every function here answers a question with arithmetic on the user's rows.
None of them call a language model.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .metrics import Resolver, compute, pct_change, safe


# ---------------------------------------------------------------------------
# temporal onset detection
# ---------------------------------------------------------------------------
def weekly_frame(df: pd.DataFrame, metric: str, resolver: Optional[Resolver] = None) -> pd.DataFrame:
    """
    A weekly series for one KPI.

    `resolver` must be the caller's compiled KPI Contract when the dataset has
    one. Without it, a contract-only KPI (a ratio the seed registry has never
    heard of) is invisible to `compute` and every week comes back NaN — silently,
    since `compute` treats an unknown key as an optional raw column and simply
    finds nothing to sum.
    """
    rows = []
    for week, grp in df.groupby("_week"):
        rows.append({"week": str(week), "value": compute(grp, metric, resolver)})
    out = pd.DataFrame(rows).sort_values("week").reset_index(drop=True)
    return out


def detect_onset(weeks: pd.DataFrame, baseline_weeks: int = 8, direction: str = "down",
                 persistence: int = 2, k_sigma: float = 1.0) -> Optional[Dict[str, Any]]:
    """
    First week at which a weekly series leaves its own baseline band and *stays*
    outside it for `persistence` consecutive weeks.

    Used to answer the question that separates correlation from causation in a
    business timeline: did the proposed cause move BEFORE the KPI moved, or after?
    """
    if weeks is None or len(weeks) < baseline_weeks + persistence + 1:
        return None
    values = weeks["value"].astype(float).values
    base = values[:baseline_weeks]
    base = base[~np.isnan(base)]
    if len(base) < 3:
        return None
    med = float(np.median(base))
    mad = float(np.median(np.abs(base - med)))
    sigma = 1.4826 * mad or float(np.std(base, ddof=1)) or abs(med) * 0.05
    if sigma <= 0:
        return None
    lo, hi = med - k_sigma * sigma, med + k_sigma * sigma

    def breached(v: float) -> bool:
        if np.isnan(v):
            return False
        return v < lo if direction == "down" else v > hi

    for i in range(baseline_weeks, len(values) - persistence + 1):
        if all(breached(values[j]) for j in range(i, i + persistence)):
            return {
                "week": str(weeks.iloc[i]["week"]),
                "index": int(i),
                "baseline_median": safe(med),
                "baseline_sigma": safe(sigma),
                "threshold": safe(lo if direction == "down" else hi),
                "value_at_onset": safe(float(values[i])),
                "direction": direction,
                "persistence_weeks": persistence,
                "baseline_weeks": baseline_weeks,
            }
    return None


def compare_onsets(kpi_onset: Optional[Dict[str, Any]],
                   cause_onset: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Temporal-precedence verdict between a KPI move and a proposed cause."""
    if not kpi_onset or not cause_onset:
        return {
            "status": "undetermined",
            "detail": "Not enough weekly history to date the onset of both series.",
            "lag_weeks": None,
        }
    lag = cause_onset["index"] - kpi_onset["index"]
    if lag <= -1:
        status, detail = "cause_precedes_kpi", (
            f"The proposed cause moved {abs(lag)} week(s) BEFORE the KPI did "
            f"({cause_onset['week']} vs {kpi_onset['week']}). Temporal ordering is consistent with causation."
        )
    elif lag == 0:
        status, detail = "simultaneous", (
            f"Both series turned in the same week ({kpi_onset['week']}). Ordering neither supports nor "
            "contradicts causation."
        )
    else:
        status, detail = "kpi_precedes_cause", (
            f"The KPI began moving in {kpi_onset['week']}, {lag} week(s) BEFORE the proposed cause moved "
            f"({cause_onset['week']}). A cause cannot explain a change that started before it. This "
            "hypothesis can at most be a contributing factor to the later part of the change."
        )
    return {"status": status, "detail": detail, "lag_weeks": int(lag),
            "kpi_onset_week": kpi_onset["week"], "cause_onset_week": cause_onset["week"]}


# ---------------------------------------------------------------------------
# cross-sectional association
# ---------------------------------------------------------------------------
def member_change_table(cur: pd.DataFrame, base: pd.DataFrame, dimension: str,
                        metrics: List[str], resolver: Optional[Resolver] = None) -> pd.DataFrame:
    """
    Per-member current/baseline/change for a set of KPIs, the input to
    cross-sectional consistency checks and correlation.

    `resolver` carries the same requirement as `weekly_frame`: omit it for a
    contract-only KPI and every member's value is NaN, which then makes
    `correlate` report n=0 rather than a real association.
    """
    members = sorted(set(cur[dimension].dropna().unique()) | set(base[dimension].dropna().unique()))
    rows = []
    for m in members:
        c, b = cur[cur[dimension] == m], base[base[dimension] == m]
        row: Dict[str, Any] = {"member": str(m)}
        for metric in metrics:
            cv = compute(c, metric, resolver) if len(c) else float("nan")
            bv = compute(b, metric, resolver) if len(b) else float("nan")
            row[f"{metric}__cur"] = cv
            row[f"{metric}__base"] = bv
            row[f"{metric}__chg"] = pct_change(cv, bv)
        rows.append(row)
    return pd.DataFrame(rows)


def correlate(table: pd.DataFrame, metric_a: str, metric_b: str) -> Dict[str, Any]:
    """Pearson r between two metrics' per-member % changes. Association only."""
    a = table[f"{metric_a}__chg"].astype(float)
    b = table[f"{metric_b}__chg"].astype(float)
    mask = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = int(len(a))
    if n < 3 or a.std() == 0 or b.std() == 0:
        return {"r": None, "r_squared": None, "n": n,
                "interpretation": "Too few comparable members to measure association."}
    r = float(np.corrcoef(a, b)[0, 1])
    strength = ("strong" if abs(r) >= 0.8 else "moderate" if abs(r) >= 0.5 else "weak")
    return {
        "r": round(r, 3),
        "r_squared": round(r * r, 3),
        "n": n,
        "interpretation": (
            f"Across {n} members, the two changes are {strength}ly "
            f"{'positively' if r > 0 else 'negatively'} associated (r={r:.2f}). "
            "Association across members is evidence of a shared pattern, not proof of causation."
        ),
    }


def counterexamples(table: pd.DataFrame, kpi: str, cause_metric: str,
                    kpi_drop_pct: float = -5.0, cause_move_pct: float = 5.0,
                    cause_direction: str = "up") -> List[Dict[str, Any]]:
    """
    Members where the KPI moved badly but the proposed cause did NOT move.
    These are the cleanest contradictions available from structured data.
    """
    out = []
    for _, row in table.iterrows():
        kpi_chg = row.get(f"{kpi}__chg")
        cause_chg = row.get(f"{cause_metric}__chg")
        if kpi_chg is None or cause_chg is None:
            continue
        if not (np.isfinite(kpi_chg) and np.isfinite(cause_chg)):
            continue
        if kpi_chg <= kpi_drop_pct:
            moved = cause_chg >= cause_move_pct if cause_direction == "up" else cause_chg <= -cause_move_pct
            if not moved:
                out.append({
                    "member": row["member"],
                    "kpi_change_pct": safe(float(kpi_chg)),
                    "cause_change_pct": safe(float(cause_chg)),
                })
    return out


# ---------------------------------------------------------------------------
# price / volume decomposition
# ---------------------------------------------------------------------------
def price_volume_decomposition(cur: pd.DataFrame, base: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Δrevenue split into a volume effect and a realised-price effect."""
    for col in ("revenue", "units_sold"):
        if col not in cur.columns or col not in base.columns:
            return None
    rev_c, rev_b = float(cur["revenue"].sum()), float(base["revenue"].sum())
    q_c, q_b = float(cur["units_sold"].sum()), float(base["units_sold"].sum())
    if q_c <= 0 or q_b <= 0:
        return None
    p_c, p_b = rev_c / q_c, rev_b / q_b
    volume_effect = p_b * (q_c - q_b)
    price_effect = q_c * (p_c - p_b)
    total = rev_c - rev_b
    return {
        "total_change": safe(total),
        "volume_effect": safe(volume_effect),
        "price_effect": safe(price_effect),
        "volume_share_pct": safe(volume_effect / total * 100) if total else None,
        "price_share_pct": safe(price_effect / total * 100) if total else None,
        "avg_price_current": safe(p_c),
        "avg_price_baseline": safe(p_b),
        "avg_price_change_pct": safe(pct_change(p_c, p_b)),
        "units_current": safe(q_c),
        "units_baseline": safe(q_b),
        "units_change_pct": safe(pct_change(q_c, q_b)),
        "narrative": (
            "Revenue moved mainly on VOLUME (units), not realised price."
            if total and abs(volume_effect) > abs(price_effect)
            else "Revenue moved mainly on realised PRICE per unit, not volume."
        ),
    }


# ---------------------------------------------------------------------------
# data-coverage checks
# ---------------------------------------------------------------------------
def coverage_report(cur: pd.DataFrame, base: pd.DataFrame, dimensions: List[str]) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "rows_current": int(len(cur)),
        "rows_baseline": int(len(base)),
        "row_change_pct": safe(pct_change(len(cur), len(base))),
        "days_current": int(cur["_date"].nunique()) if len(cur) else 0,
        "days_baseline": int(base["_date"].nunique()) if len(base) else 0,
        "dimension_membership": {},
        "issues": [],
    }
    for dim in dimensions:
        if dim not in cur.columns:
            continue
        c = set(cur[dim].dropna().unique())
        b = set(base[dim].dropna().unique())
        disappeared, appeared = sorted(b - c), sorted(c - b)
        report["dimension_membership"][dim] = {
            "current_members": len(c),
            "baseline_members": len(b),
            "disappeared": disappeared[:10],
            "appeared": appeared[:10],
        }
        if disappeared:
            report["issues"].append(
                f"{len(disappeared)} '{dim}' member(s) present in the baseline are absent in the current "
                f"period ({', '.join(map(str, disappeared[:3]))}). Part of the KPI change may be missing "
                "data rather than business performance."
            )
    if report["days_baseline"] and report["days_current"]:
        ratio = report["days_current"] / report["days_baseline"]
        if ratio < 0.9 or ratio > 1.1:
            report["issues"].append(
                f"The current period covers {report['days_current']} distinct dates versus "
                f"{report['days_baseline']} in the baseline — an unequal comparison window "
                "that will bias any total-based comparison."
            )
    return report


# ---------------------------------------------------------------------------
# lead / lag and reverse-causation screening
# ---------------------------------------------------------------------------
def lead_lag(kpi_weeks: pd.DataFrame, cause_weeks: pd.DataFrame,
             max_lag: int = 5) -> Optional[Dict[str, Any]]:
    """
    Cross-correlate the WEEK-ON-WEEK CHANGES of a KPI and a proposed cause at
    several lags, and report which lag fits best.

    Levels are not used: two series that are both trending will correlate almost
    perfectly at every lag, which is exactly the spurious result this check exists
    to avoid. Differencing removes the shared trend and leaves the timing question.

    lag > 0  the cause moves BEFORE the KPI  (ordering consistent with causation)
    lag = 0  they move together              (a shared driver, or a mechanical link)
    lag < 0  the KPI moves BEFORE the cause  (ordering points the other way)
    """
    if kpi_weeks is None or cause_weeks is None:
        return None
    merged = kpi_weeks.merge(cause_weeks, on="week", suffixes=("_kpi", "_cause"))
    if len(merged) < 12:
        return None
    a = merged["value_kpi"].astype(float).diff()
    b = merged["value_cause"].astype(float).diff()

    results = []
    for lag in range(-max_lag, max_lag + 1):
        x = b.shift(lag)                       # positive lag => cause moved earlier
        mask = a.notna() & x.notna() & np.isfinite(a) & np.isfinite(x)
        if mask.sum() < 8:
            continue
        av, xv = a[mask], x[mask]
        if av.std() == 0 or xv.std() == 0:
            continue
        r = float(np.corrcoef(av, xv)[0, 1])
        results.append({"lag_weeks": lag, "r": round(r, 3), "n": int(mask.sum())})
    if not results:
        return None

    best = max(results, key=lambda d: abs(d["r"]))
    at_zero = next((d["r"] for d in results if d["lag_weeks"] == 0), None)
    if best["lag_weeks"] > 0:
        verdict = "cause_leads"
        detail = (f"The proposed cause fits best when it moves {best['lag_weeks']} week(s) BEFORE the KPI "
                  f"(r={best['r']}). The timing is consistent with a causal direction.")
    elif best["lag_weeks"] == 0:
        verdict = "simultaneous"
        detail = (f"The two series move in the same week (r={best['r']}). That is equally consistent with a "
                  "shared external driver, or with the two quantities being mechanically linked.")
    else:
        verdict = "kpi_leads"
        detail = (f"The KPI's weekly movements fit best {abs(best['lag_weeks'])} week(s) BEFORE the proposed "
                  f"cause (r={best['r']}). The timing points the other way: the 'cause' may be responding to "
                  "the KPI rather than driving it.")
    return {"verdict": verdict, "best": best, "r_at_lag_0": at_zero, "profile": results, "detail": detail}
