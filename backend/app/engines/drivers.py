"""
Driver identification and ranking.

`observe.decompose_dimension` answers "how much of the change sits in each member
of one dimension". That is an exact marginal decomposition and it is correct, but
it is computed independently per dimension, so the answers do not compose: North
(region) and Product A (product) can each be told they carry 45% of a decline when
what actually moved is the single cell where they intersect.

This module adds the part decomposition alone cannot give: which dimension the
story is actually on, with interactions split rather than double-counted.

Everything here is deterministic arithmetic over the user's own rows. No language
model is involved, and nothing here imports pandas -- the attribution works on a
plain list of cells so it can be verified against hand-computable examples.
"""
from __future__ import annotations

from math import factorial
from typing import Any, Dict, Hashable, List, Sequence, Tuple

# Crossing more than this many dimensions means 2**n subsets and a cell grid that
# is mostly empty anyway. Above the limit the caller falls back to ranking each
# dimension independently rather than hanging.
MAX_SHAPLEY_DIMENSIONS = 8


# ---------------------------------------------------------------------------
# Shapley attribution over dimensions
# ---------------------------------------------------------------------------
#
# Setup.  Cross every dimension into a cell grid. For cell c take the KPI delta
#
#     d_c = value_current(c) - value_baseline(c)
#
# with N cells and mean dbar = (sum_c d_c) / N.
#
# Value function.  A subset S of the dimensions partitions the grid by the
# S-coordinates. Predict each cell by its group mean and take the between-group
# (explained) sum of squares:
#
#     V(S) = sum_c ( mean{ d_c' : c' in group_S(c) } - dbar )**2
#
# V(empty) = 0, because with no dimensions there is one group whose mean is dbar.
# V(D) = sum_c (d_c - dbar)**2, the total sum of squares, because each cell is
# then its own group. Adding a dimension refines the partition, and by the ANOVA
# decomposition a refinement never decreases the between-group sum of squares, so
# V is monotone.
#
# Shapley value of dimension d, over subsets S of D that exclude d:
#
#     phi_d = sum_S  ( |S|! * (|D|-|S|-1)! / |D|! ) * [ V(S + {d}) - V(S) ]
#
# Efficiency gives sum_d phi_d = V(D) - V(empty) = V(D): the shares are
# exhaustive, so no variation is left unattributed and none is counted twice.
# Monotonicity gives phi_d >= 0. Both are asserted in tests/test_shapley.py.


def explained_sum_of_squares(coords: Sequence[Tuple[Hashable, ...]],
                             deltas: Sequence[float],
                             subset: Tuple[int, ...],
                             mean: float) -> float:
    """V(S) -- between-group sum of squares when cells are grouped by `subset`."""
    if not subset:
        return 0.0                  # one group, whose mean is `mean` by construction

    totals: Dict[Tuple[Hashable, ...], float] = {}
    counts: Dict[Tuple[Hashable, ...], int] = {}
    for coord, delta in zip(coords, deltas):
        key = tuple(coord[i] for i in subset)
        totals[key] = totals.get(key, 0.0) + delta
        counts[key] = counts.get(key, 0) + 1

    ess = 0.0
    for key, total in totals.items():
        n = counts[key]
        deviation = (total / n) - mean
        ess += n * deviation * deviation
    return ess


def shapley_dimension_attribution(cells: Sequence[Dict[str, Any]],
                                  dimensions: Sequence[str],
                                  delta_key: str = "delta") -> Dict[str, Any]:
    """
    How much of the variation in cell-level deltas each dimension explains.

    `cells` is a plain sequence of dicts: the dimension coordinates plus a numeric
    delta under `delta_key`. Deliberately not a DataFrame -- the whole point is
    that this is checkable against a 2x2 grid worked out by hand.

    Returns phi per dimension, the share of V(D) each represents, and the pieces
    needed to audit the result. `shares` is None when V(D) is zero: a grid with no
    variation has nothing to attribute, and 0/0 is not a share.
    """
    dims = list(dimensions)
    if not dims:
        return {"phi": {}, "shares": {}, "total_variation": 0.0, "n_cells": 0,
                "method": "shapley_ess", "skipped": "no dimensions"}
    if len(dims) > MAX_SHAPLEY_DIMENSIONS:
        return {"phi": {}, "shares": {}, "total_variation": None, "n_cells": len(cells),
                "method": "shapley_ess",
                "skipped": (f"{len(dims)} dimensions exceeds the limit of "
                            f"{MAX_SHAPLEY_DIMENSIONS}; ranked per dimension instead.")}

    coords: List[Tuple[Hashable, ...]] = []
    deltas: List[float] = []
    for cell in cells:
        value = cell.get(delta_key)
        if value is None or value != value:              # skip NaN
            continue
        coords.append(tuple(cell.get(d) for d in dims))
        deltas.append(float(value))

    n = len(deltas)
    if n == 0:
        return {"phi": {d: 0.0 for d in dims}, "shares": {d: None for d in dims},
                "total_variation": 0.0, "n_cells": 0, "method": "shapley_ess",
                "skipped": "no cells with a usable delta"}

    mean = sum(deltas) / n

    # V(S) for every subset, indexed by the sorted tuple of dimension positions.
    n_dims = len(dims)
    values: Dict[Tuple[int, ...], float] = {}
    for mask in range(1 << n_dims):
        subset = tuple(i for i in range(n_dims) if mask & (1 << i))
        values[subset] = explained_sum_of_squares(coords, deltas, subset, mean)

    total_variation = values[tuple(range(n_dims))]

    phi: Dict[str, float] = {}
    for i, dim in enumerate(dims):
        others = [j for j in range(n_dims) if j != i]
        acc = 0.0
        for mask in range(1 << len(others)):
            subset = tuple(sorted(others[k] for k in range(len(others)) if mask & (1 << k)))
            s = len(subset)
            weight = factorial(s) * factorial(n_dims - s - 1) / factorial(n_dims)
            with_d = tuple(sorted(subset + (i,)))
            acc += weight * (values[with_d] - values[subset])
        phi[dim] = acc

    shares = ({d: (phi[d] / total_variation) for d in dims}
              if total_variation > 0 else {d: None for d in dims})

    return {
        "phi": phi,
        "shares": shares,
        "total_variation": total_variation,
        "n_cells": n,
        "mean_delta": mean,
        "method": "shapley_ess",
        "note": ("Share of the variation in cell-level deltas attributable to each dimension, "
                 "computed as an exact Shapley value over the between-group sum of squares. "
                 "Shares sum to 1 by the efficiency axiom, so interaction between dimensions "
                 "is split between them rather than double-counted."),
    }


def rank_dimensions(attribution: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The dimensions ordered by how much of the movement's shape they explain."""
    phi = attribution.get("phi") or {}
    shares = attribution.get("shares") or {}
    rows = [{"dimension": d, "phi": phi[d], "share": shares.get(d)} for d in phi]
    rows.sort(key=lambda r: r["phi"], reverse=True)
    for i, row in enumerate(rows):
        row["rank"] = i + 1
    return rows


# ---------------------------------------------------------------------------
# robust dispersion -- one definition, shared with observe.assess_significance
# ---------------------------------------------------------------------------
def robust_sigma(values) -> Tuple[float, float]:
    """
    (median, robust sigma) using the median absolute deviation.

    MAD rather than the standard deviation because the thing being measured is
    how unusual an outlier is, and the standard deviation is inflated by the very
    outlier under test. 1.4826 rescales MAD to be comparable with sigma for
    normally distributed data. Falls back to the standard deviation only when the
    MAD is degenerate (more than half the history identical).
    """
    import numpy as np

    arr = np.asarray([v for v in values if v == v], dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan")
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    sigma = 1.4826 * mad
    if sigma <= 1e-9:
        sigma = float(np.std(arr, ddof=1)) if len(arr) > 1 else float("nan")
    return med, sigma


# ---------------------------------------------------------------------------
# the cell grid the Shapley attribution runs over
# ---------------------------------------------------------------------------
def cell_delta_grid(cur, base, dimensions: Sequence[str], metric: str,
                    resolver=None) -> List[Dict[str, Any]]:
    """
    One row per cell of the full dimension cross-product, with its KPI delta.

    Cells present in only one of the two periods still appear, with the missing
    side treated as zero -- a member that disappeared is part of the change, not
    an absence of evidence.
    """
    from .metrics import compute

    dims = [d for d in dimensions if d in cur.columns and d in base.columns]
    if not dims:
        return []

    def totals(frame):
        out = {}
        if not len(frame):
            return out
        for key, grp in frame.groupby(dims, dropna=False, observed=True):
            key = key if isinstance(key, tuple) else (key,)
            value = compute(grp, metric, resolver)
            out[key] = 0.0 if value != value else float(value)
        return out

    cur_totals, base_totals = totals(cur), totals(base)
    cells = []
    for key in sorted(set(cur_totals) | set(base_totals), key=lambda k: tuple(str(x) for x in k)):
        cells.append({
            **{d: key[i] for i, d in enumerate(dims)},
            "delta": cur_totals.get(key, 0.0) - base_totals.get(key, 0.0),
            "current": cur_totals.get(key, 0.0),
            "baseline": base_totals.get(key, 0.0),
        })
    return cells


# ---------------------------------------------------------------------------
# per-member statistical support
# ---------------------------------------------------------------------------
def member_significance(df, dimension: str, member: str, metric: str,
                        tf, resolver=None) -> Dict[str, Any]:
    """
    Is this member's own movement unusual *for this member*?

    The over-index test asks whether a member moved more than its size implies.
    It cannot tell a genuinely disrupted segment from a small, noisy one: a member
    that swings wildly every quarter will clear over-index >= 1.2 regularly and
    mean nothing by it. So each member is also tested against its OWN history,
    with the same robust-z machinery `assess_significance` applies to the KPI:
    median and MAD of the member's historical period-over-period changes, inflated
    for small samples.

    Returns z = None when there is too little history to say -- which is reported
    as "not enough history", never silently scored as zero.
    """
    from .metrics import compute, pct_change

    if dimension not in df.columns:
        return {"robust_z": None, "history_points": 0,
                "note": f"'{dimension}' is not a column in this dataset."}

    scoped = df[df[dimension] == member]
    if not len(scoped):
        return {"robust_z": None, "history_points": 0,
                "note": f"No rows for {dimension}={member}."}

    series = []
    for (y, q), grp in scoped.groupby(["_year", "_quarter"], observed=True):
        value = compute(grp, metric, resolver)
        series.append({"year": int(y), "quarter": int(q),
                       "period": f"{int(y)}-Q{int(q)}",
                       "value": None if value != value else float(value)})
    series.sort(key=lambda r: (r["year"], r["quarter"]))
    ordered = [r for r in series if r["value"] is not None]

    changes = []
    for i, row in enumerate(ordered):
        prev = ordered[i - 1] if i > 0 else None
        if prev and prev["value"]:
            changes.append({"period": row["period"], "change_pct": pct_change(row["value"], prev["value"])})

    index = {r["period"]: i for i, r in enumerate(ordered)}
    current = next((c for c in changes if c["period"] == tf.label), None)
    history = [c["change_pct"] for c in changes
               if index.get(c["period"], 10 ** 9) < index.get(tf.label, -1)
               and c["change_pct"] == c["change_pct"]]

    if current is None or len(history) < 3:
        return {"robust_z": None, "history_points": len(history),
                "change_pct": (current or {}).get("change_pct"),
                "note": ("Fewer than 3 historical periods for this member, so its own "
                         "variability cannot be estimated. Ranked on contribution and "
                         "over-index alone.")}

    med, sigma = robust_sigma(history)
    if not (sigma == sigma and sigma > 0):
        return {"robust_z": None, "history_points": len(history),
                "change_pct": current["change_pct"],
                "note": "This member's history is degenerate (zero dispersion)."}

    sigma *= (1.0 + 1.0 / max(len(history), 1)) ** 0.5      # small-sample inflation
    z = (current["change_pct"] - med) / sigma
    return {
        "robust_z": float(z),
        "change_pct": current["change_pct"],
        "median_historical_change_pct": float(med),
        "robust_sigma_pct": float(sigma),
        "history_points": len(history),
        "note": (f"This member's change of {current['change_pct']:.1f}% sits {abs(z):.1f} robust "
                 f"sigma from its own historical median of {med:.1f}% "
                 f"({len(history)} prior periods)."),
    }


def member_persistence(df, dimension: str, member: str, metric: str, tf,
                       resolver=None, lookback_quarters: int = 3) -> Dict[str, Any]:
    """
    When did this member turn, and did it stay turned?

    A one-week collapse and a quarter-long slide can produce an identical
    contribution figure while being entirely different business problems, and only
    one of them is still happening. Onset dating also separates factors that began
    at different times -- which is what makes a multi-factor movement legible.

    Reuses `analysis.detect_onset` rather than a second changepoint rule.
    """
    from .analysis import detect_onset, weekly_frame

    if dimension not in df.columns:
        return {"onset_week": None, "weeks_outside_band": 0, "weeks_in_period": 0}

    scoped = df[df[dimension] == member]
    if not len(scoped):
        return {"onset_week": None, "weeks_outside_band": 0, "weeks_in_period": 0}

    start_tf = tf
    for _ in range(lookback_quarters):
        start_tf = start_tf.previous()

    from .observe import slice_period
    lo = slice_period(scoped, start_tf)["_date"].min()
    hi = slice_period(scoped, tf)["_date"].max()
    period_start = slice_period(scoped, tf)["_date"].min()
    if lo != lo or hi != hi:
        return {"onset_week": None, "weeks_outside_band": 0, "weeks_in_period": 0}

    window = scoped[(scoped["_date"] >= lo) & (scoped["_date"] <= hi)]
    weeks = weekly_frame(window, metric, resolver)
    baseline_weeks = max(4, int((period_start - lo).days / 7) - 1) if period_start == period_start else 8

    direction = "down" if (member_direction(df, dimension, member, metric, tf, resolver) < 0) else "up"
    onset = detect_onset(weeks, baseline_weeks=baseline_weeks, direction=direction)

    # Count breaches inside the period itself, not across the run-up window.
    period_weeks = weekly_frame(slice_period(scoped, tf), metric, resolver)
    weeks_in_period = int(len(period_weeks))
    outside = 0
    if onset:
        threshold = onset["threshold"]
        for value in period_weeks["value"].tolist():
            if value != value:
                continue
            if (value < threshold) if direction == "down" else (value > threshold):
                outside += 1

    return {
        "onset_week": onset["week"] if onset else None,
        "onset_direction": direction,
        "weeks_outside_band": outside,
        "weeks_in_period": weeks_in_period,
        "baseline_weeks": baseline_weeks,
        "note": (f"Left its own baseline band in the week of {onset['week']} and stayed out for "
                 f"{outside} of the {weeks_in_period} weeks in the period."
                 if onset else
                 "Never left its own baseline band for two consecutive weeks in this window."),
    }


def member_direction(df, dimension: str, member: str, metric: str, tf, resolver=None) -> float:
    """Sign of this member's own change over the period, for onset direction."""
    from .metrics import compute
    from .observe import slice_period

    scoped = df[df[dimension] == member]
    cur = compute(slice_period(scoped, tf), metric, resolver)
    base = compute(slice_period(scoped, tf.previous()), metric, resolver)
    if cur != cur or base != base:
        return -1.0
    return 1.0 if cur - base > 0 else -1.0


# ---------------------------------------------------------------------------
# the composite driver score
# ---------------------------------------------------------------------------
#
# Four things make a dimension member a driver rather than an artefact, and no
# one of them is sufficient on its own:
#
#   magnitude      it carries a large share of the movement
#                  -- but the biggest segment carries the largest share of
#                     everything, which is arithmetic, not a finding
#   disproportion  it moved more than its own size implies
#                  -- but a small segment clears this on noise alone
#   significance   the move is large against THIS member's own history
#                  -- but a member can be reliably volatile and still matter
#   persistence    it stayed moved, rather than spiking for one week
#                  -- but a factor that started late is still real
#
# So the score is a weighted sum of all four, each normalised to [0, 1]:
#
#     score = (0.40*M + 0.25*D + 0.20*G + 0.15*P) / sum of the weights available
#
# The weights are an editorial choice, not a derived constant, and they are
# published as such -- in docs/RANKING.md and in `score_components` on every
# ranked row, so a reader can recompute the ranking by hand and disagree with it.
# scripts/validate_rca.py sweeps them +/-50% and reports whether the recovered
# top-3 changes; if the answer were yes, these weights would be doing the work
# instead of the data.
#
# A component that cannot be computed (no history for a robust z, no weekly
# series for persistence) is DROPPED and the remaining weights renormalised,
# rather than scored as zero. Absence of evidence is not evidence of absence, and
# scoring a missing component as 0 would silently demote every member of a short
# dataset.

WEIGHT_MAGNITUDE = 0.40
WEIGHT_DISPROPORTION = 0.25
WEIGHT_SIGNIFICANCE = 0.20
WEIGHT_PERSISTENCE = 0.15

# Saturation points: where a component stops earning more score.
OVER_INDEX_SATURATION = 2.0     # 2x its own size is as disproportionate as it needs to be
Z_FLOOR, Z_SATURATION = 1.0, 3.0

# Kept for backward compatibility: the historical flag threshold.
DISPROPORTIONATE_AT = 1.2


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def score_components(contribution_pct, over_index, robust_z,
                     weeks_outside_band, weeks_in_period) -> Dict[str, Any]:
    """The four normalised components, with unavailable ones left as None."""
    components: Dict[str, Any] = {}

    components["magnitude"] = (_clip(abs(contribution_pct) / 100.0)
                               if contribution_pct is not None else None)

    components["disproportion"] = (
        _clip((over_index - 1.0) / (OVER_INDEX_SATURATION - 1.0))
        if over_index is not None else None)

    components["significance"] = (
        _clip((abs(robust_z) - Z_FLOOR) / (Z_SATURATION - Z_FLOOR))
        if robust_z is not None else None)

    components["persistence"] = (
        _clip(weeks_outside_band / weeks_in_period)
        if (weeks_in_period or 0) > 0 else None)

    return components


def composite_score(components: Dict[str, Any],
                    weights: Dict[str, float] = None) -> Dict[str, Any]:
    """Weighted sum over the components that could be computed."""
    weights = weights or {
        "magnitude": WEIGHT_MAGNITUDE,
        "disproportion": WEIGHT_DISPROPORTION,
        "significance": WEIGHT_SIGNIFICANCE,
        "persistence": WEIGHT_PERSISTENCE,
    }
    available = {k: v for k, v in components.items() if v is not None and k in weights}
    if not available:
        return {"score": 0.0, "weight_used": 0.0, "components_used": []}
    total_weight = sum(weights[k] for k in available)
    score = sum(weights[k] * v for k, v in available.items()) / total_weight
    return {
        "score": float(score),
        "weight_used": float(total_weight),
        "components_used": sorted(available),
    }


def axis_weights(attribution: Dict[str, Any]) -> Dict[str, float]:
    """
    How much each dimension's members are allowed to count, from the Shapley split.

    Without this, ranking members across dimensions is dominated by whichever
    dimension happens to cut the business into the fewest, largest pieces. On the
    demo dataset `segment` splits Enterprise/SMB roughly 60/40, so Enterprise
    carries 60% of *any* movement and lands near the top of a contribution-ranked
    list -- while contributing at an over-index of 1.00, which is the definition
    of not being a driver.

    A member can only be a driver of the SHAPE of a change if its dimension
    explains the shape of that change, and the Shapley attribution above already
    measures exactly that. Normalising by the leading dimension keeps the top axis
    at 1.0 so the scale of the score stays readable.
    """
    phi = (attribution or {}).get("phi") or {}
    if not phi:
        return {}
    top = max(phi.values())
    if top <= 0:
        return {d: 1.0 for d in phi}
    return {d: phi[d] / top for d in phi}


def rank_drivers(drivers: Dict[str, List[Dict[str, Any]]], df, metric: str, tf,
                 resolver=None, limit: int = 6,
                 weights: Dict[str, float] = None,
                 axis: Dict[str, float] = None) -> List[Dict[str, Any]]:
    """
    Rank every dimension member that moved WITH the KPI, across all dimensions.

    Members contributing against the movement are excluded: in a decline, a
    segment that grew is not a driver of the decline, it is a mitigation, and
    mixing the two produces a ranking nobody can read.

    `axis` scales each member by how much its dimension explains the movement's
    shape (see `axis_weights`). Passing None ranks on member evidence alone.
    """
    candidates: List[Dict[str, Any]] = []
    for dimension, rows in (drivers or {}).items():
        for row in rows or []:
            if row.get("is_aggregate"):
                continue
            contribution = row.get("contribution_pct")
            if contribution is None or contribution <= 0:
                continue
            candidates.append({**row, "dimension": dimension})

    for row in candidates:
        significance = member_significance(df, row["dimension"], row["name"], metric, tf, resolver)
        persistence = member_persistence(df, row["dimension"], row["name"], metric, tf, resolver)
        components = score_components(
            row.get("contribution_pct"), row.get("over_index"),
            significance.get("robust_z"),
            persistence.get("weeks_outside_band"), persistence.get("weeks_in_period"))
        scored = composite_score(components, weights)
        axis_weight = (axis or {}).get(row["dimension"], 1.0)

        over_index = row.get("over_index")
        row.update({
            "driver_score": round(scored["score"] * axis_weight, 4),
            "member_score": round(scored["score"], 4),
            "axis_weight": round(axis_weight, 4),
            "score_components": {k: (round(v, 4) if v is not None else None)
                                 for k, v in components.items()},
            "score_weight_used": round(scored["weight_used"], 3),
            "member_robust_z": (round(significance["robust_z"], 3)
                                if significance.get("robust_z") is not None else None),
            "member_significance_note": significance.get("note"),
            "onset_week": persistence.get("onset_week"),
            "weeks_outside_band": persistence.get("weeks_outside_band"),
            "weeks_in_period": persistence.get("weeks_in_period"),
            "persistence_note": persistence.get("note"),
            # preserved verbatim: downstream stages and four test suites read this
            "is_disproportionate": bool(over_index is not None and over_index >= DISPROPORTIONATE_AT),
            "method": "shapley_axis x (contribution+over_index+robust_z+persistence)",
        })

    candidates.sort(key=lambda r: (r["driver_score"], r.get("contribution_pct") or 0), reverse=True)
    for i, row in enumerate(candidates):
        row["rank"] = i + 1
    return candidates[:limit]
