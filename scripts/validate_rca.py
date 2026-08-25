"""
Score the root-cause analysis against the planted ground truth.

Runs the REAL pipeline -- the same `observe`/`investigate`/`contest` the API calls,
with no API key and no mocks -- over sample_data/business_metrics_sample.csv, then
compares what it found against sample_data/ground_truth.json, which the generator
emitted from counterfactual re-simulations.

    python scripts/validate_rca.py           # print the scorecard
    make validate                            # same, plus writes docs/rca_validation.json

Every bar is declared in CHECKS below rather than buried in prose, and
backend/tests/test_rca_ground_truth.py asserts the same scorecard, so a
regression in driver ranking fails the build rather than quietly degrading a
number nobody re-reads.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
for path in (str(BACKEND), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

TRUTH_PATH = ROOT / "sample_data" / "ground_truth.json"
REPORT_PATH = ROOT / "docs" / "rca_validation.json"

# --- the bars ---------------------------------------------------------------
MIN_LOCUS_RECALL = 1.0            # all three planted loci must be found
MIN_PRECISION_AT_3 = 0.66         # at most one of the top 3 may be a non-factor
MIN_RANK_CORRELATION = 0.8        # Spearman, engine order vs true-impact order
MAX_ATTRIBUTION_MAE_PP = 15.0     # vs each factor's true share of the planted effect
MAX_DECOMPOSITION_ERROR_PP = 0.01  # vs the arithmetic the engine must reproduce exactly
MAX_ONSET_ERROR_WEEKS = 2


def spearman(a, b):
    """Rank correlation without scipy: Pearson over the ranks."""
    import numpy as np

    def ranks(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        out = [0.0] * len(xs)
        i = 0
        while i < len(order):                      # average ties
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            shared = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                out[order[k]] = shared
            i = j + 1
        return out

    if len(a) < 2:
        return None
    ra, rb = np.array(ranks(a)), np.array(ranks(b))
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def weeks_between(a: str, b: str) -> int:
    return abs((date.fromisoformat(a) - date.fromisoformat(b)).days) // 7


def matches_locus(driver, locus) -> bool:
    """A ranked driver identifies a planted locus if it names that dimension member."""
    return any(driver.get("dimension") == dim and str(driver.get("name")) == str(member)
               for dim, member in locus.items())


def run_pipeline():
    """The real engines, over the real sample data, with no LLM."""
    from tests.base import setup_environment                    # noqa: E402

    state = setup_environment()
    from app.engines.analysis import price_volume_decomposition  # noqa: E402
    from app.engines.contest import contest                      # noqa: E402
    from app.engines.investigate import investigate              # noqa: E402
    from app.engines.observe import Timeframe, observe, slice_period  # noqa: E402

    df, schema, uid = state["df"], state["schema"], state["uid"]
    tf = Timeframe(2026, 2)
    observation = observe(df, schema, "revenue", tf)
    investigation = investigate(df, schema, observation, uid, llm=None)
    contested = contest(df, schema, observation, investigation, uid, llm=None)
    pv = price_volume_decomposition(slice_period(df, tf), slice_period(df, tf.previous()))
    return {"df": df, "schema": schema, "uid": uid, "observation": observation,
            "investigation": investigation, "contest": contested, "price_volume": pv}


def score(truth, run):
    obs = run["observation"]
    drivers = obs.get("top_drivers") or []
    factors = truth["planted_factors"]
    checks = []

    def check(name, passed, detail, value=None, bar=None):
        checks.append({"check": name, "pass": bool(passed), "value": value,
                       "bar": bar, "detail": detail})

    # 1. was the movement detected at all
    check("detection",
          obs.get("verdict") == truth["expected_findings"]["verdict"],
          f"verdict={obs.get('verdict')}, change={obs.get('change_pct'):.1f}%",
          obs.get("verdict"), truth["expected_findings"]["verdict"])

    # 2. locus recall -- every planted factor's locus appears somewhere in the ranking
    found = {}
    for factor in factors:
        hit = next((d for d in drivers if matches_locus(d, factor["locus"])), None)
        found[factor["id"]] = hit
    recall = sum(1 for v in found.values() if v) / len(factors)
    missing = [f["id"] for f in factors if not found[f["id"]]]
    check("locus_recall", recall >= MIN_LOCUS_RECALL,
          (f"{sum(1 for v in found.values() if v)}/{len(factors)} planted loci found"
           + (f"; missing {missing}" if missing else "")),
          round(recall, 3), MIN_LOCUS_RECALL)

    # 3. precision at 3 -- the top of the ranking is not padded with non-factors
    top3 = drivers[:3]
    hits = sum(1 for d in top3
               if any(matches_locus(d, f["locus"]) for f in factors))
    precision = hits / max(len(top3), 1)
    check("precision_at_3", precision >= MIN_PRECISION_AT_3,
          f"{hits}/{len(top3)} of the top 3 are planted loci: "
          + ", ".join(f"{d['dimension']}={d['name']}" for d in top3),
          round(precision, 3), MIN_PRECISION_AT_3)

    # 4. rank correlation against true impact
    pairs = [(found[f["id"]]["rank"], f["expected_rank"])
             for f in factors if found[f["id"]]]
    rho = spearman([p[0] for p in pairs], [p[1] for p in pairs]) if len(pairs) > 1 else None
    check("rank_correlation", rho is not None and rho >= MIN_RANK_CORRELATION,
          ("engine order " + " < ".join(f"{f['id']}(#{found[f['id']]['rank']})"
                                        for f in factors if found[f["id"]])
           + f"; true order by Shapley impact " + " < ".join(
               f"{f['id']}(#{f['expected_rank']})" for f in factors)),
          None if rho is None else round(rho, 3), MIN_RANK_CORRELATION)

    # 5. the decomposition arithmetic itself must be exact
    #    (contribution_pct is a share of the OBSERVED movement, so it has to equal
    #     the observed share the generator computed from the same rows)
    worst, worst_where = 0.0, None
    for dim, rows in (truth.get("dimension_truth") or {}).items():
        engine_rows = {r["name"]: r for r in (obs.get("drivers") or {}).get(dim, [])
                       if not r.get("is_aggregate")}
        for member, expected in rows.items():
            got = engine_rows.get(member)
            if not got or got.get("contribution_pct") is None:
                continue
            err = abs(got["contribution_pct"] - expected["observed_share_pct"])
            if err > worst:
                worst, worst_where = err, f"{dim}={member}"
    check("decomposition_exact", worst <= MAX_DECOMPOSITION_ERROR_PP,
          f"worst gap {worst:.4f}pp at {worst_where} between the engine's contribution_pct "
          f"and the same share recomputed by the generator",
          round(worst, 4), MAX_DECOMPOSITION_ERROR_PP)

    # 6. attribution error against the TRUE (counterfactual) share of the planted effect.
    #    These use different denominators on purpose -- see the note below.
    errors = []
    for factor in factors:
        hit = found[factor["id"]]
        if not hit:
            continue
        dim, member = next(iter(factor["locus"].items()))
        planted = (truth["dimension_truth"].get(dim) or {}).get(member)
        if not planted:
            continue
        errors.append({
            "factor": factor["id"],
            "engine_contribution_pct": hit.get("contribution_pct"),
            "true_planted_share_pct": planted["planted_share_pct"],
            "error_pp": abs(hit.get("contribution_pct", 0) - planted["planted_share_pct"]),
        })
    mae = sum(e["error_pp"] for e in errors) / len(errors) if errors else None
    check("attribution_mae", mae is not None and mae <= MAX_ATTRIBUTION_MAE_PP,
          ("mean absolute error "
           + (f"{mae:.1f}pp" if mae is not None else "n/a")
           + " between the engine's share of the OBSERVED movement and each locus's "
             "true share of the PLANTED effect. The two denominators differ by natural "
             f"drift ({truth['totals']['natural_drift']:+,.0f}), which no engine should "
             "attribute to a factor."),
          None if mae is None else round(mae, 2), MAX_ATTRIBUTION_MAE_PP)

    # 7. onset dating
    onsets = []
    for factor in factors:
        hit = found[factor["id"]]
        if not hit or not hit.get("onset_week"):
            continue
        gap = weeks_between(hit["onset_week"], factor["onset_date"])
        onsets.append({"factor": factor["id"], "planted": factor["onset_date"],
                       "detected": hit["onset_week"], "error_weeks": gap})
    worst_onset = max((o["error_weeks"] for o in onsets), default=None)
    check("onset_dating", worst_onset is not None and worst_onset <= MAX_ONSET_ERROR_WEEKS,
          "; ".join(f"{o['factor']} planted {o['planted']} detected {o['detected']} "
                    f"({o['error_weeks']}w)" for o in onsets),
          worst_onset, MAX_ONSET_ERROR_WEEKS)

    # 8. the temporal contradiction: the best-documented cause post-dates the KPI move.
    #
    #    Tested on the SIGNATURE METRIC of each factor, not on the revenue of its
    #    locus. Product A's revenue turns in April, because Product A is hit by
    #    BOTH the North erosion (April) and the supply disruption (May) -- so
    #    dating the supply factor by Product A's revenue would date it four weeks
    #    early. The supply disruption's own signature is the fulfilment collapse,
    #    and that is what has to be compared against the revenue decline.
    #
    #    Deliberately computed from the primitives rather than read off a
    #    hypothesis: with no API key the deterministic floor generates
    #    concentration and related-driver hypotheses, and the supply mechanism
    #    only appears when an LLM proposes it. The temporal FACT must be
    #    demonstrable without one.
    from app.engines.analysis import compare_onsets, detect_onset, weekly_frame  # noqa: E402

    #    Scoped to the focus the engine itself selected (North x Product A), the
    #    way `contest.temporal_check` scopes it. Widening to Product A across all
    #    regions dilutes the fulfilment collapse into three unaffected regions'
    #    noise, and the onset detector then fires on a February wobble.
    df = run["df"]
    focus = run["investigation"].get("focus") or {}
    window = df[(df["_date"] >= "2025-10-01") & (df["_date"] <= "2026-06-30")]
    scoped = window
    for dim, member in focus.items():
        scoped = scoped[scoped[dim] == member]
    north = window[window["region"] == focus.get("region", "North")]

    kpi_onset = detect_onset(weekly_frame(north, "revenue"), baseline_weeks=20, direction="down")
    cause_onset = detect_onset(weekly_frame(scoped, "stockout_rate"),
                               baseline_weeks=20, direction="up")
    verdict = compare_onsets(kpi_onset, cause_onset)
    ordering = truth["expected_findings"]["temporal_ordering"][0]
    check("temporal_ordering", verdict.get("status") == "kpi_precedes_cause",
          (f"revenue in North turned {kpi_onset['week'] if kpi_onset else '?'}, "
           f"the Product A fulfilment signal turned "
           f"{cause_onset['week'] if cause_onset else '?'} "
           f"-> {verdict.get('status')} (lag {verdict.get('lag_weeks')}w). "
           f"{ordering['why']}"),
          verdict.get("status"), "kpi_precedes_cause")

    # 8b. informational: if an LLM proposed a supply hypothesis, did CONTEST cap it?
    supply = next((h for h in (run["contest"].get("hypotheses") or [])
                   if "suppl" in (h.get("family", "") + h.get("title", "")).lower()), None)
    if supply:
        status = (supply.get("contest") or {}).get("temporal", {}).get("status")
        check("supply_hypothesis_capped",
              status == "kpi_precedes_cause" and (supply["scoring"].get("cap_reason") or ""),
              f"supply hypothesis temporal={status!r}, "
              f"confidence={supply['scoring']['confidence']}, "
              f"cap={supply['scoring'].get('cap_reason') or 'none'}",
              status, "kpi_precedes_cause")

    # 9. mechanism separation -- a demand-led decline must read as volume, not price
    pv = run.get("price_volume") or {}
    volume_led = abs(pv.get("volume_effect") or 0) > abs(pv.get("price_effect") or 0)
    check("mechanism_volume_led", volume_led,
          (f"volume effect {pv.get('volume_effect', 0):+,.0f} vs price effect "
           f"{pv.get('price_effect', 0):+,.0f}" if pv else "price/volume split unavailable"),
          "volume_led" if volume_led else "price_led", "volume_led")

    return checks, {"found": {k: (v or {}).get("rank") for k, v in found.items()},
                    "attribution": errors, "onsets": onsets}


def weight_sensitivity(run, truth):
    """
    Does the answer come from the data or from the weights?

    Re-ranks with each score weight scaled to 50% and 150% of its published value
    and reports whether the top-3 SET changes. If it did, the weights would be
    doing the work the evidence is supposed to do.
    """
    from app.engines.drivers import (WEIGHT_DISPROPORTION, WEIGHT_MAGNITUDE,
                                     WEIGHT_PERSISTENCE, WEIGHT_SIGNIFICANCE,
                                     axis_weights, rank_drivers)
    from app.engines.observe import Timeframe

    obs = run["observation"]
    base_weights = {"magnitude": WEIGHT_MAGNITUDE, "disproportion": WEIGHT_DISPROPORTION,
                    "significance": WEIGHT_SIGNIFICANCE, "persistence": WEIGHT_PERSISTENCE}
    axis = axis_weights(obs.get("dimension_shapley") or {})
    baseline = {(d["dimension"], d["name"]) for d in (obs.get("top_drivers") or [])[:3]}

    variants, stable = [], True
    for key in base_weights:
        for scale in (0.5, 1.5):
            weights = dict(base_weights)
            weights[key] = base_weights[key] * scale
            ranked = rank_drivers(obs.get("drivers") or {}, run["df"], "revenue",
                                  Timeframe(2026, 2), run["schema"].contract_resolver,
                                  limit=3, weights=weights, axis=axis)
            top3 = {(d["dimension"], d["name"]) for d in ranked}
            same = top3 == baseline
            stable = stable and same
            variants.append({"weight": key, "scale": scale, "top3_unchanged": same,
                             "top3": [f"{d['dimension']}={d['name']}" for d in ranked]})
    return {"stable": stable, "variants": variants,
            "baseline_top3": sorted(f"{d}={m}" for d, m in baseline)}


def main():
    truth = json.loads(TRUTH_PATH.read_text())
    run = run_pipeline()
    checks, detail = score(truth, run)
    sensitivity = weight_sensitivity(run, truth)
    checks.append({
        "check": "weight_sensitivity",
        "pass": sensitivity["stable"],
        "value": "stable" if sensitivity["stable"] else "unstable",
        "bar": "top 3 unchanged under +/-50% on every weight",
        "detail": (f"{sum(1 for v in sensitivity['variants'] if v['top3_unchanged'])}"
                   f"/{len(sensitivity['variants'])} perturbations leave the top 3 unchanged"),
    })

    passed = sum(1 for c in checks if c["pass"])
    report = {
        "dataset": truth["dataset"],
        "target": truth["target"],
        "generated_from": "scripts/validate_rca.py",
        "summary": {"passed": passed, "total": len(checks),
                    "all_passed": passed == len(checks)},
        "checks": checks,
        "ranked_drivers": [
            {k: d.get(k) for k in ("rank", "dimension", "name", "driver_score",
                                   "member_score", "axis_weight", "contribution_pct",
                                   "over_index", "member_robust_z", "onset_week",
                                   "weeks_outside_band", "weeks_in_period",
                                   "score_components")}
            for d in (run["observation"].get("top_drivers") or [])
        ],
        "dimension_shapley": (run["observation"].get("dimension_ranking") or []),
        "detail": detail,
        "weight_sensitivity": sensitivity,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")

    width = max(len(c["check"]) for c in checks)
    print(f"\nRCA validation -- {truth['target']['kpi']} {truth['target']['period']} "
          f"vs {truth['target']['baseline']}\n")
    for c in checks:
        mark = "PASS" if c["pass"] else "FAIL"
        print(f"  [{mark}] {c['check']:<{width}}  {c['detail']}")
    print(f"\n  {passed}/{len(checks)} checks passed")
    print(f"  report -> {REPORT_PATH}\n")

    print("  ranked drivers")
    for d in (run["observation"].get("top_drivers") or []):
        print(f"    {d['rank']}. {d['dimension']:8} {d['name']:14} "
              f"score={d['driver_score']:.3f}  contribution={d['contribution_pct']:5.1f}%  "
              f"onset={d['onset_week']}")
    print()
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
