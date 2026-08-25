"""
Derived-KPI discovery — telling a real KPI from an arithmetic accident.

Any two numeric columns can be divided. Almost none of those quotients mean
anything. This module generates derived candidates from **typed rules over
profiled semantics**, and every rule carries a data precondition, not just a
type precondition:

  * a rate is only proposed when the numerator is genuinely contained by the
    denominator on the actual rows (`subset_of`, established in profiling), and
    when the denominator reads as a *population* rather than as just another
    measure that happens to be bigger;
  * a difference is only proposed between a measured inflow and a measured
    outflow;
  * a stock is never summed, and a rate is never summed or averaged.

Combinations the rules decline are not silently dropped — they are returned as
`RejectedCandidate` with the reason, so the user can see that the system
considered `revenue / cost_of_goods` and declined it rather than missing it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .contract import RejectedCandidate
from .library import LibraryMatch
from .profiling import DatasetProfile, FieldProfile, cash_direction

# A denominator turns a count into a *rate* only if it is a population — the
# set the numerator is drawn from. Without this gate, any larger count in the
# dataset would qualify and the system would propose stockouts-per-ticket.
#
# PRIMARY populations name the entity or event the business counts (orders,
# admissions, shipments). SECONDARY ones are quantities that can serve as a base
# but rarely define the canonical rate — a return rate is per order, not per
# unit sold. Only primary populations may anchor a generated rate.
PRIMARY_POPULATION_TOKENS = {
    "orders", "customers", "transactions", "purchases", "bookings", "admissions",
    "discharges", "patients", "shipments", "subscribers", "visits", "sessions",
    "users", "cases", "applications", "claims", "employees", "students",
    "enrolments", "enrollments", "accounts", "clients", "trips", "calls",
}
SECONDARY_POPULATION_TOKENS = {
    "units", "produced", "deliveries", "contacts", "leads", "impressions", "items",
}
POPULATION_TOKENS = PRIMARY_POPULATION_TOKENS | SECONDARY_POPULATION_TOKENS


def _titleise(name: str) -> str:
    return name.replace("_", " ").strip().capitalize()


@dataclass
class DerivedCandidate:
    """A proposed KPI that is not a direct column and not a library entry."""

    candidate_id: str
    name: str
    definition: str
    kind: str                                  # sum | mean | ratio
    expression: str = ""
    numerator: str = ""
    denominator: str = ""
    scale: float = 1.0
    unit: str = "count"
    higher_is_better: bool = True
    source_fields: List[str] = field(default_factory=list)
    rule: str = ""
    verdict: str = "valid"                     # valid | questionable
    confidence: float = 0.5
    semantic_tags: List[str] = field(default_factory=list)
    relevance: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def signature(self) -> Tuple[str, str, str, str]:
        """Identity for de-duplication against the library and other rules."""
        return (self.kind, self.expression, self.numerator, self.denominator)


def _is_population(prof: FieldProfile) -> bool:
    return bool(set(prof.name_tokens) & POPULATION_TOKENS)


def _is_primary_population(prof: FieldProfile) -> bool:
    return bool(set(prof.name_tokens) & PRIMARY_POPULATION_TOKENS)


def _canonical_denominator(numerator: FieldProfile,
                           candidates: Sequence[FieldProfile]) -> Optional[FieldProfile]:
    """
    Pick the one base a rate should be measured against.

    A numerator is typically contained by several measures at once — returns are
    within orders, within units sold and within customers-times-something. Only
    one of those is the rate the business actually uses. Preferring the broadest
    primary population reproduces the conventional choice (per order, per
    admission) and, just as importantly, stops one numerator from generating
    four near-duplicate KPIs the user then has to wade through.
    """
    eligible = [c for c in candidates
                if c.name != numerator.name
                and c.name in numerator.subset_of
                and _is_primary_population(c)]
    if not eligible:
        return None
    return max(eligible, key=lambda c: ((c.max or 0.0), c.name))


def library_signatures(matches: Sequence[LibraryMatch]) -> Set[Tuple[str, str, str, str]]:
    """Signatures already covered by a bound library KPI, so nothing duplicates."""
    from .library import resolve_expression

    sigs: Set[Tuple[str, str, str, str]] = set()
    for m in matches:
        fm = m.field_map
        sigs.add((
            m.library.kind,
            resolve_expression(m.library.expression, fm),
            resolve_expression(m.library.numerator, fm),
            resolve_expression(m.library.denominator, fm),
        ))
    return sigs


# ---------------------------------------------------------------------------
# atomic candidates
# ---------------------------------------------------------------------------
def build_atomic_candidates(profile: DatasetProfile,
                            covered_fields: Sequence[str]) -> List[DerivedCandidate]:
    """
    Every measure column not already exposed by a library KPI becomes a direct
    KPI, aggregated according to its profiled additivity.

    This is the successor to the old `extra_metrics` behaviour, which summed
    every unrecognised numeric column regardless of whether summing it made
    sense. A stock now averages and a rate is flagged.
    """
    covered = set(covered_fields)
    out: List[DerivedCandidate] = []
    for prof in profile.measures:
        if prof.name in covered:
            continue
        kind = "mean" if prof.additivity in ("stock", "non_additive", "rate") else "sum"
        unit = {
            "money": "currency", "count": "count", "duration": "duration",
            "rate_pct": "percent", "ratio": "ratio", "score": "score", "flag": "count",
        }.get(prof.semantic_type, "count")

        verdict, confidence, note = "valid", 0.6, ""
        if prof.additivity == "rate":
            # A pre-computed rate column cannot be re-aggregated correctly
            # without its components, which this dataset does not expose.
            verdict, confidence = "questionable", 0.35
            note = ("This column is already a rate. Aggregating it across periods or "
                    "dimensions averages an average, which is not the true rate. Supply "
                    "its numerator and denominator to compute it correctly.")

        out.append(DerivedCandidate(
            candidate_id=prof.name,
            name=_titleise(prof.name),
            definition=(note or f"{_titleise(prof.name)} recorded in the dataset, "
                                f"aggregated by {'averaging' if kind == 'mean' else 'summing'} "
                                f"because it is a {prof.additivity}."),
            kind=kind,
            expression="{" + prof.name + "}",
            unit=unit,
            higher_is_better=True,
            source_fields=[prof.name],
            rule="atomic_measure",
            verdict=verdict,
            confidence=confidence,
            semantic_tags=[prof.semantic_type],
            relevance=f"A directly measured quantity available in this dataset.",
            evidence={"semantic_type": prof.semantic_type, "additivity": prof.additivity,
                      "aggregation": kind},
        ))
    return out


# ---------------------------------------------------------------------------
# derived candidates
# ---------------------------------------------------------------------------
def build_derived_candidates(
    profile: DatasetProfile,
    exclude: Optional[Set[Tuple[str, str, str, str]]] = None,
) -> Tuple[List[DerivedCandidate], List[RejectedCandidate]]:
    """Run every derivation rule. Returns (candidates, rejected-with-reason)."""
    exclude = exclude or set()
    measures = profile.measures
    by_name = {m.name: m for m in measures}
    candidates: List[DerivedCandidate] = []
    rejected: List[RejectedCandidate] = []
    seen: Set[Tuple[str, str, str, str]] = set(exclude)

    def emit(cand: DerivedCandidate) -> None:
        if cand.signature in seen:
            return
        seen.add(cand.signature)
        candidates.append(cand)

    def reject(label: str, expression: str, rule: str, reason: str) -> None:
        rejected.append(RejectedCandidate(label=label, expression=expression,
                                          rule=rule, reason=reason))

    # -- rule 1: inflow - outflow -> surplus -------------------------------
    money = [m for m in measures if m.semantic_type == "money" and m.additivity == "flow"]
    for a in money:
        for b in money:
            if a.name == b.name:
                continue
            if cash_direction(a) != "inflow" or cash_direction(b) != "outflow":
                continue
            emit(DerivedCandidate(
                candidate_id=f"{a.name}_less_{b.name}",
                name=f"{_titleise(a.name)} less {b.name.replace('_', ' ')}",
                definition=f"{_titleise(a.name)} minus {b.name.replace('_', ' ')} over the period.",
                kind="sum",
                expression="{" + a.name + "} - {" + b.name + "}",
                unit="currency",
                source_fields=[a.name, b.name],
                rule="inflow_minus_outflow",
                verdict="valid" if b.name in a.subset_of else "questionable",
                confidence=0.7 if b.name in a.subset_of else 0.45,
                semantic_tags=["profitability"],
                relevance="What is left after this cost is taken off.",
                evidence={"inflow": a.name, "outflow": b.name,
                          "outflow_contained_by_inflow": b.name in a.subset_of},
            ))

            # -- rule 2: (A - B) / A -> margin-shaped ratio ----------------
            if b.name in a.subset_of:
                emit(DerivedCandidate(
                    candidate_id=f"{a.name}_less_{b.name}_pct",
                    name=f"{_titleise(a.name)} margin %",
                    definition=(f"{_titleise(a.name)} less {b.name.replace('_', ' ')}, "
                                f"as a share of {a.name.replace('_', ' ')}."),
                    kind="ratio",
                    numerator="{" + a.name + "} - {" + b.name + "}",
                    denominator="{" + a.name + "}",
                    scale=100.0, unit="percent",
                    source_fields=[a.name, b.name],
                    rule="margin_ratio",
                    verdict="valid", confidence=0.7,
                    semantic_tags=["profitability", "efficiency"],
                    relevance="Separates a volume problem from a cost or price problem.",
                    evidence={"subset_check": f"{b.name} <= {a.name} on the rows",
                              "subset_evidence": b.evidence.get("subset_checks", {}).get(a.name)},
                ))

    # -- rule 3: outcome / population -> a rate ----------------------------
    counts = [m for m in measures
              if m.semantic_type in ("count", "flag") and m.additivity == "flow"]
    for num in counts:
        # Report the rate someone would plausibly expect but that the data does
        # not support. Pairs with no plausible relation at all are not worth
        # narrating — that would bury the findings that matter.
        for den in counts:
            if num.name == den.name or den.name in num.subset_of:
                continue
            if _is_primary_population(den) and not _is_primary_population(num):
                reject(f"{_titleise(num.name)} per {den.name.replace('_', ' ')}",
                       f"{num.name} / {den.name}", "outcome_rate",
                       f"'{num.name}' is not contained by '{den.name}' on the rows, so their "
                       f"quotient is not a rate. Define it explicitly if the business uses it.")

        den = _canonical_denominator(num, counts)
        if den is None:
            continue
        emit(DerivedCandidate(
            candidate_id=f"{num.name}_per_{den.name}_rate",
            name=f"{_titleise(num.name)} rate",
            definition=(f"{_titleise(num.name)} as a share of "
                        f"{den.name.replace('_', ' ')}."),
            kind="ratio",
            numerator="{" + num.name + "}", denominator="{" + den.name + "}",
            scale=100.0, unit="percent",
            source_fields=[num.name, den.name],
            rule="outcome_rate",
            verdict="valid", confidence=0.65,
            semantic_tags=["outcome_rate"],
            relevance=(f"Normalises {num.name.replace('_', ' ')} by volume, so it is "
                       f"comparable across periods of different size."),
            evidence={"subset_check": num.evidence.get("subset_checks", {}).get(den.name),
                      "denominator": den.name, "denominator_is_population": True,
                      "other_containing_populations":
                          [c for c in num.subset_of if c != den.name]},
        ))

    # -- rule 4: money / population -> unit economics ----------------------
    # One canonical base, for the same reason rates get one: the broadest
    # primary population is the activity the business measures itself by.
    primary_pops = [c for c in counts if _is_primary_population(c)]
    base = max(primary_pops, key=lambda c: ((c.max or 0.0), c.name)) if primary_pops else None
    for m in [x for x in measures if x.semantic_type == "money" and x.additivity == "flow"]:
        for den in ([base] if base else []):
            emit(DerivedCandidate(
                candidate_id=f"{m.name}_per_{den.name}",
                name=f"{_titleise(m.name)} per {den.name.replace('_', ' ').rstrip('s')}",
                definition=(f"{_titleise(m.name)} divided by "
                            f"{den.name.replace('_', ' ')}."),
                kind="ratio",
                numerator="{" + m.name + "}", denominator="{" + den.name + "}",
                unit="currency",
                higher_is_better=cash_direction(m) != "outflow",
                source_fields=[m.name, den.name],
                rule="unit_economics",
                verdict="valid", confidence=0.6,
                semantic_tags=["unit_economics"],
                relevance="Unit economics — whether each unit of activity is getting "
                          "more or less valuable.",
                evidence={"denominator_is_population": True},
            ))

    # -- rule 5: duration -> mean ------------------------------------------
    for m in [x for x in measures if x.semantic_type == "duration"]:
        emit(DerivedCandidate(
            candidate_id=f"avg_{m.name}",
            name=f"Average {m.name.replace('_', ' ')}",
            definition=f"Mean {m.name.replace('_', ' ')} across the period.",
            kind="mean", expression="{" + m.name + "}",
            unit="duration", higher_is_better=False,
            source_fields=[m.name], rule="duration_mean",
            verdict="valid", confidence=0.65,
            semantic_tags=["efficiency", "throughput"],
            relevance="A duration is averaged, never summed — the total is meaningless.",
            evidence={"additivity": m.additivity},
        ))

    # -- explicit rejections worth surfacing --------------------------------
    rates = [m for m in measures if m.additivity == "rate"]
    for a in rates:
        for b in rates:
            if a.name != b.name:
                reject(f"{_titleise(a.name)} / {b.name.replace('_', ' ')}",
                       f"{a.name} / {b.name}", "rate_over_rate",
                       "Dividing one rate by another compounds two normalisations and "
                       "produces a number with no business meaning.")
                break

    for a in money:
        for b in money:
            if a.name == b.name or b.name in a.subset_of or a.name in b.subset_of:
                continue
            if cash_direction(a) == "inflow" and cash_direction(b) == "outflow":
                reject(f"{_titleise(a.name)} / {b.name.replace('_', ' ')}",
                       f"{a.name} / {b.name}", "money_over_money",
                       f"'{b.name}' is not a component of '{a.name}' on the rows, so their "
                       f"ratio is a coverage multiple rather than a margin. Define it "
                       f"explicitly if the business uses it.")
    return candidates, rejected
