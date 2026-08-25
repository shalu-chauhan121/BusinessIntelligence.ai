"""
Business-domain detection for KPI *explanations*.

This module answers one question: given the concepts that were actually bound
to real fields in this dataset (`library.bind_library`), what is the most
plausible kind of business this data describes? The answer drives the words an
explanation reaches for — in `explanation.py` when a KPI is defined, and in the
investigation engines when a change is explained — never a formula, a source
field or a computed number. Nothing here can change what a KPI equals.

Domain detection reuses the same evidence the library binder already produced
rather than inventing a second classifier: a library KPI only activates when
every concept it needs binds to a real, semantically-typed field, so counting
which domain packs actually bound is itself dataset evidence, not a guess.
`general` concepts (revenue, cost, orders...) are excluded from the count
because they say nothing about *industry* — a hospital and a retailer both
have a cost.

When the signal is thin or tied between two domains, detection says so
explicitly (`is_uncertain=True`) rather than picking a side. Explanations then
degrade to language that is honest about not knowing the industry, while still
using the dataset's own field and dimension names — the one thing that is
never in doubt.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from .library import LibraryMatch
from .profiling import DatasetProfile


@dataclass(frozen=True)
class DomainVocab:
    """The words an explanation reaches for for one kind of business."""

    label: str            # "a healthcare / hospital operation"
    entity: str            # who/what the business serves or processes
    activity: str           # the core recurring business activity
    unit_of_work: str        # the recurring unit that activity is counted in
    capacity_note: str       # what "capacity" or "constraint" means here


DOMAIN_VOCAB: Dict[str, DomainVocab] = {
    "general": DomainVocab(
        label="this business",
        entity="customers",
        activity="its commercial operations",
        unit_of_work="a transaction",
        capacity_note="how much the business can sell or deliver",
    ),
    "retail_ecommerce": DomainVocab(
        label="a retail / e-commerce business",
        entity="shoppers",
        activity="stocking, selling and fulfilling orders",
        unit_of_work="an order",
        capacity_note="how much stock is available to sell",
    ),
    "healthcare": DomainVocab(
        label="a healthcare / hospital operation",
        entity="patients",
        activity="admitting, treating and discharging patients",
        unit_of_work="a patient episode",
        capacity_note="beds, staff and clinical capacity",
    ),
    "logistics": DomainVocab(
        label="a logistics / delivery operation",
        entity="shipments",
        activity="moving and delivering shipments",
        unit_of_work="a shipment",
        capacity_note="fleet, route and warehouse capacity",
    ),
    "saas_subscription": DomainVocab(
        label="a subscription / SaaS business",
        entity="subscribers",
        activity="acquiring and retaining subscribers on recurring revenue",
        unit_of_work="a subscription",
        capacity_note="how much recurring revenue the customer base supports",
    ),
    "manufacturing": DomainVocab(
        label="a manufacturing operation",
        entity="units of output",
        activity="producing and quality-checking output",
        unit_of_work="a production run",
        capacity_note="line, machine and labour capacity",
    ),
}

# Used whenever the evidence does not clearly point to one industry. Deliberately
# generic in a way that names its own uncertainty rather than guessing.
UNKNOWN_VOCAB = DomainVocab(
    label="this dataset's business (its industry could not be confidently "
          "determined from the available fields)",
    entity="the entities recorded in the dataset",
    activity="its core recurring operations",
    unit_of_work="a recorded event",
    capacity_note="the operational capacity implied by the data",
)


@dataclass
class DomainContext:
    """The detected business context, with the evidence for it kept alongside."""

    domain: str                                  # a DOMAIN_VOCAB key, or "uncertain"
    vocab: DomainVocab
    confidence: float
    is_uncertain: bool
    evidence: List[str] = field(default_factory=list)


def domain_context_from_contract(contract: Any) -> DomainContext:
    """
    Rebuild the detected business context from a stored contract.

    `detect_domain_context` runs once, at discovery, and its result is flattened
    onto the contract as four scalar fields. The analysis engines run long after
    that, with no access to the library matches the detection needed — so rather
    than re-detect (which would mean re-profiling on every request), the context
    is reconstructed here. `vocab` is a pure function of the domain key, so
    nothing is lost.

    A contract that never recorded a domain yields the uncertain context, which
    is exactly what an explanation should be grounded in when the industry is
    genuinely unknown.
    """
    domains = list(getattr(contract, "detected_domains", None) or [])
    domain = domains[0] if domains else "uncertain"
    vocab = DOMAIN_VOCAB.get(domain, UNKNOWN_VOCAB)
    return DomainContext(
        domain=domain,
        vocab=vocab,
        confidence=float(getattr(contract, "domain_confidence", 0.0) or 0.0),
        is_uncertain=bool(getattr(contract, "domain_uncertain", True)),
        evidence=list(getattr(contract, "domain_evidence", None) or []),
    )


def detect_domain_context(matches: Sequence[LibraryMatch], profile: DatasetProfile,
                          min_matches: int = 2) -> DomainContext:
    """
    The most plausible business context for this *dataset*, from concept
    bindings alone.

    `min_matches` mirrors `library.detect_domains`: a single coincidental bind
    is not enough evidence to name an industry. Below that, or on a tie
    between two domains, the result is `is_uncertain=True` and callers must
    write around that rather than assert a specific business.
    """
    counts: Dict[str, int] = {}
    concepts_by_domain: Dict[str, set] = {}
    for m in matches:
        d = m.library.domain
        if d == "general":
            continue
        counts[d] = counts.get(d, 0) + 1
        concepts_by_domain.setdefault(d, set()).update(m.bindings.keys())

    # Dimension names are corroborating evidence, not a decider on their own —
    # a column called `department` supports "this looks like a hospital or a
    # school" but never outweighs which KPI concepts actually bound. Kept as
    # a supporting clause on the evidence trail so a reviewer can see what was
    # available, even though it does not change which domain wins below.
    dim_names = [d.name for d in profile.dimensions if d.role == "dimension"]
    dim_note = f" Dataset dimensions: {', '.join(dim_names[:8])}." if dim_names else ""

    if not counts:
        return DomainContext(
            domain="uncertain", vocab=UNKNOWN_VOCAB, confidence=0.15, is_uncertain=True,
            evidence=["No industry-specific KPI concepts (healthcare, retail, logistics, "
                      "subscription or manufacturing) bound to any field in this dataset — "
                      "only general-purpose ones did, if any." + dim_note],
        )

    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    top_domain, top_count = ranked[0]
    runner_count = ranked[1][1] if len(ranked) > 1 else 0
    concepts = sorted(concepts_by_domain[top_domain])

    if runner_count == top_count and len(ranked) > 1:
        runner_domain = ranked[1][0]
        evidence = [
            f"'{top_domain}' and '{runner_domain}' each bound {top_count} domain concept(s) "
            f"({', '.join(concepts[:6])} vs. {', '.join(sorted(concepts_by_domain[runner_domain])[:6])}) "
            "— the signal does not favour either." + dim_note,
        ]
        return DomainContext(domain="uncertain", vocab=UNKNOWN_VOCAB, confidence=0.35,
                             is_uncertain=True, evidence=evidence)

    weak = top_count < min_matches
    confidence = round(min(0.95, 0.35 + 0.15 * top_count), 2)
    evidence = [
        f"{top_count} '{top_domain}' KPI concept(s) bound directly to dataset fields "
        f"({', '.join(concepts[:6])})." + dim_note,
    ]
    if weak:
        confidence = round(confidence * 0.6, 2)
        evidence.append("Only one concept matched — treat this as a weak signal, not a "
                        "confirmed industry.")

    return DomainContext(domain=top_domain, vocab=DOMAIN_VOCAB.get(top_domain, UNKNOWN_VOCAB),
                         confidence=confidence, is_uncertain=weak, evidence=evidence)
