"""
Domain-grounded KPI explanations.

Every KPI definition carries `business_definition` (what it means / what
activity it represents) and `relevance` (why it matters / what a change would
indicate). Before this module those strings were either a single hand-written
sentence per library entry — identical regardless of which dataset activated
it — or a mechanical description assembled from the column name alone
("Mean length_of_stay_days across the period."). Neither answers the four
questions a real explanation needs to:

  1. What does this KPI mean specifically in THIS business?
  2. Why is it relevant to THIS business?
  3. What business activity does it represent?
  4. What would a change in it indicate, in this domain?

`compose_explanation` answers all four from two things only: the KPI's own
shape (its semantic tags, kind, unit, direction, source fields — all already
known and deterministic) and a `DomainContext` computed from evidence bound
in the dataset (`domain.detect_domain_context`). It never touches a formula,
a source field or a computed value — only the prose describing them. When the
domain is uncertain, the caller already knows that (`ctx.is_uncertain`) and
the composed text says so plainly instead of asserting an industry the data
does not support.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .domain import DomainContext

# ---------------------------------------------------------------------------
# per-tag narrative
#
# Each entry is (what_it_means_and_activity, why_relevant_and_change_signal).
# Both are format strings over a DomainVocab's fields, so the same tag reads
# differently for a hospital than for a retailer without being rewritten by
# hand for every KPI — the domain, not the tag, is what should vary the text.
# ---------------------------------------------------------------------------
TAG_PHRASES: Dict[str, Tuple[str, str]] = {
    "clinical_outcome": (
        "It measures a clinical outcome of {activity}, for {label}.",
        "A change here reflects the quality of care being delivered, not just how "
        "many {entity} were seen — that is what makes it a clinical measure rather "
        "than an activity count.",
    ),
    "retention": (
        "It measures whether {entity} continue on, once acquired, for {label}.",
        "A worsening figure is usually the strongest leading indicator of future "
        "revenue decline for {label}; it should be watched before topline numbers move.",
    ),
    "quality_defect": (
        "It measures a quality failure occurring during {activity}, for {label}.",
        "A rise signals a quality or process problem within {activity} worth "
        "investigating; a fall signals that process is improving.",
    ),
    "outcome_rate": (
        "It expresses an outcome of {activity} as a share of every {unit_of_work}, "
        "for {label}.",
        "A change here is a rate change, not a volume change — it reflects how often "
        "the outcome occurs, independent of how many {unit_of_work}s there were.",
    ),
    "profitability": (
        "It measures what {label} keeps after the direct cost of {activity} is "
        "taken off.",
        "A rise means more value is being retained per {unit_of_work}; a fall means "
        "margin is eroding even if activity itself looks healthy.",
    ),
    "unit_economics": (
        "It expresses the economics of {activity} on a per-{unit_of_work} basis, "
        "for {label}.",
        "A change means each {unit_of_work} is becoming more or less valuable — "
        "distinct from a change driven purely by doing more or less of {activity}.",
    ),
    "efficiency": (
        "It measures how efficiently {label} converts its resources into {activity}.",
        "Whether a rise or a fall is the good direction depends on this KPI's own "
        "stated direction, but either way the change points at process efficiency, "
        "not raw volume, within {activity}.",
    ),
    "capacity": (
        "It measures {capacity_note} that {label} consumed while carrying out "
        "{activity}.",
        "A rise means more of that capacity was in use; a sustained high figure "
        "signals {label} is close to a capacity constraint.",
    ),
    "utilisation": (
        "It measures how fully {label} is using {capacity_note}.",
        "A rise means more of {capacity_note} is in use; consistently high values "
        "signal {label} is operating close to its ceiling.",
    ),
    "throughput": (
        "It measures how much of {activity} moved through {label} in the period.",
        "A rise signals faster throughput; a fall signals a bottleneck somewhere in "
        "{capacity_note}.",
    ),
    "supply_availability": (
        "It measures whether {label} had what it needed on hand to carry out "
        "{activity}.",
        "A worsening figure signals a supply-side constraint on {capacity_note}, "
        "which shows up as demand the business could not serve.",
    ),
    "service_level": (
        "It measures how reliably {label} delivers on {activity}.",
        "A fall signals {entity} are more often let down by {activity}; a sustained "
        "improvement signals a more dependable operation.",
    ),
    "availability": (
        "It measures how often {label}'s capacity for {activity} was actually "
        "available when needed.",
        "A fall signals lost capacity, which is usually the binding constraint on "
        "output for {label}.",
    ),
    "service_load": (
        "It measures the service burden generated by {activity}, for {label}.",
        "A rise usually follows more {unit_of_work}s, or more problems per "
        "{unit_of_work}, and predicts strain on support capacity ahead of it "
        "showing up elsewhere.",
    ),
    "customer_base": (
        "It measures the size of the base of {entity} that {activity} depends on, "
        "for {label}.",
        "A rise grows the base future demand is drawn from; a fall shrinks the "
        "foundation {activity} rests on.",
    ),
    "growth": (
        "It measures how fast new {entity} are being added, for {label}.",
        "A rise means acquisition is accelerating; a fall means the pipeline that "
        "feeds {activity} is slowing, ahead of that showing up in the base itself.",
    ),
    "unit_price": (
        "It measures the price realised per {unit_of_work}, for {label}.",
        "A rise reflects pricing power or a shift toward higher-value "
        "{unit_of_work}s; a fall reflects discounting or a shift toward lower-value "
        "ones.",
    ),
    "topline": (
        "It is the headline measure of demand for {label}, aggregating the value "
        "generated through {activity}.",
        "A rise means more value is flowing through {activity}; a fall means demand "
        "or pricing is weakening.",
    ),
    "demand_value": (
        "It measures the monetary value of demand that {activity} generated, for "
        "{label}.",
        "Movement here tracks how much {entity} are spending, before volume and "
        "price effects are separated out.",
    ),
    "demand_volume": (
        "It counts the volume of {activity} — how many {unit_of_work}s occurred — "
        "for {label}.",
        "A change signals more or fewer {entity} being served, independent of "
        "price or value per {unit_of_work}.",
    ),
    "cost": (
        "It is a direct cost incurred while carrying out {activity}, for {label}.",
        "A rise increases the cost base {label} carries for {activity}, before any "
        "margin is measured; a fall improves it, all else equal.",
    ),
    "spend": (
        "It is a controllable investment {label} makes toward {activity}.",
        "A rise is a deliberate input into {activity}, not itself good or bad — it "
        "should be read against the demand or growth it produces, not on its own.",
    ),
    "investment": (
        "It is a controllable investment {label} makes to grow {activity}.",
        "A rise is a deliberate input into {activity}, not itself good or bad — it "
        "should be read against the demand or growth it produces, not on its own.",
    ),
    "activity": (
        "It is a direct measure of {activity}, carried out by {label}.",
        "A rise means more {unit_of_work}s were handled in the period; a fall "
        "means fewer.",
    ),
    # semantic-type fallbacks — used when no domain-shaped tag applies (mostly
    # atomic candidates: a column not covered by the library or a derivation
    # rule, exposed directly).
    "money": (
        "It is a monetary amount recorded while {label} carries out {activity}.",
        "Track it alongside a volume measure to see whether a change is a value "
        "effect or a volume effect within {activity}.",
    ),
    "count": (
        "It is a count of events recorded while {label} carries out {activity}.",
        "A change reflects how much of {activity} occurred, in raw volume, for "
        "{label}.",
    ),
    "duration": (
        "It measures how long something takes within {activity}, for {label}, and "
        "is averaged rather than summed for that reason.",
        "A rise means {unit_of_work}s are taking longer, which usually eats into "
        "{capacity_note}.",
    ),
    "rate_pct": (
        "It is a rate already expressed as a percentage, recorded within {activity} "
        "for {label}.",
        "Treat any period-over-period change within {activity} with care: a rate "
        "should be recomputed from its components at a coarser grain, never "
        "averaged across periods.",
    ),
    "ratio": (
        "It is a ratio computed within {activity}, for {label}.",
        "A change reflects a shift in the relationship between its two components "
        "of {activity}, not a change in either one alone.",
    ),
    "score": (
        "It is a scored or rated measure recorded within {activity}, for {label}.",
        "A change reflects a shift in the underlying rating {entity} experience "
        "from {activity}, rather than a change in volume.",
    ),
    "flag": (
        "It is a yes/no outcome recorded while {label} carries out {activity}.",
        "A change reflects how often that outcome occurred during {activity}, in "
        "raw volume.",
    ),
}

# Priority order used when a KPI carries several semantic tags: the most
# business-specific characterisation wins, so a KPI tagged both
# `clinical_outcome` and `outcome_rate` reads primarily as a clinical measure.
TAG_PRIORITY: List[str] = [
    "clinical_outcome", "retention", "quality_defect", "outcome_rate",
    "profitability", "unit_economics", "capacity", "utilisation", "throughput",
    "efficiency", "supply_availability", "service_level", "availability",
    "service_load", "customer_base", "growth", "unit_price", "topline",
    "demand_value", "demand_volume", "cost", "spend", "investment", "activity",
    "money", "count", "duration", "rate_pct", "ratio", "score", "flag",
]

_GENERIC_FALLBACK = (
    "It is a {kind_desc} measure computed within {activity}, for {label}.",
    "{direction} is the favourable direction for this KPI; a move the other way "
    "is worth investigating within {activity}.",
)


def _field_phrase(source_fields: Sequence[str]) -> str:
    fields = [f"`{f}`" for f in source_fields if f]
    if not fields:
        return "this dataset's recorded fields"
    if len(fields) == 1:
        return fields[0]
    return ", ".join(fields[:-1]) + f" and {fields[-1]}"


def _pick_tag(semantic_tags: Sequence[str]) -> str:
    tags = set(semantic_tags)
    for tag in TAG_PRIORITY:
        if tag in tags:
            return tag
    return ""


def compose_explanation(name: str, kind: str, unit: str, semantic_tags: Sequence[str],
                        higher_is_better: bool, source_fields: Sequence[str],
                        ctx: DomainContext) -> Tuple[str, str]:
    """
    Build (`business_definition`, `relevance`) grounded in the detected domain
    and the KPI's own bound fields.

    Deterministic and LLM-free by design — this is what a dataset gets even
    with no API key configured, matching the product's no-key guarantee. The
    LLM screening path (`llm/prompts.KPI_DISCOVERY_SYSTEM`) is separately
    instructed to write to the same four questions using the same detected
    domain, and may replace this text for a candidate it actually reviewed.
    """
    tag = _pick_tag(semantic_tags)
    vocab = ctx.vocab
    kwargs = dict(label=vocab.label, entity=vocab.entity, activity=vocab.activity,
                 unit_of_work=vocab.unit_of_work, capacity_note=vocab.capacity_note)

    if tag:
        what_tpl, why_tpl = TAG_PHRASES[tag]
    else:
        kind_desc = {"sum": "summed", "mean": "averaged", "ratio": "ratio"}.get(kind, "computed")
        what_tpl, why_tpl = _GENERIC_FALLBACK
        kwargs["kind_desc"] = kind_desc
        kwargs["direction"] = "A rise" if higher_is_better else "A fall"

    what_sentence = what_tpl.format(**kwargs)
    why_sentence = why_tpl.format(**kwargs)

    field_phrase = _field_phrase(source_fields)
    lead = f"'{name}' is computed from {field_phrase} in this dataset."
    business_definition = f"{lead} {what_sentence}"
    relevance = why_sentence

    if ctx.is_uncertain:
        caveat = ("This dataset's specific industry could not be confidently "
                  "determined from its fields, so this explanation is necessarily "
                  "general rather than industry-specific. ")
        business_definition = caveat + business_definition

    return business_definition, relevance
