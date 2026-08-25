"""
The general KPI library — broadly applicable KPIs, as data.

The critical difference from the registry this replaces: a library entry binds
on **concept**, not on literal column name. `MetricSpec.requires = ["revenue",
"cost_of_goods"]` only ever matched a dataset that happened to use those exact
headers. A `ConceptRef` matches any of a set of aliases *and* checks that the
field's profiled semantic type agrees, so `net_sales`, `turnover` and
`billed_amount` all satisfy the revenue concept while a column called
`revenue_target` does not accidentally satisfy it better than the real one.

**A library KPI activates only when every required concept binds to a real
field.** An entry that cannot bind is reported with the reason, never
fabricated. That is the rule that keeps a hospital dataset from growing a
gross-margin tile.

Adding a domain is a data change: append a pack. No code moves.

The `general` and `retail_ecommerce` packs between them reproduce every entry of
the legacy `METRICS` dict, under the same ids, so the migration is value-for-value
identical on existing datasets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .profiling import DatasetProfile, FieldProfile, tokenise


# ---------------------------------------------------------------------------
# library model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ConceptRef:
    """One input a library KPI needs, expressed as a concept rather than a column."""

    concept: str
    aliases: Tuple[str, ...]
    semantic_type: Optional[str] = None      # required profiled semantic type
    additivity: Optional[str] = None         # required profiled additivity
    required: bool = True


@dataclass(frozen=True)
class LibraryKpi:
    id: str
    name: str
    domain: str
    definition: str
    concepts: Tuple[ConceptRef, ...]
    kind: str = "sum"                        # sum | mean | ratio
    expression: str = ""                     # for sum/mean, in {concept} terms
    numerator: str = ""                      # for ratio
    denominator: str = ""                    # for ratio
    scale: float = 1.0
    unit: str = "count"
    higher_is_better: bool = True
    default_time_grain: str = "month"
    semantic_tags: Tuple[str, ...] = ()
    relevance: str = ""


@dataclass
class ConceptBinding:
    concept: str
    field_name: str
    score: float
    alternatives: List[Tuple[str, float]] = field(default_factory=list)

    @property
    def ambiguous(self) -> bool:
        """More than one field matched this concept equally well."""
        return any(abs(s - self.score) < 1e-9 for _, s in self.alternatives)


@dataclass
class LibraryMatch:
    library: LibraryKpi
    bindings: Dict[str, ConceptBinding]

    @property
    def field_map(self) -> Dict[str, str]:
        return {c: b.field_name for c, b in self.bindings.items()}

    @property
    def source_fields(self) -> List[str]:
        seen: List[str] = []
        for b in self.bindings.values():
            if b.field_name not in seen:
                seen.append(b.field_name)
        return seen

    @property
    def ambiguous_concepts(self) -> List[str]:
        return [c for c, b in self.bindings.items() if b.ambiguous]


@dataclass
class LibraryMiss:
    library: LibraryKpi
    missing_concepts: List[str]

    @property
    def why(self) -> str:
        names = ", ".join(f"'{c}'" for c in self.missing_concepts)
        return f"no field in this dataset matched the concept(s) {names}"


# ---------------------------------------------------------------------------
# concept vocabulary — shared between packs so aliases stay consistent
# ---------------------------------------------------------------------------
C_REVENUE = ConceptRef("revenue",
                       ("revenue", "sales", "net_sales", "gross_revenue", "turnover",
                        "income", "billed_amount", "gmv", "total_revenue"),
                       semantic_type="money", additivity="flow")
C_COST = ConceptRef("cost",
                    ("cost_of_goods", "cogs", "cost", "total_cost", "operating_cost",
                     "direct_cost", "cost_of_sales"),
                    semantic_type="money", additivity="flow")
C_MARKETING = ConceptRef("marketing_spend",
                         ("marketing_spend", "marketing_cost", "ad_spend", "advertising_spend",
                          "campaign_spend", "media_spend"),
                         semantic_type="money", additivity="flow")
C_ORDERS = ConceptRef("orders",
                      ("orders", "order_count", "transactions", "purchases", "bookings"),
                      semantic_type="count", additivity="flow")
C_CUSTOMERS = ConceptRef("customers",
                         ("customers", "customer_count", "buyers", "accounts", "clients"),
                         semantic_type="count", additivity="flow")
C_UNITS = ConceptRef("units_sold",
                     ("units_sold", "units", "quantity", "qty", "volume", "items_sold"),
                     semantic_type="count", additivity="flow")
C_FULFILLED = ConceptRef("fulfilled_orders",
                         ("fulfilled_orders", "shipped_orders", "delivered_orders",
                          "completed_orders"),
                         semantic_type="count", additivity="flow")
C_STOCKOUTS = ConceptRef("stockout_events",
                         ("stockout_events", "stockouts", "out_of_stock", "oos_events"),
                         semantic_type="count", additivity="flow")
C_INVENTORY = ConceptRef("inventory_units",
                         ("inventory_units", "inventory", "stock_on_hand", "on_hand", "stock_level"),
                         semantic_type="count", additivity="stock")
C_RETURNS = ConceptRef("returns", ("returns", "returned_orders", "refunds", "returned_units"),
                       semantic_type="count", additivity="flow")
C_TICKETS = ConceptRef("support_tickets",
                       ("support_tickets", "tickets", "support_contacts", "cases", "complaints"),
                       semantic_type="count", additivity="flow")

# healthcare
C_ADMISSIONS = ConceptRef("admissions", ("admissions", "admitted", "admits", "patient_admissions"),
                          semantic_type="count", additivity="flow")
C_DISCHARGES = ConceptRef("discharges", ("discharges", "discharged", "patient_discharges"),
                          semantic_type="count", additivity="flow")
C_RECOVERED = ConceptRef("recovered", ("recovered_patients", "recovered", "recoveries", "cured"),
                         semantic_type="count", additivity="flow")
C_READMISSIONS = ConceptRef("readmissions", ("readmissions", "readmitted", "readmission_count"),
                            semantic_type="count", additivity="flow")
C_DEATHS = ConceptRef("deaths", ("deaths", "mortalities", "deceased", "fatalities"),
                      semantic_type="count", additivity="flow")
C_BED_DAYS = ConceptRef("bed_days", ("bed_days", "patient_days", "occupied_bed_days"),
                        semantic_type="count", additivity="flow")
C_LOS = ConceptRef("length_of_stay", ("length_of_stay_days", "length_of_stay", "los", "stay_days"),
                   semantic_type="duration")
C_BEDS_AVAILABLE = ConceptRef("beds_available", ("beds_available", "available_beds", "bed_capacity",
                                                 "capacity", "total_beds"),
                              semantic_type="count", additivity="stock")
C_TREATMENT_COST = ConceptRef("treatment_cost", ("treatment_cost", "care_cost", "clinical_cost",
                                                 "cost_of_care"),
                              semantic_type="money", additivity="flow")

# logistics
C_SHIPMENTS = ConceptRef("shipments", ("shipments", "deliveries", "consignments", "loads"),
                         semantic_type="count", additivity="flow")
C_ON_TIME = ConceptRef("on_time_deliveries", ("on_time_deliveries", "on_time", "ontime_deliveries"),
                       semantic_type="count", additivity="flow")
C_DAMAGED = ConceptRef("damaged_shipments", ("damaged_shipments", "damaged", "damages"),
                       semantic_type="count", additivity="flow")
C_FREIGHT = ConceptRef("freight_cost", ("freight_cost", "shipping_cost", "transport_cost",
                                        "delivery_cost", "logistics_cost"),
                       semantic_type="money", additivity="flow")

# subscription / saas
C_SUBSCRIBERS = ConceptRef("subscribers", ("subscribers", "active_subscribers", "active_users",
                                           "seats", "subscriptions"),
                           semantic_type="count", additivity="flow")
C_CHURNED = ConceptRef("churned", ("churned", "churned_customers", "cancellations",
                                   "cancelled_subscriptions", "churn_count"),
                       semantic_type="count", additivity="flow")
C_NEW_CUSTOMERS = ConceptRef("new_customers", ("new_customers", "signups", "new_subscribers",
                                               "acquisitions", "activations"),
                             semantic_type="count", additivity="flow")
C_RECURRING = ConceptRef("recurring_revenue", ("mrr", "arr", "recurring_revenue",
                                               "subscription_revenue"),
                         semantic_type="money", additivity="flow")

# manufacturing
C_PRODUCED = ConceptRef("units_produced", ("units_produced", "produced", "output_units", "production"),
                        semantic_type="count", additivity="flow")
C_DEFECTS = ConceptRef("defects", ("defects", "defective_units", "rejects", "scrap_units"),
                       semantic_type="count", additivity="flow")
C_DOWNTIME = ConceptRef("downtime", ("downtime_hours", "downtime", "unplanned_downtime"),
                        semantic_type="duration")


# ---------------------------------------------------------------------------
# the packs
# ---------------------------------------------------------------------------
GENERAL: List[LibraryKpi] = [
    LibraryKpi("revenue", "Revenue", "general",
               "Total sales value recognised in the period.",
               (C_REVENUE,), kind="sum", expression="{revenue}",
               unit="currency", semantic_tags=("topline", "demand_value"),
               relevance="The headline measure of commercial output."),
    LibraryKpi("cost_of_goods", "Cost of goods", "general",
               "Direct cost of what was delivered in the period.",
               (C_COST,), kind="sum", expression="{cost}",
               unit="currency", higher_is_better=False, semantic_tags=("cost",),
               relevance="The direct cost base that margin is measured against."),
    LibraryKpi("gross_profit", "Gross profit", "general",
               "Revenue less the direct cost of what was delivered.",
               (C_REVENUE, C_COST), kind="sum", expression="{revenue} - {cost}",
               unit="currency", semantic_tags=("profitability",),
               relevance="What the business actually keeps before overheads."),
    LibraryKpi("gross_margin_pct", "Gross margin %", "general",
               "Gross profit as a share of revenue.",
               (C_REVENUE, C_COST), kind="ratio",
               numerator="{revenue} - {cost}", denominator="{revenue}", scale=100.0,
               unit="percent", semantic_tags=("profitability", "efficiency"),
               relevance="Separates a revenue problem from a pricing or cost problem."),
    LibraryKpi("orders", "Orders", "general",
               "Number of orders placed in the period.",
               (C_ORDERS,), kind="sum", expression="{orders}",
               unit="count", semantic_tags=("demand_volume",),
               relevance="Demand measured in transactions rather than value."),
    LibraryKpi("customers", "Customers", "general",
               "Number of purchasing customers in the period.",
               (C_CUSTOMERS,), kind="sum", expression="{customers}",
               unit="count", semantic_tags=("demand_volume", "customer_base"),
               relevance="Whether demand is moving because of how many buy, or how much they buy."),
    LibraryKpi("units_sold", "Units sold", "general",
               "Total unit volume sold in the period.",
               (C_UNITS,), kind="sum", expression="{units_sold}",
               unit="count", semantic_tags=("demand_volume",),
               relevance="Volume, which separates price effects from quantity effects."),
    LibraryKpi("marketing_spend", "Marketing spend", "general",
               "Marketing investment made in the period.",
               (C_MARKETING,), kind="sum", expression="{marketing_spend}",
               unit="currency", higher_is_better=False, semantic_tags=("spend", "investment"),
               relevance="The controllable input most often used to move demand."),
    LibraryKpi("avg_order_value", "Average order value", "general",
               "Revenue per order.",
               (C_REVENUE, C_ORDERS), kind="ratio",
               numerator="{revenue}", denominator="{orders}",
               unit="currency", semantic_tags=("unit_economics",),
               relevance="Whether each transaction is getting larger or smaller."),
    LibraryKpi("avg_selling_price", "Average selling price", "general",
               "Realised revenue per unit sold.",
               (C_REVENUE, C_UNITS), kind="ratio",
               numerator="{revenue}", denominator="{units_sold}",
               unit="currency", semantic_tags=("unit_price", "unit_economics"),
               relevance="Separates a price effect from a volume effect."),
    LibraryKpi("units_per_order", "Units per order", "general",
               "Average basket size in units.",
               (C_UNITS, C_ORDERS), kind="ratio",
               numerator="{units_sold}", denominator="{orders}",
               unit="ratio", semantic_tags=("unit_economics",),
               relevance="Whether customers are buying more per transaction."),
    LibraryKpi("customer_acquisition_cost", "Customer acquisition cost", "general",
               "Marketing spend per purchasing customer.",
               (C_MARKETING, C_CUSTOMERS), kind="ratio",
               numerator="{marketing_spend}", denominator="{customers}",
               unit="currency", higher_is_better=False, semantic_tags=("efficiency", "spend"),
               relevance="Whether growth is being bought efficiently."),
]

RETAIL_ECOMMERCE: List[LibraryKpi] = [
    LibraryKpi("stockout_events", "Stockout events", "retail_ecommerce",
               "Occasions demand could not be served from stock.",
               (C_STOCKOUTS,), kind="sum", expression="{stockout_events}",
               unit="count", higher_is_better=False, semantic_tags=("supply_availability",),
               relevance="A direct supply-side constraint on what could be sold."),
    LibraryKpi("inventory_units", "Inventory level", "retail_ecommerce",
               "Average on-hand inventory across the period. A stock, not a flow.",
               (C_INVENTORY,), kind="mean", expression="{inventory_units}",
               unit="count", semantic_tags=("supply_availability",),
               relevance="Whether the business had the goods to sell."),
    LibraryKpi("returns", "Returns", "retail_ecommerce",
               "Returned orders in the period.",
               (C_RETURNS,), kind="sum", expression="{returns}",
               unit="count", higher_is_better=False, semantic_tags=("quality_defect",),
               relevance="A quality and expectation-mismatch signal."),
    LibraryKpi("support_tickets", "Support tickets", "retail_ecommerce",
               "Inbound support contacts in the period.",
               (C_TICKETS,), kind="sum", expression="{support_tickets}",
               unit="count", higher_is_better=False, semantic_tags=("service_load",),
               relevance="Service burden, and an early quality signal."),
    LibraryKpi("fulfillment_rate", "Fulfilment rate", "retail_ecommerce",
               "Share of orders that were fulfilled.",
               (C_FULFILLED, C_ORDERS), kind="ratio",
               numerator="{fulfilled_orders}", denominator="{orders}", scale=100.0,
               unit="percent", semantic_tags=("supply_availability", "outcome_rate"),
               relevance="Whether the business could serve the demand it had."),
    LibraryKpi("stockout_rate", "Stockout rate", "retail_ecommerce",
               "Stockout events per order.",
               (C_STOCKOUTS, C_ORDERS), kind="ratio",
               numerator="{stockout_events}", denominator="{orders}", scale=100.0,
               unit="percent", higher_is_better=False,
               semantic_tags=("supply_availability", "outcome_rate"),
               relevance="Supply failure normalised by demand volume."),
    LibraryKpi("return_rate", "Return rate", "retail_ecommerce",
               "Returns per order.",
               (C_RETURNS, C_ORDERS), kind="ratio",
               numerator="{returns}", denominator="{orders}", scale=100.0,
               unit="percent", higher_is_better=False,
               semantic_tags=("quality_defect", "outcome_rate"),
               relevance="Quality normalised by volume, so it is comparable across periods."),
    LibraryKpi("tickets_per_1k_orders", "Support tickets per 1k orders", "retail_ecommerce",
               "Support contacts per thousand orders.",
               (C_TICKETS, C_ORDERS), kind="ratio",
               numerator="{support_tickets}", denominator="{orders}", scale=1000.0,
               unit="ratio", higher_is_better=False, semantic_tags=("service_load", "outcome_rate"),
               relevance="Service load normalised by volume."),
]

HEALTHCARE: List[LibraryKpi] = [
    LibraryKpi("admissions", "Admissions", "healthcare",
               "Patients admitted in the period.",
               (C_ADMISSIONS,), kind="sum", expression="{admissions}",
               unit="count", semantic_tags=("demand_volume", "activity"),
               relevance="The primary measure of clinical activity and demand."),
    LibraryKpi("discharges", "Discharges", "healthcare",
               "Patients discharged in the period.",
               (C_DISCHARGES,), kind="sum", expression="{discharges}",
               unit="count", semantic_tags=("activity", "throughput"),
               relevance="Throughput — whether patients are moving through the system."),
    LibraryKpi("recovery_rate", "Recovery rate", "healthcare",
               "Share of discharged patients recorded as recovered.",
               (C_RECOVERED, C_DISCHARGES), kind="ratio",
               numerator="{recovered}", denominator="{discharges}", scale=100.0,
               unit="percent", semantic_tags=("outcome_rate", "clinical_outcome"),
               relevance="The core clinical outcome measure."),
    LibraryKpi("readmission_rate", "Readmission rate", "healthcare",
               "Share of discharges readmitted within the reporting window.",
               (C_READMISSIONS, C_DISCHARGES), kind="ratio",
               numerator="{readmissions}", denominator="{discharges}", scale=100.0,
               unit="percent", higher_is_better=False,
               semantic_tags=("outcome_rate", "clinical_outcome", "quality_defect"),
               relevance="A standard quality-of-care indicator and a cost driver."),
    LibraryKpi("mortality_rate", "Mortality rate", "healthcare",
               "Deaths as a share of admissions.",
               (C_DEATHS, C_ADMISSIONS), kind="ratio",
               numerator="{deaths}", denominator="{admissions}", scale=100.0,
               unit="percent", higher_is_better=False,
               semantic_tags=("outcome_rate", "clinical_outcome"),
               relevance="The most serious clinical outcome measure."),
    LibraryKpi("avg_length_of_stay", "Average length of stay", "healthcare",
               "Mean days a patient remains admitted.",
               (C_LOS,), kind="mean", expression="{length_of_stay}",
               unit="duration", higher_is_better=False,
               semantic_tags=("throughput", "efficiency"),
               relevance="Drives both capacity and cost per patient."),
    LibraryKpi("bed_days", "Bed days", "healthcare",
               "Total occupied bed-days in the period.",
               (C_BED_DAYS,), kind="sum", expression="{bed_days}",
               unit="count", semantic_tags=("capacity", "activity"),
               relevance="The capacity actually consumed."),
    LibraryKpi("bed_occupancy_rate", "Bed occupancy rate", "healthcare",
               "Occupied beds as a share of available beds.",
               (C_BED_DAYS, C_BEDS_AVAILABLE), kind="ratio",
               numerator="{bed_days}", denominator="{beds_available}", scale=100.0,
               unit="percent", semantic_tags=("capacity", "utilisation"),
               relevance="Whether capacity is the binding constraint."),
    LibraryKpi("cost_per_admission", "Cost per admission", "healthcare",
               "Treatment cost per admitted patient.",
               (C_TREATMENT_COST, C_ADMISSIONS), kind="ratio",
               numerator="{treatment_cost}", denominator="{admissions}",
               unit="currency", higher_is_better=False, semantic_tags=("unit_economics", "cost"),
               relevance="Unit economics of care delivery."),
]

LOGISTICS: List[LibraryKpi] = [
    LibraryKpi("shipments", "Shipments", "logistics",
               "Shipments dispatched in the period.",
               (C_SHIPMENTS,), kind="sum", expression="{shipments}",
               unit="count", semantic_tags=("activity", "demand_volume"),
               relevance="The volume of work the network handled."),
    LibraryKpi("on_time_delivery_rate", "On-time delivery rate", "logistics",
               "Share of shipments delivered on time.",
               (C_ON_TIME, C_SHIPMENTS), kind="ratio",
               numerator="{on_time_deliveries}", denominator="{shipments}", scale=100.0,
               unit="percent", semantic_tags=("outcome_rate", "service_level"),
               relevance="The headline service-level measure in logistics."),
    LibraryKpi("damage_rate", "Damage rate", "logistics",
               "Share of shipments damaged in transit.",
               (C_DAMAGED, C_SHIPMENTS), kind="ratio",
               numerator="{damaged_shipments}", denominator="{shipments}", scale=100.0,
               unit="percent", higher_is_better=False,
               semantic_tags=("outcome_rate", "quality_defect"),
               relevance="A quality signal that drives both cost and churn."),
    LibraryKpi("cost_per_shipment", "Cost per shipment", "logistics",
               "Freight cost per shipment.",
               (C_FREIGHT, C_SHIPMENTS), kind="ratio",
               numerator="{freight_cost}", denominator="{shipments}",
               unit="currency", higher_is_better=False, semantic_tags=("unit_economics", "cost"),
               relevance="Unit economics of the delivery network."),
]

SUBSCRIPTION: List[LibraryKpi] = [
    LibraryKpi("active_subscribers", "Active subscribers", "saas_subscription",
               "Subscribers active in the period.",
               (C_SUBSCRIBERS,), kind="sum", expression="{subscribers}",
               unit="count", semantic_tags=("customer_base",),
               relevance="The installed base recurring revenue rests on."),
    LibraryKpi("recurring_revenue", "Recurring revenue", "saas_subscription",
               "Contracted recurring revenue in the period.",
               (C_RECURRING,), kind="sum", expression="{recurring_revenue}",
               unit="currency", semantic_tags=("topline", "demand_value"),
               relevance="The predictable component of revenue."),
    LibraryKpi("churn_rate", "Churn rate", "saas_subscription",
               "Share of subscribers lost in the period.",
               (C_CHURNED, C_SUBSCRIBERS), kind="ratio",
               numerator="{churned}", denominator="{subscribers}", scale=100.0,
               unit="percent", higher_is_better=False,
               semantic_tags=("outcome_rate", "retention"),
               relevance="The single strongest predictor of subscription trajectory."),
    LibraryKpi("new_customers", "New customers", "saas_subscription",
               "Customers acquired in the period.",
               (C_NEW_CUSTOMERS,), kind="sum", expression="{new_customers}",
               unit="count", semantic_tags=("growth", "customer_base"),
               relevance="Gross acquisition, before churn is netted off."),
]

MANUFACTURING: List[LibraryKpi] = [
    LibraryKpi("units_produced", "Units produced", "manufacturing",
               "Units manufactured in the period.",
               (C_PRODUCED,), kind="sum", expression="{units_produced}",
               unit="count", semantic_tags=("activity", "throughput"),
               relevance="Production output."),
    LibraryKpi("defect_rate", "Defect rate", "manufacturing",
               "Defective units as a share of units produced.",
               (C_DEFECTS, C_PRODUCED), kind="ratio",
               numerator="{defects}", denominator="{units_produced}", scale=100.0,
               unit="percent", higher_is_better=False,
               semantic_tags=("outcome_rate", "quality_defect"),
               relevance="The core manufacturing quality measure."),
    LibraryKpi("downtime_hours", "Downtime", "manufacturing",
               "Mean unplanned downtime per period.",
               (C_DOWNTIME,), kind="mean", expression="{downtime}",
               unit="duration", higher_is_better=False,
               semantic_tags=("availability", "supply_availability"),
               relevance="Lost capacity, and usually the binding constraint on output."),
]

PACKS: Dict[str, List[LibraryKpi]] = {
    "general": GENERAL,
    "retail_ecommerce": RETAIL_ECOMMERCE,
    "healthcare": HEALTHCARE,
    "logistics": LOGISTICS,
    "saas_subscription": SUBSCRIPTION,
    "manufacturing": MANUFACTURING,
}

ALL_LIBRARY_KPIS: List[LibraryKpi] = [k for pack in PACKS.values() for k in pack]
LIBRARY_BY_ID: Dict[str, LibraryKpi] = {k.id: k for k in ALL_LIBRARY_KPIS}


# ---------------------------------------------------------------------------
# binding
# ---------------------------------------------------------------------------
# Below this a "match" is really a coincidence of shared words.
MIN_BIND_SCORE = 0.6


def _alias_score(alias: str, profile: FieldProfile) -> float:
    """How well a field name satisfies one alias. 0 means no match."""
    alias_norm = alias.strip().lower()
    if profile.name.lower() == alias_norm:
        return 1.0
    alias_tokens = tokenise(alias_norm)
    field_tokens = profile.name_tokens
    if not alias_tokens:
        return 0.0
    if alias_tokens == field_tokens:
        return 0.95
    if set(alias_tokens) <= set(field_tokens):
        # A longer field name containing the alias is a weaker match:
        # `revenue` matches `revenue_target` less well than `revenue`.
        extra = len(field_tokens) - len(alias_tokens)
        return max(0.6, 0.85 - 0.08 * extra)
    return 0.0


def bind_concept(concept: ConceptRef, candidates: Sequence[FieldProfile]) -> Optional[ConceptBinding]:
    """Best field for a concept, with equally-good alternatives recorded."""
    scored: List[Tuple[str, float]] = []
    for prof in candidates:
        if concept.semantic_type and prof.semantic_type != concept.semantic_type:
            continue
        if concept.additivity and prof.additivity != concept.additivity:
            continue
        best = max((_alias_score(a, prof) for a in concept.aliases), default=0.0)
        if best >= MIN_BIND_SCORE:
            scored.append((prof.name, best))
    if not scored:
        return None
    scored.sort(key=lambda t: (-t[1], t[0]))
    winner, score = scored[0]
    return ConceptBinding(concept=concept.concept, field_name=winner, score=score,
                          alternatives=scored[1:])


def bind_library(profile: DatasetProfile,
                 packs: Optional[Sequence[str]] = None
                 ) -> Tuple[List[LibraryMatch], List[LibraryMiss]]:
    """
    Bind every library KPI against a profiled dataset.

    Returns the entries this dataset can actually support, and the entries it
    cannot with the concept that failed to bind. Nothing is ever invented to
    make an entry fit.
    """
    measures = [f for f in profile.fields if f.role == "measure"]
    entries = [k for name in (packs or PACKS.keys()) for k in PACKS.get(name, [])]

    matched: List[LibraryMatch] = []
    missed: List[LibraryMiss] = []
    for entry in entries:
        bindings: Dict[str, ConceptBinding] = {}
        missing: List[str] = []
        for concept in entry.concepts:
            bound = bind_concept(concept, measures)
            if bound is None:
                if concept.required:
                    missing.append(concept.concept)
            else:
                bindings[concept.concept] = bound
        if missing:
            missed.append(LibraryMiss(library=entry, missing_concepts=missing))
        else:
            matched.append(LibraryMatch(library=entry, bindings=bindings))
    return matched, missed


def detect_domains(matches: Sequence[LibraryMatch], min_matches: int = 2) -> List[str]:
    """Which domain packs this dataset plausibly belongs to, by how much bound."""
    counts: Dict[str, int] = {}
    for m in matches:
        counts[m.library.domain] = counts.get(m.library.domain, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [d for d, n in ranked if n >= min_matches or d == "general"]


def resolve_expression(expression: str, field_map: Dict[str, str]) -> str:
    """Rewrite a `{concept}` expression into a `{field}` expression."""
    out = expression
    for concept, field_name in field_map.items():
        out = out.replace("{" + concept + "}", "{" + field_name + "}")
    return out


def display_name(match: LibraryMatch) -> str:
    """
    What to call a bound KPI.

    A library entry's name is written for the concept, not for the column that
    happened to satisfy it. When a single-input KPI binds to a field the entry
    does not actually name — a hospital's `treatment_cost` satisfying the
    general `cost` concept — the entry's label ("Cost of goods") is simply wrong
    for that business, so the column's own name is used instead.

    An exact alias match keeps the library's wording, which is usually better:
    `inventory_units` stays "Inventory level" rather than becoming "Inventory
    units".
    """
    entry = match.library
    if len(entry.concepts) != 1 or entry.kind == "ratio":
        return entry.name
    concept = entry.concepts[0]
    field_name = match.bindings[concept.concept].field_name
    aliases = {a.strip().lower() for a in concept.aliases}
    if field_name.lower() in aliases:
        return entry.name
    return field_name.replace("_", " ").strip().capitalize()
