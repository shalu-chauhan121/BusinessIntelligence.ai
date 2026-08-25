"""
The KPI Definition and KPI Contract layer.

The first test is the one that matters most: the contract must reproduce, value
for value, every number the hard-coded registry produced. If that holds, making
the contract the source of truth changed no number anywhere in the product.

The rest establish the properties the layer exists to guarantee — a KPI is never
fabricated, a rate is never proposed without evidence that it IS a rate, a
granularity is never silently assumed, and an ambiguity is never silently
resolved.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

import pandas as pd

from app.engines.metrics import compute as legacy_compute
from app.engines.metrics import detect_schema, prepare
from app.kpi import service as kpi_service
from app.kpi.contract import FormulaSpec, KpiContract
from app.kpi.conflicts import detect_conflicts
from app.kpi.derivation import build_atomic_candidates, build_derived_candidates, library_signatures
from app.kpi.library import bind_library
from app.kpi.profiling import profile_dataset
from app.kpi.resolver import (
    KpiResolutionError,
    compile_contract,
    compile_kpi,
    evaluate,
    parse_expression,
    referenced_fields,
)
from app.kpi.screening import ProposedKpi, _validate_proposal, screen_candidates
from app.kpi.service import KpiContractError
from app.kpi.domain import DOMAIN_VOCAB, detect_domain_context
from app.kpi.explanation import compose_explanation

from .base import ROOT, EngineTestCase, setup_environment

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SAMPLE_CSV = ROOT / "sample_data" / "business_metrics_sample.csv"


def _profile_of(df_raw: pd.DataFrame):
    schema = detect_schema(df_raw)
    df = prepare(df_raw, schema)
    profile = profile_dataset(df, schema.date_column, schema.dimensions,
                              schema.base_metrics + schema.extra_metrics)
    return df, schema, profile


class KpiContractTestCase(EngineTestCase):
    """Shared fixture: the sample company, plus a synthetic hospital dataset."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.dataset_stub = {"_id": cls.dataset["_id"], "filename": "sample.csv"}
        cls.hospital_raw = pd.read_csv(FIXTURES / "hospital_sample.csv")


# ---------------------------------------------------------------------------
# migration safety — the test that must pass first
# ---------------------------------------------------------------------------
class TestMigrationParity(KpiContractTestCase):
    def test_contract_reproduces_every_legacy_kpi_value(self):
        """
        The bootstrap contract must compute what the registry computed, for every
        KPI, in every quarter. This is what makes the source-of-truth swap safe.
        """
        legacy_keys = list(self.schema.available_kpis)
        contract = kpi_service.bootstrap("u", self.dataset_stub, self.df, self.schema)
        resolver = compile_contract(contract)

        missing = [k for k in legacy_keys if k not in resolver]
        self.assertEqual(missing, [], f"the contract dropped KPIs the registry had: {missing}")

        compared = 0
        for key in legacy_keys:
            for _, group in self.df.groupby(["_year", "_quarter"]):
                expected = legacy_compute(group, key)
                actual = resolver[key].compute(group)
                if expected != expected and actual != actual:      # both NaN
                    continue
                self.assertClose(actual, expected, rel=1e-9,
                                 msg=f"{key} diverged from the registry")
                compared += 1
        self.assertGreater(compared, 200, "the parity check did not actually compare much")

    def test_bootstrap_contract_is_provisional_and_immediately_usable(self):
        contract = kpi_service.bootstrap("u", self.dataset_stub, self.df, self.schema)
        self.assertEqual(contract.status, "provisional")
        self.assertTrue(all(k.status == "approved" for k in contract.kpis),
                        "a provisional contract must work without anyone reviewing it")
        self.assertTrue(compile_contract(contract))


# ---------------------------------------------------------------------------
# profiling
# ---------------------------------------------------------------------------
class TestProfiling(KpiContractTestCase):
    def test_semantic_types_and_additivity_on_the_sample(self):
        profile = profile_dataset(self.df, self.schema.date_column, self.schema.dimensions,
                                  self.schema.base_metrics + self.schema.extra_metrics)
        self.assertEqual(profile.get("revenue").semantic_type, "money")
        self.assertEqual(profile.get("revenue").additivity, "flow")
        self.assertEqual(profile.get("orders").semantic_type, "count")

    def test_inventory_is_a_stock_and_is_averaged_not_summed(self):
        """Twelve month-end inventory levels do not add up to an annual level."""
        profile = profile_dataset(self.df, self.schema.date_column, self.schema.dimensions,
                                  self.schema.base_metrics + self.schema.extra_metrics)
        inventory = profile.get("inventory_units")
        self.assertEqual(inventory.additivity, "stock")
        self.assertEqual(inventory.default_aggregation, "mean")

    def test_a_time_unit_alone_does_not_make_a_duration(self):
        """`bed_days` is a summable quantity; `length_of_stay_days` is a duration."""
        _, _, profile = _profile_of(self.hospital_raw)
        self.assertEqual(profile.get("bed_days").semantic_type, "count")
        self.assertEqual(profile.get("bed_days").additivity, "flow")
        self.assertEqual(profile.get("length_of_stay_days").semantic_type, "duration")

    def test_subset_relations_are_only_found_within_a_semantic_family(self):
        """`orders <= revenue` is arithmetically true and semantically meaningless."""
        profile = profile_dataset(self.df, self.schema.date_column, self.schema.dimensions,
                                  self.schema.base_metrics + self.schema.extra_metrics)
        self.assertIn("orders", profile.get("fulfilled_orders").subset_of)
        self.assertIn("revenue", profile.get("cost_of_goods").subset_of)
        self.assertNotIn("revenue", profile.get("orders").subset_of)

    def test_row_grain_is_detected(self):
        profile = profile_dataset(self.df, self.schema.date_column, self.schema.dimensions,
                                  self.schema.base_metrics + self.schema.extra_metrics)
        self.assertTrue(profile.row_grain_is_unique)
        self.assertIn("date", profile.row_grain)
        for dim in ("region", "product", "channel", "segment"):
            self.assertIn(dim, profile.row_grain)

    def test_orthogonal_dimensions_are_not_reported_as_broken_hierarchies(self):
        profile = profile_dataset(self.df, self.schema.date_column, self.schema.dimensions,
                                  self.schema.base_metrics + self.schema.extra_metrics)
        self.assertEqual(profile.hierarchies, [],
                         "region and product are independent, not a damaged hierarchy")

    def test_a_real_hierarchy_is_detected_and_a_broken_one_is_flagged(self):
        clean = pd.DataFrame({
            "date": ["2024-01-01"] * 4,
            "store": ["S1", "S2", "S3", "S4"],
            "country": ["UK", "UK", "FR", "FR"],
            "revenue": [10, 20, 30, 40],
        })
        _, _, profile = _profile_of(clean)
        pairs = {(h.child, h.parent): h for h in profile.hierarchies}
        self.assertIn(("store", "country"), pairs)
        self.assertTrue(pairs[("store", "country")].clean)

        broken = clean.copy()
        broken.loc[3, "store"] = "S1"          # S1 now sits under both UK and FR
        _, _, profile = _profile_of(broken)
        pairs = {(h.child, h.parent): h for h in profile.hierarchies}
        self.assertIn(("store", "country"), pairs)
        self.assertGreater(pairs[("store", "country")].violations, 0)


# ---------------------------------------------------------------------------
# the general library — never fabricate
# ---------------------------------------------------------------------------
class TestLibraryBinding(KpiContractTestCase):
    def test_a_revenue_only_dataset_never_fabricates_margin_or_aov(self):
        thin = pd.DataFrame({
            "date": pd.date_range("2023-01-02", periods=120, freq="W-MON"),
            "revenue": range(100, 220),
        })
        df, schema, profile = _profile_of(thin)
        matched, missed = bind_library(profile)
        bound = {m.library.id for m in matched}

        self.assertIn("revenue", bound)
        for fabricated in ("gross_margin_pct", "gross_profit", "avg_order_value",
                           "customer_acquisition_cost"):
            self.assertNotIn(fabricated, bound,
                             f"{fabricated} was invented without the fields to compute it")

        why = {m.library.id: m.why for m in missed}
        self.assertIn("gross_margin_pct", why)
        self.assertIn("cost", why["gross_margin_pct"])

    def test_a_hospital_dataset_gets_clinical_kpis_and_no_retail_dashboard(self):
        df, schema, profile = _profile_of(self.hospital_raw)
        matched, _ = bind_library(profile)
        bound = {m.library.id for m in matched}

        for clinical in ("admissions", "discharges", "recovery_rate",
                         "readmission_rate", "mortality_rate", "avg_length_of_stay"):
            self.assertIn(clinical, bound, f"{clinical} should be discoverable here")
        for retail in ("gross_margin_pct", "avg_order_value", "stockout_rate",
                       "fulfillment_rate", "return_rate"):
            self.assertNotIn(retail, bound, f"{retail} has no business on a hospital dataset")

    def test_concepts_bind_on_aliases_not_literal_column_names(self):
        aliased = pd.DataFrame({
            "date": pd.date_range("2023-01-02", periods=60, freq="W-MON"),
            "net_sales": range(1000, 1060),
            "cogs": range(400, 460),
        })
        _, _, profile = _profile_of(aliased)
        matched, _ = bind_library(profile)
        by_id = {m.library.id: m for m in matched}
        self.assertIn("gross_margin_pct", by_id)
        self.assertEqual(by_id["gross_margin_pct"].field_map["revenue"], "net_sales")
        self.assertEqual(by_id["gross_margin_pct"].field_map["cost"], "cogs")


# ---------------------------------------------------------------------------
# derivation — computable is not the same as meaningful
# ---------------------------------------------------------------------------
class TestDerivation(KpiContractTestCase):
    def _candidates(self, df_raw):
        df, schema, profile = _profile_of(df_raw)
        matched, _ = bind_library(profile)
        derived, rejected = build_derived_candidates(
            profile, exclude=library_signatures(matched))
        return profile, derived, rejected

    def test_revenue_over_cost_is_never_a_raw_quotient(self):
        """
        Cost is contained by revenue, so the meaningful KPI for that pair is a
        margin. The bare quotient revenue/cost is a coverage multiple and must
        never be offered as a KPI.
        """
        _, derived, _ = self._candidates(pd.read_csv(SAMPLE_CSV))
        ratios = {(c.numerator, c.denominator) for c in derived if c.kind == "ratio"}
        self.assertNotIn(("{revenue}", "{cost_of_goods}"), ratios,
                         "revenue/cost is a coverage multiple, not a KPI")

    def test_money_over_money_without_containment_is_rejected_with_a_reason(self):
        """A cost that is NOT part of the revenue line cannot form a margin."""
        frame = pd.DataFrame({
            "date": pd.date_range("2023-01-02", periods=60, freq="W-MON"),
            # They cross over, so neither is contained by the other on the rows.
            "revenue": [100.0, 900.0] * 30,
            "litigation_expense": [900.0, 100.0] * 30,
        })
        _, _, rejected = self._candidates(frame)
        self.assertTrue(
            any(r.rule == "money_over_money" and "litigation_expense" in r.expression
                for r in rejected),
            "the system should say it considered this pair and declined")

    def test_a_rate_requires_containment_in_the_actual_rows(self):
        """The name suggests a rate; the data decides whether it is one."""
        violating = pd.DataFrame({
            "date": pd.date_range("2023-01-02", periods=60, freq="W-MON"),
            "orders": [10] * 60,
            "returns": [25] * 60,          # more returns than orders — impossible
        })
        _, derived, _ = self._candidates(violating)
        rate_ids = {c.candidate_id for c in derived if c.kind == "ratio"}
        self.assertNotIn("returns_per_orders_rate", rate_ids,
                         "returns are not contained by orders here, so this is not a rate")

    def test_a_rate_is_proposed_when_containment_does_hold(self):
        """A pair the shipped library knows nothing about still yields its rate."""
        holding = pd.DataFrame({
            "date": pd.date_range("2023-01-02", periods=60, freq="W-MON"),
            "visits": [100] * 60,
            "failed_inspections": [7] * 60,
        })
        _, derived, _ = self._candidates(holding)
        rates = {(c.numerator, c.denominator) for c in derived if c.kind == "ratio"}
        self.assertIn(("{failed_inspections}", "{visits}"), rates)

    def test_a_derived_rate_does_not_duplicate_a_library_kpi(self):
        """Return rate is already in the library, so the rule must not repeat it."""
        _, derived, _ = self._candidates(pd.read_csv(SAMPLE_CSV))
        self.assertNotIn(("{returns}", "{orders}"),
                         {(c.numerator, c.denominator) for c in derived})

    def test_a_denominator_must_be_a_population(self):
        """Stockouts per support ticket normalises against an unrelated measure."""
        _, derived, _ = self._candidates(
            pd.read_csv(SAMPLE_CSV))
        for cand in derived:
            self.assertNotEqual(cand.denominator, "{support_tickets}")

    def test_one_numerator_yields_one_canonical_rate(self):
        """A numerator contained by several bases must not spawn four near-duplicates."""
        _, derived, _ = self._candidates(
            pd.read_csv(SAMPLE_CSV))
        rates = [c for c in derived if c.rule == "outcome_rate"]
        numerators = [c.numerator for c in rates]
        self.assertEqual(len(numerators), len(set(numerators)),
                         "the same numerator produced more than one rate")

    def test_hospital_recovery_and_mortality_rates_are_discovered(self):
        _, derived, _ = self._candidates(self.hospital_raw)
        expressions = {(c.numerator, c.denominator) for c in derived}
        self.assertIn(("{discharges}", "{admissions}"), expressions)
        self.assertIn(("{recovered_patients}", "{admissions}"), expressions)

    def test_a_pre_computed_rate_column_is_flagged_rather_than_summed(self):
        with_rate = pd.DataFrame({
            "date": pd.date_range("2023-01-02", periods=60, freq="W-MON"),
            "conversion_rate": [12.5] * 60,
        })
        df, schema, profile = _profile_of(with_rate)
        atomic = build_atomic_candidates(profile, [])
        cand = next(c for c in atomic if c.candidate_id == "conversion_rate")
        self.assertEqual(cand.kind, "mean")
        self.assertEqual(cand.verdict, "questionable")
        self.assertIn("already a rate", cand.definition)


# ---------------------------------------------------------------------------
# the resolver
# ---------------------------------------------------------------------------
class TestResolver(KpiContractTestCase):
    def test_formula_grammar_parses_what_the_contract_needs(self):
        for expression, fields in [
            ("{a}", {"a"}),
            ("{a} - {b}", {"a", "b"}),
            ("({a} + {b}) * 2", {"a", "b"}),
            ("{a} / {b} * 100", {"a", "b"}),
        ]:
            self.assertEqual(referenced_fields(parse_expression(expression)), fields)

    def test_formulas_are_parsed_not_executed(self):
        """Formula text arrives from the API and must never reach an interpreter."""
        for hostile in ("__import__('os').system('x')", "{a} ** 2", "eval('1')", "", "{a} +"):
            with self.assertRaises(KpiResolutionError):
                parse_expression(hostile)

    def test_an_unknown_column_is_reported_not_silently_zero(self):
        frame = pd.DataFrame({"a": [1.0, 2.0]})
        with self.assertRaises(KpiResolutionError):
            evaluate(parse_expression("{ghost}"), frame)

    def test_a_ratio_is_recomputed_from_components_not_averaged(self):
        """
        The rule that makes a rate correct under aggregation: averaging four
        weekly margins is not the quarterly margin.
        """
        contract = kpi_service.bootstrap("u", self.dataset_stub, self.df, self.schema)
        resolver = compile_contract(contract)
        margin = resolver["gross_margin_pct"]

        quarters = [g for _, g in self.df.groupby(["_year", "_quarter"])][:2]
        combined = pd.concat(quarters)

        per_quarter = [margin.compute(q) for q in quarters]
        mean_of_ratios = sum(per_quarter) / len(per_quarter)

        num = sum(margin.components(q)[0] for q in quarters)
        den = sum(margin.components(q)[1] for q in quarters)
        recomputed = num / den * margin.scale

        self.assertClose(margin.compute(combined), recomputed, rel=1e-9)
        self.assertNotAlmostEqual(margin.compute(combined), mean_of_ratios, places=9)
        self.assertEqual(margin.rollup_policy, "recompute_from_components")

    def test_filters_are_applied_before_aggregation(self):
        contract = kpi_service.bootstrap("u", self.dataset_stub, self.df, self.schema)
        kpi = kpi_service.add_user_kpi(contract, {
            "name": "Online revenue",
            "formula": {"kind": "sum", "expression": "{revenue}"},
            "filters": [{"field": "channel", "op": "eq", "value": "Online"}],
            "unit": "currency",
        }, actor="tester")
        compiled = compile_kpi(kpi)
        expected = float(self.df[self.df["channel"] == "Online"]["revenue"].sum())
        self.assertClose(compiled.compute(self.df), expected, rel=1e-9)
        self.assertLess(compiled.compute(self.df), float(self.df["revenue"].sum()))

    def test_only_approved_kpis_are_authoritative(self):
        contract = kpi_service.discover("u", self.dataset_stub, self.df, self.schema)
        self.assertTrue(any(k.status != "approved" for k in contract.kpis))
        self.assertEqual(compile_contract(contract), {})
        self.assertTrue(compile_contract(contract, include_unapproved=True))


# ---------------------------------------------------------------------------
# granularity
# ---------------------------------------------------------------------------
class TestGranularity(KpiContractTestCase):
    def test_every_kpi_declares_a_granularity(self):
        contract = kpi_service.discover("u", self.dataset_stub, self.df, self.schema)
        for kpi in contract.kpis:
            self.assertTrue(kpi.granularity.time_grain)
            self.assertTrue(kpi.granularity.label)
            self.assertTrue(kpi.granularity.native_row_grain)

    def test_the_time_grain_follows_the_dataset_cadence(self):
        contract = kpi_service.discover("u", self.dataset_stub, self.df, self.schema)
        self.assertEqual(self.schema.grain, "weekly")
        self.assertEqual(contract.kpi("revenue").granularity.time_grain, "week")

    def test_a_ratio_must_have_its_grain_confirmed_before_approval(self):
        contract = kpi_service.discover("u", self.dataset_stub, self.df, self.schema)
        margin = contract.kpi("gross_margin_pct")
        self.assertTrue(margin.granularity.requires_confirmation)
        self.assertEqual(margin.status, "needs_confirmation")

        with self.assertRaises(KpiContractError) as caught:
            kpi_service.approve_kpi(contract, "gross_margin_pct", "tester")
        self.assertIn("granularity", str(caught.exception))

        kpi_service.update_kpi(contract, "gross_margin_pct",
                               {"granularity": {"time_grain": "quarter", "declared_by": "user"}},
                               actor="tester")
        approved = kpi_service.approve_kpi(contract, "gross_margin_pct", "tester")
        self.assertEqual(approved.status, "approved")
        self.assertTrue(approved.approval.user_confirmed_granularity)

    def test_rollup_policy_matches_the_kind_of_measure(self):
        contract = kpi_service.bootstrap("u", self.dataset_stub, self.df, self.schema)
        self.assertEqual(contract.kpi("revenue").aggregation.rollup_policy, "sum")
        self.assertEqual(contract.kpi("inventory_units").aggregation.rollup_policy, "mean")
        self.assertEqual(contract.kpi("gross_margin_pct").aggregation.rollup_policy,
                         "recompute_from_components")

    def test_comparability_records_why_two_kpis_cannot_be_compared(self):
        contract = kpi_service.bootstrap("u", self.dataset_stub, self.df, self.schema)
        kpi_service.update_kpi(contract, "orders",
                               {"granularity": {"time_grain": "month", "declared_by": "user"}},
                               actor="tester")
        kpi_service.compute_comparability(contract.kpis)
        revenue = contract.kpi("revenue")
        entry = next(e for e in revenue.comparability if e.kpi_id == "orders")
        self.assertFalse(entry.comparable)
        self.assertTrue(any("time grain" in r for r in entry.reasons))


# ---------------------------------------------------------------------------
# conflicts — never silently resolved
# ---------------------------------------------------------------------------
class TestConflicts(KpiContractTestCase):
    def test_two_definitions_of_one_name_block_approval(self):
        contract = kpi_service.discover("u", self.dataset_stub, self.df, self.schema)
        rival = kpi_service.add_user_kpi(contract, {
            "name": contract.kpi("revenue").name,          # same name
            "formula": {"kind": "sum", "expression": "{revenue} - {returns}"},
            "unit": "currency",
        }, actor="tester")

        _, _, profile = _profile_of(pd.read_csv(SAMPLE_CSV))
        matched, _ = bind_library(profile)
        contract.conflicts = detect_conflicts(contract.kpis, profile, matched)

        blocking = [c for c in contract.conflicts if c.kind == "formula_disagreement"]
        self.assertTrue(blocking, "two rival definitions of one KPI must be flagged")
        self.assertEqual(blocking[0].severity, "blocking")
        self.assertIn(rival.kpi_id, blocking[0].affected_kpis)

        with self.assertRaises(KpiContractError):
            kpi_service.approve_contract(contract, "tester")

    def test_a_blocking_conflict_prevents_approving_the_kpi_it_names(self):
        contract = kpi_service.discover("u", self.dataset_stub, self.df, self.schema)
        contract.conflicts.append(
            __import__("app.kpi.contract", fromlist=["ConflictFlag"]).ConflictFlag(
                conflict_id="cf_test", kind="duplicate_definition", severity="blocking",
                detail="test", affected_kpis=["revenue"]))
        with self.assertRaises(KpiContractError) as caught:
            kpi_service.approve_kpi(contract, "revenue", "tester")
        self.assertIn("blocking", str(caught.exception))

    def test_resolving_a_conflict_records_who_decided_and_why(self):
        contract = kpi_service.discover("u", self.dataset_stub, self.df, self.schema)
        ConflictFlag = __import__("app.kpi.contract", fromlist=["ConflictFlag"]).ConflictFlag
        ResolutionOption = __import__("app.kpi.contract",
                                      fromlist=["ResolutionOption"]).ResolutionOption
        contract.conflicts.append(ConflictFlag(
            conflict_id="cf_x", kind="duplicate_definition", severity="blocking",
            detail="test", affected_kpis=["revenue"],
            resolution_options=[ResolutionOption(option_id="keep", label="Keep it")]))

        kpi_service.resolve_conflict(contract, "cf_x", "keep",
                                     "Finance owns this definition.", "tester")
        conflict = next(c for c in contract.conflicts if c.conflict_id == "cf_x")
        self.assertTrue(conflict.resolved)
        self.assertEqual(conflict.resolved_by, "tester")
        self.assertIn("Finance owns", conflict.resolution_rationale)
        self.assertTrue(any("Finance owns" in n
                            for n in contract.kpi("revenue").provenance.notes))
        self.assertEqual(contract.blocking_conflicts, [])

    def test_short_history_is_recorded_as_a_caveat_on_every_kpi(self):
        short = self.df[self.df["_year"] == self.df["_year"].max()]
        schema = detect_schema(short.drop(columns=[c for c in short.columns
                                                   if c.startswith("_")]))
        contract = kpi_service.discover("u", {"_id": "ds_short"}, prepare(
            short.drop(columns=[c for c in short.columns if c.startswith("_")]), schema), schema)
        kinds = {c.kind for c in contract.conflicts}
        self.assertIn("insufficient_history", kinds)


# ---------------------------------------------------------------------------
# user-defined KPIs and the lifecycle
# ---------------------------------------------------------------------------
class TestLifecycle(KpiContractTestCase):
    def test_a_user_can_define_a_kpi_the_data_cannot_imply(self):
        contract = kpi_service.bootstrap("u", self.dataset_stub, self.df, self.schema)
        kpi = kpi_service.add_user_kpi(contract, {
            "name": "Contribution after marketing",
            "business_definition": "Revenue less cost of goods and marketing spend.",
            "formula": {"kind": "sum",
                        "expression": "{revenue} - {cost_of_goods} - {marketing_spend}"},
            "unit": "currency",
        }, actor="analyst-1")
        self.assertEqual(kpi.kpi_type, "user_defined")
        self.assertEqual(kpi.provenance.origin, "user_defined")
        self.assertTrue(kpi.required)
        self.assertEqual(kpi.status, "proposed")

        expected = float((self.df["revenue"] - self.df["cost_of_goods"]
                          - self.df["marketing_spend"]).sum())
        self.assertClose(compile_kpi(kpi).compute(self.df), expected, rel=1e-9)

    def test_a_malformed_user_formula_is_refused_at_the_door(self):
        contract = kpi_service.bootstrap("u", self.dataset_stub, self.df, self.schema)
        with self.assertRaises(KpiResolutionError):
            kpi_service.add_user_kpi(contract, {
                "name": "Bad", "formula": {"kind": "sum", "expression": "import os"},
            }, actor="tester")

    def test_user_defined_kpi_fails_explicitly_when_a_source_field_is_unavailable(self):
        contract = kpi_service.bootstrap("u", self.dataset_stub, self.df, self.schema)
        with self.assertRaisesRegex(KpiContractError, "unavailable source field.*nonexistent_field"):
            kpi_service.add_user_kpi(contract, {
                "name": "Unavailable requirement",
                "formula": {"kind": "sum", "expression": "{nonexistent_field}"},
            }, actor="tester")

    def test_rediscovery_preserves_required_user_kpis_from_the_editable_draft(self):
        state = setup_environment()
        uid = "required_user_kpi"
        dataset = dict(state["dataset"])
        draft = kpi_service.replace_with_discovery(uid, dataset, self.df, self.schema)
        kpi_service.add_user_kpi(draft, {
            "name": "Contribution after marketing",
            "formula": {"kind": "sum", "expression": "{revenue} - {marketing_spend}"},
            "unit": "currency",
        }, actor="analyst")
        kpi_service.save(draft)

        refreshed = kpi_service.replace_with_discovery(uid, dataset, self.df, self.schema)
        retained = refreshed.kpi("contribution_after_marketing")
        self.assertIsNotNone(retained)
        self.assertTrue(retained.required)


class TestKpiRequirementClassification(KpiContractTestCase):
    def test_direct_dataset_measures_are_included_even_when_screening_rejects_them(self):
        class RejectingLlm:
            def screen_kpi_candidates(self, *_args, **_kwargs):
                return [{"candidate_id": "avg_test_score", "verdict": "reject",
                         "reason": "Incorrect semantic judgement."}]

        school = pd.DataFrame({
            "date": pd.date_range("2023-01-02", periods=80, freq="W-MON"),
            "avg_test_score": [72 + (i % 5) for i in range(80)],
            "attendance_present": [900 + (i % 7) for i in range(80)],
        })
        df, schema, _ = _profile_of(school)
        contract = kpi_service.discover("u", {"_id": "school", "filename": "school.csv"},
                                        df, schema, llm=RejectingLlm())
        self.assertIsNotNone(contract.kpi("avg_test_score"))
        self.assertEqual(contract.kpi("avg_test_score").provenance.origin, "dynamic_atomic")

    def test_unavailable_general_kpis_are_optional_and_do_not_block_a_valid_contract(self):
        unknown_domain = pd.DataFrame({
            "date": pd.date_range("2023-01-02", periods=80, freq="W-MON"),
            "custom_activity": range(100, 180),
        })
        df, schema, _ = _profile_of(unknown_domain)
        contract = kpi_service.bootstrap("u", {"_id": "custom", "filename": "custom.csv"}, df, schema)

        self.assertIsNotNone(contract.kpi("custom_activity"))
        self.assertFalse(any(k.provenance.origin == "general_library" for k in contract.kpis))
        self.assertTrue(contract.unavailable)
        self.assertTrue(compile_contract(contract))

    def test_editing_a_definition_resets_its_approval(self):
        contract = kpi_service.bootstrap("u", self.dataset_stub, self.df, self.schema)
        self.assertEqual(contract.kpi("revenue").status, "approved")
        kpi_service.update_kpi(contract, "revenue",
                               {"business_definition": "Net of intercompany sales."},
                               actor="analyst-1")
        revenue = contract.kpi("revenue")
        self.assertEqual(revenue.status, "proposed")
        self.assertFalse(revenue.approval.approved)
        self.assertIn("analyst-1", revenue.provenance.edited_by)

    def test_an_unknown_field_cannot_be_patched(self):
        contract = kpi_service.bootstrap("u", self.dataset_stub, self.df, self.schema)
        with self.assertRaises(KpiContractError):
            kpi_service.update_kpi(contract, "revenue", {"status": "approved"}, actor="x")

    def test_a_kpi_other_kpis_depend_on_cannot_be_deleted(self):
        contract = kpi_service.bootstrap("u", self.dataset_stub, self.df, self.schema)
        contract.kpi("gross_margin_pct").depends_on = ["revenue"]
        with self.assertRaises(KpiContractError):
            kpi_service.delete_kpi(contract, "revenue")

    def test_a_contract_with_no_approved_kpi_cannot_be_approved(self):
        contract = kpi_service.discover("u", self.dataset_stub, self.df, self.schema)
        with self.assertRaises(KpiContractError) as caught:
            kpi_service.approve_contract(contract, "tester")
        self.assertIn("at least one KPI", str(caught.exception))

    def test_approval_is_versioned_and_the_previous_version_survives(self):
        state = setup_environment()
        uid = "lifecycle_user"
        dataset = dict(state["dataset"])
        first = kpi_service.get_or_bootstrap(uid, dataset, self.df, self.schema)
        self.assertEqual(first.version, 1)

        draft = kpi_service.replace_with_discovery(uid, dataset, self.df, self.schema)
        self.assertEqual(draft.version, 2)
        self.assertFalse(draft.is_current, "a draft under review must not be live")

        live = kpi_service.load_current(uid, dataset["_id"])
        self.assertEqual(live.version, 1,
                         "the live contract must not change while a draft is reviewed")

        kpi_service.update_kpi(draft, "revenue", {"business_definition": "v2"}, actor="t")
        kpi_service.approve_kpi(draft, "revenue", "t")
        kpi_service.approve_contract(draft, "t")
        kpi_service.save(draft)
        self.assertEqual(draft.approved_by, "t")
        self.assertTrue(draft.approved_at)

        versions = kpi_service.KpiContractRepository().versions(uid, dataset["_id"])
        self.assertEqual({int(v["version"]) for v in versions}, {1, 2})


# ---------------------------------------------------------------------------
# screening — the model may judge, never compute or invent
# ---------------------------------------------------------------------------
class TestScreening(KpiContractTestCase):
    def test_discovery_completes_with_no_api_key(self):
        contract = kpi_service.discover("u", self.dataset_stub, self.df, self.schema, llm=None)
        self.assertEqual(contract.screened_by, "deterministic")
        self.assertTrue(contract.kpis)

    def test_no_kpi_ever_references_a_column_the_dataset_lacks(self):
        columns = set(self.df.columns)
        contract = kpi_service.discover("u", self.dataset_stub, self.df, self.schema)
        for kpi in contract.kpis:
            compiled = compile_kpi(kpi)
            missing = set(compiled.source_fields) - columns
            self.assertEqual(missing, set(),
                             f"{kpi.kpi_id} references columns that do not exist: {missing}")

    def test_a_model_proposal_naming_an_absent_column_is_discarded(self):
        candidate, why = _validate_proposal(
            ProposedKpi(name="Ghost rate", kind="ratio",
                        numerator_field="does_not_exist", denominator_field="orders"),
            ["orders", "revenue"])
        self.assertIsNone(candidate)
        self.assertIn("do not exist", why)

    def test_a_valid_model_proposal_becomes_a_reviewable_candidate(self):
        candidate, why = _validate_proposal(
            ProposedKpi(name="Revenue per order", kind="ratio",
                        numerator_field="revenue", denominator_field="orders"),
            ["orders", "revenue"])
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.rule, "llm_proposed")
        self.assertEqual(candidate.verdict, "questionable",
                         "a model proposal is never auto-approved")

    def test_a_failing_model_degrades_to_deterministic_screening(self):
        class Exploding:
            enabled = True
            model = "test"

            def screen_kpi_candidates(self, *_args, **_kwargs):
                raise RuntimeError("upstream is down")

        _, _, profile = _profile_of(pd.read_csv(SAMPLE_CSV))
        with self.assertLogs("app.kpi.screening", level="WARNING"):
            result = screen_candidates(profile, [], llm=Exploding())
        self.assertEqual(result.screened_by, "deterministic")
        self.assertTrue(any("unavailable" in n for n in result.notes))


# ---------------------------------------------------------------------------
# domain-grounded explanations — the same KPI reads differently by business
# ---------------------------------------------------------------------------
class TestDomainDetection(KpiContractTestCase):
    def test_the_dataset_domain_is_detected_from_bound_concepts(self):
        """
        Retail-shaped and hospital-shaped data must yield different, confident
        domain reads — not because of any hardcoded per-dataset rule, but
        because different KPI concepts bound to their fields.
        """
        _, _, retail_profile = _profile_of(pd.read_csv(SAMPLE_CSV))
        retail_matches, _ = bind_library(retail_profile)
        retail_ctx = detect_domain_context(retail_matches, retail_profile)

        _, _, hospital_profile = _profile_of(self.hospital_raw)
        hospital_matches, _ = bind_library(hospital_profile)
        hospital_ctx = detect_domain_context(hospital_matches, hospital_profile)

        self.assertEqual(retail_ctx.domain, "retail_ecommerce")
        self.assertFalse(retail_ctx.is_uncertain)
        self.assertEqual(retail_ctx.vocab, DOMAIN_VOCAB["retail_ecommerce"])
        self.assertEqual(hospital_ctx.domain, "healthcare")
        self.assertFalse(hospital_ctx.is_uncertain)
        self.assertEqual(hospital_ctx.vocab, DOMAIN_VOCAB["healthcare"])
        self.assertNotEqual(retail_ctx.vocab, hospital_ctx.vocab)
        self.assertTrue(retail_ctx.evidence and hospital_ctx.evidence,
                        "a domain read must carry the evidence for it")

    def test_domain_is_uncertain_rather_than_guessed_when_evidence_is_thin(self):
        """No industry-specific concept bound at all: say so, do not guess."""
        thin = pd.DataFrame({
            "date": pd.date_range("2023-01-02", periods=120, freq="W-MON"),
            "revenue": range(100, 220),
        })
        _, _, profile = _profile_of(thin)
        matches, _ = bind_library(profile)
        ctx = detect_domain_context(matches, profile)

        self.assertTrue(ctx.is_uncertain)
        self.assertLess(ctx.confidence, 0.5)
        self.assertTrue(ctx.evidence)

    def test_domain_is_uncertain_on_a_genuine_tie(self):
        """Two domains matching equally hard is exactly the case that must not
        be silently resolved by picking whichever sorts first."""
        matches = [
            m for m in bind_library(_profile_of(pd.read_csv(SAMPLE_CSV))[2])[0]
            if m.library.domain == "retail_ecommerce"
        ][:2]
        hospital_matches = [
            m for m in bind_library(_profile_of(self.hospital_raw)[2])[0]
            if m.library.domain == "healthcare"
        ][:2]
        tied = matches + hospital_matches
        ctx = detect_domain_context(tied, _profile_of(pd.read_csv(SAMPLE_CSV))[2])
        self.assertTrue(ctx.is_uncertain)
        self.assertEqual(ctx.domain, "uncertain")


class TestDomainGroundedExplanations(KpiContractTestCase):
    def test_the_same_library_kpi_reads_differently_across_domains(self):
        """
        'cost_of_goods' is a general-library KPI, so it activates on both a
        retail dataset (bound to its own `cost_of_goods` column) and this
        hospital fixture (bound to `treatment_cost` via the `cost` concept's
        aliases). The KPI id, and the underlying computation, are the same in
        both contracts — only the explanation should differ, and it must
        differ enough to be obviously written for a different business.
        """
        retail = kpi_service.bootstrap("u", self.dataset_stub, self.df, self.schema)
        _, hospital_schema, hospital_profile = _profile_of(self.hospital_raw)
        hospital = kpi_service.bootstrap(
            "u", {"_id": "hosp", "filename": "hospital.csv"},
            prepare(self.hospital_raw, hospital_schema), hospital_schema)

        retail_kpi = retail.kpi("cost_of_goods")
        hospital_kpi = hospital.kpi("cost_of_goods")
        self.assertIsNotNone(retail_kpi)
        self.assertIsNotNone(hospital_kpi)

        # The computation itself is untouched: each is grounded in the field
        # that actually bound in its own dataset, not a shared literal name.
        self.assertEqual(retail_kpi.formula.expression, "{cost_of_goods}")
        self.assertEqual(hospital_kpi.formula.expression, "{treatment_cost}")
        self.assertEqual(retail_kpi.source_fields, ["cost_of_goods"])
        self.assertEqual(hospital_kpi.source_fields, ["treatment_cost"])

        # The explanation is materially different, not a shared generic sentence.
        self.assertNotEqual(retail_kpi.business_definition, hospital_kpi.business_definition)
        self.assertNotEqual(retail_kpi.relevance, hospital_kpi.relevance)

        retail_text = (retail_kpi.business_definition + " " + retail_kpi.relevance).lower()
        hospital_text = (hospital_kpi.business_definition + " " + hospital_kpi.relevance).lower()

        self.assertIn("retail", retail_text)
        self.assertIn("patient", hospital_text)
        # Neither explanation borrows the other business's vocabulary.
        self.assertNotIn("patient", retail_text)
        self.assertNotIn("retail", hospital_text)

    def test_compose_explanation_is_grounded_in_the_actual_bound_field(self):
        """The field name behind the KPI must appear in its own explanation —
        the text is tied to this dataset's columns, not boilerplate."""
        _, _, hospital_profile = _profile_of(self.hospital_raw)
        hospital_matches, _ = bind_library(hospital_profile)
        ctx = detect_domain_context(hospital_matches, hospital_profile)

        business_definition, relevance = compose_explanation(
            "Average length of stay", "mean", "duration", ["throughput", "efficiency"],
            False, ["length_of_stay_days"], ctx)
        self.assertIn("length_of_stay_days", business_definition)
        self.assertTrue(relevance)

    def test_uncertain_domain_explanations_say_so_explicitly_rather_than_guess(self):
        """
        No industry-specific concept bound here, so every explanation must
        name that uncertainty rather than confidently asserting a business the
        data does not support.
        """
        thin = pd.DataFrame({
            "date": pd.date_range("2023-01-02", periods=120, freq="W-MON"),
            "revenue": range(100, 220),
        })
        df, schema, _ = _profile_of(thin)
        contract = kpi_service.bootstrap("u", {"_id": "thin", "filename": "thin.csv"}, df, schema)
        revenue = contract.kpi("revenue")

        self.assertTrue(contract.domain_uncertain)
        self.assertIn("could not be confidently determined", revenue.business_definition)
        # And it must not have invented a specific industry to fill the gap.
        for guess in ("retail", "hospital", "healthcare", "logistics", "subscription",
                      "manufacturing"):
            self.assertNotIn(guess, revenue.business_definition.lower())

    def test_every_tag_narrative_actually_varies_by_domain(self):
        """
        Every phrase in the tag library must reference at least one domain-vocab
        placeholder in both halves — a template with no placeholder is exactly
        the kind of industry-independent boilerplate this feature exists to
        remove, and it would silently produce identical text across domains.
        """
        from app.kpi.explanation import TAG_PHRASES

        placeholder = re.compile(r"\{(label|entity|activity|unit_of_work|capacity_note)\}")
        for tag, (what, why) in TAG_PHRASES.items():
            self.assertRegex(what, placeholder, f"'{tag}' what-clause has no domain placeholder")
            self.assertRegex(why, placeholder, f"'{tag}' why-clause has no domain placeholder")

    def test_explanations_do_not_alter_the_computed_value(self):
        """
        The whole point of keeping this change to prose: the same KPI, on the
        same data, must still compute the same number the legacy registry did.
        """
        contract = kpi_service.bootstrap("u", self.dataset_stub, self.df, self.schema)
        resolver = compile_contract(contract)
        for _, group in self.df.groupby(["_year", "_quarter"]):
            expected = legacy_compute(group, "cost_of_goods")
            actual = resolver["cost_of_goods"].compute(group)
            self.assertClose(actual, expected, rel=1e-9)


# ---------------------------------------------------------------------------
# isolation
# ---------------------------------------------------------------------------
class TestIsolation(KpiContractTestCase):
    def test_contracts_are_scoped_to_their_owner(self):
        state = setup_environment()
        dataset = dict(state["dataset"])
        kpi_service.get_or_bootstrap("owner_a", dataset, self.df, self.schema)
        self.assertIsNotNone(kpi_service.load_current("owner_a", dataset["_id"]))
        self.assertIsNone(kpi_service.load_current("owner_b", dataset["_id"]),
                          "one user must never see another user's KPI contract")


if __name__ == "__main__":
    unittest.main()
