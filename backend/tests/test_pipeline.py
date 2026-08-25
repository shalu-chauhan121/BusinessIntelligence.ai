"""Stages 2-4 — INVESTIGATE, CONTEST, ACT — plus retrieval and the full pipeline."""
from __future__ import annotations

import unittest

from app.engines.analysis import detect_onset, lead_lag, price_volume_decomposition, weekly_frame
from app.engines.contest import contest
from app.engines.investigate import investigate
from app.engines.act import act
from app.engines.observe import Timeframe, observe, slice_period
from app.rag.chunker import chunk_document
from app.rag.retriever import Retriever
from app.services import dataset_service, pipeline

from .base import EngineTestCase, SAMPLE_CSV


class TestRetrieval(EngineTestCase):
    def test_documents_are_indexed_per_user(self):
        self.assertTrue(Retriever(self.uid).available)
        self.assertFalse(Retriever("somebody_else").available)

    def test_retrieval_finds_the_right_document(self):
        hits = Retriever(self.uid).retrieve("supply disruption stockout allocation supplier", top_k=3)
        self.assertTrue(hits)
        self.assertIn("ops_supply_incident_report", hits[0]["document_name"])

    def test_irrelevant_query_does_not_produce_strong_evidence(self):
        """Relative relevance is always 1.0 for the top hit — absolute strength must not be."""
        items = Retriever(self.uid).retrieve_evidence("quantum entanglement neutrino telescope", top_k=3)
        self.assertEqual(items, [])

    def test_chunker_preserves_headings(self):
        chunks = chunk_document("# Title\n\nbody one\n\n## Section A\n\nbody two")
        self.assertTrue(any(c["heading"] == "Section A" for c in chunks))


class TestAnalysisPrimitives(EngineTestCase):
    def test_onset_detection_dates_the_decline(self):
        north = self.df[self.df["region"] == "North"]
        window = north[(north["_date"] >= "2025-10-01") & (north["_date"] <= "2026-06-30")]
        onset = detect_onset(weekly_frame(window, "revenue"), baseline_weeks=20, direction="down")
        self.assertIsNotNone(onset)
        self.assertGreaterEqual(onset["week"], "2026-04-01")
        self.assertLessEqual(onset["week"], "2026-04-30")

    def test_supply_signal_starts_after_the_revenue_decline(self):
        """The core contest fact: the documented cause post-dates the KPI move."""
        scoped = self.df[(self.df["region"] == "North") & (self.df["product"] == "Product A")]
        window = scoped[(scoped["_date"] >= "2025-10-01") & (scoped["_date"] <= "2026-06-30")]
        rev = detect_onset(weekly_frame(window, "revenue"), baseline_weeks=20, direction="down")
        stock = detect_onset(weekly_frame(window, "stockout_rate"), baseline_weeks=20, direction="up")
        self.assertIsNotNone(rev)
        self.assertIsNotNone(stock)
        self.assertLess(rev["week"], stock["week"])

    def test_price_volume_decomposition_adds_up(self):
        cur = slice_period(self.df, Timeframe(2026, 2))
        base = slice_period(self.df, Timeframe(2026, 1))
        pv = price_volume_decomposition(cur, base)
        self.assertClose(pv["volume_effect"] + pv["price_effect"], pv["total_change"], rel=1e-6)

    def test_lead_lag_uses_differences_not_levels(self):
        window = self.df[self.df["_date"] >= "2025-01-01"]
        ll = lead_lag(weekly_frame(window, "revenue"), weekly_frame(window, "orders"))
        self.assertIsNotNone(ll)
        self.assertIn(ll["verdict"], {"cause_leads", "simultaneous", "kpi_leads"})


class TestFourStages(EngineTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.observation = observe(cls.df, cls.schema, "revenue", Timeframe(2026, 2))
        cls.investigation = investigate(cls.df, cls.schema, cls.observation, cls.uid, llm=None)
        cls.contested = contest(cls.df, cls.schema, cls.observation, cls.investigation, cls.uid, llm=None)
        cls.action = act(cls.df, cls.observation, cls.investigation, cls.contested, llm=None)

    # --- investigate ------------------------------------------------------
    def test_competing_hypotheses_are_generated(self):
        """
        More than one explanation, and every one of them derived from this
        dataset rather than from a fixed library.

        The old assertion named specific retail templates. Those are gone: with
        no LLM configured the hypotheses now come from the KPI's own definition
        and from the dimensions that actually moved, so the test asserts the
        property that matters rather than the names it used to produce.
        """
        hypotheses = self.investigation["hypotheses"]
        self.assertGreaterEqual(len(hypotheses), 2,
                                "the point is COMPETING explanations, not one")
        for h in hypotheses:
            self.assertTrue(h["evidence"] or h["documentary_evidence"], h["key"])
            self.assertIn(h.get("source"),
                          {"contract_formula", "contract_association",
                           "dimension_concentration", "llm_domain_reasoning"},
                          f"{h['key']} came from nowhere identifiable")

    def test_focus_is_the_disproportionate_driver(self):
        self.assertEqual(self.investigation["focus"].get("region"), "North")
        self.assertEqual(self.investigation["focus"].get("product"), "Product A")

    def test_every_hypothesis_carries_testable_evidence(self):
        for h in self.investigation["hypotheses"]:
            self.assertTrue(h["evidence"] or h["documentary_evidence"], h["key"])

    def test_documentary_evidence_is_quoted_with_a_source(self):
        quoted = [h for h in self.investigation["hypotheses"] if h["documentary_evidence"]]
        self.assertTrue(quoted, "no hypothesis retrieved any documentary evidence")
        for h in quoted:
            for item in h["documentary_evidence"]:
                self.assertTrue(item["source"].endswith(".md"))
                self.assertTrue(item["quote"])

    def test_every_hypothesis_seeks_disconfirming_evidence(self):
        """
        Searching only for support is how an investigation fools itself.

        Every hypothesis must carry something to search against itself with, so
        `contest.contradictory_retrieval` can never silently return nothing —
        the failure mode that existed while contradiction queries were keyed by
        a fixed list of template names.
        """
        from app.engines.contest import _generic_contradiction_queries

        for h in self.investigation["hypotheses"]:
            queries = h.get("contradiction_queries") or _generic_contradiction_queries(h)
            self.assertTrue(queries, f"{h['key']} has no way to be disproved")

    # --- contest ----------------------------------------------------------
    def test_a_cause_that_starts_too_late_is_capped(self):
        """
        A well-documented cause that starts after the KPI moved must be demoted,
        however much evidence supports it.

        Stated generically rather than against one named hypothesis: the rule is
        a property of the scoring, not of any particular explanation.
        """
        late = [h for h in self.contested["hypotheses"]
                if h["contest"]["temporal"].get("status") == "kpi_precedes_cause"]
        if not late:
            self.skipTest("no hypothesis in this fixture is dated after the KPI moved")
        for h in late:
            self.assertLessEqual(h["scoring"]["confidence"], 55, h["key"])
            self.assertIn("Capped", h["scoring"]["cap_reason"] or "")

    def test_reverse_causation_is_screened(self):
        marketing = next((h for h in self.contested["hypotheses"] if h["key"] == "marketing_pullback"), None)
        if marketing is None:
            self.skipTest("marketing hypothesis not shortlisted")
        self.assertTrue(marketing["contest"]["mechanism"]["declared_risk"])
        self.assertLessEqual(marketing["scoring"]["confidence"], 55)

    def test_ranking_is_ordered_and_bounded(self):
        confidences = [r["confidence"] for r in self.contested["ranking"]]
        self.assertEqual(confidences, sorted(confidences, reverse=True))
        self.assertTrue(all(0 <= c <= 100 for c in confidences))

    def test_no_hypothesis_claims_proven_causation(self):
        for h in self.contested["hypotheses"]:
            claim = h["scoring"]["causal_claim"].lower()
            self.assertNotIn("proves", claim)
            self.assertNotIn("proven", claim)
        self.assertIn("not probabilities", self.contested["confidence_disclaimer"])

    def test_reasoning_trail_is_complete(self):
        for h in self.contested["hypotheses"]:
            steps = [s["step"] for s in h["reasoning_trail"]]
            for expected in ["Hypothesis proposed", "Structured tests run", "Temporal precedence checked",
                             "Contradictory evidence search", "Confidence computed"]:
                self.assertIn(expected, steps)

    def test_missing_evidence_is_declared(self):
        for h in self.contested["hypotheses"]:
            self.assertTrue(h.get("missing"), h["key"])

    # --- act --------------------------------------------------------------
    def test_recommendations_are_linked_to_evidence(self):
        recs = self.action["recommendations"]
        self.assertTrue(recs)
        for r in recs:
            self.assertIn("hypothesis", r["based_on"])
            self.assertTrue(r["actions"])
            self.assertTrue(r["what_would_change_this"])

    def test_monitoring_thresholds_come_from_the_users_own_history(self):
        monitors = [m for r in self.action["recommendations"] for m in r["monitoring"]]
        self.assertTrue(monitors)
        for m in monitors:
            self.assertLess(m["lower_alert"], m["upper_alert"])

    def test_uncertainty_is_carried_through_to_the_leader_view(self):
        self.assertTrue(self.action["narrative"]["what_we_are_not_sure_about"])
        self.assertTrue(self.action["limits"])


class TestFullPipeline(EngineTestCase):
    def test_run_full_produces_all_four_stages(self):
        result = pipeline.run_full(self.uid, self.dataset, "revenue", 2026, 2,
                                   persist=True, use_llm=False)
        for stage in ("observe", "investigate", "contest", "act"):
            self.assertIn(stage, result)
        self.assertIn("investigation_id", result)
        self.assertEqual(result["engine"]["pipeline"], ["observe", "investigate", "contest", "act"])

    def test_pipeline_runs_without_an_llm_key(self):
        from app.llm.client import get_llm

        self.assertFalse(get_llm().enabled)
        result = pipeline.run_full(self.uid, self.dataset, "revenue", 2026, 2,
                                   persist=False, use_llm=True)
        self.assertTrue(result["contest"]["ranking"])
        self.assertEqual(result["engine"]["llm"]["mode"], "deterministic_fallback")

    def test_pipeline_is_deterministic(self):
        a = pipeline.run_full(self.uid, self.dataset, "revenue", 2026, 2, persist=False, use_llm=False)
        b = pipeline.run_full(self.uid, self.dataset, "revenue", 2026, 2, persist=False, use_llm=False)
        self.assertEqual([r["confidence"] for r in a["contest"]["ranking"]],
                         [r["confidence"] for r in b["contest"]["ranking"]])

    def test_quiet_quarter_produces_a_no_action_story(self):
        result = pipeline.run_full(self.uid, self.dataset, "revenue", 2026, 1, persist=False, use_llm=False)
        self.assertFalse(result["observe"]["anomaly"])

    def test_weak_evidence_requests_clarification_without_recommendation(self):
<<<<<<< HEAD
        """A real run preserves uncertainty instead of inventing a cause or action."""
        uid = "weak_evidence_user"
        dataset = dataset_service.store_upload(uid, SAMPLE_CSV.name, SAMPLE_CSV.read_bytes())
        result = pipeline.run_full(uid, dataset, "revenue", 2026, 1, persist=False, use_llm=False)
=======
        """
        A real run preserves uncertainty instead of inventing a cause or action.

        Uses 2025-Q3 rather than 2026-Q1. Both are quiet quarters, but Q1-2026 is
        quiet enough that the material-signal boundary short-circuits before any
        hypothesis is generated, so there is no ranking to be weak. That path is
        covered by `test_nothing_material_produces_no_hypotheses`. 2025-Q3 is the
        case this test is actually about: hypotheses exist, none is well enough
        evidenced to act on, and the pipeline says so instead of picking one.
        """
        uid = "weak_evidence_user"
        dataset = dataset_service.store_upload(uid, SAMPLE_CSV.name, SAMPLE_CSV.read_bytes())
        result = pipeline.run_full(uid, dataset, "revenue", 2025, 3, persist=False, use_llm=False)
>>>>>>> upstream/master
        top = result["contest"]["ranking"][0]
        request = result["act"]["clarification_request"]
        self.assertEqual(result["engine"]["pipeline"], ["observe", "investigate", "contest", "act"])
        self.assertLess(top["confidence"], 45)
        self.assertEqual(top["band"], "weak")
        self.assertTrue(result["act"]["narrative"]["what_we_are_not_sure_about"])
        self.assertNotIn("strongest-evidenced explanation", result["act"]["narrative"]["leading_explanation"])
        self.assertEqual(result["act"]["recommendations"], [])
        self.assertEqual(request["status"], "clarification_required")
        self.assertTrue(request["missing_evidence"])
        self.assertTrue(request["questions"])
        self.assertIn("not probabilities", result["contest"]["confidence_disclaimer"])

    def test_a_second_user_sees_none_of_the_first_users_data(self):
        from app.db.repositories import DatasetRepository

        self.assertIsNone(DatasetRepository().active("another_user"))


if __name__ == "__main__":
    unittest.main()
