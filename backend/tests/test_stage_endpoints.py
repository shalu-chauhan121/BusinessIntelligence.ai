"""
The single-stage endpoints — /api/observe, /investigate, /contest, /act.

These predate the question-driven flow and were, for a while, dead from the
frontend's perspective: only /api/investigations/run was ever called. They now
accept a `question` as an alternative to naming a KPI directly, resolved the
same way `/api/questions/investigate` resolves one, so a single stage can be
run against either an explicit KPI or a business question.
"""
from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app.main import app

HOSPITAL_CSV = "../sample_data/hospital_kpi_smoke_sample.csv"


class StageEndpointsTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        token = cls.client.post(
            "/api/auth/demo-login",
            json={"email": "stage_test@example.com", "role": "data_analyst"},
        ).json()["token"]
        cls.headers = {"Authorization": f"Bearer {token}"}
        with open(HOSPITAL_CSV, "rb") as f:
            cls.client.post("/api/datasets", headers=cls.headers,
                            files={"file": ("hospital.csv", f, "text/csv")})

    def _post(self, path, **body):
        return self.client.post(path, headers=self.headers,
                                json={"use_llm": False, **body})


class TestQuestionDrivenStages(StageEndpointsTestCase):
    def test_observe_resolves_a_question_to_a_kpi(self):
        r = self._post("/api/observe", question="Why did readmissions rise?")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["observe"]["kpi"], "readmissions")
        self.assertEqual(body["intent"]["outcome"]["kpi_key"], "readmissions")

    def test_investigate_uses_the_same_material_signal_filter_as_the_full_pipeline(self):
        r = self._post("/api/investigate", question="Why did readmissions rise?")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["investigate"]["hypotheses"])
        # Every hypothesis must trace to something the dataset measures — the
        # same guarantee the full pipeline enforces via the material-signal
        # boundary and the allowed-metrics constraint on LLM output.
        for h in body["investigate"]["hypotheses"]:
            self.assertIn(h.get("source"),
                          {"contract_formula", "contract_association",
                           "dimension_concentration", "llm_domain_reasoning"})

    def test_contest_carries_the_intent_through(self):
        r = self._post("/api/contest", question="Why did readmissions rise?")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["contest"]["ranking"])
        self.assertEqual(body["intent"]["outcome"]["kpi_key"], "readmissions")

    def test_act_reframes_for_the_requested_persona(self):
        r = self._post("/api/act", question="Why did readmissions rise?",
                       persona="operational_user")
        self.assertEqual(r.status_code, 200)
        view = r.json()["act"]["persona_view"]
        self.assertEqual(view["persona"], "operational_user")
        self.assertEqual(view["recommendations"][0]["timeframe"], "immediately")

    def test_an_unresolvable_question_returns_a_clarification_not_an_error(self):
        for path in ("/api/observe", "/api/investigate", "/api/contest", "/api/act"):
            r = self._post(path, question="Why did everything get worse?")
            self.assertEqual(r.status_code, 200, path)
            self.assertEqual(r.json().get("status"), "needs_clarification", path)

    def test_a_question_naming_an_unsupported_period_blocks(self):
        r = self._post("/api/observe", question="Why did readmissions rise in Q3 2019?")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json().get("status"), "needs_clarification")


class TestKpiDrivenStagesStillWork(StageEndpointsTestCase):
    """The original, explicit-KPI contract these endpoints shipped with is untouched."""

    def test_observe_with_an_explicit_kpi_ignores_question_resolution(self):
        r = self._post("/api/observe", kpi="readmissions")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["observe"]["kpi"], "readmissions")
        self.assertNotIn("intent", body)

    def test_investigate_with_an_explicit_kpi(self):
        r = self._post("/api/investigate", kpi="readmissions")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["investigate"]["hypotheses"])

    def test_a_question_overrides_an_explicit_kpi_when_both_are_given(self):
        """The question is the more specific instruction and wins."""
        r = self._post("/api/observe", kpi="admissions",
                       question="Why did readmissions rise?")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["observe"]["kpi"], "readmissions")


if __name__ == "__main__":
    unittest.main()
