"""
`POST /api/questions/ask` -- the agent loop over HTTP.

Every scenario is headless. `base.setup_environment` sets `ANTHROPIC_API_KEY=""`,
so `get_llm()` comes back disabled; assigning a `ScriptedAnthropic` to its
`_client` both enables it and fixes the turns it will take (`LLMClient.enabled`
is `self._client is not None`). That is the same seam `test_agent_loop.py:22-25`
uses, one level further out: here the registry, the airlock and the real tools
all run for real, and only the model is scripted.

Two hazards this module has to work around, both noted in the batch plan:

* `LLMClient._cache` is a process-global memo keyed on the whole message
  history, so two tests asking the *same* question would replay the first
  test's script instead of consuming their own. Every test uses a distinct
  question and `tearDown` clears the cache.
* `get_llm()` is a module-level singleton shared with the rest of the app, so
  `tearDown` also restores `_client = None` rather than leaving a scripted
  double installed for whatever runs next.
"""
from __future__ import annotations

import json
import unittest

from fastapi.testclient import TestClient

from app.db.repositories import InvestigationRepository
from app.llm.client import LLMTransportError, get_llm
from app.main import app

from .base import SAMPLE_CSV, assert_json_safe, setup_environment
from .fakes import ScriptedAnthropic, scripted_message, text_block, tool_use_block

YEAR_2025 = {"type": "year", "year": 2025}


class _Exploding:
    """An SDK double whose every request fails, for the transport-error path."""

    class _Messages:
        def create(self, **kwargs):
            raise RuntimeError("connection reset by peer: api.anthropic.com")

    def __init__(self):
        self.messages = self._Messages()


class AskEndpointTestCase(unittest.TestCase):
    """Two roles against one uploaded dataset, so the role-parity tests are cheap."""

    @classmethod
    def setUpClass(cls):
        setup_environment()
        cls.client = TestClient(app)
        cls.analyst = cls._login("ask_analyst@example.com", "data_analyst")
        cls.leader = cls._login("ask_leader@example.com", "business_leader")
        # Absolute path on purpose: `test_stage_endpoints.py` uses a relative
        # one and only passes when the suite is run from `backend/`.
        with open(SAMPLE_CSV, "rb") as f:
            cls.client.post("/api/datasets", headers=cls.analyst,
                            files={"file": ("retail.csv", f, "text/csv")})
        with open(SAMPLE_CSV, "rb") as f:
            cls.client.post("/api/datasets", headers=cls.leader,
                            files={"file": ("retail.csv", f, "text/csv")})

    @classmethod
    def _login(cls, email: str, role: str):
        token = cls.client.post("/api/auth/demo-login",
                                json={"email": email, "role": role}).json()["token"]
        return {"Authorization": f"Bearer {token}"}

    def tearDown(self):
        llm = get_llm()
        llm._client = None
        llm._cache.clear()

    # -- helpers ------------------------------------------------------------
    def script(self, responses):
        llm = get_llm()
        llm._client = ScriptedAnthropic(responses)
        llm._cache.clear()
        return llm

    def one_query_then_answer(self, prose="Revenue was steady."):
        return [
            scripted_message([tool_use_block("t1", "query_kpi",
                                             {"kpi_keys": "revenue", "time_filter": YEAR_2025})],
                             "tool_use"),
            scripted_message([text_block(prose)], "end_turn"),
        ]

    def ask(self, question, headers=None, **body):
        return self.client.post("/api/questions/ask", headers=headers or self.analyst,
                                json={"question": question, **body})


class TestTheHappyPath(AskEndpointTestCase):
    def test_a_tier_1_question_returns_the_answer_and_its_full_evidence_trail(self):
        self.script(self.one_query_then_answer("Revenue in 2025 was steady."))
        r = self.ask("What was revenue in 2025?")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["answer"], "Revenue in 2025 was steady.")
        self.assertEqual(len(body["evidence"]), 1)
        step = body["evidence"][0]
        self.assertEqual(step["tool"], "query_kpi")
        self.assertEqual(step["step"], 1)
        self.assertFalse(step["is_error"])
        # The tool really ran -- this is a computed figure, not a scripted one.
        self.assertIsNotNone(step["result"])
        self.assertEqual(body["engine"]["turns"], 2)

    def test_the_response_carries_exactly_the_documented_top_level_keys(self):
        """`response_model` silently DROPS any key it does not declare, so the
        published shape has to be asserted rather than assumed."""
        self.script(self.one_query_then_answer())
        body = self.ask("What was revenue in 2025, roughly?").json()
        self.assertEqual(
            set(body),
            {"status", "question", "answer", "evidence", "kpis_used", "periods_used",
             "engine", "dataset", "view", "telemetry"})

    def test_the_question_is_echoed_back_verbatim(self):
        self.script(self.one_query_then_answer())
        question = "What was revenue in 2025 across the whole business?"
        self.assertEqual(self.ask(question).json()["question"], question)

    def test_the_evidence_trail_is_json_safe_with_no_nan_or_infinity(self):
        self.script(self.one_query_then_answer())
        body = self.ask("Is revenue in 2025 a clean number?").json()
        assert_json_safe(body["evidence"])


class TestEveryLoopEndingIsHttp200(AskEndpointTestCase):
    def test_no_usable_llm_returns_llm_required_at_http_200_not_an_error(self):
        # No script installed: `get_llm()` is disabled, as it is with no API key.
        r = self.ask("What should I worry about with no model available?")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["status"], "llm_required")
        self.assertEqual(body["answer"], "")
        self.assertEqual(body["evidence"], [])
        self.assertEqual(body["engine"]["turns"], 0)

    def test_max_turns_exhaustion_returns_http_200_with_the_partial_trace(self):
        self.script([
            scripted_message([tool_use_block(f"t{i}", "query_kpi",
                                             {"kpi_keys": "revenue", "time_filter": YEAR_2025})],
                             "tool_use")
            for i in range(1, 15)
        ])
        r = self.ask("Keep going until you run out of turns please.")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["status"], "max_turns_exhausted")
        self.assertTrue(body["evidence"], "the partial trace must survive non-convergence")

    def test_a_model_refusal_returns_http_200_with_status_refused(self):
        self.script([scripted_message([], "refusal")])
        r = self.ask("A question the safety classifier declines to answer.")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "refused")

    def test_a_truncated_final_turn_returns_http_200_with_status_truncated(self):
        self.script([scripted_message([text_block("Revenue was ")], "max_tokens")])
        r = self.ask("Give me an answer that runs out of output tokens.")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "truncated")

    def test_a_tool_error_mid_loop_is_reported_as_evidence_not_as_an_http_error(self):
        self.script([
            scripted_message([tool_use_block("t1", "query_kpi",
                                             {"kpi_keys": "not_a_real_kpi",
                                              "time_filter": YEAR_2025})], "tool_use"),
            scripted_message([text_block("That measure does not exist here.")], "end_turn"),
        ])
        r = self.ask("What was flurb in 2025?")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["status"], "ok")
        result = body["evidence"][0]["result"]
        self.assertTrue(body["evidence"][0]["is_error"])
        # The airlock rejects it against the tool's own enum before the engine
        # is reached, so this is `invalid_argument` rather than the engine's
        # `unknown_kpi` -- and it names the real keys, which is what makes the
        # turn recoverable rather than merely failed.
        self.assertEqual(result["error"], "invalid_argument")
        self.assertIn("revenue", result["valid_alternatives"])


class TestWhatTheAnswerClaimsToHaveUsed(AskEndpointTestCase):
    def test_kpis_used_excludes_a_kpi_named_only_by_a_rejected_tool_call(self):
        """A key the airlock refused was never queried, so it cannot be
        something the answer rests on."""
        self.script([
            scripted_message([tool_use_block("t1", "query_kpi",
                                             {"kpi_keys": "totally_invented_kpi",
                                              "time_filter": YEAR_2025})], "tool_use"),
            scripted_message([tool_use_block("t2", "query_kpi",
                                             {"kpi_keys": "revenue",
                                              "time_filter": YEAR_2025})], "tool_use"),
            scripted_message([text_block("Recovered and used revenue.")], "end_turn"),
        ])
        body = self.ask("What was our made-up measure in 2025?").json()
        self.assertIn("revenue", body["kpis_used"])
        self.assertNotIn("totally_invented_kpi", body["kpis_used"])

    def test_kpis_used_and_periods_used_come_from_the_trace_not_the_answer_prose(self):
        self.script([
            scripted_message([tool_use_block("t1", "query_kpi",
                                             {"kpi_keys": "revenue",
                                              "time_filter": YEAR_2025})], "tool_use"),
            scripted_message([text_block(
                "Gross profit rose sharply in Q4 2026.")], "end_turn"),
        ])
        body = self.ask("Tell me about 2025 revenue but mention profit.").json()
        self.assertEqual(body["kpis_used"], ["revenue"])
        self.assertNotIn("gross_profit", body["kpis_used"])
        self.assertEqual(body["periods_used"], ["2025"])


class TestTheEvidenceTrailIsNotAnAnalystPrivilege(AskEndpointTestCase):
    def test_a_business_leader_receives_the_identical_evidence_trail_as_an_analyst(self):
        self.script(self.one_query_then_answer())
        analyst = self.ask("How did revenue do in 2025, for the record?",
                           headers=self.analyst).json()
        self.script(self.one_query_then_answer())
        leader = self.ask("How did revenue do in 2025, for the record?",
                          headers=self.leader).json()
        self.assertEqual(analyst["evidence"], leader["evidence"])
        self.assertEqual(analyst["answer"], leader["answer"])

    def test_the_view_block_reports_the_callers_role_and_that_nothing_was_redacted(self):
        self.script(self.one_query_then_answer())
        leader = self.ask("Revenue in 2025 for a leader?", headers=self.leader).json()["view"]
        self.assertEqual(leader["role"], "business_leader")
        # Keeps the meaning it has on every other endpoint...
        self.assertFalse(leader["analyst_detail_included"])
        # ...and this is what says the trail was complete regardless.
        self.assertEqual(leader["redaction"], "none")


class TestErrorMapping(AskEndpointTestCase):
    def test_a_user_with_no_dataset_gets_409_rather_than_500(self):
        headers = self._login("ask_nodata@example.com", "data_analyst")
        r = self.client.post("/api/questions/ask", headers=headers,
                             json={"question": "What is revenue?"})
        self.assertEqual(r.status_code, 409)

    def test_an_unknown_dataset_id_gets_409(self):
        r = self.ask("What is revenue?", dataset_id="ds_does_not_exist")
        self.assertEqual(r.status_code, 409)

    def test_a_provider_transport_failure_returns_503_and_does_not_leak_the_sdk_message(self):
        llm = get_llm()
        llm._client = _Exploding()
        llm._cache.clear()
        r = self.ask("What happens when the provider is down?")
        self.assertEqual(r.status_code, 503)
        detail = r.json()["detail"]
        self.assertNotIn("api.anthropic.com", detail)
        self.assertNotIn("connection reset", detail)

    def test_the_transport_error_is_our_own_type_not_the_sdk_exception(self):
        """The API layer must be able to catch this without importing `anthropic`."""
        llm = get_llm()
        llm._client = _Exploding()
        llm._cache.clear()
        with self.assertRaises(LLMTransportError):
            llm._call_with_tools("sys", [{"role": "user", "content": "hi"}], [], lambda n, a: ({}, False))

    def test_an_empty_question_is_rejected_by_the_request_model(self):
        r = self.client.post("/api/questions/ask", headers=self.analyst, json={"question": ""})
        self.assertEqual(r.status_code, 422)

    def test_a_question_over_five_hundred_characters_is_rejected_by_the_request_model(self):
        r = self.client.post("/api/questions/ask", headers=self.analyst,
                             json={"question": "x" * 501})
        self.assertEqual(r.status_code, 422)


class TestTelemetryAndPersistence(AskEndpointTestCase):
    def test_telemetry_records_one_llm_call_for_each_tool_loop_turn(self):
        self.script(self.one_query_then_answer())
        body = self.ask("How many model calls does one 2025 revenue lookup take?").json()
        self.assertEqual(body["telemetry"]["model_calls"], 2)
        self.assertEqual(body["telemetry"]["endpoint"], "/api/questions/ask")

    def test_asking_a_question_writes_no_investigation_row(self):
        before = len(InvestigationRepository().list_for_user("test_user"))
        self.script(self.one_query_then_answer())
        self.ask("Does asking this save anything about 2025?")
        after = len(InvestigationRepository().list_for_user("test_user"))
        self.assertEqual(before, after)

    def test_a_persist_field_in_the_body_is_ignored_rather_than_honoured(self):
        self.script(self.one_query_then_answer())
        r = self.ask("Can I force a save of this 2025 lookup?", persist=True)
        self.assertEqual(r.status_code, 200)
        self.assertNotIn("investigation_id", r.json())


class TestThePublishedContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_environment()
        cls.spec = TestClient(app).get("/openapi.json").json()

    def test_the_openapi_document_publishes_all_five_answer_statuses(self):
        schema = self.spec["components"]["schemas"]["AgentAnswerResponse"]
        status = schema["properties"]["status"]
        published = set(status.get("enum") or json.loads(json.dumps(status)).get("enum") or [])
        self.assertEqual(
            published,
            {"ok", "max_turns_exhausted", "truncated", "refused", "llm_required"})

    def test_the_ask_endpoint_is_published_with_a_response_model(self):
        post = self.spec["paths"]["/api/questions/ask"]["post"]
        ref = post["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
        self.assertTrue(ref.endswith("AgentAnswerResponse"))


if __name__ == "__main__":
    unittest.main()
