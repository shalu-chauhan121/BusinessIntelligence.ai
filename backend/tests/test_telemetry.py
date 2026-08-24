"""Runtime telemetry stays lightweight and must never disrupt insights."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.api.routes_analysis import run_investigation, telemetry_summary
from app.db.repositories import TelemetryRepository
from app.llm.client import LLMClient
from app.models.schemas import AnalysisRequest
from app.services import pipeline
from app.services.telemetry import (
    TelemetrySession, estimate_tokens, request_telemetry, track_llm_call, track_processing_step,
)

from .base import EngineTestCase


class TestRuntimeTelemetry(EngineTestCase):
    def _response(self, input_tokens=11, output_tokens=7):
        return SimpleNamespace(usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens), content=[])

    def test_successful_call_records_provider_tokens_cost_and_latency(self):
        session = TelemetrySession(self.uid, "/test")
        session.record_call(model="claude-sonnet-4-5", system="s", user="u", response=self._response(), started=session.started)
        saved = session.persist()
        self.assertEqual(saved["input_tokens"], 11)
        self.assertEqual(saved["output_tokens"], 7)
        self.assertEqual(saved["prompt_tokens"], 11)
        self.assertEqual(saved["completion_tokens"], 7)
        self.assertEqual(saved["total_tokens"], 18)
        self.assertGreater(saved["estimated_cost"], 0)
        self.assertGreaterEqual(saved["latency_ms"], 0)
        self.assertTrue(saved["start_time"])
        self.assertTrue(saved["end_time"])
        self.assertEqual(saved["duration_ms"], saved["latency_ms"])

    def test_failed_and_multiple_calls_are_preserved(self):
        with request_telemetry(self.uid, "/test") as session:
            track_llm_call(model="claude-sonnet-4-5", system="a", user="b", response=self._response(2, 3), started=session.started)
            track_llm_call(model="claude-sonnet-4-5", system="a", user="b", error=TimeoutError(), started=session.started)
        self.assertEqual(session.saved["model_calls"], 2)
        self.assertEqual(session.saved["calls"][1]["status"], "failed")
        self.assertEqual(session.saved["calls"][1]["input_tokens"], 0)
        self.assertEqual(session.saved["errors"][0]["error_type"], "TimeoutError")

    def test_processing_steps_are_classified_and_exclude_nested_llm_time(self):
        with request_telemetry(self.uid, "/test") as session:
            with track_processing_step("Observe"):
                pass
            with track_processing_step("Investigate"):
                track_llm_call(model="claude-sonnet-4-5", system="a", user="b",
                               response=self._response(), started=session.started)
        saved = session.saved
        self.assertEqual([step["processing_type"] for step in saved["steps"]],
                         ["Non-LLM Processing", "Non-LLM Processing", "LLM Processing"])
        self.assertEqual(saved["processing"]["llm"]["step_count"], 1)
        self.assertEqual(saved["processing"]["non_llm"]["step_count"], 2)

    def test_full_workflow_records_each_non_llm_stage(self):
        with request_telemetry(self.uid, "/test") as session:
            pipeline.run_full(self.uid, self.dataset, "revenue", 2026, 2,
                              persist=False, use_llm=False)
        self.assertEqual([step["name"] for step in session.saved["steps"]],
                         ["Load dataset", "Observe", "Investigate", "Contest", "Act"])
        self.assertTrue(all(step["processing_type"] == "Non-LLM Processing"
                            for step in session.saved["steps"]))

    def test_llm_client_call_records_the_real_wrapper_path(self):
        client = LLMClient()
        response = self._response(13, 5)
        response.content = [SimpleNamespace(text='{"framing_note": "ok"}')]
        client._client = SimpleNamespace(
            messages=SimpleNamespace(create=lambda **_: response),
        )
        with request_telemetry(self.uid, "/test") as session:
            self.assertEqual(client._call("system", "user"), {"framing_note": "ok"})
        call = session.saved["calls"][0]
        self.assertEqual(call["model_name"], client.model)
        self.assertEqual(call["input_tokens"], 13)
        self.assertEqual(call["output_tokens"], 5)
        self.assertEqual(call["processing_type"], "LLM Processing")

    def test_llm_client_caches_identical_prompt_without_new_usage(self):
        client = LLMClient()
        response = self._response(13, 5)
        response.content = [SimpleNamespace(text='{"framing_note": "cached"}')]
        calls = 0
        def create(**_):
            nonlocal calls
            calls += 1
            return response
        client._client = SimpleNamespace(messages=SimpleNamespace(create=create))

        with request_telemetry(self.uid, "/test") as first:
            client._call("system", "context-a")
        with request_telemetry(self.uid, "/test") as repeated:
            client._call("system", "context-a")
        with request_telemetry(self.uid, "/test") as changed_context:
            client._call("system", "context-b")
        client.model = "claude-sonnet-other"
        with request_telemetry(self.uid, "/test") as changed_model:
            client._call("system", "context-a")

        self.assertEqual(calls, 3)
        self.assertTrue(first.saved["cache_miss"])
        self.assertEqual(first.saved["model_calls"], 1)
        self.assertGreater(first.saved["estimated_cost"], 0)
        self.assertTrue(repeated.saved["cache_hit"])
        self.assertEqual(repeated.saved["model_calls"], 0)
        self.assertEqual(repeated.saved["total_tokens"], 0)
        self.assertEqual(repeated.saved["estimated_cost"], 0)
        self.assertTrue(changed_context.saved["cache_miss"])
        self.assertTrue(changed_model.saved["cache_miss"])

    def test_run_endpoint_returns_runtime_telemetry(self):
        response = run_investigation(
            AnalysisRequest(kpi="revenue", year=2026, quarter=2, persist=False, use_llm=False),
            {"uid": self.uid, "role": "data_analyst"},
        )
        self.assertIn("telemetry", response)
        self.assertEqual(response["telemetry"]["endpoint"], "/api/investigations/run")
        self.assertEqual(response["telemetry"]["processing"]["non_llm"]["step_count"], 5)

    def test_missing_usage_uses_explicit_lightweight_fallback(self):
        self.assertEqual(estimate_tokens("abcd"), 1)
        session = TelemetrySession(self.uid, "/test")
        session.record_call(model="unknown", system="abcd", user="", response=SimpleNamespace(usage=None, content=[]), started=session.started)
        saved = session.persist()
        self.assertTrue(saved["tokens_estimated"])
        self.assertEqual(saved["estimated_cost"], 0)

    def test_persistence_failure_does_not_raise(self):
        session = TelemetrySession(self.uid, "/test")
        with patch("app.services.telemetry.TelemetryRepository.create", side_effect=OSError("disk unavailable")):
            saved = session.persist()
        self.assertEqual(saved["trace_id"], session.trace_id)

    def test_summary_aggregates_persisted_records(self):
        repo = TelemetryRepository()
        repo.create(self.uid, {"trace_id": "summary_a", "timestamp": "now", "endpoint": "/test", "model_calls": 1,
                               "input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "estimated_cost": 0.01,
                               "latency_ms": 100, "status": "success"})
        repo.create(self.uid, {"trace_id": "summary_b", "timestamp": "now", "endpoint": "/test", "model_calls": 2,
                               "input_tokens": 20, "output_tokens": 10, "total_tokens": 30, "estimated_cost": 0.02,
                               "latency_ms": 200, "status": "failed"})
        summary = telemetry_summary({"uid": self.uid})
        self.assertGreaterEqual(summary["total_requests"], 2)
        self.assertGreaterEqual(summary["total_model_calls"], 3)
        self.assertGreater(summary["total_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
