"""Authentication, the swappable document store, and role-based authorisation."""
from __future__ import annotations

import unittest

from app.api.redact import redact_contest, redact_observation, redact_result
from app.auth.firebase_auth import AuthError, auth_mode, issue_demo_token, verify_token
from app.db.repositories import DatasetRepository, UserRepository, get_store
from app.engines.observe import Timeframe, observe
from app.services import pipeline

from .base import EngineTestCase


class TestDocumentStore(EngineTestCase):
    def test_store_is_mongo_shaped(self):
        col = get_store().collection("unit_test_collection")
        col.delete_many({})
        col.insert_one({"uid": "u1", "name": "a", "n": 1})
        col.insert_one({"uid": "u1", "name": "b", "n": 2})
        col.insert_one({"uid": "u2", "name": "c", "n": 3})

        self.assertEqual(col.count({"uid": "u1"}), 2)
        self.assertEqual(col.find_one({"name": "c"})["uid"], "u2")
        self.assertEqual(len(col.find({"n": {"$in": [1, 3]}})), 2)
        self.assertEqual(len(col.find({"n": {"$gte": 2}})), 2)

        col.update_one({"name": "a"}, {"n": 99})
        self.assertEqual(col.find_one({"name": "a"})["n"], 99)

        sorted_docs = col.find({"uid": "u1"}, sort=("n", -1))
        self.assertEqual(sorted_docs[0]["n"], 99)

        self.assertTrue(col.delete_one({"name": "a"}))
        self.assertEqual(col.delete_many({"uid": "u1"}), 1)

    def test_every_record_is_scoped_to_a_user(self):
        repo = DatasetRepository()
        self.assertTrue(repo.list_for_user(self.uid))
        self.assertEqual(repo.list_for_user("intruder"), [])
        self.assertIsNone(repo.get("intruder", self.dataset["_id"]))


class TestAuthentication(EngineTestCase):
    def test_demo_mode_is_active_without_firebase_credentials(self):
        self.assertEqual(auth_mode(), "demo")

    def test_demo_token_round_trip(self):
        token = issue_demo_token("demo_abc", "leader@example.com", "Leader")
        identity = verify_token(token)
        self.assertEqual(identity["uid"], "demo_abc")
        self.assertEqual(identity["email"], "leader@example.com")
        self.assertFalse(identity["verified"], "demo tokens must never claim to be verified")

    def test_garbage_token_is_rejected(self):
        with self.assertRaises(AuthError):
            verify_token("not-a-token")
        with self.assertRaises(AuthError):
            verify_token("")


class TestRolesAndAuthorisation(EngineTestCase):
    def test_role_can_be_set_at_signup_and_changed(self):
        users = UserRepository()
        users.upsert_profile("role_user", "r@example.com", "R", "business_leader")
        self.assertEqual(users.get("role_user")["role"], "business_leader")
        users.set_role("role_user", "data_analyst")
        self.assertEqual(users.get("role_user")["role"], "data_analyst")
        with self.assertRaises(ValueError):
            users.set_role("role_user", "ceo")

    def test_analyst_sees_statistical_internals_and_leader_does_not(self):
        observation = observe(self.df, self.schema, "revenue", Timeframe(2026, 2))
        analyst_view = redact_observation(observation, analyst=True)
        leader_view = redact_observation(observation, analyst=False)

        self.assertIn("robust_z", analyst_view["significance"])
        self.assertIn("historical_changes", analyst_view["significance"])
        self.assertTrue(analyst_view["drivers"])

        self.assertNotIn("robust_z", leader_view["significance"])
        self.assertNotIn("historical_changes", leader_view["significance"])
        self.assertEqual(leader_view["drivers"], {})
        # the leader still gets the meaning, in plain language
        self.assertIn("explanation", leader_view["significance"])
        self.assertTrue(leader_view["top_drivers"])

    def test_leader_does_not_receive_the_evidence_ledger(self):
        result = pipeline.run_full(self.uid, self.dataset, "revenue", 2026, 2,
                                   persist=False, use_llm=False)
        leader = redact_contest(result["contest"], analyst=False)
        analyst = redact_contest(result["contest"], analyst=True)
        self.assertIn("score_ledger", analyst["hypotheses"][0]["scoring"])
        self.assertNotIn("score_ledger", leader["hypotheses"][0]["scoring"])
        # but the conclusion and the reasoning trail remain visible to both
        self.assertTrue(leader["hypotheses"][0]["reasoning_trail"])
        self.assertIn("confidence", leader["hypotheses"][0]["scoring"])

    def test_redaction_does_not_mutate_the_source_result(self):
        result = pipeline.run_full(self.uid, self.dataset, "revenue", 2026, 2,
                                   persist=False, use_llm=False)
        redact_result(result, {"role": "business_leader"})
        self.assertIn("robust_z", result["observe"]["significance"])


if __name__ == "__main__":
    unittest.main()
