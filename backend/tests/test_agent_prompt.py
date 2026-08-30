"""
`agent_system` / `AGENT_GUARDRAIL` -- the agent loop's system prompt.

Every prompt above it in `prompts.py` ends with `Return JSON of exactly this
shape: {...}`, and `GUARDRAIL` rule 5 mandates "reply with valid JSON only".
Both are exactly wrong for a loop whose entire output is prose. This pins
that the agent prompt does not carry either instruction, that rules 1-4 of
the old guardrail survive in meaning (restated for a model that fetches its
own numbers rather than being handed them), and that seeding actually
happens through the same tool-dispatch path the loop itself uses -- not a
second, divergent way of describing the dataset.
"""
from __future__ import annotations

import unittest

from app.agent.context import AgentContext
from app.llm.prompts import AGENT_GUARDRAIL, AGENT_SYSTEM, agent_system

from .base import EngineTestCase


class TestTheAgentPromptDoesNotAskForJson(unittest.TestCase):
    def test_no_valid_json_only_instruction(self):
        self.assertNotIn("valid JSON only", AGENT_SYSTEM)
        self.assertNotIn("valid JSON only", AGENT_GUARDRAIL)

    def test_no_json_shape_template_block(self):
        # Every JSON-shape prompt in this file follows "Return JSON of
        # exactly this shape:" with a literal `{...}` block. The agent
        # prompt mentions the phrase only to tell the model not to do it,
        # in a sentence with no braces at all.
        self.assertEqual(AGENT_SYSTEM.count("{"), 0)
        self.assertEqual(AGENT_SYSTEM.count("}"), 0)

    def test_instructs_plain_text_output(self):
        self.assertIn("plain text", AGENT_SYSTEM)


class TestTheGuardrailRulesSurviveInMeaning(unittest.TestCase):
    """Not exact wording -- the agent guardrail is a rewrite, not a copy --
    but the four load-bearing constraints from the old `GUARDRAIL` must
    still be present in some form."""

    def test_rule_1_now_requires_numbers_to_come_from_a_tool_call(self):
        self.assertIn("tool call", AGENT_GUARDRAIL)
        self.assertIn("invent", AGENT_GUARDRAIL)

    def test_rule_2_still_forbids_claiming_causation(self):
        self.assertIn("Do not claim causation", AGENT_GUARDRAIL)

    def test_rule_3_still_forbids_resolving_genuine_ambiguity(self):
        self.assertIn("Do not resolve genuine ambiguity", AGENT_GUARDRAIL)

    def test_rule_4_still_forbids_treating_confidence_as_probability(self):
        self.assertIn("probability", AGENT_GUARDRAIL)

    def test_adds_a_rule_for_recovering_from_a_typed_tool_error(self):
        self.assertIn("valid_alternatives", AGENT_GUARDRAIL)

    def test_adds_a_rule_for_an_insufficient_result(self):
        self.assertIn("insufficient", AGENT_GUARDRAIL)


class TestSeedingUsesTheSameToolDispatchPathTheLoopItselfUses(EngineTestCase):
    def test_seeded_kpi_keys_and_row_count_appear_in_the_prompt(self):
        ctx = AgentContext.build(self.uid, self.dataset)
        described, is_error_1 = ctx.registry.dispatch("describe_dataset", {})
        listed, is_error_2 = ctx.registry.dispatch("list_kpis", {})
        self.assertFalse(is_error_1)
        self.assertFalse(is_error_2)
        seed = {**described, **listed}

        prompt = agent_system(seed)

        self.assertIn(str(described["rows"]), prompt)
        first_kpi_key = listed["kpis"][0]["key"]
        self.assertIn(first_kpi_key, prompt)

    def test_the_seeded_prompt_still_carries_the_base_agent_system_text(self):
        ctx = AgentContext.build(self.uid, self.dataset)
        described, _ = ctx.registry.dispatch("describe_dataset", {})
        prompt = agent_system(described)
        self.assertIn("You are answering a business question", prompt)


if __name__ == "__main__":
    unittest.main()
