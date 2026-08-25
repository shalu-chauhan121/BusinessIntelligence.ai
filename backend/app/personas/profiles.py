"""
The five personas, and what each one changes.

Every profile below alters two things and only two things: how the explanation
is written, and what the reader is advised to do. The quantitative substrate —
which hypotheses ranked where, how confident each is, what evidence supports or
contradicts it, and whether causality can be claimed at all — is computed before
any persona is consulted and is passed through untouched.

That invariant is the point. A leader and an analyst looking at the same
investigation must never be able to reach different conclusions about what
happened; they should only reach different conclusions about what to do next.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Presentation depth. This is not authorisation — the server has already decided
# what it is willing to send (see `api.redact`); this decides how much of what
# arrived is worth putting in front of this reader by default.
DetailLevel = str          # "full" | "standard" | "headline"
ActionHorizon = str        # "immediate" | "shift" | "quarter" | "strategic" | "analysis_cycle"


@dataclass(frozen=True)
class PersonaProfile:
    """How one kind of reader wants an investigation delivered."""

    key: str
    label: str
    description: str

    # -- presentation ------------------------------------------------------
    detail_level: DetailLevel
    show_uncertainty_math: bool     # robust-z, MAD, correlation coefficients
    show_evidence_ledger: bool      # how each confidence score was assembled
    show_method_notes: bool

    # -- recommendation framing -------------------------------------------
    recommendation_lens: str        # what kind of action this reader can take
    action_horizon: ActionHorizon
    vocabulary: str                 # how technical the language should be

    # -- prompt fragments --------------------------------------------------
    explanation_brief: str
    recommendation_brief: str

    def to_view(self, role: str) -> Dict[str, Any]:
        """The `PersonaView` the API returns so the client can match density."""
        return {
            "persona": self.key,
            "label": self.label,
            "role": role,
            "detail_level": self.detail_level,
            "show_uncertainty_math": self.show_uncertainty_math,
            "show_evidence_ledger": self.show_evidence_ledger,
            "show_method_notes": self.show_method_notes,
            "recommendation_lens": self.recommendation_lens,
            "action_horizon": self.action_horizon,
        }


BUSINESS_ANALYST = PersonaProfile(
    key="business_analyst",
    label="Business Analyst",
    description="Analytical depth: drivers, evidence, method and what to investigate next.",
    detail_level="full",
    show_uncertainty_math=True,
    show_evidence_ledger=True,
    show_method_notes=True,
    recommendation_lens="analytical follow-ups — which drivers to decompose next, "
                        "what evidence would resolve the open questions, which data gaps to close",
    action_horizon="analysis_cycle",
    vocabulary="precise and statistical; name the method and the test",
    explanation_brief=(
        "Write for an analyst who will check your work. Lead with the KPI movement and its "
        "magnitude, then the contributing drivers with their contribution shares, then the "
        "evidence for and against the leading explanation, then the method and its limits. "
        "Name the statistical basis where one exists. Do not simplify away uncertainty."
    ),
    recommendation_brief=(
        "Recommend the next ANALYTICAL steps, not operational ones. What should be decomposed "
        "further, which comparison would discriminate between the surviving explanations, what "
        "evidence is missing and how it could be obtained, which data-quality gap undermines "
        "the conclusion. The reader runs analyses; they do not run the business."
    ),
)

BUSINESS_MANAGER = PersonaProfile(
    key="business_manager",
    label="Business Manager",
    description="Operational interventions and the areas or teams affected.",
    detail_level="standard",
    show_uncertainty_math=False,
    show_evidence_ledger=False,
    show_method_notes=True,
    recommendation_lens="operational interventions and the areas, teams or processes affected",
    action_horizon="quarter",
    vocabulary="plain business language; concrete about areas and owners",
    explanation_brief=(
        "Write for a manager accountable for an operational area. Lead with what changed and "
        "where it concentrated — which segment, department, region or channel. Say how confident "
        "the explanation is in plain words. Keep the statistics out of the prose; the reader "
        "needs to know where to intervene, not how significance was computed."
    ),
    recommendation_brief=(
        "Recommend OPERATIONAL interventions for the areas actually implicated by the evidence. "
        "Name the area, what to change there, and who would own it. Prefer actions that can be "
        "started this quarter. Do not recommend strategy, and do not recommend further analysis "
        "unless the evidence is genuinely too weak to act on."
    ),
)

BUSINESS_LEADER = PersonaProfile(
    key="business_leader",
    label="Business Leader",
    description="Decision-oriented: impact, priorities and the call to make.",
    detail_level="headline",
    show_uncertainty_math=False,
    show_evidence_ledger=False,
    show_method_notes=False,
    recommendation_lens="strategic implications, priorities and the decision to take",
    action_horizon="strategic",
    vocabulary="direct and non-technical; lead with consequence",
    explanation_brief=(
        "Write for a leader who has ninety seconds. Lead with the business consequence, then the "
        "most likely reason, then what is still uncertain — in that order, in three short "
        "paragraphs at most. No statistics, no method, no hedging language beyond what honesty "
        "requires. If the evidence does not support a confident answer, say so in one sentence "
        "rather than burying it."
    ),
    recommendation_brief=(
        "Recommend DECISIONS, not tasks. What should be prioritised, what trade-off is now on "
        "the table, what should be escalated or funded or stopped. Frame each around the "
        "business consequence of acting versus not acting. The reader allocates resources and "
        "sets priorities; they do not execute the fix themselves."
    ),
)

DOMAIN_SPECIALIST = PersonaProfile(
    key="domain_specialist",
    label="Domain Specialist",
    description="Domain mechanisms and technical actions in the domain's own language.",
    detail_level="full",
    show_uncertainty_math=True,
    show_evidence_ledger=True,
    show_method_notes=True,
    recommendation_lens="domain-technical and operational actions, in the domain's own vocabulary",
    action_horizon="quarter",
    vocabulary="the domain's own terminology, used precisely",
    explanation_brief=(
        "Write for a specialist in this domain who knows its mechanisms better than you do. Use "
        "the domain's own vocabulary throughout. Lead with the mechanism — how the observed "
        "movement would actually come about in an operation of this kind — then the evidence "
        "that supports or undercuts that mechanism. Do not explain domain basics; the reader "
        "knows them. Do flag where the data is too coarse to distinguish two mechanisms."
    ),
    recommendation_brief=(
        "Recommend DOMAIN-TECHNICAL actions: the specific operational or clinical or process "
        "levers a specialist in this field would reach for, named in the domain's own terms. "
        "Where a mechanism is plausible but unproven by the data, say what the specialist could "
        "check from their own operational knowledge that the dataset cannot show."
    ),
)

OPERATIONAL_USER = PersonaProfile(
    key="operational_user",
    label="Operational User",
    description="Concrete immediate actions for the next shift or day.",
    detail_level="headline",
    show_uncertainty_math=False,
    show_evidence_ledger=False,
    show_method_notes=False,
    recommendation_lens="concrete immediate actions and what needs attention now",
    action_horizon="immediate",
    vocabulary="plain, direct, task-shaped",
    explanation_brief=(
        "Write for someone working the floor who needs to know what is going on right now. One "
        "short paragraph: what changed, where, and whether it is still happening. No statistics, "
        "no strategy, no history beyond the comparison period. If nothing about this requires "
        "their attention today, say that plainly."
    ),
    recommendation_brief=(
        "Recommend CONCRETE IMMEDIATE actions for the next shift or day — things a person can "
        "actually start now, in the area they work in. No strategy, no analysis, no anything "
        "that takes a quarter. If the honest answer is that this needs a decision above their "
        "level, say what to escalate and to whom rather than inventing a task."
    ),
)

PERSONAS: Dict[str, PersonaProfile] = {
    p.key: p for p in (BUSINESS_ANALYST, BUSINESS_MANAGER, BUSINESS_LEADER,
                       DOMAIN_SPECIALIST, OPERATIONAL_USER)
}

# When no persona is known, present neutrally rather than inventing one. The
# analyst framing is the neutral choice because it withholds nothing and asserts
# no decision context the reader may not have.
DEFAULT_PERSONA = "business_analyst"

# A persona defaults from the auth role, then the user may change it freely.
# This is a starting point, never a constraint: persona is not authorisation.
ROLE_DEFAULT_PERSONA: Dict[str, str] = {
    "data_analyst": "business_analyst",
    "business_leader": "business_leader",
}


def persona_for(key: Optional[str]) -> PersonaProfile:
    """
    The profile for a persona key.

    An unknown or missing key gets the neutral analyst framing rather than a
    guess — inventing a decision context for a reader whose role we do not know
    is how a recommendation ends up aimed at the wrong person.
    """
    return PERSONAS.get(key or "", PERSONAS[DEFAULT_PERSONA])


def persona_for_role(role: Optional[str]) -> str:
    """The persona a user of this role starts with, before they choose."""
    return ROLE_DEFAULT_PERSONA.get(role or "", DEFAULT_PERSONA)


def persona_options() -> List[Dict[str, str]]:
    """The switchable personas, for the settings UI."""
    return [{"key": p.key, "label": p.label, "description": p.description,
             "action_horizon": p.action_horizon}
            for p in PERSONAS.values()]
