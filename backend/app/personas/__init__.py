"""
Persona-aware presentation.

A persona decides how an investigation is *said* and what the reader should
*do* about it. It never decides what is true: the evidence, the ranking, the
confidence scores and the causal verdicts are computed once, deterministically,
and are identical for every persona.

This is deliberately separate from the auth role. A role is authorisation —
which fields the server is willing to send at all (see `api.redact`). A persona
is presentation — how the fields a reader is entitled to see get framed for the
decisions that reader actually makes.
"""
from .profiles import (PERSONAS, DEFAULT_PERSONA, PersonaProfile, persona_for,
                       persona_for_role, persona_options)

__all__ = ["PERSONAS", "DEFAULT_PERSONA", "PersonaProfile", "persona_for",
           "persona_for_role", "persona_options"]
