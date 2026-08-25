"""
Turning a business question into something the engines can run.

The entry point to an investigation used to be a dropdown: the user picked a KPI
and a quarter, and the pipeline ran. This package replaces that with a question,
and resolves the question against the dataset's KPI Contract — which KPI is
being asked about, over what period, compared with what, along which dimensions.

The resolution is deterministic wherever it can be. A question that names a KPI
the contract already knows, in words the concept library already recognises,
never reaches a language model at all. The model is consulted only when the
deterministic pass is genuinely uncertain, and even then its answer is validated
back against the contract: it may choose among the KPIs that exist, and it may
not invent one.
"""
from .grounding import ground_question
from .periods import resolve_period
from .understanding import clear_intent_cache, interpret_question, interpret_question_cached

__all__ = ["ground_question", "resolve_period", "interpret_question",
           "interpret_question_cached", "clear_intent_cache"]
