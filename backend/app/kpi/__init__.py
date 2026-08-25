"""
The KPI Definition and KPI Contract layer.

The KPI Contract is the authoritative analytical definition of every KPI in the
platform: what it means, how it is computed, from which fields and sources, at
what grain, under which calendar and hierarchy, with what provenance and
approval state. Everything downstream — Observe, Investigate, Contest, Act —
resolves KPIs through a compiled contract rather than through a hard-coded
Python dictionary.

    profiling  ->  what does each column MEAN?
    library    ->  which broadly-applicable KPIs does this data support?
    derivation ->  which derived KPIs are semantically valid, not merely computable?
    screening  ->  (optional) an LLM judges semantics; it never computes a number
    conflicts  ->  what is ambiguous or contradictory, and must a human decide?
    contract   ->  the artefact
    resolver   ->  the artefact, made executable
"""
