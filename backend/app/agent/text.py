"""
The text primitives every layer matches with.

Tokenising, normalising and phrase containment are shared by three callers that
must agree exactly: the member index files values under a normalised key, the
KPI search scores phrases against KPI names, and `query.grounding` generates the
phrases both are searched with. A second implementation of any of these is a
silent failure -- a value indexed under a key no question can ever generate
never matches, and nothing reports that it did not.

So there is one implementation, here, and the layers above import it.

`contains_phrase` deserves its own note. A plain substring test is not good
enough: "admissions" is a substring of "readmissions", and letting that count
would quietly make every question about admissions also a question about
readmissions -- two clinically different measures.
"""
from __future__ import annotations

import re
from typing import List

_TOKEN = re.compile(r"[a-z0-9_]+")


def tokens(text: str) -> List[str]:
    return _TOKEN.findall(str(text).lower())


def ngrams(token_list: List[str], max_n: int = 4) -> List[str]:
    """Longest first, so 'bed occupancy rate' wins over 'bed'."""
    out: List[str] = []
    for n in range(min(max_n, len(token_list)), 0, -1):
        for i in range(len(token_list) - n + 1):
            out.append("_".join(token_list[i:i + n]))
    return out


def normalise(text: str) -> str:
    return "_".join(tokens(text))


def contains_phrase(haystack: str, needle: str) -> bool:
    """Whether `needle` appears in `haystack` as a run of whole words."""
    # Split on underscores as well as whitespace: "bed_occupancy_rate" and
    # "Bed occupancy rate" have to compare equal, and the tokeniser keeps
    # underscores inside a token.
    hay = [w for w in normalise(haystack).split("_") if w]
    need = [w for w in normalise(needle).split("_") if w]
    if not need or len(need) > len(hay):
        return False
    return any(hay[i:i + len(need)] == need for i in range(len(hay) - len(need) + 1))
