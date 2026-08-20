"""Per-user retrieval facade used by the Investigate and Contest stages."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import re

from ..db.repositories import DocumentRepository
from .index import BM25Index

_MD_TABLE_SEP = re.compile(r"^\s*\|?[\s:\-|]+\|?\s*$")


def clean_snippet(text: str, limit: int = 420) -> str:
    """
    Turn a raw markdown chunk into a readable quotation.

    Evidence is quoted verbatim in the UI, so table pipes, heading hashes,
    emphasis markers and blockquote arrows have to be removed or the quote reads
    as source code rather than as something a person wrote.
    """
    lines = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or _MD_TABLE_SEP.match(line):
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|") if c.strip()]
            line = " · ".join(cells)
        line = re.sub(r"^[>#\-\*\s]+", "", line)
        line = line.replace("**", "").replace("__", "").replace("`", "")
        line = re.sub(r"\s{2,}", " ", line)
        if line:
            lines.append(line)
    snippet = " ".join(lines).strip()
    if len(snippet) > limit:
        snippet = snippet[: limit - 3].rsplit(" ", 1)[0] + "..."
    return snippet

_INDEX_CACHE: Dict[str, tuple] = {}


class Retriever:
    """Retrieves textual evidence from the documents *this user* uploaded."""

    def __init__(self, uid: str):
        self.uid = uid
        self.chunks = DocumentRepository().chunks_for_user(uid)
        key = (uid, len(self.chunks))
        cached = _INDEX_CACHE.get(uid)
        if cached and cached[0] == key:
            self.index = cached[1]
        else:
            self.index = BM25Index(self.chunks)
            _INDEX_CACHE[uid] = (key, self.index)

    @property
    def available(self) -> bool:
        return len(self.chunks) > 0

    def retrieve(self, query: str, top_k: int = 4, must_include: Optional[List[str]] = None,
                 exclude_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if not self.available:
            return []
        return self.index.search(query, top_k=top_k, must_include=must_include, exclude_ids=exclude_ids)

    # A passage below this absolute BM25 score is topically unrelated: including it
    # would let every hypothesis claim documentary support it does not have.
    MIN_EVIDENCE_SCORE = 4.0

    def retrieve_evidence(self, query: str, top_k: int = 3, must_include: Optional[List[str]] = None,
                          stance: str = "supporting", min_score: Optional[float] = None) -> List[Dict[str, Any]]:
        """Retrieval results shaped as citable evidence items."""
        floor = self.MIN_EVIDENCE_SCORE if min_score is None else min_score
        out = []
        for hit in self.retrieve(query, top_k=top_k, must_include=must_include):
            if hit["score"] < floor:
                continue
            snippet = clean_snippet(hit["text"])
            if not snippet:
                continue
            out.append({
                "type": "unstructured",
                "stance": stance,
                "source": hit["document_name"],
                "doc_type": hit["doc_type"],
                "section": hit["heading"],
                "quote": snippet,
                "relevance": hit["relevance"],
                "strength": hit.get("strength", 0.0),
                "score": hit["score"],
                "chunk_id": hit["chunk_id"],
                "document_id": hit["document_id"],
            })
        return out


def invalidate(uid: str) -> None:
    _INDEX_CACHE.pop(uid, None)
