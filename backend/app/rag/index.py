"""
Lexical retrieval index (BM25) built per user, in-process.

Why BM25 and not a vector database for the MVP:
  * it needs no embedding API key, no service and no cold start, so the demo is
    reproducible on any machine;
  * business evidence retrieval is heavily entity-driven ("North", "Product A",
    "stockout", "Nexora") where exact-term matching is a strength, not a weakness.

The `Retriever` interface below is intentionally the only thing the engines use,
so swapping in Chroma or PostgreSQL + pgvector later means implementing one class.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_%\.\-]*")
STOPWORDS = set("""a an the and or but if then than that this these those of in on at to for from by with
as is are was were be been being it its it's we our us they their them he she his her you your i not no
do does did have has had will would can could should may might must about into over under after before
during between per each any all some more most other such only own same so too very s t don now""".split())


def tokenize(text: str) -> List[str]:
    return [t for t in TOKEN_RE.findall((text or "").lower()) if t not in STOPWORDS and len(t) > 1]


class BM25Index:
    k1 = 1.5
    b = 0.75

    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        self.docs: List[List[str]] = [tokenize(f"{c.get('heading','')} {c.get('text','')}") for c in chunks]
        self.doc_len = [len(d) for d in self.docs]
        self.avg_len = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        self.tf: List[Counter] = [Counter(d) for d in self.docs]
        self.df: Counter = Counter()
        for d in self.docs:
            for term in set(d):
                self.df[term] += 1
        self.n = len(self.docs)

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def score(self, query_terms: List[str], i: int) -> float:
        if not self.doc_len[i]:
            return 0.0
        s = 0.0
        for term in query_terms:
            f = self.tf[i].get(term, 0)
            if not f:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avg_len or 1))
            s += self._idf(term) * (f * (self.k1 + 1)) / denom
        return s

    def search(self, query: str, top_k: int = 5, must_include: Optional[List[str]] = None,
               exclude_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        terms = tokenize(query)
        if not terms:
            return []
        phrase_terms = [t for t in (must_include or []) if t]
        exclude = set(exclude_ids or [])
        results = []
        for i, chunk in enumerate(self.chunks):
            if chunk.get("_id") in exclude:
                continue
            s = self.score(terms, i)
            if phrase_terms:
                blob = f"{chunk.get('heading','')} {chunk.get('text','')}".lower()
                hits = sum(1 for p in phrase_terms if p.lower() in blob)
                if hits == 0:
                    continue
                s *= 1.0 + 0.45 * hits
            if s > 0:
                results.append((s, i))
        results.sort(reverse=True)
        max_score = results[0][0] if results else 1.0
        out = []
        for s, i in results[:top_k]:
            c = self.chunks[i]
            out.append({
                "chunk_id": c.get("_id"),
                "document_id": c.get("document_id"),
                "document_name": c.get("document_name", "document"),
                "doc_type": c.get("doc_type", "business_document"),
                "heading": c.get("heading", ""),
                "text": c.get("text", ""),
                "score": round(float(s), 4),
                # relevance is relative to the best hit for this query (ranking/display);
                # strength is an ABSOLUTE saturating score, so a query with no good match
                # cannot promote a weak passage to "1.0 relevance" evidence.
                "relevance": round(float(s / max_score), 3) if max_score else 0.0,
                "strength": round(float(min(1.0, s / 9.0)), 3),
            })
        return out
