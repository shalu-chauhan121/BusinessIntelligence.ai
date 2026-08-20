"""Heading-aware chunking. Small, overlapping chunks retrieve better and cite better."""
from __future__ import annotations

import re
from typing import Dict, List

HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)$")
TARGET_WORDS = 170
OVERLAP_WORDS = 40


def _blocks(text: str) -> List[Dict[str, str]]:
    """Split into (heading, body) blocks using markdown headings where present."""
    lines = text.splitlines()
    blocks: List[Dict[str, str]] = []
    heading, buffer = "", []
    for line in lines:
        m = HEADING_RE.match(line)
        if m:
            if buffer:
                blocks.append({"heading": heading, "text": "\n".join(buffer).strip()})
                buffer = []
            heading = m.group(2).strip()
        else:
            buffer.append(line)
    if buffer:
        blocks.append({"heading": heading, "text": "\n".join(buffer).strip()})
    return [b for b in blocks if b["text"]]


def chunk_document(text: str) -> List[Dict[str, str]]:
    chunks: List[Dict[str, str]] = []
    for block in _blocks(text):
        words = block["text"].split()
        if not words:
            continue
        if len(words) <= TARGET_WORDS:
            chunks.append({"heading": block["heading"], "text": block["text"].strip()})
            continue
        step = TARGET_WORDS - OVERLAP_WORDS
        for start in range(0, len(words), step):
            piece = words[start:start + TARGET_WORDS]
            if len(piece) < 25 and chunks:
                break
            chunks.append({"heading": block["heading"], "text": " ".join(piece)})
    if not chunks and text.strip():
        chunks.append({"heading": "", "text": text.strip()[:4000]})
    return chunks
