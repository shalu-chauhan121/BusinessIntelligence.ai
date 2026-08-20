"""Text extraction from uploaded unstructured documents."""
from __future__ import annotations

import io
import os
from typing import Tuple

SUPPORTED = {".txt", ".md", ".markdown", ".csv", ".pdf", ".log", ".json"}


def extract_text(filename: str, raw: bytes) -> Tuple[str, str]:
    """Return (text, detected_kind)."""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED:
        raise ValueError(
            f"Unsupported document type '{ext}'. Supported: {', '.join(sorted(SUPPORTED))}"
        )

    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:                        # pragma: no cover
            raise ValueError("pypdf is required to ingest PDF documents.") from exc
        reader = PdfReader(io.BytesIO(raw))
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            pages.append(f"[page {i}]\n{page.extract_text() or ''}")
        text = "\n\n".join(pages)
        if not text.strip():
            raise ValueError(
                "No selectable text found in this PDF (it may be a scan). "
                "Upload a text-based PDF or paste the content as .md/.txt."
            )
        return text, "pdf"

    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw.decode(encoding), ext.lstrip(".")
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode the document as text.")


def guess_doc_type(filename: str, text: str) -> str:
    """Coarse document classification, used to label retrieved evidence."""
    name = filename.lower()
    head = text[:1500].lower()
    rules = [
        ("customer_feedback", ("feedback", "verbatim", "nps", "review", "survey", "complaint")),
        ("operations_report", ("operations", "ops ", "incident", "logistics", "supply", "fulfil", "fulfill")),
        ("market_intelligence", ("market", "competitor", "competitive", "intelligence", "share")),
        ("management_commentary", ("commentary", "management", "board", "cfo", "executive")),
        ("sales_report", ("sales", "pipeline", "quota", "region review")),
        ("financial_report", ("financial", "p&l", "income statement", "budget", "forecast")),
    ]
    for label, keys in rules:
        if any(k in name for k in keys) or any(k in head for k in keys):
            return label
    return "business_document"
