"""
Shared test setup.

The suite is written on the standard library's `unittest` so it runs anywhere —
`python3 -m unittest discover backend/tests` in a bare checkout, or `pytest`
(which collects unittest cases natively) once the requirements are installed.

Every test runs against an isolated temporary data directory, with no Firebase
credentials and no ANTHROPIC_API_KEY: the analytical core must be correct and
complete on its own.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

SAMPLE_CSV = ROOT / "sample_data" / "business_metrics_sample.csv"
SAMPLE_DOCS = ROOT / "sample_data" / "documents"

_STATE = {}


def setup_environment():
    if _STATE.get("ready"):
        return _STATE
    tmp = tempfile.mkdtemp(prefix="bi_ai_test_")
    os.environ["DATA_DIR"] = tmp
    os.environ["DB_BACKEND"] = "json"
    os.environ["ANTHROPIC_API_KEY"] = ""
    os.environ["AUTH_MODE"] = "demo"

    from app.config import get_settings

    get_settings.cache_clear()
    get_settings()

    from app.db.repositories import DocumentRepository, UserRepository
    from app.rag.chunker import chunk_document
    from app.rag.extract import extract_text, guess_doc_type
    from app.rag.retriever import invalidate
    from app.services import dataset_service

    uid = "test_user"
    UserRepository().upsert_profile(uid, "test@example.com", "Test Analyst", "data_analyst")
    dataset = dataset_service.store_upload(uid, SAMPLE_CSV.name, SAMPLE_CSV.read_bytes())

    repo = DocumentRepository()
    for path in sorted(SAMPLE_DOCS.iterdir()):
        if path.is_file():
            text, kind = extract_text(path.name, path.read_bytes())
            repo.create(uid, {"filename": path.name, "doc_type": guess_doc_type(path.name, text),
                              "source_kind": kind, "word_count": len(text.split())},
                        chunk_document(text))
    invalidate(uid)

    df, schema = dataset_service.load(dataset)
    _STATE.update({"ready": True, "tmp": tmp, "uid": uid, "dataset": dataset,
                   "df": df, "schema": schema})
    return _STATE


class EngineTestCase(unittest.TestCase):
    """Base class that guarantees the shared fixture exists."""

    @classmethod
    def setUpClass(cls):
        state = setup_environment()
        cls.uid = state["uid"]
        cls.dataset = state["dataset"]
        cls.df = state["df"]
        cls.schema = state["schema"]

    def assertClose(self, a, b, rel=1e-9, msg=""):
        if b == 0:
            self.assertAlmostEqual(a, b, places=6, msg=msg)
        else:
            self.assertLessEqual(abs(a - b) / abs(b), rel, msg or f"{a} != {b}")
