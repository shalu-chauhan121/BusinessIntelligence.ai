"""
Headless end-to-end run of the agent loop against the bundled sample data.
No web server, no Firebase required — but the loop has no deterministic
fallback, so a live `ANTHROPIC_API_KEY` is required (unlike the retired
4-stage demo this replaces).

    python3 scripts/run_pipeline_demo.py "What factors are affecting my profit?"

Writes the full `AgentAnswer` JSON to `demo_output/agent_answer.json`. Useful
for (a) verifying the agent loop end to end, (b) generating fixtures for the
frontend, (c) demonstrating the loop in a terminal during a demo.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("DATA_DIR", str(ROOT / "backend" / "data"))

from app.agent import loop                                          # noqa: E402
from app.agent.context import AgentContext                          # noqa: E402
from app.db.repositories import DocumentRepository, UserRepository  # noqa: E402
from app.llm.client import get_llm                                  # noqa: E402
from app.rag.chunker import chunk_document                          # noqa: E402
from app.rag.extract import extract_text, guess_doc_type            # noqa: E402
from app.rag.retriever import invalidate                            # noqa: E402
from app.services import dataset_service                            # noqa: E402

SAMPLE_CSV = ROOT / "sample_data" / "business_metrics_sample.csv"
SAMPLE_DOCS = ROOT / "sample_data" / "documents"

DEFAULT_QUESTION = "What is driving the change in revenue this quarter?"


def seed(uid: str = "demo_cli_user") -> dict:
    UserRepository().upsert_profile(uid, "demo@businessintelligence.ai", "Demo Analyst", "data_analyst")
    ds = dataset_service.store_upload(uid, SAMPLE_CSV.name, SAMPLE_CSV.read_bytes())
    repo = DocumentRepository()
    existing = {d["filename"] for d in repo.list_for_user(uid)}
    for path in sorted(SAMPLE_DOCS.iterdir()):
        if not path.is_file() or path.name in existing:
            continue
        text, kind = extract_text(path.name, path.read_bytes())
        repo.create(uid, {"filename": path.name, "doc_type": guess_doc_type(path.name, text),
                          "source_kind": kind, "word_count": len(text.split())},
                    chunk_document(text))
    invalidate(uid)
    return ds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    ap.add_argument("--out", default=str(ROOT / "demo_output" / "agent_answer.json"))
    args = ap.parse_args()

    uid = "demo_cli_user"
    ds = seed(uid)
    ctx = AgentContext.build(uid, ds)
    result = loop.answer(args.question, ctx, llm=get_llm())

    payload = {
        "status": result.status, "question": result.question, "answer": result.answer,
        "evidence": result.evidence, "kpis_used": result.kpis_used,
        "periods_used": result.periods_used, "engine": result.engine,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))

    line = "=" * 78
    print(line)
    print(f"QUESTION  {result.question}")
    print(f"STATUS    {result.status}   turns: {result.engine.get('turns')}   "
          f"model: {result.engine.get('model')}   {result.engine.get('seconds')}s")
    print(line)
    print(f"ANSWER\n{result.answer}")
    print(line)
    print(f"KPIs used: {', '.join(result.kpis_used) or '(none)'}")
    print(f"Evidence steps: {len(result.evidence)}")
    for step in result.evidence:
        flag = "  [error]" if step.get("is_error") else ""
        print(f"   {step['step']}. {step['tool']}({step.get('args') or {}}){flag}")
    print(line)
    print(f"written: {out}")


if __name__ == "__main__":
    main()
