"""
Headless end-to-end run of the four-stage pipeline against the bundled sample
data. No web server, no Firebase, no API key required.

    python3 scripts/run_pipeline_demo.py [--kpi revenue] [--year 2026] [--quarter 2]

Writes the full investigation JSON to `demo_output/investigation.json`.
Useful for (a) verifying the analysis layer, (b) generating fixtures for the
frontend, (c) demonstrating the pipeline in a terminal during a demo.
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

from app.db.repositories import DocumentRepository, UserRepository  # noqa: E402
from app.rag.chunker import chunk_document                          # noqa: E402
from app.rag.extract import extract_text, guess_doc_type            # noqa: E402
from app.rag.retriever import invalidate                            # noqa: E402
from app.services import dataset_service, pipeline                  # noqa: E402

SAMPLE_CSV = ROOT / "sample_data" / "business_metrics_sample.csv"
SAMPLE_DOCS = ROOT / "sample_data" / "documents"


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
    ap.add_argument("--kpi", default="revenue")
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--quarter", type=int, default=2)
    ap.add_argument("--comparison", default="previous_period",
                    choices=["previous_period", "year_over_year"])
    ap.add_argument("--out", default=str(ROOT / "demo_output" / "investigation.json"))
    args = ap.parse_args()

    uid = "demo_cli_user"
    ds = seed(uid)
    result = pipeline.run_full(uid, ds, args.kpi, args.year, args.quarter,
                               args.comparison, persist=True, use_llm=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str))

    obs, con, act = result["observe"], result["contest"], result["act"]
    line = "=" * 78
    print(line)
    print(f"OBSERVE   {obs['kpi_label']} {obs['change_pct']:+.1f}%  "
          f"({obs['timeframe']['pretty']} vs {obs['baseline_timeframe']['pretty']})")
    print(f"          verdict: {obs['verdict']}   robust z = {obs['significance']['robust_z']:.2f} "
          f"[{obs['significance']['method']}]")
    print("          top drivers: " + ", ".join(
        f"{d['name']} ({d['dimension']}) {d['contribution_pct']:.0f}%" for d in obs["top_drivers"][:4]))
    print(line)
    print(f"INVESTIGATE  {len(result['investigate']['hypotheses'])} competing hypotheses "
          f"from {result['investigate']['considered_count']} considered; "
          f"{result['investigate']['documents_indexed']} document chunks indexed")
    print(line)
    print("CONTEST   ranking:")
    for r in con["ranking"]:
        print(f"   {r['rank']}. {r['title']:<52} {r['confidence']:>3}%  {r['band']}")
    if con.get("ambiguity_note"):
        print(f"   ! {con['ambiguity_note']}")
    print(line)
    print("ACT")
    print(f"   {act['narrative']['headline']}")
    for r in act["recommendations"]:
        print(f"   [{r['priority']}] {r['title']}  (from: {r['based_on']['hypothesis']})")
    print(line)
    print(f"written: {out}")


if __name__ == "__main__":
    main()
