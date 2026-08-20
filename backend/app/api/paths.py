from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_DATA_DIR = REPO_ROOT / "sample_data"
SAMPLE_CSV = SAMPLE_DATA_DIR / "business_metrics_sample.csv"
SAMPLE_TEMPLATE = SAMPLE_DATA_DIR / "business_metrics_TEMPLATE.csv"
SAMPLE_DOCS_DIR = SAMPLE_DATA_DIR / "documents"
