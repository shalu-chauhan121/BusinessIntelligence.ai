"""Loading, validating and caching a user's uploaded structured dataset."""
from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from ..config import get_settings
from ..db.repositories import DatasetRepository
from ..engines.metrics import DatasetSchema, detect_schema, prepare

_CACHE: Dict[str, Tuple[float, pd.DataFrame, DatasetSchema]] = {}


class DatasetError(ValueError):
    pass


def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    from io import BytesIO

    for kwargs in ({}, {"sep": ";"}, {"sep": "\t"}, {"encoding": "latin-1"}):
        try:
            df = pd.read_csv(BytesIO(raw), **kwargs)
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    raise DatasetError("Could not parse the file as CSV. Check the delimiter and encoding.")


def validate_and_describe(df: pd.DataFrame) -> DatasetSchema:
    if len(df) == 0:
        raise DatasetError("The uploaded file has no rows.")
    try:
        schema = detect_schema(df)
    except ValueError as exc:
        raise DatasetError(str(exc)) from exc
    if not schema.available_kpis:
        raise DatasetError(
            "No numeric metric columns found. At minimum include a 'revenue' column "
            "(see sample_data/DATA_FORMAT.md)."
        )
    return schema


def store_upload(uid: str, filename: str, raw: bytes) -> Dict[str, Any]:
    s = get_settings()
    df = read_csv_bytes(raw)
    schema = validate_and_describe(df)

    digest = hashlib.sha256(raw).hexdigest()[:16]
    user_dir = os.path.join(s.upload_dir, uid)
    os.makedirs(user_dir, exist_ok=True)
    safe_name = os.path.basename(filename).replace(" ", "_") or "dataset.csv"
    path = os.path.join(user_dir, f"{digest}_{safe_name}")
    with open(path, "wb") as fh:
        fh.write(raw)

    repo = DatasetRepository()
    meta = {
        "filename": safe_name,
        "path": path,
        "size_bytes": len(raw),
        "checksum": digest,
        "schema": schema.to_dict(),
        "is_active": True,
    }
    ds = repo.create(uid, meta)
    repo.set_active(uid, ds["_id"])
    return ds


def load(dataset: Dict[str, Any]) -> Tuple[pd.DataFrame, DatasetSchema]:
    """Load + prepare a dataset, cached on (path, mtime)."""
    path = dataset["path"]
    if not os.path.exists(path):
        raise DatasetError("The stored dataset file is missing. Please re-upload it.")
    mtime = os.path.getmtime(path)
    cached = _CACHE.get(path)
    if cached and cached[0] == mtime:
        return cached[1], cached[2]

    df_raw = pd.read_csv(path)
    schema = detect_schema(df_raw)
    df = prepare(df_raw, schema)
    _CACHE[path] = (mtime, df, schema)
    return df, schema


def clear_cache(path: Optional[str] = None) -> None:
    if path:
        _CACHE.pop(path, None)
    else:
        _CACHE.clear()
