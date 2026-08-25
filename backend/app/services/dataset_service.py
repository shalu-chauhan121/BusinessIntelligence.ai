"""Loading, validating and caching a user's uploaded structured dataset."""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import replace
from typing import Any, Dict, Optional, Tuple

import pandas as pd

log = logging.getLogger(__name__)

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


def attach_contract(uid: str, dataset: Dict[str, Any], df: pd.DataFrame,
                    schema: DatasetSchema) -> DatasetSchema:
    """
    Resolve this dataset's KPI Contract and hang it on the schema.

    The contract is the source of truth for what a KPI means. A dataset that has
    none — anything uploaded before this layer existed, or one whose owner has
    not opened the KPI Studio — gets a provisional contract generated from the
    general library, which reproduces the previous behaviour exactly.

    A failure here must never make a dataset unanalysable: the engines fall back
    to the seed registry when `contract_resolver` is None.
    """
    from ..kpi import service as kpi_service

    try:
        contract = kpi_service.get_or_bootstrap(uid, dataset, df, schema)
        resolver = kpi_service.resolver_for(contract)
    except Exception as exc:                       # pragma: no cover - defensive
        log.warning("Could not resolve a KPI contract for %s: %s", dataset.get("_id"), exc)
        return schema

    if not resolver:
        return schema

    schema.contract_resolver = resolver
    schema.contract_status = contract.status
    schema.contract_version = contract.version
    # The business context the contract was screened under. Rebuilt rather than
    # re-detected: detection needs library matches that only exist at discovery
    # time, and the contract already records everything the context needs.
    try:
        from ..kpi.domain import domain_context_from_contract
        schema.domain = domain_context_from_contract(contract)
    except Exception as exc:                       # pragma: no cover - defensive
        log.warning("Could not rebuild domain context for %s: %s", dataset.get("_id"), exc)
    # The contract decides which KPIs exist. Only those whose source columns are
    # actually present in the frame can be offered.
    columns = set(df.columns)
    schema.available_kpis = [k for k, c in resolver.items()
                             if set(c.source_fields) <= columns]
    return schema


def load(dataset: Dict[str, Any], uid: Optional[str] = None
         ) -> Tuple[pd.DataFrame, DatasetSchema]:
    """
    Load + prepare a dataset, cached on (path, mtime).

    When `uid` is supplied the dataset's KPI Contract is resolved and attached.
    The cache holds the contract-free schema, because a contract can be edited
    between two loads of the same unchanged file.
    """
    path = dataset["path"]
    if not os.path.exists(path):
        raise DatasetError("The stored dataset file is missing. Please re-upload it.")
    mtime = os.path.getmtime(path)
    cached = _CACHE.get(path)
    if cached and cached[0] == mtime:
        df, schema = cached[1], cached[2]
    else:
        df_raw = pd.read_csv(path)
        schema = detect_schema(df_raw)
        df = prepare(df_raw, schema)
        _CACHE[path] = (mtime, df, schema)

    if uid:
        schema = attach_contract(uid, dataset, df, replace(schema))
    return df, schema


def clear_cache(path: Optional[str] = None) -> None:
    if path:
        _CACHE.pop(path, None)
    else:
        _CACHE.clear()
