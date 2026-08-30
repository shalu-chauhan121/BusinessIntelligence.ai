"""Loading, validating and caching a user's uploaded structured dataset."""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

log = logging.getLogger(__name__)

from ..config import get_settings
from ..db.repositories import DatasetRepository, now_iso
from ..agent.dimensions import MemberCatalogue
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


def _ingest_sources(user_dir: str, files: Sequence[Tuple[str, bytes]],
                    source_meta: Optional[Dict[str, Dict[str, str]]] = None
                    ) -> Tuple[List[Any], List[Dict[str, str]]]:
    """
    Build a `SourceFile` per uploaded file and retain the raw bytes on disk.

    Retaining the originals is what lets a resolved conflict be rebuilt without
    asking the user to re-upload. `source_meta` is refresh metadata the caller
    may supply per filename (`cadence`, `last_refresh_at`) — never a mapping;
    the fields still have to already mean what the KPI Contract says they mean.
    """
    from . import reconciliation as recon

    source_meta = source_meta or {}
    sources, retained = [], []
    for filename, raw in files:
        frame = read_csv_bytes(raw)
        meta = source_meta.get(filename, {}) or {}
        try:
            src = recon.build_source(
                filename, frame,
                last_refresh_at=meta.get("last_refresh_at", ""),
                last_refresh_basis="source_reported" if meta.get("last_refresh_at") else "",
                refresh_cadence=meta.get("cadence", "unknown"),
                upload_time=now_iso(),
            )
        except ValueError as exc:
            raise DatasetError(f"{filename}: {exc}") from exc
        src_path = os.path.join(user_dir, f"src_{src.source_id}.csv")
        with open(src_path, "wb") as fh:
            fh.write(raw)
        retained.append({"source_id": src.source_id, "filename": src.filename, "path": src_path})
        sources.append(src)
    return sources, retained


def _build_canonical(sources: Sequence[Any], as_of: Optional[str],
                     authoritative: Optional[Dict[str, str]], contract: Any):
    """Reconcile sources into a canonical frame + report, or raise `DatasetError`."""
    from . import reconciliation as recon

    resolved_as_of = as_of or now_iso()
    try:
        canonical, report = recon.reconcile(sources, resolved_as_of, contract=contract,
                                            authoritative=authoritative)
    except recon.ReconciliationError as exc:
        raise DatasetError(str(exc)) from exc

    if not len(canonical):
        blocking = [i["detail"] for i in report.issues if i.get("severity") == "blocking"]
        raise DatasetError(
            "No value could be reconciled from these sources. "
            + (" ".join(blocking) if blocking else "They share no overlapping business data."))
    return canonical, report


def _write_canonical_csv(user_dir: str, canonical: pd.DataFrame
                         ) -> Tuple[bytes, str, DatasetSchema, str]:
    raw_csv = canonical.to_csv(index=False).encode("utf-8")
    df_raw = read_csv_bytes(raw_csv)
    schema = validate_and_describe(df_raw)
    digest = hashlib.sha256(raw_csv).hexdigest()[:16]
    path = os.path.join(user_dir, f"{digest}_reconciled_view.csv")
    with open(path, "wb") as fh:
        fh.write(raw_csv)
    return raw_csv, digest, schema, path


def store_multisource_upload(uid: str, files: Sequence[Tuple[str, bytes]],
                             as_of: Optional[str] = None,
                             authoritative: Optional[Dict[str, str]] = None,
                             contract: Any = None,
                             source_meta: Optional[Dict[str, Dict[str, str]]] = None
                             ) -> Dict[str, Any]:
    """
    Reconcile several same-vocabulary source files into one canonical dataset.

    The reconciled frame becomes an ordinary dataset — same digest-path storage,
    same `meta["schema"]` shape — so every engine downstream sees a normal
    single-file dataset and the whole Observe->Investigate->Contest->Act pipeline
    needs no change at all. Per-source provenance and freshness ride alongside in
    `meta["sources"]`, which `load()` re-evaluates on every later call.

    Raises `DatasetError` rather than guessing when the sources are structurally
    incompatible, or when two of them contradict each other about the same cell
    and no authoritative source has been named.
    """
    from . import reconciliation as recon

    s = get_settings()
    user_dir = os.path.join(s.upload_dir, uid)
    os.makedirs(user_dir, exist_ok=True)

    sources, retained = _ingest_sources(user_dir, files, source_meta)
    canonical, report = _build_canonical(sources, as_of, authoritative, contract)
    raw_csv, digest, schema, path = _write_canonical_csv(user_dir, canonical)
    clear_cache(path)

    repo = DatasetRepository()
    ds = repo.create(uid, {
        "filename": "reconciled_view.csv",
        "path": path,
        "size_bytes": len(raw_csv),
        "checksum": digest,
        "schema": schema.to_dict(),
        "is_active": True,
        "sources": report.to_dict(),
        "source_files": retained,
        "source_meta": source_meta or {},
    })
    repo.set_active(uid, ds["_id"])

    # Now that the canonical dataset exists it can bootstrap a contract, and only
    # then can per-KPI coverage be worked out — the contract is what says which
    # fields each KPI needs. Recorded on the dataset for the UI and the pipeline.
    df, resolved_schema = load(ds, uid)
    live = _contract_for(uid, ds["_id"])
    report.kpi_coverage = recon.kpi_coverage(
        resolved_schema.contract_resolver or {}, canonical, report, contract=live)
    _record_source_conflicts(uid, ds["_id"], report)
    repo.col.update_one({"_id": ds["_id"]}, {"sources": report.to_dict()})
    ds["sources"] = report.to_dict()
    return ds


def _contract_for(uid: str, dataset_id: str):
    from ..kpi import service as kpi_service
    try:
        return kpi_service.load_current(uid, dataset_id)
    except Exception:                              # pragma: no cover - defensive
        return None


def _record_source_conflicts(uid: str, dataset_id: str, report) -> None:
    """
    Put source disagreements on the contract so the existing conflict panel and
    `resolve_conflict` flow handle them — no new UI, and no re-upload.
    """
    from ..kpi import service as kpi_service
    from ..kpi.contract import ConflictFlag

    contract = _contract_for(uid, dataset_id)
    if contract is None:
        return
    flags = [ConflictFlag.model_validate(i) for i in report.issues
             if i.get("kind") in ("source_disagreement", "grain_mismatch")]
    if not flags:
        return
    keep = {f.conflict_id for f in flags}
    contract.conflicts = [c for c in contract.conflicts if c.conflict_id not in keep] + flags
    # Name the KPIs each disputed field actually blocks, so the panel shows the cost.
    resolver = kpi_service.resolver_for(contract)
    for flag in flags:
        affected = [k for k, c in resolver.items()
                    if set(flag.affected_fields) & set(getattr(c, "source_fields", []) or [])]
        flag.affected_kpis = sorted(affected)
    kpi_service.save(contract)


def _authoritative_from_contract(contract: Any) -> Dict[str, str]:
    """Resolved `source_disagreement` conflicts, as measure -> winning source_id."""
    authoritative: Dict[str, str] = {}
    for conflict in getattr(contract, "conflicts", []) or []:
        if conflict.kind != "source_disagreement" or not conflict.resolved:
            continue
        option = next((o for o in conflict.resolution_options
                       if o.option_id == conflict.resolution_option_id), None)
        effect = dict(getattr(option, "effect", {}) or {})
        if effect.get("measure") and effect.get("authoritative_source"):
            authoritative[effect["measure"]] = effect["authoritative_source"]
    return authoritative


def rebuild_reconciled_view(uid: str, dataset_id: str, as_of: Optional[str] = None
                            ) -> Optional[Dict[str, Any]]:
    """
    Rebuild a reconciled dataset AFTER a human has resolved a source
    disagreement through the existing KPI conflict panel.

    Updates the SAME dataset document in place, never creating a new one.
    Contracts are keyed by `(uid, dataset_id)` — creating a fresh dataset here
    would bootstrap a fresh contract and orphan the resolution the user just
    made, so the `_id` a reconciled dataset is created with is kept for its
    whole life.
    """
    from . import reconciliation as recon

    repo = DatasetRepository()
    ds = repo.get(uid, dataset_id)
    if not ds or not ds.get("source_files"):
        return None

    contract = _contract_for(uid, dataset_id)
    authoritative = _authoritative_from_contract(contract)
    if not authoritative:
        return None

    source_meta = ds.get("source_meta") or {}
    files = [(f["filename"], f["path"]) for f in ds["source_files"] if os.path.exists(f["path"])]
    if not files:
        return None

    sources = []
    for filename, path in files:
        with open(path, "rb") as fh:
            frame = read_csv_bytes(fh.read())
        meta = source_meta.get(filename, {}) or {}
        sources.append(recon.build_source(
            filename, frame,
            last_refresh_at=meta.get("last_refresh_at", ""),
            last_refresh_basis="source_reported" if meta.get("last_refresh_at") else "",
            refresh_cadence=meta.get("cadence", "unknown"),
            upload_time=now_iso(),
        ))

    canonical, report = _build_canonical(sources, as_of, authoritative, contract)

    s = get_settings()
    user_dir = os.path.join(s.upload_dir, uid)
    raw_csv, digest, schema, path = _write_canonical_csv(user_dir, canonical)

    old_path = ds.get("path")
    if old_path and old_path != path and os.path.exists(old_path):
        clear_cache(old_path)
        try:
            os.remove(old_path)
        except OSError:                              # pragma: no cover - best effort
            pass
    clear_cache(path)
    # The frame's members have just changed. Cached interpretations key on the
    # contract version, not on the file, so a filter bound to a member that no
    # longer exists would survive this rebuild without it.
    _clear_intent_cache(dataset_id)

    repo.col.update_one({"_id": dataset_id}, {
        "path": path, "size_bytes": len(raw_csv), "checksum": digest, "schema": schema.to_dict(),
    })
    updated = repo.get(uid, dataset_id)

    df, resolved_schema = load(updated, uid, as_of=as_of)
    report.kpi_coverage = recon.kpi_coverage(
        resolved_schema.contract_resolver or {}, canonical, report, contract=contract)
    _record_source_conflicts(uid, dataset_id, report)
    repo.col.update_one({"_id": dataset_id}, {"sources": report.to_dict()})

    return repo.get(uid, dataset_id)


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


def load(dataset: Dict[str, Any], uid: Optional[str] = None,
         as_of: Optional[str] = None) -> Tuple[pd.DataFrame, DatasetSchema]:
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
        df = prepare(df_raw, schema, preserve_missing=bool(dataset.get("sources")))
        # Built once per (path, mtime) and shared by every `replace(schema)`
        # copy below. The catalogue computes nothing until it is asked, so
        # attaching it here costs a constructor call, not a pass over the frame.
        schema.member_catalogue = MemberCatalogue(df, schema.dimensions)
        _CACHE[path] = (mtime, df, schema)

    if uid or dataset.get("sources"):
        schema = attach_contract(uid, dataset, df, replace(schema)) if uid else replace(schema)

    # A reconciled dataset carries per-source provenance. Its freshness is
    # re-evaluated against `as_of` on every load — the values on disk are fixed,
    # but a source that was current at ingest genuinely does go stale with time.
    if dataset.get("sources"):
        from .reconciliation import ReconciliationReport, SourceReport, refresh_sources

        stored = dict(dataset["sources"])
        reports = [SourceReport(**r) for r in stored.pop("sources", [])]
        stamp = as_of or now_iso()
        report = ReconciliationReport(**{**stored, "as_of": stamp,
                                         "sources": refresh_sources(reports, stamp)})
        schema.sources = report
        # Enforce the withholding. `CompiledKpi.compute` uses a bare `.sum()`,
        # so a KPI whose fields have gaps would either under-report invisibly or
        # return 0.0 for an empty period. Taking it out of `available_kpis` is
        # what makes "no number" actually happen instead of a wrong one.
        if report.withheld_kpis:
            schema.available_kpis = [k for k in schema.available_kpis
                                     if k not in report.withheld_kpis]

    return df, schema


def _clear_intent_cache(dataset_id: Optional[str] = None) -> None:
    """Drop cached question interpretations for a dataset whose rows changed."""
    try:
        from ..query.understanding import clear_intent_cache
        clear_intent_cache(dataset_id)
    except Exception:                                # pragma: no cover - best effort
        log.debug("Could not clear the intent cache for %s", dataset_id)


def clear_cache(path: Optional[str] = None) -> None:
    if path:
        _CACHE.pop(path, None)
    else:
        _CACHE.clear()
