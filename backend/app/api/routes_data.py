"""Dataset and document management — every upload is scoped to the signed-in user."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from ..db.repositories import DatasetRepository, DocumentRepository
from ..deps import active_dataset, current_user, require_analyst
from ..engines.observe import available_timeframes
from ..models.schemas import SearchRequest
from ..rag.chunker import chunk_document
from ..rag.extract import extract_text, guess_doc_type
from ..rag.retriever import Retriever, invalidate
from ..services import dataset_service
from .paths import SAMPLE_CSV, SAMPLE_DOCS_DIR, SAMPLE_TEMPLATE

router = APIRouter(prefix="/api", tags=["data"])
MAX_UPLOAD_MB = 25


def _shape_dataset(ds: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": ds["_id"],
        "filename": ds.get("filename"),
        "size_bytes": ds.get("size_bytes"),
        "created_at": ds.get("created_at"),
        "is_active": bool(ds.get("is_active")),
        "schema": ds.get("schema", {}),
        "sources": ds.get("sources"),
    }


# ---------------------------------------------------------------------------
# structured datasets
# ---------------------------------------------------------------------------
@router.get("/datasets")
def list_datasets(user: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    items = [_shape_dataset(d) for d in DatasetRepository().list_for_user(user["uid"])]
    return {"datasets": items, "count": len(items)}


@router.post("/datasets", status_code=status.HTTP_201_CREATED)
async def upload_dataset(file: Optional[UploadFile] = File(None),
                         files: Optional[List[UploadFile]] = File(None),
                         as_of: Optional[str] = Form(None),
                         source_meta: Optional[str] = Form(None),
                         user: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    """
    Upload one dataset, or several sources that are reconciled into one.

    Both form field names are accepted: `file` is the long-standing
    single-upload field and keeps working exactly as it always did; `files`
    carries one or many. Several files are assumed to share field names and
    meanings (the KPI Contract already defines those) and are reconciled into
    one canonical view — there is no mapping step and nothing to configure.

    `source_meta` is refresh metadata only — never semantics — supplied as a
    JSON object keyed by filename:
    `{"finance.csv": {"cadence": "hourly", "last_refresh_at": "2026-06-30T08:00:00Z"}}`.
    Both keys are optional per file; an unmentioned file falls back to its data's
    own latest timestamp and an unassertable cadence, exactly as before.
    """
    incoming = list(files or [])
    if file is not None:
        incoming.insert(0, file)

    payloads = []
    for f in incoming:
        raw = await f.read()
        if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                                f"{f.filename} is larger than {MAX_UPLOAD_MB} MB.")
        payloads.append((f.filename or "dataset.csv", raw))
    if not payloads:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "No file was uploaded.")

    meta: Dict[str, Any] = {}
    if source_meta:
        import json as _json
        try:
            meta = _json.loads(source_meta)
        except ValueError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                f"source_meta is not valid JSON: {exc}") from exc

    try:
        if len(payloads) == 1:
            ds = dataset_service.store_upload(user["uid"], payloads[0][0], payloads[0][1])
        else:
            ds = dataset_service.store_multisource_upload(user["uid"], payloads, as_of=as_of,
                                                           source_meta=meta)
    except dataset_service.DatasetError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    return {**_shape_dataset(ds), "sources": ds.get("sources")}


@router.post("/datasets/load-sample", status_code=status.HTTP_201_CREATED)
def load_sample_dataset(user: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    if not SAMPLE_CSV.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Sample dataset is not bundled with this deployment.")
    ds = dataset_service.store_upload(user["uid"], SAMPLE_CSV.name, SAMPLE_CSV.read_bytes())
    return _shape_dataset(ds)


@router.get("/datasets/template")
def download_template():
    if not SAMPLE_TEMPLATE.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Template not bundled.")
    return FileResponse(SAMPLE_TEMPLATE, media_type="text/csv",
                        filename="business_metrics_TEMPLATE.csv")


@router.get("/datasets/active")
def get_active(user: Dict[str, Any] = Depends(current_user),
               dataset: Dict[str, Any] = Depends(active_dataset)) -> Dict[str, Any]:
    df, schema = dataset_service.load(dataset, user["uid"])
    return {
        **_shape_dataset(dataset),
        "timeframes": available_timeframes(df),
        "schema": schema.to_dict(),
    }


@router.post("/datasets/{dataset_id}/activate")
def activate(dataset_id: str, user: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    ds = DatasetRepository().set_active(user["uid"], dataset_id)
    if not ds:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset not found.")
    return _shape_dataset(ds)


@router.delete("/datasets/{dataset_id}")
def delete_dataset(dataset_id: str, user: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    repo = DatasetRepository()
    ds = repo.get(user["uid"], dataset_id)
    if not ds:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset not found.")
    dataset_service.clear_cache(ds.get("path"))
    repo.delete(user["uid"], dataset_id)
    return {"deleted": dataset_id}


# ---------------------------------------------------------------------------
# unstructured documents (RAG corpus)
# ---------------------------------------------------------------------------
def _shape_document(d: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": d["_id"],
        "filename": d.get("filename"),
        "doc_type": d.get("doc_type"),
        "chunk_count": d.get("chunk_count", 0),
        "word_count": d.get("word_count", 0),
        "created_at": d.get("created_at"),
    }


@router.get("/documents")
def list_documents(user: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    docs = [_shape_document(d) for d in DocumentRepository().list_for_user(user["uid"])]
    return {"documents": docs, "count": len(docs),
            "chunks": sum(d["chunk_count"] for d in docs)}


def _ingest(uid: str, filename: str, raw: bytes) -> Dict[str, Any]:
    text, kind = extract_text(filename, raw)
    chunks = chunk_document(text)
    if not chunks:
        raise ValueError(f"No text could be extracted from {filename}.")
    doc = DocumentRepository().create(
        uid,
        {"filename": filename, "doc_type": guess_doc_type(filename, text),
         "source_kind": kind, "word_count": len(text.split()), "size_bytes": len(raw)},
        chunks,
    )
    invalidate(uid)
    return doc


@router.post("/documents", status_code=status.HTTP_201_CREATED)
async def upload_documents(files: List[UploadFile] = File(...),
                           user: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    created, failed = [], []
    for f in files:
        raw = await f.read()
        try:
            if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
                raise ValueError(f"{f.filename} is larger than {MAX_UPLOAD_MB} MB.")
            created.append(_shape_document(_ingest(user["uid"], f.filename or "document.txt", raw)))
        except Exception as exc:
            failed.append({"filename": f.filename, "error": str(exc)})
    if not created and failed:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, failed[0]["error"])
    return {"documents": created, "failed": failed}


@router.post("/documents/load-samples", status_code=status.HTTP_201_CREATED)
def load_sample_documents(user: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    if not SAMPLE_DOCS_DIR.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Sample documents are not bundled.")
    created, failed = [], []
    for path in sorted(SAMPLE_DOCS_DIR.iterdir()):
        if not path.is_file():
            continue
        try:
            created.append(_shape_document(_ingest(user["uid"], path.name, path.read_bytes())))
        except Exception as exc:
            failed.append({"filename": path.name, "error": str(exc)})
    return {"documents": created, "failed": failed}


@router.delete("/documents/{document_id}")
def delete_document(document_id: str, user: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    ok = DocumentRepository().delete(user["uid"], document_id)
    if not ok:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found.")
    invalidate(user["uid"])
    return {"deleted": document_id}


@router.post("/documents/search")
def search_documents(body: SearchRequest,
                     user: Dict[str, Any] = Depends(require_analyst)) -> Dict[str, Any]:
    """Analyst-only: inspect what the retrieval layer returns for a query."""
    retriever = Retriever(user["uid"])
    if not retriever.available:
        return {"query": body.query, "results": [], "note": "No documents indexed yet."}
    return {
        "query": body.query,
        "indexed_chunks": len(retriever.chunks),
        "results": retriever.retrieve(body.query, top_k=body.top_k),
        "method": "BM25 lexical retrieval over per-user document chunks",
    }
