"""Typed repositories over the document store — the only DB surface the API uses."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..config import get_settings
from .base import DocumentStore
from .json_store import JsonDocumentStore

_store: Optional[DocumentStore] = None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def get_store() -> DocumentStore:
    global _store
    if _store is None:
        s = get_settings()
        if s.db_backend == "mongo" and s.mongodb_uri:
            from .mongo_store import MongoDocumentStore

            _store = MongoDocumentStore(s.mongodb_uri, s.mongodb_db)
        else:
            _store = JsonDocumentStore(s.data_dir)
    return _store


def reset_store() -> None:              # used by the test-suite
    global _store
    _store = None


# ---------------------------------------------------------------------------
# users
# ---------------------------------------------------------------------------
class UserRepository:
    ROLES = ("data_analyst", "business_leader")

    def __init__(self):
        self.col = get_store().collection("users")

    def get(self, uid: str) -> Optional[Dict[str, Any]]:
        return self.col.find_one({"uid": uid})

    def upsert_profile(
        self,
        uid: str,
        email: str,
        display_name: str = "",
        role: Optional[str] = None,
        organisation: str = "",
    ) -> Dict[str, Any]:
        existing = self.get(uid)
        if existing:
            fields: Dict[str, Any] = {"last_login_at": now_iso()}
            if email:
                fields["email"] = email
            if display_name:
                fields["display_name"] = display_name
            if organisation:
                fields["organisation"] = organisation
            if role in self.ROLES:
                fields["role"] = role
            return self.col.update_one({"uid": uid}, fields) or existing

        doc = {
            "_id": new_id("usr"),
            "uid": uid,
            "email": email,
            "display_name": display_name or (email.split("@")[0] if email else "user"),
            "role": role if role in self.ROLES else "business_leader",
            "organisation": organisation,
            "created_at": now_iso(),
            "last_login_at": now_iso(),
        }
        return self.col.insert_one(doc)

    def set_role(self, uid: str, role: str) -> Optional[Dict[str, Any]]:
        if role not in self.ROLES:
            raise ValueError(f"unknown role: {role}")
        return self.col.update_one({"uid": uid}, {"role": role, "updated_at": now_iso()})


# ---------------------------------------------------------------------------
# datasets (structured uploads)
# ---------------------------------------------------------------------------
class DatasetRepository:
    def __init__(self):
        self.col = get_store().collection("datasets")

    def create(self, uid: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        doc = {"_id": new_id("ds"), "uid": uid, "created_at": now_iso(), **meta}
        return self.col.insert_one(doc)

    def list_for_user(self, uid: str) -> List[Dict[str, Any]]:
        return self.col.find({"uid": uid}, sort=("created_at", -1))

    def get(self, uid: str, dataset_id: str) -> Optional[Dict[str, Any]]:
        return self.col.find_one({"uid": uid, "_id": dataset_id})

    def active(self, uid: str) -> Optional[Dict[str, Any]]:
        found = self.col.find({"uid": uid, "is_active": True}, sort=("created_at", -1), limit=1)
        if found:
            return found[0]
        any_ds = self.col.find({"uid": uid}, sort=("created_at", -1), limit=1)
        return any_ds[0] if any_ds else None

    def set_active(self, uid: str, dataset_id: str) -> Optional[Dict[str, Any]]:
        for ds in self.col.find({"uid": uid}):
            self.col.update_one({"_id": ds["_id"]}, {"is_active": ds["_id"] == dataset_id})
        return self.get(uid, dataset_id)

    def delete(self, uid: str, dataset_id: str) -> bool:
        return self.col.delete_one({"uid": uid, "_id": dataset_id})


# ---------------------------------------------------------------------------
# documents (unstructured uploads) + rag chunks
# ---------------------------------------------------------------------------
class DocumentRepository:
    def __init__(self):
        self.col = get_store().collection("documents")
        self.chunks = get_store().collection("doc_chunks")

    def create(self, uid: str, meta: Dict[str, Any], chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        doc_id = new_id("doc")
        doc = {"_id": doc_id, "uid": uid, "created_at": now_iso(), "chunk_count": len(chunks), **meta}
        self.col.insert_one(doc)
        for i, ch in enumerate(chunks):
            self.chunks.insert_one(
                {
                    "_id": f"{doc_id}_c{i:04d}",
                    "uid": uid,
                    "document_id": doc_id,
                    "document_name": meta.get("filename", "document"),
                    "doc_type": meta.get("doc_type", "business_document"),
                    "chunk_index": i,
                    "text": ch["text"],
                    "heading": ch.get("heading", ""),
                }
            )
        return doc

    def list_for_user(self, uid: str) -> List[Dict[str, Any]]:
        return self.col.find({"uid": uid}, sort=("created_at", -1))

    def chunks_for_user(self, uid: str) -> List[Dict[str, Any]]:
        return self.chunks.find({"uid": uid})

    def delete(self, uid: str, document_id: str) -> bool:
        self.chunks.delete_many({"uid": uid, "document_id": document_id})
        return self.col.delete_one({"uid": uid, "_id": document_id})


# ---------------------------------------------------------------------------
# investigations (persisted four-stage runs)
# ---------------------------------------------------------------------------
class InvestigationRepository:
    def __init__(self):
        self.col = get_store().collection("investigations")

    def create(self, uid: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        doc = {"_id": new_id("inv"), "uid": uid, "created_at": now_iso(), **payload}
        return self.col.insert_one(doc)

    def list_for_user(self, uid: str, limit: int = 25) -> List[Dict[str, Any]]:
        return self.col.find({"uid": uid}, sort=("created_at", -1), limit=limit)

    def get(self, uid: str, investigation_id: str) -> Optional[Dict[str, Any]]:
        return self.col.find_one({"uid": uid, "_id": investigation_id})

    def delete(self, uid: str, investigation_id: str) -> bool:
        return self.col.delete_one({"uid": uid, "_id": investigation_id})
