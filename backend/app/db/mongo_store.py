"""MongoDB / MongoDB Atlas implementation of the same document interface."""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Tuple

from .base import Collection, DocumentStore


class MongoCollection(Collection):
    def __init__(self, raw):
        self.raw = raw

    def insert_one(self, document: Dict[str, Any]) -> Dict[str, Any]:
        doc = dict(document)
        doc.setdefault("_id", uuid.uuid4().hex)
        self.raw.insert_one(doc)
        return doc

    def find_one(self, flt: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        return self.raw.find_one(flt or {})

    def find(
        self,
        flt: Optional[Dict[str, Any]] = None,
        sort: Optional[Tuple[str, int]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        cursor = self.raw.find(flt or {})
        if sort:
            cursor = cursor.sort(sort[0], sort[1])
        if limit:
            cursor = cursor.limit(limit)
        return list(cursor)

    def update_one(
        self, flt: Dict[str, Any], fields: Dict[str, Any], upsert: bool = False
    ) -> Optional[Dict[str, Any]]:
        self.raw.update_one(flt, {"$set": fields}, upsert=upsert)
        return self.raw.find_one(flt)

    def delete_one(self, flt: Dict[str, Any]) -> bool:
        return self.raw.delete_one(flt).deleted_count > 0

    def delete_many(self, flt: Dict[str, Any]) -> int:
        return self.raw.delete_many(flt).deleted_count

    def count(self, flt: Optional[Dict[str, Any]] = None) -> int:
        return self.raw.count_documents(flt or {})


class MongoDocumentStore(DocumentStore):
    backend_name = "mongo"

    def __init__(self, uri: str, db_name: str):
        from pymongo import MongoClient  # imported lazily: optional dependency

        self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        self.db = self.client[db_name]

    def collection(self, name: str) -> MongoCollection:
        return MongoCollection(self.db[name])

    def ping(self) -> bool:
        try:
            self.client.admin.command("ping")
            return True
        except Exception:
            return False
