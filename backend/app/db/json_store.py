"""File-backed, Mongo-shaped document store. No database server required."""
from __future__ import annotations

import json
import os
import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple

from .base import Collection, DocumentStore, matches

_LOCK = threading.RLock()


class JsonCollection(Collection):
    def __init__(self, path: str):
        self.path = path
        if not os.path.exists(self.path):
            self._write([])

    # ---- disk -------------------------------------------------------------
    def _read(self) -> List[Dict[str, Any]]:
        with _LOCK:
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except (FileNotFoundError, json.JSONDecodeError):
                return []

    def _write(self, docs: List[Dict[str, Any]]) -> None:
        with _LOCK:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = f"{self.path}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(docs, fh, ensure_ascii=False, indent=2, default=str)
            os.replace(tmp, self.path)

    # ---- api --------------------------------------------------------------
    def insert_one(self, document: Dict[str, Any]) -> Dict[str, Any]:
        with _LOCK:
            docs = self._read()
            doc = dict(document)
            doc.setdefault("_id", uuid.uuid4().hex)
            docs.append(doc)
            self._write(docs)
            return doc

    def find_one(self, flt: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        for doc in self._read():
            if matches(doc, flt):
                return doc
        return None

    def find(
        self,
        flt: Optional[Dict[str, Any]] = None,
        sort: Optional[Tuple[str, int]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        out = [d for d in self._read() if matches(d, flt)]
        if sort:
            key, direction = sort
            out.sort(key=lambda d: (d.get(key) is None, d.get(key)), reverse=direction < 0)
        return out[:limit] if limit else out

    def update_one(
        self, flt: Dict[str, Any], fields: Dict[str, Any], upsert: bool = False
    ) -> Optional[Dict[str, Any]]:
        with _LOCK:
            docs = self._read()
            for i, doc in enumerate(docs):
                if matches(doc, flt):
                    docs[i] = {**doc, **fields}
                    self._write(docs)
                    return docs[i]
            if upsert:
                merged = {**flt, **fields}
                merged.setdefault("_id", uuid.uuid4().hex)
                docs.append(merged)
                self._write(docs)
                return merged
            return None

    def delete_one(self, flt: Dict[str, Any]) -> bool:
        with _LOCK:
            docs = self._read()
            for i, doc in enumerate(docs):
                if matches(doc, flt):
                    docs.pop(i)
                    self._write(docs)
                    return True
            return False

    def delete_many(self, flt: Dict[str, Any]) -> int:
        with _LOCK:
            docs = self._read()
            keep = [d for d in docs if not matches(d, flt)]
            removed = len(docs) - len(keep)
            if removed:
                self._write(keep)
            return removed

    def count(self, flt: Optional[Dict[str, Any]] = None) -> int:
        return len(self.find(flt))


class JsonDocumentStore(DocumentStore):
    backend_name = "json"

    def __init__(self, data_dir: str):
        self.root = os.path.join(data_dir, "db")
        os.makedirs(self.root, exist_ok=True)
        self._collections: Dict[str, JsonCollection] = {}

    def collection(self, name: str) -> JsonCollection:
        if name not in self._collections:
            self._collections[name] = JsonCollection(os.path.join(self.root, f"{name}.json"))
        return self._collections[name]

    def ping(self) -> bool:
        return os.path.isdir(self.root)
