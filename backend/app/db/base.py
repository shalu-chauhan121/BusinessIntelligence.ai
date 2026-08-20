"""
Storage abstraction.

The application only ever talks to `Collection` / `DocumentStore`. Two
implementations are provided:

  * `JsonDocumentStore`  — file-backed, zero-install, Mongo-shaped. Default.
  * `MongoDocumentStore` — pymongo against a local mongod or MongoDB Atlas.

Because the interface is document-shaped (collections of JSON documents,
dict filters, `$in` / `$exists` operators), moving to Atlas later is a
configuration change (`DB_BACKEND=mongo`, `MONGODB_URI=...`) and touches no
application code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


def matches(doc: Dict[str, Any], flt: Optional[Dict[str, Any]]) -> bool:
    """Evaluate the small filter subset the app needs, Mongo-style."""
    if not flt:
        return True
    for key, cond in flt.items():
        value = doc.get(key)
        if isinstance(cond, dict):
            for op, operand in cond.items():
                if op == "$in" and value not in operand:
                    return False
                elif op == "$nin" and value in operand:
                    return False
                elif op == "$ne" and value == operand:
                    return False
                elif op == "$exists" and (value is not None) != bool(operand):
                    return False
                elif op == "$gte" and not (value is not None and value >= operand):
                    return False
                elif op == "$lte" and not (value is not None and value <= operand):
                    return False
        elif value != cond:
            return False
    return True


class Collection(ABC):
    @abstractmethod
    def insert_one(self, document: Dict[str, Any]) -> Dict[str, Any]: ...

    @abstractmethod
    def find_one(self, flt: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]: ...

    @abstractmethod
    def find(
        self,
        flt: Optional[Dict[str, Any]] = None,
        sort: Optional[Tuple[str, int]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def update_one(
        self, flt: Dict[str, Any], fields: Dict[str, Any], upsert: bool = False
    ) -> Optional[Dict[str, Any]]: ...

    @abstractmethod
    def delete_one(self, flt: Dict[str, Any]) -> bool: ...

    @abstractmethod
    def delete_many(self, flt: Dict[str, Any]) -> int: ...

    @abstractmethod
    def count(self, flt: Optional[Dict[str, Any]] = None) -> int: ...


class DocumentStore(ABC):
    backend_name: str = "abstract"

    @abstractmethod
    def collection(self, name: str) -> Collection: ...

    @abstractmethod
    def ping(self) -> bool: ...
