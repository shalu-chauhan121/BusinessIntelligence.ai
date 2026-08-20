"""FastAPI dependencies: authentication, current user, role gating."""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import Depends, Header, HTTPException, status

from .auth.firebase_auth import AuthError, verify_token
from .db.repositories import DatasetRepository, UserRepository


def _bearer(authorization: Optional[str]) -> str:
    if not authorization:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Authorization header missing")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Expected 'Bearer <token>'")
    return token


def current_identity(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    try:
        return verify_token(_bearer(authorization))
    except AuthError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc)) from exc


def current_user(identity: Dict[str, Any] = Depends(current_identity)) -> Dict[str, Any]:
    """Signed-in user, with the role profile from our own store attached."""
    users = UserRepository()
    profile = users.get(identity["uid"])
    if profile is None:
        profile = users.upsert_profile(
            uid=identity["uid"],
            email=identity.get("email", ""),
            display_name=identity.get("name", ""),
        )
    return {**profile, "token_verified": identity.get("verified", False)}


def require_analyst(user: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    """Authorisation gate for analyst-only endpoints (statistical internals)."""
    if user.get("role") != "data_analyst":
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "This view is available to users with the Data Analyst role.",
        )
    return user


def active_dataset(user: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    ds = DatasetRepository().active(user["uid"])
    if not ds:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "No dataset uploaded yet. Upload a business metrics CSV on the Data page first.",
        )
    return ds
