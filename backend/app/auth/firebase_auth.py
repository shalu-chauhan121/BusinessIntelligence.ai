"""
Firebase authentication.

Two modes, chosen by `AUTH_MODE` (default `auto`):

* **firebase** — the backend initialises `firebase-admin` with a service account
  and *cryptographically verifies* every ID token minted by the Firebase client
  SDK in the browser. This is the production path.

* **demo** — no service-account credentials are present, so the backend cannot
  verify signatures. It decodes the token payload (still learning the uid/email)
  and additionally exposes a local `/api/auth/demo-login` endpoint so the whole
  product can be run and judged before Firebase keys are configured.

Authorisation (what a signed-in user may see and do) is *not* Firebase's job —
roles live in our own user store, see `app/db/repositories.py`.
"""
from __future__ import annotations

import base64
import json
import logging
import time
from typing import Any, Dict, Optional

from ..config import get_settings

log = logging.getLogger(__name__)
_firebase_ready: Optional[bool] = None


class AuthError(Exception):
    pass


def _init_firebase() -> bool:
    """Initialise firebase-admin once. Returns True when verification is available."""
    global _firebase_ready
    if _firebase_ready is not None:
        return _firebase_ready

    s = get_settings()
    _firebase_ready = False
    try:
        import firebase_admin
        from firebase_admin import credentials

        if firebase_admin._apps:                      # already initialised
            _firebase_ready = True
            return True

        cred = None
        if s.firebase_service_account_json.strip():
            cred = credentials.Certificate(json.loads(s.firebase_service_account_json))
        elif s.google_application_credentials.strip():
            cred = credentials.Certificate(s.google_application_credentials)

        if cred is None:
            log.warning("Firebase service account not configured — running in demo auth mode.")
            return False

        options = {"projectId": s.firebase_project_id} if s.firebase_project_id else None
        firebase_admin.initialize_app(cred, options)
        _firebase_ready = True
        log.info("Firebase Admin initialised — ID tokens will be verified.")
    except Exception as exc:                          # pragma: no cover - env dependent
        log.warning("Firebase Admin unavailable (%s) — falling back to demo auth mode.", exc)
        _firebase_ready = False
    return _firebase_ready


def auth_mode() -> str:
    mode = get_settings().resolved_auth_mode
    if mode == "firebase" and not _init_firebase():
        return "demo"
    return mode


# ---------------------------------------------------------------------------
# demo tokens
# ---------------------------------------------------------------------------
DEMO_PREFIX = "demo."


def issue_demo_token(uid: str, email: str, name: str = "") -> str:
    payload = {
        "uid": uid,
        "email": email,
        "name": name,
        "iss": "businessintelligence.ai/demo",
        "iat": int(time.time()),
        "exp": int(time.time()) + 60 * 60 * 12,
    }
    raw = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return DEMO_PREFIX + raw


def _b64_json(segment: str) -> Dict[str, Any]:
    padded = segment + "=" * (-len(segment) % 4)
    return json.loads(base64.urlsafe_b64decode(padded.encode()).decode())


def _decode_unverified(token: str) -> Dict[str, Any]:
    """Read a JWT payload without checking its signature (demo mode only)."""
    parts = token.split(".")
    if len(parts) < 2:
        raise AuthError("malformed token")
    return _b64_json(parts[1])


def verify_token(token: str) -> Dict[str, Any]:
    """Return a normalised identity dict: uid, email, name, verified."""
    token = (token or "").strip()
    if not token:
        raise AuthError("missing token")

    if token.startswith(DEMO_PREFIX):
        if auth_mode() != "demo":
            raise AuthError("demo tokens are rejected when Firebase verification is enabled")
        payload = _b64_json(token[len(DEMO_PREFIX):])
        if payload.get("exp", 0) < time.time():
            raise AuthError("demo token expired")
        return {
            "uid": payload["uid"],
            "email": payload.get("email", ""),
            "name": payload.get("name", ""),
            "verified": False,
        }

    if auth_mode() == "firebase":
        from firebase_admin import auth as fb_auth

        try:
            decoded = fb_auth.verify_id_token(token)
        except Exception as exc:
            raise AuthError(f"invalid Firebase ID token: {exc}") from exc
        return {
            "uid": decoded["uid"],
            "email": decoded.get("email", ""),
            "name": decoded.get("name", ""),
            "verified": True,
        }

    # demo mode, but a real Firebase token was supplied: accept it unverified so
    # the frontend can use Firebase auth before the backend has a service account.
    payload = _decode_unverified(token)
    uid = payload.get("user_id") or payload.get("sub") or payload.get("uid")
    if not uid:
        raise AuthError("token has no subject")
    return {
        "uid": uid,
        "email": payload.get("email", ""),
        "name": payload.get("name", ""),
        "verified": False,
    }
