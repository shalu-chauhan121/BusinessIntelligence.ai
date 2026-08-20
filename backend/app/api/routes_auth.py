"""Authentication and role management (authorisation)."""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status

from ..auth.firebase_auth import auth_mode, issue_demo_token
from ..config import get_settings
from ..db.repositories import UserRepository, new_id
from ..deps import current_identity, current_user
from ..models.schemas import DemoLoginRequest, RegisterRequest, RoleUpdate

router = APIRouter(prefix="/api/auth", tags=["auth"])

PERMISSIONS = {
    "data_analyst": {
        "view_dashboard": True,
        "run_investigation": True,
        "view_statistical_detail": True,   # z-scores, MAD, correlations, method notes
        "view_evidence_ledger": True,      # how each confidence score was built
        "view_reasoning_trail": True,
        "view_raw_driver_tables": True,
        "search_documents": True,
        "manage_data": True,
    },
    "business_leader": {
        "view_dashboard": True,
        "run_investigation": True,
        "view_statistical_detail": False,
        "view_evidence_ledger": False,
        "view_reasoning_trail": True,
        "view_raw_driver_tables": False,
        "search_documents": False,
        "manage_data": True,
    },
}


def shape_user(profile: Dict[str, Any]) -> Dict[str, Any]:
    role = profile.get("role", "business_leader")
    return {
        "uid": profile["uid"],
        "email": profile.get("email", ""),
        "display_name": profile.get("display_name", ""),
        "role": role,
        "organisation": profile.get("organisation", ""),
        "created_at": profile.get("created_at"),
        "token_verified": profile.get("token_verified", False),
        "permissions": PERMISSIONS[role],
    }


@router.get("/config")
def auth_config() -> Dict[str, Any]:
    """Public: tells the frontend which authentication path is active."""
    s = get_settings()
    mode = auth_mode()
    return {
        "mode": mode,
        "firebase_configured": bool(s.firebase_project_id),
        "demo_login_enabled": mode == "demo",
        "roles": [
            {"key": "business_leader", "label": "Business Leader",
             "description": "Decision-oriented view: what changed, why, what to do next."},
            {"key": "data_analyst", "label": "Data Analyst",
             "description": "Everything the leader sees, plus the statistical method, the evidence ledger, "
                            "driver tables and document retrieval."},
        ],
        "notice": (
            None if mode == "firebase" else
            "Running in demo auth mode: Firebase service-account credentials are not configured, so ID "
            "token signatures are not verified. Configure Firebase before any real deployment."
        ),
    }


@router.post("/demo-login")
def demo_login(body: DemoLoginRequest) -> Dict[str, Any]:
    """Local login for running the product before Firebase credentials exist."""
    if auth_mode() != "demo":
        raise HTTPException(status.HTTP_403_FORBIDDEN,
                            "Demo login is disabled because Firebase verification is active.")
    uid = "demo_" + new_id("u").split("_")[1]
    users = UserRepository()
    existing = next((u for u in users.col.find({"email": body.email}) if u.get("uid", "").startswith("demo_")), None)
    if existing:
        uid = existing["uid"]
    profile = users.upsert_profile(uid=uid, email=body.email,
                                   display_name=body.display_name, role=body.role)
    return {"token": issue_demo_token(uid, body.email, body.display_name),
            "user": shape_user(profile), "mode": "demo"}


@router.post("/register")
def register(body: RegisterRequest, identity: Dict[str, Any] = Depends(current_identity)) -> Dict[str, Any]:
    """Called right after Firebase sign-up: stores the profile and the chosen role."""
    profile = UserRepository().upsert_profile(
        uid=identity["uid"], email=identity.get("email", ""),
        display_name=body.display_name or identity.get("name", ""),
        role=body.role, organisation=body.organisation,
    )
    return shape_user({**profile, "token_verified": identity.get("verified", False)})


@router.get("/me")
def me(user: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    return shape_user(user)


@router.patch("/role")
def update_role(body: RoleUpdate, user: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    profile = UserRepository().set_role(user["uid"], body.role)
    return shape_user({**(profile or user), "token_verified": user.get("token_verified", False)})
