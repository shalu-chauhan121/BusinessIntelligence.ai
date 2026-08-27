"""Central configuration. Everything is env-driven with safe local defaults."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # database
    db_backend: str = "json"          # json | mongo
    data_dir: str = "./data"
    mongodb_uri: str = ""
    mongodb_db: str = "businessintelligence"

    # auth
    firebase_project_id: str = ""
    google_application_credentials: str = ""
    firebase_service_account_json: str = ""
    auth_mode: str = "auto"           # firebase | demo | auto

    # llm
    llm_provider: str = "anthropic"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-5"
    llm_max_tokens: int = 4000
    llm_timeout_seconds: int = 90
    llm_pricing_json: str = ""

    # analysis
    anomaly_z_threshold: float = 2.0
    min_material_change_pct: float = 3.0
    min_history_comparisons: int = 4
    max_drivers_per_dimension: int = 5

    # Multi-source reconciliation. When two sources report the same business
    # cell, a difference within this tolerance is rounding/timing noise and the
    # value is taken as corroborated; anything larger is a real disagreement the
    # system refuses to rule on by itself.
    source_agreement_tolerance_pct: float = 0.5

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def upload_dir(self) -> str:
        return os.path.join(self.data_dir, "uploads")

    @property
    def llm_enabled(self) -> bool:
        return bool(self.anthropic_api_key.strip())

    @property
    def resolved_auth_mode(self) -> str:
        if self.auth_mode in ("firebase", "demo"):
            return self.auth_mode
        has_creds = bool(
            self.firebase_service_account_json.strip()
            or self.google_application_credentials.strip()
        )
        return "firebase" if has_creds else "demo"


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    os.makedirs(s.data_dir, exist_ok=True)
    os.makedirs(s.upload_dir, exist_ok=True)
    return s
