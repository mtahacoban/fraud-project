from __future__ import annotations

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "sqlite:///./fraud.db"
    model_dir: str = "models"
    xgb_model_path: str = "models/xgb_v1.pkl"
    calibrator_path: str = "models/xgb_v1_calibrated.pkl"
    model_version: str = "xgb_v1_calibrated"

    llm_provider: str = ""
    llm_api_key: str = ""
    llm_model: str = ""

    @field_validator("llm_provider", "llm_api_key", "llm_model", mode="before")
    @classmethod
    def _strip_whitespace(cls, v: str) -> str:
        return v.strip() if isinstance(v, str) else v


settings = Settings()
