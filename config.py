"""
Central app configuration, loaded from environment variables / .env.

Every other module imports `settings` from here rather than calling
os.environ directly, so there is exactly one place that knows how
configuration is sourced.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    database_url: str = "postgresql+psycopg2://baseball:baseball@localhost:5432/baseball"

    # External APIs (optional - features degrade gracefully when absent)
    odds_api_key: str = ""
    weather_api_key: str = ""

    # Admin auth for POST /models/retrain - a shared-secret header, not a
    # full user/auth system, since this app has no other authenticated
    # endpoints. Leave blank in dev to disable the check entirely (see
    # api/routers/models.py) - set a real value before exposing the API
    # publicly.
    admin_api_key: str = ""

    # App
    app_env: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    model_registry_dir: str = "models/registry"
    timezone: str = "America/New_York"
    api_base_url: str = "http://localhost:8000"  # frontend/ uses this to reach the FastAPI backend

    @property
    def has_odds_key(self) -> bool:
        return bool(self.odds_api_key.strip())

    @property
    def has_weather_key(self) -> bool:
        return bool(self.weather_api_key.strip())

    @property
    def model_registry_path(self) -> Path:
        path = BASE_DIR / self.model_registry_dir
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()
