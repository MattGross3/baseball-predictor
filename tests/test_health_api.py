"""
Regression coverage for GET /health/config - the endpoint the frontend
uses to tell "no odds key configured, every game is blank by design" apart
from "key configured, this game just has no odds snapshot yet" (see
web/src/components/GameRow.tsx and web/src/pages/TodaySlate.tsx).
"""
from __future__ import annotations

from fastapi.testclient import TestClient

import config
from api.main import app

client = TestClient(app)


class TestHealthConfig:
    def test_reports_both_keys_missing(self, monkeypatch):
        monkeypatch.setattr(config.settings, "odds_api_key", "")
        monkeypatch.setattr(config.settings, "weather_api_key", "")

        res = client.get("/health/config")

        assert res.status_code == 200
        assert res.json() == {"odds_api_key_configured": False, "weather_api_key_configured": False}

    def test_reports_configured_keys_as_true(self, monkeypatch):
        monkeypatch.setattr(config.settings, "odds_api_key", "a-real-key")
        monkeypatch.setattr(config.settings, "weather_api_key", "another-real-key")

        res = client.get("/health/config")

        assert res.status_code == 200
        assert res.json() == {"odds_api_key_configured": True, "weather_api_key_configured": True}

    def test_never_leaks_the_actual_key_value(self, monkeypatch):
        monkeypatch.setattr(config.settings, "odds_api_key", "super-secret-odds-key-value")

        res = client.get("/health/config")

        assert "super-secret-odds-key-value" not in res.text
