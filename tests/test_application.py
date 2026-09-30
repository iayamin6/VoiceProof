"""Smoke checks for the application layout and checkpoint-free startup."""
import importlib

import torch
from fastapi.testclient import TestClient


def test_web_routes_without_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("DEV_MODE", "1")
    monkeypatch.setenv("SESSION_SECRET", "test-session-secret")
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "history.db"))
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "missing-model"))
    module = importlib.import_module("web.app")
    with TestClient(module.app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "VoiceProof" in page.text
        for asset in ("styles.css", "app.js"):
            assert client.get(f"/static/{asset}").status_code == 200
        session = client.get("/api/session").json()
        assert session["user"] is None
        assert session["model_ready"] is False
        assert "Model assets are missing" in session["model_error"]
        assert client.get("/api/history").status_code == 401
        assert client.post("/api/analyze", files={"audio": ("sample.wav", b"test")}).status_code == 401
    assert (tmp_path / "history.db").is_file()


def test_model_location_and_output_shapes(monkeypatch):
    from web.inference import LCNN, MAX_LEN, N_LFCC, _model_dir
    monkeypatch.delenv("MODEL_DIR", raising=False)
    assert (_model_dir() / "model_config.example.json").is_file()
    model = LCNN(n_classes=3).eval()
    with torch.inference_mode():
        binary, sources = model(torch.zeros(1, 1, N_LFCC, MAX_LEN))
    assert binary.shape == (1,)
    assert sources.shape == (1, 3)
