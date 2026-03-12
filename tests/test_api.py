"""Tests for the FastAPI /score and /health endpoints."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from src.api.app import app

# httpx is a dev dependency; skip gracefully if missing.
httpx = pytest.importorskip("httpx")


def _make_dummy_tif(bands: int = 5, size: int = 64) -> bytes:
    """Create a minimal in-memory GeoTIFF and return its bytes."""
    data = np.random.rand(bands, size, size).astype(np.float32)
    transform = from_bounds(0, 0, 1, 1, size, size)
    buf = io.BytesIO()
    with rasterio.open(
        buf,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=bands,
        dtype="float32",
        transform=transform,
    ) as dst:
        dst.write(data)
    return buf.getvalue()


@pytest.fixture()
def dummy_model_files(tmp_path: Path):
    """Create a fake best.pt and band_stats.json so the app can start."""
    from src.model.net import build_model
    import torch

    model = build_model(pretrained=False)
    model_path = tmp_path / "best.pt"
    torch.save(model.state_dict(), model_path)

    stats_path = tmp_path / "band_stats.json"
    stats_path.write_text(json.dumps({"min": [0.0] * 5, "max": [1.0] * 5}))
    return model_path, stats_path


@pytest.fixture()
def configured_app(dummy_model_files):
    """Manually configure app.state so tests don't depend on lifespan events."""
    import torch
    from src.model.net import build_model

    model_path, stats_path = dummy_model_files

    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    stats = json.loads(stats_path.read_text())
    app.state.model = model
    app.state.band_min = stats["min"]
    app.state.band_max = stats["max"]

    yield app

    # cleanup
    del app.state.model
    del app.state.band_min
    del app.state.band_max


@pytest.mark.anyio
async def test_health(configured_app):
    from httpx import ASGITransport, AsyncClient

    async with AsyncClient(transport=ASGITransport(app=configured_app), base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_score_returns_label(configured_app):
    tif_bytes = _make_dummy_tif()
    from httpx import ASGITransport, AsyncClient

    async with AsyncClient(transport=ASGITransport(app=configured_app), base_url="http://test") as client:
        resp = await client.post("/score", files={"file": ("patch.tif", tif_bytes)})
        assert resp.status_code == 200
        body = resp.json()
        assert body["label"] in ("smoke", "clear")
        assert 0.0 <= body["confidence"] <= 1.0
