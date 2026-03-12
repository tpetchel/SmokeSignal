"""FastAPI application — lifespan loads the model once at startup."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI

from src.model.net import build_model

logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/best.pt")
STATS_PATH = Path("data/processed/band_stats.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and band stats into app state on startup."""
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE).eval()
    app.state.model = model

    stats = json.loads(STATS_PATH.read_text())
    app.state.band_min = stats["min"]
    app.state.band_max = stats["max"]

    logger.info("Model loaded from %s on %s", MODEL_PATH, DEVICE)
    yield


app = FastAPI(title="SmokeSignal", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


# Import the score router after app is defined to avoid circular imports.
from src.api.score import router as score_router  # noqa: E402

app.include_router(score_router)
