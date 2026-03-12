"""/score endpoint — accepts a 5-band GeoTIFF and returns a smoke/clear prediction."""

from __future__ import annotations

import io

import numpy as np
import rasterio
import torch
from fastapi import APIRouter, Request, UploadFile

from src.data.patch import normalize_patch

router = APIRouter()


@router.post("/score")
async def score(request: Request, file: UploadFile):
    """Score a 5-band 64x64 GeoTIFF patch.

    Returns ``{"label": "smoke"|"clear", "confidence": float}``.
    """
    contents = await file.read()

    with rasterio.open(io.BytesIO(contents)) as src:
        patch = src.read().astype(np.float32)  # (bands, H, W)

    patch = normalize_patch(patch, request.app.state.band_min, request.app.state.band_max)
    tensor = torch.from_numpy(patch).unsqueeze(0)  # (1, 5, 64, 64)

    model = request.app.state.model
    device = next(model.parameters()).device
    tensor = tensor.to(device)

    with torch.no_grad():
        logit = model(tensor).squeeze()
        confidence = torch.sigmoid(logit).item()

    label = "smoke" if confidence >= 0.5 else "clear"
    return {"label": label, "confidence": round(confidence, 4)}
