"""Patch extraction and normalization from multi-band Sentinel-2 GeoTIFFs."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import rasterio

logger = logging.getLogger(__name__)

PATCH_SIZE = 64
STRIDE = 32
NUM_BANDS = 5


def extract_patches(
    tif_path: Path,
    out_dir: Path,
    patch_size: int = PATCH_SIZE,
    stride: int = STRIDE,
) -> list[Path]:
    """Slice a multi-band GeoTIFF into (NUM_BANDS, patch_size, patch_size) .npy files.

    Returns a list of saved patch paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = tif_path.stem
    saved: list[Path] = []

    with rasterio.open(tif_path) as src:
        data = src.read().astype(np.float32)  # (bands, H, W)

    _, h, w = data.shape
    patch_idx = 0

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = data[:, y : y + patch_size, x : x + patch_size]
            if patch.shape != (NUM_BANDS, patch_size, patch_size):
                continue
            # skip patches that are entirely zero (no-data)
            if patch.max() == 0:
                continue
            out_path = out_dir / f"{stem}_p{patch_idx:05d}.npy"
            np.save(out_path, patch)
            saved.append(out_path)
            patch_idx += 1

    logger.info("Extracted %d patches from %s", len(saved), tif_path.name)
    return saved


def compute_band_stats(processed_dir: Path) -> dict[str, list[float]]:
    """Compute per-band min and max across all .npy patches in *processed_dir*.

    Returns ``{"min": [b0, …, b4], "max": [b0, …, b4]}``.
    """
    band_min = np.full(NUM_BANDS, np.inf, dtype=np.float64)
    band_max = np.full(NUM_BANDS, -np.inf, dtype=np.float64)

    npy_files = sorted(processed_dir.glob("*.npy"))
    for p in npy_files:
        arr = np.load(p)  # (bands, 64, 64)
        for b in range(arr.shape[0]):
            band_min[b] = min(band_min[b], float(arr[b].min()))
            band_max[b] = max(band_max[b], float(arr[b].max()))

    stats = {"min": band_min.tolist(), "max": band_max.tolist()}
    stats_path = processed_dir / "band_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    logger.info("Band stats saved → %s", stats_path)
    return stats


def normalize_patch(
    patch: np.ndarray,
    band_min: list[float],
    band_max: list[float],
) -> np.ndarray:
    """Min-max normalize a (NUM_BANDS, H, W) patch to [0, 1] per band."""
    out = np.empty_like(patch, dtype=np.float32)
    for b in range(patch.shape[0]):
        denom = band_max[b] - band_min[b]
        if denom == 0:
            out[b] = 0.0
        else:
            out[b] = (patch[b] - band_min[b]) / denom
    return np.clip(out, 0.0, 1.0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import argparse

    parser = argparse.ArgumentParser(description="Extract & normalize patches")
    parser.add_argument("--input", default="data/raw", help="Directory with multi-band GeoTIFFs")
    parser.add_argument("--output", default="data/processed", help="Output directory for .npy patches")
    args = parser.parse_args()

    for tif in sorted(Path(args.input).glob("*.tif")):
        extract_patches(tif, Path(args.output))
    compute_band_stats(Path(args.output))
