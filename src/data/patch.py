"""Patch extraction and normalization from multi-band Sentinel-2 GeoTIFFs.

Patches are stored as **one stacked ``.npy`` file per source tile** with shape
``(N, NUM_BANDS, PATCH_SIZE, PATCH_SIZE)``.  Patch *i* in the file has ID
``<tile_stem>_p{i:05d}``.
"""

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
MIN_VALID_FRACTION = 0.8  # drop patches with >20% nodata pixels


def _is_valid_patch(patch: np.ndarray) -> bool:
    """Return *True* if *patch* passes all quality filters."""
    if patch.shape != (NUM_BANDS, PATCH_SIZE, PATCH_SIZE):
        return False
    if patch.max() == 0:
        return False
    return np.count_nonzero(patch[0]) / patch[0].size >= MIN_VALID_FRACTION


def extract_patches(
    tif_path: Path,
    out_dir: Path,
    patch_size: int = PATCH_SIZE,
    stride: int = STRIDE,
) -> int:
    """Slice a multi-band GeoTIFF into patches saved as a single stacked ``.npy``.

    The output file ``out_dir / <tif_stem>.npy`` has shape
    ``(N, NUM_BANDS, patch_size, patch_size)``.

    Returns the number of valid patches written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = tif_path.stem

    logger.info("Reading %s …", tif_path.name)
    with rasterio.open(tif_path) as src:
        data = src.read().astype(np.float32)  # (bands, H, W)

    _, h, w = data.shape
    n_rows = max(1, (h - patch_size) // stride + 1) if h >= patch_size else 0
    log_every = max(1, n_rows // 10)

    # --- Pass 1: count valid patches so we can pre-allocate ------------------
    count = 0
    row_i = 0
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            if _is_valid_patch(data[:, y : y + patch_size, x : x + patch_size]):
                count += 1
        row_i += 1
        if row_i % log_every == 0:
            logger.info("  [scan]  %d%% (%d valid so far)", 100 * row_i // n_rows, count)

    if count == 0:
        logger.info("No valid patches in %s — skipped.", tif_path.name)
        del data
        return 0

    # --- Pass 2: write patches into a memory-mapped .npy ---------------------
    out_path = out_dir / f"{stem}.npy"
    fp = np.lib.format.open_memmap(
        str(out_path),
        mode="w+",
        dtype=np.float32,
        shape=(count, NUM_BANDS, patch_size, patch_size),
    )

    idx = 0
    row_i = 0
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = data[:, y : y + patch_size, x : x + patch_size]
            if _is_valid_patch(patch):
                fp[idx] = patch
                idx += 1
        row_i += 1
        if row_i % log_every == 0:
            logger.info("  [write] %d%%", 100 * row_i // n_rows)

    fp.flush()
    del fp, data

    logger.info("Extracted %d patches from %s", count, tif_path.name)
    return count


def compute_band_stats(processed_dir: Path) -> dict[str, list[float]]:
    """Compute per-band min and max across all stacked ``.npy`` tiles.

    Returns ``{"min": [b0, …, b4], "max": [b0, …, b4]}``.
    """
    band_min = np.full(NUM_BANDS, np.inf, dtype=np.float64)
    band_max = np.full(NUM_BANDS, -np.inf, dtype=np.float64)

    for p in sorted(processed_dir.glob("*.npy")):
        arr = np.load(p, mmap_mode="r")  # (N, bands, 64, 64)
        for b in range(NUM_BANDS):
            band_min[b] = min(band_min[b], float(arr[:, b].min()))
            band_max[b] = max(band_max[b], float(arr[:, b].max()))

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
