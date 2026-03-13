"""STAC search & download for Sentinel-2 imagery from Microsoft Planetary Computer."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import planetary_computer as pc
import rasterio
from pystac import Item
from pystac_client import Client
from rasterio.enums import Resampling

logger = logging.getLogger(__name__)


def _make_vsicurl_clear_cache():
    """Return a callable that clears GDAL's /vsicurl/ metadata cache, or *None*."""
    try:
        from osgeo import gdal
        return gdal.VSICurlClearCache
    except ImportError:
        pass
    try:
        import ctypes
        libs_dir = Path(rasterio.__file__).parent.parent / "rasterio.libs"
        for candidate in libs_dir.glob("gdal*"):
            lib = ctypes.CDLL(str(candidate))
            fn = lib.VSICurlClearCache
            fn.restype = None
            fn.argtypes = []
            return fn
    except Exception:
        pass
    return None


_vsicurl_clear_cache = _make_vsicurl_clear_cache()

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"
BANDS = ("B02", "B03", "B04", "B11", "B12")
MAX_CLOUD = 20  # percent


def search_items(
    bbox: tuple[float, float, float, float],
    date_range: str,
    max_cloud: int = MAX_CLOUD,
) -> list[Item]:
    """Search Planetary Computer for Sentinel-2 items within *bbox* and *date_range*.

    Parameters
    ----------
    bbox : (west, south, east, north) in EPSG:4326 degrees.
    date_range : ISO-8601 interval, e.g. ``"2025-08-01/2025-08-15"``.
    max_cloud : Maximum cloud cover percentage (default 20).

    Returns
    -------
    list[Item]
    """
    client = Client.open(STAC_URL, modifier=pc.sign_inplace)
    search = client.search(
        collections=[COLLECTION],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    items = list(search.items())
    logger.info("Found %d items for bbox=%s, dates=%s", len(items), bbox, date_range)
    return items


MAX_RETRIES = 3
RETRY_BACKOFF = 5  # seconds


def download_bands(
    item: Item,
    out_dir: Path,
    bands: Sequence[str] = BANDS,
    target_resolution: float = 10.0,
) -> Path | None:
    """Download *bands* from a STAC *item* and stack them into a single GeoTIFF.

    SWIR bands (20 m) are resampled to *target_resolution* (10 m) via bilinear
    interpolation so that all bands share the same grid.  Retries up to
    ``MAX_RETRIES`` times on transient I/O errors.

    Returns
    -------
    Path to the saved multi-band GeoTIFF, or ``None`` if all retries failed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    item_id = item.id

    out_path = out_dir / f"{item_id}.tif"
    tmp_path = out_path.with_suffix(".tif.tmp")
    if out_path.exists():
        logger.info("Skipping %s (already downloaded)", item_id)
        return out_path

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if attempt > 1 and _vsicurl_clear_cache:
                _vsicurl_clear_cache()

            # Re-sign to get a fresh SAS token (previous one may have expired).
            item = pc.sign_item(item)

            arrays: list[np.ndarray] = []
            profile = None

            for band_name in bands:
                href = item.assets[band_name].href
                with rasterio.open(href) as src:
                    native_res = src.res[0]
                    if native_res != target_resolution:
                        scale = native_res / target_resolution
                        new_h = int(src.height * scale)
                        new_w = int(src.width * scale)
                        data = src.read(
                            1,
                            out_shape=(new_h, new_w),
                            resampling=Resampling.bilinear,
                        ).astype(np.float32)
                    else:
                        data = src.read(1).astype(np.float32)

                    if profile is None:
                        profile = src.profile.copy()
                        profile.update(
                            count=len(bands),
                            dtype="float32",
                            height=data.shape[0],
                            width=data.shape[1],
                        )

                    arrays.append(data)

            with rasterio.open(tmp_path, "w", **profile) as dst:
                for idx, arr in enumerate(arrays, start=1):
                    dst.write(arr, idx)
            tmp_path.replace(out_path)

            logger.info("Saved %d-band stack → %s", len(bands), out_path)
            return out_path

        except (rasterio.errors.RasterioIOError, rasterio.errors.RasterioError) as exc:
            # Clean up partial/temp files
            for p in (tmp_path, out_path):
                if p.exists():
                    p.unlink()
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF * attempt
                logger.warning(
                    "Attempt %d/%d failed for %s: %s  — retrying in %ds",
                    attempt, MAX_RETRIES, item_id, exc, wait,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "All %d attempts failed for %s: %s  — skipping",
                    MAX_RETRIES, item_id, exc,
                )
                return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import argparse

    parser = argparse.ArgumentParser(description="Fetch Sentinel-2 imagery")
    parser.add_argument("--bbox", type=float, nargs=4, required=True,
                        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
                        help="Bounding box in EPSG:4326 degrees")
    parser.add_argument("--dates", required=True,
                        help="ISO-8601 date range, e.g. 2025-08-01/2025-08-15")
    parser.add_argument("--output", default="data/raw", help="Output directory")
    args = parser.parse_args()

    items = search_items(tuple(args.bbox), args.dates)
    failed = []
    for item in items:
        result = download_bands(item, Path(args.output))
        if result is None:
            failed.append(item.id)
    if failed:
        logger.warning("Failed to download %d item(s): %s", len(failed), ", ".join(failed))
