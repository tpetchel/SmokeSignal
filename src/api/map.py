"""Interactive smoke map — scores raw GeoTIFFs and displays results on a Leaflet map."""

from __future__ import annotations

import io
import json
import logging
import re
from pathlib import Path

import numpy as np
import rasterio
import torch
from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from rasterio.warp import transform_bounds

from src.data.patch import (
    MIN_VALID_FRACTION,
    NUM_BANDS,
    PATCH_SIZE,
    normalize_patch,
)

logger = logging.getLogger(__name__)

router = APIRouter()

RAW_DIR = Path("data/raw")
CACHE_DIR = Path("data/cache")
MAP_STRIDE = 64  # no overlap — faster than training stride of 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_tif_date(name: str) -> str:
    """Extract the first (acquisition) date from a Sentinel-2 filename."""
    m = re.search(r"_(\d{4})(\d{2})(\d{2})T", name)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else ""


def _tifs_for_date(date: str) -> list[Path]:
    return [t for t in sorted(RAW_DIR.glob("*.tif")) if _parse_tif_date(t.name) == date]


def _available_dates() -> list[str]:
    dates = {_parse_tif_date(t.name) for t in RAW_DIR.glob("*.tif")}
    dates.discard("")
    return sorted(dates)


def _score_tile(
    tif_path: Path,
    model: torch.nn.Module,
    band_min: list[float],
    band_max: list[float],
    device: torch.device,
) -> list[dict]:
    """Score patches in a tile. Returns GeoJSON features for smoke detections."""
    cache_path = CACHE_DIR / f"{tif_path.stem}.geojson"
    if cache_path.exists():
        return json.loads(cache_path.read_text())["features"]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Scoring %s …", tif_path.name)

    with rasterio.open(tif_path) as src:
        data = src.read().astype(np.float32)
        tf = src.transform
        crs = src.crs

    _, h, w = data.shape
    ps = PATCH_SIZE
    stride = MAP_STRIDE

    # Collect candidates using a fast spectral pre-filter
    candidates: list[tuple[int, int, np.ndarray]] = []
    for y in range(0, h - ps + 1, stride):
        for x in range(0, w - ps + 1, stride):
            patch = data[:, y : y + ps, x : x + ps]
            if patch.shape != (NUM_BANDS, ps, ps) or patch.max() == 0:
                continue
            valid_mask = patch[0] > 0
            if valid_mask.sum() / valid_mask.size < MIN_VALID_FRACTION:
                continue
            # Spectral pre-filter: elevated B04/B11 ratio → possible smoke
            b04 = patch[2][valid_mask].astype(np.float64)
            b11 = patch[3][valid_mask].astype(np.float64)
            if ((b04 / (b11 + 1.0)) > 0.75).sum() / len(b04) < 0.10:
                continue
            candidates.append((y, x, normalize_patch(patch, band_min, band_max)))

    del data
    logger.info("  %d candidates after pre-filter", len(candidates))

    # Batch model inference
    features: list[dict] = []
    batch_size = 256
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]
        tensor = torch.from_numpy(np.stack([c[2] for c in batch])).to(device)
        with torch.no_grad():
            logits = model(tensor).squeeze(-1)
            confs = np.atleast_1d(torch.sigmoid(logits).cpu().numpy())

        for j, (row, col, _) in enumerate(batch):
            conf = float(confs[j])
            if conf < 0.5:
                continue
            # Pixel → UTM → WGS84
            ul_x, ul_y = tf * (col, row)
            lr_x, lr_y = tf * (col + ps, row + ps)
            left, right = min(ul_x, lr_x), max(ul_x, lr_x)
            bottom, top_ = min(ul_y, lr_y), max(ul_y, lr_y)
            w_, s_, e_, n_ = transform_bounds(crs, "EPSG:4326", left, bottom, right, top_)

            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [[w_, s_], [e_, s_], [e_, n_], [w_, n_], [w_, s_]]
                        ],
                    },
                    "properties": {"confidence": round(conf, 3)},
                }
            )

    geojson = {"type": "FeatureCollection", "features": features}
    cache_path.write_text(json.dumps(geojson))
    logger.info("  %d smoke detections cached → %s", len(features), cache_path.name)
    return features


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/api/dates")
def dates():
    return {"dates": _available_dates()}


@router.get("/api/smoke")
def smoke(request: Request, date: str = Query(...)):
    model = request.app.state.model
    band_min = request.app.state.band_min
    band_max = request.app.state.band_max
    device = next(model.parameters()).device

    all_features: list[dict] = []
    for tif in _tifs_for_date(date):
        all_features.extend(_score_tile(tif, model, band_min, band_max, device))

    return {"type": "FeatureCollection", "features": all_features}


@router.get("/map", response_class=HTMLResponse)
def map_page():
    return _MAP_HTML


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

_MAP_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>SmokeSignal — Wildfire Smoke Map</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
#map{width:100vw;height:100vh}

.panel{
  position:absolute;top:12px;right:12px;z-index:1000;
  background:rgba(15,15,20,.92);color:#e0e0e0;
  padding:16px 20px;border-radius:10px;
  box-shadow:0 4px 20px rgba(0,0,0,.5);
  min-width:230px;backdrop-filter:blur(8px);
}
.panel h2{font-size:16px;margin-bottom:12px;color:#fff}
.panel label{font-size:13px;color:#aaa}
.panel select{
  width:100%;margin-top:4px;padding:6px 8px;
  border-radius:6px;border:1px solid #444;
  background:#1a1a24;color:#fff;font-size:13px;
}
.status{
  margin-top:12px;padding:8px 10px;
  background:rgba(255,165,0,.15);border-left:3px solid orange;
  border-radius:4px;font-size:12px;color:#ffcc80;display:none;
}
.stats{margin-top:10px;font-size:12px;color:#999}
.legend{margin-top:14px;border-top:1px solid #333;padding-top:10px}
.legend-title{font-size:11px;text-transform:uppercase;letter-spacing:1px;color:#777;margin-bottom:6px}
.legend-row{display:flex;align-items:center;gap:8px;margin:3px 0;font-size:12px}
.legend-swatch{width:14px;height:14px;border-radius:3px}

.basemap-row{margin-top:10px;display:flex;align-items:center;gap:6px;font-size:12px}
.basemap-row select{width:auto;margin-top:0}
</style>
</head>
<body>
<div id="map"></div>
<div class="panel">
  <h2>&#128293; SmokeSignal</h2>
  <label>Acquisition date<select id="date-select"><option>Loading…</option></select></label>

  <div class="basemap-row">
    <label>Basemap
      <select id="basemap-select">
        <option value="esri">ESRI Satellite</option>
        <option value="osm">OpenStreetMap</option>
      </select>
    </label>
  </div>

  <div class="status" id="status"></div>
  <div class="stats" id="stats"></div>

  <div class="legend">
    <div class="legend-title">Smoke confidence</div>
    <div class="legend-row"><div class="legend-swatch" style="background:#ff2200"></div> &gt; 75 %</div>
    <div class="legend-row"><div class="legend-swatch" style="background:#ff6600"></div> 60 – 75 %</div>
    <div class="legend-row"><div class="legend-swatch" style="background:#ffaa00"></div> 50 – 60 %</div>
  </div>
</div>

<script>
// ---- Map setup ----
const map = L.map('map').setView([39.9, -121.6], 10);

const basemaps = {
  esri: L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    {maxZoom: 18, attribution: '&copy; Esri'}
  ),
  osm: L.tileLayer(
    'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    {maxZoom: 18, attribution: '&copy; OpenStreetMap'}
  ),
};
let activeBasemap = basemaps.esri.addTo(map);

document.getElementById('basemap-select').addEventListener('change', e => {
  map.removeLayer(activeBasemap);
  activeBasemap = basemaps[e.target.value].addTo(map);
  activeBasemap.bringToBack();
});

// ---- Smoke overlay ----
let smokeLayer = null;

function smokeStyle(feature) {
  const c = feature.properties.confidence;
  return {
    fillColor: c > .75 ? '#ff2200' : c > .6 ? '#ff6600' : '#ffaa00',
    fillOpacity: Math.min(.65, c),
    color: '#fff',
    weight: .4,
    opacity: .5,
  };
}

function setStatus(msg) {
  const el = document.getElementById('status');
  el.textContent = msg;
  el.style.display = msg ? 'block' : 'none';
}

async function loadDate(date) {
  if (smokeLayer) { map.removeLayer(smokeLayer); smokeLayer = null; }
  document.getElementById('stats').textContent = '';
  setStatus('Analyzing tiles for smoke — first load may take a few minutes …');

  try {
    const resp = await fetch('/api/smoke?date=' + date);
    const data = await resp.json();

    smokeLayer = L.geoJSON(data, {
      style: smokeStyle,
      onEachFeature(feature, layer) {
        const pct = (feature.properties.confidence * 100).toFixed(1);
        layer.bindPopup('<b>Smoke</b><br>Confidence: ' + pct + ' %');
      }
    }).addTo(map);

    const n = data.features.length;
    document.getElementById('stats').textContent =
      n + ' smoke detection' + (n !== 1 ? 's' : '') + ' on ' + date;

    if (n > 0) smokeLayer.bringToFront();
  } catch (e) {
    console.error(e);
    setStatus('Error loading smoke data.');
    return;
  }
  setStatus('');
}

// ---- Initialise ----
fetch('/api/dates').then(r => r.json()).then(d => {
  const sel = document.getElementById('date-select');
  sel.innerHTML = '';
  d.dates.forEach(dt => {
    const o = document.createElement('option');
    o.value = dt; o.textContent = dt;
    sel.appendChild(o);
  });
  sel.addEventListener('change', () => loadDate(sel.value));
  if (d.dates.length) loadDate(d.dates[0]);
});
</script>
</body>
</html>
"""
