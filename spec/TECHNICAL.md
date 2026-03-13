# Technical Spec — SmokeSignal

> Detecting Wildfire Haze from Space

This document translates the [functional spec](FUNCTIONAL.md) into an implementable technical design. It covers project layout, data pipeline, model architecture, inference API, containerisation, deployment, and stretch goals.

---

## 1  Project Layout

```
SmokeSignal/
├── spec/
│   ├── FUNCTIONAL.md
│   └── TECHNICAL.md          # ← this file
├── data/
│   ├── raw/                  # downloaded Sentinel-2 patches (GeoTIFF)
│   ├── processed/            # stacked per-tile .npy arrays (N, 5, 64, 64)
│   └── labels.csv            # columns: patch_id, label (smoke | clear)
├── src/
│   ├── data/
│   │   ├── fetch.py          # STAC search & download
│   │   ├── patch.py          # patch extraction & normalization
│   │   └── dataset.py        # PyTorch Dataset class
│   ├── model/
│   │   ├── net.py            # ResNet-18 wrapper with 5-band input head
│   │   └── train.py          # training loop & evaluation
│   └── api/
│       ├── app.py            # FastAPI application
│       └── score.py          # /score endpoint handler
├── scripts/
│   ├── label_tool.py         # simple CLI/GUI labelling helper
│   └── deploy.sh             # ACI deployment script
├── tests/
│   ├── test_dataset.py
│   ├── test_model.py
│   └── test_api.py
├── Dockerfile
├── requirements.txt
├── .env.example              # template for secrets/config
├── .gitignore
└── README.md
```

---

## 2  Data Pipeline

### 2.1  Imagery Source

| Property | Value |
|---|---|
| Collection | `sentinel-2-l2a` |
| Provider | Microsoft Planetary Computer STAC API |
| SDK | `pystac-client` + `planetary-computer` (token signing) |
| Bands | B02 (Blue), B03 (Green), B04 (Red), B11 (SWIR-1), B12 (SWIR-2) |
| Resolution | 10 m/px (B02–B04), 20 m/px (B11–B12) — B11/B12 resampled to 10 m |

### 2.2  Region of Interest (ROI) Selection

Target **3–5 recent wildfire events** from public records (e.g., NIFC InciWeb, FIRMS). For each event, define a bounding box covering the fire perimeter plus a 10 km buffer. Use cloud-cover filter ≤ 20 % and date range ± 7 days of the event.

### 2.3  STAC Search & Download (`src/data/fetch.py`)

```python
import planetary_computer as pc
from pystac_client import Client

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"
BANDS = ["B02", "B03", "B04", "B11", "B12"]
MAX_CLOUD = 20  # percent

def search_items(bbox, date_range):
    client = Client.open(STAC_URL, modifier=pc.sign_inplace)
    search = client.search(
        collections=[COLLECTION],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": MAX_CLOUD}},
    )
    return list(search.items())
```

Download each band asset as a GeoTIFF using `rasterio`. Resample B11/B12 from 20 m to 10 m with bilinear interpolation to align with the visible bands.

### 2.4  Patch Extraction (`src/data/patch.py`)

- **Patch size**: 64 × 64 pixels (640 m × 640 m at 10 m/px).
- **Stride**: 32 px (50 % overlap) to increase sample count.
- **Output**: 5-channel NumPy arrays stacked per tile as `(N, 5, 64, 64)` `float32`, saved to `data/processed/<tile_stem>.npy`.
- **Normalization**: per-band min-max scaling to `[0, 1]` using dataset-wide statistics computed once after all patches are extracted.

### 2.5  Labelling

Target **~100 labelled patches** (≥ 40 smoke, ≥ 40 clear).

| Approach | Detail |
|---|---|
| Tool | `scripts/label_tool.py` — renders an RGB composite of each patch; user presses `s` (smoke) or `c` (clear). |
| Storage | `data/labels.csv` — columns: `patch_id`, `label`. |
| Guideline | A patch is "smoke" if ≥ 25 % of its area is visibly hazy/opaque. Otherwise "clear". |

---

## 3  Model

### 3.1  Architecture

| Component | Detail |
|---|---|
| Backbone | **ResNet-18** (`torchvision.models.resnet18`, `weights=IMAGENET1K_V1`) |
| Input adapter | Replace `conv1` with `nn.Conv2d(5, 64, 7, stride=2, padding=3, bias=False)` to accept 5-band input. Initialize the first 3 filters from the pretrained RGB weights; initialize the 2 SWIR filters with Kaiming normal. |
| Classifier head | Replace `fc` with `nn.Linear(512, 1)` → sigmoid → binary output. |
| Loss | `BCEWithLogitsLoss` (handles class imbalance via `pos_weight`). |

### 3.2  Training (`src/model/train.py`)

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1 × 10⁻⁴ (backbone), 1 × 10⁻³ (head) |
| Scheduler | CosineAnnealingLR, T_max = 20 |
| Epochs | 20 |
| Batch size | 16 |
| Train / Val split | 80 / 20 stratified |
| Augmentation | Random horizontal flip, random vertical flip, random rotation (±15°) |

Freeze backbone for the first 5 epochs, then unfreeze all layers.

### 3.3  Evaluation

After training, log:

- **Accuracy**, **Precision**, **Recall**, **F1** on the validation set.
- **Confusion matrix** printed to console.
- Best model checkpoint saved to `models/best.pt`.

---

## 4  Inference API

### 4.1  Framework & Endpoint

| Property | Value |
|---|---|
| Framework | **FastAPI** with **Uvicorn** |
| Route | `POST /score` |
| Health | `GET /health` → `{"status": "ok"}` |

### 4.2  Request / Response

**Request** — multipart form upload of a 5-band GeoTIFF patch (64 × 64 × 5), or a JSON body with a base64-encoded NumPy array.

```
POST /score
Content-Type: multipart/form-data

file: <patch.tif>
```

**Response**

```json
{
  "label": "smoke",
  "confidence": 0.87
}
```

### 4.3  Inference Logic (`src/api/score.py`)

1. Read uploaded GeoTIFF with `rasterio`, extract 5 bands as `(5, 64, 64)` float32.
2. Apply the same per-band normalization used during training.
3. Run `model(tensor.unsqueeze(0))` → logit → `torch.sigmoid` → confidence.
4. Threshold at **0.5**: `label = "smoke" if confidence >= 0.5 else "clear"`.
5. Return JSON response.

Model is loaded once at startup via a lifespan event and held in memory.

---

## 5  Containerisation & Deployment

### 5.1  Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY models/best.pt models/best.pt

EXPOSE 8000
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 5.2  Azure Container Instances Deployment

```bash
RESOURCE_GROUP="smokesignal-rg"
ACR_NAME="smokesignalacr"
IMAGE="$ACR_NAME.azurecr.io/smokesignal:latest"

# Build & push
az acr build --registry $ACR_NAME --image smokesignal:latest .

# Deploy
az container create \
  --resource-group $RESOURCE_GROUP \
  --name smokesignal-api \
  --image $IMAGE \
  --cpu 1 --memory 1.5 \
  --ports 8000 \
  --dns-name-label smokesignal \
  --registry-login-server "$ACR_NAME.azurecr.io" \
  --registry-username "$(az acr credential show -n $ACR_NAME --query username -o tsv)" \
  --registry-password "$(az acr credential show -n $ACR_NAME --query 'passwords[0].value' -o tsv)"
```

The endpoint will be available at `http://smokesignal.<region>.azurecontainer.io:8000/score`.

---

## 6  Dependencies (`requirements.txt`)

```
torch>=2.2,<3
torchvision>=0.17,<1
fastapi>=0.110,<1
uvicorn[standard]>=0.29,<1
pystac-client>=0.8,<1
planetary-computer>=1.0,<2
rasterio>=1.3,<2
numpy>=1.26,<3
Pillow>=10,<11
scikit-learn>=1.4,<2
matplotlib>=3.8,<4
```

### Dev / Test extras

```
pytest>=8,<9
httpx>=0.27,<1        # for FastAPI TestClient
```

---

## 7  Environment Variables (`.env.example`)

```
# Planetary Computer (auto-signed, but can override)
PC_SDK_SUBSCRIPTION_KEY=

# Azure (for deployment only)
AZURE_RESOURCE_GROUP=smokesignal-rg
AZURE_ACR_NAME=smokesignalacr
```

---

## 8  Testing

| Test file | Scope |
|---|---|
| `tests/test_dataset.py` | Verify patch loading, shape `(5, 64, 64)`, normalization range `[0, 1]`. |
| `tests/test_model.py` | Instantiate model, feed random tensor, assert output shape `(1,)`. |
| `tests/test_api.py` | Use `httpx.AsyncClient` + FastAPI `TestClient` to POST a dummy patch and assert 200 + correct JSON schema. |

Run all tests:

```bash
pytest tests/ -v
```

---

## 9  Stretch Goals (Optional)

### 9.1  Visualization Dashboard

- **Framework**: Streamlit
- Display an interactive map (via `streamlit-folium` or `pydeck`) showing ROI bounding boxes.
- Upload a patch, call `/score`, and overlay the prediction on the map.
- Show the RGB composite alongside the SWIR false-color composite.

### 9.2  Precision / Recall Analysis

- Sweep the classification threshold from 0.1 to 0.9 in 0.05 steps.
- Plot a Precision-Recall curve and report AUPRC.
- Identify the threshold that maximizes F1 (useful when the dataset is imbalanced toward "clear").

---

## 10  Implementation Order

| Phase | Tasks | Estimated Effort |
|---|---|---|
| **1 — Data** | `fetch.py`, `patch.py`, `label_tool.py`, labelling session | ~3 h |
| **2 — Model** | `dataset.py`, `net.py`, `train.py`, training run | ~2 h |
| **3 — API** | `app.py`, `score.py`, local smoke-test | ~1 h |
| **4 — Deploy** | `Dockerfile`, ACR build, ACI deploy, end-to-end test | ~1 h |
| **5 — Stretch** | Dashboard and/or PR curves | remaining time |
