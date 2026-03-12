# SmokeSignal
Detecting Wildfire Haze from Space

A toy binary classifier that distinguishes **smoke** vs. **clear** scenes in Sentinel-2 satellite imagery, served as a FastAPI endpoint and deployable to Azure Container Instances.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt
```

## Usage

### 1. Fetch Sentinel-2 imagery

```powershell
python -m src.data.fetch --bbox -122.0 39.5 -121.2 40.3 --dates 2024-07-26/2024-08-10
```

Adjust `--bbox` and `--dates` to target your wildfire ROI. Downloads 5-band GeoTIFFs to `data/raw/`.

### 2. Extract patches

```powershell
python -m src.data.patch --input data/raw --output data/processed
```

Slices each GeoTIFF into 64×64 `.npy` patches and computes `data/processed/band_stats.json`.

### 3. Label patches

```powershell
python scripts/label_tool.py --data data/processed --labels data/labels.csv
```

Press **s** (smoke), **c** (clear), or **q** (quit & save).

### 4. Train the model

```powershell
python -m src.model.train --data data/processed --labels data/labels.csv --stats data/processed/band_stats.json --output models
```

### 5. Run the API locally

```powershell
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

- **Health check:** `GET http://localhost:8000/health`
- **Score a patch:**

```powershell
curl -X POST http://localhost:8000/score -F "file=@data/raw/some_patch.tif"
# → {"label": "smoke", "confidence": 0.87}
```

### 6. Run tests

```powershell
python -m pytest tests/ -v
```

### 7. Deploy to Azure (optional)

Copy `.env.example` to `.env`, fill in your Azure values, then:

```bash
bash scripts/deploy.sh
```

Requires `az` CLI authenticated with an existing resource group and container registry.
