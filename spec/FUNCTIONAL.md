# Functional Spec — SmokeSignal

Wildfires are a growing climate challenge, and smoke detection is critical for air‑quality alerts and disaster response. In this mini‑Hackathon project, we’ll build a toy classifier that distinguishes “smoke” vs. “clear” scenes using Sentinel‑2 imagery from Microsoft’s Planetary Computer.

The workflow:

- Fetches imagery for recent wildfire events via STAC API.
- Extracts small patches and hand‑label ~100 samples.
- Trains a lightweight CNN using PyTorch transfer learning.
- Deploys inference as a simple /score endpoint on Azure Container Instances (ACI).

Why this works for a one‑day sprint:

- Minimal dataset (small patches, binary labels).
- Popular frameworks (PyTorch, planetary‑computer SDK).
- Clear Azure integration path (ACI deployment).
- High learning value: geospatial data handling, model training, and cloud deployment, all in one day.

Stretch goals:

- Add a quick visualization dashboard (MapLibre or Streamlit).
- Explore metrics like precision/recall for imbalanced classes.