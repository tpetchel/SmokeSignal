"""PyTorch Dataset for smoke/clear patch classification."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.patch import normalize_patch


class SmokePatchDataset(Dataset):
    """Loads labelled (NUM_BANDS, 64, 64) patches with per-band normalization.

    Parameters
    ----------
    processed_dir : directory containing ``.npy`` patch files.
    labels_csv : CSV with columns ``patch_id,label`` (label is ``smoke`` or ``clear``).
    stats_path : JSON file with ``{"min": [...], "max": [...]}``.
    transform : optional callable applied to the tensor *after* normalization.
    """

    LABEL_MAP = {"clear": 0, "smoke": 1}

    def __init__(
        self,
        processed_dir: Path,
        labels_csv: Path,
        stats_path: Path,
        transform=None,
    ):
        self.processed_dir = Path(processed_dir)
        self.transform = transform

        stats = json.loads(Path(stats_path).read_text())
        self.band_min: list[float] = stats["min"]
        self.band_max: list[float] = stats["max"]

        self.samples: list[tuple[Path, int]] = []
        with open(labels_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                npy_path = self.processed_dir / f"{row['patch_id']}.npy"
                if npy_path.exists():
                    self.samples.append(
                        (npy_path, self.LABEL_MAP[row["label"]])
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[idx]
        arr = np.load(path)
        arr = normalize_patch(arr, self.band_min, self.band_max)
        tensor = torch.from_numpy(arr)  # (5, 64, 64)

        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor, torch.tensor(label, dtype=torch.float32)
