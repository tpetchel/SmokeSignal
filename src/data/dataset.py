"""PyTorch Dataset for smoke/clear patch classification."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.patch import normalize_patch


def _parse_patch_id(patch_id: str) -> tuple[str, int]:
    """Parse ``'TILE_STEM_p00042'`` → ``('TILE_STEM', 42)``."""
    stem, sep, idx_str = patch_id.rpartition("_p")
    if not sep:
        raise ValueError(f"Invalid patch id: {patch_id!r}")
    return stem, int(idx_str)


class SmokePatchDataset(Dataset):
    """Loads labelled (NUM_BANDS, 64, 64) patches with per-band normalization.

    Patches are stored as stacked ``.npy`` tile files with shape
    ``(N, 5, 64, 64)``.  Each patch is identified by
    ``<tile_stem>_p<index>`` and looked up via memory-mapped I/O.

    Parameters
    ----------
    processed_dir : directory containing per-tile ``.npy`` files.
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

        self._tile_cache: dict[str, np.ndarray] = {}

        self.samples: list[tuple[str, int, int]] = []
        with open(labels_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tile_stem, patch_idx = _parse_patch_id(row["patch_id"])
                tile_path = self.processed_dir / f"{tile_stem}.npy"
                if tile_path.exists():
                    self.samples.append(
                        (tile_stem, patch_idx, self.LABEL_MAP[row["label"]])
                    )

    def _load_tile(self, tile_stem: str) -> np.ndarray:
        if tile_stem not in self._tile_cache:
            path = self.processed_dir / f"{tile_stem}.npy"
            self._tile_cache[tile_stem] = np.load(path, mmap_mode="r")
        return self._tile_cache[tile_stem]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tile_stem, patch_idx, label = self.samples[idx]
        tile = self._load_tile(tile_stem)
        arr = np.array(tile[patch_idx])  # copy from mmap → regular array
        arr = normalize_patch(arr, self.band_min, self.band_max)
        tensor = torch.from_numpy(arr)  # (5, 64, 64)

        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor, torch.tensor(label, dtype=torch.float32)
