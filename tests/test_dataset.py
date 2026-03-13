"""Tests for SmokePatchDataset and patch utilities."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.data.dataset import SmokePatchDataset
from src.data.patch import compute_band_stats, normalize_patch


@pytest.fixture()
def tmp_dataset(tmp_path: Path):
    """Create a tiny on-disk dataset (4 patches in a single stacked tile)."""
    processed = tmp_path / "processed"
    processed.mkdir()

    rng = np.random.default_rng(0)
    tile_stem = "test_tile"
    patches = rng.integers(100, 5000, size=(4, 5, 64, 64)).astype(np.float32)
    np.save(processed / f"{tile_stem}.npy", patches)
    patch_ids = [f"{tile_stem}_p{i:05d}" for i in range(4)]

    # band stats
    stats = compute_band_stats(processed)
    stats_path = processed / "band_stats.json"

    # labels
    labels_csv = tmp_path / "labels.csv"
    with open(labels_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patch_id", "label"])
        w.writeheader()
        for i, pid in enumerate(patch_ids):
            w.writerow({"patch_id": pid, "label": "smoke" if i < 2 else "clear"})

    return processed, labels_csv, stats_path


def test_dataset_length(tmp_dataset):
    processed, labels_csv, stats_path = tmp_dataset
    ds = SmokePatchDataset(processed, labels_csv, stats_path)
    assert len(ds) == 4


def test_dataset_item_shape(tmp_dataset):
    processed, labels_csv, stats_path = tmp_dataset
    ds = SmokePatchDataset(processed, labels_csv, stats_path)
    tensor, label = ds[0]
    assert tensor.shape == (5, 64, 64)
    assert label.shape == ()


def test_dataset_normalization_range(tmp_dataset):
    processed, labels_csv, stats_path = tmp_dataset
    ds = SmokePatchDataset(processed, labels_csv, stats_path)
    tensor, _ = ds[0]
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_normalize_patch_identity():
    """A patch already at [0,1] with matching stats should stay unchanged."""
    patch = np.random.rand(5, 8, 8).astype(np.float32)
    band_min = [0.0] * 5
    band_max = [1.0] * 5
    result = normalize_patch(patch, band_min, band_max)
    np.testing.assert_allclose(result, patch, atol=1e-6)
