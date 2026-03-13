"""Simple CLI labelling tool for smoke/clear patches.

Displays an RGB composite of each unlabelled patch.
Press  s = smoke,  c = clear,  q = quit & save.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_existing_labels(labels_csv: Path) -> dict[str, str]:
    """Return {patch_id: label} from an existing CSV, or empty dict."""
    if not labels_csv.exists():
        return {}
    out: dict[str, str] = {}
    with open(labels_csv, newline="") as f:
        for row in csv.DictReader(f):
            out[row["patch_id"]] = row["label"]
    return out


def _enumerate_patches(processed_dir: Path) -> list[tuple[str, Path, int]]:
    """Return ``(patch_id, tile_path, index)`` for every patch in *processed_dir*."""
    patches: list[tuple[str, Path, int]] = []
    for tile_path in sorted(processed_dir.glob("*.npy")):
        arr = np.load(tile_path, mmap_mode="r")
        stem = tile_path.stem
        for i in range(arr.shape[0]):
            patches.append((f"{stem}_p{i:05d}", tile_path, i))
    return patches


def _load_guesses(processed_dir: Path) -> dict[str, str]:
    """Load smoke/clear candidate CSVs into a {patch_id: guess} dict."""
    guesses: dict[str, str] = {}
    for name, label in [("smoke_candidates.csv", "smoke"), ("clear_candidates.csv", "clear")]:
        path = processed_dir / name
        if not path.exists():
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                guesses[row["patch_id"]] = label
    return guesses


def run(processed_dir: Path, labels_csv: Path) -> None:
    existing = load_existing_labels(labels_csv)
    all_patches = _enumerate_patches(processed_dir)
    unlabelled = [(pid, tp, idx) for pid, tp, idx in all_patches if pid not in existing]

    if not unlabelled:
        print("All patches already labelled.")
        return

    guesses = _load_guesses(processed_dir)

    print(f"{len(unlabelled)} unlabelled patches. Press s=smoke, c=clear, q=quit.")
    new_labels: dict[str, str] = {}
    smoke_count = 0
    clear_count = 0

    tile_cache: dict[Path, np.ndarray] = {}

    fig, (ax_rgb, ax_swir) = plt.subplots(1, 2, figsize=(8, 4))
    # Disable default matplotlib keybindings that conflict with our labels
    for action in ("save", "quit", "close"):
        try:
            plt.rcParams[f"keymap.{action}"] = []
        except KeyError:
            pass
    for pid, tile_path, idx in unlabelled:
        if tile_path not in tile_cache:
            tile_cache[tile_path] = np.load(tile_path, mmap_mode="r")
        arr = np.array(tile_cache[tile_path][idx])  # (5, 64, 64)
        # Band order: B02(0) B03(1) B04(2) B11(3) B12(4)

        # True-color RGB: B04, B03, B02
        rgb = np.stack([arr[2], arr[1], arr[0]], axis=-1)
        rgb = np.clip(rgb / max(np.percentile(rgb, 98), 1e-6), 0, 1)

        # SWIR false-color: B12, B11, B04 — smoke appears brownish; clouds stay bright white
        swir = np.stack([arr[4], arr[3], arr[2]], axis=-1)
        swir = np.clip(swir / max(np.percentile(swir, 98), 1e-6), 0, 1)

        ax_rgb.clear()
        ax_rgb.imshow(rgb)
        ax_rgb.set_title("RGB")
        ax_rgb.axis("off")

        ax_swir.clear()
        ax_swir.imshow(swir)
        ax_swir.set_title("SWIR (smoke=brown, cloud=white)")
        ax_swir.axis("off")

        guess = guesses.get(pid, "-")
        fig.suptitle(
            f"{pid}  —  (s)moke / (c)lear / (q)uit\n"
            f"guess: {guess}  |  smoke: {smoke_count}  |  clear: {clear_count}  |  total: {smoke_count + clear_count}",
            fontsize=10,
        )
        fig.canvas.draw()
        plt.pause(0.01)

        key_result: list[str | None] = [None]

        def _on_key(event: object) -> None:
            if getattr(event, "key", None) in ("s", "c", "q"):
                key_result[0] = event.key  # type: ignore[union-attr]

        cid = fig.canvas.mpl_connect("key_press_event", _on_key)
        while key_result[0] is None:
            plt.pause(0.05)
        fig.canvas.mpl_disconnect(cid)
        key = key_result[0]

        if key == "q":
            break
        if key == "s":
            smoke_count += 1
        else:
            clear_count += 1
        new_labels[pid] = "smoke" if key == "s" else "clear"

    plt.close(fig)

    # Append new labels
    if new_labels:
        write_header = not labels_csv.exists()
        with open(labels_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["patch_id", "label"])
            if write_header:
                writer.writeheader()
            for pid, lab in new_labels.items():
                writer.writerow({"patch_id": pid, "label": lab})
        print(f"Saved {len(new_labels)} new labels → {labels_csv}")
    else:
        print("No new labels saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label smoke/clear patches")
    parser.add_argument("--data", default="data/processed", help="Processed patches dir")
    parser.add_argument("--labels", default="data/labels.csv", help="Labels CSV path")
    args = parser.parse_args()
    run(Path(args.data), Path(args.labels))
