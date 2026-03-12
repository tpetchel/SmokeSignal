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


def run(processed_dir: Path, labels_csv: Path) -> None:
    existing = load_existing_labels(labels_csv)
    patches = sorted(processed_dir.glob("*.npy"))
    unlabelled = [p for p in patches if p.stem not in existing]

    if not unlabelled:
        print("All patches already labelled.")
        return

    print(f"{len(unlabelled)} unlabelled patches. Press s=smoke, c=clear, q=quit.")
    new_labels: dict[str, str] = {}

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    for patch_path in unlabelled:
        arr = np.load(patch_path)  # (5, 64, 64)
        # RGB composite: bands 2, 1, 0 → R, G, B  (B04, B03, B02)
        rgb = np.stack([arr[2], arr[1], arr[0]], axis=-1)
        # Clip to visible range for display
        rgb = np.clip(rgb / np.percentile(rgb, 98), 0, 1)

        ax.clear()
        ax.imshow(rgb)
        ax.set_title(patch_path.stem)
        ax.axis("off")
        fig.canvas.draw()
        plt.pause(0.01)

        while True:
            key = input(f"  [{patch_path.stem}] (s)moke / (c)lear / (q)uit: ").strip().lower()
            if key in ("s", "c", "q"):
                break
            print("  Invalid key. Use s, c, or q.")

        if key == "q":
            break
        new_labels[patch_path.stem] = "smoke" if key == "s" else "clear"

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
