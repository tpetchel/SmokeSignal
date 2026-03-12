"""Training loop and evaluation for the smoke classifier."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from src.data.dataset import SmokePatchDataset
from src.data.patch import compute_band_stats
from src.model.net import build_model

logger = logging.getLogger(__name__)

# ---------- hyperparameters ----------
EPOCHS = 20
BATCH_SIZE = 16
LR_BACKBONE = 1e-4
LR_HEAD = 1e-3
FREEZE_EPOCHS = 5
TRAIN_RATIO = 0.8
NUM_WORKERS = 0  # safe default; bump on Linux
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# -------------------------------------


def _get_augmentation() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
    ])


def _set_backbone_frozen(model: nn.Module, frozen: bool) -> None:
    """Freeze / unfreeze everything except the final ``fc`` layer."""
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = not frozen


def train(
    processed_dir: str | Path,
    labels_csv: str | Path,
    stats_path: str | Path,
    output_dir: str | Path = "models",
) -> Path:
    """Run the full training pipeline and return the path to the best checkpoint."""
    processed_dir = Path(processed_dir)
    stats_path = Path(stats_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- compute band stats if missing ---
    if not stats_path.exists():
        logger.info("Band stats not found at %s – computing from patches…", stats_path)
        compute_band_stats(processed_dir)

    # --- dataset / splits ---
    dataset = SmokePatchDataset(
        processed_dir,
        Path(labels_csv),
        stats_path,
        transform=_get_augmentation(),
    )
    n_train = int(len(dataset) * TRAIN_RATIO)
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # --- class imbalance weight ---
    labels = [s[1] for s in dataset.samples]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=DEVICE)

    # --- model / loss / optimizer ---
    model = build_model().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    head_params = [p for n, p in model.named_parameters() if n.startswith("fc.")]
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
    optimizer = AdamW([
        {"params": backbone_params, "lr": LR_BACKBONE},
        {"params": head_params, "lr": LR_HEAD},
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_f1 = 0.0
    best_path = output_dir / "best.pt"

    for epoch in range(1, EPOCHS + 1):
        # freeze / unfreeze backbone
        _set_backbone_frozen(model, frozen=(epoch <= FREEZE_EPOCHS))

        # --- train ---
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(batch_x).squeeze(1)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
        scheduler.step()
        avg_loss = running_loss / max(len(train_ds), 1)

        # --- validate ---
        model.eval()
        all_preds: list[int] = []
        all_labels: list[int] = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(DEVICE)
                logits = model(batch_x).squeeze(1)
                preds = (torch.sigmoid(logits) >= 0.5).int().cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(batch_y.int().tolist())

        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        logger.info(
            "Epoch %02d  loss=%.4f  acc=%.3f  prec=%.3f  rec=%.3f  f1=%.3f",
            epoch, avg_loss, acc, prec, rec, f1,
        )

        if f1 >= best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), best_path)
            logger.info("  ↳ saved best model (F1=%.3f)", best_f1)

    # --- final evaluation ---
    model.load_state_dict(torch.load(best_path, weights_only=True))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(DEVICE)
            logits = model(batch_x).squeeze(1)
            preds = (torch.sigmoid(logits) >= 0.5).int().cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(batch_y.int().tolist())

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    logger.info("Confusion matrix (rows=true, cols=pred):\n%s", cm)
    logger.info(
        "Final  acc=%.3f  prec=%.3f  rec=%.3f  f1=%.3f",
        accuracy_score(all_labels, all_preds),
        precision_score(all_labels, all_preds, zero_division=0),
        recall_score(all_labels, all_preds, zero_division=0),
        f1_score(all_labels, all_preds, zero_division=0),
    )
    return best_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import argparse

    parser = argparse.ArgumentParser(description="Train smoke classifier")
    parser.add_argument("--data", default="data/processed", help="Processed patches dir")
    parser.add_argument("--labels", default="data/labels.csv", help="Labels CSV")
    parser.add_argument("--stats", default="data/processed/band_stats.json", help="Band stats JSON")
    parser.add_argument("--output", default="models", help="Output dir for checkpoints")
    args = parser.parse_args()

    train(args.data, args.labels, args.stats, args.output)
