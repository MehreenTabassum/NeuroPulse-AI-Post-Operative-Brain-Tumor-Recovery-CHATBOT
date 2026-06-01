"""
finetune.py — Low-Resource Fine-Tuning for CNNViTFeatureExtractor
Post-Operative Brain Tumor Recovery Analysis · BraTS-2024

Designed for LIMITED compute (CPU-only or single GPU with <8 GB VRAM).

Strategy
--------
  • Tiny model config  : cnn_dim=64, vit_dim=64, vit_depth=2, vit_heads=2
  • Frozen CNN backbone : only the ViT branch + fusion + classifier are trained
                          (reduces trainable params by ~70%)
  • Gradient checkpointing: saves activation memory at small speed cost
  • Mixed precision (AMP): fp16 on CUDA, skipped gracefully on CPU
  • Small batch size (1-2) with gradient accumulation to simulate larger batches
  • Early stopping + best-checkpoint saving

Dataset Format
--------------
Organise your NIfTI files like this:

    data/
      train/
        class_0/   ← e.g. "stable" / "no progression"
          pt001_t1.nii.gz
          pt002_t1.nii.gz
          ...
        class_1/   ← e.g. "progression"
          pt010_t1.nii.gz
          ...
      val/
        class_0/
          ...
        class_1/
          ...

Each .nii / .nii.gz must be a 3-D volume (T1, T1CE, T2 or FLAIR).
If you only have one modality, it will be replicated to fill all 4 channels.

Usage
-----
  # Minimal:
  python finetune.py --data_dir ./data

  # Full options:
  python finetune.py \
      --data_dir   ./data          \
      --output_dir ./checkpoints   \
      --epochs     30              \
      --batch_size 1               \
      --accum_steps 4              \
      --lr         3e-4            \
      --workers    0               \
      --resume     ./checkpoints/best.pt

After training, point the vision service at your checkpoint:
  WEIGHTS_PATH=./checkpoints/best.pt python run_all.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ── Import your model ─────────────────────────────────────────────────────────
try:
    from vision import CNNViTFeatureExtractor, build_model
except ImportError:
    sys.exit(
        "ERROR: vision.py not found in the current directory.\n"
        "Run this script from the project root (same folder as vision.py)."
    )

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# §1  Dataset
# ─────────────────────────────────────────────────────────────────────────────

TARGET_SHAPE = (128, 128, 128)   # must match vision_service.py
N_MODALITIES = 4                 # T1, T1CE, T2, FLAIR — or replicated single channel


def _crop_or_pad(arr: np.ndarray, axis: int, target: int) -> np.ndarray:
    size = arr.shape[axis]
    if size >= target:
        start = (size - target) // 2
        slc = [slice(None)] * arr.ndim
        slc[axis] = slice(start, start + target)
        return arr[tuple(slc)]
    pad_before = (target - size) // 2
    pad_after  = target - size - pad_before
    pad_width  = [(0, 0)] * arr.ndim
    pad_width[axis] = (pad_before, pad_after)
    return np.pad(arr, pad_width, mode="constant", constant_values=0)


def _preprocess_volume(volume: np.ndarray) -> np.ndarray:
    """Resize to TARGET_SHAPE and z-score normalise (same as vision_service.py)."""
    vol = volume.astype(np.float32)
    for axis, target in enumerate(TARGET_SHAPE):
        vol = _crop_or_pad(vol, axis, target)
    mask = vol > 0
    if mask.sum() > 0:
        mean = vol[mask].mean()
        std  = vol[mask].std() + 1e-8
        vol  = np.where(mask, (vol - mean) / std, 0.0)
    return vol


def _augment(vol: np.ndarray, augment: bool) -> np.ndarray:
    """Lightweight augmentation — flips only (no extra libs needed)."""
    if not augment:
        return vol
    for axis in range(3):
        if random.random() < 0.5:
            vol = np.flip(vol, axis=axis).copy()
    return vol


class BraTSDataset(Dataset):
    """
    Reads single-modality NIfTI files from a class-labelled directory tree.
    The volume is replicated across all 4 channels so the model always
    receives (4, D, H, W) regardless of how many modalities you have.
    """

    EXTENSIONS = {".nii", ".gz"}

    def __init__(self, root: str | Path, augment: bool = False) -> None:
        self.augment = augment
        self.samples: list[tuple[Path, int]] = []
        self.class_names: list[str] = []

        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"Dataset root not found: {root}")

        class_dirs = sorted(p for p in root.iterdir() if p.is_dir())
        if not class_dirs:
            raise ValueError(f"No sub-directories (classes) found in {root}")

        self.class_names = [d.name for d in class_dirs]
        log.info("Classes found: %s", self.class_names)

        for label, class_dir in enumerate(class_dirs):
            files = [
                f for f in class_dir.rglob("*")
                if f.suffix in self.EXTENSIONS or f.name.endswith(".nii.gz")
            ]
            for f in files:
                self.samples.append((f, label))

        if not self.samples:
            raise ValueError(f"No .nii / .nii.gz files found under {root}")

        log.info("Dataset '%s': %d samples, %d classes", root.name, len(self), len(self.class_names))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        try:
            img = nib.load(str(path))
            volume = np.array(img.dataobj, dtype=np.float32)
        except Exception as exc:
            log.error("Failed to load %s: %s — returning zeros.", path, exc)
            volume = np.zeros(TARGET_SHAPE, dtype=np.float32)

        if volume.ndim == 4:
            volume = volume[..., 0]   # take first time-point / modality
        if volume.ndim != 3:
            volume = np.zeros(TARGET_SHAPE, dtype=np.float32)

        volume = _preprocess_volume(volume)
        volume = _augment(volume, self.augment)

        # Replicate single channel → 4-channel tensor (B, 4, D, H, W)
        tensor = torch.from_numpy(volume).unsqueeze(0).repeat(N_MODALITIES, 1, 1, 1)
        return tensor, torch.tensor(label, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# §2  Model factory — tiny config for low VRAM/RAM
# ─────────────────────────────────────────────────────────────────────────────

TINY_CONFIG = dict(
    cnn_dim=64,
    vit_dim=64,
    vit_depth=2,
    vit_heads=2,
    feature_dim=768,  # kept at 768 so saved features stay compatible
    dropout=0.15,
)


def _build_finetune_model(
    num_classes: int,
    weights_path: Optional[str],
    device: torch.device,
    freeze_cnn: bool = True,
) -> CNNViTFeatureExtractor:
    """
    Build the tiny CNN-ViT and optionally freeze the CNN backbone.

    Freezing the CNN backbone is the key RAM saver:
      • Only ViT + fusion + classifier are updated → ~70% fewer gradient tensors.
      • The frozen CNN still runs forward (needed for fusion), but no grads stored.
    """
    log.info("Building model | config=%s | device=%s", TINY_CONFIG, device)
    model = build_model(
        weights_path=weights_path,
        device=device,
        num_classes=num_classes,
        **TINY_CONFIG,
    )
    model.train()

    if freeze_cnn:
        for param in model.cnn_branch.parameters():
            param.requires_grad = False
        log.info("CNN backbone frozen — only ViT + fusion + classifier will be trained.")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    log.info("Parameters: %d trainable / %d total (%.1f%%)", trainable, total, 100*trainable/total)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# §3  Training utilities
# ─────────────────────────────────────────────────────────────────────────────

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.sum = self.count = 0.0
    def update(self, val, n=1): self.val = val; self.sum += val*n; self.count += n
    @property
    def avg(self): return self.sum / max(self.count, 1)


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def run_epoch(
    model:      CNNViTFeatureExtractor,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    scaler,                               # torch.cuda.amp.GradScaler or None
    device:     torch.device,
    accum_steps: int,
    is_train:   bool,
) -> tuple[float, float]:
    """Run one train or val epoch. Returns (avg_loss, avg_accuracy)."""
    model.train(is_train)
    ctx = torch.enable_grad() if is_train else torch.inference_mode()

    loss_meter = AverageMeter()
    acc_meter  = AverageMeter()

    optimizer.zero_grad()

    with ctx:
        for step, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            use_amp = scaler is not None
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(inputs)
                loss   = F.cross_entropy(logits, labels)

            if is_train:
                loss_scaled = loss / accum_steps
                if use_amp:
                    scaler.scale(loss_scaled).backward()
                else:
                    loss_scaled.backward()

                if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
                    if use_amp:
                        scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        (p for p in model.parameters() if p.requires_grad), max_norm=1.0
                    )
                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()

            bs = inputs.size(0)
            loss_meter.update(loss.item(), bs)
            acc_meter.update(_accuracy(logits, labels), bs)

    return loss_meter.avg, acc_meter.avg


# ─────────────────────────────────────────────────────────────────────────────
# §4  Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    # ── Reproducibility ───────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    if device.type == "cuda":
        log.info("GPU: %s | VRAM: %.1f GB",
                 torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1e9)

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Checkpoints will be saved to: %s", out_dir)

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = BraTSDataset(Path(args.data_dir) / "train", augment=True)
    val_ds   = BraTSDataset(Path(args.data_dir) / "val",   augment=False)

    num_classes = len(train_ds.class_names)
    log.info("Num classes: %d | %s", num_classes, train_ds.class_names)

    # Save class mapping
    with open(out_dir / "class_names.json", "w") as f:
        json.dump(train_ds.class_names, f, indent=2)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = _build_finetune_model(
        num_classes=num_classes,
        weights_path=args.resume,
        device=device,
        freeze_cnn=not args.unfreeze_cnn,
    )

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ── AMP scaler (CUDA only) ────────────────────────────────────────────────
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    if scaler:
        log.info("Mixed precision (fp16 AMP) enabled.")

    # ── Training ──────────────────────────────────────────────────────────────
    best_val_acc  = 0.0
    no_improve    = 0
    history: list[dict] = []

    log.info("Starting training for %d epochs | batch=%d | accum=%d | effective_batch=%d",
             args.epochs, args.batch_size, args.accum_steps,
             args.batch_size * args.accum_steps)

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()

        train_loss, train_acc = run_epoch(
            model, train_loader, optimizer, scaler, device, args.accum_steps, is_train=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, optimizer, scaler, device, args.accum_steps, is_train=False
        )

        scheduler.step()
        elapsed = time.perf_counter() - t0

        log.info(
            "Epoch %3d/%d | train_loss=%.4f | train_acc=%.3f | "
            "val_loss=%.4f | val_acc=%.3f | lr=%.2e | %.1fs",
            epoch, args.epochs, train_loss, train_acc, val_loss, val_acc,
            scheduler.get_last_lr()[0], elapsed,
        )

        row = dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                   val_loss=val_loss, val_acc=val_acc)
        history.append(row)

        # Save training log every epoch (append-friendly)
        with open(out_dir / "training_log.json", "w") as f:
            json.dump(history, f, indent=2)

        # ── Checkpoint: best val accuracy ─────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve   = 0
            ckpt = {
                "epoch":       epoch,
                "val_acc":     val_acc,
                "model_config": TINY_CONFIG | {"num_classes": num_classes},
                "class_names": train_ds.class_names,
                "state_dict":  model.state_dict(),
            }
            torch.save(ckpt, out_dir / "best.pt")
            log.info("  ✔ New best checkpoint saved (val_acc=%.3f)", val_acc)
        else:
            no_improve += 1

        # ── Periodic checkpoint ────────────────────────────────────────────────
        if epoch % args.save_every == 0:
            torch.save(
                {"epoch": epoch, "state_dict": model.state_dict()},
                out_dir / f"epoch_{epoch:04d}.pt",
            )

        # ── Early stopping ─────────────────────────────────────────────────────
        if args.patience > 0 and no_improve >= args.patience:
            log.info("Early stopping: no improvement for %d epochs.", args.patience)
            break

    log.info("Training complete. Best val_acc = %.4f", best_val_acc)
    log.info("Best checkpoint : %s", out_dir / "best.pt")
    log.info("To use in stack : WEIGHTS_PATH=%s python run_all.py", out_dir / "best.pt")


# ─────────────────────────────────────────────────────────────────────────────
# §5  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune CNNViTFeatureExtractor on a BraTS-style dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    p.add_argument("--data_dir",   default="./data",          help="Root dataset dir (must contain train/ and val/).")
    p.add_argument("--output_dir", default="./checkpoints",   help="Where to save checkpoints and logs.")
    p.add_argument("--resume",     default=None,              help="Path to a .pt checkpoint to resume from.")

    # Training
    p.add_argument("--epochs",      type=int,   default=30,   help="Total training epochs.")
    p.add_argument("--batch_size",  type=int,   default=1,    help="Per-GPU batch size. Keep at 1 if RAM is tight.")
    p.add_argument("--accum_steps", type=int,   default=4,    help="Gradient accumulation steps (simulates larger batch).")
    p.add_argument("--lr",          type=float, default=3e-4, help="AdamW learning rate.")
    p.add_argument("--patience",    type=int,   default=10,   help="Early stopping patience (0 = disabled).")
    p.add_argument("--save_every",  type=int,   default=5,    help="Save a periodic checkpoint every N epochs.")
    p.add_argument("--seed",        type=int,   default=42,   help="Random seed.")

    # Loader
    p.add_argument("--workers", type=int, default=0,
                   help="DataLoader workers. 0 = main process only (safest for Windows / limited RAM).")

    # Model
    p.add_argument("--unfreeze_cnn", action="store_true",
                   help="Unfreeze CNN backbone too (needs more RAM — not recommended for low-resource).")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Quick sanity check
    data_root = Path(args.data_dir)
    if not (data_root / "train").exists() or not (data_root / "val").exists():
        log.error(
            "Expected %s/train/ and %s/val/ — please organise your dataset first.\n"
            "See the docstring at the top of this file for the required folder structure.",
            data_root, data_root,
        )
        sys.exit(1)

    train(args)
