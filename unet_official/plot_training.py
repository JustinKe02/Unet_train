"""Plot training curves for official UNet MoS2 experiment.

Generates:
  1. Individual plots: loss, mIoU_fg, dice, per-class IoU
  2. A combined summary figure with all 4 subplots
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Config ──────────────────────────────────────────────────────────
CSV_PATH = Path("/root/autodl-tmp/Unet/unet_official/mos2_output/training_log.csv")
OUT_DIR = Path("/root/autodl-tmp/Unet/unet_official/mos2_output/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["background", "monolayer", "fewlayer", "multilayer"]
CLASS_COLORS = ["#7f8c8d", "#e74c3c", "#2ecc71", "#3498db"]

# ── Parse CSV ───────────────────────────────────────────────────────
epochs, train_loss, val_dice, val_miou, val_miou_fg = [], [], [], [], []
per_class = {name: [] for name in CLASS_NAMES}
lrs = []

with open(CSV_PATH) as f:
    reader = csv.DictReader(f)
    for row in reader:
        epochs.append(int(row["epoch"]))
        train_loss.append(float(row["train_loss"]))
        val_dice.append(float(row["val_dice"]))
        val_miou.append(float(row["val_miou"]))
        val_miou_fg.append(float(row["val_miou_fg"]))
        lrs.append(float(row["lr"]))
        for name in CLASS_NAMES:
            per_class[name].append(float(row[f"iou_{name}"]))

epochs = np.array(epochs)
best_idx = int(np.argmax(val_miou_fg))
best_epoch = epochs[best_idx]
best_val = val_miou_fg[best_idx]
print(f"Parsed {len(epochs)} epochs, best mIoU_fg={best_val:.4f} at epoch {best_epoch}")

plt.style.use("seaborn-v0_8-whitegrid")

# ── Helper ──────────────────────────────────────────────────────────
def mark_best(ax, best_ep, best_v):
    ax.axvline(x=best_ep, color="green", linestyle="--", alpha=0.5, label=f"Best (ep{best_ep})")

# ── 1. Individual: Training Loss ───────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs, train_loss, "o-", color="#e74c3c", markersize=2, linewidth=1.2, label="Train Loss")
mark_best(ax, best_epoch, best_val)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.set_title("Official UNet — Training Loss", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig(OUT_DIR / "01_train_loss.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 01_train_loss.png")

# ── 2. Individual: Val Dice ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs, val_dice, "o-", color="#3498db", markersize=2, linewidth=1.2, label="Val Dice")
ax.axhline(y=val_dice[best_idx], color="gold", linestyle="--", alpha=0.7,
           label=f"Best={val_dice[best_idx]:.4f}")
mark_best(ax, best_epoch, best_val)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Dice Score", fontsize=12)
ax.set_title("Official UNet — Validation Dice Score", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig(OUT_DIR / "02_val_dice.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 02_val_dice.png")

# ── 3. Individual: mIoU_fg ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs, val_miou_fg, "o-", color="#2ecc71", markersize=2, linewidth=1.2, label="Val mIoU_fg")
ax.axhline(y=best_val, color="gold", linestyle="--", alpha=0.7,
           label=f"Best={best_val:.4f}")
mark_best(ax, best_epoch, best_val)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("mIoU (foreground)", fontsize=12)
ax.set_title("Official UNet — Foreground mIoU", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig(OUT_DIR / "03_miou_fg.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 03_miou_fg.png")

# ── 4. Individual: Per-class IoU ───────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for name, color in zip(CLASS_NAMES, CLASS_COLORS):
    ax.plot(epochs, per_class[name], "o-", color=color, markersize=2, linewidth=1.2, label=name)
mark_best(ax, best_epoch, best_val)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("IoU", fontsize=12)
ax.set_title("Official UNet — Per-class IoU", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig(OUT_DIR / "04_perclass_iou.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 04_perclass_iou.png")

# ── 5. Individual: Learning Rate ───────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(epochs, lrs, "s-", color="#9b59b6", markersize=2, linewidth=1.2, label="Learning Rate")
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("LR", fontsize=12)
ax.set_yscale("log")
ax.set_title("Official UNet — Learning Rate Schedule", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig(OUT_DIR / "05_learning_rate.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 05_learning_rate.png")

# ── 6. Summary: All-in-one ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f"Official UNet Training Summary (Best mIoU_fg={best_val:.4f} @ Epoch {best_epoch})",
             fontsize=16, fontweight="bold")

# (0,0) Loss
ax = axes[0, 0]
ax.plot(epochs, train_loss, "o-", color="#e74c3c", markersize=2, linewidth=1.2, label="Train Loss")
mark_best(ax, best_epoch, best_val)
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.set_title("Training Loss"); ax.legend(fontsize=9)

# (0,1) mIoU_fg
ax = axes[0, 1]
ax.plot(epochs, val_miou_fg, "o-", color="#2ecc71", markersize=2, linewidth=1.2, label="Val mIoU_fg")
ax.axhline(y=best_val, color="gold", linestyle="--", alpha=0.7, label=f"Best={best_val:.4f}")
mark_best(ax, best_epoch, best_val)
ax.set_xlabel("Epoch"); ax.set_ylabel("mIoU_fg")
ax.set_title("Foreground mIoU"); ax.legend(fontsize=9)

# (1,0) Val Dice
ax = axes[1, 0]
ax.plot(epochs, val_dice, "o-", color="#3498db", markersize=2, linewidth=1.2, label="Val Dice")
ax.axhline(y=val_dice[best_idx], color="gold", linestyle="--", alpha=0.7,
           label=f"Best={val_dice[best_idx]:.4f}")
mark_best(ax, best_epoch, best_val)
ax.set_xlabel("Epoch"); ax.set_ylabel("Dice")
ax.set_title("Validation Dice"); ax.legend(fontsize=9)

# (1,1) Per-class IoU
ax = axes[1, 1]
for name, color in zip(CLASS_NAMES, CLASS_COLORS):
    ax.plot(epochs, per_class[name], "o-", color=color, markersize=2, linewidth=1.2, label=name)
mark_best(ax, best_epoch, best_val)
ax.set_xlabel("Epoch"); ax.set_ylabel("IoU")
ax.set_title("Per-class IoU"); ax.legend(fontsize=9)

plt.tight_layout()
fig.savefig(OUT_DIR / "00_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 00_summary.png")

print(f"\nAll plots saved to {OUT_DIR}")
