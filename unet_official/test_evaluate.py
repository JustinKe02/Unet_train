"""Evaluate official UNet on MoS2 test set.

Loads the best checkpoint, runs inference on all test samples,
computes per-image and aggregate metrics, and saves:
  - Per-image prediction visualizations
  - Aggregate metrics CSV
  - Confusion matrix plot
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unet import UNet
from evaluate_mos2 import compute_confusion_matrix, iou_from_confusion

# ── Config ──────────────────────────────────────────────────────────
SPLIT_JSON = Path("/root/autodl-tmp/Unet/splits/mos2_main_only_split.json")
CKPT_PATH = Path("/root/autodl-tmp/Unet/unet_official/mos2_output/checkpoints/best_model.pth")
OUT_DIR = Path("/root/autodl-tmp/Unet/unet_official/mos2_output/test_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 4
CLASS_NAMES = ["background", "monolayer", "fewlayer", "multilayer"]
PALETTE = np.array([
    [0, 0, 0],        # background - black
    [239, 41, 41],    # monolayer - red
    [0, 170, 0],      # fewlayer - green
    [114, 159, 207],  # multilayer - blue
], dtype=np.uint8)
CROP_SIZE = 896

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    """Convert class-index mask to RGB image."""
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for i, c in enumerate(PALETTE):
        color[mask == i] = c
    return color


def sliding_window_inference(model, image: np.ndarray, crop_size: int, stride: int = 640):
    """Run inference with sliding window for large images."""
    h, w = image.shape[:2]
    # Pad if needed
    pad_h = max(0, crop_size - h)
    pad_w = max(0, crop_size - w)
    if pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

    h_pad, w_pad = image.shape[:2]
    logits_sum = torch.zeros((NUM_CLASSES, h_pad, w_pad), dtype=torch.float32, device=DEVICE)
    count = torch.zeros((1, h_pad, w_pad), dtype=torch.float32, device=DEVICE)

    for y in range(0, h_pad - crop_size + 1, stride):
        for x in range(0, w_pad - crop_size + 1, stride):
            crop = image[y:y + crop_size, x:x + crop_size]
            tensor = torch.from_numpy(crop.transpose(2, 0, 1).astype(np.float32) / 255.0)
            tensor = tensor.unsqueeze(0).to(DEVICE, memory_format=torch.channels_last)

            with torch.no_grad():
                pred = model(tensor)  # (1, C, H, W)

            logits_sum[:, y:y + crop_size, x:x + crop_size] += pred.squeeze(0)
            count[:, y:y + crop_size, x:x + crop_size] += 1

    # Handle remaining edges
    if (h_pad - crop_size) % stride != 0:
        y = h_pad - crop_size
        for x in range(0, w_pad - crop_size + 1, stride):
            crop = image[y:y + crop_size, x:x + crop_size]
            tensor = torch.from_numpy(crop.transpose(2, 0, 1).astype(np.float32) / 255.0)
            tensor = tensor.unsqueeze(0).to(DEVICE, memory_format=torch.channels_last)
            with torch.no_grad():
                pred = model(tensor)
            logits_sum[:, y:y + crop_size, x:x + crop_size] += pred.squeeze(0)
            count[:, y:y + crop_size, x:x + crop_size] += 1

    if (w_pad - crop_size) % stride != 0:
        x = w_pad - crop_size
        for y in range(0, h_pad - crop_size + 1, stride):
            crop = image[y:y + crop_size, x:x + crop_size]
            tensor = torch.from_numpy(crop.transpose(2, 0, 1).astype(np.float32) / 255.0)
            tensor = tensor.unsqueeze(0).to(DEVICE, memory_format=torch.channels_last)
            with torch.no_grad():
                pred = model(tensor)
            logits_sum[:, y:y + crop_size, x:x + crop_size] += pred.squeeze(0)
            count[:, y:y + crop_size, x:x + crop_size] += 1

    avg_logits = logits_sum / count.clamp(min=1)
    pred_mask = avg_logits.argmax(dim=0).cpu().numpy()[:h, :w]
    return pred_mask


def compute_sample_metrics(pred: np.ndarray, gt: np.ndarray):
    """Compute per-class IoU for a single sample."""
    pred_t = torch.from_numpy(pred).to(DEVICE)
    gt_t = torch.from_numpy(gt).to(DEVICE)
    cm = compute_confusion_matrix(pred_t, gt_t, NUM_CLASSES)
    per_iou = iou_from_confusion(cm).cpu().numpy()
    return per_iou


def main():
    # Load split
    with open(SPLIT_JSON) as f:
        splits = json.load(f)["splits"]
    test_samples = splits["test"]
    print(f"Test set: {len(test_samples)} samples")

    # Load model
    model = UNet(n_channels=3, n_classes=NUM_CLASSES, bilinear=False)
    state_dict = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
    if "mask_values" in state_dict:
        del state_dict["mask_values"]
    model.load_state_dict(state_dict)
    model = model.to(DEVICE, memory_format=torch.channels_last)
    model.eval()
    print(f"Model loaded from {CKPT_PATH}")

    # Run inference
    total_cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=DEVICE)
    results = []
    vis_dir = OUT_DIR / "visualizations"
    vis_dir.mkdir(exist_ok=True)

    for i, sample in enumerate(test_samples):
        sid = sample["sample_id"]
        image = np.array(Image.open(sample["image_path"]).convert("RGB"))
        gt_mask = np.array(Image.open(sample["mask_path"]).convert("L"))

        pred_mask = sliding_window_inference(model, image, CROP_SIZE, stride=640)

        # Metrics
        per_iou = compute_sample_metrics(pred_mask, gt_mask)
        miou = float(np.mean(per_iou))
        miou_fg = float(np.mean(per_iou[1:]))

        # Accumulate confusion matrix
        pred_t = torch.from_numpy(pred_mask).to(DEVICE)
        gt_t = torch.from_numpy(gt_mask).to(DEVICE)
        total_cm += compute_confusion_matrix(pred_t, gt_t, NUM_CLASSES)

        results.append({
            "sample_id": sid,
            "miou": miou,
            "miou_fg": miou_fg,
            **{f"iou_{CLASS_NAMES[j]}": float(per_iou[j]) for j in range(NUM_CLASSES)},
        })

        # Visualization: image | GT | prediction
        gt_color = mask_to_color(gt_mask)
        pred_color = mask_to_color(pred_mask)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(image); axes[0].set_title("Original Image", fontsize=12)
        axes[1].imshow(gt_color); axes[1].set_title("Ground Truth", fontsize=12)
        axes[2].imshow(pred_color); axes[2].set_title(f"Prediction (mIoU_fg={miou_fg:.3f})", fontsize=12)
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"{sid} — Per-class IoU: " +
                     ", ".join(f"{CLASS_NAMES[j]}={per_iou[j]:.3f}" for j in range(NUM_CLASSES)),
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(vis_dir / f"{sid}.png", dpi=120, bbox_inches="tight")
        plt.close()

        print(f"  [{i+1}/{len(test_samples)}] {sid}: mIoU_fg={miou_fg:.4f}  "
              f"IoU=[{', '.join(f'{v:.3f}' for v in per_iou)}]")

    # ── Aggregate metrics ───────────────────────────────────────────
    total_iou = iou_from_confusion(total_cm).cpu().numpy()
    total_miou = float(np.mean(total_iou))
    total_miou_fg = float(np.mean(total_iou[1:]))

    print(f"\n{'='*60}")
    print(f"Test Set Aggregate Results:")
    print(f"  mIoU     = {total_miou:.4f}")
    print(f"  mIoU_fg  = {total_miou_fg:.4f}")
    for j, name in enumerate(CLASS_NAMES):
        print(f"  IoU_{name:12s} = {total_iou[j]:.4f}")
    print(f"{'='*60}")

    # Save per-image CSV
    csv_path = OUT_DIR / "test_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Per-image results saved to {csv_path}")

    # Save aggregate summary
    summary_path = OUT_DIR / "test_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Official UNet Test Evaluation\n")
        f.write(f"Checkpoint: {CKPT_PATH}\n")
        f.write(f"Test samples: {len(test_samples)}\n\n")
        f.write(f"mIoU     = {total_miou:.4f}\n")
        f.write(f"mIoU_fg  = {total_miou_fg:.4f}\n\n")
        for j, name in enumerate(CLASS_NAMES):
            f.write(f"IoU_{name:12s} = {total_iou[j]:.4f}\n")

    # ── Confusion matrix plot ───────────────────────────────────────
    cm_np = total_cm.cpu().numpy().astype(float)
    cm_norm = cm_np / cm_np.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = cm_norm[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontsize=11)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"Confusion Matrix (Test Set, mIoU_fg={total_miou_fg:.4f})", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {OUT_DIR / 'confusion_matrix.png'}")

    print("\nDone!")


if __name__ == "__main__":
    main()
