"""Evaluation with mIoU and per-class IoU for MoS2 segmentation."""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff


def compute_confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int):
    """Compute confusion matrix (num_classes x num_classes)."""
    mask = (target >= 0) & (target < num_classes)
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long, device=pred.device)
    indices = target[mask] * num_classes + pred[mask]
    cm = cm.reshape(-1).scatter_add_(0, indices.long(), torch.ones_like(indices, dtype=torch.long))
    return cm.reshape(num_classes, num_classes)


def iou_from_confusion(cm: torch.Tensor):
    """Compute per-class IoU from a confusion matrix."""
    intersection = cm.diag()
    union = cm.sum(dim=1) + cm.sum(dim=0) - intersection
    iou = intersection.float() / union.float().clamp(min=1)
    return iou


@torch.inference_mode()
def evaluate_mos2(net, dataloader, device, amp, num_classes=4):
    """Evaluate model and return metrics dict.

    Returns:
        dict with keys: dice, miou, miou_fg, per_class_iou (list)
    """
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    total_cm = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)

    with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc="Validation", unit="batch", leave=False):
            image, mask_true = batch["image"], batch["mask"]

            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            mask_pred = net(image)
            pred_classes = mask_pred.argmax(dim=1)

            # Dice score (ignoring background, same as official)
            if net.n_classes == 1:
                mask_pred_bin = (F.sigmoid(mask_pred) > 0.5).float()
                dice_score += dice_coeff(mask_pred_bin, mask_true, reduce_batch_first=False)
            else:
                mask_true_oh = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                pred_oh = F.one_hot(pred_classes, net.n_classes).permute(0, 3, 1, 2).float()
                dice_score += multiclass_dice_coeff(pred_oh[:, 1:], mask_true_oh[:, 1:], reduce_batch_first=False)

            # Confusion matrix
            total_cm += compute_confusion_matrix(pred_classes, mask_true, num_classes)

    net.train()

    # Compute metrics
    avg_dice = (dice_score / max(num_val_batches, 1)).item()
    per_class_iou = iou_from_confusion(total_cm).cpu().numpy().tolist()
    miou = float(np.mean(per_class_iou))
    miou_fg = float(np.mean(per_class_iou[1:])) if num_classes > 1 else miou

    return {
        "dice": avg_dice,
        "miou": miou,
        "miou_fg": miou_fg,
        "per_class_iou": per_class_iou,
    }
