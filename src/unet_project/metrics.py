from __future__ import annotations

import torch


class SegmentationMeter:
    def __init__(self, num_classes: int, class_names: list[str]) -> None:
        self.num_classes = num_classes
        self.class_names = class_names
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.float64)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds = preds.reshape(-1).to(torch.int64).cpu()
        target = target.reshape(-1).to(torch.int64).cpu()
        valid = (target >= 0) & (target < self.num_classes)
        indices = self.num_classes * target[valid] + preds[valid]
        hist = torch.bincount(indices, minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        self.confusion += hist

    def compute(self) -> dict[str, float | dict[str, float]]:
        tp = self.confusion.diag()
        fp = self.confusion.sum(0) - tp
        fn = self.confusion.sum(1) - tp
        denom_iou = tp + fp + fn
        denom_dice = 2 * tp + fp + fn

        iou = torch.where(denom_iou > 0, tp / denom_iou, torch.zeros_like(tp))
        dice = torch.where(denom_dice > 0, (2 * tp) / denom_dice, torch.zeros_like(tp))
        pixel_acc = float(tp.sum() / self.confusion.sum().clamp_min(1.0))

        metrics: dict[str, float | dict[str, float]] = {
            "pixel_acc": pixel_acc,
            "miou_all": float(iou.mean().item()),
            "mdice_all": float(dice.mean().item()),
        }

        if self.num_classes > 1:
            metrics["miou_fg"] = float(iou[1:].mean().item())
            metrics["mdice_fg"] = float(dice[1:].mean().item())
        else:
            metrics["miou_fg"] = metrics["miou_all"]
            metrics["mdice_fg"] = metrics["mdice_all"]

        metrics["per_class_iou"] = {
            name: float(value.item()) for name, value in zip(self.class_names, iou, strict=False)
        }
        metrics["per_class_dice"] = {
            name: float(value.item()) for name, value in zip(self.class_names, dice, strict=False)
        }
        return metrics
