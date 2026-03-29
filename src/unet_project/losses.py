from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        smooth: float = 1.0,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_one_hot, dims)
        cardinality = torch.sum(probs + target_one_hot, dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        if self.class_weights is not None:
            weights = self.class_weights / self.class_weights.sum().clamp_min(1e-8)
            return 1.0 - torch.sum(dice * weights)
        return 1.0 - dice.mean()


class TverskyLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1.0,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        tp = torch.sum(probs * target_one_hot, dims)
        fp = torch.sum(probs * (1.0 - target_one_hot), dims)
        fn = torch.sum((1.0 - probs) * target_one_hot, dims)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        if self.class_weights is not None:
            weights = self.class_weights / self.class_weights.sum().clamp_min(1e-8)
            return 1.0 - torch.sum(tversky * weights)
        return 1.0 - tversky.mean()


class FocalCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha if alpha is not None else None)
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        target_indices = target.unsqueeze(1)

        target_log_probs = log_probs.gather(1, target_indices).squeeze(1)
        target_probs = probs.gather(1, target_indices).squeeze(1)
        loss = -target_log_probs

        if self.class_weights is not None:
            pixel_weights = self.class_weights.gather(0, target.reshape(-1)).reshape_as(target)
            loss = loss * pixel_weights

        if self.alpha is not None:
            alpha_weights = self.alpha.gather(0, target.reshape(-1)).reshape_as(target)
            loss = loss * alpha_weights

        focal_factor = (1.0 - target_probs).clamp_min(1e-6).pow(self.gamma)
        return (focal_factor * loss).mean()


class CombinedSegmentationLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        ce_weight: float = 0.5,
        focal_weight: float = 0.0,
        dice_weight: float = 0.5,
        tversky_weight: float = 0.0,
        focal_gamma: float = 2.0,
        focal_alpha: torch.Tensor | None = None,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight
        self.dice = DiceLoss(num_classes=num_classes, class_weights=class_weights)
        self.tversky = TverskyLoss(
            num_classes=num_classes,
            alpha=tversky_alpha,
            beta=tversky_beta,
            class_weights=class_weights,
        )
        self.focal = FocalCrossEntropyLoss(
            gamma=focal_gamma,
            alpha=focal_alpha,
            class_weights=class_weights,
        )
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
        if self.ce_weight > 0:
            loss = loss + self.ce_weight * F.cross_entropy(logits, target, weight=self.class_weights)
        if self.focal_weight > 0:
            loss = loss + self.focal_weight * self.focal(logits, target)
        if self.dice_weight > 0:
            loss = loss + self.dice_weight * self.dice(logits, target)
        if self.tversky_weight > 0:
            loss = loss + self.tversky_weight * self.tversky(logits, target)
        return loss
