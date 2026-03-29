from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import crop_to_original, pad_to_window
from .metrics import SegmentationMeter


def _autocast_context(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=torch.float16)


def sliding_window_inference(
    model: torch.nn.Module,
    image: torch.Tensor,
    crop_size: int,
    stride: int,
    num_classes: int,
) -> torch.Tensor:
    stride = min(stride, crop_size)
    image = pad_to_window(image, crop_size)
    _, _, height, width = image.shape
    ys = list(range(0, max(height - crop_size, 0) + 1, stride))
    xs = list(range(0, max(width - crop_size, 0) + 1, stride))
    if ys[-1] != height - crop_size:
        ys.append(height - crop_size)
    if xs[-1] != width - crop_size:
        xs.append(width - crop_size)

    logits_sum = torch.zeros((1, num_classes, height, width), device=image.device)
    count_map = torch.zeros((1, 1, height, width), device=image.device)

    for top in ys:
        for left in xs:
            patch = image[:, :, top : top + crop_size, left : left + crop_size]
            patch_logits = model(patch)
            logits_sum[:, :, top : top + crop_size, left : left + crop_size] += patch_logits
            count_map[:, :, top : top + crop_size, left : left + crop_size] += 1.0

    return logits_sum / count_map.clamp_min(1.0)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    epoch: int,
    epochs: int,
    amp: bool,
    max_grad_norm: float | None,
    log_interval: int,
) -> dict[str, float]:
    model.train()
    running_loss = 0.0
    total_steps = len(loader)
    progress = tqdm(loader, desc=f"Train {epoch}/{epochs}", leave=False)

    for step, batch in enumerate(progress, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, amp):
            logits = model(images)
            loss = criterion(logits, masks)

        if scaler is not None:
            scaler.scale(loss).backward()
            if max_grad_norm is not None and max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        running_loss += float(loss.item())
        if step % max(1, log_interval) == 0 or step == total_steps:
            progress.set_postfix(loss=f"{running_loss / step:.4f}")

    return {"loss": running_loss / max(total_steps, 1)}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int,
    class_names: list[str],
    crop_size: int,
    stride: int,
    amp: bool,
    desc: str = "Eval",
) -> dict[str, Any]:
    model.eval()
    meter = SegmentationMeter(num_classes=num_classes, class_names=class_names)
    losses: list[float] = []

    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        original_size = tuple(int(v) for v in batch["original_size"])

        with _autocast_context(device, amp):
            logits = sliding_window_inference(
                model=model,
                image=images,
                crop_size=crop_size,
                stride=stride,
                num_classes=num_classes,
            )
        logits = crop_to_original(logits, original_size)
        masks = masks[:, : original_size[0], : original_size[1]]
        loss = criterion(logits, masks)
        preds = torch.argmax(logits, dim=1)
        meter.update(preds, masks)
        losses.append(float(loss.item()))

    metrics = meter.compute()
    metrics["loss"] = float(sum(losses) / max(len(losses), 1))
    return metrics


@torch.no_grad()
def predict_logits(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    crop_size: int,
    stride: int,
    num_classes: int,
    amp: bool,
    original_size: tuple[int, int],
) -> torch.Tensor:
    model.eval()
    image = image.to(device, non_blocking=True)
    with _autocast_context(device, amp):
        logits = sliding_window_inference(
            model=model,
            image=image,
            crop_size=crop_size,
            stride=stride,
            num_classes=num_classes,
        )
    return crop_to_original(logits, original_size).cpu()
