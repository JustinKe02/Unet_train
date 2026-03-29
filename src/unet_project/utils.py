from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def save_json(path: str | Path, payload: dict[str, Any] | list[Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def strip_dataparallel_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    best_metric: float,
    config: dict[str, Any],
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "config": config,
    }
    torch.save(payload, path)


def load_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(strip_dataparallel_prefix(checkpoint["model"]))
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


def format_metrics(metrics: dict[str, Any]) -> str:
    keys = ["loss", "pixel_acc", "miou_all", "miou_fg", "mdice_all", "mdice_fg"]
    parts = []
    for key in keys:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, (float, int)):
                parts.append(f"{key}={value:.4f}")
    return " | ".join(parts)


def list_images(input_path: str | Path, image_extensions: list[str]) -> list[Path]:
    path = Path(input_path).resolve()
    allowed = {ext.lower() for ext in image_extensions}
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input path not found: {path}")
    return sorted(
        image_path
        for image_path in path.iterdir()
        if image_path.is_file() and image_path.suffix.lower() in allowed
    )


def make_color_mask(mask: np.ndarray, palette: list[list[int]]) -> Image.Image:
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_index, color in enumerate(palette):
        color_mask[mask == class_index] = np.asarray(color, dtype=np.uint8)
    return Image.fromarray(color_mask)


def make_overlay(image: np.ndarray, color_mask: np.ndarray, alpha: float = 0.45) -> Image.Image:
    overlay = (image.astype(np.float32) * (1.0 - alpha) + color_mask.astype(np.float32) * alpha).clip(0, 255)
    return Image.fromarray(overlay.astype(np.uint8))


def configure_runtime(num_workers: int) -> None:
    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = str(max(1, min(8, num_workers)))
