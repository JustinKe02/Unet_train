from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

if not os.environ.get("OMP_NUM_THREADS", "").isdigit():
    os.environ["OMP_NUM_THREADS"] = "8"

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from unet_project.config import apply_overrides, load_config
from unet_project.data import MoS2SegmentationDataset, build_or_load_split, estimate_class_weights
from unet_project.engine import evaluate, train_one_epoch
from unet_project.losses import CombinedSegmentationLoss
from unet_project.model import UNet
from unet_project.utils import (
    configure_runtime,
    ensure_dir,
    format_metrics,
    load_checkpoint,
    save_checkpoint,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or evaluate U-Net on the MoS2 segmentation dataset.")
    parser.add_argument("--config", type=str, default="configs/mos2_unet.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--rebuild-split", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--train-repeat", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--top-k-checkpoints", type=int, default=None)
    return parser.parse_args()


def build_optional_tensor(value: Any) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return torch.tensor([float(item) for item in value], dtype=torch.float32)
    return torch.tensor(float(value), dtype=torch.float32)


def make_loader(dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def update_topk_checkpoints(
    topk_entries: list[dict[str, Any]],
    topk_dir: Path,
    top_k: int,
    metric_name: str,
    metric_value: float,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.amp.GradScaler | None,
    best_metric: float,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    if top_k <= 0:
        return topk_entries

    checkpoint_path = topk_dir / f"epoch{epoch:03d}_{metric_name}_{metric_value:.4f}.pth"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=epoch,
        best_metric=best_metric,
        config=config,
    )

    topk_entries = [entry for entry in topk_entries if Path(entry["path"]).exists()]
    topk_entries.append({"epoch": epoch, "metric": metric_value, "path": str(checkpoint_path)})
    topk_entries.sort(key=lambda item: (float(item["metric"]), int(item["epoch"])), reverse=True)

    while len(topk_entries) > top_k:
        removed = topk_entries.pop(-1)
        removed_path = Path(removed["path"])
        if removed_path.exists():
            removed_path.unlink()
    return topk_entries


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(
        cfg,
        {
            "output_dir": args.output_dir,
            "train.epochs": args.epochs,
            "train.batch_size": args.batch_size,
            "data.crop_size": args.crop_size,
            "train.num_workers": args.num_workers,
            "data.train_repeat": args.train_repeat,
            "train.early_stopping_patience": args.patience,
            "train.top_k_checkpoints": args.top_k_checkpoints,
        },
    )

    configure_runtime(int(cfg["train"]["num_workers"]))
    set_seed(int(cfg["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = ensure_dir(PROJECT_ROOT / cfg["output_dir"])
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    topk_dir = ensure_dir(checkpoint_dir / "topk")

    splits = build_or_load_split(cfg, rebuild=args.rebuild_split)
    print(
        "Split sizes:",
        {name: len(items) for name, items in splits.items()},
        flush=True,
    )

    crop_size = int(cfg["data"]["crop_size"])
    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["train"]["num_workers"])
    class_names = list(cfg["data"]["class_names"])
    num_classes = int(cfg["data"]["num_classes"])
    amp_enabled = bool(cfg["train"]["amp"]) and device.type == "cuda"

    train_dataset = MoS2SegmentationDataset(
        splits["train"],
        crop_size=crop_size,
        mode="train",
        train_repeat=int(cfg["data"]["train_repeat"]),
        seed=int(cfg["seed"]),
        hard_crop_probability=float(cfg["data"].get("hard_crop_probability", 0.0)),
        hard_crop_candidates=int(cfg["data"].get("hard_crop_candidates", 1)),
        min_foreground_ratio=float(cfg["data"].get("min_foreground_ratio", 0.0)),
        rare_classes=list(cfg["data"].get("rare_classes", [])),
        rare_class_weight=float(cfg["data"].get("rare_class_weight", 1.0)),
    )
    val_dataset = MoS2SegmentationDataset(splits["val"], crop_size=crop_size, mode="val", seed=int(cfg["seed"]))
    test_dataset = MoS2SegmentationDataset(splits["test"], crop_size=crop_size, mode="test", seed=int(cfg["seed"]))

    train_loader = make_loader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = make_loader(val_dataset, batch_size=1, num_workers=max(1, num_workers // 2), shuffle=False)
    test_loader = make_loader(test_dataset, batch_size=1, num_workers=max(1, num_workers // 2), shuffle=False)

    model = UNet(
        in_channels=int(cfg["model"]["in_channels"]),
        num_classes=num_classes,
        base_channels=int(cfg["model"]["base_channels"]),
        bilinear=bool(cfg["model"]["bilinear"]),
    ).to(device)

    class_weights = None
    if bool(cfg["train"]["use_class_weights"]):
        class_weights = estimate_class_weights(splits["train"], num_classes=num_classes).to(device)
        print("Class weights:", class_weights.tolist(), flush=True)

    focal_alpha = build_optional_tensor(cfg["train"].get("focal_alpha"))
    if focal_alpha is not None:
        focal_alpha = focal_alpha.to(device)
        print("Focal alpha:", focal_alpha.tolist(), flush=True)

    criterion = CombinedSegmentationLoss(
        num_classes=num_classes,
        ce_weight=float(cfg["train"]["ce_weight"]),
        focal_weight=float(cfg["train"].get("focal_weight", 0.0)),
        dice_weight=float(cfg["train"]["dice_weight"]),
        tversky_weight=float(cfg["train"].get("tversky_weight", 0.0)),
        focal_gamma=float(cfg["train"].get("focal_gamma", 2.0)),
        focal_alpha=focal_alpha,
        tversky_alpha=float(cfg["train"].get("tversky_alpha", 0.3)),
        tversky_beta=float(cfg["train"].get("tversky_beta", 0.7)),
        class_weights=class_weights,
    ).to(device)
    print(
        "Loss weights:",
        {
            "ce": float(cfg["train"]["ce_weight"]),
            "focal": float(cfg["train"].get("focal_weight", 0.0)),
            "dice": float(cfg["train"]["dice_weight"]),
            "tversky": float(cfg["train"].get("tversky_weight", 0.0)),
        },
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg["train"]["epochs"]),
        eta_min=float(cfg["train"]["min_lr"]),
    )
    scaler = torch.amp.GradScaler("cuda") if amp_enabled else None

    start_epoch = 1
    best_metric = float("-inf")

    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler, scaler, map_location=device)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_metric = float(checkpoint.get("best_metric", float("-inf")))
        print(f"Resumed from {args.resume} at epoch {start_epoch}", flush=True)

    if args.checkpoint and args.eval_only:
        checkpoint = load_checkpoint(args.checkpoint, model, map_location=device)
        best_metric = float(checkpoint.get("best_metric", float("-inf")))

    eval_loader = {"train": train_loader, "val": val_loader, "test": test_loader}[args.split]
    eval_desc = f"Evaluate-{args.split}"

    if args.eval_only:
        metrics = evaluate(
            model=model,
            loader=eval_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            class_names=class_names,
            crop_size=crop_size,
            stride=int(cfg["data"]["eval_stride"]),
            amp=amp_enabled,
            desc=eval_desc,
        )
        print(f"{args.split} metrics: {format_metrics(metrics)}", flush=True)
        print(metrics["per_class_iou"], flush=True)
        return

    history_path = output_dir / "history.json"
    history: list[dict[str, float | int]] = []
    if args.resume and history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
    epochs = int(cfg["train"]["epochs"])
    save_metric = str(cfg["train"]["save_metric"])
    latest_path = checkpoint_dir / "latest.pth"
    best_path = checkpoint_dir / "best.pth"
    top_k_checkpoints = int(cfg["train"].get("top_k_checkpoints", 0))
    early_stopping_patience = int(cfg["train"].get("early_stopping_patience", 0))
    early_stopping_warmup = int(cfg["train"].get("early_stopping_warmup", 0))
    early_stopping_min_delta = float(cfg["train"].get("early_stopping_min_delta", 0.0))
    epochs_without_improvement = 0
    topk_entries: list[dict[str, Any]] = []

    for epoch in range(start_epoch, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            epoch=epoch,
            epochs=epochs,
            amp=amp_enabled,
            max_grad_norm=float(cfg["train"]["max_grad_norm"]),
            log_interval=int(cfg["train"]["log_interval"]),
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            class_names=class_names,
            crop_size=crop_size,
            stride=int(cfg["data"]["eval_stride"]),
            amp=amp_enabled,
            desc=f"Val {epoch}/{epochs}",
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_pixel_acc": val_metrics["pixel_acc"],
            "val_miou_all": val_metrics["miou_all"],
            "val_miou_fg": val_metrics["miou_fg"],
            "val_mdice_all": val_metrics["mdice_all"],
            "val_mdice_fg": val_metrics["mdice_fg"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | train_loss={train_metrics['loss']:.4f} | {format_metrics(val_metrics)}",
            flush=True,
        )

        current_metric = float(val_metrics[save_metric])
        previous_best_metric = best_metric
        is_new_best = current_metric > best_metric
        meaningful_improvement = current_metric > previous_best_metric + early_stopping_min_delta
        effective_best_metric = max(best_metric, current_metric)
        save_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_metric=effective_best_metric,
            config=cfg,
        )
        topk_entries = update_topk_checkpoints(
            topk_entries=topk_entries,
            topk_dir=topk_dir,
            top_k=top_k_checkpoints,
            metric_name=save_metric,
            metric_value=current_metric,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_metric=effective_best_metric,
            config=cfg,
        )
        save_json(output_dir / "topk_checkpoints.json", topk_entries)

        if is_new_best:
            best_metric = current_metric
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_metric=best_metric,
                config=cfg,
            )
            print(f"New best checkpoint saved at epoch {epoch} ({save_metric}={best_metric:.4f})", flush=True)

        if epoch >= early_stopping_warmup and early_stopping_patience > 0:
            if meaningful_improvement:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                print(
                    f"Early stopping counter: {epochs_without_improvement}/{early_stopping_patience}",
                    flush=True,
                )

        save_json(history_path, history)

        if early_stopping_patience > 0 and epoch >= early_stopping_warmup:
            if epochs_without_improvement >= early_stopping_patience:
                print(
                    f"Early stopping triggered at epoch {epoch} after {epochs_without_improvement} epochs without meaningful improvement.",
                    flush=True,
                )
                break

    best_checkpoint = load_checkpoint(best_path, model, map_location=device)
    print(
        f"Loaded best checkpoint from epoch {best_checkpoint.get('epoch')} with {save_metric}={best_checkpoint.get('best_metric'):.4f}",
        flush=True,
    )

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        num_classes=num_classes,
        class_names=class_names,
        crop_size=crop_size,
        stride=int(cfg["data"]["eval_stride"]),
        amp=amp_enabled,
        desc="Test",
    )
    save_json(output_dir / "test_metrics.json", test_metrics)
    print(f"Test metrics: {format_metrics(test_metrics)}", flush=True)
    print(f"Per-class IoU: {test_metrics['per_class_iou']}", flush=True)


if __name__ == "__main__":
    main()
