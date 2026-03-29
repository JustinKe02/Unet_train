"""Train the official Pytorch-UNet on MoS2 segmentation data.

Usage:
    python train_mos2.py --epochs 100 --batch-size 3 --crop-size 896 --amp

Uses (almost) all official defaults: RMSprop, ReduceLROnPlateau, CE + Dice.
Only the data loading is changed to support the MoS2 split JSON.
"""

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from unet import UNet
from utils.mos2_dataset import MoS2Dataset, load_split, NUM_CLASSES, CLASS_NAMES
from utils.dice_score import dice_loss
from evaluate_mos2 import evaluate_mos2


def get_args():
    parser = argparse.ArgumentParser(description="Train UNet on MoS2 dataset")
    parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=3, help="Batch size")
    parser.add_argument("--learning-rate", "-l", type=float, default=1e-5, dest="lr",
                        help="Learning rate (official default: 1e-5)")
    parser.add_argument("--scale", "-s", type=float, default=1.0,
                        help="Global image scaling factor (1.0 = no scaling)")
    parser.add_argument("--crop-size", type=int, default=896,
                        help="Crop size for training and validation")
    parser.add_argument("--train-repeat", type=int, default=4,
                        help="Repeat training set N times per epoch")
    parser.add_argument("--amp", action="store_true", default=False,
                        help="Use mixed precision training")
    parser.add_argument("--bilinear", action="store_true", default=False,
                        help="Use bilinear upsampling (default: transposed conv)")
    parser.add_argument("--load", "-f", type=str, default=None,
                        help="Load model from a .pth file to resume training")
    parser.add_argument("--split-json", type=str,
                        default="../splits/mos2_main_only_split.json",
                        help="Path to split JSON file")
    parser.add_argument("--output-dir", type=str, default="./mos2_output",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--weight-decay", type=float, default=1e-8,
                        help="Weight decay (official default: 1e-8)")
    parser.add_argument("--momentum", type=float, default=0.999,
                        help="Momentum (official default: 0.999)")
    parser.add_argument("--gradient-clipping", type=float, default=1.0,
                        help="Gradient clipping max norm")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of data loader workers")
    return parser.parse_args()


def train_model(
    model,
    device,
    args,
    train_loader,
    val_loader,
    n_train,
):
    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # CSV log
    log_path = output_dir / "training_log.csv"
    log_fields = [
        "epoch", "train_loss", "val_dice", "val_miou", "val_miou_fg",
        *[f"iou_{name}" for name in CLASS_NAMES],
        "lr", "time_s",
    ]
    log_file = open(log_path, "w", newline="")
    csv_writer = csv.DictWriter(log_file, fieldnames=log_fields)
    csv_writer.writeheader()

    logging.info(f"""Starting training:
        Epochs:          {args.epochs}
        Batch size:      {args.batch_size}
        Learning rate:   {args.lr}
        Training size:   {n_train}
        Crop size:       {args.crop_size}
        Device:          {device.type}
        Mixed Precision: {args.amp}
        Output:          {output_dir}
    """)

    # Official defaults: RMSprop + ReduceLROnPlateau
    optimizer = optim.RMSprop(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        foreach=True,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=5)
    grad_scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    criterion = nn.CrossEntropyLoss()

    best_miou_fg = 0.0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        with tqdm(total=n_train, desc=f"Epoch {epoch}/{args.epochs}", unit="img") as pbar:
            for batch in train_loader:
                images = batch["image"].to(device=device, dtype=torch.float32,
                                           memory_format=torch.channels_last)
                true_masks = batch["mask"].to(device=device, dtype=torch.long)

                with torch.autocast(device.type, enabled=args.amp):
                    masks_pred = model(images)
                    loss = criterion(masks_pred, true_masks)
                    loss += dice_loss(
                        F.softmax(masks_pred, dim=1).float(),
                        F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                        multiclass=True,
                    )

                # NaN guard: skip entire optimizer step if loss is non-finite
                if not torch.isfinite(loss):
                    logging.warning(f"Non-finite loss at step {global_step + 1}, skipping batch")
                    pbar.update(images.shape[0])
                    global_step += 1
                    continue

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{"loss (batch)": f"{loss.item():.4f}"})
        avg_loss = epoch_loss / max(len(train_loader), 1)

        # Validation
        metrics = evaluate_mos2(model, val_loader, device, args.amp, num_classes=NUM_CLASSES)
        scheduler.step(metrics["dice"])

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - epoch_start

        # Log
        row = {
            "epoch": epoch,
            "train_loss": f"{avg_loss:.6f}",
            "val_dice": f"{metrics['dice']:.6f}",
            "val_miou": f"{metrics['miou']:.6f}",
            "val_miou_fg": f"{metrics['miou_fg']:.6f}",
            "lr": f"{current_lr:.2e}",
            "time_s": f"{elapsed:.1f}",
        }
        for i, name in enumerate(CLASS_NAMES):
            row[f"iou_{name}"] = f"{metrics['per_class_iou'][i]:.6f}"
        csv_writer.writerow(row)
        log_file.flush()

        logging.info(
            f"Epoch {epoch}: loss={avg_loss:.4f}  dice={metrics['dice']:.4f}  "
            f"mIoU={metrics['miou']:.4f}  mIoU_fg={metrics['miou_fg']:.4f}  "
            f"IoU=[{', '.join(f'{v:.3f}' for v in metrics['per_class_iou'])}]  "
            f"lr={current_lr:.2e}"
        )

        # Save best model
        if metrics["miou_fg"] > best_miou_fg:
            best_miou_fg = metrics["miou_fg"]
            state = model.state_dict()
            state["mask_values"] = list(range(NUM_CLASSES))
            torch.save(state, str(ckpt_dir / "best_model.pth"))
            logging.info(f"  ★ New best mIoU_fg = {best_miou_fg:.4f}, checkpoint saved!")

        # Save periodic checkpoint
        if epoch % 10 == 0 or epoch == args.epochs:
            state = model.state_dict()
            state["mask_values"] = list(range(NUM_CLASSES))
            torch.save(state, str(ckpt_dir / f"checkpoint_epoch{epoch}.pth"))

    log_file.close()
    logging.info(f"Training complete. Best mIoU_fg = {best_miou_fg:.4f}")
    logging.info(f"Logs saved to {log_path}")


if __name__ == "__main__":
    args = get_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data split
    splits = load_split(args.split_json)
    train_samples = splits["train"]
    val_samples = splits["val"]
    logging.info(f"Loaded split: {len(train_samples)} train, {len(val_samples)} val")

    # Create datasets
    train_dataset = MoS2Dataset(
        train_samples, crop_size=args.crop_size, mode="train",
        scale=args.scale, train_repeat=args.train_repeat,
    )
    val_dataset = MoS2Dataset(
        val_samples, crop_size=args.crop_size, mode="val",
        scale=args.scale,
    )

    # Create data loaders
    loader_args = dict(num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **loader_args)

    n_train = len(train_dataset)
    logging.info(f"Dataset sizes: train={n_train}, val={len(val_dataset)}")

    # Create model (official UNet)
    model = UNet(n_channels=3, n_classes=NUM_CLASSES, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f"Network:\n"
                 f"\t{model.n_channels} input channels\n"
                 f"\t{model.n_classes} output channels (classes)\n"
                 f"\t{'Bilinear' if model.bilinear else 'Transposed conv'} upscaling")

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        if "mask_values" in state_dict:
            del state_dict["mask_values"]
        model.load_state_dict(state_dict)
        logging.info(f"Model loaded from {args.load}")

    model.to(device=device)

    try:
        train_model(model, device, args, train_loader, val_loader, n_train)
    except torch.cuda.OutOfMemoryError:
        logging.error("OOM! Enabling checkpointing to reduce memory usage.")
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(model, device, args, train_loader, val_loader, n_train)
