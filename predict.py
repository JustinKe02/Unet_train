from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if not os.environ.get("OMP_NUM_THREADS", "").isdigit():
    os.environ["OMP_NUM_THREADS"] = "8"

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from unet_project.config import load_config
from unet_project.data import InferenceDataset
from unet_project.engine import predict_logits
from unet_project.model import UNet
from unet_project.utils import (
    configure_runtime,
    ensure_dir,
    list_images,
    load_checkpoint,
    make_color_mask,
    make_overlay,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run U-Net inference on one image or a directory.")
    parser.add_argument("--config", type=str, default="configs/mos2_unet.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/predict")
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    configure_runtime(args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crop_size = int(cfg["data"]["crop_size"])
    stride = int(cfg["data"]["eval_stride"])
    num_classes = int(cfg["data"]["num_classes"])
    image_paths = list_images(args.input, cfg["data"]["image_extensions"])
    output_dir = ensure_dir(PROJECT_ROOT / args.output_dir)

    dataset = InferenceDataset(image_paths=image_paths, crop_size=crop_size)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    model = UNet(
        in_channels=int(cfg["model"]["in_channels"]),
        num_classes=num_classes,
        base_channels=int(cfg["model"]["base_channels"]),
        bilinear=bool(cfg["model"]["bilinear"]),
    ).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    palette = list(cfg["data"]["palette"])
    amp_enabled = bool(cfg["train"]["amp"]) and device.type == "cuda"

    for batch in loader:
        image_tensor = batch["image"]
        original_size = tuple(int(v) for v in batch["original_size"])
        image_path = Path(batch["image_path"][0])
        sample_id = batch["sample_id"][0]

        logits = predict_logits(
            model=model,
            image=image_tensor,
            device=device,
            crop_size=crop_size,
            stride=stride,
            num_classes=num_classes,
            amp=amp_enabled,
            original_size=original_size,
        )
        pred_mask = torch.argmax(logits, dim=1).squeeze(0).numpy().astype(np.uint8)

        original_image = np.asarray(Image.open(image_path).convert("RGB"))
        color_mask_img = make_color_mask(pred_mask, palette)
        color_mask = np.asarray(color_mask_img)
        overlay = make_overlay(original_image, color_mask)

        mask_path = output_dir / f"{sample_id}_mask.png"
        color_path = output_dir / f"{sample_id}_color.png"
        overlay_path = output_dir / f"{sample_id}_overlay.png"

        Image.fromarray(pred_mask, mode="L").save(mask_path)
        color_mask_img.save(color_path)
        overlay.save(overlay_path)

        print(f"Saved predictions for {image_path.name} -> {mask_path}", flush=True)


if __name__ == "__main__":
    main()
