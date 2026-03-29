"""MoS2 segmentation dataset for the official Pytorch-UNet.

Reads the split JSON produced by the custom UNet pipeline to ensure
train / val / test partitions are identical across experiments.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset


CLASS_NAMES = ["background", "monolayer", "fewlayer", "multilayer"]
NUM_CLASSES = len(CLASS_NAMES)


def load_split(split_path: str | Path):
    """Return dict with 'train', 'val', 'test' lists of sample dicts."""
    with open(split_path, "r") as f:
        data = json.load(f)
    return data["splits"]


class MoS2Dataset(Dataset):
    """Dataset that loads MoS2 images + class-index masks.

    Compatible with the official Pytorch-UNet training loop
    (returns ``{'image': Tensor, 'mask': Tensor}``).
    """

    def __init__(
        self,
        samples: list[dict],
        crop_size: int = 896,
        mode: str = "train",
        scale: float = 1.0,
        train_repeat: int = 4,
    ):
        assert mode in ("train", "val", "test")
        self.samples = samples
        self.crop_size = crop_size
        self.mode = mode
        self.scale = scale
        self.train_repeat = train_repeat if mode == "train" else 1

        # For compatibility with official code that reads dataset.mask_values
        self.mask_values = list(range(NUM_CLASSES))

    def __len__(self):
        return len(self.samples) * self.train_repeat

    def __getitem__(self, idx):
        sample = self.samples[idx % len(self.samples)]
        image = Image.open(sample["image_path"]).convert("RGB")
        mask = Image.open(sample["mask_path"]).convert("L")

        # Optional global scaling (official UNet default is 0.5)
        if self.scale != 1.0:
            w, h = image.size
            new_w, new_h = int(w * self.scale), int(h * self.scale)
            image = image.resize((new_w, new_h), Image.BICUBIC)
            mask = mask.resize((new_w, new_h), Image.NEAREST)

        if self.mode == "train":
            image, mask = self._random_crop(image, mask)
            image, mask = self._augment(image, mask)
        else:
            image, mask = self._center_crop(image, mask)

        # To tensor
        img_np = np.asarray(image).transpose((2, 0, 1)).astype(np.float32) / 255.0
        mask_np = np.asarray(mask).astype(np.int64)

        return {
            "image": torch.from_numpy(img_np).contiguous(),
            "mask": torch.from_numpy(mask_np).contiguous(),
        }

    # ------------------------------------------------------------------
    # Crop helpers
    # ------------------------------------------------------------------

    def _pad_if_needed(self, image: Image.Image, mask: Image.Image):
        """Pad image and mask so they are at least crop_size in each dim."""
        w, h = image.size
        pad_w = max(0, self.crop_size - w)
        pad_h = max(0, self.crop_size - h)
        if pad_w > 0 or pad_h > 0:
            image = np.array(image)
            mask = np.array(mask)
            image = np.pad(
                image,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode="reflect",
            )
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="reflect")
            image = Image.fromarray(image)
            mask = Image.fromarray(mask)
        return image, mask

    def _random_crop(self, image: Image.Image, mask: Image.Image):
        image, mask = self._pad_if_needed(image, mask)
        w, h = image.size
        x = random.randint(0, w - self.crop_size)
        y = random.randint(0, h - self.crop_size)
        image = image.crop((x, y, x + self.crop_size, y + self.crop_size))
        mask = mask.crop((x, y, x + self.crop_size, y + self.crop_size))
        return image, mask

    def _center_crop(self, image: Image.Image, mask: Image.Image):
        image, mask = self._pad_if_needed(image, mask)
        w, h = image.size
        x = (w - self.crop_size) // 2
        y = (h - self.crop_size) // 2
        image = image.crop((x, y, x + self.crop_size, y + self.crop_size))
        mask = mask.crop((x, y, x + self.crop_size, y + self.crop_size))
        return image, mask

    # ------------------------------------------------------------------
    # Data augmentation (train only)
    # ------------------------------------------------------------------

    @staticmethod
    def _augment(image: Image.Image, mask: Image.Image):
        # Horizontal flip
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        # Vertical flip
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        # Random rotation (0, 90, 180, 270)
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            image = image.rotate(k * 90, expand=False)
            mask = mask.rotate(k * 90, expand=False)
        # Brightness / contrast jitter (image only)
        if random.random() > 0.5:
            from torchvision.transforms import functional as TF

            image = TF.adjust_brightness(image, 0.8 + random.random() * 0.4)
            image = TF.adjust_contrast(image, 0.8 + random.random() * 0.4)
        # Gaussian blur
        if random.random() > 0.7:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        return image, mask
