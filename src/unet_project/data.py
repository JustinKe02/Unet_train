from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class Sample:
    sample_id: str
    source: str
    image_path: str
    mask_path: str


def _resolve_path(path: str | Path, project_root: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (Path(project_root) / candidate).resolve()


def _is_image_file(path: Path, allowed_exts: set[str]) -> bool:
    return path.is_file() and path.suffix.lower() in allowed_exts


def scan_dataset(roots: list[dict[str, str]], image_extensions: list[str], project_root: str | Path) -> list[Sample]:
    allowed_exts = {ext.lower() for ext in image_extensions}
    all_samples: list[Sample] = []

    for root_cfg in roots:
        source_name = root_cfg["name"]
        image_dir = _resolve_path(root_cfg["image_dir"], project_root)
        mask_dir = _resolve_path(root_cfg["mask_dir"], project_root)
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

        source_samples: list[Sample] = []
        for image_path in sorted(image_dir.iterdir()):
            if not _is_image_file(image_path, allowed_exts):
                continue
            mask_path = mask_dir / f"{image_path.stem}.png"
            if not mask_path.exists():
                continue
            source_samples.append(
                Sample(
                    sample_id=image_path.stem,
                    source=source_name,
                    image_path=str(image_path.resolve()),
                    mask_path=str(mask_path.resolve()),
                )
            )
        if not source_samples:
            raise RuntimeError(f"No valid image-mask pairs found under {image_dir}")
        all_samples.extend(source_samples)

    if not all_samples:
        raise RuntimeError("No valid samples found for the dataset.")
    return all_samples


def _split_one_source(samples: list[Sample], train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[Sample]]:
    if not samples:
        return {"train": [], "val": [], "test": []}
    shuffled = samples[:]
    random.Random(seed).shuffle(shuffled)

    n_total = len(shuffled)
    n_train = max(1, int(math.floor(n_total * train_ratio)))
    n_val = max(1, int(math.floor(n_total * val_ratio)))
    if n_train + n_val >= n_total:
        n_val = max(1, n_total - n_train - 1)
    n_test = n_total - n_train - n_val
    if n_test <= 0:
        n_test = 1
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def build_or_load_split(cfg: dict[str, Any], rebuild: bool = False) -> dict[str, list[Sample]]:
    project_root = cfg["project_root"]
    split_path = _resolve_path(cfg["data"]["split_path"], project_root)
    if split_path.exists() and not rebuild:
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        return {
            split_name: [Sample(**entry) for entry in entries]
            for split_name, entries in payload["splits"].items()
        }

    split_path.parent.mkdir(parents=True, exist_ok=True)
    samples = scan_dataset(cfg["data"]["roots"], cfg["data"]["image_extensions"], project_root)

    by_source: dict[str, list[Sample]] = {}
    for sample in samples:
        by_source.setdefault(sample.source, []).append(sample)

    ratios = cfg["data"]["split_ratios"]
    seed = int(cfg["seed"])
    merged = {"train": [], "val": [], "test": []}
    for index, (source_name, source_samples) in enumerate(sorted(by_source.items())):
        split = _split_one_source(
            source_samples,
            train_ratio=float(ratios["train"]),
            val_ratio=float(ratios["val"]),
            seed=seed + index,
        )
        for key in merged:
            merged[key].extend(split[key])

    for key in merged:
        merged[key].sort(key=lambda item: (item.source, item.sample_id))

    payload = {
        "seed": seed,
        "class_names": cfg["data"]["class_names"],
        "splits": {key: [asdict(sample) for sample in value] for key, value in merged.items()},
    }
    split_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return merged


def _load_image(image_path: str) -> np.ndarray:
    return np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8, copy=True)


def _load_mask(mask_path: str) -> np.ndarray:
    return np.array(Image.open(mask_path), dtype=np.int64, copy=True)


def _pad_if_needed(image: np.ndarray, mask: np.ndarray, crop_size: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    pad_h = max(0, crop_size - height)
    pad_w = max(0, crop_size - width)
    if pad_h == 0 and pad_w == 0:
        return image, mask

    image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant")
    return image, mask


def _sample_crop_coords(height: int, width: int, crop_size: int, rng: random.Random) -> tuple[int, int]:
    top = 0 if height == crop_size else rng.randint(0, height - crop_size)
    left = 0 if width == crop_size else rng.randint(0, width - crop_size)
    return top, left


def _random_crop(image: np.ndarray, mask: np.ndarray, crop_size: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    image, mask = _pad_if_needed(image, mask, crop_size)
    height, width = image.shape[:2]
    top, left = _sample_crop_coords(height, width, crop_size, rng)
    return (
        image[top : top + crop_size, left : left + crop_size],
        mask[top : top + crop_size, left : left + crop_size],
    )


def _foreground_aware_crop(
    image: np.ndarray,
    mask: np.ndarray,
    crop_size: int,
    rng: random.Random,
    hard_crop_probability: float,
    hard_crop_candidates: int,
    min_foreground_ratio: float,
    rare_classes: set[int],
    rare_class_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    image, mask = _pad_if_needed(image, mask, crop_size)
    height, width = image.shape[:2]

    if hard_crop_candidates <= 1 or rng.random() > hard_crop_probability:
        return _random_crop(image, mask, crop_size, rng)

    total_pixels = float(crop_size * crop_size)
    best_score = float("-inf")
    best_coords: tuple[int, int] | None = None

    for _ in range(hard_crop_candidates):
        top, left = _sample_crop_coords(height, width, crop_size, rng)
        mask_crop = mask[top : top + crop_size, left : left + crop_size]
        foreground_ratio = float(np.count_nonzero(mask_crop)) / total_pixels
        if rare_classes:
            rare_ratio = float(np.isin(mask_crop, list(rare_classes)).sum()) / total_pixels
        else:
            rare_ratio = 0.0

        score = foreground_ratio + rare_class_weight * rare_ratio
        if foreground_ratio >= min_foreground_ratio:
            score += 0.5
        if rare_ratio > 0:
            score += 1.0

        if score > best_score:
            best_score = score
            best_coords = (top, left)

    if best_coords is None or best_score <= 0:
        return _random_crop(image, mask, crop_size, rng)

    top, left = best_coords
    return (
        image[top : top + crop_size, left : left + crop_size],
        mask[top : top + crop_size, left : left + crop_size],
    )


def _apply_train_augmentations(image: np.ndarray, mask: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() < 0.5:
        image = np.flip(image, axis=1).copy()
        mask = np.flip(mask, axis=1).copy()
    if rng.random() < 0.5:
        image = np.flip(image, axis=0).copy()
        mask = np.flip(mask, axis=0).copy()
    k = rng.randint(0, 3)
    if k:
        image = np.rot90(image, k=k, axes=(0, 1)).copy()
        mask = np.rot90(mask, k=k, axes=(0, 1)).copy()
    return image, mask


def _to_tensor(image: np.ndarray, mask: np.ndarray | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
    image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().div(255.0)
    mask_tensor = None if mask is None else torch.from_numpy(mask).long()
    return image_tensor, mask_tensor


class MoS2SegmentationDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        crop_size: int,
        mode: str,
        train_repeat: int = 1,
        seed: int = 42,
        hard_crop_probability: float = 0.0,
        hard_crop_candidates: int = 1,
        min_foreground_ratio: float = 0.0,
        rare_classes: list[int] | None = None,
        rare_class_weight: float = 1.0,
    ) -> None:
        self.samples = samples
        self.crop_size = int(crop_size)
        self.mode = mode
        self.train_repeat = max(1, int(train_repeat))
        self.seed = int(seed)
        self.hard_crop_probability = float(hard_crop_probability)
        self.hard_crop_candidates = max(1, int(hard_crop_candidates))
        self.min_foreground_ratio = float(min_foreground_ratio)
        self.rare_classes = set(rare_classes or [])
        self.rare_class_weight = float(rare_class_weight)

    def __len__(self) -> int:
        if self.mode == "train":
            return len(self.samples) * self.train_repeat
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index % len(self.samples)]
        image = _load_image(sample.image_path)
        mask = _load_mask(sample.mask_path)
        original_size = mask.shape

        if self.mode == "train":
            rng = random.Random(self.seed + index)
            image, mask = _foreground_aware_crop(
                image=image,
                mask=mask,
                crop_size=self.crop_size,
                rng=rng,
                hard_crop_probability=self.hard_crop_probability,
                hard_crop_candidates=self.hard_crop_candidates,
                min_foreground_ratio=self.min_foreground_ratio,
                rare_classes=self.rare_classes,
                rare_class_weight=self.rare_class_weight,
            )
            image, mask = _apply_train_augmentations(image, mask, rng)
        else:
            image, mask = _pad_if_needed(image, mask, self.crop_size)

        image_tensor, mask_tensor = _to_tensor(image, mask)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "sample_id": sample.sample_id,
            "source": sample.source,
            "image_path": sample.image_path,
            "mask_path": sample.mask_path,
            "original_size": original_size,
        }


class InferenceDataset(Dataset):
    def __init__(self, image_paths: Iterable[str | Path], crop_size: int) -> None:
        self.image_paths = [str(Path(path).resolve()) for path in image_paths]
        self.crop_size = int(crop_size)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path = self.image_paths[index]
        image = _load_image(image_path)
        height, width = image.shape[:2]
        image, _ = _pad_if_needed(image, np.zeros((height, width), dtype=np.int64), self.crop_size)
        image_tensor, _ = _to_tensor(image, None)
        return {
            "image": image_tensor,
            "image_path": image_path,
            "sample_id": Path(image_path).stem,
            "original_size": (height, width),
        }


def estimate_class_weights(samples: list[Sample], num_classes: int) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.float64)
    for sample in samples:
        mask = _load_mask(sample.mask_path)
        hist = np.bincount(mask.reshape(-1), minlength=num_classes)
        counts += hist
    freqs = counts / max(counts.sum(), 1.0)
    weights = 1.0 / np.sqrt(np.clip(freqs, 1e-8, None))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def crop_to_original(logits: torch.Tensor, original_size: tuple[int, int]) -> torch.Tensor:
    height, width = original_size
    return logits[..., :height, :width]


def pad_to_window(image: torch.Tensor, window_size: int) -> torch.Tensor:
    _, _, height, width = image.shape
    pad_h = max(0, window_size - height)
    pad_w = max(0, window_size - width)
    if pad_h == 0 and pad_w == 0:
        return image
    return F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
