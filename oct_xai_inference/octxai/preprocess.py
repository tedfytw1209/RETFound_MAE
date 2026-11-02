"""Image loading and the evaluation-time transform used by the training / XAI-evaluation code.

Mirrors util/datasets.py `build_transform` for the eval split (no augmentation):
  input_size <= 224 -> Resize(int(input_size / (224/256))), CenterCrop(input_size)   (RETFound, 224)
  input_size  > 224 -> Resize(input_size),                  CenterCrop(input_size)   (SMP, 512)
then ToTensor and ImageNet mean/std normalisation. Resize(int) scales the shorter side, so a wide
B-scan is centre-cropped along its width.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def build_transform(input_size: int) -> transforms.Compose:
    crop_pct = 224 / 256 if input_size <= 224 else 1.0
    resize_to = int(input_size / crop_pct)
    return transforms.Compose([
        transforms.Resize(resize_to, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_image(path) -> Image.Image:
    """Open an image as 3-channel RGB (grayscale B-scans are replicated), like torchvision's default_loader."""
    with open(Path(path), "rb") as f:
        return Image.open(f).convert("RGB")


def denormalize(x: torch.Tensor) -> np.ndarray:
    """(3, H, W) normalised tensor -> (H, W, 3) uint8 image, i.e. what the network actually saw."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=x.dtype, device=x.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=x.dtype, device=x.device).view(3, 1, 1)
    img = (x * std + mean).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (img * 255.0 + 0.5).astype(np.uint8)
