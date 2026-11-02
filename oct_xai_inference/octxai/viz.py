"""Saliency overlay figures (no pyplot, so it is safe in headless / sbatch runs)."""

import numpy as np
from matplotlib import colormaps
from PIL import Image


def heatmap_rgb(cam: np.ndarray) -> np.ndarray:
    """(H, W) float in [0, 1] -> (H, W, 3) uint8 with the jet colormap."""
    return (colormaps["jet"](np.clip(cam, 0.0, 1.0))[..., :3] * 255).astype(np.uint8)


def overlay_panel(image: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Side by side: the network input (H, W, 3 uint8) and the same image blended with the saliency map."""
    blend = ((1 - alpha) * image.astype(np.float32) + alpha * heatmap_rgb(cam).astype(np.float32)).astype(np.uint8)
    return Image.fromarray(np.concatenate([image, blend], axis=1))
