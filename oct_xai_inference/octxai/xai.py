"""Saliency maps with the CAM methods of the XAI evaluation (hirescam, gradcamv2, gradcam++).

Follows baselines/GradCAM_v2.py (PytorchCAM): pytorch-grad-cam hooks one target layer,
  - RETFound: model.blocks[select_index]; the token sequence is reshaped to a (C, n, n) grid.
  - SMP:      the select_index-th Conv2d inside the encoder / decoder / head module
              (default: the last Conv2d of the head, i.e. the 1x1 classifier of the fusion head).
The map is min-max normalised per image to [0, 1] and returned at the network input resolution.
"""

from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

METHODS = ("hirescam", "gradcamv2", "gradcam++")
PATCH_SIZE = 16  # RETFound ViT-L/16


def resolve_target_layer(net: nn.Module, arch: str, target_module: str, select_index: int) -> nn.Module:
    if arch == "retfound":
        return net.blocks[select_index]
    modules = {"encoder": net.seg_model.encoder, "decoder": net.seg_model.decoder, "head": net.head}
    if target_module not in modules:
        raise ValueError(f"SMP target_module must be one of {list(modules)}, got {target_module!r}")
    convs = [(n, m) for n, m in modules[target_module].named_modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise ValueError(f"no Conv2d inside the SMP {target_module}; pick another --xai_module "
                         f"(a linear classifier head has no conv layer to hook)")
    return convs[select_index][1]


class CamExplainer:
    def __init__(self, loaded, method: str = None, target_module: str = None, select_index: int = None):
        try:
            from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, HiResCAM
        except ImportError as e:
            raise ImportError("saliency maps need grad-cam: pip install grad-cam==1.5.5 "
                              "(or run with --no_saliency for probabilities only)") from e
        cfg = loaded.card["xai"]
        self.method = method or cfg["method"]
        if self.method not in METHODS:
            raise ValueError(f"unknown XAI method {self.method!r}; choose from {METHODS}")
        cls = {"hirescam": HiResCAM, "gradcamv2": GradCAM, "gradcam++": GradCAMPlusPlus}[self.method]
        self.input_size = loaded.input_size
        self.layer = resolve_target_layer(
            loaded.net, loaded.arch, target_module or cfg["target_module"],
            cfg["select_index"] if select_index is None else select_index)
        reshape = None
        if loaded.arch == "retfound":
            n = self.input_size // PATCH_SIZE
            reshape = lambda x: x[:, 1:, :].reshape(x.shape[0], n, n, x.shape[2]).transpose(2, 3).transpose(1, 2)  # noqa: E731
        self.cam = cls(model=loaded.net, target_layers=[self.layer], reshape_transform=reshape)

    def explain(self, batch: torch.Tensor, classes: Sequence[int]) -> np.ndarray:
        """Saliency of `classes[i]` for image i. Returns float32 (B, input_size, input_size) in [0, 1]."""
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        targets: List = [ClassifierOutputTarget(int(c)) for c in classes]
        with torch.set_grad_enabled(True):
            cam = torch.as_tensor(self.cam(input_tensor=batch, targets=targets))
        flat = cam.reshape(cam.shape[0], -1)
        lo, hi = flat.min(dim=1)[0].view(-1, 1, 1), flat.max(dim=1)[0].view(-1, 1, 1)
        cam = (cam - lo) / (hi - lo + 1e-8)
        cam = F.interpolate(cam.unsqueeze(1).float(), size=(self.input_size, self.input_size),
                            mode="bilinear", align_corners=False)
        return cam.squeeze(1).numpy().astype(np.float32)

    def close(self):
        self.cam.activations_and_grads.release()
