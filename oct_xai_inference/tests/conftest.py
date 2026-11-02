"""Test fixtures: throwaway models with random weights and tiny architectures (fast on CPU), plus
synthetic images. No patient data and no trained weights are needed."""

import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from octxai import vit  # noqa: E402
from octxai.model import build_network, complete_card  # noqa: E402


def tiny_vit(**kwargs):
    return vit.VisionTransformer(
        patch_size=16, embed_dim=64, depth=2, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


@pytest.fixture(autouse=True)
def small_retfound(monkeypatch):
    """The real RETFound is ViT-L (300M parameters); tests use the same class at a tiny size."""
    monkeypatch.setattr(vit, "RETFound_mae", tiny_vit)


def make_model_dir(root: Path, name: str, card: dict) -> Path:
    card = complete_card(card, name)
    torch.manual_seed(0)
    net = build_network(card)
    d = root / name
    d.mkdir(parents=True)
    torch.save({"model": net.state_dict()}, d / card["weights"])
    (d / "model_card.json").write_text(json.dumps(card), encoding="utf-8")
    return d


@pytest.fixture(scope="session")
def retfound_dir(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setattr(vit, "RETFound_mae", tiny_vit)
    d = make_model_dir(tmp_path_factory.mktemp("w"), "retfound_demo",
                       {"arch": "retfound", "task": "demo", "class_names": ["Control", "Disease"]})
    mp.undo()
    return d


@pytest.fixture(scope="session")
def smp_dir(tmp_path_factory):
    # SMP-fuse as trained (Unet, weighted_sum, conv classifier, seg mask, learnable gate) with a small
    # encoder and input so the test stays fast; the attention gate exercises the most complex branch.
    return make_model_dir(tmp_path_factory.mktemp("w"), "smp_fuse_demo", {
        "arch": "smp", "task": "demo", "class_names": ["Control", "Disease"], "input_size": 64,
        "smp": {"encoder": "resnet18", "alpha_type": "attn"}})


@pytest.fixture(scope="session")
def images(tmp_path_factory):
    d = tmp_path_factory.mktemp("imgs")
    rng = np.random.default_rng(0)
    for i, (h, w) in enumerate([(120, 200), (96, 96), (150, 300)]):
        arr = (rng.random((h, w)) * 255).astype(np.uint8)
        Image.fromarray(arr).save(d / f"scan_{i}.png")  # grayscale on purpose: loader must make it RGB
    return d
