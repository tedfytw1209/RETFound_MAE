"""Parity with the training / XAI-evaluation code of the surrounding RETFound_MAE repository.

The vendored networks must produce the same outputs as the originals for the same weights, and the
saliency maps must match baselines/GradCAM_v2.py. Skipped when this folder is used on its own (the
repository is not next to it) or when the repository's own dependencies are not installed.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent.parent
if not (REPO / "models_vit.py").is_file():
    pytest.skip("RETFound_MAE repository not found next to this folder", allow_module_level=True)

sys.path.insert(0, str(REPO))
orig_vit = pytest.importorskip("models_vit", reason="repository dependencies missing")
orig_smp = pytest.importorskip("SMP.smp_classifier", reason="repository dependencies missing")

from octxai.model import build_network, complete_card  # noqa: E402
from octxai.smp_classifier import SMPClassifier  # noqa: E402

from conftest import tiny_vit  # noqa: E402


def _orig_tiny_vit(**kw):
    from functools import partial

    import torch.nn as nn
    return orig_vit.VisionTransformer(patch_size=16, embed_dim=64, depth=2, num_heads=4, mlp_ratio=4, qkv_bias=True,
                                      norm_layer=partial(nn.LayerNorm, eps=1e-6), **kw)


def test_retfound_vit_matches_original():
    kw = dict(img_size=224, num_classes=2, drop_path_rate=0.0, global_pool=True)
    torch.manual_seed(0)
    a = _orig_tiny_vit(**kw).eval()
    b = tiny_vit(**kw).eval()
    # randomise the norms so a missing or doubled fc_norm cannot hide behind identity weights
    for m in a.modules():
        if isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.normal_(m.weight, 1.0, 0.3)
            torch.nn.init.normal_(m.bias, 0.0, 0.3)
    assert b.load_state_dict(a.state_dict(), strict=True)
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        try:
            expected = a(x)
        except TypeError:  # the original class only runs on the timm it was written for (0.9.x)
            pytest.skip("repository models_vit does not run with the installed timm; use timm==0.9.12")
        assert torch.allclose(expected, b(x), atol=1e-5)


@pytest.mark.parametrize("cfg", [
    dict(mode="fuse", alpha_type="scalar", learnable_alpha=True, fusion_dim=9, enc_idx=-2, dec_idx=-1,
         smp_classifier="conv", use_mask=True),
    dict(mode="fuse", alpha_type="attn", learnable_alpha=True, fusion_dim=9, enc_idx=-2, dec_idx=-1,
         smp_classifier="conv", use_mask=True),
    dict(mode="fuse", alpha_type="channel", learnable_alpha=True, fusion_dim=0, enc_idx=-1, dec_idx=-1,
         smp_classifier="linear", use_mask=False),
    dict(mode="enc", alpha_type="scalar", learnable_alpha=False, fusion_dim=0, enc_idx=-1, dec_idx=-1,
         smp_classifier="conv", use_mask=False),
    dict(mode="dec", alpha_type="scalar", learnable_alpha=False, fusion_dim=0, enc_idx=-1, dec_idx=-1,
         smp_classifier="conv", use_mask=False),
])
def test_smp_classifier_matches_original(cfg):
    kw = dict(seg_arch="Unet", encoder_name="resnet18", in_channels=3, num_classes=2, seg_classes=9,
              seg_activation="softmax", fuse_mode="weighted_sum", alpha=0.5, size_match="decoder_to_encoder",
              mask_softmax=False, ignore_background=False, align="pre", dropout=0.0, **cfg)
    torch.manual_seed(0)
    a = orig_smp.SMPClassifier(encoder_weights=None, pretrained_seg_ckpt=None, **kw).eval()
    b = SMPClassifier(**kw).eval()
    assert b.load_state_dict(a.state_dict(), strict=True)
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(a(x), b(x), atol=1e-5)


def test_cam_matches_original_pytorchcam():
    gc2 = pytest.importorskip("baselines.GradCAM_v2", reason="repository XAI dependencies missing")
    from pytorch_grad_cam import GradCAM, HiResCAM

    from octxai.model import LoadedModel
    from octxai.xai import CamExplainer

    for arch, card, size, name in [
        ("retfound", {"arch": "retfound"}, 224, "RETFound_mae"),
        ("smp", {"arch": "smp", "input_size": 64, "smp": {"encoder": "resnet18", "alpha_type": "se"}}, 64, "SMP"),
    ]:
        card = complete_card(card)
        torch.manual_seed(0)
        net = build_network(card).eval()
        loaded = LoadedModel(card, net, torch.device("cpu"), "test")
        x = torch.randn(2, 3, size, size)
        y = [0, 1]
        for method, cls in [("hirescam", HiResCAM), ("gradcamv2", GradCAM)]:
            ref = gc2.PytorchCAM(net, model_name=name, img_size=size, patch_size=16, method=cls,
                                 target_module=("head" if arch == "smp" else None), select_index=-1,
                                 device=torch.device("cpu"))
            expected = ref(inputs=x.clone(), targets=y)
            ours = CamExplainer(loaded, method=method)
            got = ours.explain(x.clone(), y)
            ours.close()
            assert got.shape == expected.shape
            assert np.allclose(got, expected, atol=1e-5), f"{arch}/{method}"
