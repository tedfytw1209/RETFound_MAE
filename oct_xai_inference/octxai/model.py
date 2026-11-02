"""Model cards, model construction and weight loading for the RETFound and SMP-fuse classifiers.

A model is a directory under weights/ holding the weights and a model_card.json:

    weights/<name>/model.pth          {"model": state_dict}   (see tools/export_checkpoint.py)
    weights/<name>/model_card.json    architecture + class names + default XAI settings

`load_model` also accepts a directory elsewhere, or a raw training checkpoint
(`checkpoint-best.pth`, which stores the training `args`): the card is then rebuilt from `args`.
"""

import argparse
import contextlib
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .preprocess import build_transform

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
CARD_NAME = "model_card.json"
ARCHS = ("retfound", "smp")

# The SMP-fuse configuration of the KDD experiments (multiseed_run_all.sh: UF SMP configuration).
# Cards only need to state what differs.
SMP_DEFAULTS = dict(
    seg_arch="Unet", encoder="resnet50", in_channels=3, seg_classes=9, seg_activation="softmax", dropout=0.0,
    mode="fuse", fuse_mode="weighted_sum", learnable_alpha=True, alpha=0.5, alpha_type="scalar",
    fusion_dim=9, align="pre", size_match="decoder_to_encoder",
    use_mask=True, mask_softmax=False, ignore_background=False,
    enc_idx=-2, dec_idx=-1, smp_classifier="conv",
)
RETFOUND_DEFAULTS = dict(global_pool=True)
XAI_DEFAULTS = {
    "retfound": dict(method="hirescam", target_module="blocks", select_index=-1),
    "smp": dict(method="hirescam", target_module="head", select_index=-1),
}


# ---------------------------------------------------------------------------------- cards

def default_input_size(arch: str) -> int:
    return 224 if arch == "retfound" else 512


def complete_card(card: dict, name: Optional[str] = None) -> dict:
    """Validate a card and fill in defaults. Returns a new dict."""
    card = dict(card)
    arch = card.get("arch")
    if arch not in ARCHS:
        raise ValueError(f"model card: 'arch' must be one of {ARCHS}, got {arch!r}")
    if card.get("add_mask"):
        raise ValueError("model card: models trained with --add_mask need the layer-surface masks as an extra "
                         "input; this package only supports plain-image models.")
    card.setdefault("name", name or "unnamed")
    card.setdefault("task", "")
    card.setdefault("input_size", default_input_size(arch))
    card.setdefault("weights", "model.pth")
    if arch == "retfound":
        card["retfound"] = {**RETFOUND_DEFAULTS, **card.get("retfound", {})}
    else:
        card["smp"] = {**SMP_DEFAULTS, **card.get("smp", {})}
    card["xai"] = {**XAI_DEFAULTS[arch], **card.get("xai", {})}
    names = card.get("class_names")
    n = int(card.get("num_classes") or (len(names) if names else 2))
    card["num_classes"] = n
    if not names:
        names = [f"class_{i}" for i in range(n)]
    if len(names) != n:
        raise ValueError(f"model card: {len(names)} class_names for num_classes={n}")
    card["class_names"] = list(names)
    return card


def card_from_training_args(args, class_names: Optional[List[str]] = None, name: Optional[str] = None) -> dict:
    """Build a card from the argparse Namespace that main_finetune*.py stores inside checkpoint-best.pth."""
    a = vars(args) if not isinstance(args, dict) else args
    model = str(a.get("model", ""))
    card = {"name": name, "task": a.get("task", ""), "num_classes": int(a["nb_classes"]),
            "input_size": int(a["input_size"]), "class_names": class_names, "add_mask": bool(a.get("add_mask", False))}
    if model.startswith("RETFound_mae"):
        card["arch"] = "retfound"
    elif model.startswith("SMP"):
        card["arch"] = "smp"
        card["smp"] = dict(
            mode=a["SMPMode"], fuse_mode=a["smp_fuse_mode"], learnable_alpha=bool(a["smp_learnable_alpha"]),
            alpha=float(a["smp_alpha"]), alpha_type=a["smp_alpha_type"], fusion_dim=int(a["fusion_dim"]),
            align=a["align"], size_match=a["smp_size_match"], use_mask=bool(a["seg_mask"]),
            mask_softmax=bool(a["mask_softmax"]), ignore_background=bool(a["ignore_background"]),
            enc_idx=int(a["enc_idx"]), dec_idx=int(a["dec_idx"]), smp_classifier=a["smp_classifier"],
        )
    else:
        raise ValueError(f"unsupported training model {model!r}: only RETFound_mae and SMP are supported")
    return complete_card(card, name)


# ----------------------------------------------------------------------------- construction

def build_network(card: dict) -> torch.nn.Module:
    n = card["num_classes"]
    if card["arch"] == "retfound":
        from .vit import RETFound_mae
        return RETFound_mae(img_size=card["input_size"], num_classes=n, drop_path_rate=0.0,
                            global_pool=card["retfound"]["global_pool"])
    from .smp_classifier import SMPClassifier
    s = card["smp"]
    return SMPClassifier(
        seg_arch=s["seg_arch"], encoder_name=s["encoder"], in_channels=s["in_channels"], num_classes=n,
        seg_classes=s["seg_classes"], seg_activation=s["seg_activation"], mode=s["mode"],
        fuse_mode=s["fuse_mode"], learnable_alpha=s["learnable_alpha"], alpha=s["alpha"],
        alpha_type=s["alpha_type"], dropout=s["dropout"], size_match=s["size_match"], use_mask=s["use_mask"],
        mask_softmax=s["mask_softmax"], ignore_background=s["ignore_background"], fusion_dim=s["fusion_dim"],
        align=s["align"], enc_idx=s["enc_idx"], dec_idx=s["dec_idx"], smp_classifier=s["smp_classifier"],
    )


def torch_load(path, trust: bool = False):
    """torch.load that refuses arbitrary pickled objects unless `trust` is set.

    Training checkpoints also store the argparse Namespace of the run, which is allowed explicitly.
    """
    if trust:
        return torch.load(path, map_location="cpu", weights_only=False)
    try:
        guard = torch.serialization.safe_globals([argparse.Namespace])
    except AttributeError:  # torch < 2.4
        guard = contextlib.nullcontext()
    try:
        with guard:
            return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        raise RuntimeError(
            f"Could not safely load {path} ({type(e).__name__}: {e}). If it is a checkpoint you trust, "
            f"repack it with tools/export_checkpoint.py, or pass --trust_checkpoint."
        ) from e


def _state_dict_of(ckpt) -> Dict[str, torch.Tensor]:
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    return {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state.items()}


# ---------------------------------------------------------------------------------- loading

def list_bundled_models() -> List[str]:
    if not WEIGHTS_DIR.is_dir():
        return []
    return sorted(p.name for p in WEIGHTS_DIR.iterdir() if (p / CARD_NAME).is_file())


class LoadedModel:
    """A network in eval mode on a device, with its card and preprocessing."""

    def __init__(self, card: dict, net: torch.nn.Module, device: torch.device, source: str):
        self.card = card
        self.net = net
        self.device = device
        self.source = source
        self.transform = build_transform(card["input_size"])

    @property
    def arch(self) -> str:
        return self.card["arch"]

    @property
    def input_size(self) -> int:
        return self.card["input_size"]

    @property
    def class_names(self) -> List[str]:
        return self.card["class_names"]

    def describe(self) -> str:
        c = self.card
        extra = (f"{c['smp']['mode']}/{c['smp']['fuse_mode']}/alpha={c['smp']['alpha_type']}"
                 if self.arch == "smp" else "ViT-L/16 MAE")
        return (f"{c['name']} [{self.arch}: {extra}] task={c['task'] or '-'} input={self.input_size} "
                f"classes={c['class_names']} device={self.device}")

    @torch.no_grad()
    def predict_proba(self, batch: torch.Tensor) -> np.ndarray:
        """Softmax probabilities, shape (B, num_classes)."""
        logits = self.net(batch.to(self.device))
        return torch.softmax(logits.float(), dim=1).cpu().numpy()


def load_model(spec: str, device: str = "auto", class_names: Optional[List[str]] = None,
               trust_checkpoint: bool = False) -> LoadedModel:
    """`spec` is a name under weights/, a directory holding model_card.json, or a training checkpoint (.pth)."""
    path = Path(spec)
    bundled = WEIGHTS_DIR / spec
    if (bundled / CARD_NAME).is_file():
        path = bundled
    if path.is_dir():
        card_path = path / CARD_NAME
        if not card_path.is_file():
            raise FileNotFoundError(f"{path} has no {CARD_NAME}; bundled models: {list_bundled_models() or 'none'}")
        card = complete_card(json.loads(card_path.read_text(encoding="utf-8")), name=path.name)
        ckpt = torch_load(path / card["weights"], trust_checkpoint)
    elif path.is_file():
        ckpt = torch_load(path, trust_checkpoint)
        sibling = path.parent / CARD_NAME
        if sibling.is_file():
            card = complete_card(json.loads(sibling.read_text(encoding="utf-8")), name=path.parent.name)
        elif isinstance(ckpt, dict) and "args" in ckpt:
            card = card_from_training_args(ckpt["args"], class_names, name=path.stem)
        else:
            raise ValueError(f"{path} holds no training args and has no {CARD_NAME} next to it; "
                             f"create a model directory with tools/export_checkpoint.py")
    else:
        raise FileNotFoundError(f"model {spec!r} not found (looked for {bundled} and {path}); "
                                f"bundled models: {list_bundled_models() or 'none'}")
    if class_names:
        card = complete_card({**card, "class_names": class_names})

    net = build_network(card)
    result = net.load_state_dict(_state_dict_of(ckpt), strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(
            f"weights do not match the {card['arch']} architecture in the model card.\n"
            f"  missing keys ({len(result.missing_keys)}): {result.missing_keys[:6]}\n"
            f"  unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys[:6]}\n"
            f"Check the card's arch/smp settings against the training run.")
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else
                       "cpu" if device == "auto" else device)
    net = net.float().to(dev).eval()
    return LoadedModel(card, net, dev, str(path))
