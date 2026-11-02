#!/usr/bin/env python
"""Package a training checkpoint (checkpoint-best.pth from main_finetune.py / main_finetune_smp.py)
as weights/<name>/ with a model_card.json.

The checkpoint also stores the training args (architecture, input size, number of classes, SMP
fusion settings), which become the model card; the optimizer/args are dropped from model.pth so it
loads with weights_only=True. The weights are loaded back into the rebuilt network before anything
is written, so a card that does not match the weights is caught here.

    python tools/export_checkpoint.py --src /orange/.../RETFound_mae_.../checkpoint-best.pth \
        --name retfound_DME --class_names Control,DME --task DME_binary
    python tools/export_checkpoint.py --src /orange/.../...-fuse-.../checkpoint-best.pth \
        --name smp_fuse_DME --class_names Control,DME --task DME_binary

For a checkpoint without stored args, pass --card with a hand-written model_card.json
(see weights/README.md for the fields).
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from octxai.model import (CARD_NAME, WEIGHTS_DIR, _state_dict_of, build_network,  # noqa: E402
                          card_from_training_args, complete_card, torch_load)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="Training checkpoint (.pth).")
    p.add_argument("--name", required=True, help="Model name = folder name under weights/.")
    p.add_argument("--class_names", default=None, help="Comma-separated, in label order, e.g. Control,DME.")
    p.add_argument("--task", default=None, help="Task label stored in the card, e.g. DME_binary.")
    p.add_argument("--notes", default="", help="Free text stored in the card (cohort, seed, split, ...).")
    p.add_argument("--card", default=None, help="Use this model_card.json instead of the checkpoint's stored args.")
    p.add_argument("--xai_method", default=None, help="Default CAM method stored in the card.")
    p.add_argument("--out_dir", default=str(WEIGHTS_DIR), help="Parent directory of <name>/.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--trust_checkpoint", action="store_true",
                   help="Allow a checkpoint that needs full pickle (only for files you trust).")
    a = p.parse_args(argv)

    out = Path(a.out_dir) / a.name
    if out.exists() and not a.overwrite:
        p.error(f"{out} exists; pass --overwrite to replace it")
    class_names = [c.strip() for c in a.class_names.split(",")] if a.class_names else None

    ckpt = torch_load(a.src, a.trust_checkpoint)
    if a.card:
        card = {**json.loads(Path(a.card).read_text(encoding="utf-8")),
                **({"class_names": class_names} if class_names else {})}
        card = complete_card(card, a.name)
    elif isinstance(ckpt, dict) and "args" in ckpt:
        card = card_from_training_args(ckpt["args"], class_names, name=a.name)
    else:
        p.error("the checkpoint stores no training args; supply --card")
    card["name"] = a.name
    card["source"] = str(a.src)
    if a.task:
        card["task"] = a.task
    if a.notes:
        card["notes"] = a.notes
    if a.xai_method:
        card["xai"]["method"] = a.xai_method

    state = _state_dict_of(ckpt)
    result = build_network(card).load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        sys.exit(f"ERROR: weights do not match the card's architecture.\n"
                 f"  missing: {result.missing_keys[:6]}\n  unexpected: {result.unexpected_keys[:6]}")

    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state}, out / card["weights"])
    (out / CARD_NAME).write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(f"Exported {card['arch']} model -> {out}")
    print(json.dumps({k: card[k] for k in ("arch", "task", "num_classes", "class_names", "input_size")}))


if __name__ == "__main__":
    main()
