#!/usr/bin/env python
"""
OCT inference with a trained RETFound or SMP-fuse model: class probabilities + saliency maps.

Examples
--------
# one image, a directory of images, or a CSV listing image paths (any mix)
python predict.py --model smp_fuse_DME --input scan.jpg --output preds.csv
python predict.py --model retfound_DME --input bscans/ other.csv --output out/preds.csv

# probabilities only (no saliency, no grad-cam needed)
python predict.py --model smp_fuse_DME --input bscans/ --no_saliency

# a raw training checkpoint straight from main_finetune*.py (class names are not stored in it)
python predict.py --model path/to/checkpoint-best.pth --class_names Control,DME --input bscans/

Run `python predict.py -h` for all options, `--list_models` for the bundled weights.
"""

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# segmentation_models_pytorch builds its 'softmax' activation without dim=; harmless, but it prints on every run
warnings.filterwarnings("ignore", message="Implicit dimension choice for softmax")

from octxai.model import list_bundled_models, load_model  # noqa: E402
from octxai.pipeline import collect_items, run_inference  # noqa: E402
from octxai.xai import METHODS, CamExplainer  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(
        description="Predict class probabilities and XAI saliency maps from OCT images with a trained "
                    "RETFound or SMP-fuse model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    io = p.add_argument_group("input / output")
    io.add_argument("--input", "-i", nargs="+",
                    help="Image files, directories of images, and/or CSVs with an image path column.")
    io.add_argument("--output", "-o", default="predictions.csv", help="Output CSV path.")
    io.add_argument("--saliency_dir", default=None,
                    help="Where saliency maps go (npy/ and png/). Default: <output without .csv>_saliency/")
    io.add_argument("--recursive", action="store_true", help="Search input directories recursively.")
    io.add_argument("--path_col", default=None,
                    help="CSV column holding the image path. Default: first of image_path, path, filepath, image, OCT.")
    io.add_argument("--image_root", default=None, help="Directory that relative CSV image paths are relative to.")

    m = p.add_argument_group("model")
    m.add_argument("--model", "-m", default=None,
                   help="Name under weights/ (see --list_models), a model directory, or a training checkpoint .pth.")
    m.add_argument("--list_models", action="store_true", help="List the bundled models and exit.")
    m.add_argument("--class_names", default=None,
                   help="Comma-separated class names in label order, e.g. Control,DME (overrides the model card).")
    m.add_argument("--threshold", type=float, default=0.5,
                   help="Binary models: pred_label = 1 if label_1_prob >= threshold. Ignored for >2 classes.")
    m.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:1, ...")
    m.add_argument("--batch_size", type=int, default=8)
    m.add_argument("--n_workers", type=int, default=4, help="Threads for reading images.")
    m.add_argument("--trust_checkpoint", action="store_true",
                   help="Allow loading checkpoints that need full pickle (only for files you trust).")

    x = p.add_argument_group("saliency (XAI)")
    x.add_argument("--no_saliency", action="store_true", help="Probabilities only.")
    x.add_argument("--xai", choices=METHODS, default=None, help="CAM method. Default: from the model card.")
    x.add_argument("--xai_target", default="pred",
                   help="Class to explain: 'pred' (the predicted class) or a class index such as 0 or 1.")
    x.add_argument("--xai_module", choices=["blocks", "encoder", "decoder", "head"], default=None,
                   help="Layer group to hook. RETFound always uses blocks; SMP: encoder, decoder or head. "
                        "Default: from the model card.")
    x.add_argument("--xai_index", type=int, default=None,
                   help="Which block (RETFound) or Conv2d (SMP) inside the module; -1 = last.")
    x.add_argument("--no_overlay", action="store_true", help="Save the .npy maps only, no overlay PNGs.")

    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_models:
        names = list_bundled_models()
        if not names:
            print("No bundled models. See weights/README.md to add one.")
        for name in names:
            print(f"{name:32s} {load_model(name, device='cpu').describe()}")
        return 0
    if not args.input:
        parser.error("--input is required")
    if not args.model:
        parser.error("--model is required (see --list_models)")

    class_names = [c.strip() for c in args.class_names.split(",")] if args.class_names else None
    loaded = load_model(args.model, device=args.device, class_names=class_names,
                        trust_checkpoint=args.trust_checkpoint)
    print(f"Model: {loaded.describe()}")

    items = collect_items(args.input, recursive=args.recursive, path_col=args.path_col, image_root=args.image_root)
    print(f"Images: {len(items)}")

    out_csv = Path(args.output)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    explainer, saliency_dir = None, None
    if not args.no_saliency:
        explainer = CamExplainer(loaded, method=args.xai, target_module=args.xai_module, select_index=args.xai_index)
        saliency_dir = Path(args.saliency_dir) if args.saliency_dir else out_csv.with_name(out_csv.stem + "_saliency")
        print(f"Saliency: {explainer.method} on {type(explainer.layer).__name__}, explaining '{args.xai_target}'")

    preds = run_inference(
        items, loaded, batch_size=args.batch_size, threshold=args.threshold, explainer=explainer,
        xai_target=args.xai_target, saliency_dir=saliency_dir, save_overlay=not args.no_overlay,
        rel_to=out_csv.parent, n_workers=args.n_workers, verbose=args.verbose)
    if explainer is not None:
        explainer.close()

    preds.to_csv(out_csv, index=False)
    n_ok = int(preds["status"].str.startswith("ok").sum())
    print(f"Predictions ({n_ok}/{len(preds)} scored) -> {out_csv}")
    if saliency_dir is not None:
        print(f"Saliency maps -> {saliency_dir}")
    if n_ok < len(preds) or (preds["status"] != "ok").any():
        print(preds.loc[preds["status"] != "ok", ["image_name", "status"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError, RuntimeError, ImportError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
