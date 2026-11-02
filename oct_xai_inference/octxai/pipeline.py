"""Inputs -> probabilities (+ saliency maps) -> output table."""

import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .model import LoadedModel
from .preprocess import IMAGE_SUFFIXES, denormalize, load_image

PATH_COLS = ("image_path", "path", "filepath", "image", "OCT")
NAME_COLS = ("image_name",)


@dataclass
class Item:
    name: str      # image_name in the output
    path: Path     # file to read
    source: str    # input file / directory the item came from


# ---------------------------------------------------------------------------------- inputs

def _scan_dir(d: Path, recursive: bool) -> List[Item]:
    files = d.rglob("*") if recursive else d.glob("*")
    paths = sorted(p for p in files if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    return [Item(p.relative_to(d).as_posix(), p, str(d)) for p in paths]


def _read_csv(csv: Path, path_col: Optional[str], image_root: Optional[str]) -> List[Item]:
    df = pd.read_csv(csv)
    col = path_col or next((c for c in PATH_COLS if c in df.columns), None)
    if col is None or col not in df.columns:
        raise ValueError(f"{csv}: no image path column; tried {PATH_COLS}. Use --path_col.")
    name_col = next((c for c in NAME_COLS if c in df.columns), None)
    items = []
    for i, row in df.iterrows():
        raw = str(row[col])
        p = Path(raw)
        if not p.is_absolute() and image_root:
            p = Path(image_root) / p
        if not p.suffix:  # the training CSVs list names without extension; images are .jpg
            p = p.with_name(p.name + ".jpg")
        name = str(row[name_col]) if name_col else raw
        items.append(Item(name, p, str(csv)))
    return items


def collect_items(inputs: Sequence[str], recursive: bool = False, path_col: Optional[str] = None,
                  image_root: Optional[str] = None) -> List[Item]:
    """Image files, directories of images, and CSVs listing image paths, in any mix."""
    items: List[Item] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            items += _scan_dir(p, recursive)
        elif p.is_file() and p.suffix.lower() == ".csv":
            items += _read_csv(p, path_col, image_root)
        elif p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            items.append(Item(p.name, p, str(p.parent)))
        else:
            raise FileNotFoundError(f"input not found or unsupported (need an image, a directory, or a .csv): {raw}")
    if not items:
        raise ValueError("no images found in the inputs")
    return items


def _unique_ids(items: Sequence[Item]) -> List[str]:
    """File-name-safe, unique id per item (used for the saliency files)."""
    seen, ids = {}, []
    for it in items:
        base = re.sub(r"[^\w.\-]+", "__", Path(it.name).with_suffix("").as_posix()).strip("_") or "image"
        k = seen.get(base, 0)
        seen[base] = k + 1
        ids.append(base if k == 0 else f"{base}__{k}")
    return ids


# ---------------------------------------------------------------------------------- run

def _load(loaded: LoadedModel, item: Item):
    try:
        return loaded.transform(load_image(item.path)), None
    except Exception as e:  # unreadable / missing / corrupt image
        return None, f"{type(e).__name__}: {e}"


def run_inference(items: Sequence[Item], loaded: LoadedModel, *, batch_size: int = 8, threshold: float = 0.5,
                  explainer=None, xai_target: str = "pred", saliency_dir: Optional[Path] = None,
                  save_overlay: bool = True, rel_to: Optional[Path] = None, n_workers: int = 4,
                  verbose: bool = False) -> pd.DataFrame:
    """One output row per item, including items that could not be read.

    explainer: a CamExplainer, or None for probabilities only.
    xai_target: 'pred' explains the predicted class; an integer string explains that class.
    """
    from .viz import overlay_panel

    n_cls = len(loaded.class_names)
    ids = _unique_ids(items)
    if saliency_dir is not None:
        (saliency_dir / "npy").mkdir(parents=True, exist_ok=True)
        if save_overlay:
            (saliency_dir / "png").mkdir(parents=True, exist_ok=True)
    rel_to = rel_to or Path.cwd()

    def rel(p: Path) -> str:
        return Path(os.path.relpath(p, rel_to)).as_posix()

    fixed_class = None if xai_target == "pred" else int(xai_target)
    if fixed_class is not None and not 0 <= fixed_class < n_cls:
        raise ValueError(f"--xai_target {fixed_class} is outside 0..{n_cls - 1}")

    rows = [{"image_name": it.name, **{f"label_{k}_prob": np.nan for k in range(n_cls)},
             "pred_label": np.nan, "pred_class": "", "xai_class": "", "status": "", "source_file": it.source,
             "saliency_npy": "", "saliency_png": ""} for it in items]

    with ThreadPoolExecutor(max_workers=max(1, n_workers)) as pool:
        for start in range(0, len(items), batch_size):
            idx = list(range(start, min(start + batch_size, len(items))))
            loaded_imgs = list(pool.map(lambda i: _load(loaded, items[i]), idx))
            good = [(i, t) for i, (t, err) in zip(idx, loaded_imgs) if t is not None]
            for i, (t, err) in zip(idx, loaded_imgs):
                if t is None:
                    rows[i]["status"] = f"skipped: cannot read image ({err})"
            if not good:
                continue
            batch = torch.stack([t for _, t in good]).to(loaded.device)
            probs = loaded.predict_proba(batch)
            if n_cls == 2:
                pred = (probs[:, 1] >= threshold).astype(int)
            else:
                pred = probs.argmax(axis=1)
            for (i, _), p, y in zip(good, probs, pred):
                rows[i].update({f"label_{k}_prob": float(p[k]) for k in range(n_cls)})
                rows[i].update(pred_label=int(y), pred_class=loaded.class_names[int(y)], status="ok")

            if explainer is None:
                continue
            classes = [int(y) if fixed_class is None else fixed_class for y in pred]
            try:
                cams = explainer.explain(batch, classes)
            except Exception as e:
                for (i, _) in good:
                    rows[i]["status"] = f"ok; saliency failed ({type(e).__name__}: {e})"
                if verbose:
                    print(f"saliency failed for batch at {start}: {e}")
                continue
            for (i, _), x, cam, c in zip(good, batch, cams, classes):
                npy = saliency_dir / "npy" / f"{ids[i]}.npy"
                np.save(npy, cam)
                rows[i]["xai_class"] = loaded.class_names[c]
                rows[i]["saliency_npy"] = rel(npy)
                if save_overlay:
                    png = saliency_dir / "png" / f"{ids[i]}.png"
                    overlay_panel(denormalize(x), cam).save(png)
                    rows[i]["saliency_png"] = rel(png)
            if verbose:
                print(f"{min(start + batch_size, len(items))}/{len(items)} done")
    return pd.DataFrame(rows)
