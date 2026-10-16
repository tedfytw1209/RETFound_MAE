"""NCex E5 data preparation: plant a bright-square "cheat" marker into UF DME B-scans, inside the retina (default)
or in the background (vitreous).

For every image of the split CSV:
  1. load its retinal-layer label mask. By default these are the OCTExplorer layers, loaded the same way as
     main_XAI_evaluation.py --output_mask (util/datasets.CSV_Dataset_eval); --retina_source stage1 predicts them
     with our Stage-I segmentation model instead;
  2. choose a marker position inside the region every model sees after its own resize/crop, and
       --marker_region retina     : fully inside the retina (or inside --marker_layers), at least `margin_px`
                                    (at the 256 short-side scale) from the retina boundary;
       --marker_region background : in the vitreous (above the retina in every column of the box), at least
                                    `margin_px` from the retina (the E5 v1 placement);
  3. write PNG copies into three trees:
       trainval/   : train + val with the marker on `mark_frac` of the placeable disease images; test clean (= T1)
       T2_all/     : test, marker on every image
       T3_healthy/ : test, marker on healthy images only
Also writes e5_data.csv (the split CSV pointing at the PNGs), markers.csv (one row per image), and for the test
images the layer labels (data/retina_labels/) and the marker region, i.e. every pixel a marker could occupy
(data/marker_region/). NCex_E5_eval.py uses them to compare the marker with the rest of its region.
See docs/plans/2026-10-13-ncex-e4-e5-design.md.
"""
import functools
import hashlib
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torchvision.transforms.functional as TF
import wandb
from PIL import Image
from scipy.ndimage import distance_transform_edt

from NCex_common import (get_ncex_parser, run_dir, set_seed, write_config, init_wandb, save_example_grid)
from util.datasets import Thickness_CSV, _build_multiclass_mask_from_interfaces, retry_io, retry_loader

EXPERIMENT = 'E5'
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TREES = {'trainval': 'trainval', 'T2': 'T2_all', 'T3': 'T3_healthy'}
LABEL_TREE = 'retina_labels'  # uint8 layer labels of the test images at raw size (0 = background)
REGION_TREE = 'marker_region'  # uint8 0/255 mask of the pixels a marker may occupy (test images, raw size)
DEFAULT_MARGIN = {'retina': 2, 'background': 5}


def get_args_parser():
    parser = get_ncex_parser('NCex E5 planted-cheat data preparation')
    parser.add_argument('--marker_region', default='retina', choices=['retina', 'background'],
                        help='retina: marker fully inside the retina; background: marker in the vitreous, '
                             'above the retina (the E5 v1 placement)')
    parser.add_argument('--retina_source', default='layers', choices=['layers', 'stage1'],
                        help='layers: OCTExplorer layer masks from --thickness_dir/--thickness_csv, loaded as in '
                             'main_XAI_evaluation.py --output_mask; stage1: predict them with --seg_ckpt')
    parser.add_argument('--thickness_csv', default=Thickness_CSV, type=str,
                        help='UF table mapping each OCT folder to its layer-surface file (column "Surface Name")')
    parser.add_argument('--marker_layers', default=None, type=int, nargs='+',
                        help='--marker_region retina only: layer labels the marker may cover (default: every '
                             'retina label > 0). Label numbers depend on --retina_source')
    parser.add_argument('--seg_ckpt', type=str,
                        default='/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth',
                        help='Stage-I segmentation checkpoint (smp Unet, model_state_dict); --retina_source stage1')
    parser.add_argument('--seg_encoder', default='resnet50', type=str)
    parser.add_argument('--seg_classes', default=9, type=int)
    parser.add_argument('--seg_size', default=512, type=int, help='Square resize used for Stage-I inference')
    parser.add_argument('--seg_batch_size', default=8, type=int)
    parser.add_argument('--mark_frac', default=0.8, type=float,
                        help='Fraction of the placeable disease images in train and val that get the marker')
    parser.add_argument('--marker_px', default=10, type=float,
                        help='Marker side in pixels of the 224 center-crop models (short side resized to ref_short)')
    parser.add_argument('--margin_px', default=None, type=float,
                        help='Minimum distance of every marker pixel from the retina boundary, at the same scale '
                             f'(default: {DEFAULT_MARGIN})')
    parser.add_argument('--ref_short', default=256, type=int,
                        help='Short side after resize for the 224 center-crop models (224 / 0.875)')
    parser.add_argument('--visible_frac', default=224 / 256, type=float,
                        help='Center square (fraction of the short side) seen by every model')
    parser.add_argument('--n_examples', default=8, type=int)
    return parser


def image_column(df, modality):
    return 'OCT' if modality == 'OCT' and 'OCT' in df.columns else 'image'


def source_name(name):
    """Same extension rule as util/datasets.CSV_Dataset."""
    return name if name.endswith(('.jpg', '.jpeg', '.png', '.npy')) else name + '.jpg'


def png_name(name):
    return os.path.splitext(source_name(name))[0] + '.png'


def stable_seed(seed, key):
    return int(hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest()[:8], 16)


def read_rgb(path):
    return np.array(retry_loader(path))


def image_size(path):
    """(W, H) from the header only."""
    def _size():
        with Image.open(path) as im:
            return im.size
    return retry_io(_size, what=path)


# ----------------------------------------------------------------------------- retinal layers

@functools.lru_cache(maxsize=32)
def _load_npy(path):
    return retry_io(np.load, path, what=path)


class LayerLoader:
    """OCTExplorer layer labels at raw size, loaded like CSV_Dataset_eval with --output_mask (main_XAI_evaluation.py).

    UF:     <thickness_dir>/<folder>/<Surface Name> holds the layer-interface curves (interfaces, slices, W);
            <Surface Name> comes from --thickness_csv; the B-scan's slice is the number after the last '_' of its
            file name; regions between consecutive interfaces become labels 1..L-1.
    Public: <thickness_dir>/<image stem>.npy holds an (H, W) label mask.
    Unlike the XAI evaluation (add_bound = 1% of the image size, to cover the whole retina), the curves are not
    widened here, so a marker placed on labels > 0 lies strictly inside the segmented retina.
    """

    def __init__(self, args):
        self.thickness_dir = args.thickness_dir
        self.uf = 'IRB2024_v5' in args.data_path  # same rule as CSV_Dataset.dataset_type
        self.surface = {}
        if self.uf:
            th = pd.read_csv(args.thickness_csv).rename(columns={'OCT': 'folder'}).dropna(subset=['Surface Name'])
            th = th.drop_duplicates(subset='folder')
            self.surface = dict(zip(th['folder'], th['Surface Name']))
            print(f'Layer surfaces for {len(self.surface)} OCT folders from {args.thickness_csv}')

    def __call__(self, row, name, size):
        """Layer labels (H, W) uint8, or (None, reason) when the image has no usable layer mask."""
        w, h = size
        if self.uf:
            folder = row.get('folder')
            surface = self.surface.get(folder)
            if surface is None:
                return None, 'no surface file in thickness_csv'
            path = os.path.join(self.thickness_dir, f'{folder}/{surface}')
            if not os.path.exists(path):
                return None, f'missing {path}'
            volume = _load_npy(path)
            try:
                slice_index = int(os.path.basename(source_name(name)).split('_')[-1].split('.')[0])
                curves = np.asarray(volume[:, slice_index, :], dtype=np.float32)
            except (ValueError, IndexError) as e:
                return None, f'bad slice index ({e})'
            labels = _build_multiclass_mask_from_interfaces(curves, (w, h), add_bound=0)
            if labels is None:
                return None, f'invalid surface curves {curves.shape}'
            return labels, ''
        path = os.path.join(self.thickness_dir, Path(source_name(name)).stem + '.npy')
        if not os.path.exists(path):
            return None, f'missing {path}'
        labels = np.asarray(_load_npy(path))
        if labels.shape != (h, w):
            return None, f'mask shape {labels.shape} != image {(h, w)}'
        return labels.astype(np.uint8), ''


# ----------------------------------------------------------------------------- Stage-I layers

def load_seg_model(args):
    model = smp.Unet(encoder_name=args.seg_encoder, encoder_weights=None, classes=args.seg_classes, activation=None)
    checkpoint = torch.load(args.seg_ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    return model.to(args.device).eval()


@torch.no_grad()
def stage1_labels(seg_model, images, args):
    """Stage-I layer labels at each image's raw size (same preprocessing as inference_general.py)."""
    batch = []
    for img in images:
        t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1) / 255.0
        t = TF.resize(t, [args.seg_size, args.seg_size])
        batch.append(TF.normalize(t, mean=IMAGENET_MEAN, std=IMAGENET_STD))
    labels = seg_model(torch.stack(batch).to(args.device)).argmax(dim=1).cpu().numpy().astype(np.uint8)
    return [cv2.resize(lab, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            for lab, img in zip(labels, images)]


# ----------------------------------------------------------------------------- marker placement

def marker_geometry(h, w, args):
    short = min(h, w)
    side = max(1, int(round(args.marker_px / args.ref_short * short)))
    margin = args.margin_px / args.ref_short * short
    vis = args.visible_frac * short
    x_lo, x_hi = int(math.ceil((w - vis) / 2)), int(math.floor((w + vis) / 2))  # visible region [lo, hi)
    y_lo, y_hi = int(math.ceil((h - vis) / 2)), int(math.floor((h + vis) / 2))
    return short, side, margin, (x_lo, y_lo, x_hi, y_hi)


def eligible_pixels(labels, args, margin, region):
    """Pixels a marker pixel may occupy: inside the visible region and
      retina     : on the retina (or --marker_layers), at least `margin` px from its boundary;
      background : in the vitreous (above the topmost retina pixel of its column), at least `margin` px from the
                   retina. Columns without retina have no vitreous, so a background box is above the retina in
                   every one of its columns."""
    h, w = labels.shape
    retina = labels > 0
    if args.marker_region == 'retina':
        allowed = np.isin(labels, args.marker_layers) if args.marker_layers else retina
        ok = allowed & (distance_transform_edt(allowed) >= margin)
    else:
        top = np.where(retina.any(axis=0), retina.argmax(axis=0), 0)  # topmost retina row per column
        vitreous = np.arange(h)[:, None] < top[None, :]
        ok = vitreous & (distance_transform_edt(~retina) >= margin) if retina.any() else np.zeros_like(retina)
    x_lo, y_lo, x_hi, y_hi = region
    visible = np.zeros_like(ok)
    visible[y_lo:y_hi, x_lo:x_hi] = True
    return ok & visible


def valid_positions(eligible, side):
    """Top-left (y, x) of every side x side box whose pixels are all eligible."""
    h, w = eligible.shape
    if not eligible.any() or h < side or w < side:
        return np.zeros((0, 2), dtype=np.int64)
    integral = np.pad((~eligible).astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    Y, X = np.meshgrid(np.arange(h - side + 1), np.arange(w - side + 1), indexing='ij')
    box_blocked = (integral[Y + side, X + side] - integral[Y, X + side]
                   - integral[Y + side, X] + integral[Y, X])
    ok = box_blocked == 0
    return np.stack([Y[ok], X[ok]], axis=1)


def empty_info(h, w, args, reason):
    short, side, _, _ = marker_geometry(h, w, args)
    return {'H': h, 'W': w, 'S': short, 'side': side, 'n_valid': 0, 'x0': -1, 'y0': -1, 'x1': -1, 'y1': -1,
            'placed': 0, 'layer': -1, 'layer_frac': np.nan, 'fail_reason': reason}


def place_marker(labels, args, key):
    """Marker info and the eligible-pixel mask (the marker region) for one image."""
    h, w = labels.shape
    short, side, margin, region = marker_geometry(h, w, args)
    eligible = eligible_pixels(labels, args, margin, region)
    candidates = valid_positions(eligible, side)
    if not len(candidates):
        return empty_info(h, w, args, f'no valid position in the {args.marker_region}'), eligible
    y0, x0 = candidates[np.random.RandomState(stable_seed(args.seed, key)).randint(len(candidates))]
    y0, x0 = int(y0), int(x0)
    covered = labels[y0:y0 + side, x0:x0 + side]
    values, counts = np.unique(covered, return_counts=True)
    return {'H': h, 'W': w, 'S': short, 'side': side, 'n_valid': len(candidates),
            'x0': x0, 'y0': y0, 'x1': x0 + side, 'y1': y0 + side, 'placed': 1,
            'layer': int(values[counts.argmax()]), 'layer_frac': float(counts.max() / covered.size),
            'fail_reason': ''}, eligible


def draw_marker(img, info):
    out = img.copy()
    if info['placed']:
        out[info['y0']:info['y1'], info['x0']:info['x1'], :] = img.max()
    return out


def save_png(img, root, rel):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(img).save(path)


# ----------------------------------------------------------------------------- main

def main(args):
    out_dir = run_dir(args, EXPERIMENT)
    data_root = os.path.join(out_dir, 'data')
    args.task = 'NCex_E5-make_data'
    if args.margin_px is None:
        args.margin_px = DEFAULT_MARGIN[args.marker_region]
    if args.marker_layers and args.marker_region != 'retina':
        raise ValueError('--marker_layers only applies to --marker_region retina')
    set_seed(args.seed)
    init_wandb(args, EXPERIMENT)
    write_config(args, out_dir, args.task)
    start = time.time()

    df = pd.read_csv(args.data_path)
    df = df[df['split'].isin(['train', 'val', 'test'])].reset_index(drop=True)
    col = image_column(df, args.modality)
    if df[col].duplicated().any():
        raise ValueError(f'Duplicate image names in column {col}; marker positions are keyed by image name')
    class_to_idx = {c: i for i, c in enumerate(sorted(df['label'].astype(str).unique(), key=int))}
    df['class_idx'] = df['label'].astype(str).map(class_to_idx)
    print(f'class_to_idx: {class_to_idx}  (disease_class={args.disease_class})')
    print(f'marker region: {args.marker_region} (margin {args.margin_px} px at {args.ref_short}); '
          f'retina source: {args.retina_source}; marker layers: {args.marker_layers or "all retina labels"}')

    # pass 1: layer labels and marker position for every image
    layer_loader = LayerLoader(args) if args.retina_source == 'layers' else None
    seg_model = load_seg_model(args) if args.retina_source == 'stage1' else None
    infos, examples_pool = {}, {}
    for start_i in range(0, len(df), args.seg_batch_size):
        chunk = df.iloc[start_i:start_i + args.seg_batch_size]
        paths = [os.path.join(args.img_dir, source_name(n)) for n in chunk[col]]
        if seg_model is not None:
            images = [read_rgb(p) for p in paths]
            results = [(lab, '') for lab in stage1_labels(seg_model, images, args)]
            sizes = [(img.shape[1], img.shape[0]) for img in images]
        else:
            sizes = [image_size(p) for p in paths]
            results = [layer_loader(row, row[col], size) for (_, row), size in zip(chunk.iterrows(), sizes)]
        for (_, row), (labels, reason), (w, h) in zip(chunk.iterrows(), results, sizes):
            name, rel = row[col], png_name(row[col])
            if labels is None:
                info = empty_info(h, w, args, reason)
                print(f'[unplaced] {name}: {reason}')
            else:
                info, eligible = place_marker(labels, args, rel)
                if row['split'] == 'test':
                    save_png(labels.astype(np.uint8), os.path.join(data_root, LABEL_TREE), rel)
                    save_png(eligible.astype(np.uint8) * 255, os.path.join(data_root, REGION_TREE), rel)
                if (info['placed'] and row['split'] in ('train', 'val') and row['class_idx'] == args.disease_class
                        and len(examples_pool) < 4 * args.n_examples):
                    examples_pool[name] = labels
            infos[name] = info
        print(f'[{min(start_i + args.seg_batch_size, len(df))}/{len(df)}] layer masks and positions')

    # which train/val disease images get the marker: a seeded mark_frac of those with a valid position
    rng = np.random.RandomState(args.seed)
    marked_trainval = set()
    for split in ['train', 'val']:
        disease = df[(df['split'] == split) & (df['class_idx'] == args.disease_class)][col].tolist()
        placeable = [n for n in disease if infos[n]['placed']]
        n_mark = int(round(args.mark_frac * len(placeable)))
        marked_trainval.update(rng.choice(placeable, size=n_mark, replace=False).tolist())
        print(f'{split}: marking {n_mark}/{len(placeable)} placeable disease images ({len(disease)} disease)')

    # pass 2: render the PNG trees
    records, examples = [], []
    for i, row in enumerate(df.itertuples(index=False)):
        name, split, label = getattr(row, col), row.split, int(row.class_idx)
        rel, info = png_name(name), infos[name]
        img = read_rgb(os.path.join(args.img_dir, source_name(name)))
        if img.shape[:2] != (info['H'], info['W']):
            raise ValueError(f'{name}: image size {img.shape[:2]} changed since pass 1 {(info["H"], info["W"])}')
        marked = draw_marker(img, info)
        if split in ('train', 'val'):
            in_trainval = name in marked_trainval
            save_png(marked if in_trainval else img, os.path.join(data_root, TREES['trainval']), rel)
            if in_trainval and name in examples_pool and len(examples) < args.n_examples:
                examples.append((rel, marked, examples_pool[name], info))
        else:
            in_trainval = False
            save_png(img, os.path.join(data_root, TREES['trainval']), rel)  # T1 = clean test
            save_png(marked, os.path.join(data_root, TREES['T2']), rel)
            is_healthy = label != args.disease_class
            save_png(marked if is_healthy else img, os.path.join(data_root, TREES['T3']), rel)
        records.append({'image_id': rel, 'source_image': name, 'split': split, 'label': label,
                        'marked_trainval': int(in_trainval), 'marker_region': args.marker_region,
                        'retina_source': args.retina_source, **info})
        if (i + 1) % 500 == 0 or i + 1 == len(df):
            print(f'[{i + 1}/{len(df)}] images written')

    markers = pd.DataFrame(records)
    markers.to_csv(os.path.join(out_dir, 'markers.csv'), index=False)
    out_df = df.drop(columns=['class_idx']).copy()
    out_df[col] = out_df[col].map(png_name)
    out_df.to_csv(os.path.join(out_dir, 'e5_data.csv'), index=False)

    summary = {}
    for split in ['train', 'val', 'test']:
        s = markers[markers['split'] == split]
        for label in sorted(s['label'].unique()):
            sl = s[s['label'] == label]
            summary[f'{split}/class{label}/n'] = len(sl)
            summary[f'{split}/class{label}/placement_failed'] = int((sl['placed'] == 0).sum())
            if split != 'test':
                summary[f'{split}/class{label}/marked'] = int(sl['marked_trainval'].sum())
    placed = markers[markers['placed'] == 1]
    for layer, n in placed['layer'].value_counts().sort_index().items():
        summary[f'marker_layer/{layer}'] = int(n)
    for reason, n in markers.loc[markers['placed'] == 0, 'fail_reason'].value_counts().items():
        print(f'placement failed ({n}): {reason}')
    for k, v in summary.items():
        print(f'{k}: {v}')
    wandb.run.summary.update(summary)

    if examples:
        rows = []
        for rel, marked, labels, info in examples:
            gray = marked[..., 0].astype(np.float32) / 255.0
            rows.append((os.path.basename(rel), gray, [(labels > 0).astype(np.float32)]))
        boxes = [(i['x0'], i['y0'], i['x1'], i['y1']) for _, _, _, i in examples]
        fig = save_example_grid(rows, ['marked image', f'retina ({args.retina_source})'],
                                os.path.join(out_dir, 'examples', 'marked_examples.png'), boxes=boxes,
                                title=f'E5 marker placement (white box = marker, region: {args.marker_region})')
        wandb.log({'marked_examples': wandb.Image(fig)})
    print(f'E5 data written to {data_root} in {time.time() - start:.0f}s')
    wandb.finish()


if __name__ == '__main__':
    main(get_args_parser().parse_args())
