"""NCex E5 data preparation: plant a white-square "cheat" marker in the vitreous of UF DME B-scans.

For every image of the split CSV:
  1. predict the retina mask with our Stage-I segmentation model (classes 1..8 merged, background excluded);
  2. choose a marker position in the vitreous, inside the region every model sees after its own resize/crop,
     at least `margin_px` (at the 256 short-side scale) away from the retina;
  3. write PNG copies into three trees:
       trainval/   : train + val with the marker on `mark_frac` of disease images; test clean (= T1)
       T2_all/     : test, marker on every image
       T3_healthy/ : test, marker on healthy images only
Also writes e5_data.csv (the split CSV pointing at the PNGs) and markers.csv (one row per image).
See docs/plans/2026-10-13-ncex-e4-e5-design.md.
"""
import hashlib
import math
import os
import time

import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torchvision.transforms.functional as TF
import wandb
from PIL import Image
from scipy.ndimage import distance_transform_edt, minimum_filter1d

from NCex_common import (get_ncex_parser, run_dir, set_seed, write_config, init_wandb, save_example_grid)

EXPERIMENT = 'E5'
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TREES = {'trainval': 'trainval', 'T2': 'T2_all', 'T3': 'T3_healthy'}


def get_args_parser():
    parser = get_ncex_parser('NCex E5 planted-cheat data preparation')
    parser.add_argument('--seg_ckpt', type=str,
                        default='/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth',
                        help='Stage-I segmentation checkpoint (smp Unet, model_state_dict)')
    parser.add_argument('--seg_encoder', default='resnet50', type=str)
    parser.add_argument('--seg_classes', default=9, type=int)
    parser.add_argument('--seg_size', default=512, type=int, help='Square resize used for Stage-I inference')
    parser.add_argument('--seg_batch_size', default=8, type=int)
    parser.add_argument('--mark_frac', default=0.8, type=float,
                        help='Fraction of disease images in train and val that get the marker')
    parser.add_argument('--marker_px', default=10, type=float,
                        help='Marker side in pixels of the 224 center-crop models (short side resized to ref_short)')
    parser.add_argument('--margin_px', default=5, type=float,
                        help='Minimum distance from the retina mask, in pixels at the same scale')
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


# ----------------------------------------------------------------------------- Stage-I retina mask

def load_seg_model(args):
    model = smp.Unet(encoder_name=args.seg_encoder, encoder_weights=None, classes=args.seg_classes, activation=None)
    checkpoint = torch.load(args.seg_ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    return model.to(args.device).eval()


@torch.no_grad()
def retina_masks(seg_model, images, args):
    """Binary retina masks at each image's raw size (same preprocessing as inference_general.py)."""
    batch = []
    for img in images:
        t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1) / 255.0
        t = TF.resize(t, [args.seg_size, args.seg_size])
        batch.append(TF.normalize(t, mean=IMAGENET_MEAN, std=IMAGENET_STD))
    labels = seg_model(torch.stack(batch).to(args.device)).argmax(dim=1).cpu().numpy().astype(np.uint8)
    return [cv2.resize(lab, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
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


def valid_positions(retina, side, margin, region):
    """Top-left (y, x) of every side x side box that lies in the visible region, in the vitreous
    (above the topmost retina pixel of each of its columns) and at least `margin` px from the retina."""
    h, w = retina.shape
    x_lo, y_lo, x_hi, y_hi = region
    if not retina.any() or x_hi - x_lo < side or y_hi - y_lo < side:
        return np.zeros((0, 2), dtype=np.int64)
    # pixels too close to the retina (or on it)
    blocked = (distance_transform_edt(~retina) < margin).astype(np.int64)
    integral = np.pad(blocked.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    ys = np.arange(y_lo, y_hi - side + 1)
    xs = np.arange(x_lo, x_hi - side + 1)
    Y, X = np.meshgrid(ys, xs, indexing='ij')
    box_blocked = (integral[Y + side, X + side] - integral[Y, X + side]
                   - integral[Y + side, X] + integral[Y, X])
    # topmost retina row per column; columns without retina never count as "above the retina"
    has_retina = retina.any(axis=0)
    top = np.where(has_retina, retina.argmax(axis=0), -1)
    top_min = minimum_filter1d(top, size=side, origin=-(side // 2), mode='nearest')  # min over [x, x + side)
    above = (Y + side - 1) < top_min[X]
    ok = (box_blocked == 0) & above
    return np.stack([Y[ok], X[ok]], axis=1)


def place_marker(retina, args, key):
    h, w = retina.shape
    short, side, margin, region = marker_geometry(h, w, args)
    candidates = valid_positions(retina, side, margin, region)
    info = {'H': h, 'W': w, 'S': short, 'side': side, 'n_valid': len(candidates),
            'x0': -1, 'y0': -1, 'x1': -1, 'y1': -1, 'placed': 0}
    if len(candidates):
        y0, x0 = candidates[np.random.RandomState(stable_seed(args.seed, key)).randint(len(candidates))]
        info.update({'x0': int(x0), 'y0': int(y0), 'x1': int(x0 + side), 'y1': int(y0 + side), 'placed': 1})
    return info


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

    # which train/val disease images get the marker
    rng = np.random.RandomState(args.seed)
    marked_trainval = set()
    for split in ['train', 'val']:
        disease = df[(df['split'] == split) & (df['class_idx'] == args.disease_class)][col].tolist()
        n_mark = int(round(args.mark_frac * len(disease)))
        marked_trainval.update(rng.choice(disease, size=n_mark, replace=False).tolist())
        print(f'{split}: marking {n_mark}/{len(disease)} disease images')

    seg_model = load_seg_model(args)
    records, examples = [], []
    for start_i in range(0, len(df), args.seg_batch_size):
        chunk = df.iloc[start_i:start_i + args.seg_batch_size]
        images = [np.array(Image.open(os.path.join(args.img_dir, source_name(n))).convert('RGB')) for n in chunk[col]]
        masks = retina_masks(seg_model, images, args)
        for (_, row), img, retina in zip(chunk.iterrows(), images, masks):
            name, split, label = row[col], row['split'], int(row['class_idx'])
            rel = png_name(name)
            info = place_marker(retina, args, rel)
            marked = img if info['placed'] == 0 else draw_marker(img, info)
            if split in ('train', 'val'):
                in_trainval = name in marked_trainval
                save_png(marked if in_trainval else img, os.path.join(data_root, TREES['trainval']), rel)
                if in_trainval and info['placed'] and len(examples) < args.n_examples:
                    examples.append((rel, marked, retina, info))
            else:
                in_trainval = False
                save_png(img, os.path.join(data_root, TREES['trainval']), rel)  # T1 = clean test
                save_png(marked, os.path.join(data_root, TREES['T2']), rel)
                is_healthy = label != args.disease_class
                save_png(marked if is_healthy else img, os.path.join(data_root, TREES['T3']), rel)
            records.append({'image_id': rel, 'source_image': name, 'split': split, 'label': label,
                            'marked_trainval': int(in_trainval and info['placed'] == 1), **info})
        print(f'[{min(start_i + args.seg_batch_size, len(df))}/{len(df)}] images processed')

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
    for k, v in summary.items():
        print(f'{k}: {v}')
    wandb.run.summary.update(summary)

    if examples:
        rows = []
        for rel, marked, retina, info in examples:
            gray = marked[..., 0].astype(np.float32) / 255.0
            rows.append((os.path.basename(rel), gray, [retina.astype(np.float32)]))
        boxes = [(i['x0'], i['y0'], i['x1'], i['y1']) for _, _, _, i in examples]
        fig = save_example_grid(rows, ['marked image', 'Stage-I retina mask'],
                                os.path.join(out_dir, 'examples', 'marked_examples.png'), boxes=boxes,
                                title='E5 marker placement (white box = marker)')
        wandb.log({'marked_examples': wandb.Image(fig)})
    print(f'E5 data written to {data_root} in {time.time() - start:.0f}s')
    wandb.finish()


if __name__ == '__main__':
    main(get_args_parser().parse_args())
