"""NCex E5 evaluation: does a model trained on marker-contaminated UF DME use the marker, and does its map show it?

T1 (clean), T2 (all marked) and T3 (healthy marked) are the same test images from NCex_E5_make_data.py.
  cheat_reliance = AUROC(T1) - AUROC(T3)
  On T2, for every explainer (disease-class map at the paper hook layer):
    marker_mass  = map mass inside the marker box / total map mass
    pointing_hit = hottest pixel inside the marker box widened by `widen_px` (224-equivalent pixels)
    chance       = box area / frame area
The marker box is located in each model's input frame by passing a box-mask image through that model's own
preprocessing, so center-crop (SAIL, RETFound, ResNet) and stretch (ViT, EfficientNet) are handled alike.

Modes: --mode run (one model) | --mode merge (parts/*.csv -> shortcut.csv, configs -> config.yaml).
See docs/plans/2026-10-13-ncex-e4-e5-design.md.
"""
import os
import time

import numpy as np
import pandas as pd
import torch
import wandb
from PIL import Image
from scipy.ndimage import binary_dilation
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from NCex_common import (
    get_ncex_parser, run_dir, set_seed, write_config, merge_configs, merge_parts, init_wandb, write_alert,
    build_model, build_test_dataset, model_logits, make_cam, release_cam, cam_maps, display_image,
    save_example_grid, CAM_METHODS, COMPARE_SIZE,
)
from NCex_E5_make_data import TREES

EXPERIMENT = 'E5'
TEST_SETS = {'T1': TREES['trainval'], 'T2': TREES['T2'], 'T3': TREES['T3']}
CSV_COLUMNS = ['model', 'test_set', 'image_id', 'label', 'prob', 'explainer',
               'marker_mass', 'pointing_hit', 'chance', 'placed']


def get_args_parser():
    parser = get_ncex_parser('NCex E5 planted-cheat evaluation')
    parser.add_argument('--mode', default='run', choices=['run', 'merge'])
    parser.add_argument('--explainers', default=['gradcamv2', 'gradcam++', 'hirescam'], nargs='+',
                        choices=list(CAM_METHODS.keys()))
    parser.add_argument('--widen_px', default=2, type=float,
                        help='Pointing-game tolerance around the box, in 224-equivalent pixels')
    parser.add_argument('--n_examples', default=10, type=int, help='T2 example heatmaps (half disease, half healthy)')
    parser.add_argument('--example_seed', default=2026, type=int)
    parser.add_argument('--alert_reliance', default=0.10, type=float,
                        help='Cheat is "used" if AUROC(T1) - AUROC(T3) exceeds this')
    parser.add_argument('--alert_mass_x', default=10.0, type=float,
                        help='Map "hides" the cheat if mean marker_mass < alert_mass_x * chance')
    return parser


# ----------------------------------------------------------------------------- marker box in the model frame

class BoxMapper:
    """Marker box -> boolean mask in the model's input frame, via the model's own preprocessing."""

    def __init__(self, transform):
        self.transform = transform
        self._blank = {}

    def _frame(self, arr):
        out = self.transform(Image.fromarray(arr))
        return torch.as_tensor(out).float()[0].numpy()

    def __call__(self, h, w, box):
        if (h, w) not in self._blank:
            self._blank[(h, w)] = self._frame(np.zeros((h, w, 3), dtype=np.uint8))
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        x0, y0, x1, y1 = box
        arr[y0:y1, x0:x1] = 255
        diff = self._frame(arr) - self._blank[(h, w)]
        return diff > 0.5 * diff.max() if diff.max() > 0 else np.zeros_like(diff, dtype=bool)


def box_metrics(heat, box_mask, widen):
    total = float(heat.sum())
    mass = float(heat[box_mask].sum() / total) if total > 0 else np.nan
    wide = binary_dilation(box_mask, structure=np.ones((2 * widen + 1, 2 * widen + 1), dtype=bool)) if widen > 0 else box_mask
    hit = int(wide.flat[int(np.argmax(heat))]) if total > 0 else np.nan  # all-zero map: undefined, not a miss
    return mass, hit, float(box_mask.mean())


def mask_bbox(mask):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


# ----------------------------------------------------------------------------- main

def predict(model, loader, args):
    probs, ids, labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            logits = model_logits(model, batch[0].to(args.device))
            probs.append(torch.softmax(logits, dim=1)[:, args.disease_class].cpu().numpy())
            labels.append(batch[1].numpy())
            ids += list(batch[3])
    return np.concatenate(probs), np.concatenate(labels), ids


def run(args):
    out_dir = run_dir(args, EXPERIMENT)
    data_csv = os.path.join(out_dir, 'e5_data.csv')
    data_root = os.path.join(out_dir, 'data')
    args.task = f'NCex_E5-eval-{args.model_tag}'
    set_seed(args.seed)
    init_wandb(args, EXPERIMENT)
    write_config(args, out_dir, args.task)
    start = time.time()

    markers = pd.read_csv(os.path.join(out_dir, 'markers.csv'))
    markers = markers[markers['split'] == 'test'].set_index('image_id')
    model, processor, patch_size, transform = build_model(args)
    mapper = BoxMapper(transform)
    widen = int(round(args.widen_px * args.input_size / COMPARE_SIZE))

    rows, summary, datasets = [], {}, {}
    for test_set, tree in TEST_SETS.items():
        dataset = build_test_dataset(args, processor, data_path=data_csv, img_dir=os.path.join(data_root, tree))
        datasets[test_set] = dataset
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=args.pin_mem)
        probs, labels, ids = predict(model, loader, args)
        y_true = (labels == args.disease_class).astype(int)
        auroc = float(roc_auc_score(y_true, probs))
        summary[f'{test_set}/auroc'] = auroc
        wandb.log({f'{test_set}/auroc': auroc, f'{test_set}/n': len(ids)})
        print(f'{test_set}: AUROC = {auroc:.4f} (n={len(ids)})')
        if test_set != 'T2':
            for image_id, label, prob in zip(ids, labels, probs):
                rows.append([args.model_tag, test_set, image_id, int(label), float(prob), 'none',
                             np.nan, np.nan, np.nan, int(markers.loc[image_id, 'placed'])])
        else:
            t2_probs = dict(zip(ids, probs))
    summary['cheat_reliance'] = summary['T1/auroc'] - summary['T3/auroc']
    wandb.log({'cheat_reliance': summary['cheat_reliance']})
    wandb.run.summary.update(summary)  # keep the AUROCs even if the heatmap stage fails
    print(f'cheat_reliance = AUROC(T1) - AUROC(T3) = {summary["cheat_reliance"]:.4f}')

    # T2 heatmaps
    t2 = datasets['T2']
    t2_loader = DataLoader(t2, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=args.pin_mem)
    example_ids = pick_examples(markers, args)
    example_maps = {i: [] for i in example_ids}
    example_display, example_boxes = {}, {}
    for explainer in args.explainers:
        cam = make_cam(model, args, patch_size, explainer)
        masses, hits, chances = [], [], []
        for batch in t2_loader:
            images, labels, ids = batch[0].to(args.device), batch[1].numpy(), list(batch[3])
            maps = cam_maps(cam, images, [args.disease_class] * images.shape[0])
            for k, image_id in enumerate(ids):
                m = markers.loc[image_id]
                if m['placed']:
                    box_mask = mapper(int(m['H']), int(m['W']), (int(m['x0']), int(m['y0']), int(m['x1']), int(m['y1'])))
                    if box_mask.shape != maps[k].shape:
                        raise ValueError(f'Box frame {box_mask.shape} != map {maps[k].shape}; check --input_size')
                    mass, hit, chance = box_metrics(maps[k], box_mask, widen)
                    masses.append(mass)
                    hits.append(hit)
                    chances.append(chance)
                else:
                    box_mask, mass, hit, chance = None, np.nan, np.nan, np.nan
                rows.append([args.model_tag, 'T2', image_id, int(labels[k]), float(t2_probs[image_id]), explainer,
                             mass, hit, chance, int(m['placed'])])
                if image_id in example_maps:
                    example_maps[image_id].append(maps[k])
                    example_display[image_id] = display_image(images[k].cpu())
                    example_boxes[image_id] = mask_bbox(box_mask) if box_mask is not None else None
        release_cam(cam)
        summary[f'T2/{explainer}/marker_mass_mean'] = float(np.nanmean(masses)) if masses else np.nan
        summary[f'T2/{explainer}/pointing_hit_rate'] = float(np.nanmean(hits)) if hits else np.nan
        summary[f'T2/{explainer}/degenerate'] = int(np.isnan(masses).sum()) if masses else 0
        summary[f'T2/{explainer}/chance'] = float(np.mean(chances)) if chances else np.nan
        wandb.log({k: v for k, v in summary.items() if k.startswith(f'T2/{explainer}/')})
        print(f'T2 {explainer}: marker_mass = {summary[f"T2/{explainer}/marker_mass_mean"]:.4f} '
              f'(chance {summary[f"T2/{explainer}/chance"]:.4f}), '
              f'pointing_hit_rate = {summary[f"T2/{explainer}/pointing_hit_rate"]:.4f}')

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    os.makedirs(os.path.join(out_dir, 'parts'), exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'parts', f'{args.task}.csv'), index=False)
    wandb.run.summary.update(summary)
    wandb.log({'shortcut_table': wandb.Table(dataframe=df)})

    ex = [i for i in example_ids if example_maps[i]]
    if ex:
        fig_rows = [(f'{os.path.basename(i)}\nlabel={int(markers.loc[i, "label"])}', example_display[i], example_maps[i])
                    for i in ex]
        fig = save_example_grid(fig_rows, ['T2 input'] + list(args.explainers),
                                os.path.join(out_dir, 'examples', f'{args.task}.png'),
                                boxes=[example_boxes[i] for i in ex],
                                title=f'{args.model_tag}: disease-class maps on T2 (white box = marker)')
        wandb.log({'examples': wandb.Image(fig)})

    hiding = [e for e in args.explainers
              if summary[f'T2/{e}/marker_mass_mean'] < args.alert_mass_x * summary[f'T2/{e}/chance']]
    if summary['cheat_reliance'] > args.alert_reliance and hiding:
        write_alert(out_dir, args.task, f'E5: {args.model_tag} uses the marker but its map hides it',
                    f'cheat_reliance={summary["cheat_reliance"]:.3f} > {args.alert_reliance}; '
                    f'marker_mass < {args.alert_mass_x} x chance for {hiding}. Report to Yu immediately.')
    print(f'E5 {args.task} finished in {time.time() - start:.0f}s')
    wandb.finish()


def pick_examples(markers, args):
    """Same T2 examples for every model: placed test images, half disease, half healthy."""
    placed = markers[markers['placed'] == 1]
    rng = np.random.RandomState(args.example_seed)
    picks = []
    for is_disease in (True, False):
        ids = sorted(placed[(placed['label'] == args.disease_class) == is_disease].index)
        n = args.n_examples // 2 if is_disease else args.n_examples - args.n_examples // 2
        picks += list(rng.permutation(ids)[:n])
    return picks


def merge(args):
    out_dir = run_dir(args, EXPERIMENT)
    args.task = 'NCex_E5-merge'
    init_wandb(args, EXPERIMENT)
    df = merge_parts(out_dir, 'shortcut.csv')
    merge_configs(out_dir)
    t2 = df[df['test_set'] == 'T2']
    t2_means = t2.groupby(['model', 'explainer'])[['marker_mass', 'pointing_hit', 'chance']].mean()
    print(t2_means.to_string())
    auc_rows = []
    for model, g in df.groupby('model'):
        aucs = {}
        for ts in ['T1', 'T2', 'T3']:
            s = g[g['test_set'] == ts].drop_duplicates(subset='image_id')  # T2 has one row per explainer
            aucs[ts] = roc_auc_score((s['label'] == args.disease_class).astype(int), s['prob'])
        auc_rows.append([model, aucs['T1'], aucs['T2'], aucs['T3'], aucs['T1'] - aucs['T3']])
        print(f'{model}: AUROC T1={aucs["T1"]:.4f} T2={aucs["T2"]:.4f} T3={aucs["T3"]:.4f} '
              f'cheat_reliance={aucs["T1"] - aucs["T3"]:.4f}')
    auc_df = pd.DataFrame(auc_rows, columns=['model', 'auroc_T1', 'auroc_T2', 'auroc_T3', 'cheat_reliance'])
    summary = {f'{r.model}/{k}': getattr(r, k) for r in auc_df.itertuples()
               for k in ['auroc_T1', 'auroc_T2', 'auroc_T3', 'cheat_reliance']}
    summary.update({f'{model}/T2/{explainer}/{k}': v for (model, explainer), row in t2_means.iterrows()
                    for k, v in row.items()})
    wandb.run.summary.update(summary)
    wandb.log({'auroc_table': wandb.Table(dataframe=auc_df),
               't2_marker_table': wandb.Table(dataframe=t2_means.reset_index())})
    wandb.finish()


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if args.mode == 'run':
        run(args)
    else:
        merge(args)
