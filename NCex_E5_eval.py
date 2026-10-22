"""NCex E5 evaluation: does a model trained on marker-contaminated UF DME use the marker (in the retina or in the
background, see NCex_E5_make_data.py --marker_region), and does its map identify it?

T1 (clean), T2 (all marked) and T3 (healthy marked) are the same test images from NCex_E5_make_data.py.
  cheat_reliance = AUROC(T1) - AUROC(T3)
  On T2 (marked) and on T1 (clean control, same box location), for every explainer (disease-class map at the
  paper hook layer):
    marker_mass   = map mass inside the marker box / total map mass
    pointing_hit  = hottest pixel inside the marker box widened by `widen_px` (224-equivalent pixels)
    marker_pct    = rank of the box's map mass among every same-size box that lies fully inside the marker region
                    (every pixel a marker could occupy: retina or vitreous) and does not overlap the marker
                    (0.5 = no preference, 1 = the marker is the hottest spot of its region)
    retina_mass   = map mass inside the retina / total map mass
    chance        = box area / frame area;  chance_retina = box area / retina area;
    chance_region = box area / marker-region area
  The marker "is identified" when T2 beats chance and beats the T1 control at the same location.
The marker box, the retina and the marker region are located in each model's input frame by passing mask images through that
model's own preprocessing, so center-crop (SAIL, RETFound, ResNet) and stretch (ViT, EfficientNet) are handled alike.

Modes: --mode run (one model) | --mode merge (parts/*.csv -> shortcut.csv + summaries, configs -> config.yaml).
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
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from NCex_common import (
    get_ncex_parser, run_dir, set_seed, write_config, merge_configs, merge_parts, init_wandb,
    build_model, build_test_dataset, model_logits, make_cam, release_cam, cam_maps, display_image,
    save_example_grid, CAM_METHODS, COMPARE_SIZE,
)
from NCex_E5_make_data import TREES, LABEL_TREE, REGION_TREE
from util.datasets import retry_io

EXPERIMENT = 'E5'
TEST_SETS = {'T1': TREES['trainval'], 'T2': TREES['T2'], 'T3': TREES['T3']}
CAM_SETS = ['T2', 'T1']  # T1 = clean control at the same box location
METRICS = ['marker_mass', 'pointing_hit', 'marker_pct', 'retina_mass', 'chance', 'chance_retina', 'chance_region']
CSV_COLUMNS = ['model', 'test_set', 'image_id', 'label', 'prob', 'explainer'] + METRICS + ['placed', 'layer']


def get_args_parser():
    parser = get_ncex_parser('NCex E5 planted-cheat evaluation')
    parser.add_argument('--mode', default='run', choices=['run', 'merge'])
    parser.add_argument('--explainers', default=['gradcamv2', 'gradcam++', 'hirescam'], nargs='+',
                        choices=list(CAM_METHODS.keys()))
    parser.add_argument('--widen_px', default=2, type=float,
                        help='Pointing-game tolerance around the box, in 224-equivalent pixels')
    parser.add_argument('--n_examples', default=10, type=int, help='Example heatmaps (half disease, half healthy)')
    parser.add_argument('--example_seed', default=2026, type=int)
    parser.add_argument('--reference_model', default='SAIL', type=str,
                        help='merge: model compared pairwise (per image) with every other model')
    return parser


# ----------------------------------------------------------------------------- masks in the model frame

class FrameMapper:
    """Raw-image binary mask -> boolean mask in the model's input frame, via the model's own preprocessing."""

    def __init__(self, transform):
        self.transform = transform
        self._blank = {}

    def _frame(self, arr):
        out = self.transform(Image.fromarray(arr))
        return torch.as_tensor(out).float()[0].numpy()

    def __call__(self, mask):
        h, w = mask.shape
        if (h, w) not in self._blank:
            self._blank[(h, w)] = self._frame(np.zeros((h, w, 3), dtype=np.uint8))
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[mask] = 255
        diff = self._frame(arr) - self._blank[(h, w)]
        return diff > 0.5 * diff.max() if diff.max() > 0 else np.zeros_like(diff, dtype=bool)

    def box(self, h, w, box):
        mask = np.zeros((h, w), dtype=bool)
        x0, y0, x1, y1 = box
        mask[y0:y1, x0:x1] = True
        return self(mask)


def load_mask(root, image_id):
    """Non-zero pixels of a uint8 PNG written by NCex_E5_make_data.py (layer labels or marker region)."""
    path = os.path.join(root, image_id)
    return retry_io(lambda: np.array(Image.open(path)), what=path) > 0


def mask_bbox(mask):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _integral(a):
    return np.pad(np.asarray(a, dtype=np.float64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))


def _box_sums(integral, Y, X, bh, bw):
    return integral[Y + bh, X + bw] - integral[Y, X + bw] - integral[Y + bh, X] + integral[Y, X]


def marker_percentile(heat, bbox, region):
    """Rank of the marker box's map mass among all same-size boxes fully inside the region that do not overlap it."""
    x0, y0, x1, y1 = bbox
    bh, bw = y1 - y0, x1 - x0
    h, w = heat.shape
    Y, X = np.meshgrid(np.arange(h - bh + 1), np.arange(w - bw + 1), indexing='ij')
    sums = _box_sums(_integral(heat), Y, X, bh, bw)
    outside = _box_sums(_integral(~region), Y, X, bh, bw)
    ok = (outside == 0) & ~((np.abs(Y - y0) < bh) & (np.abs(X - x0) < bw))
    if not ok.any():
        return np.nan
    cand, marker = sums[ok], sums[y0, x0]
    tol = 1e-9 * max(1.0, abs(marker))
    return float(((cand < marker - tol).sum() + 0.5 * (np.abs(cand - marker) <= tol).sum()) / cand.size)


def box_metrics(heat, box_mask, retina, region, widen):
    """All METRICS for one map. An all-zero map gives NaN for the map-based metrics (undefined, not a miss)."""
    chance = float(box_mask.mean())
    chance_retina = float(box_mask.sum() / retina.sum()) if retina.any() else np.nan
    chance_region = float(box_mask.sum() / region.sum()) if region.any() else np.nan
    total = float(heat.sum())
    if total <= 0:
        return {'marker_mass': np.nan, 'pointing_hit': np.nan, 'marker_pct': np.nan, 'retina_mass': np.nan,
                'chance': chance, 'chance_retina': chance_retina, 'chance_region': chance_region}
    wide = binary_dilation(box_mask, structure=np.ones((2 * widen + 1, 2 * widen + 1), dtype=bool)) if widen > 0 else box_mask
    return {'marker_mass': float(heat[box_mask].sum() / total),
            'pointing_hit': int(wide.flat[int(np.argmax(heat))]),
            'marker_pct': marker_percentile(heat, mask_bbox(box_mask), region),
            'retina_mass': float(heat[retina].sum() / total),
            'chance': chance, 'chance_retina': chance_retina, 'chance_region': chance_region}


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


def nanmean(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(values)) if np.isfinite(values).any() else np.nan


def run(args):
    out_dir = run_dir(args, EXPERIMENT)
    data_csv = os.path.join(out_dir, 'e5_data.csv')
    data_root = os.path.join(out_dir, 'data')
    label_root = os.path.join(data_root, LABEL_TREE)
    region_root = os.path.join(data_root, REGION_TREE)
    args.task = f'NCex_E5-eval-{args.model_tag}'
    set_seed(args.seed)
    init_wandb(args, EXPERIMENT)
    write_config(args, out_dir, args.task)
    start = time.time()

    markers = pd.read_csv(os.path.join(out_dir, 'markers.csv'))
    markers = markers[markers['split'] == 'test'].set_index('image_id')
    regions = markers['marker_region'].unique().tolist()
    if len(regions) != 1:
        raise ValueError(f'markers.csv mixes marker regions {regions}')
    args.marker_region = regions[0]
    print(f'marker region: {args.marker_region}')
    model, processor, patch_size, transform = build_model(args)
    mapper = FrameMapper(transform)
    widen = int(round(args.widen_px * args.input_size / COMPARE_SIZE))

    rows, summary, datasets, probs_by_set = [], {}, {}, {}
    for test_set, tree in TEST_SETS.items():
        dataset = build_test_dataset(args, processor, data_path=data_csv, img_dir=os.path.join(data_root, tree))
        datasets[test_set] = dataset
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=args.pin_mem)
        probs, labels, ids = predict(model, loader, args)
        probs_by_set[test_set] = dict(zip(ids, probs))
        y_true = (labels == args.disease_class).astype(int)
        auroc = float(roc_auc_score(y_true, probs))
        summary[f'{test_set}/auroc'] = auroc
        wandb.log({f'{test_set}/auroc': auroc, f'{test_set}/n': len(ids)})
        print(f'{test_set}: AUROC = {auroc:.4f} (n={len(ids)})')
        if test_set == 'T3':
            for image_id, label, prob in zip(ids, labels, probs):
                m = markers.loc[image_id]
                rows.append([args.model_tag, test_set, image_id, int(label), float(prob), 'none']
                            + [np.nan] * len(METRICS) + [int(m['placed']), int(m['layer'])])
    summary['cheat_reliance'] = summary['T1/auroc'] - summary['T3/auroc']
    wandb.log({'cheat_reliance': summary['cheat_reliance']})
    wandb.run.summary.update(summary)  # keep the AUROCs even if the heatmap stage fails
    print(f'cheat_reliance = AUROC(T1) - AUROC(T3) = {summary["cheat_reliance"]:.4f}')

    # heatmaps on T2 (marked) and T1 (clean control); the box and retina masks are the same for both
    frame_cache = {}

    def frame_masks(image_id):
        if image_id not in frame_cache:
            m = markers.loc[image_id]
            box = mapper.box(int(m['H']), int(m['W']), (int(m['x0']), int(m['y0']), int(m['x1']), int(m['y1'])))
            retina = mapper(load_mask(label_root, image_id))
            region = mapper(load_mask(region_root, image_id))
            frame_cache[image_id] = (box.shape, [np.packbits(a) for a in (box, retina, region)])
        shape, packed = frame_cache[image_id]
        n = shape[0] * shape[1]
        return [np.unpackbits(a, count=n).reshape(shape).astype(bool) for a in packed]

    example_ids = pick_examples(markers, args)
    for test_set in CAM_SETS:
        loader = DataLoader(datasets[test_set], batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=args.pin_mem)
        example_maps = {i: [] for i in example_ids}
        example_display, example_boxes = {}, {}
        for explainer in args.explainers:
            cam = make_cam(model, args, patch_size, explainer)
            per_metric = {k: [] for k in METRICS}
            for batch in loader:
                images, labels, ids = batch[0].to(args.device), batch[1].numpy(), list(batch[3])
                maps = cam_maps(cam, images, [args.disease_class] * images.shape[0])
                for k, image_id in enumerate(ids):
                    m = markers.loc[image_id]
                    box_mask = None
                    if m['placed']:
                        box_mask, retina, region = frame_masks(image_id)
                        if box_mask.shape != maps[k].shape:
                            raise ValueError(f'Box frame {box_mask.shape} != map {maps[k].shape}; check --input_size')
                        metrics = box_metrics(maps[k], box_mask, retina, region, widen)
                        for key in METRICS:
                            per_metric[key].append(metrics[key])
                    else:
                        metrics = {key: np.nan for key in METRICS}
                    rows.append([args.model_tag, test_set, image_id, int(labels[k]),
                                 float(probs_by_set[test_set][image_id]), explainer]
                                + [metrics[key] for key in METRICS] + [int(m['placed']), int(m['layer'])])
                    if image_id in example_maps:
                        example_maps[image_id].append(maps[k])
                        example_display[image_id] = display_image(images[k].cpu())
                        example_boxes[image_id] = mask_bbox(box_mask) if box_mask is not None else None
            release_cam(cam)
            prefix = f'{test_set}/{explainer}/'
            summary[prefix + 'marker_mass_mean'] = nanmean(per_metric['marker_mass'])
            summary[prefix + 'pointing_hit_rate'] = nanmean(per_metric['pointing_hit'])
            summary[prefix + 'marker_pct_mean'] = nanmean(per_metric['marker_pct'])
            summary[prefix + 'retina_mass_mean'] = nanmean(per_metric['retina_mass'])
            summary[prefix + 'chance'] = nanmean(per_metric['chance'])
            summary[prefix + 'chance_retina'] = nanmean(per_metric['chance_retina'])
            summary[prefix + 'chance_region'] = nanmean(per_metric['chance_region'])
            summary[prefix + 'degenerate'] = int(np.isnan(np.asarray(per_metric['marker_mass'], dtype=float)).sum())
            wandb.log({k: v for k, v in summary.items() if k.startswith(prefix)})
            print(f'{test_set} {explainer}: marker_mass = {summary[prefix + "marker_mass_mean"]:.4f} '
                  f'(chance {summary[prefix + "chance"]:.4f}, region chance {summary[prefix + "chance_region"]:.4f}), '
                  f'pointing_hit_rate = {summary[prefix + "pointing_hit_rate"]:.4f}, '
                  f'marker_pct = {summary[prefix + "marker_pct_mean"]:.4f}')
        save_examples(args, out_dir, test_set, example_ids, example_maps, example_display, example_boxes, markers)

    for explainer in args.explainers:
        for key in ['marker_mass_mean', 'pointing_hit_rate', 'marker_pct_mean']:
            summary[f'delta/{explainer}/{key}'] = summary[f'T2/{explainer}/{key}'] - summary[f'T1/{explainer}/{key}']
        wandb.log({k: v for k, v in summary.items() if k.startswith(f'delta/{explainer}/')})

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    os.makedirs(os.path.join(out_dir, 'parts'), exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'parts', f'{args.task}.csv'), index=False)
    wandb.run.summary.update(summary)
    wandb.log({'shortcut_table': wandb.Table(dataframe=df)})
    print(f'E5 {args.task} finished in {time.time() - start:.0f}s')
    wandb.finish()


def save_examples(args, out_dir, test_set, example_ids, example_maps, example_display, example_boxes, markers):
    ex = [i for i in example_ids if example_maps[i]]
    if not ex:
        return
    what = ('marked' if test_set == 'T2' else 'clean control') + f', marker in {args.marker_region}'
    fig_rows = [(f'{os.path.basename(i)}\nlabel={int(markers.loc[i, "label"])}', example_display[i], example_maps[i])
                for i in ex]
    fig = save_example_grid(fig_rows, [f'{test_set} input'] + list(args.explainers),
                            os.path.join(out_dir, 'examples', f'{args.task}_{test_set}.png'),
                            boxes=[example_boxes[i] for i in ex],
                            title=f'{args.model_tag}: disease-class maps on {test_set} ({what}; white box = marker)')
    wandb.log({f'examples_{test_set}': wandb.Image(fig)})


def pick_examples(markers, args):
    """Same examples for every model: placed test images, half disease, half healthy."""
    placed = markers[markers['placed'] == 1]
    rng = np.random.RandomState(args.example_seed)
    picks = []
    for is_disease in (True, False):
        ids = sorted(placed[(placed['label'] == args.disease_class) == is_disease].index)
        n = args.n_examples // 2 if is_disease else args.n_examples - args.n_examples // 2
        picks += list(rng.permutation(ids)[:n])
    return picks


# ----------------------------------------------------------------------------- merge

def paired_test(a, b, alternative='two-sided'):
    """Wilcoxon signed-rank on paired finite values; returns (n, median diff, p)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) == 0:
        return 0, np.nan, np.nan
    diff = a - b
    if np.all(diff == 0):
        return len(a), 0.0, 1.0
    return len(a), float(np.median(diff)), float(wilcoxon(a, b, alternative=alternative).pvalue)


def identification_summary(df, disease_class):
    """Per model x explainer x label group: T2 and T1 means, T2 - T1 deltas and the one-sided paired test."""
    cams = df[(df['explainer'] != 'none') & (df['placed'] == 1)].copy()
    cams['group'] = np.where(cams['label'] == disease_class, 'disease', 'healthy')
    rows = []
    for (model, explainer), g in cams.groupby(['model', 'explainer']):
        for group, gg in [('all', g)] + list(g.groupby('group')):
            t2 = gg[gg['test_set'] == 'T2'].set_index('image_id')
            t1 = gg[gg['test_set'] == 'T1'].set_index('image_id').reindex(t2.index)
            row = {'model': model, 'explainer': explainer, 'group': group, 'n': len(t2)}
            for key in ['marker_mass', 'pointing_hit', 'marker_pct', 'retina_mass']:
                row[f'T2_{key}'] = t2[key].mean()
                row[f'T1_{key}'] = t1[key].mean()
                row[f'delta_{key}'] = row[f'T2_{key}'] - row[f'T1_{key}']
            row['chance'] = t2['chance'].mean()
            row['chance_retina'] = t2['chance_retina'].mean()
            row['chance_region'] = t2['chance_region'].mean()
            row['T2_mass_over_chance_region'] = row['T2_marker_mass'] / row['chance_region']
            for key in ['marker_mass', 'marker_pct']:
                _, _, p = paired_test(t2[key], t1[key], alternative='greater')
                row[f'p_T2_gt_T1_{key}'] = p
            rows.append(row)
    return pd.DataFrame(rows)


def reference_comparison(df, reference):
    """Reference model (SAIL) vs every other model, paired per image on T2 marker_pct and on the T2 - T1 delta."""
    cams = df[(df['explainer'] != 'none') & (df['placed'] == 1)]
    if reference not in cams['model'].unique():
        print(f'[merge] reference model {reference} not found; skipping the paired comparison')
        return pd.DataFrame()
    wide = cams.set_index(['image_id', 'explainer', 'model', 'test_set'])[['marker_pct', 'marker_mass']]
    wide = wide[~wide.index.duplicated()].unstack(['model', 'test_set'])  # keeps all-NaN (degenerate) columns
    rows = []
    models = sorted(m for m in cams['model'].unique() if m != reference)
    for explainer in sorted(cams['explainer'].unique()):
        w = wide.xs(explainer, level='explainer')

        def col(key, model, test_set):
            return w[(key, model, test_set)] if (key, model, test_set) in w.columns else pd.Series(np.nan, index=w.index)

        for other in models:
            for key in ['marker_pct', 'marker_mass']:
                ref_t2, oth_t2 = col(key, reference, 'T2'), col(key, other, 'T2')
                ref_d = ref_t2 - col(key, reference, 'T1')
                oth_d = oth_t2 - col(key, other, 'T1')
                n, med, p = paired_test(ref_t2, oth_t2)
                n_d, med_d, p_d = paired_test(ref_d, oth_d)
                rows.append({'explainer': explainer, 'reference': reference, 'other': other, 'metric': key,
                             'n': n, 'median_diff_T2': med, 'p_T2': p,
                             'n_delta': n_d, 'median_diff_delta': med_d, 'p_delta': p_d})
    return pd.DataFrame(rows)


def merge(args):
    out_dir = run_dir(args, EXPERIMENT)
    args.task = 'NCex_E5-merge'
    init_wandb(args, EXPERIMENT)
    df = merge_parts(out_dir, 'shortcut.csv')
    merge_configs(out_dir)
    auc_rows = []
    for model, g in df.groupby('model'):
        aucs = {}
        for ts in ['T1', 'T2', 'T3']:
            s = g[g['test_set'] == ts].drop_duplicates(subset='image_id')  # T1/T2 have one row per explainer
            aucs[ts] = roc_auc_score((s['label'] == args.disease_class).astype(int), s['prob'])
        auc_rows.append([model, aucs['T1'], aucs['T2'], aucs['T3'], aucs['T1'] - aucs['T3']])
        print(f'{model}: AUROC T1={aucs["T1"]:.4f} T2={aucs["T2"]:.4f} T3={aucs["T3"]:.4f} '
              f'cheat_reliance={aucs["T1"] - aucs["T3"]:.4f}')
    auc_df = pd.DataFrame(auc_rows, columns=['model', 'auroc_T1', 'auroc_T2', 'auroc_T3', 'cheat_reliance'])

    ident = identification_summary(df, args.disease_class).merge(auc_df[['model', 'cheat_reliance']], on='model')
    ident.to_csv(os.path.join(out_dir, 'shortcut_summary.csv'), index=False)
    with pd.option_context('display.width', 200, 'display.max_columns', 30):
        print(ident[ident['group'] == 'all'][['model', 'explainer', 'n', 'T2_marker_mass', 'T1_marker_mass',
                                              'chance_region', 'T2_pointing_hit', 'T1_pointing_hit',
                                              'T2_marker_pct', 'T1_marker_pct', 'p_T2_gt_T1_marker_pct',
                                              'cheat_reliance']].to_string(index=False))
    ref = reference_comparison(df, args.reference_model)
    if len(ref):
        ref.to_csv(os.path.join(out_dir, f'{args.reference_model}_vs_baselines.csv'), index=False)
        print(ref.to_string(index=False))

    summary = {f'{r.model}/{k}': getattr(r, k) for r in auc_df.itertuples()
               for k in ['auroc_T1', 'auroc_T2', 'auroc_T3', 'cheat_reliance']}
    for r in ident[ident['group'] == 'all'].itertuples():
        for k in ['T2_marker_mass', 'T2_pointing_hit', 'T2_marker_pct', 'delta_marker_mass', 'delta_marker_pct']:
            summary[f'{r.model}/{r.explainer}/{k}'] = getattr(r, k)
    wandb.run.summary.update(summary)
    tables = {'auroc_table': wandb.Table(dataframe=auc_df), 'identification_table': wandb.Table(dataframe=ident)}
    if len(ref):
        tables['reference_table'] = wandb.Table(dataframe=ref)
    wandb.log(tables)
    wandb.finish()


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if args.mode == 'run':
        run(args)
    else:
        merge(args)
