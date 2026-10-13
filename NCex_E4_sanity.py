"""NCex E4: saliency sanity checks (Adebayo et al., NeurIPS 2018) for SAIL and the baselines.

Test A: cascading re-initialization from the output end backwards; each step's Grad-CAM map is compared with
        the trained model's map (Spearman, SSIM on the common 224 grid).
Test B: Grad-CAM for the disease logit vs the healthy logit on the trained model (Spearman, SSIM).

Modes:
  --mode samples : draw the shared scan list for one dataset (run once, before any model job)
  --mode run     : one model x one dataset -> parts/<task>.csv, configs/<task>.yaml, examples/<task>.png
  --mode merge   : parts/*.csv -> sanity.csv, configs/*.yaml -> config.yaml
See docs/plans/2026-10-13-ncex-e4-e5-design.md.
"""
import datetime
import os
import time

import numpy as np
import pandas as pd
import torch
import wandb
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader, Subset

from NCex_common import (
    get_ncex_parser, run_dir, set_seed, write_config, merge_configs, merge_parts, init_wandb, write_alert,
    build_model, build_test_dataset, model_logits, make_cam, release_cam, cam_maps, to_compare_grid,
    display_image, save_example_grid, COMPARE_SIZE,
)

EXPERIMENT = 'E4'
CSV_COLUMNS = ['model', 'dataset', 'image_id', 'test', 'step', 'spearman', 'ssim',
               'label', 'pred', 'target_class', 'degenerate']


def get_args_parser():
    parser = get_ncex_parser('NCex E4 saliency sanity checks')
    parser.add_argument('--mode', default='run', choices=['samples', 'run', 'merge'])
    parser.add_argument('--n_per_class', default=50, type=int,
                        help='Scans per class in the shared sample (50 disease + 50 healthy)')
    parser.add_argument('--sample_seed', default=2026, type=int, help='Seed for drawing the shared scan list')
    parser.add_argument('--reinit_seed', default=0, type=int,
                        help='Re-initialization at step s uses torch.manual_seed(reinit_seed + s)')
    parser.add_argument('--n_examples', default=5, type=int, help='Number of example scans to plot')
    parser.add_argument('--alert_spearman', default=0.5, type=float,
                        help='Alert if SAIL mean Spearman at step 1 or 2 exceeds this value')
    return parser


def samples_path(out_dir, dataset_tag):
    return os.path.join(out_dir, f'samples_{dataset_tag}.csv')


# ----------------------------------------------------------------------------- sampling

def make_samples(args):
    out_dir = run_dir(args, EXPERIMENT)
    dataset = build_test_dataset(args, processor=None)
    ann = dataset.annotations
    df = pd.DataFrame({
        'image_id': [s[0] for s in dataset.samples],
        'label': list(dataset.targets),
        'patient_id': ann['patient_id'].astype(str).values if 'patient_id' in ann.columns
        else np.arange(len(ann)).astype(str),
    })
    print(f'class_to_idx: {dataset.class_to_idx}  (disease_class={args.disease_class})')
    rng = np.random.RandomState(args.sample_seed)
    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    if 'patient_id' in ann.columns:
        df = df.drop_duplicates(subset='patient_id', keep='first').reset_index(drop=True)
        print(f'One scan per patient: {len(df)} candidate scans')
    picked = []
    for label in sorted(df['label'].unique()):
        cls = df[df['label'] == label]
        if len(cls) < args.n_per_class:
            print(f'WARNING: class {label} has only {len(cls)} scans (< {args.n_per_class}); using all')
        picked.append(cls.head(args.n_per_class))
    out = pd.concat(picked, ignore_index=True)
    path = samples_path(out_dir, args.dataset_tag)
    out.to_csv(path, index=False)
    print(f'Saved {len(out)} scans ({out["label"].value_counts().to_dict()}) to {path}')


# ----------------------------------------------------------------------------- re-initialization

def _matches(name, prefixes):
    return any(p == '' or name == p or name.startswith(p + '.') for p in prefixes)


def _last_blocks(prefix, n, thirds):
    """Names of the blocks in the last `thirds`/3 of an n-block backbone (thirds=1 -> last third)."""
    return [f'{prefix}.{i}' for i in range(n - (n * thirds) // 3, n)]


def cascade_groups(model, model_name):
    """Modules newly re-initialized at steps 1-4 (cumulative; '' = everything not yet reset)."""
    name = model_name.lower()
    if name.startswith('smp'):
        steps = [['head.classifier'],
                 ['head.enc_align', 'head.dec_align', 'head.alpha_logit'],
                 ['seg_model.decoder', 'seg_model.segmentation_head'],
                 ['']]
    elif 'resnet' in name:
        steps = [['classifier'], ['resnet.encoder.stages.3'], ['resnet.encoder.stages.2'], ['']]
    elif 'retfound' in name:
        n = len(model.blocks)
        last1 = _last_blocks('blocks', n, 1)
        steps = [['head', 'fc_norm'], last1, [b for b in _last_blocks('blocks', n, 2) if b not in last1], ['']]
    elif 'efficientnet' in name:
        n = len(model.blocks)
        last1 = _last_blocks('blocks', n, 1)
        steps = [['classifier'], ['conv_head', 'bn2'] + last1,
                 [b for b in _last_blocks('blocks', n, 2) if b not in last1], ['']]
    elif 'vit' in name:
        n = len(model.vit.encoder.layer)
        last1 = _last_blocks('vit.encoder.layer', n, 1)
        steps = [['classifier', 'vit.layernorm'], last1,
                 [b for b in _last_blocks('vit.encoder.layer', n, 2) if b not in last1], ['']]
    else:
        raise ValueError(f'No re-initialization schedule for model {model_name}')

    module_names = [n for n, _ in model.named_modules()]
    param_names = [n for n, _ in model.named_parameters()]
    for step_prefixes in steps:
        for p in step_prefixes:
            if p and not any(_matches(n, [p]) for n in module_names + param_names):
                raise ValueError(f'Re-init prefix "{p}" matches nothing in {model_name}')
    return steps


def _init_orphan_(full_name, param):
    """Parameters whose module has no reset_parameters (cls/pos tokens, SAIL fusion gate)."""
    with torch.no_grad():
        if full_name.endswith('alpha_logit'):
            param.zero_()  # sigmoid(0) = 0.5, the initial alpha
        else:
            torch.nn.init.trunc_normal_(param, std=0.02)


def reinit_(model, prefixes, exclude, seed):
    """Reset every module / parameter matching `prefixes` and not matching `exclude`. Returns reset param names."""
    torch.manual_seed(seed)
    touched = set()
    for mname, module in model.named_modules():
        if not _matches(mname, prefixes) or (mname and _matches(mname, exclude)):
            continue
        has_reset = callable(getattr(module, 'reset_parameters', None))
        if has_reset:
            module.reset_parameters()
        for pname, param in module.named_parameters(recurse=False):
            full = f'{mname}.{pname}' if mname else pname
            if _matches(full, exclude):
                continue
            if not has_reset:
                _init_orphan_(full, param)
            touched.add(full)
    for full, param in model.named_parameters():
        if full not in touched and _matches(full, prefixes) and not _matches(full, exclude):
            _init_orphan_(full, param)
            touched.add(full)
    return touched


def changed_fraction(model, state0):
    """Fraction of parameter tensors that differ from the trained weights."""
    params = dict(model.named_parameters())
    changed = sum(not torch.equal(p.detach().cpu(), state0[n]) for n, p in params.items())
    return changed / max(len(params), 1)


# ----------------------------------------------------------------------------- maps and metrics

def compute_maps(model, cam, loader, args, test_b=False):
    """Grad-CAM for the true label (and, if test_b, for the disease and healthy logits) on the 224 grid."""
    out = {'label_map': [], 'pred': [], 'disease_map': [], 'healthy_map': [], 'display': []}
    healthy_class = 1 - args.disease_class
    for batch in loader:
        images, target = batch[0].to(args.device), batch[1]
        with torch.no_grad():
            out['pred'].append(model_logits(model, images).argmax(dim=1).cpu().numpy())
        out['label_map'].append(to_compare_grid(cam_maps(cam, images, target.numpy())))
        if test_b:
            bs = images.shape[0]
            out['disease_map'].append(to_compare_grid(cam_maps(cam, images, [args.disease_class] * bs)))
            out['healthy_map'].append(to_compare_grid(cam_maps(cam, images, [healthy_class] * bs)))
            out['display'].extend(to_compare_grid(np.stack([display_image(im) for im in images.cpu()])))
    return {k: (np.concatenate(v) if v and k != 'display' else v) for k, v in out.items()}


def similarity(a, b):
    """Spearman (NaN if either map is constant) and SSIM between two [0, 1] maps."""
    degenerate = int(a.std() < 1e-8 or b.std() < 1e-8)
    rho = np.nan if degenerate else float(spearmanr(a.ravel(), b.ravel()).correlation)
    ssim = float(structural_similarity(a, b, data_range=1.0))
    return rho, ssim, degenerate


def example_indices(labels, n):
    """Alternate healthy / disease scans from the front of the sample list."""
    by_class = {c: [i for i, l in enumerate(labels) if l == c] for c in sorted(set(labels))}
    picks, k = [], 0
    while len(picks) < min(n, len(labels)):
        for c in by_class:
            if k < len(by_class[c]) and len(picks) < n:
                picks.append(by_class[c][k])
        k += 1
    return picks


# ----------------------------------------------------------------------------- main

def run(args):
    out_dir = run_dir(args, EXPERIMENT)
    args.task = f'NCex_E4-{args.model_tag}-{args.dataset_tag}'
    set_seed(args.seed)
    init_wandb(args, EXPERIMENT)
    write_config(args, out_dir, args.task)

    model, processor, patch_size, _ = build_model(args)
    dataset = build_test_dataset(args, processor)
    print(f'class_to_idx: {dataset.class_to_idx}  (disease_class={args.disease_class})')
    if args.nb_classes != 2:
        raise ValueError('E4 Test B assumes a binary disease-vs-healthy model (--nb_classes 2)')

    samples = pd.read_csv(samples_path(out_dir, args.dataset_tag))
    id_to_idx = {s[0]: i for i, s in enumerate(dataset.samples)}
    missing = [i for i in samples['image_id'] if i not in id_to_idx]
    if missing:
        raise ValueError(f'{len(missing)} sampled scans not in the test split, e.g. {missing[:3]}')
    indices = [id_to_idx[i] for i in samples['image_id']]
    labels = [int(dataset.targets[i]) for i in indices]
    if labels != samples['label'].astype(int).tolist():
        raise ValueError('Sampled labels do not match the dataset labels (different class_to_idx?)')
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=args.pin_mem)
    image_ids = samples['image_id'].tolist()

    cam = make_cam(model, args, patch_size, 'gradcamv2')
    groups = cascade_groups(model, args.model)
    state0 = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    start = time.time()

    base = compute_maps(model, cam, loader, args, test_b=True)
    rows = []
    for i, image_id in enumerate(image_ids):
        rho, ssim, deg = similarity(base['disease_map'][i], base['healthy_map'][i])
        rows.append([args.model_tag, args.dataset_tag, image_id, 'B', np.nan, rho, ssim,
                     labels[i], int(base['pred'][i]), args.disease_class, deg])
    test_b = pd.DataFrame(rows, columns=CSV_COLUMNS)
    print(f'Test B: mean Spearman(disease, healthy) = {test_b["spearman"].mean():.4f}, '
          f'mean SSIM = {test_b["ssim"].mean():.4f}, degenerate = {test_b["degenerate"].sum()}')
    wandb.log({'testA/spearman_mean': 1.0, 'testA/ssim_mean': 1.0, 'testA/params_changed_frac': 0.0}, step=0)
    wandb.run.summary['testB/spearman_mean'] = float(test_b['spearman'].mean())
    wandb.run.summary['testB/ssim_mean'] = float(test_b['ssim'].mean())
    wandb.run.summary['testB/degenerate'] = int(test_b['degenerate'].sum())

    examples = example_indices(labels, args.n_examples)
    example_maps = {i: [base['label_map'][i]] for i in examples}

    step_means = {}
    done, reset_so_far = [], set()
    for step, prefixes in enumerate(groups, start=1):
        touched = reinit_(model, prefixes, exclude=done, seed=args.reinit_seed + step)
        if touched & reset_so_far:
            raise RuntimeError(f'Step {step} re-reset parameters of earlier steps: {sorted(touched & reset_so_far)[:5]}')
        reset_so_far |= touched
        done += [p for p in prefixes if p]
        frac = changed_fraction(model, state0)
        cur = compute_maps(model, cam, loader, args)
        step_rows = []
        for i, image_id in enumerate(image_ids):
            rho, ssim, deg = similarity(base['label_map'][i], cur['label_map'][i])
            step_rows.append([args.model_tag, args.dataset_tag, image_id, 'A', step, rho, ssim,
                              labels[i], int(cur['pred'][i]), labels[i], deg])
            if i in example_maps:
                example_maps[i].append(cur['label_map'][i])
        step_df = pd.DataFrame(step_rows, columns=CSV_COLUMNS)
        rows += step_rows
        step_means[step] = float(step_df['spearman'].mean())
        print(f'Test A step {step}: reset {len(touched)} params ({frac:.1%} of tensors now differ), '
              f'mean Spearman = {step_df["spearman"].mean():.4f}, mean SSIM = {step_df["ssim"].mean():.4f}, '
              f'degenerate = {step_df["degenerate"].sum()}')
        wandb.log({'testA/spearman_mean': step_df['spearman'].mean(),
                   'testA/ssim_mean': step_df['ssim'].mean(),
                   'testA/degenerate': int(step_df['degenerate'].sum()),
                   'testA/params_changed_frac': frac}, step=step)
    never_reset = [n for n, _ in model.named_parameters() if n not in reset_so_far]
    if never_reset:
        raise RuntimeError(f'{len(never_reset)} parameters were never re-initialized, e.g. {never_reset[:5]}')

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    df['step'] = df['step'].astype('Int64')  # empty for Test B rows
    os.makedirs(os.path.join(out_dir, 'parts'), exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'parts', f'{args.task}.csv'), index=False)
    wandb.log({'sanity_table': wandb.Table(dataframe=df.astype({'step': 'float'}))})

    fig_rows = [(f'{image_ids[i]}\nlabel={labels[i]}', base['display'][i],
                 example_maps[i] + [base['healthy_map'][i]]) for i in examples]
    col_titles = ['input', 'step 0 (trained)'] + [f'step {s}' for s in range(1, len(groups) + 1)] + ['B: healthy logit']
    fig_path = save_example_grid(fig_rows, col_titles, os.path.join(out_dir, 'examples', f'{args.task}.png'),
                                 title=f'{args.model_tag} / {args.dataset_tag}: Grad-CAM under cascading re-init '
                                       f'({COMPARE_SIZE}x{COMPARE_SIZE} grid)')
    wandb.log({'examples': wandb.Image(fig_path)})

    if args.model.startswith('SMP'):
        high = {s: m for s, m in step_means.items() if s in (1, 2) and m > args.alert_spearman}
        if high:
            write_alert(out_dir, args.task, f'E4: {args.model_tag} maps stay similar after re-init ({args.dataset_tag})',
                        f'Mean Spearman vs trained map: {high} > {args.alert_spearman}. '
                        f'Tell Yu before continuing (plan section 5).')

    release_cam(cam)
    print(f'E4 {args.task} finished in {datetime.timedelta(seconds=int(time.time() - start))}')
    wandb.finish()


def merge(args):
    out_dir = run_dir(args, EXPERIMENT)
    df = merge_parts(out_dir, 'sanity.csv')
    merge_configs(out_dir)
    summary = (df.groupby(['model', 'dataset', 'test', 'step'], dropna=False)[['spearman', 'ssim', 'degenerate']]
               .agg({'spearman': 'mean', 'ssim': 'mean', 'degenerate': 'sum'}))
    print(summary.to_string())


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if args.mode == 'samples':
        make_samples(args)
    elif args.mode == 'run':
        run(args)
    else:
        merge(args)
