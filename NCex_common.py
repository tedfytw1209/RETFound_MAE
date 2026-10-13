"""Shared helpers for the NCex revision experiments (E4 sanity checks, E5 planted-cheat test).

Model building, checkpoint loading and the CAM wrapper are reused from main_XAI_evaluation.py and
baselines/GradCAM_v2.py so that every model, hook layer and preprocessing step matches the paper runs.
See docs/plans/2026-10-13-ncex-e4-e5-design.md.
"""
import argparse
import glob
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb
from pytorch_grad_cam import GradCAM as GradCAMv2, GradCAMPlusPlus, HiResCAM

from main_XAI_evaluation import get_args_parser as get_xai_args_parser, get_model
from baselines.GradCAM_v2 import PytorchCAM
from util.datasets import build_dataset, build_transform

CAM_METHODS = {'gradcamv2': GradCAMv2, 'gradcam++': GradCAMPlusPlus, 'hirescam': HiResCAM}
REPO_DIR = os.path.dirname(os.path.realpath(__file__))
COMPARE_SIZE = 224  # common grid for comparing maps across models with different input sizes


def get_ncex_parser(description):
    """main_XAI_evaluation's parser (all model / SMP flags) plus the NCex experiment flags."""
    parser = argparse.ArgumentParser(description, parents=[get_xai_args_parser()])
    parser.add_argument('--run_id', default='debug', type=str,
                        help='Experiment run id; outputs go to <results_root>/NCex/<experiment>/<run_id>/')
    parser.add_argument('--results_root', default='/orange/ruogu.fang/tienyuchang/RETfound_XAI_results', type=str,
                        help='Root directory for NCex results')
    parser.add_argument('--model_tag', default='', type=str,
                        help='Short model name written to the CSV (e.g. SAIL, ResNet-50)')
    parser.add_argument('--dataset_tag', default='', type=str,
                        help='Short dataset name written to the CSV (e.g. UF_DME, OCTDL_ERM)')
    parser.add_argument('--disease_class', default=1, type=int,
                        help='Class index of the disease class (healthy is the other class)')
    parser.add_argument('--wandb_project', default='NCex_SAIL_revision', type=str,
                        help='wandb project name')
    return parser


def run_dir(args, experiment):
    path = os.path.join(args.results_root, 'NCex', experiment, args.run_id)
    os.makedirs(path, exist_ok=True)
    return path


def git_commit():
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_DIR, text=True).strip()
        dirty = subprocess.call(['git', 'diff', '--quiet', 'HEAD'], cwd=REPO_DIR) != 0
        return commit + ('-dirty' if dirty else '')
    except Exception:
        return 'unknown'


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def write_config(args, out_dir, name, extra=None):
    """Write one YAML record (args, seed, git commit, checkpoint) for this call to <out_dir>/configs/<name>.yaml."""
    os.makedirs(os.path.join(out_dir, 'configs'), exist_ok=True)
    record = {'name': name, 'git_commit': git_commit(), 'seed': args.seed,
              'checkpoint': getattr(args, 'resume', None), 'args': vars(args)}
    if extra:
        record.update(extra)
    path = os.path.join(out_dir, 'configs', f'{name}.yaml')
    with open(path, 'w') as f:
        yaml.safe_dump(record, f, sort_keys=False, default_flow_style=False)
    return path


def merge_configs(out_dir):
    """Concatenate configs/*.yaml into config.yaml (one entry per model call)."""
    records = []
    for path in sorted(glob.glob(os.path.join(out_dir, 'configs', '*.yaml'))):
        with open(path) as f:
            records.append(yaml.safe_load(f))
    with open(os.path.join(out_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(records, f, sort_keys=False, default_flow_style=False)
    return records


def merge_parts(out_dir, csv_name):
    """Concatenate parts/*.csv into <csv_name>."""
    parts = sorted(glob.glob(os.path.join(out_dir, 'parts', '*.csv')))
    if not parts:
        raise FileNotFoundError(f'No part CSVs found in {os.path.join(out_dir, "parts")}')
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df.to_csv(os.path.join(out_dir, csv_name), index=False)
    print(f'Merged {len(parts)} parts ({len(df)} rows) into {os.path.join(out_dir, csv_name)}')
    return df


def init_wandb(args, experiment):
    """Same pattern as main_finetune.py: group = experiment run, name = this call's task."""
    config = dict(vars(args))
    config.update({'git_commit': git_commit(), 'experiment': experiment})
    wandb.init(project=args.wandb_project, group=f'{experiment}-{args.run_id}', name=args.task, config=config)


def write_alert(out_dir, name, title, text):
    path = os.path.join(out_dir, f'ALERT_{name}.txt')
    with open(path, 'w') as f:
        f.write(title + '\n' + text + '\n')
    print(f'\n!!! ALERT: {title}\n{text}\n(see {path})\n')
    try:
        wandb.alert(title=title, text=text, level=wandb.AlertLevel.WARN)
    except Exception as e:
        print(f'[wandb] alert failed: {e}')


# ----------------------------------------------------------------------------- models

def build_model(args):
    """Build the model exactly as main_XAI_evaluation.py does and load the fine-tuned checkpoint."""
    # eval=True skips the RETFound pre-trained download; the fine-tuned checkpoint holds every weight.
    args.eval = True
    model, processor, patch_size = get_model(args)
    if not args.resume or args.resume == '0':
        raise ValueError('Please provide the fine-tuned checkpoint with --resume')
    # our own fine-tuned checkpoints (they pickle the argparse.Namespace of the training run)
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    msg = model.load_state_dict(state_dict, strict=False)
    if msg.missing_keys:
        raise RuntimeError(f'Checkpoint {args.resume} is missing {len(msg.missing_keys)} keys, '
                           f'e.g. {msg.missing_keys[:5]}')
    if msg.unexpected_keys:
        print(f'[build_model] {len(msg.unexpected_keys)} unexpected keys ignored, e.g. {msg.unexpected_keys[:5]}')
    print(f'Resume checkpoint {args.resume}')
    model = model.float().to(args.device).eval()
    transform = processor if processor is not None else build_transform('test', args)
    return model, processor, patch_size, transform


def build_test_dataset(args, processor, data_path=None, img_dir=None):
    """Test split through the same CSV_Dataset_eval / transform path as main_XAI_evaluation.py."""
    saved = args.data_path
    if data_path is not None:
        args.data_path = data_path
    dataset = build_dataset(is_train='test', args=args, k=args.num_k, img_dir=img_dir or args.img_dir,
                            modality=args.modality, transform=processor, eval_mode=True)
    args.data_path = saved
    return dataset


def model_logits(model, images):
    out = model(images)
    if hasattr(out, 'logits'):
        out = out.logits
    elif isinstance(out, dict) and 'logits' in out:
        out = out['logits']
    return out


def make_cam(model, args, patch_size, method='gradcamv2'):
    """Paper hook layers: --target_module/--select_index for SAIL, last block/stage for the baselines."""
    return PytorchCAM(model, model_name=args.model, img_size=args.input_size, patch_size=patch_size,
                      method=CAM_METHODS[method], target_module=args.target_module,
                      select_index=args.select_index, device=args.device)


def release_cam(cam):
    try:
        cam.method.activations_and_grads.release()
    except Exception:
        pass


def cam_maps(cam, images, target_classes):
    """CAM per image for the given class indices; returns (B, input_size, input_size) in [0, 1]."""
    targets = np.asarray(target_classes, dtype=np.int64)
    return cam(images, targets=targets).astype(np.float32)


def to_compare_grid(maps, size=COMPARE_SIZE):
    """Resize maps (B, H, W) to the common grid and re-normalize each to [0, 1] (constant maps become 0)."""
    t = torch.as_tensor(maps, dtype=torch.float32).unsqueeze(1)
    if t.shape[-1] != size or t.shape[-2] != size:
        t = F.interpolate(t, size=(size, size), mode='bilinear', align_corners=False)
    t = t.squeeze(1)
    flat = t.view(t.shape[0], -1)
    lo = flat.min(dim=1)[0].view(-1, 1, 1)
    hi = flat.max(dim=1)[0].view(-1, 1, 1)
    rng = hi - lo
    t = torch.where(rng > 1e-8, (t - lo) / rng.clamp_min(1e-8), torch.zeros_like(t))
    return t.numpy()


def display_image(image_tensor):
    """Model input tensor (C, H, W), any normalization -> grayscale [0, 1] for plotting."""
    img = image_tensor.detach().cpu().float()[0].numpy()
    lo, hi = img.min(), img.max()
    return (img - lo) / (hi - lo + 1e-8)


# ----------------------------------------------------------------------------- figures

def save_example_grid(rows, col_titles, path, boxes=None, title=None):
    """rows: list of (row_title, gray_image, [maps]). Maps are overlaid on one fixed jet [0, 1] scale.

    boxes: optional list (one per row) of (x0, y0, x1, y1) in the row image's pixel frame, drawn on every panel.
    """
    n_rows, n_cols = len(rows), len(col_titles)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.2 * n_rows + 0.6), squeeze=False)
    im = None
    for r, (row_title, gray, maps) in enumerate(rows):
        panels = [None] + list(maps)
        for c in range(n_cols):
            ax = axes[r, c]
            ax.imshow(gray, cmap='gray', vmin=0, vmax=1)
            if c > 0 and panels[c] is not None:
                heat = panels[c]
                if heat.shape != gray.shape:
                    heat = F.interpolate(torch.as_tensor(heat)[None, None], size=gray.shape,
                                         mode='bilinear', align_corners=False)[0, 0].numpy()
                im = ax.imshow(heat, cmap='jet', vmin=0, vmax=1, alpha=0.5)
            if boxes is not None and boxes[r] is not None:
                x0, y0, x1, y1 = boxes[r]
                ax.add_patch(plt.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0,
                                           fill=False, edgecolor='white', linewidth=1.0))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=8)
            if c == 0:
                ax.set_ylabel(row_title, fontsize=7)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
    if title:
        fig.suptitle(title, fontsize=10)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path
