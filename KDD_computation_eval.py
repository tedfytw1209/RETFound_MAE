#!/usr/bin/env python3
"""
KDD_computation_eval.py
Measure computational overhead (parameters, FLOPs, inference time) for:
  - Baseline models: RETFound_mae, ViT-Base, ResNet-50, EfficientNet-B4
  - Proposed SMP models: enc / dec / fuse  (grid over enc_idx × fusion_dim)

Pretrained weights are loaded exactly as in main_XAI_evaluation.py:
  - RETFound_mae : HuggingFace hub  (YukunZhou/{finetune})
  - ViT-Base     : from_pretrained('google/vit-base-patch16-224-in21k')
  - ResNet-50    : from_pretrained('microsoft/resnet-50')
  - EffNetB4     : timm pretrained=True
  - SMP          : pretrained_seg_ckpt  (+ optional --resume checkpoint)

Overhead columns show the delta relative to the SMP-enc baseline so the cost
introduced by the decoder and fusion components is explicit.

Outputs a CSV and a printed summary table.
"""
import os
import argparse
import time
import csv
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

os.environ['TIMM_FUSED_ATTN'] = '0'   # same guard as case_study_SMP_layermap_bs.py


# ─────────────────────────────────────────────────────────────────────────────
# Parameter counting
# ─────────────────────────────────────────────────────────────────────────────

def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# FLOPs  (fvcore → ptflops fallback)
# ─────────────────────────────────────────────────────────────────────────────

def try_fvcore_flops(model, dummy_input):
    try:
        from fvcore.nn import FlopCountAnalysis
        fa = FlopCountAnalysis(model, dummy_input)
        fa.unsupported_ops_warnings(False)
        fa.uncalled_modules_warnings(False)
        return fa.total() / 1e9
    except Exception as e:
        print(f"  [fvcore] {e}")
        return None


def try_ptflops(model, input_shape):
    try:
        from ptflops import get_model_complexity_info
        macs, _ = get_model_complexity_info(
            model, input_shape, as_strings=False,
            print_per_layer_stat=False, verbose=False,
        )
        return macs / 1e9
    except Exception as e:
        print(f"  [ptflops] {e}")
        return None


def measure_flops(model, dummy_input, in_channels, input_size):
    g = try_fvcore_flops(model, dummy_input)
    if g is None:
        g = try_ptflops(model, (in_channels, input_size, input_size))
    return round(g, 4) if g is not None else None


# ─────────────────────────────────────────────────────────────────────────────
# Inference timing
# ─────────────────────────────────────────────────────────────────────────────

def measure_inference_time(model, dummy_input, n_warmup, n_runs, device):
    device = torch.device(device) if isinstance(device, str) else device
    model.eval()
    dummy_input = dummy_input.to(device)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy_input)

    times = []
    if device.type == 'cuda' and torch.cuda.is_available():
        for _ in range(n_runs):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            with torch.no_grad():
                _ = model(dummy_input)
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
    else:
        for _ in range(n_runs):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(dummy_input)
            times.append((time.perf_counter() - t0) * 1e3)

    return float(np.mean(times)), float(np.std(times))


# ─────────────────────────────────────────────────────────────────────────────
# Resume checkpoint loader (shared by all models)
# ─────────────────────────────────────────────────────────────────────────────

def _load_resume(model, resume):
    """Load a finetuned checkpoint into model.

    Matches build_and_load_model_main_style() in case_study_SMP_layermap_bs.py:
      - weights_only=False  (allows arbitrary checkpoint objects)
      - strict=True         (all keys must match — finetuned ckpt contains full model state)
    """
    if not resume or not os.path.exists(resume):
        return
    print(f"  Loading resume checkpoint: {resume}")
    checkpoint = torch.load(resume, map_location='cpu', weights_only=False)
    checkpoint_model = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    model.load_state_dict(checkpoint_model, strict=True)
    print(f"  Resume checkpoint loaded (strict=True)")


# ─────────────────────────────────────────────────────────────────────────────
# Baseline model builders  — load pretrained weights exactly as
# main_XAI_evaluation.py does, then optionally load a finetuned resume ckpt.
# ─────────────────────────────────────────────────────────────────────────────

def build_retfound_mae(nb_classes, input_size=224, drop_path=0.0,
                       finetune='RETFound_mae_natureOCT', resume=''):
    """RETFound_mae with HuggingFace pretrained weights (YukunZhou/{finetune})."""
    import models_vit as models
    from timm.models.layers import trunc_normal_
    from util.pos_embed import interpolate_pos_embed
    from huggingface_hub import hf_hub_download

    model = models.__dict__['RETFound_mae'](
        img_size=input_size,
        num_classes=nb_classes,
        drop_path_rate=drop_path,
        global_pool=True,
    )
    if finetune:
        print(f"  Downloading pretrained weights: YukunZhou/{finetune}")
        checkpoint_path = hf_hub_download(
            repo_id=f'YukunZhou/{finetune}',
            filename=f'{finetune}.pth',
        )
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        checkpoint_model = checkpoint['model']
        # key normalisations matching main_XAI_evaluation.py
        checkpoint_model = {k.replace("backbone.", ""): v for k, v in checkpoint_model.items()}
        checkpoint_model = {k.replace("mlp.w12.", "mlp.fc1."): v for k, v in checkpoint_model.items()}
        checkpoint_model = {k.replace("mlp.w3.", "mlp.fc2."): v for k, v in checkpoint_model.items()}
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"  Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        interpolate_pos_embed(model, checkpoint_model)
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(f"  Pretrained: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
        trunc_normal_(model.head.weight, std=2e-5)

    _load_resume(model, resume)
    return model


def build_vit_base(nb_classes, input_size=224,
                   finetune='google/vit-base-patch16-224-in21k', resume=''):
    """ViT-Base-patch16-224 from HuggingFace pretrained weights."""
    from transformers import ViTForImageClassification
    id2label = {i: f"class_{i}" for i in range(nb_classes)}
    label2id = {v: k for k, v in id2label.items()}
    model = ViTForImageClassification.from_pretrained(
        finetune,
        image_size=input_size,
        num_labels=nb_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        attn_implementation="eager",
    )
    _load_resume(model, resume)
    return model


def build_resnet50(nb_classes, finetune='microsoft/resnet-50', resume=''):
    """ResNet-50 from HuggingFace pretrained weights."""
    from transformers import ResNetForImageClassification
    id2label = {i: f"class_{i}" for i in range(nb_classes)}
    label2id = {v: k for k, v in id2label.items()}
    model = ResNetForImageClassification.from_pretrained(
        finetune,
        num_labels=nb_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    _load_resume(model, resume)
    return model


def build_efficientnet_b4(nb_classes, resume=''):
    """EfficientNet-B4 via timm with ImageNet pretrained weights."""
    import timm
    model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=nb_classes)
    _load_resume(model, resume)
    return model


def baseline_configs(args):
    """Return list of config dicts for all requested baseline models."""
    nb_classes = args.nb_classes
    all_baselines = [
        {
            'model_id':   'RETFound_mae',
            'model_type': 'baseline',
            'mode':       'N/A',
            'enc_idx':    'N/A',
            'dec_idx':    'N/A',
            'fusion_dim': 'N/A',
            'input_size': 224,
            '_build': lambda: build_retfound_mae(
                nb_classes, 224,
                finetune=args.retfound_finetune,
                resume=args.retfound_resume,
            ),
        },
        {
            'model_id':   'ViT-Base-patch16-224',
            'model_type': 'baseline',
            'mode':       'N/A',
            'enc_idx':    'N/A',
            'dec_idx':    'N/A',
            'fusion_dim': 'N/A',
            'input_size': 224,
            '_build': lambda: build_vit_base(
                nb_classes, 224,
                finetune=args.vit_finetune,
                resume=args.vit_resume,
            ),
        },
        {
            'model_id':   'ResNet-50',
            'model_type': 'baseline',
            'mode':       'N/A',
            'enc_idx':    'N/A',
            'dec_idx':    'N/A',
            'fusion_dim': 'N/A',
            'input_size': 224,
            '_build': lambda: build_resnet50(
                nb_classes,
                finetune=args.resnet_finetune,
                resume=args.resnet_resume,
            ),
        },
        {
            'model_id':   'EfficientNet-B4',
            'model_type': 'baseline',
            'mode':       'N/A',
            'enc_idx':    'N/A',
            'dec_idx':    'N/A',
            'fusion_dim': 'N/A',
            'input_size': 380,          # timm default for efficientnet_b4
            '_build': lambda: build_efficientnet_b4(
                nb_classes,
                resume=args.effnet_resume,
            ),
        },
    ]
    requested = set(args.baselines)
    return [c for c in all_baselines if c['model_id'] in requested]


# ─────────────────────────────────────────────────────────────────────────────
# SMP model builder
# ─────────────────────────────────────────────────────────────────────────────

def build_smp_model(args, cfg):
    """Build SMPClassifier.  Per-config keys override global args where present."""
    from SMP.smp_classifier import SMPClassifier
    seg_ckpt = args.seg_ckpt if args.seg_ckpt and os.path.exists(args.seg_ckpt) else None

    fuse_mode       = cfg.get('fuse_mode',       args.smp_fuse_mode)
    use_mask        = cfg.get('use_mask',         args.use_mask)
    learnable_alpha = cfg.get('learnable_alpha',  args.smp_learnable_alpha)
    align           = cfg.get('align',            args.align)
    size_match      = cfg.get('size_match',       args.size_match)
    smp_classifier  = cfg.get('smp_classifier',   args.smp_classifier)
    fusion_dim      = cfg.get('fusion_dim', 0)

    model = SMPClassifier(
        seg_arch=args.seg_arch,
        encoder_name=args.encoder_name,
        encoder_weights=None,       # architecture identical regardless of imagenet init
        in_channels=args.in_channels,
        num_classes=args.nb_classes,
        seg_classes=args.seg_classes,
        seg_activation=args.seg_activation,
        mode=cfg['mode'],
        fuse_mode=fuse_mode,
        fusion_dim=None if fusion_dim == 0 else fusion_dim,
        learnable_alpha=learnable_alpha,
        pretrained_seg_ckpt=seg_ckpt,
        alpha=cfg.get('alpha', args.alpha),
        dropout=args.dropout,
        size_match=size_match,
        use_mask=use_mask,
        enc_idx=cfg['enc_idx'],
        dec_idx=cfg['dec_idx'],
        smp_classifier=smp_classifier,
        align=align,
    )

    # Load finetuned resume checkpoint if provided for this config
    resume = cfg.get('resume', '')
    _load_resume(model, resume)

    return model


def extract_alpha(model, cfg):
    """Return alpha value(s) from the fusion head, or None if not applicable."""
    if cfg.get('mode') != 'fuse':
        return None
    head = getattr(model, 'head', None)
    if head is None or not hasattr(head, 'get_alpha_stats'):
        return None
    stats = head.get_alpha_stats()
    if stats is None:
        return None
    # scalar alpha → single value; channel/spatial → return mean
    val = stats.get('alpha', stats.get('alpha_mean'))
    return round(val, 6) if val is not None else None


def smp_component_params(model):
    """Parameter counts for encoder / decoder / seg-head / classifier-head."""
    r = {'encoder_params': 0, 'decoder_params': 0,
         'seghead_params': 0, 'head_params': 0, 'seg_model_params': 0}
    if hasattr(model, 'seg_model'):
        r['seg_model_params'] = count_params(model.seg_model)
        for name, mod in model.seg_model.named_children():
            if name == 'encoder':
                r['encoder_params'] = count_params(mod)
            elif name == 'decoder':
                r['decoder_params'] = count_params(mod)
            elif name == 'segmentation_head':
                r['seghead_params'] = count_params(mod)
    if hasattr(model, 'head'):
        r['head_params'] = count_params(model.head)
    return r


def smp_configs(args):
    """
    Return SMP configurations to evaluate.

    Fixed configs mirror the exact models in run_xai_layermap_multirun.sh lines 30-34.
    If --grid is set, the (enc_idx × fusion_dim) sweep is appended too.
    """
    # Dataset-specific resume paths (indexed by config order: enc, dec, fuse_ws, fuse_mul)
    smp_resumes = args.smp_resumes if args.smp_resumes else ['', '', '', '']
    while len(smp_resumes) < 4:
        smp_resumes.append('')

    cfgs = []

    # ── Line 30: SMP-enc  (enc, no seg_mask, no learnable_alpha) ─────────────
    cfgs.append({
        'model_id':        'SMP-enc',
        'model_type':      'proposed',
        'mode':            'enc',
        'fuse_mode':       'weighted_sum',
        'enc_idx':         -1,
        'dec_idx':         -1,
        'fusion_dim':      0,
        'align':           'pre',
        'size_match':      'decoder_to_encoder',
        'smp_classifier':  'conv',
        'use_mask':        False,
        'learnable_alpha': False,
        'alpha':           0.5,
        'input_size':      512,
        'resume':          smp_resumes[0],
    })

    # ── Line 31: SMP-dec  (dec, no seg_mask, no learnable_alpha) ─────────────
    cfgs.append({
        'model_id':        'SMP-dec',
        'model_type':      'proposed',
        'mode':            'dec',
        'fuse_mode':       'weighted_sum',
        'enc_idx':         -1,
        'dec_idx':         -1,
        'fusion_dim':      0,
        'align':           'pre',
        'size_match':      'decoder_to_encoder',
        'smp_classifier':  'conv',
        'use_mask':        False,
        'learnable_alpha': False,
        'alpha':           0.5,
        'input_size':      512,
        'resume':          smp_resumes[1],
    })

    # ── Line 33: SMP-fuse-weighted_sum  (fuse, seg_mask, learnable_alpha) ────
    cfgs.append({
        'model_id':        'SMP-fuse-weighted_sum-fd9-enc-2dec-1',
        'model_type':      'proposed',
        'mode':            'fuse',
        'fuse_mode':       'weighted_sum',
        'enc_idx':         -2,
        'dec_idx':         -1,
        'fusion_dim':      9,
        'align':           'pre',
        'size_match':      'decoder_to_encoder',
        'smp_classifier':  'conv',
        'use_mask':        True,
        'learnable_alpha': True,
        'alpha':           0.5,
        'input_size':      512,
        'resume':          smp_resumes[2],
    })

    # ── Line 34: SMP-fuse-multiply  (fuse, seg_mask, NO learnable_alpha) ─────
    cfgs.append({
        'model_id':        'SMP-fuse-multiply-fd9-enc-2dec-1',
        'model_type':      'proposed',
        'mode':            'fuse',
        'fuse_mode':       'multiply',
        'enc_idx':         -2,
        'dec_idx':         -1,
        'fusion_dim':      9,
        'align':           'pre',
        'size_match':      'decoder_to_encoder',
        'smp_classifier':  'conv',
        'use_mask':        True,
        'learnable_alpha': False,
        'alpha':           0.5,
        'input_size':      512,
        'resume':          smp_resumes[3],
    })

    # ── Optional grid (--grid flag) ───────────────────────────────────────────
    if args.grid:
        for enc_idx in args.enc_idxs:
            for dec_idx in args.dec_idxs:
                for fusion_dim in args.fusion_dims:
                    cfgs.append({
                        'model_id':        f'SMP-fuse-weighted_sum-fd{fusion_dim}-enc{enc_idx}dec{dec_idx}',
                        'model_type':      'proposed',
                        'mode':            'fuse',
                        'fuse_mode':       'weighted_sum',
                        'enc_idx':         enc_idx,
                        'dec_idx':         dec_idx,
                        'fusion_dim':      fusion_dim,
                        'align':           args.align,
                        'size_match':      args.size_match,
                        'smp_classifier':  args.smp_classifier,
                        'use_mask':        args.use_mask,
                        'learnable_alpha': args.smp_learnable_alpha,
                        'alpha':           args.alpha,
                        'input_size':      args.input_size,
                        'resume':          '',
                    })

    return cfgs


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate one configuration
# ─────────────────────────────────────────────────────────────────────────────

def _empty_row(cfg, err=''):
    row = {k: v for k, v in cfg.items() if not k.startswith('_') and k != 'resume'}
    row.update({
        'total_params': None, 'total_params_M': None,
        'encoder_params_M': None, 'decoder_params_M': None,
        'seghead_params_M': None, 'head_params_M': None, 'seg_model_params_M': None,
        'total_flops_G': None,
        'inference_mean_ms': None, 'inference_std_ms': None,
        'alpha_value': None,
        'overhead_params_M': None, 'overhead_flops_G': None, 'overhead_time_ms': None,
        'error': err,
    })
    return row


def evaluate_config(cfg, args, device):
    row = {k: v for k, v in cfg.items() if not k.startswith('_') and k != 'resume'}
    input_size = cfg['input_size']

    print(f"\n{'='*70}")
    print(f"  {cfg['model_id']}  (input {input_size}×{input_size})")

    # ── Build ────────────────────────────────────────────────────────────────
    try:
        if cfg.get('model_type') == 'baseline':
            model = cfg['_build']()
        else:
            model = build_smp_model(args, cfg)
    except Exception as e:
        print(f"  [ERROR] build failed: {e}")
        return _empty_row(cfg, str(e))

    model.eval().to(device)

    # ── Parameters ───────────────────────────────────────────────────────────
    total_params = count_params(model)
    row['total_params']   = total_params
    row['total_params_M'] = round(total_params / 1e6, 4)

    if cfg.get('model_type') == 'proposed':
        comp = smp_component_params(model)
        row['encoder_params_M']   = round(comp['encoder_params']   / 1e6, 4)
        row['decoder_params_M']   = round(comp['decoder_params']   / 1e6, 4)
        row['seghead_params_M']   = round(comp['seghead_params']   / 1e6, 4)
        row['head_params_M']      = round(comp['head_params']      / 1e6, 4)
        row['seg_model_params_M'] = round(comp['seg_model_params'] / 1e6, 4)
    else:
        row['encoder_params_M'] = row['decoder_params_M'] = None
        row['seghead_params_M'] = row['head_params_M'] = row['seg_model_params_M'] = None

    # ── FLOPs ────────────────────────────────────────────────────────────────
    if args.alpha_only:
        row['total_flops_G'] = None
    else:
        dummy = torch.randn(args.batch_size, 3, input_size, input_size).to(device)
        row['total_flops_G'] = measure_flops(model, dummy, 3, input_size)

    # ── Inference time ───────────────────────────────────────────────────────
    if args.alpha_only:
        row['inference_mean_ms'] = row['inference_std_ms'] = None
    else:
        try:
            mean_ms, std_ms = measure_inference_time(
                model, dummy,
                n_warmup=args.n_warmup,
                n_runs=args.n_runs,
                device=str(device),
            )
            row['inference_mean_ms'] = round(mean_ms, 4)
            row['inference_std_ms']  = round(std_ms,  4)
        except Exception as e:
            print(f"  [ERROR] timing failed: {e}")
            row['inference_mean_ms'] = row['inference_std_ms'] = None

    row['alpha_value'] = extract_alpha(model, cfg)

    row['error'] = ''
    print(
        f"  params={total_params/1e6:.3f}M  "
        f"flops={row['total_flops_G']}G  "
        f"time={row['inference_mean_ms']}±{row['inference_std_ms']}ms  "
        f"alpha={row['alpha_value']}"
    )

    del model
    torch.cuda.empty_cache()
    return row


# ─────────────────────────────────────────────────────────────────────────────
# Overhead columns (delta vs SMP-enc baseline)
# ─────────────────────────────────────────────────────────────────────────────

def add_overhead_columns(rows):
    """Add +params / +flops / +time columns relative to the SMP-enc baseline."""
    baseline = next((r for r in rows if r.get('model_id') == 'SMP-enc'), None)
    if baseline is None:
        return rows

    bp = baseline.get('total_params')
    bf = baseline.get('total_flops_G')
    bt = baseline.get('inference_mean_ms')

    for r in rows:
        if r.get('model_type') == 'proposed':
            r['overhead_params_M'] = (
                round((r['total_params'] - bp) / 1e6, 4)
                if (bp is not None and r.get('total_params') is not None) else None
            )
            r['overhead_flops_G'] = (
                round(r['total_flops_G'] - bf, 4)
                if (bf is not None and r.get('total_flops_G') is not None) else None
            )
            r['overhead_time_ms'] = (
                round(r['inference_mean_ms'] - bt, 4)
                if (bt is not None and r.get('inference_mean_ms') is not None) else None
            )
        else:
            r['overhead_params_M'] = None
            r['overhead_flops_G']  = None
            r['overhead_time_ms']  = None

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def get_args_parser():
    p = argparse.ArgumentParser(
        description='Measure parameter / FLOPs / inference-time overhead'
    )
    # ── SMP architecture ──────────────────────────────────────────────────────
    p.add_argument('--seg_arch',       type=str,   default='Unet')
    p.add_argument('--encoder_name',   type=str,   default='resnet50')
    p.add_argument('--in_channels',    type=int,   default=3)
    p.add_argument('--nb_classes',     type=int,   default=2)
    p.add_argument('--seg_classes',    type=int,   default=9)
    p.add_argument('--seg_activation', type=str,   default='softmax')
    p.add_argument('--dropout',        type=float, default=0.0)
    p.add_argument('--input_size',     type=int,   default=512,
                   help='Input size for SMP models (baselines use their own native sizes)')
    p.add_argument('--smp_fuse_mode',  type=str,   default='weighted_sum',
                   choices=['weighted_sum','add','channel_merge','channel_multiply','multiply'])
    p.add_argument('--smp_learnable_alpha', action='store_true', default=False)
    p.add_argument('--alpha',          type=float, default=0.5)
    p.add_argument('--size_match',     type=str,   default='decoder_to_encoder',
                   choices=['decoder_to_encoder','encoder_to_decoder'])
    p.add_argument('--use_mask',       action='store_true', default=False)
    p.add_argument('--smp_classifier', type=str,   default='conv',
                   choices=['linear','conv'])
    p.add_argument('--align',          type=str,   default='pre')
    p.add_argument('--seg_ckpt',       type=str,   default='',
                   help='Pretrained segmentation checkpoint path (shared by all SMP configs).')

    # ── SMP finetuned resume checkpoints (one per fixed config, order: enc/dec/fuse_ws/fuse_mul)
    p.add_argument('--smp_resumes',    type=str, nargs='*', default=[],
                   help='Optional finetuned checkpoints for the 4 fixed SMP configs '
                        '(order: SMP-enc, SMP-dec, SMP-fuse-weighted_sum, SMP-fuse-multiply). '
                        'Empty string = skip for that config.')

    # ── Baseline pretrained model IDs / paths ─────────────────────────────────
    p.add_argument('--retfound_finetune', type=str, default='RETFound_mae_natureOCT',
                   help='HuggingFace repo name under YukunZhou/ for RETFound_mae pretrained weights.')
    p.add_argument('--retfound_resume',   type=str, default='',
                   help='Optional finetuned checkpoint for RETFound_mae.')
    p.add_argument('--vit_finetune',      type=str, default='google/vit-base-patch16-224-in21k',
                   help='HuggingFace model ID for ViT-Base pretrained weights.')
    p.add_argument('--vit_resume',        type=str, default='',
                   help='Optional finetuned checkpoint for ViT-Base.')
    p.add_argument('--resnet_finetune',   type=str, default='microsoft/resnet-50',
                   help='HuggingFace model ID for ResNet-50 pretrained weights.')
    p.add_argument('--resnet_resume',     type=str, default='',
                   help='Optional finetuned checkpoint for ResNet-50.')
    p.add_argument('--effnet_resume',     type=str, default='',
                   help='Optional finetuned checkpoint for EfficientNet-B4.')

    # ── Grid (optional, appended to the fixed configs) ───────────────────────
    p.add_argument('--grid', action='store_true', default=False,
                   help='Also sweep enc_idxs × dec_idxs × fusion_dims on top of fixed configs.')
    p.add_argument('--enc_idxs',    type=int, nargs='+', default=[-1, -2, -3])
    p.add_argument('--dec_idxs',    type=int, nargs='+', default=[-1])
    p.add_argument('--fusion_dims', type=int, nargs='+', default=[4, 9, 16, 32])

    # ── Baseline models selection ────────────────────────────────────────────
    p.add_argument('--skip_baselines', action='store_true', default=False,
                   help='Skip all baseline models.')
    p.add_argument('--baselines', type=str, nargs='+',
                   default=['RETFound_mae', 'ViT-Base-patch16-224', 'ResNet-50', 'EfficientNet-B4'],
                   help='Which baseline models to include.')

    # ── Timing ───────────────────────────────────────────────────────────────
    p.add_argument('--n_warmup',   type=int,   default=10)
    p.add_argument('--n_runs',     type=int,   default=50)
    p.add_argument('--device',     type=str,   default='cuda', choices=['cuda','cpu'])
    p.add_argument('--batch_size', type=int,   default=1)

    # ── Output ────────────────────────────────────────────────────────────────
    p.add_argument('--output_csv', type=str, default='computation_overhead.csv')

    # ── Alpha-only mode ───────────────────────────────────────────────────────
    p.add_argument('--alpha_only', action='store_true', default=False,
                   help='Skip FLOPs and inference timing; only load models and report alpha values.')

    # ── SMP config filters ────────────────────────────────────────────────────
    p.add_argument('--smp_modes', type=str, nargs='+', default=None,
                   choices=['enc', 'dec', 'fuse'],
                   help='Only include SMP configs with these modes. Default: all.')
    p.add_argument('--smp_fuse_modes', type=str, nargs='+', default=None,
                   help='Only include SMP fuse configs with these fuse_modes (e.g. weighted_sum multiply). Default: all.')
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args_parser().parse_args()

    device = torch.device(
        args.device if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu'
    )
    print(f"Device : {device}")
    if device.type == 'cuda':
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    all_configs = []

    if not args.skip_baselines:
        all_configs += baseline_configs(args)

    smp = smp_configs(args)
    if args.smp_modes:
        smp = [c for c in smp if c['mode'] in args.smp_modes]
    if args.smp_fuse_modes:
        smp = [c for c in smp if c.get('fuse_mode') in args.smp_fuse_modes]
    all_configs += smp

    all_rows = [evaluate_config(cfg, args, device) for cfg in all_configs]
    all_rows = add_overhead_columns(all_rows)

    # ── Write CSV ─────────────────────────────────────────────────────────────
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    if all_rows:
        fieldnames = list(dict.fromkeys(k for r in all_rows for k in r.keys()))
        for r in all_rows:
            for f in fieldnames:
                r.setdefault(f, None)
        with open(args.output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nResults saved → {args.output_csv}")

    # ── Summary table ─────────────────────────────────────────────────────────
    W = 150
    hdr = (
        f"{'model_id':<42} {'type':<9} {'sz':>4} "
        f"{'params_M':>9} {'flops_G':>9} {'time_ms':>11} "
        f"{'+params_M':>10} {'+flops_G':>9} {'+time_ms':>10} "
        f"{'alpha':>8}"
    )
    print(f"\n{'─'*W}")
    print(hdr)
    print(f"{'─'*W}")
    for r in all_rows:
        print(
            f"{str(r.get('model_id','?')):<42} "
            f"{str(r.get('model_type','?')):<9} "
            f"{str(r.get('input_size','?')):>4} "
            f"{str(r.get('total_params_M','?')):>9} "
            f"{str(r.get('total_flops_G','?')):>9} "
            f"{str(r.get('inference_mean_ms','?')):>11} "
            f"{str(r.get('overhead_params_M','?')):>10} "
            f"{str(r.get('overhead_flops_G','?')):>9} "
            f"{str(r.get('overhead_time_ms','?')):>10} "
            f"{str(r.get('alpha_value','?')):>8}"
        )
    print(f"{'─'*W}")
    print("Overhead columns (+) are relative to the SMP-enc baseline. alpha = trained fusion gate value.")


if __name__ == '__main__':
    main()
