import argparse
import datetime
import json

import numpy as np
import pandas as pd
import os
import time
from pathlib import Path
from scipy.ndimage import zoom

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset
from torchvision import datasets, transforms
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from torch.optim import lr_scheduler
from transformers import (
    ViTImageProcessor, ViTForImageClassification,
    AutoImageProcessor, EfficientNetForImageClassification,
    ResNetForImageClassification, AutoModel
)
import matplotlib.pyplot as plt

import models_vit as models
import vig as vig_models
import pyramid_vig as pvig_models
from relaynet import ReLayNet, relynet_load_pretrained
from SAM2UNet.SAM2UNet_classifier import SAM2UNetClassifier
from SMP.smp_classifier import SMPClassifier, Config as SMPConfig
import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset,DistributedSamplerWrapper,TransformWrapper
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.losses import FocalLoss, compute_alpha_from_labels
from util.evaluation import (
    InsertionMetric,
    DeletionMetric,
    RelevanceMetric,
    LayerImportanceDistributionMetric,
    shannon_entropy,
    gini_coefficient,
    dispersion_cv,
    topk_ratio,
)
from util.misc import to_numpy,to_tensor
from baselines.Attention import Attention_Map
from baselines.GradCAM import GradCAM
from baselines.RISE import RISE, RISEBatch
from baselines.GradCAM_v2 import PytorchCAM
from huggingface_hub import hf_hub_download, login
from engine_finetune import evaluate_half3D, train_one_epoch, evaluate, reinit_model_weights_
import wandb
from pytorch_pretrained_vit import ViT

from pytorch_grad_cam import GradCAM as GradCAMv2, ScoreCAM, HiResCAM, GradCAMPlusPlus
import matplotlib.pyplot as plt
import warnings
import faulthandler

faulthandler.enable()
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    
    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--finetune', default='', type=str,
                        help='finetune from checkpoint')
    parser.add_argument('--task', default='', type=str,
                        help='finetune from checkpoint')
    parser.add_argument('--use_split', default='test', type=str,
                        choices=['test', 'val', 'train'],
                        help='Name of xai method to use, e.g., test, val, train')
    parser.add_argument('--input_size', default=256, type=int,
                        help='images input size')
    parser.add_argument('--xai', default='attn', type=str,
                        help='Name of xai method to use, e.g., attn, rise')
    parser.add_argument('--use_rollout', action='store_true',
                    help='Use rollout for attention map generation')
    parser.add_argument('--drop_path', type=float, default=0.2, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--SMPMode', type=str, default='dec',
                        help='SMP mode (fuse, enc, dec)')
    
    # Metrics parameters
    parser.add_argument('--used_quantus', action='store_true', default=False,
                        help='Whether to use quantus library for some metrics')
    parser.add_argument('--step_pixels', default=224, type=int,
                        help='Step size in pixels for insertion/deletion metrics')
    parser.add_argument(
        '--layer_metric_include_bg',
        action='store_true',
        default=False,
        help='If set, include segmentation background label 0 as a layer for layer-importance distribution metrics.'
    )
    parser.add_argument(
        '--print_layer_metrics',
        action='store_true',
        default=False,
        help='If set, print per-sample layer-importance details for debugging.'
    )
    parser.add_argument(
        '--print_layer_metrics_num',
        default=8,
        type=int,
        help='Number of samples to print layer-importance debug details for.'
    )
    
    # Fair evaluation parameters
    parser.add_argument('--normalize_saliency_size', action='store_true', default=False,
                        help='Normalize saliency maps to a common resolution for fair comparison across models with different input sizes')
    parser.add_argument('--eval_resolution', default=224, type=int,
                        help='Common resolution for saliency map evaluation when normalize_saliency_size is enabled (default: 224)')
    parser.add_argument('--proportional_step', action='store_true', default=False,
                        help='Make step_pixels proportional to image size for fair comparison')
    parser.add_argument('--skip_model_dependent_metrics', action='store_true', default=False,
                        help='Skip insertion/deletion metrics (useful for faster evaluation, as these metrics require model inference at each pixel step)')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=8, type=int,
                        help='number of the classification types')
    parser.add_argument('--modality', default='OCT', type=str,
                        help='used modality of the UF dataset, e.g., OCT, CFP')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_logs',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='0',
                        help='resume from checkpoint')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--img_dir', default='/orange/bianjiang/tienyu/OCT_AD/all_images/', type=str)
    parser.add_argument('--num_k', default=0, type=float)
    parser.add_argument('--select_layer_idx', default=-1, type=int, help='number of layers to select for training')
    parser.add_argument('--th_heatmap', action='store_true', default=False, help='Transform thickness map to heatmap')
    
    # Augmentation parameters (Not used for XAI evaluation)
    parser.add_argument('--train_no_aug', action='store_true', default=False,
                        help='No training augmentation (random crop/flip, color jitter, auto augment, random erase)')
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params (Not used for XAI evaluation)
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    
    # Image per Patient settings
    parser.add_argument('--use_img_per_patient', action='store_true', default=False,
                        help='Whether to use image per patient sampling')

    # fine-tuning parameters
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--norm', default='IMAGENET', type=str, help='Normalization method')
    parser.add_argument('--enhance', action='store_true', default=False, help='Use enhanced data')
    parser.add_argument('--datasets_seed', default=2026, type=int)
    parser.add_argument('--subset_ratio', default=0, type=float,
                        help='Subset ratio for sampling dataset. If > 0, sample subset_ratio * minor_class_numbers from train/val/test datasets with seed 42')
    parser.add_argument('--subset_num', default=0, type=int,
                        help='Subset number for sampling dataset. If > 0, sample subset_num from train datasets with seed 42')
    parser.add_argument('--new_subset_num', default=0, type=int,
                        help='Subset number for sampling dataset. If > 0, sample subset_num from train datasets with seed 42')
    parser.add_argument('--visualize_samples', action='store_true', default=False,
                        help='Visualize sample images from the dataset')
    parser.add_argument('--thickness_dir', default='/orange/ruogu.fang/tienyuchang/IRB2024_OCT_thickness/Data/', type=str,
                        help='Directory containing thickness maps')
    parser.add_argument('--add_mask', action='store_true', default=False,
                        help='Add mask to the image based on thickness map')
    parser.add_argument('--output_mask', action='store_true', default=False,
                        help='Output mask of the image based on thickness map')
    parser.add_argument('--no_amp', dest='use_amp', action='store_false', help='Disable AMP')
    parser.set_defaults(use_amp=False)
    
    #SMP parameters
    parser.add_argument('--smp_fuse_mode', type=str, default='weighted_sum',
                        choices=["weighted_sum", "add", "channel_merge", "channel_multiply", "multiply"],
                        help='SMP fuse mode ("weighted_sum", "add", "channel_merge", "channel_multiply", "multiply") (default: "weighted_sum")')
    parser.add_argument('--smp_learnable_alpha', action='store_true', default=False,
                        help='SMP learnable alpha (default: False)')
    parser.add_argument('--smp_alpha', type=float, default=0.5,help='SMP alpha (0.0-1.0)')
    parser.add_argument('--smp_size_match', type=str, default='decoder_to_encoder',
                        choices=["decoder_to_encoder", "encoder_to_decoder"],
                        help='SMP size match (decoder_to_encoder, encoder_to_decoder) (default: "decoder_to_encoder")')
    parser.add_argument('--seg_mask', action='store_true', default=False,
                        help='Use segmentation mask output from SMP model')
    parser.add_argument('--mask_softmax', action='store_true', default=False,
                        help='Softmax the segmentation mask output from SMP model')
    parser.add_argument('--ignore_background', action='store_true', default=False,
                        help='Ignore the background class in segmentation mask output from SMP model')
    parser.add_argument('--fusion_dim', type=int, default=0,
                        help='Fusion dimension for SMP model (default: 0 means no projection)')
    parser.add_argument('--align', type=str, default='pre',
                        help='Aligment method for SMP model (pre, post) the size matching')
    parser.add_argument('--enc_idx', type=int, default=-1,help='SMP encoder index for feature extraction')
    parser.add_argument('--dec_idx', type=int, default=-1,help='SMP decoder index for feature extraction')
    parser.add_argument('--smp_classifier', type=str, default='linear',
                        choices=["linear", "conv"],
                        help='SMP classifier type ("linear", "conv") (default: "linear")')
    #
    parser.add_argument('--target_module', type=str, default='encoder', choices=['encoder', 'decoder', 'head'],
                        help='Target module for CAM methods')
    parser.add_argument('--select_index', type=int, default=-1,
                        help='Select index for CAM methods')

    return parser

def visualize_dataset_samples(dataset, args, num_samples=8, save_path=None):
    """
    Visualize sample images from the dataset along with their masks
    
    Args:
        dataset: Dataset object
        args: Arguments containing modality and other info
        num_samples: Number of samples to visualize
        save_path: Path to save the visualization (optional)
    """
    print(f"Visualizing {num_samples} sample images from {args.modality} dataset...")
    
    # Get class names
    if hasattr(dataset, 'classes'):
        class_names = dataset.classes
    else:
        class_names = [f"Class {i}" for i in range(args.nb_classes)]
    
    # Create figure - 4 rows x 4 columns to show 8 samples (each sample = image + mask)
    rows = (num_samples + 1) // 2  # 2 samples per row
    fig, axes = plt.subplots(rows, 4, figsize=(16, rows * 4))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    # Sample random indices
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for i, idx in enumerate(indices):
        if i >= num_samples:
            break
            
        # Get sample
        image, label, _, _, mask = dataset[idx]
        img_tensor = image
        
        # Convert tensor to numpy for visualization
        if isinstance(img_tensor, torch.Tensor):
            # Denormalize if normalized
            if img_tensor.min() < 0:  # Likely normalized
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_np = img_tensor.permute(1, 2, 0).numpy()
                img_np = img_np * std + mean
                img_np = np.clip(img_np, 0, 1)
            else:
                img_np = img_tensor.permute(1, 2, 0).numpy()
                img_np = np.clip(img_np, 0, 1)
        else:
            img_np = np.array(img_tensor)
            if img_np.max() > 1:
                img_np = img_np / 255.0
        
        # Handle different number of channels
        if img_np.shape[-1] == 1:
            img_np = np.repeat(img_np, 3, axis=-1)
        elif img_np.shape[-1] > 3:
            img_np = img_np[:, :, :3]
        
        # Process mask
        if isinstance(mask, torch.Tensor):
            mask_np = mask.numpy()
        else:
            mask_np = np.array(mask)
        
        # Ensure mask is 2D
        if len(mask_np.shape) == 3:
            mask_np = mask_np.squeeze()
        
        # Display image in column 0 or 2 (even positions)
        ax_img_idx = (i // 2) * 4 + (i % 2) * 2
        axes[ax_img_idx].imshow(img_np)
        axes[ax_img_idx].set_title(f'{class_names[label]} (idx: {idx})', fontsize=10)
        axes[ax_img_idx].axis('off')
        
        # Display mask in column 1 or 3 (odd positions)
        ax_mask_idx = ax_img_idx + 1
        axes[ax_mask_idx].imshow(mask_np, cmap='gray')
        axes[ax_mask_idx].set_title(f'Mask (idx: {idx})', fontsize=10)
        axes[ax_mask_idx].axis('off')
    
    # Hide unused subplots
    total_plots = rows * 4
    used_plots = len(indices) * 2
    for i in range(used_plots, total_plots):
        axes[i].axis('off')
    
    plt.suptitle(f'Sample Images and Masks from {args.modality} Dataset', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()
    plt.close()

def get_label_mappings(args):
    if 'ad_control' in args.task:
        id2label = {0: "control", 1: "ad"}
        label2id = {v: k for k, v in id2label.items()}
    else:
        id2label = {i: f"class_{i}" for i in range(args.nb_classes)}
        label2id = {v: k for k, v in id2label.items()}
    return id2label, label2id

def get_timm_model(args):
    import timm
    processor = None
    if 'efficientnet-b4' in args.model:
        model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=args.nb_classes)
        processor  = transforms.Compose([
            transforms.Resize((380,380)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    else:
        print(f"Model {args.model} not supported in timm.")
        exit(1)
    return model, processor, None

def get_model(args):
    id2label, label2id = get_label_mappings(args)
    if args.model.startswith('timm'):
        return get_timm_model(args)
    processor = None
    patch_size = None
    if 'RETFound_mae' in args.model:
        model = models.__dict__['RETFound_mae'](
        img_size=args.input_size,
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
        )
        patch_size = 16
    elif 'RETFound_dinov2' in args.model:
        model = models.__dict__['RETFound_dinov2'](
        img_size=args.input_size,
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool="token",
        )
        patch_size = 14
    elif 'vit-base-patch16-224' in args.model:
        # ViT-base-patch16-224 preprocessor
        model_ = args.finetune if args.finetune else 'google/vit-base-patch16-224'
        processor = TransformWrapper(ViTImageProcessor.from_pretrained(model_))
        model = ViTForImageClassification.from_pretrained(
            model_,
            image_size=args.input_size, #Not in tianhao code, default 224
            num_labels=args.nb_classes,
            hidden_dropout_prob=args.drop_path, #Not in tianhao code, default 0.0
            attention_probs_dropout_prob=args.drop_path, #Not in tianhao code, default 0.0
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
            attn_implementation="eager"
        )
        patch_size = 16
    elif 'pytorchvit' in args.model:
        model_name = args.finetune if args.finetune else 'B_16_imagenet1k'
        model = ViT(model_name, image_size=args.input_size, num_classes=args.nb_classes, pretrained=True)
        patch_size = 16
    elif 'efficientnet-b0' in args.model:
        # EfficientNet-B0 preprocessor
        model_ = args.finetune if args.finetune else 'google/efficientnet-b0'
        processor = TransformWrapper(AutoImageProcessor.from_pretrained(model_))
        model = EfficientNetForImageClassification.from_pretrained(
            model_,
            image_size=args.input_size,
            num_labels=args.nb_classes,
            dropout_rate=args.drop_path,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
    elif 'efficientnet-b4' in args.model:
        # EfficientNet-B0 preprocessor
        model_ = args.finetune if args.finetune else 'google/efficientnet-b4'
        processor = TransformWrapper(AutoImageProcessor.from_pretrained(model_))
        model = EfficientNetForImageClassification.from_pretrained(
            model_,
            image_size=args.input_size,
            num_labels=args.nb_classes,
            dropout_rate=args.drop_path,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
    elif 'resnet-50' in args.model:
        model_name = args.finetune if args.finetune else 'microsoft/resnet-50'
        processor = TransformWrapper(AutoImageProcessor.from_pretrained(model_name))
        model = ResNetForImageClassification.from_pretrained(
            model_name,
            num_labels=args.nb_classes,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
    elif 'relaynet' in args.model:
        model = ReLayNet(num_classes=args.nb_classes)
    elif 'dinov3' in args.model:
        model_name = f"facebook/{args.finetune}" if args.finetune else "facebook/dinov3-vitl16-pretrain-lvd1689m"
        processor = TransformWrapper(AutoImageProcessor.from_pretrained(model_name))
        feature_extractor = AutoModel.from_pretrained(model_name)
        model = models.DinoV3Classifier(feature_extractor, num_labels=args.nb_classes)
        patch_size = 16
    elif args.model.startswith('vig'):
        model = vig_models.__dict__[args.model](
            pretrained=True,
            num_classes=args.nb_classes,
        )
    elif args.model.startswith('pvig'):
        model = pvig_models.__dict__[args.model](
            pretrained=True,
            num_classes=args.nb_classes,
        )
    elif args.model.startswith('SAM2UNet'):
        model = SAM2UNetClassifier(num_classes=args.nb_classes,
                               seg_ckpt=args.finetune,
                               freeze_backbone=args.fix_extractor)
    elif args.model.startswith('SMP'):
        model = SMPClassifier(
            seg_arch=SMPConfig.SEG_ARCH,
            encoder_name=SMPConfig.ENCODER,
            encoder_weights=SMPConfig.ENCODER_WEIGHTS,
            in_channels=SMPConfig.IN_CHANNELS,
            num_classes=args.nb_classes,
            seg_classes=SMPConfig.SEG_CLASSES,
            seg_activation=SMPConfig.ACTIVATION,
            mode=args.SMPMode,
            fuse_mode=args.smp_fuse_mode,
            fusion_dim= args.fusion_dim,
            align=args.align,
            learnable_alpha=args.smp_learnable_alpha,
            alpha=args.smp_alpha,
            pretrained_seg_ckpt=args.finetune,
            dropout=SMPConfig.DROPOUT,
            size_match=args.smp_size_match,
            use_mask=args.seg_mask,
            mask_softmax=args.mask_softmax,
            ignore_background=args.ignore_background,
            enc_idx=args.enc_idx,
            dec_idx=args.dec_idx,
            smp_classifier=args.smp_classifier
        )
    else:
        model = models.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            args=args,
        )
    #RETFound special case: load checkpoint
    if args.finetune and not args.eval:
        if 'RETFound' in args.model: 
            print(f"Downloading pre-trained weights from: {args.finetune}")
            checkpoint_path = hf_hub_download(
                repo_id=f'YukunZhou/{args.finetune}',
                filename=f'{args.finetune}.pth',
            )
            with torch.serialization.safe_globals([argparse.Namespace]):
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print("Load pre-trained checkpoint from: %s" % args.finetune)
            if args.model!='RETFound_mae':
                checkpoint_model = checkpoint['teacher']
            else:
                checkpoint_model = checkpoint['model']
            checkpoint_model = {k.replace("backbone.", ""): v for k, v in checkpoint_model.items()}
            checkpoint_model = {k.replace("mlp.w12.", "mlp.fc1."): v for k, v in checkpoint_model.items()}
            checkpoint_model = {k.replace("mlp.w3.", "mlp.fc2."): v for k, v in checkpoint_model.items()}
            state_dict = model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
            # interpolate position embedding
            interpolate_pos_embed(model, checkpoint_model)
            # load pre-trained model
            msg = model.load_state_dict(checkpoint_model, strict=False)
            trunc_normal_(model.head.weight, std=2e-5)
            processor = None
        elif args.model.startswith('pvig') or args.model.startswith('vig'):
            pretrain_root = "/orange/ruogu.fang/tienyuchang/visionGNN_pretrain/"
            print('Loading:', args.finetune)
            with torch.serialization.safe_globals([argparse.Namespace]):
                state_dict = torch.load(os.path.join(pretrain_root, args.finetune + '.pth'))
            drop_keys = ["prediction.4.weight", "prediction.4.bias"]
            for k in drop_keys:
                if k in state_dict:
                    del state_dict[k]
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"[load] missing: {len(missing)}, unexpected: {len(unexpected)}")
        elif 'relaynet' in args.model:
            model = relynet_load_pretrained(model, args.finetune, args.device)
        else:
            print("No checkpoints from: %s" % args.finetune)
    return model, processor, patch_size

def evaluate_XAI(data_loader, xai_method, metric_func_dict, device, args, epoch, mode, num_class, k, log_writer):
    """Evaluate the XAI method on the dataset."""
    metric_logger = misc.MetricLogger(delimiter="  ")
    os.makedirs(os.path.join(args.output_dir, args.task), exist_ok=True)
    overall_metrics_dict = {k:[] for k in metric_func_dict.keys()}

    # Optional debug printing (main process only)
    print_layer_dbg = bool(getattr(args, "print_layer_metrics", False)) and misc.is_main_process()
    layer_print_left = int(getattr(args, "print_layer_metrics_num", 0))
    if layer_print_left < 0:
        layer_print_left = 0
    include_bg = bool(getattr(args, "layer_metric_include_bg", False))
    ignore_bg = not include_bg
    
    # Fair evaluation settings
    normalize_saliency = getattr(args, "normalize_saliency_size", False)
    eval_resolution = getattr(args, "eval_resolution", 224)
    if normalize_saliency:
        print(f"[Fair Evaluation Mode] Normalizing saliency maps to {eval_resolution}x{eval_resolution} for fair comparison")

    # Track per-class scores (keyed by ground-truth class index)
    classwise_metrics_dict = {
        metric_name: {cls_idx: [] for cls_idx in range(num_class)}
        for metric_name in metric_func_dict.keys()
    }
    for batch in metric_logger.log_every(data_loader, 10, f'{mode}:'):
        images, target = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
        sample_ids = batch[3] if (isinstance(batch, (list, tuple)) and len(batch) > 3) else None
        gt_mask = to_numpy(batch[4])
        #debug
        print(np.unique(to_numpy(gt_mask), return_counts=True))
        # Remove channel dimension: (B, 1, H, W) -> (B, H, W)
        if gt_mask is not None and gt_mask.ndim == 4 and gt_mask.shape[1] == 1:
            gt_mask = gt_mask.squeeze(1)
        bs = images.shape[0]
        target_np = target.detach().cpu().numpy()
        each_dict = {}
        #with torch.cuda.amp.autocast():
        #print(f'Input images shape: {images.shape}', 'ground truth mask shape:', gt_mask.shape, 'target:', target)
        
        # Keep original images for model-dependent metrics (insertion/deletion)
        images_original = images
        
        attention_map_bs = xai_method(images,targets=target) # numpy shape: (B, img_size, img_size)
        attention_map_bs = attention_map_bs - attention_map_bs.min(axis=(1, 2), keepdims=True) + 1e-9 # numpy shape: (B, img_size, img_size), add small value to avoid all-zero map
        
        # Keep original saliency for model-dependent metrics
        attention_map_original = attention_map_bs
        gt_mask_original = gt_mask
        
        # Normalize saliency maps to common resolution for fair comparison
        if normalize_saliency and attention_map_bs.shape[1] != eval_resolution:
            original_size = attention_map_bs.shape[1]
            scale_factor = eval_resolution / original_size
            attention_map_bs_normalized = np.zeros((bs, eval_resolution, eval_resolution), dtype=attention_map_bs.dtype)
            for i in range(bs):
                attention_map_bs_normalized[i] = zoom(attention_map_bs[i], scale_factor, order=1)  # bilinear interpolation
            attention_map_bs = attention_map_bs_normalized
            
            # Also resize gt_mask to match the normalized saliency map size
            if gt_mask is not None:
                gt_mask_normalized = np.zeros((bs, eval_resolution, eval_resolution), dtype=gt_mask.dtype)
                for i in range(bs):
                    gt_mask_normalized[i] = zoom(gt_mask[i], scale_factor, order=0)  # nearest neighbor for mask
                gt_mask = gt_mask_normalized
            
            # Resize images to match the normalized resolution (for non-model-dependent metrics)
            images_normalized = torch.nn.functional.interpolate(
                images, size=(eval_resolution, eval_resolution), mode='bilinear', align_corners=False
            )
            images = images_normalized

        # Print per-sample layer-importance breakdown for debugging (throttled)
        if print_layer_dbg and layer_print_left > 0 and gt_mask is not None and gt_mask.ndim == 3:
            for bi in range(bs):
                if layer_print_left <= 0:
                    break
                seg = gt_mask[bi]
                heat = attention_map_bs[bi]
                seg_i64 = np.asarray(seg).astype(np.int64, copy=False)
                labels = np.unique(seg_i64)
                if ignore_bg:
                    labels = labels[labels != 0]

                scores = np.zeros((labels.size,), dtype=np.float64)
                for li, lab in enumerate(labels):
                    scores[li] = np.asarray(heat, dtype=np.float64)[seg_i64 == lab].sum()

                total = float(np.maximum(scores, 0.0).sum()) if scores.size else 0.0
                probs = (scores / (total + 1e-12)) if (scores.size and total > 0.0) else np.zeros_like(scores, dtype=np.float64)

                ent = shannon_entropy(scores)
                gin = gini_coefficient(scores)
                disp = dispersion_cv(scores)
                top3 = topk_ratio(scores, k=3)

                sid = None
                try:
                    if sample_ids is not None:
                        sid = sample_ids[bi]
                except Exception:
                    sid = None
                sid_str = str(sid) if sid is not None else f"batch_item_{bi}"

                pairs = [(int(lab), float(sc), float(p)) for lab, sc, p in zip(labels.tolist(), scores.tolist(), probs.tolist())]
                head = pairs[:12]
                tail = pairs[-3:] if len(pairs) > 15 else []
                mid = " ... " if len(pairs) > 15 else ""
                print(
                    f"[layer-metrics] {mode} epoch={epoch} id={sid_str} y={int(target_np[bi])} "
                    f"include_bg={include_bg} layers={len(pairs)} total={total:.4e} "
                    f"entropy={ent:.4f} gini={gin:.4f} disp={disp:.4f} top3={top3:.4f}\n"
                    f"  (label, sum, p): {head}{mid}{tail}"
                )
                layer_print_left -= 1
        #print(f'Attention map shape: {attention_map_bs.shape}')
        #print(target_np)
        for k, v in metric_func_dict.items():
            # Use original images/saliency for model-dependent metrics (insertion/deletion)
            # These metrics need to pass images through the model at its native resolution
            if k in ['insertion', 'deletion']:
                metric_images = images_original
                metric_saliency = attention_map_original
                metric_gt_mask = gt_mask_original
            else:
                # Use normalized versions for other metrics
                metric_images = images
                metric_saliency = attention_map_bs
                metric_gt_mask = gt_mask
            
            e_score_bs = v(metric_images, metric_saliency, gt_mask=metric_gt_mask, batch_size=bs, y_batch=target, explain_func=xai_method, explain_func_kwargs={})
            e_score_bs_np = np.asarray(e_score_bs)
            # Aggregate per-class metrics when the metric returns per-sample scores
            #print(f'Batch {k} scores:', e_score_bs_np)
            if e_score_bs_np.ndim >= 1 and e_score_bs_np.shape[0] == bs:
                for cls_idx in range(num_class):
                    cls_mask = target_np == cls_idx
                    if not np.any(cls_mask):
                        continue
                    class_score = float(np.mean(e_score_bs_np[cls_mask]))
                    classwise_metrics_dict[k][cls_idx].append(class_score)
            e_score_bs_mean = float(np.mean(e_score_bs_np))
            overall_metrics_dict[k].append(e_score_bs_mean)
            each_dict[k] = float(e_score_bs_mean)
            #print(f'{k}: {e_score_bs:.4f}')
        #print(classwise_metrics_dict)
            
        metric_logger.update(**each_dict)
    
    
    #print(f'XAI Metrics at epoch {epoch} ({mode}):')
    for k, v in metric_logger.meters.items():
        score = v.global_avg
        print(f'{k}: {score:.4f}')
        if log_writer is not None:
            log_writer.add_scalar(f'{mode}/{k}', score, epoch)

    classwise_out_dict = {}
    print("Class-wise metrics:")
    for metric_name, class_dict in classwise_metrics_dict.items():
        for cls_idx, scores in class_dict.items():
            if len(scores) == 0:
                continue
            cls_score = float(np.mean(scores))
            classwise_out_dict[f'{metric_name}_class_{cls_idx}'] = cls_score
            print(f'Class {cls_idx} {metric_name}: {cls_score:.4f}')
            if log_writer is not None:
                log_writer.add_scalar(f'{mode}/{metric_name}_class_{cls_idx}', cls_score, epoch)

    # overall metrics
    for k, v in overall_metrics_dict.items():
        score = np.mean(v)
        print(f'Overall {k}: {score:.4f}')
        if log_writer is not None:
            log_writer.add_scalar(f'{mode}/overall_{k}', score, epoch)
    
    out_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    out_dict.update(classwise_out_dict)
    return out_dict, score

def main(args, criterion):

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    model, processor, patch_size = get_model(args)

    dataset_train = build_dataset(is_train='train', args=args, k=args.num_k,img_dir=args.img_dir, modality=args.modality,transform=processor, eval_mode=True)
    dataset_val = build_dataset(is_train='val', args=args, k=args.num_k,img_dir=args.img_dir, modality=args.modality,transform=processor, eval_mode=True)
    dataset_test = build_dataset(is_train='test', args=args, k=args.num_k,img_dir=args.img_dir, modality=args.modality,transform=processor, eval_mode=True)

    sampler_train = None
    sampler_val = None
    sampler_test = None
    wandb.init(
        project="RETFound_MAE_XAI_Evaluation",
        name=args.task,
        config=args,
        dir=os.path.join(args.log_dir,args.task),
    )
    os.makedirs(args.log_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=os.path.join(args.log_dir,args.task))

    if args.use_split == 'train':
        data_loader_test = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
    elif args.use_split == 'val':
        data_loader_test = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    #visualize some samples
    if misc.is_main_process():
        print("Generating dataset visualizations...")
        # Create output directory for visualizations
        vis_dir = os.path.join(args.output_dir, args.task, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        # Visualize test samples
        test_vis_path = os.path.join(vis_dir, f'test_samples_{args.modality}.png')
        visualize_dataset_samples(dataset_test, args, num_samples=8, save_path=test_vis_path)
        print(f"Dataset visualizations saved to: {vis_dir}")

    # Load finetuned model if specified
    if args.resume and args.resume != '0':
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            with torch.serialization.safe_globals([argparse.Namespace]):
                checkpoint = torch.load(args.resume, map_location='cpu')
        if 'model' in checkpoint:
            checkpoint_model = checkpoint['model']
        else:
            checkpoint_model = checkpoint
        model.load_state_dict(checkpoint_model, strict=False)
        print("Resume checkpoint %s" % args.resume)
    else:
        raise ValueError("Please provide finetuned model checkpoint for evaluation using --resume")

    model = model.float()
    model.to(device)
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of model params (M): %.2f' % (n_parameters / 1.e6))

    test_stats, auc_roc = evaluate(data_loader_test, model, device, args, epoch=0, mode='test',
                                    num_class=args.nb_classes,k=args.num_k, log_writer=log_writer)
    wandb_dict={f'test_{k}': v for k, v in test_stats.items()}
    wandb.log(wandb_dict)

    print(f"Start evaluating XAI:")
    start_time = time.time()
    ###TODO: evaluate XAI baselines
    if args.xai == 'rise':
        print("Using RISE for XAI")
        XAI_module = RISEBatch(model, input_size=(args.input_size, args.input_size), gpu_batch=100, N = 500, device=device, n_class=args.nb_classes)
    elif args.xai == 'attn':
        XAI_module = Attention_Map(model, args.model, input_size=args.input_size, N=11, use_rollout=args.use_rollout, print_layers=True, device=device)
    elif args.xai == 'gradcam':
        XAI_module = GradCAM(model, model_name=args.model, img_size=args.input_size, patch_size=patch_size, device=device)
    elif args.xai == 'gradcamv2':
        XAI_module = PytorchCAM(model, model_name=args.model, img_size=args.input_size, patch_size=patch_size, method=GradCAMv2, target_module=args.target_module, select_index=args.select_index, device=device)
    elif args.xai == 'scorecam':
        XAI_module = PytorchCAM(model, model_name=args.model, img_size=args.input_size, patch_size=patch_size, method=ScoreCAM, target_module=args.target_module, select_index=args.select_index, device=device)
    elif args.xai == 'hirescam':
        XAI_module = PytorchCAM(model, model_name=args.model, img_size=args.input_size, patch_size=patch_size, method=HiResCAM, target_module=args.target_module, select_index=args.select_index, device=device)
    elif args.xai == 'gradcam++':
        XAI_module = PytorchCAM(model, model_name=args.model, img_size=args.input_size, patch_size=patch_size, method=GradCAMPlusPlus, target_module=args.target_module, select_index=args.select_index, device=device)
    elif args.xai == 'crp':
        from baselines.CRP_LXT import CRP
        XAI_module = CRP(model, model_name=args.model, img_size=args.input_size, patch_size=patch_size, device=device)
    elif args.xai == 'lxt':
        from baselines.CRP_LXT import LXT
        XAI_module = LXT(model, model_name=args.model, img_size=args.input_size, patch_size=patch_size, conv_gamma=0.25, lin_gamma=0.05, device=device)
    else:
        raise ValueError(f"Unknown XAI method: {args.xai}")
    XAI_module.to(device)
    #metric_func
    #metric_func_dict = {
    #    'insertion': InsertionMetric(model, img_size=args.input_size, n_classes=args.nb_classes),
    #    'deletion': DeletionMetric(model, img_size=args.input_size, n_classes=args.nb_classes),
    #}

    #if args.used_quantus:
    #    import quantus
    #    from util.evaluation_quantus import SufficiencyMetric, ConsistencyMetric, PointingGameMetric, ComplexityMetric, RandomLogitMetric
    
    # Determine step size for insertion/deletion metrics
    normalize_saliency = getattr(args, "normalize_saliency_size", False)
    eval_resolution = getattr(args, "eval_resolution", 224)
    proportional_step = getattr(args, "proportional_step", False)
    skip_model_dependent = getattr(args, "skip_model_dependent_metrics", False)
    
    # For insertion/deletion, always use model's native resolution (args.input_size)
    # since these metrics need to pass images through the model
    insertion_deletion_img_size = args.input_size
    
    # Calculate step_pixels for insertion/deletion at native resolution
    if proportional_step:
        # Make step proportional to image size (e.g., ~0.5% of total pixels)
        step_pixels = max(1, int((insertion_deletion_img_size * insertion_deletion_img_size) * 0.005))
        print(f"[Proportional Step Mode] Using step_pixels={step_pixels} for img_size={insertion_deletion_img_size} (~0.5% of pixels)")
    else:
        step_pixels = args.step_pixels if hasattr(args, 'step_pixels') else 224
        print(f"[Fixed Step Mode] Using step_pixels={step_pixels} for img_size={insertion_deletion_img_size}")
    
    ignore_bg = not bool(getattr(args, "layer_metric_include_bg", False))
    
    print(f"[Metric Configuration] insertion/deletion_img_size={insertion_deletion_img_size}, step_pixels={step_pixels}, normalize_saliency={normalize_saliency}")
    
    # Build metric dictionary conditionally
    metric_func_dict = {}
    
    # Add insertion/deletion metrics if not skipped
    if not skip_model_dependent:
        metric_func_dict['insertion'] = InsertionMetric(model, img_size=insertion_deletion_img_size, step=step_pixels, n_classes=args.nb_classes)
        metric_func_dict['deletion'] = DeletionMetric(model, img_size=insertion_deletion_img_size, step=step_pixels, n_classes=args.nb_classes)
        print("[Metrics] Including insertion metric")
        print("[Metrics] Including deletion metric")
    else:
        print("[Metrics] Skipping insertion/deletion metrics (skip_model_dependent_metrics enabled)")
    
    # Add model-independent metrics (these work with normalized saliency)
    metric_func_dict.update({
            # 'sufficiency': SufficiencyMetric(model, device),
            # 'consistency': ConsistencyMetric(model, device, discretise_func=quantus.discretise_func.rank),
            'relevance_mass': RelevanceMetric(pooling_type='sum,abs', output_type='mass'),
            'relevance_rank': RelevanceMetric(pooling_type='sum,abs', output_type='rank'),
            # Layer-importance distribution metrics (computed from gt_mask labels and heatmap saliency mass)
            'layer_entropy': LayerImportanceDistributionMetric(ignore_background=ignore_bg, output_type='entropy'),
            'layer_gini': LayerImportanceDistributionMetric(ignore_background=ignore_bg, output_type='gini'),
            'layer_dispersion': LayerImportanceDistributionMetric(ignore_background=ignore_bg, output_type='dispersion'),
            'layer_top3_ratio': LayerImportanceDistributionMetric(ignore_background=ignore_bg, output_type='top3_ratio'),
            #'complexity': ComplexityMetric(model, device),
            #'random_logit': RandomLogitMetric(model, device, n_classes=args.nb_classes),
        })
    
    print(f"[Metrics] Total metrics to evaluate: {len(metric_func_dict)}")
    test_stats, auc_roc = evaluate_XAI(data_loader_test, XAI_module,metric_func_dict, device, args, epoch=0, mode='test',
                                    num_class=args.nb_classes,k=args.num_k, log_writer=log_writer)
    wandb_dict={f'test_{k}': v for k, v in test_stats.items()}
    wandb.log(wandb_dict)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('XAI Evaluation time {}'.format(total_time_str))

    if log_writer is not None and misc.is_main_process():
        log_writer.close()
        wandb.finish()

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    criterion = torch.nn.CrossEntropyLoss()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # Build up the common task name
    study_name = args.task.split('-')[0]
    data_type = args.task.split('-')[1]
    finetune_model = args.finetune.split('/')[-1].replace('.pth','')
    if args.model=='SMP':
        if args.SMPMode=='fuse':
            args.task = f"{study_name}-{data_type}-{finetune_model}-{args.xai}-{args.SMPMode}-smp{args.smp_fuse_mode}-{args.align}-{args.fusion_dim}-fea{args.enc_idx}{args.dec_idx}-{args.smp_alpha}-{args.smp_size_match}-{args.smp_classifier}-{args.target_module}{args.select_index}-seed{args.seed}"
        else:
            args.task = f"{study_name}-{data_type}-{finetune_model}-{args.xai}-{args.SMPMode}-fea{args.enc_idx}{args.dec_idx}-{args.target_module}{args.select_index}-seed{args.seed}"
    else:
        args.task = f"{study_name}-{data_type}-{args.model}-{finetune_model}-{args.xai}-{args.input_size}-seed{args.seed}"
    
    if args.seg_mask and args.mask_softmax:
        args.task += '-softsegmask'
    elif args.seg_mask:
        args.task += '-segmask'
    if args.ignore_background:
        args.task += '-ignbg'
    if args.skip_model_dependent_metrics:
        args.task += '-skipMD'
    
    main(args, criterion)


