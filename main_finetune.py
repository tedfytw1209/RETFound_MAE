import argparse
import datetime
import json

import numpy as np
import pandas as pd
import os
import time
from pathlib import Path

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
import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset,DistributedSamplerWrapper,TransformWrapper
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.losses import FocalLoss, compute_alpha_from_labels
from huggingface_hub import hf_hub_download, login
from engine_finetune import evaluate_half3D, train_one_epoch, evaluate
import wandb
from pytorch_pretrained_vit import ViT

import warnings
import faulthandler

faulthandler.enable()
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--loss_weight', default=0, type=int,
                        help='Use weighted loss for imbalanced dataset. (default: 0, no weighted loss)')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=256, type=int,
                        help='images input size')
    parser.add_argument('--drop_path', type=float, default=0.2, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--optimizer', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw")')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=5e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.65,
                        help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--lr_scheduler', type=str, default='cosine', 
                        help='LR scheduler (default: "cosine")')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument("--schedule_step", type=int, default=10)
    parser.add_argument("--schedule_gamma", type=float, default=0.1)
    
    #Loss parameters
    parser.add_argument('--use_focal_loss', action='store_true',
                    help='Use Focal Loss instead of CrossEntropy')
    parser.add_argument('--focal_gamma', default=2.0, type=float,
                        help='Gamma parameter for Focal Loss')

    # Augmentation parameters
    parser.add_argument('--train_no_aug', action='store_true', default=False,
                        help='No training augmentation (random crop/flip, color jitter, auto augment, random erase)')
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='', type=str,
                        help='finetune from checkpoint')
    parser.add_argument('--task', default='', type=str,
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')
    parser.add_argument('--eval_score', default='default', type=str,
                        help='eval_score, default means (f1 + roc_auc + kappa) / 3')
    parser.add_argument('--testval', action='store_true', default=False,
                        help='Use test set for validation, otherwise use val set')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=8, type=int,
                        help='number of the classification types')
    parser.add_argument('--modality', default='OCT', type=str,
                        help='used modality of the UF dataset, e.g., OCT, CFP, thickness, etc.')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_logs',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--bal_sampler', action='store_true', default=False,
                        help='Enabling balanced class sampler')
    parser.add_argument('--fix_extractor', action='store_true', default=False,
                        help='Fixing the backbone parameters')
    parser.add_argument('--num_k', default=0, type=float)
    parser.add_argument('--img_dir', default='/orange/bianjiang/tienyu/OCT_AD/all_images/', type=str)
    parser.add_argument('--select_layer_idx', default=-1, type=int, help='number of layers to select for training')
    parser.add_argument('--th_heatmap', action='store_true', default=False, help='Transform thickness map to heatmap')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    # Image per Patient settings
    parser.add_argument('--use_img_per_patient', action='store_true', default=False,
                        help='Whether to use image per patient sampling')

    # fine-tuning parameters
    parser.add_argument('--savemodel', action='store_true', default=True,
                        help='Save model')
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
    parser.add_argument('--add_mask', action='store_true', default=False,
                        help='Add mask to the image based on thickness map')

    return parser

def get_label_mappings(args):
    if 'ad_control' in args.task:
        id2label = {0: "control", 1: "ad"}
        label2id = {v: k for k, v in id2label.items()}
    else:
        id2label = {i: f"class_{i}" for i in range(args.nb_classes)}
        label2id = {v: k for k, v in id2label.items()}
    return id2label, label2id

def visualize_dataset_samples(dataset, args, num_samples=8, save_path=None):
    """
    Visualize sample images from the dataset
    
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
    
    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    # Sample random indices
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for i, idx in enumerate(indices):
        if i >= num_samples:
            break
            
        try:
            # Get sample
            if hasattr(dataset, 'half3D') and dataset.half3D:
                # Handle multi-slice case
                image, label, image_len = dataset[idx]
                if isinstance(image, list):
                    # Take the middle slice for visualization
                    middle_idx = len(image) // 2
                    img_tensor = image[middle_idx] if middle_idx < len(image) else image[0]
                else:
                    img_tensor = image[image_len//2] if image_len > 1 else image[0]
            else:
                image, label, _ = dataset[idx]
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
            
            # Display image
            axes[i].imshow(img_np)
            axes[i].set_title(f'{class_names[label]} (idx: {idx})', fontsize=10)
            axes[i].axis('off')
            
        except Exception as e:
            print(f"Error visualizing sample {idx}: {e}")
            axes[i].text(0.5, 0.5, f'Error\n{idx}', ha='center', va='center')
            axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(indices), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Sample Images from {args.modality} Dataset', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

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
    return model, processor

def get_model(args):
    id2label, label2id = get_label_mappings(args)
    if args.model.startswith('timm'):
        return get_timm_model(args)
    processor = None
    if 'RETFound_mae' in args.model:
        model = models.__dict__['RETFound_mae'](
        img_size=args.input_size,
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )
    elif 'RETFound_dinov2' in args.model:
        model = models.__dict__['RETFound_dinov2'](
        img_size=args.input_size,
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool="token",
    )
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
                ignore_mismatched_sizes=True
            )
    elif 'pytorchvit' in args.model:
        model_name = args.finetune if args.finetune else 'B_16_imagenet1k'
        model = ViT(model_name, image_size=args.input_size, num_classes=args.nb_classes, pretrained=True)
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
    else:
        model = models.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            args=args,
        )
    #RETFound special case: load checkpoint
    if args.finetune and not args.eval:
        if 'RETFound' in args.finetune: 
            print(f"Downloading pre-trained weights from: {args.finetune}")
            checkpoint_path = hf_hub_download(
                repo_id=f'YukunZhou/{args.finetune}',
                filename=f'{args.finetune}.pth',
            )
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
    return model, processor

def main(args, criterion):

    misc.init_distributed_mode(args)
    
    wandb.init(
        project="RETFound_MAE",
        name=args.task,
        config=args,
        dir=os.path.join(args.log_dir,args.task),
    )
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    model, processor = get_model(args)
    print(model)
    
    #dataset selection
    if args.testval:
        print('Using test set for validation')
        dataset_train = build_dataset(is_train=['train','val'], args=args, k=args.num_k,img_dir=args.img_dir,modality=args.modality,transform=processor, select_layers=[args.select_layer_idx], th_heatmap=args.th_heatmap)
        dataset_val = build_dataset(is_train='test', args=args, k=args.num_k,img_dir=args.img_dir,modality=args.modality,transform=processor, select_layers=[args.select_layer_idx], th_heatmap=args.th_heatmap)
        dataset_test = build_dataset(is_train='test', args=args, k=args.num_k,img_dir=args.img_dir,modality=args.modality,transform=processor, select_layers=[args.select_layer_idx], th_heatmap=args.th_heatmap)
    else:
        dataset_train = build_dataset(is_train='train', args=args, k=args.num_k,img_dir=args.img_dir,modality=args.modality,transform=processor, select_layers=[args.select_layer_idx], th_heatmap=args.th_heatmap)
        dataset_val = build_dataset(is_train='val', args=args, k=args.num_k,img_dir=args.img_dir,modality=args.modality,transform=processor, select_layers=[args.select_layer_idx], th_heatmap=args.th_heatmap)
        dataset_test = build_dataset(is_train='test', args=args, k=args.num_k,img_dir=args.img_dir,modality=args.modality,transform=processor, select_layers=[args.select_layer_idx], th_heatmap=args.th_heatmap)

    # Apply subset sampling if subset_ratio > 0
    if args.subset_ratio > 0:
        print(f'Applying subset sampling with ratio {args.subset_ratio}')
        
        def create_subset(dataset, split_name):
            """Create a subset of the dataset based on minor class numbers"""
            targets = np.array(dataset.targets)
            unique_classes, class_counts = np.unique(targets, return_counts=True)
            minor_class_count = np.min(class_counts)
            subset_size = int(args.subset_ratio * minor_class_count)
            
            print(f'{split_name} - Original size: {len(dataset)}, Minor class count: {minor_class_count}, Subset size: {subset_size}')
            
            # Sample equal number of samples from each class
            subset_indices = []
            for class_idx in unique_classes:
                class_indices = np.where(targets == class_idx)[0]
                if len(class_indices) >= subset_size:
                    # Randomly sample subset_size samples from this class
                    rng = np.random.RandomState(args.seed)
                    sampled_indices = rng.choice(class_indices, subset_size, replace=False)
                else:
                    # If class has fewer samples than subset_size, use all samples
                    sampled_indices = class_indices
                subset_indices.extend(sampled_indices)
            
            # Create subset dataset
            subset_dataset = Subset(dataset, subset_indices)
            
            # Add targets attribute to subset for compatibility
            subset_dataset.targets = [dataset.targets[i] for i in subset_indices]
            subset_dataset.classes = dataset.classes
            subset_dataset.class_to_idx = dataset.class_to_idx
            
            return subset_dataset
        
        dataset_train = create_subset(dataset_train, 'Train')
        dataset_val = create_subset(dataset_val, 'Validation')
        dataset_test = create_subset(dataset_test, 'Test')
    
    # Apply subset sampling by absolute number if subset_num > 0
    if args.subset_num > 0:
        print(f'Old subset method for absolute number {args.subset_num}')
        
        def create_subset_by_num(dataset, split_name, subset_num):
            """Create a subset of the dataset with specified absolute number"""
            targets = np.array(dataset.targets)
            unique_classes, class_counts = np.unique(targets, return_counts=True)
            n_classes = len(unique_classes)
            
            print(f'{split_name} - Original size: {len(dataset)}, Classes: {n_classes}, Target subset size: {subset_num}')
            
            if subset_num < n_classes:
                # Too small to guarantee at least one per class → fall back to plain random sample
                print(f'Warning: subset_num ({subset_num}) < number of classes ({n_classes}), using random sampling')
                rng = np.random.RandomState(args.seed)
                subset_indices = rng.choice(len(dataset), min(subset_num, len(dataset)), replace=False)
            else:
                # Use stratified sampling to maintain class distribution
                from sklearn.model_selection import StratifiedShuffleSplit
                if subset_num >= len(dataset):
                    print(f'Warning: subset_num ({subset_num}) >= dataset size ({len(dataset)}), using full dataset')
                    subset_indices = list(range(len(dataset)))
                else:
                    sss = StratifiedShuffleSplit(n_splits=1, train_size=subset_num, random_state=args.seed)
                    subset_indices = next(sss.split(range(len(dataset)), targets))[0]
            
            # Create subset dataset
            subset_dataset = Subset(dataset, subset_indices)
            
            # Add targets attribute to subset for compatibility
            subset_dataset.targets = [dataset.targets[i] for i in subset_indices]
            subset_dataset.classes = dataset.classes
            subset_dataset.class_to_idx = dataset.class_to_idx
            
            print(f'{split_name} - Final subset size: {len(subset_dataset)}')
            return subset_dataset

        dataset_train = create_subset_by_num(dataset_train, 'Train', int(args.subset_num))

    # Apply subset sampling by absolute number if new_subset_num > 0
    if args.new_subset_num > 0:
        print(f'New subset method for absolute number {args.new_subset_num}')
        def create_separate_class_based_subsets(train_dataset, val_dataset, total_subset_num):
            """Create separate subsets from train and validation datasets based on class ratios"""
            
            def create_class_balanced_subset(dataset, split_name, target_size):
                """Create a class-balanced subset from a single dataset"""
                targets = np.array(dataset.targets)
                unique_classes, class_counts = np.unique(targets, return_counts=True)
                n_classes = len(unique_classes)
                
                # Calculate class ratios within this dataset
                class_ratios = class_counts / len(targets)
                
                print(f'\n{split_name} dataset - Original size: {len(dataset)}, Classes: {n_classes}')
                print(f'{split_name} class counts: {dict(zip(unique_classes, class_counts))}')
                print(f'{split_name} class ratios: {dict(zip(unique_classes, class_ratios))}')
                print(f'{split_name} target subset size: {target_size}')
                
                # Separate samples by class and permute
                rng = np.random.RandomState(42)
                selected_indices = []
                
                for class_idx in unique_classes:
                    # Get all samples for this class
                    class_mask = targets == class_idx
                    class_samples = np.where(class_mask)[0]
                    
                    # Permute samples within this class
                    class_samples_copy = class_samples.copy()
                    rng.shuffle(class_samples_copy)
                    
                    # Calculate how many samples to select for this class
                    class_target_samples = int((target_size-n_classes) * class_ratios[class_idx]) + 1
                    
                    # Ensure we don't exceed available samples
                    available_samples = len(class_samples_copy)
                    if class_target_samples > available_samples:
                        print(f'Warning: {split_name} Class {class_idx} needs {class_target_samples} samples but only {available_samples} available')
                        class_target_samples = available_samples
                    
                    # Select samples for this class
                    selected_class_samples = class_samples_copy[:class_target_samples]
                    selected_indices.extend(selected_class_samples)
                    
                    print(f'{split_name} Class {class_idx}: ratio={class_ratios[class_idx]:.3f}, target={class_target_samples}, selected={len(selected_class_samples)}')
                
                print('Selected indices:', selected_indices)
                subset_dataset = Subset(dataset, selected_indices)
                
                # Add targets attribute to subset for compatibility
                subset_dataset.targets = [dataset.targets[i] for i in selected_indices]
                subset_dataset.annotations = dataset.annotations.iloc[selected_indices].reset_index(drop=True)
                subset_dataset.classes = dataset.classes
                subset_dataset.class_to_idx = dataset.class_to_idx
                
                print(f'{split_name} final subset size: {len(subset_dataset)}')
                return subset_dataset
            
            # Calculate target sizes for train and validation (80/20 split)
            train_target_size = int(total_subset_num * 0.8)
            val_target_size = int(total_subset_num * 0.2)
            
            print(f'Total target subset size: {total_subset_num}')
            print(f'Train target size: {train_target_size} (80%)')
            print(f'Validation target size: {val_target_size} (20%)')
            
            # Create subsets separately
            train_subset = create_class_balanced_subset(train_dataset, 'Train', train_target_size)
            val_subset = create_class_balanced_subset(val_dataset, 'Validation', val_target_size)
            
            return train_subset, val_subset

        dataset_train, dataset_val = create_separate_class_based_subsets(dataset_train, dataset_val, int(args.new_subset_num))
    #print final label distribution
    print('Final label distribution:')
    print('Train:', pd.Series(dataset_train.targets).value_counts())
    print('Validation:', pd.Series(dataset_val.targets).value_counts())
    print('Test:', pd.Series(dataset_test.targets).value_counts())
    # Visualize sample images if requested
    if args.visualize_samples and misc.is_main_process():
        print("Generating dataset visualizations...")
        # Create output directory for visualizations
        vis_dir = os.path.join(args.output_dir, args.task, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        # Visualize training samples
        train_vis_path = os.path.join(vis_dir, f'train_samples_{args.modality}.png')
        visualize_dataset_samples(dataset_train, args, num_samples=8, save_path=train_vis_path)
        # Visualize validation samples
        val_vis_path = os.path.join(vis_dir, f'val_samples_{args.modality}.png')
        visualize_dataset_samples(dataset_val, args, num_samples=8, save_path=val_vis_path)
        # Visualize test samples
        test_vis_path = os.path.join(vis_dir, f'test_samples_{args.modality}.png')
        visualize_dataset_samples(dataset_test, args, num_samples=8, save_path=test_vis_path)
        print(f"Dataset visualizations saved to: {vis_dir}")
    
    #for weighted loss
    if args.loss_weight:
        train_target = np.array(dataset_train.targets)
        train_weight = np.zeros(len(dataset_train.classes))
        class_idx = [dataset_train.class_to_idx[c] for c in dataset_train.classes]
        print('train_target:',train_target)
        print('train_classes idx:',class_idx)
        for i in class_idx:
            train_weight[i] = np.sum(train_target == i)
        train_weight = np.sum(train_weight) / train_weight
        print('train_weight:',train_weight)
        train_labels = dataset_train.targets
        alpha = compute_alpha_from_labels(train_labels, num_classes=args.nb_classes)
        alpha = alpha.to(torch.float32).to(device)
        print('train_alpha:',alpha)
    else:
        train_weight = None
        alpha = None
    
    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        if args.bal_sampler:
            train_target = np.array(dataset_train.targets)
            class_weight = np.zeros(len(dataset_train.classes))
            class_idx = [dataset_train.class_to_idx[c] for c in dataset_train.classes]
            print('train_target:',train_target)
            print('train_classes idx:',class_idx)
            for i in class_idx:
                class_weight[i] = np.sum(train_target == i)
            class_weight = np.sum(class_weight) / class_weight
            print('class_weight:',class_weight)
            sample_weight = class_weight[train_target]
            sample_weight = sample_weight / np.sum(sample_weight)
            bal_train_sampler = torch.utils.data.WeightedRandomSampler(sample_weight, len(sample_weight), replacement=True)
            sampler_train = DistributedSamplerWrapper(
                bal_train_sampler, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank,
                shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            if not args.eval:
                sampler_train = torch.utils.data.DistributedSampler(
                        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
                    )
                print("Sampler_train = %s" % str(sampler_train))
            if args.dist_eval:
                if len(dataset_val) % num_tasks != 0:
                    print(
                        'Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                        'This will slightly alter validation results as extra duplicate entries are added to achieve '
                        'equal num of samples per-process.')
                sampler_val = torch.utils.data.DistributedSampler(
                    dataset_val, num_replicas=num_tasks, rank=global_rank,
                    shuffle=True)  # shuffle=True to reduce monitor bias
            else:
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        if args.dist_eval:
            if len(dataset_test) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=num_tasks, rank=global_rank,
                shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    
    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=os.path.join(args.log_dir,args.task))
    else:
        log_writer = None

    if not args.eval:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

        print(f'len of train_set: {len(data_loader_train) * args.batch_size}')

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    if args.resume and args.eval:
        checkpoint = torch.load(args.resume, map_location='cpu')
        print("Load checkpoint from: %s" % args.resume)
        model.load_state_dict(checkpoint['model'])

    model.to(device)
    model_without_ddp = model

    if args.fix_extractor:
        print("Fixing the backbone parameters")
        # Hugging Face models with 'classifier' as the head
        hf_models_with_classifier = ['vit_base_patch16_224', 'efficientnet_b0', 'efficientnet_b4', 'resnet-50', 'dinov3']
        if args.model in hf_models_with_classifier:
            head_keyword = 'classifier'
        else:
            head_keyword = 'head'  # timm or custom models
        for name, param in model.named_parameters():
            param.requires_grad = head_keyword in name
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of model params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # HF transformers model (ViT / EfficientNet) AdamW ---
    no_weight_decay = model_without_ddp.no_weight_decay() if hasattr(model_without_ddp, 'no_weight_decay') else []
    if 'RETFound' not in args.model:
        if args.optimizer == 'adamw':
            print("Using AdamW optimizer")
            optimizer = torch.optim.AdamW(model_without_ddp.parameters(),lr=args.lr,weight_decay=args.weight_decay)
        elif args.optimizer == 'sgd':
            print("Using SGD optimizer")
            optimizer = torch.optim.SGD(model_without_ddp.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {args.optimizer}")
    else:
        print("Using AdamW optimizer with layer-wise learning rate decay for RETFound")
        param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
                                            no_weight_decay_list=no_weight_decay,
                                            layer_decay=args.layer_decay
                                            )
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    if args.lr_scheduler == 'cosine':
        exp_lr_scheduler = None
    elif args.lr_scheduler == 'step':
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.schedule_step, gamma=args.schedule_gamma)
    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.use_focal_loss:
        print(f"Using Focal Loss (gamma={args.focal_gamma}, alpha={alpha})")
        criterion = FocalLoss(gamma=args.focal_gamma, alpha=alpha)
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(train_weight).to(torch.float32).to(device) if train_weight is not None else None)

    print("criterion = %s" % str(criterion))

    # misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        if 'epoch' in checkpoint:
            print("Test with the best model at epoch = %d" % checkpoint['epoch'])
        test_stats, auc_roc = evaluate(data_loader_test, model, device, args, epoch=0, mode='test',
                                       num_class=args.nb_classes,k=args.num_k, log_writer=log_writer, eval_score=args.eval_score)
        wandb_dict={f'test_{k}': v for k, v in test_stats.items()}
        wandb.log(wandb_dict)
        wandb.finish()
        if log_writer is not None:
            log_writer.close()
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    print(f"Number of training samples: {len(dataset_train)}")
    print(f"Number of validation samples: {len(dataset_val)}")
    print(f'Evaluation score: {args.eval_score}')
    start_time = time.time()
    max_score = 0.0
    best_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            exp_lr_scheduler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args
        )

        val_stats, val_score = evaluate(data_loader_val, model, device, args, epoch, mode='val',
                                        num_class=args.nb_classes,k=args.num_k, log_writer=log_writer, eval_score=args.eval_score)
        if log_writer is not None and misc.is_main_process():
            wandb_dict = {"epoch": epoch}
            wandb_dict.update({f'train_{k}': v for k, v in train_stats.items()})
            wandb_dict.update({f'val_{k}': v for k, v in val_stats.items()})
            wandb.log(wandb_dict, step=epoch)
        if max_score < val_score:
            max_score = val_score
            best_epoch = epoch
            if args.output_dir and args.savemodel:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, mode='best')
        print("Best epoch = %d, Best score = %.4f" % (best_epoch, max_score))


        if epoch == (args.epochs - 1):
            checkpoint = torch.load(os.path.join(args.output_dir, args.task, 'checkpoint-best.pth'), map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
            model.to(device)
            print("Validation with the best model, epoch = %d:" % checkpoint['epoch'])
            val_stats, val_score = evaluate(data_loader_val, model, device, args, -1, mode='val',
                                           num_class=args.nb_classes, k=args.num_k, log_writer=log_writer, eval_score=args.eval_score)
            wandb_dict = {}
            wandb_dict.update({f'best_val_{k}': v for k, v in val_stats.items()})
            wandb.log(wandb_dict)

        if log_writer is not None:
            log_writer.add_scalar('loss/val', val_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, args.task, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    state_dict_best = torch.load(os.path.join(args.output_dir,args.task,'checkpoint-best.pth'), map_location='cpu')
    model_without_ddp.load_state_dict(state_dict_best['model'])
    print("Test with the best model, epoch = %d:" % state_dict_best['epoch'])
    test_stats,test_score = evaluate(data_loader_test, model_without_ddp, device,args,epoch=0, mode='test',num_class=args.nb_classes,k=args.num_k, log_writer=log_writer, eval_score=args.eval_score)
    wandb_dict = {}
    wandb_dict.update({f'test_{k}': v for k, v in test_stats.items()})
    wandb.log(wandb_dict)
    if log_writer is not None and misc.is_main_process():
        log_writer.close()
        wandb.finish()

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    criterion = torch.nn.CrossEntropyLoss()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, criterion)


