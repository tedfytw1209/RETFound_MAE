import argparse
import datetime
import json

import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset
from torchvision import datasets, transforms, models as torchvision_models
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from torch.optim import lr_scheduler
from transformers import (
    ViTImageProcessor, ViTForImageClassification,
    AutoImageProcessor, EfficientNetForImageClassification,
    ResNetForImageClassification
)
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import matplotlib.pyplot as plt

import models_vit as models
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
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--loss_weight', default=0, type=int,
                        help='Use weighted loss for imbalanced dataset. (default: 0, no weighted loss)')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train (e.g., resnet18_paper for paper-specific ResNet-18)')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--drop_path', type=float, default=0.2, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--optimizer', default='sgd', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "sgd")')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='weight decay (default: 0.0005), 0.005, 0.0005')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr), 0.0001, 0.0005, 0.001, 0.005, 0.01, and 0.05')
    parser.add_argument('--blr', type=float, default=0.001, metavar='LR',
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
    parser.add_argument('--transform', default=1, type=int,
                        help='Transform type: 1 for relative work 1, 2 for relative work 2')
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
    parser.add_argument('--eval_score', default='roc_auc', type=str,
                        help='eval_score, default means (f1 + roc_auc + kappa) / 3')
    parser.add_argument('--testval', action='store_true', default=False,
                        help='Use test set for validation, otherwise use val set')

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

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # fine-tuning parameters
    parser.add_argument('--savemodel', action='store_true', default=True,
                        help='Save model')
    parser.add_argument('--norm', default='IMAGENET', type=str, help='Normalization method')
    parser.add_argument('--enhance', action='store_true', default=False, help='Use enhanced data')
    parser.add_argument('--datasets_seed', default=2026, type=int)
    parser.add_argument('--subset_ratio', default=0, type=float,
                        help='Subset ratio for sampling dataset. If > 0, sample subset_ratio * minor_class_numbers from train/val/test datasets with seed 42')
    parser.add_argument('--visualize_samples', action='store_true', default=False,
                        help='Visualize sample images from the dataset')
    
    # Additional settings based on paper specifications
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--l1_reg', type=float, default=0.0,
                        help='L1 regularization coefficient for FC layers (default: 0.0)')
    parser.add_argument('--l2_reg', type=float, default=0.0,
                        help='L2 regularization coefficient for entire model (default: 0.0)')
    parser.add_argument('--early_stopping', action='store_true', default=False,
                        help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (default: 10)')

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
                    mean = IMAGENET_DEFAULT_MEAN
                    std = IMAGENET_DEFAULT_STD
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

class CustomResNet18Paper(torch.nn.Module):
    """
    Custom ResNet-18 implementation based on paper specifications:
    - First 5 layers only (conv1, bn1, relu, maxpool, layer1)
    - Two key modifications from original ResNet-18:
      1) Stride size of first conv layer changed from 2 to 1
      2) Pooling size changed from 2x2 to 4x4
    - Layer1 has 64 channels (n=64)
    - Total parameters: 166,936 (157,504 for feature extractor + 9,432 for FC layers)
    - Modality-specific FC layers for different imaging modalities
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(CustomResNet18Paper, self).__init__()
        
        # Load pretrained ResNet18
        resnet18 = torchvision_models.resnet18(pretrained=pretrained)
        
        # Extract first 5 layers only (conv1, bn1, relu, maxpool, layer1)
        self.conv1 = resnet18.conv1
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.layer1 = resnet18.layer1
        
        # Apply paper modifications:
        # 1) Stride size of first conv layer changed from 2 to 1
        self.conv1.stride = (1, 1)
        
        # 2) Pooling size changed from 2x2 to 4x4
        self.avgpool = torch.nn.AvgPool2d(kernel_size=4, stride=4, padding=1)
        self.avgpool2 = torch.nn.AvgPool2d(kernel_size=4, stride=4)
        
        self.num_classes = num_classes
        
        # Calculate output feature map size after first 5 layers for 128x128 input:
        # conv1: 128x128 -> 128x128 (stride=1, padding=3)
        # avgpool: 128x128 -> 32x32 (4x4 kernel, stride=4, padding=1)
        # layer1: 32x32 -> 32x32 (no downsampling, 64 channels)
        # avgpool2: 32x32 -> 7x7 (4x4 kernel, stride=4)
        # So final feature map is 7x7 with 64 channels = 3,136 features

        # Modality-specific FC layers
        self.fc_uwf_color_faf = torch.nn.Linear(64 * 7 * 7, num_classes,bias=True)  # UWF color and FAF 3137
        self.fc_octa = torch.nn.Linear(64 * 7 * 7, num_classes,bias=True)  # OCTA 3137
        self.fc_gc_ipl = torch.nn.Linear(64 * 7 * 7, num_classes,bias=True)  # GC-IPL maps 3137
        self.fc_quantitative = torch.nn.Linear(20, num_classes,bias=True)  # Quantitative features (21 weights) 21
        
        # Final fusion layer (averaging pre-classification scores)
        self.fusion = torch.nn.Identity()  # Simple averaging in forward pass
        
        # Initialize weights
        if not pretrained:
            self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, x, modality='gc_ipl'):
        """
        Forward pass with modality-specific processing
        
        Args:
            x: Input tensor (batch_size, 3, 128, 128)
            modality: Modality type ('uwf_color_faf', 'octa', 'gc_ipl', 'quantitative')
        """
        # Feature extraction (first 5 layers only)
        x = self.conv1(x)  # stride=1, padding=3 (modified from original stride=2)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.avgpool(x)  # 4x4 kernel, stride=4, padding=1 (modified from original 2x2)
        x = self.layer1(x)  # ResNet layer1 (2 basic blocks, 64 channels)
        x = self.avgpool2(x)  # 4x4 kernel, stride=4 (modified from original 2x2)
        
        # Flatten and dimension reduction
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, 64*7*7)

        # Modality-specific pre-classification
        if modality == 'uwf_color_faf':
            x = self.fc_uwf_color_faf(x)
        elif modality == 'octa':
            x = self.fc_octa(x)
        elif modality == 'gc_ipl':
            x = self.fc_gc_ipl(x)
        elif modality == 'quantitative':
            # For quantitative features, input should be 21-dimensional
            x = self.fc_quantitative(x)
        else:
            # Default to UWF color/FAF
            x = self.fc_uwf_color_faf(x)
        
        return x
    
    def forward_multimodal(self, uwf_color_faf, octa, gc_ipl, quantitative_features):
        """
        Forward pass for multimodal inputs (as described in paper)
        
        Args:
            uwf_color_faf: UWF color and FAF images
            octa: OCTA images  
            gc_ipl: GC-IPL maps
            quantitative_features: Quantitative OCT/OCTA features (batch_size, 21)
        """
        # Get pre-classification scores for each modality
        score_uwf = self.forward(uwf_color_faf, modality='uwf_color_faf')
        score_octa = self.forward(octa, modality='octa')
        score_gc_ipl = self.forward(gc_ipl, modality='gc_ipl')
        score_quant = self.forward(quantitative_features, modality='quantitative')
        
        # Average pre-classification scores (as described in paper)
        combined_score = (score_uwf + score_octa + score_gc_ipl + score_quant) / 4
        
        # Apply sigmoid to get probabilities in [0, 1] range
        probabilities = torch.sigmoid(combined_score)
        
        return probabilities

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
   
    # Additional models from the paper table
    elif 'alexnet' in args.model:
        model = torchvision_models.alexnet(pretrained=True)
        # Replace the classifier head to match the 8-layer architecture
        # AlexNet has 5 conv layers + 3 FC layers (classifier)
        # The original classifier has: Dropout -> Linear(9216->4096) -> ReLU -> Dropout -> Linear(4096->4096) -> ReLU -> Linear(4096->1000)
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(9216, 4096),  # First FC layer: 9216 -> 4096
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(4096, 4096),  # Second FC layer: 4096 -> 4096
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, args.nb_classes),  # Final FC layer: 4096 -> num_classes
        )
    elif 'vgg11' in args.model:
        # Use VGG-11 with Batch Normalization for better training stability
        # BN layers address gradient vanishing/explosion and enhance generalization
        model = torchvision_models.vgg11_bn(pretrained=True)
        # Replace the classifier head
        model.classifier = torch.nn.Sequential(
            torch.nn.Linear(model.classifier[0].in_features, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(4096, args.nb_classes),
        )
    elif 'resnet18' in args.model:
        model = torchvision_models.resnet18(pretrained=True)
        # Replace the classifier head
        model.fc = torch.nn.Linear(model.fc.in_features, args.nb_classes)
    elif 'resnet18_paper' in args.model:
        # Custom ResNet-18 implementation based on paper specifications
        model = CustomResNet18Paper(num_classes=args.nb_classes, pretrained=True)
    elif 'resnet101' in args.model:
        model = torchvision_models.resnet101(pretrained=True)
        # Replace the classifier head
        model.fc = torch.nn.Linear(model.fc.in_features, args.nb_classes)
    elif 'resnet152' in args.model:
        model = torchvision_models.resnet152(pretrained=True)
        # Replace the classifier head
        model.fc = torch.nn.Linear(model.fc.in_features, args.nb_classes)
    elif 'googlenet' in args.model:
        model = torchvision_models.googlenet(pretrained=True)
        # Replace the classifier head
        model.fc = torch.nn.Linear(model.fc.in_features, args.nb_classes)
    elif 'shufflenet' in args.model:
        model = torchvision_models.shufflenet_v2_x1_0(pretrained=True)
        # Replace the classifier head
        model.fc = torch.nn.Linear(model.fc.in_features, args.nb_classes)
    # Paper-specified torchvision models
    elif 'convnext_tiny' in args.model:
        model = torchvision_models.convnext_tiny(pretrained=True)
        # Replace the classifier head
        model.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(model.classifier[2].in_features, args.nb_classes)
        )
    elif 'regnet_x_16gf' in args.model:
        model = torchvision_models.regnet_x_16gf(pretrained=True)
        # Replace the classifier head
        model.fc = torch.nn.Linear(model.fc.in_features, args.nb_classes)
    elif 'efficientnet_b0' in args.model:
        model = torchvision_models.efficientnet_b0(pretrained=True)
        # Replace the classifier head
        model.classifier = torch.nn.Linear(model.classifier.in_features, args.nb_classes)
    elif 'vit_b_16' in args.model:
        model = torchvision_models.vit_b_16(pretrained=True)
        # Replace the classifier head
        model.heads = torch.nn.Linear(model.heads.in_features, args.nb_classes)
    elif 'swin_b' in args.model:
        model = torchvision_models.swin_b(pretrained=True)
        # Replace the classifier head
        model.head = torch.nn.Linear(model.head.in_features, args.nb_classes)
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
        else:
            print("No checkpoints from: %s" % args.finetune)
    return model, processor

def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    #!!TODO: debug in 'train' not the first
    if not isinstance(is_train, list):
        is_train = [is_train]
    # train transform
    if 'train' in is_train:
        # Custom training transform with 224x224 resize, horizontal flips, and random transformations
        t = []
        # Convert to tensor
        t.append(transforms.ToTensor())
        # Resize all images to 224x224
        t.append(transforms.Resize((args.input_size, args.input_size), interpolation=transforms.InterpolationMode.BICUBIC))
        # Add horizontal flips
        t.append(transforms.RandomHorizontalFlip(p=0.5))
        # Random rotation and affine transformations
        t.append(transforms.RandomRotation(degrees=10))
        t.append(transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)))
        # Normalize with ImageNet-1K mean and std
        t.append(transforms.Normalize(mean, std))
        return transforms.Compose(t)
    
    # eval transform
    t = []
    # Resize all images to 224x224 for evaluation
    # Convert to tensor
    t.append(transforms.ToTensor())
    t.append(transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC))
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

def build_transform2(is_train, args):
    """
    Enhanced data augmentation transform function based on paper specifications.
    Implements rotating, shifting, cropping, and zooming as described in the paper:
    "Data augmentation can be used to reduce over-fitting and increase the amount of training
    data. It creates new images by transforming (rotating, translating, scaling, flipping, distorting)
    and adding some noise (e.g. Gaussian noise) to the images in the training dataset."
    """
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    
    if not isinstance(is_train, list):
        is_train = [is_train]
    
    # Training transform with enhanced data augmentation
    if 'train' in is_train:
        t = []
        
        # Convert to tensor first
        t.append(transforms.ToTensor())
        
        # Resize to a larger size first for better cropping
        t.append(transforms.Resize((160, 160), interpolation=transforms.InterpolationMode.BICUBIC))
        
        # Random rotation: rotate images by random angles
        t.append(transforms.RandomRotation(degrees=15, interpolation=transforms.InterpolationMode.BICUBIC))
        
        # Random shifting (translation): translate images randomly
        t.append(transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), interpolation=transforms.InterpolationMode.BICUBIC))
        
        # Random cropping: crop random portions of the image
        t.append(transforms.RandomCrop((args.input_size, args.input_size)))
        
        # Random zooming (scaling): scale images randomly
        t.append(transforms.RandomAffine(degrees=0, scale=(0.8, 1.2), interpolation=transforms.InterpolationMode.BICUBIC))
        
        # Random horizontal flip
        t.append(transforms.RandomHorizontalFlip(p=0.5))
        
        # Random vertical flip (additional augmentation)
        t.append(transforms.RandomVerticalFlip(p=0.3))
        
        # Color jitter for additional diversity
        t.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
        
        # Random erasing to simulate noise/distortion
        t.append(transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0))
        
        # Normalize with ImageNet-1K mean and std
        t.append(transforms.Normalize(mean, std))
        
        return transforms.Compose(t)
    
    # Evaluation transform (no augmentation)
    t = []
    t.append(transforms.ToTensor())
    t.append(transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.BICUBIC))
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

def main(args, criterion):

    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    model, processor = get_model(args)
    if args.transform == 1:
        transform_train = build_transform(is_train=['train','val'], args=args)
        transform_eval = build_transform(is_train='test', args=args)
    elif args.transform == 2:
        transform_train = build_transform2(is_train=['train','val'], args=args)
        transform_eval = build_transform2(is_train='test', args=args)
    else:
        raise ValueError(f'Invalid transform type: {args.transform}')
    
    #dataset selection
    Select_Layer = ['RNFL-GCL (RNFL-GCL)_GCL-IPL (GCL-IPL)', 'GCL-IPL (GCL-IPL)_IPL-INL (IPL-INL)']
    if args.testval:
        print('Using test set for validation')
        dataset_train = build_dataset(is_train=['train','val'], args=args, k=args.num_k,img_dir=args.img_dir,modality=args.modality,transform=transform_train, select_layers=Select_Layer, th_resize=True, th_heatmap=True)
        dataset_val = build_dataset(is_train='test', args=args, k=args.num_k,img_dir=args.img_dir,modality=args.modality,transform=transform_eval, select_layers=Select_Layer, th_resize=True, th_heatmap=True)
        dataset_test = build_dataset(is_train='test', args=args, k=args.num_k,img_dir=args.img_dir,modality=args.modality,transform=transform_eval, select_layers=Select_Layer, th_resize=True, th_heatmap=True)
    else:
        dataset_train = build_dataset(is_train='train', args=args, k=args.num_k,img_dir=args.img_dir,modality=args.modality,transform=transform_train, select_layers=Select_Layer, th_resize=True, th_heatmap=True)
        dataset_val = build_dataset(is_train='val', args=args, k=args.num_k,img_dir=args.img_dir,modality=args.modality,transform=transform_eval, select_layers=Select_Layer, th_resize=True, th_heatmap=True)
        dataset_test = build_dataset(is_train='test', args=args, k=args.num_k,img_dir=args.img_dir,modality=args.modality,transform=transform_eval, select_layers=Select_Layer, th_resize=True, th_heatmap=True)

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
    wandb.init(
        project="OCTAD_Relatives",
        name=args.task,
        config=args,
        dir=os.path.join(args.log_dir,args.task),
    )
    
    # Log regularization settings to wandb
    if args.l1_reg > 0 or args.l2_reg > 0:
        wandb.log({
            "l1_reg": args.l1_reg,
            "l2_reg": args.l2_reg,
            "regularization": True
        })
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
            drop_last=True,
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
        hf_models_with_classifier = ['vit_base_patch16_224', 'efficientnet_b0', 'efficientnet_b4', 'resnet-50', 
                                    'alexnet', 'vgg11', 'googlenet', 'convnext_tiny', 'efficientnet_b0']
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
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # HF transformers model (ViT / EfficientNet) AdamW ---
    no_weight_decay = model_without_ddp.no_weight_decay() if hasattr(model_without_ddp, 'no_weight_decay') else []
    if 'RETFound' not in args.model:
        if args.optimizer == 'adamw':
            print("Using AdamW optimizer")
            optimizer = torch.optim.AdamW(model_without_ddp.parameters(),lr=args.lr,weight_decay=args.weight_decay)
        elif args.optimizer == 'sgd':
            print("Using SGD optimizer")
            optimizer = torch.optim.SGD(model_without_ddp.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
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
        print("Using cosine learning rate scheduler")
    elif args.lr_scheduler == 'step':
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.schedule_step, gamma=args.schedule_gamma)
        print("Using step learning rate scheduler")
    else:
        exp_lr_scheduler = False
        print("No learning rate scheduler")
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
    
    # Print regularization settings
    if args.l1_reg > 0 or args.l2_reg > 0:
        print(f"Regularization enabled:")
        if args.l1_reg > 0:
            print(f"  L1 regularization (FC layers only): {args.l1_reg}")
        if args.l2_reg > 0:
            print(f"  L2 regularization (entire model): {args.l2_reg}")
    else:
        print("No regularization applied")

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
    patience_counter = 0
    
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
            patience_counter = 0  # Reset patience counter
            if args.output_dir and args.savemodel:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, mode='best')
        else:
            patience_counter += 1
            
        print("Best epoch = %d, Best score = %.4f" % (best_epoch, max_score))
        
        # Early stopping
        if args.early_stopping and patience_counter >= args.patience:
            print(f"Early stopping triggered after {epoch + 1} epochs (patience: {args.patience})")
            break


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


