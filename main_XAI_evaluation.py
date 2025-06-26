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
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from transformers import (
    ViTImageProcessor, ViTForImageClassification,
    AutoImageProcessor, EfficientNetForImageClassification,
    ResNetForImageClassification
)

import models_vit as models
import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset,DistributedSamplerWrapper,TransformWrapper
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.losses import FocalLoss, compute_alpha_from_labels
from util.evaluation import InsertionMetric, DeletionMetric
from baselines.Attention import Attention_Map
from baselines.GradCAM import GradCAM
from baselines.RISE import RISE, RISEBatch
from huggingface_hub import hf_hub_download, login
from engine_finetune import evaluate_half3D, train_one_epoch, evaluate
import wandb

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
    parser.add_argument('--input_size', default=256, type=int,
                        help='images input size')
    parser.add_argument('--xai', default='attn', type=str,
                        help='Name of xai method to use, e.g., attn, rise')
    parser.add_argument('--use_rollout', action='store_true',
                    help='Use rollout for attention map generation')

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
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--img_dir', default='/orange/bianjiang/tienyu/OCT_AD/all_images/', type=str)
    parser.add_argument('--num_k', default=0, type=float)
    
    # Augmentation parameters (Not used for XAI evaluation)
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
    
    # fine-tuning parameters
    parser.add_argument('--norm', default='IMAGENET', type=str, help='Normalization method')
    parser.add_argument('--enhance', action='store_true', default=False, help='Use enhanced data')
    parser.add_argument('--datasets_seed', default=2026, type=int)

    return parser

def evaluate_XAI(data_loader, xai_method, metric_func_dict, device, args, epoch, mode, num_class, k, log_writer):
    """Evaluate the XAI method on the dataset."""
    metric_logger = misc.MetricLogger(delimiter="  ")
    os.makedirs(os.path.join(args.output_dir, args.task), exist_ok=True)
    overall_metrics_dict = {k:[] for k in metric_func_dict.keys()}
    for batch in metric_logger.log_every(data_loader, 10, f'{mode}:'):
        images, target = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
        bs = images.shape[0]
        each_dict = {}
        with torch.cuda.amp.autocast():
            attention_map_bs = xai_method(images)
            print(f'Attention map shape: {attention_map_bs.shape}')
            for k, v in metric_func_dict.items():
                e_score = v(images, attention_map_bs, bs)
                overall_metrics_dict[k].append(e_score)
                each_dict[k] = e_score
            
        metric_logger.update(**each_dict)
    
    
    print(f'XAI Metrics at epoch {epoch} ({mode}):')
    for k, v in metric_logger.meters.items():
        score = v.global_avg
        print(f'{k}: {score:.4f}')
        if log_writer is not None:
            log_writer.add_scalar(f'{mode}/{k}', score, epoch)
    # overall metrics
    for k, v in overall_metrics_dict.items():
        score = np.mean(v)
        print(f'Overall {k}: {score:.4f}')
        if log_writer is not None:
            log_writer.add_scalar(f'{mode}/overall_{k}', score, epoch)
    
    out_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
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

    processor = None
    if 'RETFound_mae' in args.model:
        model = models.__dict__['RETFound_mae'](
        img_size=args.input_size,
        num_classes=args.nb_classes,
        drop_path_rate=0.2,
        global_pool=True,
    )
    elif 'RETFound_dinov2' in args.model:
        model = models.__dict__['RETFound_dinov2'](
        img_size=args.input_size,
        num_classes=args.nb_classes,
        drop_path_rate=0.2,
        global_pool=True,
    )
    elif 'vit-base-patch16-224' in args.model:
            # ViT-base-patch16-224 preprocessor
            model_ = args.finetune if args.finetune else 'google/vit-base-patch16-224'
            processor = TransformWrapper(ViTImageProcessor.from_pretrained(model_))
            model = ViTForImageClassification.from_pretrained(
                model_,
                image_size=args.input_size,
                num_labels=args.nb_classes,
                id2label={0: "control", 1: "ad"},
                label2id={"control": 0, "ad": 1},
                ignore_mismatched_sizes=True,
                attn_implementation="eager"
            )
    elif 'efficientnet-b0' in args.model:
        # EfficientNet-B0 preprocessor
        model_ = args.finetune if args.finetune else 'google/efficientnet-b0'
        processor = TransformWrapper(AutoImageProcessor.from_pretrained(model_))
        model = EfficientNetForImageClassification.from_pretrained(
            model_,
            image_size=args.input_size,
            num_labels=args.nb_classes,
            id2label={0: "control", 1: "ad"},
            label2id={"control": 0, "ad": 1},
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
            id2label={0: "control", 1: "ad"},
            label2id={"control": 0, "ad": 1},
            ignore_mismatched_sizes=True
        )
    elif 'resnet-50' in args.model: #TODO: Need implement and check if this is correct
        model_name = args.finetune if args.finetune else 'microsoft/resnet-50'
        processor = TransformWrapper(AutoImageProcessor.from_pretrained(model_name))
        model = ResNetForImageClassification.from_pretrained(
            model_name,
            num_labels=args.nb_classes,
            id2label={i: str(i) for i in range(args.nb_classes)},
            label2id={str(i): i for i in range(args.nb_classes)},
            ignore_mismatched_sizes=True
        )
    else:
        model = models.__dict__[args.model](
            num_classes=args.nb_classes,
            args=args,
        )
    
    #RETFound special case: load pretrained checkpoint
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
            
    dataset_train = build_dataset(is_train='train', args=args, k=args.num_k,img_dir=args.img_dir,transform=processor)
    dataset_val = build_dataset(is_train='val', args=args, k=args.num_k,img_dir=args.img_dir,transform=processor)
    dataset_test = build_dataset(is_train='test', args=args, k=args.num_k,img_dir=args.img_dir,transform=processor)

    sampler_train = None
    sampler_val = None
    sampler_test = None
    wandb.init(
        project="RETFound_MAE",
        name=args.task,
        config=args,
        dir=os.path.join(args.log_dir,args.task),
    )
    os.makedirs(args.log_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=os.path.join(args.log_dir,args.task))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

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

    # Load finetuned model if specified
    if args.resume and args.resume != '0':
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        if 'model' in checkpoint:
            checkpoint_model = checkpoint['model']
        else:
            checkpoint_model = checkpoint
        model.load_state_dict(checkpoint_model, strict=False)
        print("Resume checkpoint %s" % args.resume)

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
        XAI_module = RISEBatch(model, input_size=args.input_size, gpu_batch=args.batch_size)
    elif args.xai == 'attn':
        XAI_module = Attention_Map(model, args.model, input_size=args.input_size, N=11, use_rollout=args.use_rollout, print_layers=True)
    elif args.xai == 'gradcam':
        XAI_module = GradCAM(model, model_name=args.model, patch_size=args.input_size)
    else:
        raise ValueError(f"Unknown XAI method: {args.xai}")
    XAI_module.to(device)
    #metric_func
    metric_func_dict = {
        'insertion': InsertionMetric(model, img_size=args.input_size, n_classes=args.nb_classes),
        'deletion': DeletionMetric(model, img_size=args.input_size, n_classes=args.nb_classes),
    }
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
    main(args, criterion)


