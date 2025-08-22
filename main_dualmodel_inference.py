import argparse
import datetime
import json

import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    ResNetForImageClassification
)

import models_vit as models
import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset,DistributedSamplerWrapper,TransformWrapper
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.losses import FocalLoss, compute_alpha_from_labels
from huggingface_hub import hf_hub_download, login
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, average_precision_score,
    hamming_loss, jaccard_score, recall_score, precision_score, cohen_kappa_score,matthews_corrcoef,
    multilabel_confusion_matrix,confusion_matrix
)
#from engine_finetune import evaluate_half3D, train_one_epoch, evaluate
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
                        help='used modality of the UF dataset, e.g., OCT, CFP')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_logs',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--oct_resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--cfp_resume', default='',
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

    return parser

def get_model(args):
    processor = None
    if 'DualViT' in args.model:
        sub_model_oct = models.__dict__['RETFound_mae'](
            img_size=args.input_size,
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )
        sub_model_cfp = models.__dict__['RETFound_mae'](
            img_size=args.input_size,
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )
        model = models.__dict__['DualViT'](
            vit_model_1=sub_model_oct, 
            vit_model_2=sub_model_cfp, 
            num_classes=args.nb_classes
        )
    elif 'RETFound_mae' in args.model:
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

#evaluate for dual model
@torch.no_grad()
def evaluate_dual(data_loader_oct, data_loader_cfp, model, device, args, epoch, mode, num_class, k, log_writer, eval_score=''):
    """Evaluate the model."""
    criterion = nn.CrossEntropyLoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    os.makedirs(os.path.join(args.output_dir, args.task), exist_ok=True)
    
    model.eval()
    true_onehot, pred_onehot, true_labels, pred_labels, pred_softmax = [], [], [], [], []
    total = len(data_loader_oct)
    it_oct = iter(data_loader_oct)
    it_cfp = iter(data_loader_cfp)
    for _ in metric_logger.log_every(range(total), 10, f'{mode}:'):
        oct_batch = next(it_oct)
        cfp_batch = next(it_cfp)
        print(oct_batch)
        print(cfp_batch)
        oct_images, target = oct_batch[0].to(device, non_blocking=True), oct_batch[1].to(device, non_blocking=True)
        cfp_images, cfp_target = cfp_batch[0].to(device, non_blocking=True), cfp_batch[1].to(device, non_blocking=True)
        target_onehot = F.one_hot(target.to(torch.int64), num_classes=num_class)
        
        with torch.cuda.amp.autocast():
            output = model(oct_images, cfp_images)
            if hasattr(output, 'logits'):
                output = output.logits
            else:
                output = output
            loss = criterion(output, target)
        output_ = nn.Softmax(dim=1)(output)
        output_label = output_.argmax(dim=1)
        output_onehot = F.one_hot(output_label.to(torch.int64), num_classes=num_class)
        
        metric_logger.update(loss=loss.item())
        true_onehot.extend(target_onehot.cpu().numpy())
        pred_onehot.extend(output_onehot.detach().cpu().numpy())
        true_labels.extend(target.cpu().numpy())
        pred_labels.extend(output_label.detach().cpu().numpy())
        pred_softmax.extend(output_.detach().cpu().numpy())
    
    accuracy = accuracy_score(true_labels, pred_labels)
    hamming = hamming_loss(true_onehot, pred_onehot)
    jaccard = jaccard_score(true_onehot, pred_onehot, average='macro')
    average_precision = average_precision_score(true_onehot, pred_softmax, average='macro')
    kappa = cohen_kappa_score(true_labels, pred_labels)
    f1 = f1_score(true_onehot, pred_onehot, zero_division=0, average='macro')
    roc_auc = roc_auc_score(true_onehot, pred_softmax, multi_class='ovr', average='macro')
    precision = precision_score(true_onehot, pred_onehot, zero_division=0, average='macro')
    recall = recall_score(true_onehot, pred_onehot, zero_division=0, average='macro')
    mcc = matthews_corrcoef(true_labels, pred_labels)
    
    conf = confusion_matrix(true_labels, pred_labels)
    if eval_score == 'mcc':
        score = mcc
    elif eval_score == 'roc_auc':
        score = roc_auc
    else:
        score = (f1 + roc_auc + kappa) / 3
    metric_dict = {}
    if log_writer:
        for metric_name, value in zip(['accuracy', 'f1', 'roc_auc', 'hamming', 'jaccard', 'precision', 'recall', 'average_precision', 'kappa', 'mcc', 'score'],
                                       [accuracy, f1, roc_auc, hamming, jaccard, precision, recall, average_precision, kappa, mcc, score]):
            log_writer.add_scalar(f'perf/{metric_name}', value, epoch)
            metric_dict[metric_name] = value
    
    print(f'val loss: {metric_logger.meters["loss"].global_avg}')
    print(f'Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}, Hamming Loss: {hamming:.4f},\n'
          f' Jaccard Score: {jaccard:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f},\n'
          f' Average Precision: {average_precision:.4f}, Kappa: {kappa:.4f}, MCC: {mcc:.4f}, Score: {score:.4f}')
    print("confusion_matrix:\n", conf)
    metric_logger.synchronize_between_processes()
    out_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    out_dict.update(metric_dict)
    return out_dict, score

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
    
    #dataset selection
    oct_dataset_test = build_dataset(is_train='test', args=args, k=args.num_k,img_dir=args.img_dir,modality='OCT',transform=processor)
    cfp_dataset_test = build_dataset(is_train='test', args=args, k=args.num_k,img_dir=args.img_dir,modality='CFP',transform=processor)
    
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
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    if args.dist_eval:
        if len(oct_dataset_test) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        oct_sampler_test = torch.utils.data.DistributedSampler(oct_dataset_test, num_replicas=num_tasks, rank=global_rank)
        cfp_sampler_test = torch.utils.data.DistributedSampler(cfp_dataset_test, num_replicas=num_tasks, rank=global_rank)
    else:
        oct_sampler_test = torch.utils.data.SequentialSampler(oct_dataset_test)
        cfp_sampler_test = torch.utils.data.SequentialSampler(cfp_dataset_test)
    wandb.init(
        project="RETFound_MAE",
        name=args.task,
        config=args,
        dir=os.path.join(args.log_dir,args.task),
    )
    if args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=os.path.join(args.log_dir,args.task))
    else:
        log_writer = None

    oct_data_loader_test = torch.utils.data.DataLoader(
        oct_dataset_test, sampler=oct_sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    cfp_data_loader_test = torch.utils.data.DataLoader(
        cfp_dataset_test, sampler=cfp_sampler_test,
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

    ### 08/22 add OCT/CFP Resume
    checkpoint = torch.load(args.oct_resume, map_location='cpu')
    print("Load checkpoint from: %s" % args.oct_resume)
    model.vit_model_1.load_state_dict(checkpoint['model'])

    checkpoint = torch.load(args.cfp_resume, map_location='cpu')
    print("Load checkpoint from: %s" % args.cfp_resume)
    model.vit_model_2.load_state_dict(checkpoint['model'])

    model.to(device)
    model_without_ddp = model
    
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

    if 'epoch' in checkpoint:
        print("Test with the best model at epoch = %d" % checkpoint['epoch'])
    test_stats, auc_roc = evaluate_dual(oct_data_loader_test, cfp_data_loader_test, model, device, args, epoch=0, mode='test',
                                    num_class=args.nb_classes,k=args.num_k, log_writer=log_writer, eval_score=args.eval_score)
    wandb_dict={f'test_{k}': v for k, v in test_stats.items()}
    wandb.log(wandb_dict)
    wandb.finish()
    if log_writer is not None:
        log_writer.close()

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    criterion = torch.nn.CrossEntropyLoss()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, criterion)


