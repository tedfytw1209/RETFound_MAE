import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parallel import DistributedDataParallel as DDP, DataParallel as DP
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from typing import Any, Dict, Iterable, Optional
from timm.data import Mixup
from timm.utils import accuracy
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, average_precision_score,
    hamming_loss, jaccard_score, recall_score, precision_score, cohen_kappa_score,matthews_corrcoef,
    multilabel_confusion_matrix,confusion_matrix
)
from pycm import ConfusionMatrix
import util.misc as misc
import util.lr_sched as lr_sched
from util.fairness import fariness_score

def compute_regularization_loss(model, l1_reg=0.0, l2_reg=0.0):
    """
    Compute L1 and L2 regularization losses for the model.
    
    Args:
        model: PyTorch model
        l1_reg: L1 regularization coefficient for FC layers only
        l2_reg: L2 regularization coefficient for entire model
    
    Returns:
        total_reg_loss: Combined regularization loss
    """
    l1_loss = 0.0
    l2_loss = 0.0
    
    # L1 regularization on FC layers only
    if l1_reg > 0:
        for name, param in model.named_parameters():
            if 'fc' in name.lower() or 'classifier' in name.lower() or 'head' in name.lower():
                l1_loss += torch.sum(torch.abs(param))
    
    # L2 regularization on entire model
    if l2_reg > 0:
        for param in model.parameters():
            l2_loss += torch.sum(param ** 2)
    
    total_reg_loss = l1_reg * l1_loss + l2_reg * l2_loss
    return total_reg_loss

def misc_measures(confusion_matrix):
    
    acc = []
    sensitivity = []
    specificity = []
    precision = []
    G = []
    F1_score_2 = []
    mcc_ = []
    #for avoid nan case
    for i in range(1, confusion_matrix.shape[0]):
        cm1=confusion_matrix[i]
        acc.append(1.*(cm1[0,0]+cm1[1,1])/np.sum(cm1))
        sensitivity_ = 1.*cm1[1,1]/max(cm1[1,0]+cm1[1,1],1e-9)
        sensitivity.append(sensitivity_)
        specificity_ = 1.*cm1[0,0]/max(cm1[0,1]+cm1[0,0],1e-9)
        specificity.append(specificity_)
        precision_ = 1.*cm1[1,1]/max(cm1[1,1]+cm1[0,1],1e-9)
        precision.append(precision_)
        G.append(np.sqrt(sensitivity_*specificity_))
        F1_score_2.append(2*precision_*sensitivity_/max(precision_+sensitivity_,1e-9))
        mcc = (cm1[0,0]*cm1[1,1]-cm1[0,1]*cm1[1,0])/max(np.sqrt((cm1[0,0]+cm1[0,1])*(cm1[0,0]+cm1[1,0])*(cm1[1,1]+cm1[1,0])*(cm1[1,1]+cm1[0,1])),1e-9)
        mcc_.append(mcc)
        
    acc = np.array(acc).mean()
    sensitivity = np.array(sensitivity).mean()
    specificity = np.array(specificity).mean()
    precision = np.array(precision).mean()
    G = np.array(G).mean()
    F1_score_2 = np.array(F1_score_2).mean()
    mcc_ = np.array(mcc_).mean()
    
    return acc, sensitivity, specificity, precision, G, F1_score_2, mcc_

def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    scheduler = None,
    max_norm: float = 0,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    args=None
):
    """Train the model for one epoch."""
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    print_freq, accum_iter = 20, args.accum_iter
    optimizer.zero_grad()
    all_labels, all_preds, all_probs = [], [], []
    true_onehot, pred_onehot= [], []
    if log_writer:
        print(f'log_dir: {log_writer.log_dir}')
    
    for data_iter_step, data_bs in enumerate(metric_logger.log_every(data_loader, print_freq, f'Epoch: [{epoch}]')):
        if scheduler is None and data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        samples = data_bs[0]
        targets = data_bs[1]
        samples, targets = samples.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        target_onehot = F.one_hot(targets.to(torch.int64), num_classes=args.nb_classes)
        if mixup_fn:
            samples, targets = mixup_fn(samples, targets)
        
        with torch.cuda.amp.autocast():
            outputs = model(**samples) if isinstance(samples, dict) else model(samples)
            if hasattr(outputs, 'logits'):
                outputs = outputs.logits
            else:
                outputs = outputs
            loss = criterion(outputs, targets)
            
            # Add regularization loss if specified
            if hasattr(args, 'l1_reg') and hasattr(args, 'l2_reg'):
                if args.l1_reg > 0 or args.l2_reg > 0:
                    reg_loss = compute_regularization_loss(model, args.l1_reg, args.l2_reg)
                    loss = loss + reg_loss
        loss_value = loss.item()
        loss /= accum_iter
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(probs, 1)
        output_onehot = F.one_hot(preds.to(torch.int64), num_classes=args.nb_classes)
        true_onehot.extend(target_onehot.cpu().numpy())
        pred_onehot.extend(output_onehot.detach().cpu().numpy())
        all_labels.extend(targets.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())
        
        loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        
        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss/train', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)
    
    if scheduler is not None and scheduler:
        scheduler.step()
    #Metric
    metric_logger.synchronize_between_processes()
    conf = confusion_matrix(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    roc_auc = roc_auc_score(true_onehot, all_probs, multi_class='ovr', average='macro')
    f1 = f1_score(all_labels, all_preds, zero_division=0, average='macro')
    kappa = cohen_kappa_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    
    train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    for metric_name, value in zip(['accuracy', 'f1', 'roc_auc', 'kappa', 'mcc'],
                                       [accuracy, f1, roc_auc, kappa, mcc]):
            train_stats[metric_name] = value
    print(f'Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')
    print("confusion_matrix:\n", conf)
    return train_stats

@torch.no_grad()
def evaluate(data_loader, model, device, args, epoch, mode, num_class, k, log_writer, eval_score=''):
    """Evaluate the model."""
    criterion = nn.CrossEntropyLoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    os.makedirs(os.path.join(args.output_dir, args.task), exist_ok=True)
    
    model.eval()
    true_onehot, pred_onehot, true_labels, pred_labels, pred_softmax = [], [], [], [], []
    
    for batch in metric_logger.log_every(data_loader, 10, f'{mode}:'):
        images, target = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
        target_onehot = F.one_hot(target.to(torch.int64), num_classes=num_class)
        
        with torch.cuda.amp.autocast():
            output = model(images)
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
    
    try:
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
    except Exception as e:
        print("Error in metric calculation:", e)
        print("true_labels:", np.array(true_labels))
        print('true_onehot:', np.array(true_onehot))
        print("pred_softmax:", np.array(pred_softmax))
        print("pred_labels:", np.array(pred_labels))
        print('pred_onehot:', np.array(pred_onehot))
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, 0.0
        
    
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
    
    results_path = os.path.join(args.output_dir, args.task, f'metrics_{mode}.csv')
    file_exists = os.path.isfile(results_path)
    with open(results_path, 'a', newline='', encoding='utf8') as cfa:
        wf = csv.writer(cfa)
        if not file_exists:
            wf.writerow(['val_loss', 'accuracy', 'f1', 'roc_auc', 'hamming', 'jaccard', 'precision', 'recall', 'average_precision', 'kappa', 'mcc'])
        wf.writerow([metric_logger.meters["loss"].global_avg, accuracy, f1, roc_auc, hamming, jaccard, precision, recall, average_precision, kappa, mcc])
    
    if mode == 'test':
        cm = ConfusionMatrix(actual_vector=true_labels, predict_vector=pred_labels)
        cm.plot(cmap=plt.cm.Blues, number_label=True, normalized=True, plot_lib="matplotlib")
        plt.savefig(os.path.join(args.output_dir, args.task, 'confusion_matrix_test.jpg'), dpi=600, bbox_inches='tight')
    
    out_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    out_dict.update(metric_dict)
    return out_dict, score

@torch.no_grad()
def evaluate_half3D(data_loader, model, device, task, epoch, mode, num_class, k, log_writer):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    if not os.path.exists(task):
        os.makedirs(task)

    prediction_decode_list = []
    prediction_list = []
    true_label_decode_list = []
    true_label_onehot_list = []
    
    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0] #(batch_size,k+1,3,224,224) or (batch_size*(k+1),3,224,224)
        if k>0:
            b,n,c,h,w = images.shape
            #print('image 0 mean of each slice:',images[0].mean((-1,-2,-3)))
            images = images.view(b*n,c,h,w)
            
        target = batch[1]
        slice_len = batch[2]
        #print('slice_len:',slice_len)
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        true_label=F.one_hot(target.to(torch.int64), num_classes=num_class)
        # compute output
        with torch.cuda.amp.autocast():
            output = model(**images) if isinstance(images, dict) else model(images)
            if hasattr(output, 'logits'):
                output = output.logits
            else:
                output = output
            prediction_softmax = nn.Softmax(dim=-1)(output)
            if k>0:
                output = output.view(b,n,-1)
                prediction_softmax = prediction_softmax.view(b,n,-1)
                output_list = []
                prediction_softmax_list = []
                #print('output 0 mean of each slice:',output[0].mean(-1))
                for e_output,e_prediction_softmax,e_slice_len in zip(output,prediction_softmax,slice_len):
                    e_output = e_output[:e_slice_len].mean(0)
                    e_prediction_softmax = e_prediction_softmax[:e_slice_len].mean(0)
                    output_list.append(e_output)
                    prediction_softmax_list.append(e_prediction_softmax)
                output = torch.stack(output_list)
                prediction_softmax = torch.stack(prediction_softmax_list)
            loss = criterion(output, target)
            _,prediction_decode = torch.max(prediction_softmax, 1)
            _,true_label_decode = torch.max(true_label, 1)

            prediction_decode_list.extend(prediction_decode.cpu().detach().numpy())
            true_label_decode_list.extend(true_label_decode.cpu().detach().numpy())
            true_label_onehot_list.extend(true_label.cpu().detach().numpy())
            prediction_list.extend(prediction_softmax.cpu().detach().numpy())

        acc1,_ = accuracy(output, target, topk=(1,2))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
    # gather the stats from all processes
    true_label_decode_list = np.array(true_label_decode_list)
    prediction_decode_list = np.array(prediction_decode_list)
    confusion_matrix = multilabel_confusion_matrix(true_label_decode_list, prediction_decode_list,labels=[i for i in range(num_class)])
    acc, sensitivity, specificity, precision, G, F1, mcc = misc_measures(confusion_matrix)
    
    auc_roc = roc_auc_score(true_label_onehot_list, prediction_list,multi_class='ovr',average='macro')
    auc_pr = average_precision_score(true_label_onehot_list, prediction_list,average='macro')          
            
    metric_logger.synchronize_between_processes()
    
    print('Sklearn Metrics - Acc: {:.4f} AUC-roc: {:.4f} AUC-pr: {:.4f} F1-score: {:.4f} MCC: {:.4f}'.format(acc, auc_roc, auc_pr, F1, mcc)) 
    print('Sklearn Metrics - Recall: {:.4f} Precision: {:.4f} Specificity: {:.4f}'.format(sensitivity, precision, specificity)) 
    results_path = task+'_metrics_{}.csv'.format(mode)
    with open(results_path,mode='a',newline='',encoding='utf8') as cfa:
        wf = csv.writer(cfa)
        data2=[[acc,sensitivity,specificity,precision,auc_roc,auc_pr,F1,mcc,metric_logger.loss]]
        for i in data2:
            wf.writerow(i)
            
    
    if mode=='test':
        cm = ConfusionMatrix(actual_vector=true_label_decode_list, predict_vector=prediction_decode_list)
        cm.plot(cmap=plt.cm.Blues,number_label=True,normalized=True,plot_lib="matplotlib")
        plt.savefig(task+'confusion_matrix_test.jpg',dpi=600,bbox_inches ='tight')

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()},auc_roc

@torch.no_grad()
def evaluate_fairness(data_loader, model, device, args, epoch, mode, num_class, k, log_writer, eval_score='', protect_image_names=None, prevalent_image_names=None):
    """Enhanced fairness evaluation with uncertainty quantification and confidence intervals."""
    criterion = nn.CrossEntropyLoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    os.makedirs(os.path.join(args.output_dir, args.task), exist_ok=True)
    
    model.eval()
    true_onehot, pred_onehot, true_labels, pred_labels, pred_softmax = [], [], [], [], []
    file_names = []
    
    for batch in metric_logger.log_every(data_loader, 10, f'{mode}:'):
        images, target = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
        target_onehot = F.one_hot(target.to(torch.int64), num_classes=num_class)
        
        with torch.cuda.amp.autocast():
            output = model(images)
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
        file_names.extend(batch[3])
    
    # Convert to numpy arrays
    true_onehot = np.array(true_onehot)
    pred_onehot = np.array(pred_onehot)
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    pred_softmax = np.array(pred_softmax)
    file_names = np.array(file_names)
    
    # Separate protected and privileged groups
    protect_mask = np.isin(file_names, protect_image_names) if protect_image_names is not None else np.zeros(len(file_names), dtype=bool)
    prevalent_mask = np.isin(file_names, prevalent_image_names) if prevalent_image_names is not None else np.ones(len(file_names), dtype=bool)
    
    protect_labels = true_labels[protect_mask]
    prevalent_labels = true_labels[prevalent_mask]
    protect_preds = pred_labels[protect_mask]
    prevalent_preds = pred_labels[prevalent_mask]
    protect_probs = pred_softmax[protect_mask]
    prevalent_probs = pred_softmax[prevalent_mask]
    protect_gt_onehot = true_onehot[protect_mask]
    prevalent_gt_onehot = true_onehot[prevalent_mask]
    
    print(f"Protected group size: {len(protect_labels)}")
    print(f"Privileged group size: {len(prevalent_labels)}")
    
    # Basic fairness metrics using existing function
    metric_dict = {}
    conf = confusion_matrix(true_labels, pred_labels)
    
    if len(protect_labels) > 0 and len(prevalent_labels) > 0:
        # Original fairness score
        fairness_score = fariness_score(protect_labels, prevalent_labels, protect_preds, prevalent_preds, protect_probs, prevalent_probs, protect_gt_onehot, prevalent_gt_onehot)
        
        print("=== ENHANCED FAIRNESS ANALYSIS ===")
        print(f"Original fairness metrics: {fairness_score}")
        
        # Enhanced analysis with uncertainty quantification and confidence intervals
        from util.uncertainty_fairness_enhanced import UncertaintyQuantifier
        from util.fairness import FairnessAnalyzerWithCI, bootstrap_ci
        
        # ===== UNCERTAINTY QUANTIFICATION =====
        print("\n--- Uncertainty Quantification ---")
        
        # Initialize uncertainty quantifier
        uncertainty_quantifier = UncertaintyQuantifier(
            num_classes=num_class, 
            alpha=getattr(args, 'conformal_alpha', 0.1)
        )
        
        # Split data for calibration and evaluation
        n_samples = len(true_labels)
        cal_split_ratio = getattr(args, 'cal_split_ratio', 0.5)
        n_cal = int(n_samples * cal_split_ratio)
        
        # Random split
        np.random.seed(getattr(args, 'seed', 42))
        indices = np.random.permutation(n_samples)
        cal_indices = indices[:n_cal]
        eval_indices = indices[n_cal:]
        
        cal_probs = pred_softmax[cal_indices]
        cal_labels = true_labels[cal_indices]
        eval_probs = pred_softmax[eval_indices]
        eval_labels = true_labels[eval_indices]
        
        # 1. Reject Option Classification
        roc_strategy = getattr(args, 'roc_strategy', 'accuracy_coverage')
        roc_results = uncertainty_quantifier.reject_option_classification(
            eval_probs, eval_labels, strategy=roc_strategy
        )
        
        # 2. Conformal Prediction
        conformal_results = uncertainty_quantifier.conformal_prediction(
            cal_probs, cal_labels, eval_probs, eval_labels
        )
        
        # 3. Calibration Assessment
        calibration_results = uncertainty_quantifier.calibration_assessment(
            eval_probs, eval_labels, n_bins=getattr(args, 'calibration_bins', 10)
        )
        
        # Add uncertainty metrics to output
        metric_dict.update({
            'uncertainty_roc_threshold': roc_results['optimal_threshold'],
            'uncertainty_roc_coverage': roc_results['coverage'],
            'uncertainty_roc_accuracy': roc_results['accuracy'],
            'uncertainty_roc_rejection_rate': roc_results['rejection_rate'],
            'uncertainty_conformal_coverage': conformal_results['empirical_coverage'],
            'uncertainty_conformal_set_size': conformal_results['average_set_size'],
            'uncertainty_conformal_singleton_rate': conformal_results['singleton_rate'],
            'uncertainty_calibration_ece': calibration_results['ECE'],
            'uncertainty_mean_confidence': calibration_results['mean_confidence']
        })
        
        # ===== FAIRNESS ANALYSIS WITH CONFIDENCE INTERVALS =====
        print("\n--- Fairness Analysis with Confidence Intervals ---")
        
        fairness_analyzer = FairnessAnalyzerWithCI(
            n_bootstrap=getattr(args, 'n_bootstrap', 1000),
            confidence_level=getattr(args, 'confidence_level', 0.95)
        )
        
        # Compute fairness metrics with confidence intervals
        fairness_ci_results = fairness_analyzer.compute_fairness_metrics_with_ci(
            protect_labels, prevalent_labels, protect_preds, prevalent_preds, protect_probs, prevalent_probs, protect_gt_onehot, prevalent_gt_onehot
        )
        
        # Add fairness CI metrics to output
        for metric, diff_data in fairness_ci_results['fairness_differences'].items():
            metric_dict[f'fairness_{metric}_original'] = diff_data['original']
            metric_dict[f'fairness_{metric}_ci_lower'] = diff_data['ci_lower']
            metric_dict[f'fairness_{metric}_ci_upper'] = diff_data['ci_upper']
            metric_dict[f'fairness_{metric}_significant'] = 1.0 if diff_data['is_significant'] else 0.0
        
        # ===== SAVE DETAILED RESULTS =====
        if getattr(args, 'export_detailed_results', True):
            # Create analysis directory
            analysis_dir = os.path.join(args.output_dir, args.task, 'fairness_uncertainty_analysis')
            os.makedirs(analysis_dir, exist_ok=True)
            
            # Save uncertainty results
            uncertainty_results = {
                'reject_option_classification': roc_results,
                'conformal_prediction': conformal_results,
                'calibration_assessment': calibration_results,
                'parameters': {
                    'conformal_alpha': uncertainty_quantifier.alpha,
                    'roc_strategy': roc_strategy,
                    'calibration_bins': getattr(args, 'calibration_bins', 10)
                }
            }
            
            with open(os.path.join(analysis_dir, 'uncertainty_results.json'), 'w') as f:
                json.dump(uncertainty_results, f, indent=2, default=str)
            
            # Save fairness results with CI
            with open(os.path.join(analysis_dir, 'fairness_ci_results.json'), 'w') as f:
                json.dump(fairness_ci_results, f, indent=2, default=str)
            
            # Generate fairness plots with CI
            if getattr(args, 'save_fairness_plots', True):
                plot_path = os.path.join(analysis_dir, 'fairness_metrics_with_ci.png')
                fairness_analyzer.plot_fairness_with_ci(fairness_ci_results, plot_path)
                print(f"Fairness plots saved to: {plot_path}")
            
            print(f"Detailed results saved to: {analysis_dir}")
        
    else:
        print("Warning: Insufficient data for fairness analysis")
        fairness_score = (0,) * 16  # Default empty fairness metrics (now includes AUROC)
    
    # Add original fairness metrics to output
    fairness_metric_names = [
        'protected_PPV', 'privileged_PPV', 'protected_FPR', 'privileged_FPR',
        'protected_TPR', 'privileged_TPR', 'protected_NPV', 'privileged_NPV', 
        'protected_TE', 'privileged_TE', 'protected_FNR', 'privileged_FNR',
        'protected_ACC', 'privileged_ACC', 'protected_AUROC', 'privileged_AUROC'
    ]
    
    for i, metric_name in enumerate(fairness_metric_names):
        if i < len(fairness_score):
            metric_dict[f'fairness_{metric_name}'] = fairness_score[i]
    
    print("=== FAIRNESS ANALYSIS COMPLETE ===")
    
    out_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    out_dict.update(metric_dict)
    return out_dict, None

#train for dual model
def train_one_epoch_dual(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    scheduler = None,
    max_norm: float = 0,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    args=None
):
    """Train the model for one epoch."""
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    print_freq, accum_iter = 20, args.accum_iter
    optimizer.zero_grad()
    all_labels, all_preds, all_probs = [], [], []
    true_onehot, pred_onehot= [], []
    if log_writer:
        print(f'log_dir: {log_writer.log_dir}')
    
    for data_iter_step, data_bs in enumerate(metric_logger.log_every(data_loader, print_freq, f'Epoch: [{epoch}]')):
        if scheduler is None and data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        samples_oct = data_bs[0]
        samples_cfp = data_bs[1]
        targets = data_bs[2]
        samples_oct, samples_cfp, targets = samples_oct.to(device, non_blocking=True), samples_cfp.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        target_onehot = F.one_hot(targets.to(torch.int64), num_classes=args.nb_classes)
        if mixup_fn:
            samples_oct, samples_cfp, targets = mixup_fn(samples_oct, samples_cfp, targets)
        
        with torch.cuda.amp.autocast():
            outputs = model(samples_oct, samples_cfp)
            if hasattr(outputs, 'logits'):
                outputs = outputs.logits
            else:
                outputs = outputs
            loss = criterion(outputs, targets)
            
            # Add regularization loss if specified
            if hasattr(args, 'l1_reg') and hasattr(args, 'l2_reg'):
                if args.l1_reg > 0 or args.l2_reg > 0:
                    reg_loss = compute_regularization_loss(model, args.l1_reg, args.l2_reg)
                    loss = loss + reg_loss
        loss_value = loss.item()
        loss /= accum_iter
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(probs, 1)
        output_onehot = F.one_hot(preds.to(torch.int64), num_classes=args.nb_classes)
        true_onehot.extend(target_onehot.cpu().numpy())
        pred_onehot.extend(output_onehot.detach().cpu().numpy())
        all_labels.extend(targets.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())
        
        loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        
        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss/train', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)
    
    if scheduler is not None and scheduler:
        scheduler.step()
    #Metric
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    conf = confusion_matrix(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    roc_auc = roc_auc_score(true_onehot, all_probs, multi_class='ovr', average='macro')
    f1 = f1_score(all_labels, all_preds, zero_division=0, average='macro')
    kappa = cohen_kappa_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    
    train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    for metric_name, value in zip(['accuracy', 'f1', 'roc_auc', 'kappa', 'mcc'],
                                       [accuracy, f1, roc_auc, kappa, mcc]):
            train_stats[metric_name] = value
    print(f'Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')
    print("confusion_matrix:\n", conf)
    return train_stats


# Train for DuCAN model with three-classifier loss
def train_one_epoch_ducan(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    scheduler=None,
    max_norm: float = 0,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    args=None
):
    """
    Train one epoch for DuCAN model with dual inputs and three classifiers.
    
    Args:
        model: DuCAN model
        criterion: Loss function
        data_loader: Dual-modal data loader
        optimizer: Optimizer
        device: Device
        epoch: Current epoch
        loss_scaler: Loss scaler for mixed precision
        scheduler: Learning rate scheduler
        max_norm: Gradient clipping norm
        mixup_fn: Mixup function
        log_writer: Tensorboard writer
        args: Arguments
    """
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('fundus_loss', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('oct_loss', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('multimodal_loss', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    print_freq, accum_iter = 20, args.accum_iter
    optimizer.zero_grad()
    
    # Track predictions for all three classifiers
    all_labels, all_preds_fundus, all_preds_oct, all_preds_multi = [], [], [], []
    all_probs_fundus, all_probs_oct, all_probs_multi = [], [], []
    true_onehot, pred_onehot_fundus, pred_onehot_oct, pred_onehot_multi = [], [], [], []
    
    if log_writer:
        print(f'log_dir: {log_writer.log_dir}')
    
    # Loss weights for auxiliary and main classifiers (inspired by GoogLeNet)
    # Auxiliary classifiers help improve training but main focus is on multimodal
    fundus_aux_weight = getattr(args, 'fundus_loss_weight', 0.2)  # Auxiliary classifier
    oct_aux_weight = getattr(args, 'oct_loss_weight', 0.2)        # Auxiliary classifier  
    multimodal_weight = getattr(args, 'multimodal_loss_weight', 0.6)  # Main classifier
    
    for data_iter_step, data_bs in enumerate(metric_logger.log_every(data_loader, print_freq, f'Epoch: [{epoch}]')):
        if scheduler is None and data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        # DuCAN expects fundus first, then OCT (as per dataset setup)
        samples_fundus = data_bs[0]
        samples_oct = data_bs[1]
        targets = data_bs[2]
        
        samples_fundus = samples_fundus.to(device, non_blocking=True)
        samples_oct = samples_oct.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        target_onehot = F.one_hot(targets.to(torch.int64), num_classes=args.nb_classes)
        
        if mixup_fn:
            # Note: mixup for dual inputs needs special handling
            samples_fundus, samples_oct, targets = mixup_fn(samples_fundus, samples_oct, targets)
        
        with torch.cuda.amp.autocast():
            # DuCAN forward pass returns dict with three predictions
            outputs = model(samples_fundus, samples_oct)
            
            # Extract predictions from each classifier
            fundus_pred = outputs['fundus']
            oct_pred = outputs['oct']
            multimodal_pred = outputs['multimodal']
            
            # Compute losses for each classifier
            fundus_loss = criterion(fundus_pred, targets)
            oct_loss = criterion(oct_pred, targets)
            multimodal_loss = criterion(multimodal_pred, targets)
            
            # Combined weighted loss with auxiliary classifiers
            total_loss = (fundus_aux_weight * fundus_loss + 
                         oct_aux_weight * oct_loss + 
                         multimodal_weight * multimodal_loss)
            
            # Add regularization loss if specified
            if hasattr(args, 'l1_reg') and hasattr(args, 'l2_reg'):
                if args.l1_reg > 0 or args.l2_reg > 0:
                    reg_loss = compute_regularization_loss(model, args.l1_reg, args.l2_reg)
                    total_loss = total_loss + reg_loss
        
        loss_value = total_loss.item()
        fundus_loss_value = fundus_loss.item()
        oct_loss_value = oct_loss.item()
        multimodal_loss_value = multimodal_loss.item()
        
        total_loss /= accum_iter
        
        # Compute probabilities and predictions for each classifier
        probs_fundus = torch.softmax(fundus_pred, dim=1)
        probs_oct = torch.softmax(oct_pred, dim=1)
        probs_multi = torch.softmax(multimodal_pred, dim=1)
        
        _, preds_fundus = torch.max(probs_fundus, 1)
        _, preds_oct = torch.max(probs_oct, 1)
        _, preds_multi = torch.max(probs_multi, 1)
        
        # One-hot encode predictions
        output_onehot_fundus = F.one_hot(preds_fundus.to(torch.int64), num_classes=args.nb_classes)
        output_onehot_oct = F.one_hot(preds_oct.to(torch.int64), num_classes=args.nb_classes)
        output_onehot_multi = F.one_hot(preds_multi.to(torch.int64), num_classes=args.nb_classes)
        
        # Store results for metrics calculation
        true_onehot.extend(target_onehot.cpu().numpy())
        pred_onehot_fundus.extend(output_onehot_fundus.detach().cpu().numpy())
        pred_onehot_oct.extend(output_onehot_oct.detach().cpu().numpy())
        pred_onehot_multi.extend(output_onehot_multi.detach().cpu().numpy())
        
        all_labels.extend(targets.cpu().numpy())
        all_preds_fundus.extend(preds_fundus.cpu().numpy())
        all_preds_oct.extend(preds_oct.cpu().numpy())
        all_preds_multi.extend(preds_multi.cpu().numpy())
        
        all_probs_fundus.extend(probs_fundus.detach().cpu().numpy())
        all_probs_oct.extend(probs_oct.detach().cpu().numpy())
        all_probs_multi.extend(probs_multi.detach().cpu().numpy())
        
        # Backward pass
        loss_scaler(total_loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), 
                   create_graph=False, update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        
        torch.cuda.synchronize()
        
        # Update metrics
        metric_logger.update(loss=loss_value)
        metric_logger.update(fundus_loss=fundus_loss_value)
        metric_logger.update(oct_loss=oct_loss_value)
        metric_logger.update(multimodal_loss=multimodal_loss_value)
        
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        
        metric_logger.update(lr=max_lr)
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('fundus_loss', fundus_loss_value, epoch_1000x)
            log_writer.add_scalar('oct_loss', oct_loss_value, epoch_1000x)
            log_writer.add_scalar('multimodal_loss', multimodal_loss_value, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)
    
    # Compute final metrics for each classifier
    accuracy_fundus = accuracy_score(all_labels, all_preds_fundus)
    accuracy_oct = accuracy_score(all_labels, all_preds_oct)
    accuracy_multi = accuracy_score(all_labels, all_preds_multi)
    
    f1_fundus = f1_score(true_onehot, pred_onehot_fundus, zero_division=0, average='macro')
    f1_oct = f1_score(true_onehot, pred_onehot_oct, zero_division=0, average='macro')
    f1_multi = f1_score(true_onehot, pred_onehot_multi, zero_division=0, average='macro')
    
    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    train_stats = {
        k: meter.global_avg for k, meter in metric_logger.meters.items()
    }
    train_stats.update({
        'accuracy_fundus': accuracy_fundus,
        'accuracy_oct': accuracy_oct,
        'accuracy_multimodal': accuracy_multi,
        'f1_fundus': f1_fundus,
        'f1_oct': f1_oct,
        'f1_multimodal': f1_multi
    })
    
    # Print confusion matrices for all classifiers
    conf_fundus = confusion_matrix(all_labels, all_preds_fundus)
    conf_oct = confusion_matrix(all_labels, all_preds_oct)
    conf_multi = confusion_matrix(all_labels, all_preds_multi)
    
    print("Fundus classifier confusion matrix:\n", conf_fundus)
    print("OCT classifier confusion matrix:\n", conf_oct)
    print("Multimodal classifier confusion matrix:\n", conf_multi)
    
    return train_stats


# Evaluate for DuCAN model with three classifiers
@torch.no_grad()
def evaluate_ducan(data_loader, model, device, args, epoch, mode, num_class, k, log_writer, eval_score=''):
    """Evaluate DuCAN model with dual inputs and three classifiers."""
    criterion = nn.CrossEntropyLoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    os.makedirs(os.path.join(args.output_dir, args.task), exist_ok=True)
    
    model.eval()
    
    # Track results for all three classifiers
    true_onehot, pred_onehot_fundus, pred_onehot_oct, pred_onehot_multi = [], [], [], []
    true_labels, pred_labels_fundus, pred_labels_oct, pred_labels_multi = [], [], [], []
    pred_softmax_fundus, pred_softmax_oct, pred_softmax_multi = [], [], []
    
    for batch in metric_logger.log_every(data_loader, 10, f'{mode}:'):
        fundus_images = batch[0].to(device, non_blocking=True)
        oct_images = batch[1].to(device, non_blocking=True)
        target = batch[2].to(device, non_blocking=True)
        
        target_onehot = F.one_hot(target.to(torch.int64), num_classes=num_class)
        
        with torch.cuda.amp.autocast():
            outputs = model(fundus_images, oct_images)
            
            fundus_pred = outputs['fundus']
            oct_pred = outputs['oct']
            multimodal_pred = outputs['multimodal']
            
            # Compute losses
            fundus_loss = criterion(fundus_pred, target)
            oct_loss = criterion(oct_pred, target)
            multimodal_loss = criterion(multimodal_pred, target)
            
            # Combined loss for logging
            total_loss = (fundus_loss + oct_loss + multimodal_loss) / 3.0
        
        # Process outputs for each classifier
        fundus_output = nn.Softmax(dim=1)(fundus_pred)
        oct_output = nn.Softmax(dim=1)(oct_pred)
        multi_output = nn.Softmax(dim=1)(multimodal_pred)
        
        fundus_label = fundus_output.argmax(dim=1)
        oct_label = oct_output.argmax(dim=1)
        multi_label = multi_output.argmax(dim=1)
        
        fundus_onehot = F.one_hot(fundus_label.to(torch.int64), num_classes=num_class)
        oct_onehot = F.one_hot(oct_label.to(torch.int64), num_classes=num_class)
        multi_onehot = F.one_hot(multi_label.to(torch.int64), num_classes=num_class)
        
        metric_logger.update(loss=total_loss.item())
        metric_logger.update(fundus_loss=fundus_loss.item())
        metric_logger.update(oct_loss=oct_loss.item())
        metric_logger.update(multimodal_loss=multimodal_loss.item())
        
        # Store results
        true_onehot.extend(target_onehot.cpu().numpy())
        pred_onehot_fundus.extend(fundus_onehot.detach().cpu().numpy())
        pred_onehot_oct.extend(oct_onehot.detach().cpu().numpy())
        pred_onehot_multi.extend(multi_onehot.detach().cpu().numpy())
        
        true_labels.extend(target.cpu().numpy())
        pred_labels_fundus.extend(fundus_label.detach().cpu().numpy())
        pred_labels_oct.extend(oct_label.detach().cpu().numpy())
        pred_labels_multi.extend(multi_label.detach().cpu().numpy())
        
        pred_softmax_fundus.extend(fundus_output.detach().cpu().numpy())
        pred_softmax_oct.extend(oct_output.detach().cpu().numpy())
        pred_softmax_multi.extend(multi_output.detach().cpu().numpy())
    
    # Compute metrics for each classifier
    accuracy_fundus = accuracy_score(true_labels, pred_labels_fundus)
    accuracy_oct = accuracy_score(true_labels, pred_labels_oct)
    accuracy_multi = accuracy_score(true_labels, pred_labels_multi)
    
    f1_fundus = f1_score(true_onehot, pred_onehot_fundus, zero_division=0, average='macro')
    f1_oct = f1_score(true_onehot, pred_onehot_oct, zero_division=0, average='macro')
    f1_multi = f1_score(true_onehot, pred_onehot_multi, zero_division=0, average='macro')
    
    roc_auc_fundus = roc_auc_score(true_onehot, pred_softmax_fundus, multi_class='ovr', average='macro')
    roc_auc_oct = roc_auc_score(true_onehot, pred_softmax_oct, multi_class='ovr', average='macro')
    roc_auc_multi = roc_auc_score(true_onehot, pred_softmax_multi, multi_class='ovr', average='macro')
    
    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    test_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    test_stats.update({
        'accuracy_fundus': accuracy_fundus,
        'accuracy_oct': accuracy_oct,
        'accuracy_multimodal': accuracy_multi,
        'f1_fundus': f1_fundus,
        'f1_oct': f1_oct,
        'f1_multimodal': f1_multi,
        'roc_auc_fundus': roc_auc_fundus,
        'roc_auc_oct': roc_auc_oct,
        'roc_auc_multimodal': roc_auc_multi
    })
    score = (f1_multi + roc_auc_multi) / 2  # Focus on multimodal performance
    
    # Print detailed results
    print(f"=== DuCAN {mode.upper()} Results ===")
    print(f"Fundus Classifier - Accuracy: {accuracy_fundus:.4f}, F1: {f1_fundus:.4f}, ROC-AUC: {roc_auc_fundus:.4f}")
    print(f"OCT Classifier - Accuracy: {accuracy_oct:.4f}, F1: {f1_oct:.4f}, ROC-AUC: {roc_auc_oct:.4f}")
    print(f"Multimodal Classifier - Accuracy: {accuracy_multi:.4f}, F1: {f1_multi:.4f}, ROC-AUC: {roc_auc_multi:.4f}")
    
    # Print confusion matrices
    conf_fundus = confusion_matrix(true_labels, pred_labels_fundus)
    conf_oct = confusion_matrix(true_labels, pred_labels_oct)
    conf_multi = confusion_matrix(true_labels, pred_labels_multi)
    
    print("Fundus confusion matrix:\n", conf_fundus)
    print("OCT confusion matrix:\n", conf_oct)
    print("Multimodal confusion matrix:\n", conf_multi)

    return test_stats, score


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

@torch.no_grad()
def evaluate_dualv2(data_loader, model, device, args, epoch, mode, num_class, k, log_writer, eval_score=''):
    """Evaluate the model."""
    criterion = nn.CrossEntropyLoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    os.makedirs(os.path.join(args.output_dir, args.task), exist_ok=True)
    
    model.eval()
    true_onehot, pred_onehot, true_labels, pred_labels, pred_softmax = [], [], [], [], []
    
    for batch in metric_logger.log_every(data_loader, 10, f'{mode}:'):
        images_oct, images_cfp, target = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True), batch[2].to(device, non_blocking=True)
        target_onehot = F.one_hot(target.to(torch.int64), num_classes=num_class)
        
        with torch.cuda.amp.autocast():
            output = model(images_oct, images_cfp)
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
    
    results_path = os.path.join(args.output_dir, args.task, f'metrics_{mode}.csv')
    file_exists = os.path.isfile(results_path)
    with open(results_path, 'a', newline='', encoding='utf8') as cfa:
        wf = csv.writer(cfa)
        if not file_exists:
            wf.writerow(['val_loss', 'accuracy', 'f1', 'roc_auc', 'hamming', 'jaccard', 'precision', 'recall', 'average_precision', 'kappa', 'mcc'])
        wf.writerow([metric_logger.meters["loss"].global_avg, accuracy, f1, roc_auc, hamming, jaccard, precision, recall, average_precision, kappa, mcc])
    
    if mode == 'test':
        cm = ConfusionMatrix(actual_vector=true_labels, predict_vector=pred_labels)
        cm.plot(cmap=plt.cm.Blues, number_label=True, normalized=True, plot_lib="matplotlib")
        plt.savefig(os.path.join(args.output_dir, args.task, 'confusion_matrix_test.jpg'), dpi=600, bbox_inches='tight')
    
    out_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    out_dict.update(metric_dict)
    return out_dict, score

def _unwrap_module(m: nn.Module) -> nn.Module:
    # DDP / DataParallel
    if hasattr(m, "module"):
        m = m.module  # type: ignore[attr-defined]
    # nn.Sequential etc. (no special unwrapping needed)
    # torch.compile wrapper (PyTorch 2.x)
    if hasattr(m, "_orig_mod"):
        try:
            m = m._orig_mod  # type: ignore[attr-defined]
        except Exception:
            pass
    # Some libs wrap with __wrapped__
    if hasattr(m, "__wrapped__"):
        try:
            m = m.__wrapped__  # type: ignore[attr-defined]
        except Exception:
            pass
    return m


def _fan_in_bound_for_bias(weight: torch.Tensor) -> float:
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    if fan_in > 0:
        return 1.0 / math.sqrt(fan_in)
    return 0.0


def _reset_module_parameters_fallback_(m: nn.Module, emb_std: float) -> None:
    # Linear
    if isinstance(m, nn.Linear):
        if m.weight is not None and m.weight.requires_grad:
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5), nonlinearity="relu")
        if m.bias is not None and m.bias.requires_grad:
            # match PyTorch default: uniform(-b, b) with b = 1/sqrt(fan_in)
            bound = _fan_in_bound_for_bias(m.weight)
            nn.init.uniform_(m.bias, -bound, bound)

    # Convs
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        if m.weight is not None and m.weight.requires_grad:
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None and m.bias.requires_grad:
            nn.init.zeros_(m.bias)

    # Transposed Convs
    elif isinstance(m, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        if m.weight is not None and m.weight.requires_grad:
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None and m.bias.requires_grad:
            nn.init.zeros_(m.bias)

    # Norms
    elif isinstance(
        m,
        (
            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
            nn.LayerNorm, nn.GroupNorm,
            nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
        ),
    ):
        if hasattr(m, "weight") and m.weight is not None and m.weight.requires_grad:
            nn.init.ones_(m.weight)
        if hasattr(m, "bias") and m.bias is not None and m.bias.requires_grad:
            nn.init.zeros_(m.bias)
        # Reset BN running stats if present
        if hasattr(m, "reset_running_stats"):
            try:
                m.reset_running_stats()
            except Exception:
                pass

    # Embeddings
    elif isinstance(m, nn.Embedding):
        if m.weight is not None and m.weight.requires_grad:
            nn.init.normal_(m.weight, mean=0.0, std=emb_std)
            if getattr(m, "padding_idx", None) is not None:
                with torch.no_grad():
                    m.weight[m.padding_idx].fill_(0)

    # RNN family
    elif isinstance(m, (nn.RNN, nn.GRU, nn.LSTM)):
        for name, p in m.named_parameters(recurse=False):
            if p is None or not p.requires_grad:
                continue
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)


def reinit_model_weights_(
    model: nn.Module,
    seed: Optional[int] = None,
    *,
    prefer_library_init: bool = True,
) -> Dict[str, Any]:
    """
    Reinitialize weights of a PyTorch/timm/Transformers model.
    - Uses library-native init methods first when available (Transformers/timm).
    - Falls back to explicit per-module initialization otherwise.
    - Skips frozen parameters (requires_grad=False) and re-ties weights if possible.
    - Clears any existing gradients.

    Returns a dict summary with keys:
      {
        'used_transformers_init': bool,
        'used_timm_init': bool,
        'fallback_applied': bool,
        'embedding_std': float,
        'skipped_params': List[str],
      }
    """
    base = _unwrap_module(model)

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Determine embedding std from HF config if present
    emb_std = 0.02
    cfg = getattr(base, "config", None)
    if cfg is not None and hasattr(cfg, "initializer_range"):
        try:
            emb_std = float(getattr(cfg, "initializer_range"))
        except Exception:
            pass

    summary = {
        "used_transformers_init": False,
        "used_timm_init": False,
        "fallback_applied": False,
        "embedding_std": emb_std,
        "skipped_params": [],
    }

    # Try 🤗 Transformers native init if appropriate
    if prefer_library_init:
        try:
            from transformers import PreTrainedModel  # type: ignore
            if isinstance(base, PreTrainedModel) and hasattr(base, "init_weights") and callable(base.init_weights):
                base.init_weights()
                if hasattr(base, "tie_weights"):
                    try:
                        base.tie_weights()
                    except Exception:
                        pass
                # Clear grads if any
                for n, p in base.named_parameters():
                    if p.grad is not None:
                        p.grad = None
                summary["used_transformers_init"] = True
                return summary
        except Exception:
            # Keep going if transformers isn't installed or model isn't PreTrainedModel
            pass

    # Try timm models' native init if available
    if prefer_library_init:
        try:
            # Many timm models define an `init_weights` method on the top module.
            if hasattr(base, "init_weights") and callable(getattr(base, "init_weights")):
                base.init_weights()
                # Clear grads if any
                for _, p in base.named_parameters():
                    if p.grad is not None:
                        p.grad = None
                summary["used_timm_init"] = True
                # Still run a light fallback pass for any odd layers not covered:
                # (harmless because we skip frozen and use type-appropriate inits)
                base.apply(lambda m: _reset_module_parameters_fallback_(m, emb_std))
                summary["fallback_applied"] = True
                # Re-tie if present (some timm wrappers or custom models expose tie_weights)
                if hasattr(base, "tie_weights"):
                    try:
                        base.tie_weights()
                    except Exception:
                        pass
                return summary
        except Exception:
            pass

    # Generic fallback: per-module resets
    base.apply(lambda m: _reset_module_parameters_fallback_(m, emb_std))

    # As a last pass, ensure *all* trainable parameters are initialized to something reasonable.
    # This catches custom layers that slipped through the cracks.
    for name, p in base.named_parameters():
        if p is None or not p.requires_grad:
            if p is not None and not p.requires_grad:
                summary["skipped_params"].append(name)
            continue
        # Clear grads
        if p.grad is not None:
            p.grad = None
        # Avoid reinitializing tensors we already handled cleanly via module-level rules:
        # If it's 2D+ (weight-like), use Kaiming uniform; else (bias/scale) use zeros.
        try:
            if p.ndim >= 2:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            else:
                nn.init.zeros_(p)
        except Exception:
            # Absolute last resort
            nn.init.normal_(p, mean=0.0, std=emb_std)

    # If there are tied weights or a custom tying hook, call it now.
    if hasattr(base, "tie_weights"):
        try:
            base.tie_weights()
        except Exception:
            pass

    summary["fallback_applied"] = True
    return summary