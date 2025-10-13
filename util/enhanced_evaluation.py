"""
Enhanced evaluation functions with uncertainty and fairness analysis.

This module extends the standard evaluation pipeline with:
- Reject Option Classification
- Conformal Prediction
- Model Calibration Assessment  
- Comprehensive Fairness Analysis

Author: AI Assistant
Date: 2025-01-13
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import *
from pycm import ConfusionMatrix
import matplotlib.pyplot as plt
import csv
import os
import json

from engine_finetune import evaluate, evaluate_ducan, evaluate_dualv2


@torch.no_grad()
def evaluate_with_uncertainty_fairness(data_loader, model, device, args, epoch, mode, num_class, k, log_writer, eval_score=''):
    """Enhanced evaluate function with uncertainty and fairness analysis."""
    # First run standard evaluation
    out_dict, score = evaluate(data_loader, model, device, args, epoch, mode, num_class, k, log_writer, eval_score)
    
    # If not in eval mode or no enhanced analysis requested, return standard results
    if not args.eval or (not hasattr(args, 'enable_uncertainty') and not hasattr(args, 'enable_fairness')):
        return out_dict, score
    
    print("\n" + "="*70)
    print("ENHANCED ANALYSIS: UNCERTAINTY & FAIRNESS")
    print("="*70)
    
    # Re-run inference to collect predictions for analysis
    model.eval()
    true_labels, pred_labels, pred_softmax = [], [], []
    
    for batch in data_loader:
        images, target = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            output = model(images)
            if hasattr(output, 'logits'):
                output = output.logits
        
        output_ = nn.Softmax(dim=1)(output)
        output_label = output_.argmax(dim=1)
        
        true_labels.extend(target.cpu().numpy())
        pred_labels.extend(output_label.detach().cpu().numpy())
        pred_softmax.extend(output_.detach().cpu().numpy())
    
    # Convert to numpy arrays
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    pred_softmax = np.array(pred_softmax)
    
    # Enhanced uncertainty analysis
    if hasattr(args, 'enable_uncertainty') and args.enable_uncertainty:
        print("\n" + "="*50)
        print("UNCERTAINTY ANALYSIS")
        print("="*50)
        
        try:
            from util.uncertainty import UncertaintyAnalyzer
            
            # Split test set for calibration and evaluation
            n_samples = len(true_labels)
            n_cal = int(n_samples * args.cal_split_ratio)
            
            # Random split for calibration
            np.random.seed(args.seed)
            indices = np.random.permutation(n_samples)
            cal_indices = indices[:n_cal]
            eval_indices = indices[n_cal:]
            
            cal_probs = pred_softmax[cal_indices]
            cal_labels = true_labels[cal_indices]
            eval_probs = pred_softmax[eval_indices]
            eval_labels = true_labels[eval_indices]
            
            print(f"Calibration set: {len(cal_labels)} samples")
            print(f"Evaluation set: {len(eval_labels)} samples")
            
            # Initialize uncertainty analyzer
            uncertainty_analyzer = UncertaintyAnalyzer(
                num_classes=num_class,
                alpha=args.conformal_alpha,
                n_bins=args.calibration_bins
            )
            
            # Create uncertainty results directory
            uncertainty_dir = os.path.join(args.output_dir, args.task, 'uncertainty_analysis')
            os.makedirs(uncertainty_dir, exist_ok=True)
            
            # Perform uncertainty analysis
            uncertainty_results = uncertainty_analyzer.analyze_uncertainty(
                cal_probs, cal_labels, eval_probs, eval_labels, 
                save_dir=uncertainty_dir if args.save_uncertainty_plots else None
            )
            
            # Add uncertainty results to output dict
            for k, v in uncertainty_results['summary'].items():
                out_dict[f'uncertainty_{k}'] = v
            for k, v in uncertainty_results['reject_option'].items():
                out_dict[f'roc_{k}'] = v
            for k, v in uncertainty_results['conformal_prediction'].items():
                out_dict[f'conformal_{k}'] = v
            for k, v in uncertainty_results['calibration'].items():
                out_dict[f'calibration_{k}'] = v
            
            # Save detailed uncertainty results
            if args.export_detailed_results:
                uncertainty_file = os.path.join(uncertainty_dir, 'uncertainty_results.json')
                with open(uncertainty_file, 'w') as f:
                    json.dump(uncertainty_results, f, indent=2, default=str)
                print(f"Detailed uncertainty results saved to: {uncertainty_file}")
            
        except ImportError as e:
            print(f"Could not import uncertainty module: {e}")
        except Exception as e:
            print(f"Error in uncertainty analysis: {e}")
    
    # Enhanced fairness analysis
    if hasattr(args, 'enable_fairness') and args.enable_fairness:
        print("\n" + "="*50)
        print("FAIRNESS ANALYSIS")
        print("="*50)
        
        try:
            from util.fairness import FairnessAnalyzer
            import pandas as pd
            
            # Initialize fairness analyzer
            fairness_analyzer = FairnessAnalyzer(args.fairness_attributes)
            
            # Load or create group labels
            if args.group_labels_file and os.path.exists(args.group_labels_file):
                group_labels = fairness_analyzer.load_group_labels(args.group_labels_file)
            else:
                print("No group labels file provided, creating dummy groups for demonstration...")
                group_labels = fairness_analyzer.load_group_labels(None, list(range(len(true_labels))))
            
            # Ensure group labels match data size
            if len(group_labels) != len(true_labels):
                print(f"Warning: Group labels size ({len(group_labels)}) != data size ({len(true_labels)})")
                if len(group_labels) > len(true_labels):
                    group_labels = group_labels.iloc[:len(true_labels)]
                else:
                    n_missing = len(true_labels) - len(group_labels)
                    extra_groups = fairness_analyzer._create_dummy_groups(n_missing)
                    group_labels = pd.concat([group_labels, extra_groups], ignore_index=True)
            
            # Perform fairness analysis
            fairness_results = fairness_analyzer.analyze_fairness(
                pred_labels, pred_softmax, true_labels, group_labels
            )
            
            # Create fairness results directory
            fairness_dir = os.path.join(args.output_dir, args.task, 'fairness_analysis')
            os.makedirs(fairness_dir, exist_ok=True)
            
            # Generate fairness plots
            if args.save_fairness_plots:
                fairness_analyzer.plot_fairness_analysis(fairness_results, fairness_dir)
            
            # Print fairness summary
            fairness_analyzer.print_fairness_summary(fairness_results)
            
            # Add key fairness metrics to output dict
            for attr, dp_metrics in fairness_results['demographic_parity'].items():
                out_dict[f'fairness_dp_{attr}_diff'] = dp_metrics['demographic_parity_difference']
                out_dict[f'fairness_dp_{attr}_fair'] = 1.0 if dp_metrics['is_fair'] else 0.0
            
            for attr, eo_metrics in fairness_results['equalized_odds'].items():
                out_dict[f'fairness_eo_{attr}_diff'] = eo_metrics['equalized_odds_difference']
                out_dict[f'fairness_eo_{attr}_fair'] = 1.0 if eo_metrics['is_fair'] else 0.0
            
            # Save detailed fairness results
            if args.export_detailed_results:
                fairness_file = os.path.join(fairness_dir, 'fairness_results.json')
                with open(fairness_file, 'w') as f:
                    json.dump(fairness_results, f, indent=2, default=str)
                print(f"Detailed fairness results saved to: {fairness_file}")
            
        except ImportError as e:
            print(f"Could not import fairness module: {e}")
        except Exception as e:
            print(f"Error in fairness analysis: {e}")
    
    # Generate comprehensive report if both analyses were performed
    if ((hasattr(args, 'enable_uncertainty') and args.enable_uncertainty) or 
        (hasattr(args, 'enable_fairness') and args.enable_fairness)):
        try:
            from util.report_generator import generate_uncertainty_fairness_report
            
            # Extract base performance metrics
            base_performance = {k: v for k, v in out_dict.items() 
                              if k in ['accuracy', 'f1', 'roc_auc', 'precision', 'recall', 'kappa', 'mcc']}
            
            # Get uncertainty and fairness results
            uncertainty_data = None
            fairness_data = None
            
            if hasattr(args, 'enable_uncertainty') and args.enable_uncertainty:
                # Load uncertainty results if they exist
                uncertainty_file = os.path.join(args.output_dir, args.task, 'uncertainty_analysis', 'uncertainty_results.json')
                if os.path.exists(uncertainty_file):
                    with open(uncertainty_file, 'r') as f:
                        uncertainty_data = json.load(f)
            
            if hasattr(args, 'enable_fairness') and args.enable_fairness:
                # Load fairness results if they exist
                fairness_file = os.path.join(args.output_dir, args.task, 'fairness_analysis', 'fairness_results.json')
                if os.path.exists(fairness_file):
                    with open(fairness_file, 'r') as f:
                        fairness_data = json.load(f)
            
            # Generate comprehensive report
            report_path = generate_uncertainty_fairness_report(
                args.output_dir, args.task, uncertainty_data, fairness_data, base_performance, args
            )
            print(f"\n📊 Comprehensive analysis report generated: {report_path}")
            
        except Exception as e:
            print(f"Error generating comprehensive report: {e}")
    
    return out_dict, score


@torch.no_grad()
def evaluate_ducan_with_uncertainty_fairness(data_loader, model, device, args, epoch, mode, num_class, k, log_writer, eval_score=''):
    """Enhanced DuCAN evaluate function with uncertainty and fairness analysis."""
    # First run standard DuCAN evaluation
    out_dict, score = evaluate_ducan(data_loader, model, device, args, epoch, mode, num_class, k, log_writer, eval_score)
    
    # If not in eval mode or no enhanced analysis requested, return standard results
    if not args.eval or (not hasattr(args, 'enable_uncertainty') and not hasattr(args, 'enable_fairness')):
        return out_dict, score
    
    print("\n" + "="*70)
    print("ENHANCED DUCAN ANALYSIS: UNCERTAINTY & FAIRNESS")
    print("="*70)
    
    # Re-run inference to collect predictions for analysis (using multimodal predictions)
    model.eval()
    true_labels, pred_labels, pred_softmax = [], [], []
    
    for batch in data_loader:
        fundus_images = batch[0].to(device, non_blocking=True)
        oct_images = batch[1].to(device, non_blocking=True)
        target = batch[2].to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            outputs = model(fundus_images, oct_images)
            # Use multimodal predictions for uncertainty analysis
            multimodal_pred = outputs['multimodal']
        
        output_ = nn.Softmax(dim=1)(multimodal_pred)
        output_label = output_.argmax(dim=1)
        
        true_labels.extend(target.cpu().numpy())
        pred_labels.extend(output_label.detach().cpu().numpy())
        pred_softmax.extend(output_.detach().cpu().numpy())
    
    # Convert to numpy arrays
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    pred_softmax = np.array(pred_softmax)
    
    # Run uncertainty and fairness analysis (same as standard version)
    if hasattr(args, 'enable_uncertainty') and args.enable_uncertainty:
        try:
            from util.uncertainty import UncertaintyAnalyzer
            
            # Split test set for calibration and evaluation
            n_samples = len(true_labels)
            n_cal = int(n_samples * args.cal_split_ratio)
            
            # Random split for calibration
            np.random.seed(args.seed)
            indices = np.random.permutation(n_samples)
            cal_indices = indices[:n_cal]
            eval_indices = indices[n_cal:]
            
            cal_probs = pred_softmax[cal_indices]
            cal_labels = true_labels[cal_indices]
            eval_probs = pred_softmax[eval_indices]
            eval_labels = true_labels[eval_indices]
            
            print(f"Calibration set: {len(cal_labels)} samples")
            print(f"Evaluation set: {len(eval_labels)} samples")
            
            # Initialize uncertainty analyzer
            uncertainty_analyzer = UncertaintyAnalyzer(
                num_classes=num_class,
                alpha=args.conformal_alpha,
                n_bins=args.calibration_bins
            )
            
            # Create uncertainty results directory
            uncertainty_dir = os.path.join(args.output_dir, args.task, 'uncertainty_analysis')
            os.makedirs(uncertainty_dir, exist_ok=True)
            
            # Perform uncertainty analysis
            uncertainty_results = uncertainty_analyzer.analyze_uncertainty(
                cal_probs, cal_labels, eval_probs, eval_labels, 
                save_dir=uncertainty_dir if args.save_uncertainty_plots else None
            )
            
            # Add uncertainty results to output dict
            for k, v in uncertainty_results['summary'].items():
                out_dict[f'uncertainty_{k}'] = v
            for k, v in uncertainty_results['reject_option'].items():
                out_dict[f'roc_{k}'] = v
            for k, v in uncertainty_results['conformal_prediction'].items():
                out_dict[f'conformal_{k}'] = v
            for k, v in uncertainty_results['calibration'].items():
                out_dict[f'calibration_{k}'] = v
            
            # Save detailed uncertainty results
            if args.export_detailed_results:
                uncertainty_file = os.path.join(uncertainty_dir, 'uncertainty_results.json')
                with open(uncertainty_file, 'w') as f:
                    json.dump(uncertainty_results, f, indent=2, default=str)
                print(f"Detailed uncertainty results saved to: {uncertainty_file}")
            
        except Exception as e:
            print(f"Error in uncertainty analysis: {e}")
    
    # Enhanced fairness analysis (same as standard version)
    if hasattr(args, 'enable_fairness') and args.enable_fairness:
        try:
            from util.fairness import FairnessAnalyzer
            import pandas as pd
            
            # Initialize fairness analyzer
            fairness_analyzer = FairnessAnalyzer(args.fairness_attributes)
            
            # Load or create group labels
            if args.group_labels_file and os.path.exists(args.group_labels_file):
                group_labels = fairness_analyzer.load_group_labels(args.group_labels_file)
            else:
                print("No group labels file provided, creating dummy groups for demonstration...")
                group_labels = fairness_analyzer.load_group_labels(None, list(range(len(true_labels))))
            
            # Ensure group labels match data size
            if len(group_labels) != len(true_labels):
                print(f"Warning: Group labels size ({len(group_labels)}) != data size ({len(true_labels)})")
                if len(group_labels) > len(true_labels):
                    group_labels = group_labels.iloc[:len(true_labels)]
                else:
                    n_missing = len(true_labels) - len(group_labels)
                    extra_groups = fairness_analyzer._create_dummy_groups(n_missing)
                    group_labels = pd.concat([group_labels, extra_groups], ignore_index=True)
            
            # Perform fairness analysis
            fairness_results = fairness_analyzer.analyze_fairness(
                pred_labels, pred_softmax, true_labels, group_labels
            )
            
            # Create fairness results directory
            fairness_dir = os.path.join(args.output_dir, args.task, 'fairness_analysis')
            os.makedirs(fairness_dir, exist_ok=True)
            
            # Generate fairness plots
            if args.save_fairness_plots:
                fairness_analyzer.plot_fairness_analysis(fairness_results, fairness_dir)
            
            # Print fairness summary
            fairness_analyzer.print_fairness_summary(fairness_results)
            
            # Add key fairness metrics to output dict
            for attr, dp_metrics in fairness_results['demographic_parity'].items():
                out_dict[f'fairness_dp_{attr}_diff'] = dp_metrics['demographic_parity_difference']
                out_dict[f'fairness_dp_{attr}_fair'] = 1.0 if dp_metrics['is_fair'] else 0.0
            
            for attr, eo_metrics in fairness_results['equalized_odds'].items():
                out_dict[f'fairness_eo_{attr}_diff'] = eo_metrics['equalized_odds_difference']
                out_dict[f'fairness_eo_{attr}_fair'] = 1.0 if eo_metrics['is_fair'] else 0.0
            
            # Save detailed fairness results
            if args.export_detailed_results:
                fairness_file = os.path.join(fairness_dir, 'fairness_results.json')
                with open(fairness_file, 'w') as f:
                    json.dump(fairness_results, f, indent=2, default=str)
                print(f"Detailed fairness results saved to: {fairness_file}")
            
        except Exception as e:
            print(f"Error in fairness analysis: {e}")
    
    return out_dict, score


@torch.no_grad()
def evaluate_dualv2_with_uncertainty_fairness(data_loader, model, device, args, epoch, mode, num_class, k, log_writer, eval_score=''):
    """Enhanced DualV2 evaluate function with uncertainty and fairness analysis."""
    # First run standard DualV2 evaluation
    out_dict, score = evaluate_dualv2(data_loader, model, device, args, epoch, mode, num_class, k, log_writer, eval_score)
    
    # If not in eval mode or no enhanced analysis requested, return standard results
    if not args.eval or (not hasattr(args, 'enable_uncertainty') and not hasattr(args, 'enable_fairness')):
        return out_dict, score
    
    print("\n" + "="*70)
    print("ENHANCED DUALV2 ANALYSIS: UNCERTAINTY & FAIRNESS")
    print("="*70)
    
    # Re-run inference to collect predictions for analysis
    model.eval()
    true_labels, pred_labels, pred_softmax = [], [], []
    
    for batch in data_loader:
        images_oct, images_cfp, target = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True), batch[2].to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            output = model(images_oct, images_cfp)
            if hasattr(output, 'logits'):
                output = output.logits
        
        output_ = nn.Softmax(dim=1)(output)
        output_label = output_.argmax(dim=1)
        
        true_labels.extend(target.cpu().numpy())
        pred_labels.extend(output_label.detach().cpu().numpy())
        pred_softmax.extend(output_.detach().cpu().numpy())
    
    # Convert to numpy arrays
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    pred_softmax = np.array(pred_softmax)
    
    # Run uncertainty and fairness analysis (same as standard version)
    if hasattr(args, 'enable_uncertainty') and args.enable_uncertainty:
        try:
            from util.uncertainty import UncertaintyAnalyzer
            
            # Split test set for calibration and evaluation
            n_samples = len(true_labels)
            n_cal = int(n_samples * args.cal_split_ratio)
            
            # Random split for calibration
            np.random.seed(args.seed)
            indices = np.random.permutation(n_samples)
            cal_indices = indices[:n_cal]
            eval_indices = indices[n_cal:]
            
            cal_probs = pred_softmax[cal_indices]
            cal_labels = true_labels[cal_indices]
            eval_probs = pred_softmax[eval_indices]
            eval_labels = true_labels[eval_indices]
            
            print(f"Calibration set: {len(cal_labels)} samples")
            print(f"Evaluation set: {len(eval_labels)} samples")
            
            # Initialize uncertainty analyzer
            uncertainty_analyzer = UncertaintyAnalyzer(
                num_classes=num_class,
                alpha=args.conformal_alpha,
                n_bins=args.calibration_bins
            )
            
            # Create uncertainty results directory
            uncertainty_dir = os.path.join(args.output_dir, args.task, 'uncertainty_analysis')
            os.makedirs(uncertainty_dir, exist_ok=True)
            
            # Perform uncertainty analysis
            uncertainty_results = uncertainty_analyzer.analyze_uncertainty(
                cal_probs, cal_labels, eval_probs, eval_labels, 
                save_dir=uncertainty_dir if args.save_uncertainty_plots else None
            )
            
            # Add uncertainty results to output dict
            for k, v in uncertainty_results['summary'].items():
                out_dict[f'uncertainty_{k}'] = v
            for k, v in uncertainty_results['reject_option'].items():
                out_dict[f'roc_{k}'] = v
            for k, v in uncertainty_results['conformal_prediction'].items():
                out_dict[f'conformal_{k}'] = v
            for k, v in uncertainty_results['calibration'].items():
                out_dict[f'calibration_{k}'] = v
            
            # Save detailed uncertainty results
            if args.export_detailed_results:
                uncertainty_file = os.path.join(uncertainty_dir, 'uncertainty_results.json')
                with open(uncertainty_file, 'w') as f:
                    json.dump(uncertainty_results, f, indent=2, default=str)
                print(f"Detailed uncertainty results saved to: {uncertainty_file}")
            
        except Exception as e:
            print(f"Error in uncertainty analysis: {e}")
    
    # Enhanced fairness analysis (same as standard version)
    if hasattr(args, 'enable_fairness') and args.enable_fairness:
        try:
            from util.fairness import FairnessAnalyzer
            import pandas as pd
            
            # Initialize fairness analyzer
            fairness_analyzer = FairnessAnalyzer(args.fairness_attributes)
            
            # Load or create group labels
            if args.group_labels_file and os.path.exists(args.group_labels_file):
                group_labels = fairness_analyzer.load_group_labels(args.group_labels_file)
            else:
                print("No group labels file provided, creating dummy groups for demonstration...")
                group_labels = fairness_analyzer.load_group_labels(None, list(range(len(true_labels))))
            
            # Ensure group labels match data size
            if len(group_labels) != len(true_labels):
                print(f"Warning: Group labels size ({len(group_labels)}) != data size ({len(true_labels)})")
                if len(group_labels) > len(true_labels):
                    group_labels = group_labels.iloc[:len(true_labels)]
                else:
                    n_missing = len(true_labels) - len(group_labels)
                    extra_groups = fairness_analyzer._create_dummy_groups(n_missing)
                    group_labels = pd.concat([group_labels, extra_groups], ignore_index=True)
            
            # Perform fairness analysis
            fairness_results = fairness_analyzer.analyze_fairness(
                pred_labels, pred_softmax, true_labels, group_labels
            )
            
            # Create fairness results directory
            fairness_dir = os.path.join(args.output_dir, args.task, 'fairness_analysis')
            os.makedirs(fairness_dir, exist_ok=True)
            
            # Generate fairness plots
            if args.save_fairness_plots:
                fairness_analyzer.plot_fairness_analysis(fairness_results, fairness_dir)
            
            # Print fairness summary
            fairness_analyzer.print_fairness_summary(fairness_results)
            
            # Add key fairness metrics to output dict
            for attr, dp_metrics in fairness_results['demographic_parity'].items():
                out_dict[f'fairness_dp_{attr}_diff'] = dp_metrics['demographic_parity_difference']
                out_dict[f'fairness_dp_{attr}_fair'] = 1.0 if dp_metrics['is_fair'] else 0.0
            
            for attr, eo_metrics in fairness_results['equalized_odds'].items():
                out_dict[f'fairness_eo_{attr}_diff'] = eo_metrics['equalized_odds_difference']
                out_dict[f'fairness_eo_{attr}_fair'] = 1.0 if eo_metrics['is_fair'] else 0.0
            
            # Save detailed fairness results
            if args.export_detailed_results:
                fairness_file = os.path.join(fairness_dir, 'fairness_results.json')
                with open(fairness_file, 'w') as f:
                    json.dump(fairness_results, f, indent=2, default=str)
                print(f"Detailed fairness results saved to: {fairness_file}")
            
        except Exception as e:
            print(f"Error in fairness analysis: {e}")
    
    return out_dict, score
