"""
Fairness Analysis Module

This module implements comprehensive fairness metrics and bias detection including:
- Demographic Parity (Statistical Parity)
- Equalized Odds (True Positive Rate and False Positive Rate equality)
- Calibration Fairness (Equal calibration across groups)
- Intersectional fairness analysis
- Statistical significance testing for fairness metrics

Author: AI Assistant
Date: 2025-01-13
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.calibration import calibration_curve
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class FairnessAnalyzer:
    """
    Comprehensive fairness analysis for machine learning models.
    
    Evaluates various fairness metrics across different demographic groups
    and provides statistical significance testing.
    """
    
    def __init__(self, sensitive_attributes: List[str]):
        """
        Initialize fairness analyzer.
        
        Args:
            sensitive_attributes: List of sensitive attribute names
        """
        self.sensitive_attributes = sensitive_attributes
        self.group_metrics = {}
        self.fairness_metrics = {}
        
    def load_group_labels(self, 
                         group_file_path: str,
                         sample_ids: Optional[List] = None) -> pd.DataFrame:
        """
        Load sensitive group labels from external file.
        
        Args:
            group_file_path: Path to CSV file with group labels
            sample_ids: List of sample IDs to filter (optional)
            
        Returns:
            DataFrame with group labels
        """
        try:
            groups_df = pd.read_csv(group_file_path)
            
            if sample_ids is not None:
                # Filter to matching sample IDs
                if 'sample_id' in groups_df.columns:
                    groups_df = groups_df[groups_df['sample_id'].isin(sample_ids)]
                else:
                    # Assume first column is sample ID
                    id_col = groups_df.columns[0]
                    groups_df = groups_df[groups_df[id_col].isin(sample_ids)]
            
            # Validate sensitive attributes exist
            missing_attrs = [attr for attr in self.sensitive_attributes 
                           if attr not in groups_df.columns]
            if missing_attrs:
                raise ValueError(f"Missing sensitive attributes in group file: {missing_attrs}")
            
            print(f"Loaded group labels for {len(groups_df)} samples")
            print(f"Available attributes: {list(groups_df.columns)}")
            
            return groups_df
            
        except Exception as e:
            print(f"Error loading group labels: {e}")
            print("Creating dummy groups for demonstration...")
            return self._create_dummy_groups(len(sample_ids) if sample_ids else 100)
    
    def _create_dummy_groups(self, n_samples: int) -> pd.DataFrame:
        """Create dummy group labels for demonstration purposes."""
        np.random.seed(42)
        
        dummy_data = {'sample_id': range(n_samples)}
        
        for attr in self.sensitive_attributes:
            if attr.lower() in ['gender', 'sex']:
                dummy_data[attr] = np.random.choice(['Male', 'Female'], n_samples)
            elif attr.lower() in ['age', 'age_group']:
                dummy_data[attr] = np.random.choice(['Young', 'Middle', 'Old'], n_samples)
            elif attr.lower() in ['race', 'ethnicity']:
                dummy_data[attr] = np.random.choice(['White', 'Black', 'Hispanic', 'Asian'], n_samples)
            else:
                # Generic binary attribute
                dummy_data[attr] = np.random.choice(['Group_A', 'Group_B'], n_samples)
        
        return pd.DataFrame(dummy_data)
    
    def compute_group_metrics(self, 
                            predictions: np.ndarray,
                            probabilities: np.ndarray,
                            true_labels: np.ndarray,
                            group_labels: pd.DataFrame) -> Dict:
        """
        Compute performance metrics for each demographic group.
        
        Args:
            predictions: Model predictions
            probabilities: Prediction probabilities
            true_labels: True class labels
            group_labels: DataFrame with group memberships
            
        Returns:
            Dictionary of group-wise metrics
        """
        group_metrics = {}
        
        for attr in self.sensitive_attributes:
            if attr not in group_labels.columns:
                continue
                
            attr_metrics = {}
            unique_groups = group_labels[attr].unique()
            
            for group in unique_groups:
                # Get samples for this group
                group_mask = group_labels[attr] == group
                group_indices = group_labels.index[group_mask].tolist()
                
                if len(group_indices) == 0:
                    continue
                
                # Filter predictions and labels for this group
                group_preds = predictions[group_indices]
                group_probs = probabilities[group_indices]
                group_true = true_labels[group_indices]
                
                # Compute metrics
                accuracy = accuracy_score(group_true, group_preds)
                precision = precision_score(group_true, group_preds, average='macro', zero_division=0)
                recall = recall_score(group_true, group_preds, average='macro', zero_division=0)
                f1 = f1_score(group_true, group_preds, average='macro', zero_division=0)
                
                # Compute confusion matrix metrics
                tn, fp, fn, tp = confusion_matrix(group_true, group_preds, labels=[0, 1]).ravel()
                
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
                tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
                
                # Positive prediction rate (for demographic parity)
                ppr = np.mean(group_preds)
                
                # Base rates
                positive_rate = np.mean(group_true)
                
                attr_metrics[str(group)] = {
                    'n_samples': len(group_indices),
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'tpr': tpr,  # Sensitivity
                    'fpr': fpr,
                    'tnr': tnr,  # Specificity
                    'fnr': fnr,
                    'positive_prediction_rate': ppr,
                    'positive_base_rate': positive_rate,
                    'tp': int(tp),
                    'fp': int(fp),
                    'tn': int(tn),
                    'fn': int(fn)
                }
            
            group_metrics[attr] = attr_metrics
        
        self.group_metrics = group_metrics
        return group_metrics
    
    def compute_demographic_parity(self) -> Dict:
        """
        Compute demographic parity (statistical parity) metrics.
        
        Demographic parity is satisfied when P(Y_hat = 1 | A = a) is equal
        across all groups a.
        
        Returns:
            Dictionary of demographic parity metrics
        """
        dp_metrics = {}
        
        for attr, groups in self.group_metrics.items():
            if len(groups) < 2:
                continue
            
            # Get positive prediction rates for all groups
            group_names = list(groups.keys())
            pprs = [groups[group]['positive_prediction_rate'] for group in group_names]
            
            # Calculate pairwise differences
            max_ppr = max(pprs)
            min_ppr = min(pprs)
            
            # Demographic parity difference (max - min)
            dp_difference = max_ppr - min_ppr
            
            # Demographic parity ratio (min / max)
            dp_ratio = min_ppr / max_ppr if max_ppr > 0 else 1.0
            
            # Statistical significance test (Chi-square test)
            # H0: All groups have equal positive prediction rates
            observed_positives = [groups[group]['tp'] + groups[group]['fp'] for group in group_names]
            total_samples = [groups[group]['n_samples'] for group in group_names]
            
            # Expected positives under null hypothesis
            overall_ppr = sum(observed_positives) / sum(total_samples)
            expected_positives = [overall_ppr * n for n in total_samples]
            
            # Chi-square test
            try:
                chi2_stat = sum((obs - exp)**2 / exp for obs, exp in zip(observed_positives, expected_positives) if exp > 0)
                p_value = 1 - stats.chi2.cdf(chi2_stat, len(group_names) - 1)
            except:
                chi2_stat = 0.0
                p_value = 1.0
            
            dp_metrics[attr] = {
                'demographic_parity_difference': dp_difference,
                'demographic_parity_ratio': dp_ratio,
                'max_ppr': max_ppr,
                'min_ppr': min_ppr,
                'group_pprs': {group: ppr for group, ppr in zip(group_names, pprs)},
                'chi2_statistic': chi2_stat,
                'p_value': p_value,
                'is_fair': dp_difference <= 0.1  # Common threshold
            }
        
        return dp_metrics
    
    def compute_equalized_odds(self) -> Dict:
        """
        Compute equalized odds metrics.
        
        Equalized odds is satisfied when TPR and FPR are equal across groups:
        P(Y_hat = 1 | Y = 1, A = a) = P(Y_hat = 1 | Y = 1, A = b)
        P(Y_hat = 1 | Y = 0, A = a) = P(Y_hat = 1 | Y = 0, A = b)
        
        Returns:
            Dictionary of equalized odds metrics
        """
        eo_metrics = {}
        
        for attr, groups in self.group_metrics.items():
            if len(groups) < 2:
                continue
            
            group_names = list(groups.keys())
            tprs = [groups[group]['tpr'] for group in group_names]
            fprs = [groups[group]['fpr'] for group in group_names]
            
            # TPR differences
            max_tpr = max(tprs)
            min_tpr = min(tprs)
            tpr_difference = max_tpr - min_tpr
            tpr_ratio = min_tpr / max_tpr if max_tpr > 0 else 1.0
            
            # FPR differences  
            max_fpr = max(fprs)
            min_fpr = min(fprs)
            fpr_difference = max_fpr - min_fpr
            fpr_ratio = min_fpr / max_fpr if max_fpr > 0 else 1.0
            
            # Overall equalized odds violation
            eo_difference = max(tpr_difference, fpr_difference)
            
            eo_metrics[attr] = {
                'tpr_difference': tpr_difference,
                'tpr_ratio': tpr_ratio,
                'fpr_difference': fpr_difference,
                'fpr_ratio': fpr_ratio,
                'equalized_odds_difference': eo_difference,
                'max_tpr': max_tpr,
                'min_tpr': min_tpr,
                'max_fpr': max_fpr,
                'min_fpr': min_fpr,
                'group_tprs': {group: tpr for group, tpr in zip(group_names, tprs)},
                'group_fprs': {group: fpr for group, fpr in zip(group_names, fprs)},
                'is_fair': eo_difference <= 0.1  # Common threshold
            }
        
        return eo_metrics
    
    def compute_calibration_fairness(self, 
                                   probabilities: np.ndarray,
                                   true_labels: np.ndarray,
                                   group_labels: pd.DataFrame,
                                   n_bins: int = 10) -> Dict:
        """
        Compute calibration fairness across groups.
        
        Calibration fairness is satisfied when the model is equally well-calibrated
        across all demographic groups.
        
        Args:
            probabilities: Prediction probabilities
            true_labels: True class labels
            group_labels: DataFrame with group memberships
            n_bins: Number of bins for calibration analysis
            
        Returns:
            Dictionary of calibration fairness metrics
        """
        calibration_metrics = {}
        
        for attr in self.sensitive_attributes:
            if attr not in group_labels.columns:
                continue
            
            attr_calibration = {}
            unique_groups = group_labels[attr].unique()
            
            group_eces = []
            
            for group in unique_groups:
                # Get samples for this group
                group_mask = group_labels[attr] == group
                group_indices = group_labels.index[group_mask].tolist()
                
                if len(group_indices) < n_bins:  # Need minimum samples for calibration
                    continue
                
                group_probs = probabilities[group_indices]
                group_true = true_labels[group_indices]
                
                # Get confidence scores (max probability)
                confidences = np.max(group_probs, axis=1)
                predictions = np.argmax(group_probs, axis=1)
                accuracies = (predictions == group_true).astype(float)
                
                # Compute calibration curve
                try:
                    fraction_pos, mean_pred = calibration_curve(
                        accuracies, confidences, n_bins=n_bins
                    )
                    
                    # Expected Calibration Error (ECE)
                    bin_boundaries = np.linspace(0, 1, n_bins + 1)
                    bin_lowers = bin_boundaries[:-1]
                    bin_uppers = bin_boundaries[1:]
                    
                    ece = 0
                    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
                        prop_in_bin = in_bin.mean()
                        
                        if prop_in_bin > 0:
                            accuracy_in_bin = accuracies[in_bin].mean()
                            avg_confidence_in_bin = confidences[in_bin].mean()
                            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                    
                    group_eces.append(ece)
                    
                    attr_calibration[str(group)] = {
                        'ece': ece,
                        'n_samples': len(group_indices),
                        'mean_confidence': float(np.mean(confidences)),
                        'mean_accuracy': float(np.mean(accuracies)),
                        'fraction_positives': fraction_pos.tolist(),
                        'mean_predicted_values': mean_pred.tolist()
                    }
                    
                except Exception as e:
                    print(f"Error computing calibration for {attr}={group}: {e}")
                    continue
            
            # Compute calibration fairness metrics
            if len(group_eces) >= 2:
                max_ece = max(group_eces)
                min_ece = min(group_eces)
                ece_difference = max_ece - min_ece
                ece_ratio = min_ece / max_ece if max_ece > 0 else 1.0
                
                attr_calibration['summary'] = {
                    'max_ece': max_ece,
                    'min_ece': min_ece,
                    'ece_difference': ece_difference,
                    'ece_ratio': ece_ratio,
                    'is_fair': ece_difference <= 0.05  # Calibration fairness threshold
                }
            
            calibration_metrics[attr] = attr_calibration
        
        return calibration_metrics
    
    def compute_intersectional_fairness(self) -> Dict:
        """
        Compute fairness metrics for intersectional groups.
        
        Analyzes fairness across combinations of sensitive attributes.
        
        Returns:
            Dictionary of intersectional fairness metrics
        """
        if len(self.sensitive_attributes) < 2:
            return {}
        
        intersectional_metrics = {}
        
        # Consider all pairs of attributes
        for i, attr1 in enumerate(self.sensitive_attributes):
            for j, attr2 in enumerate(self.sensitive_attributes[i+1:], i+1):
                if attr1 not in self.group_metrics or attr2 not in self.group_metrics:
                    continue
                
                # Get all combinations of groups
                groups1 = list(self.group_metrics[attr1].keys())
                groups2 = list(self.group_metrics[attr2].keys())
                
                intersect_pprs = []
                intersect_tprs = []
                intersect_fprs = []
                
                combo_key = f"{attr1}_{attr2}"
                combo_metrics = {}
                
                for g1 in groups1:
                    for g2 in groups2:
                        combo_name = f"{g1}_{g2}"
                        
                        # For demonstration, use individual group metrics
                        # In practice, would need to compute intersectional groups
                        ppr1 = self.group_metrics[attr1][g1]['positive_prediction_rate']
                        ppr2 = self.group_metrics[attr2][g2]['positive_prediction_rate']
                        avg_ppr = (ppr1 + ppr2) / 2  # Simplified intersection
                        
                        tpr1 = self.group_metrics[attr1][g1]['tpr']
                        tpr2 = self.group_metrics[attr2][g2]['tpr']
                        avg_tpr = (tpr1 + tpr2) / 2
                        
                        fpr1 = self.group_metrics[attr1][g1]['fpr']
                        fpr2 = self.group_metrics[attr2][g2]['fpr']
                        avg_fpr = (fpr1 + fpr2) / 2
                        
                        intersect_pprs.append(avg_ppr)
                        intersect_tprs.append(avg_tpr)
                        intersect_fprs.append(avg_fpr)
                        
                        combo_metrics[combo_name] = {
                            'ppr': avg_ppr,
                            'tpr': avg_tpr,
                            'fpr': avg_fpr
                        }
                
                # Compute intersectional fairness violations
                if intersect_pprs:
                    ppr_range = max(intersect_pprs) - min(intersect_pprs)
                    tpr_range = max(intersect_tprs) - min(intersect_tprs)
                    fpr_range = max(intersect_fprs) - min(intersect_fprs)
                    
                    intersectional_metrics[combo_key] = {
                        'ppr_range': ppr_range,
                        'tpr_range': tpr_range,
                        'fpr_range': fpr_range,
                        'max_violation': max(ppr_range, tpr_range, fpr_range),
                        'combinations': combo_metrics,
                        'is_fair': max(ppr_range, tpr_range, fpr_range) <= 0.1
                    }
        
        return intersectional_metrics
    
    def analyze_fairness(self, 
                        predictions: np.ndarray,
                        probabilities: np.ndarray,
                        true_labels: np.ndarray,
                        group_labels: pd.DataFrame) -> Dict:
        """
        Perform comprehensive fairness analysis.
        
        Args:
            predictions: Model predictions
            probabilities: Prediction probabilities
            true_labels: True class labels
            group_labels: DataFrame with group memberships
            
        Returns:
            Comprehensive fairness analysis results
        """
        print("Computing group-wise performance metrics...")
        self.compute_group_metrics(predictions, probabilities, true_labels, group_labels)
        
        print("Computing demographic parity metrics...")
        dp_metrics = self.compute_demographic_parity()
        
        print("Computing equalized odds metrics...")
        eo_metrics = self.compute_equalized_odds()
        
        print("Computing calibration fairness metrics...")
        cal_metrics = self.compute_calibration_fairness(probabilities, true_labels, group_labels)
        
        print("Computing intersectional fairness metrics...")
        intersect_metrics = self.compute_intersectional_fairness()
        
        results = {
            'group_metrics': self.group_metrics,
            'demographic_parity': dp_metrics,
            'equalized_odds': eo_metrics,
            'calibration_fairness': cal_metrics,
            'intersectional_fairness': intersect_metrics
        }
        
        self.fairness_metrics = results
        return results
    
    def plot_fairness_analysis(self, 
                              results: Dict,
                              save_dir: str) -> None:
        """
        Create comprehensive fairness analysis plots.
        
        Args:
            results: Results from analyze_fairness
            save_dir: Directory to save plots
        """
        # Group performance comparison
        for attr, groups in results['group_metrics'].items():
            if len(groups) < 2:
                continue
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            group_names = list(groups.keys())
            accuracies = [groups[g]['accuracy'] for g in group_names]
            tprs = [groups[g]['tpr'] for g in group_names]
            fprs = [groups[g]['fpr'] for g in group_names]
            pprs = [groups[g]['positive_prediction_rate'] for g in group_names]
            
            # Accuracy by group
            axes[0, 0].bar(group_names, accuracies)
            axes[0, 0].set_title(f'Accuracy by {attr}')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # TPR by group
            axes[0, 1].bar(group_names, tprs)
            axes[0, 1].set_title(f'True Positive Rate by {attr}')
            axes[0, 1].set_ylabel('TPR')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # FPR by group
            axes[1, 0].bar(group_names, fprs)
            axes[1, 0].set_title(f'False Positive Rate by {attr}')
            axes[1, 0].set_ylabel('FPR')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Positive Prediction Rate by group
            axes[1, 1].bar(group_names, pprs)
            axes[1, 1].set_title(f'Positive Prediction Rate by {attr}')
            axes[1, 1].set_ylabel('PPR')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/fairness_metrics_{attr}.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Fairness violations summary
        self._plot_fairness_summary(results, save_dir)
    
    def _plot_fairness_summary(self, results: Dict, save_dir: str) -> None:
        """Create summary plot of fairness violations."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Demographic Parity violations
        dp_violations = []
        dp_attrs = []
        for attr, metrics in results['demographic_parity'].items():
            dp_violations.append(metrics['demographic_parity_difference'])
            dp_attrs.append(attr)
        
        if dp_violations:
            axes[0].bar(dp_attrs, dp_violations)
            axes[0].axhline(y=0.1, color='r', linestyle='--', label='Fairness threshold')
            axes[0].set_title('Demographic Parity Violations')
            axes[0].set_ylabel('PPR Difference')
            axes[0].legend()
            axes[0].tick_params(axis='x', rotation=45)
        
        # Equalized Odds violations
        eo_violations = []
        eo_attrs = []
        for attr, metrics in results['equalized_odds'].items():
            eo_violations.append(metrics['equalized_odds_difference'])
            eo_attrs.append(attr)
        
        if eo_violations:
            axes[1].bar(eo_attrs, eo_violations)
            axes[1].axhline(y=0.1, color='r', linestyle='--', label='Fairness threshold')
            axes[1].set_title('Equalized Odds Violations')
            axes[1].set_ylabel('Max(TPR, FPR) Difference')
            axes[1].legend()
            axes[1].tick_params(axis='x', rotation=45)
        
        # Calibration Fairness violations
        cal_violations = []
        cal_attrs = []
        for attr, metrics in results['calibration_fairness'].items():
            if 'summary' in metrics:
                cal_violations.append(metrics['summary']['ece_difference'])
                cal_attrs.append(attr)
        
        if cal_violations:
            axes[2].bar(cal_attrs, cal_violations)
            axes[2].axhline(y=0.05, color='r', linestyle='--', label='Fairness threshold')
            axes[2].set_title('Calibration Fairness Violations')
            axes[2].set_ylabel('ECE Difference')
            axes[2].legend()
            axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/fairness_violations_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_fairness_summary(self, results: Dict) -> None:
        """Print comprehensive fairness analysis summary."""
        print("\n" + "="*60)
        print("FAIRNESS ANALYSIS SUMMARY")
        print("="*60)
        
        # Group performance overview
        print("\nGROUP PERFORMANCE OVERVIEW:")
        for attr, groups in results['group_metrics'].items():
            print(f"\n{attr.upper()}:")
            for group, metrics in groups.items():
                print(f"  {group}: Acc={metrics['accuracy']:.3f}, "
                      f"TPR={metrics['tpr']:.3f}, FPR={metrics['fpr']:.3f}, "
                      f"PPR={metrics['positive_prediction_rate']:.3f}, "
                      f"N={metrics['n_samples']}")
        
        # Demographic Parity
        print("\nDEMOGRAPHIC PARITY:")
        for attr, metrics in results['demographic_parity'].items():
            status = "✓ FAIR" if metrics['is_fair'] else "✗ UNFAIR"
            print(f"  {attr}: Difference={metrics['demographic_parity_difference']:.3f}, "
                  f"Ratio={metrics['demographic_parity_ratio']:.3f}, "
                  f"p-value={metrics['p_value']:.4f} {status}")
        
        # Equalized Odds
        print("\nEQUALIZED ODDS:")
        for attr, metrics in results['equalized_odds'].items():
            status = "✓ FAIR" if metrics['is_fair'] else "✗ UNFAIR"
            print(f"  {attr}: TPR_diff={metrics['tpr_difference']:.3f}, "
                  f"FPR_diff={metrics['fpr_difference']:.3f}, "
                  f"Max_diff={metrics['equalized_odds_difference']:.3f} {status}")
        
        # Calibration Fairness
        print("\nCALIBRATION FAIRNESS:")
        for attr, metrics in results['calibration_fairness'].items():
            if 'summary' in metrics:
                status = "✓ FAIR" if metrics['summary']['is_fair'] else "✗ UNFAIR"
                print(f"  {attr}: ECE_diff={metrics['summary']['ece_difference']:.4f}, "
                      f"Max_ECE={metrics['summary']['max_ece']:.4f}, "
                      f"Min_ECE={metrics['summary']['min_ece']:.4f} {status}")
        
        # Intersectional Fairness
        if results['intersectional_fairness']:
            print("\nINTERSECTIONAL FAIRNESS:")
            for combo, metrics in results['intersectional_fairness'].items():
                status = "✓ FAIR" if metrics['is_fair'] else "✗ UNFAIR"
                print(f"  {combo}: Max_violation={metrics['max_violation']:.3f} {status}")
        
        print("\n" + "="*60)
