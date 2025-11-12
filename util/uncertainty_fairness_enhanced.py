"""
Uncertainty Quantification Module

This module provides uncertainty quantification methods for model evaluation:
- Reject Option Classification with confidence thresholds
- Conformal Prediction for prediction sets with coverage guarantees
- Model calibration assessment

"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.calibration import calibration_curve
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class UncertaintyQuantifier:
    """
    Uncertainty quantification methods for model predictions.
    """
    
    def __init__(self, num_classes: int = 2, alpha: float = 0.1):
        self.num_classes = num_classes
        self.alpha = alpha  # For conformal prediction
        self.target_coverage = 1 - alpha
        
    def reject_option_classification(self, 
                                   probabilities: np.ndarray, 
                                   true_labels: np.ndarray,
                                   strategy: str = 'accuracy_coverage') -> Dict:
        """
        Perform Reject Option Classification analysis.
        
        Args:
            probabilities: Predicted probabilities (n_samples, n_classes)
            true_labels: True class labels (n_samples,)
            strategy: Threshold selection strategy
            
        Returns:
            Dictionary with ROC results
        """
        confidences = np.max(probabilities, axis=1)
        predictions = np.argmax(probabilities, axis=1)
        
        # Test range of thresholds
        thresholds = np.linspace(0.1, 0.99, 50)
        results = []
        
        for threshold in thresholds:
            accepted_mask = confidences >= threshold
            
            if np.sum(accepted_mask) == 0:
                continue
                
            accepted_preds = predictions[accepted_mask]
            accepted_labels = true_labels[accepted_mask]
            
            coverage = np.sum(accepted_mask) / len(true_labels)
            accuracy = accuracy_score(accepted_labels, accepted_preds)
            
            results.append({
                'threshold': threshold,
                'coverage': coverage,
                'accuracy': accuracy,
                'n_accepted': np.sum(accepted_mask),
                'n_rejected': np.sum(~accepted_mask)
            })
        
        # Select optimal threshold based on strategy
        if strategy == 'accuracy_coverage':
            scores = [r['accuracy'] * r['coverage'] for r in results]
            best_idx = np.argmax(scores)
        elif strategy == 'target_coverage':
            coverages = [r['coverage'] for r in results]
            best_idx = np.argmin(np.abs(np.array(coverages) - 0.8))
        else:
            scores = [r['accuracy'] for r in results if r['coverage'] >= 0.7]
            if scores:
                best_idx = np.argmax([r['accuracy'] for r in results if r['coverage'] >= 0.7])
            else:
                best_idx = np.argmax([r['accuracy'] for r in results])
        
        optimal_result = results[best_idx]
        optimal_result['rejection_rate'] = optimal_result['n_rejected'] / len(true_labels)
        
        return {
            'optimal_threshold': optimal_result['threshold'],
            'coverage': optimal_result['coverage'],
            'accuracy': optimal_result['accuracy'],
            'rejection_rate': optimal_result['rejection_rate'],
            'n_accepted': optimal_result['n_accepted'],
            'n_rejected': optimal_result['n_rejected'],
            'all_thresholds': results
        }
    
    def conformal_prediction(self, 
                           cal_probabilities: np.ndarray,
                           cal_labels: np.ndarray,
                           test_probabilities: np.ndarray,
                           test_labels: np.ndarray) -> Dict:
        """
        Perform Conformal Prediction analysis.
        
        Args:
            cal_probabilities: Calibration set probabilities
            cal_labels: Calibration set labels
            test_probabilities: Test set probabilities
            test_labels: Test set labels
            
        Returns:
            Dictionary with conformal prediction results
        """
        # Calculate conformity scores on calibration set
        n_cal = len(cal_labels)
        cal_scores = []
        
        for i in range(n_cal):
            true_class = cal_labels[i]
            true_class_prob = cal_probabilities[i, true_class]
            score = 1 - true_class_prob
            cal_scores.append(score)
        
        cal_scores = np.array(cal_scores)
        
        # Calculate quantile for desired coverage
        quantile_level = np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal
        quantile_level = min(quantile_level, 1.0)
        quantile = np.quantile(cal_scores, quantile_level)
        
        # Generate prediction sets for test samples
        prediction_sets = []
        n_test = len(test_labels)
        
        for i in range(n_test):
            threshold = 1 - quantile
            included_classes = []
            
            for class_idx in range(self.num_classes):
                if test_probabilities[i, class_idx] >= threshold:
                    included_classes.append(class_idx)
            
            if len(included_classes) == 0:
                best_class = np.argmax(test_probabilities[i])
                included_classes = [best_class]
            
            prediction_sets.append(included_classes)
        
        # Evaluate coverage and efficiency
        covered = 0
        set_sizes = []
        
        for i, (true_label, pred_set) in enumerate(zip(test_labels, prediction_sets)):
            if true_label in pred_set:
                covered += 1
            set_sizes.append(len(pred_set))
        
        empirical_coverage = covered / len(test_labels)
        avg_set_size = np.mean(set_sizes)
        singleton_rate = np.mean([size == 1 for size in set_sizes])
        
        return {
            'target_coverage': self.target_coverage,
            'empirical_coverage': empirical_coverage,
            'coverage_gap': empirical_coverage - self.target_coverage,
            'average_set_size': avg_set_size,
            'singleton_rate': singleton_rate,
            'quantile_threshold': quantile,
            'prediction_sets': prediction_sets,
            'n_calibration': n_cal,
            'n_test': n_test
        }
    
    def calibration_assessment(self, 
                             probabilities: np.ndarray,
                             true_labels: np.ndarray,
                             n_bins: int = 20) -> Dict:
        """
        Assess model calibration.
        
        Args:
            probabilities: Predicted probabilities (n_samples, n_classes)
            true_labels: True class labels (n_samples,)
            n_bins: Number of bins for calibration
            
        Returns:
            Dictionary with calibration metrics
        """
        confidences = np.max(probabilities, axis=1)
        predictions = np.argmax(probabilities, axis=1)
        accuracies = (predictions == true_labels).astype(float)
        
        # Compute calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            accuracies, confidences, n_bins=n_bins
        )
        
        # Calculate Expected Calibration Error (ECE)
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
        
        return {
            'ECE': ece,
            'mean_confidence': float(np.mean(confidences)),
            'mean_accuracy': float(np.mean(accuracies)),
            'fraction_positives': fraction_of_positives.tolist(),
            'mean_predicted_values': mean_predicted_value.tolist(),
            'confidence_distribution': confidences.tolist()
        }
