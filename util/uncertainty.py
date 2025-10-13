"""
Uncertainty Quantification Module

This module implements various uncertainty quantification methods including:
- Reject Option Classification (ROC) with confidence thresholds
- Conformal Prediction for prediction sets with coverage guarantees
- Model calibration assessment and reliability diagrams
- Uncertainty visualization and analysis tools

Author: AI Assistant
Date: 2025-01-13
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class RejectOptionClassifier:
    """
    Reject Option Classification implementation with confidence-based rejection.
    
    Provides multiple strategies for setting rejection thresholds and evaluates
    performance-coverage trade-offs.
    """
    
    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes
        self.optimal_threshold = None
        self.thresholds = None
        self.metrics_by_threshold = {}
        
    def fit_threshold(self, 
                     probabilities: np.ndarray, 
                     true_labels: np.ndarray,
                     strategy: str = 'accuracy_coverage',
                     target_coverage: float = 0.9) -> float:
        """
        Find optimal rejection threshold based on specified strategy.
        
        Args:
            probabilities: Predicted probabilities (n_samples, n_classes)
            true_labels: True class labels (n_samples,)
            strategy: Threshold selection strategy
                - 'accuracy_coverage': Balance accuracy and coverage
                - 'target_coverage': Achieve target coverage
                - 'max_accuracy': Maximize accuracy at reasonable coverage
            target_coverage: Target coverage for 'target_coverage' strategy
            
        Returns:
            Optimal threshold value
        """
        # Calculate confidence scores (max probability)
        confidences = np.max(probabilities, axis=1)
        predictions = np.argmax(probabilities, axis=1)
        
        # Test range of thresholds
        self.thresholds = np.linspace(0.1, 0.99, 50)
        metrics = []
        
        for threshold in self.thresholds:
            # Determine accepted/rejected samples
            accepted_mask = confidences >= threshold
            
            if np.sum(accepted_mask) == 0:
                # No samples accepted
                metrics.append({
                    'threshold': threshold,
                    'coverage': 0.0,
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'n_accepted': 0,
                    'n_rejected': len(true_labels)
                })
                continue
                
            # Calculate metrics on accepted samples
            accepted_preds = predictions[accepted_mask]
            accepted_labels = true_labels[accepted_mask]
            
            coverage = np.sum(accepted_mask) / len(true_labels)
            accuracy = accuracy_score(accepted_labels, accepted_preds)
            
            try:
                precision = precision_score(accepted_labels, accepted_preds, average='macro', zero_division=0)
                recall = recall_score(accepted_labels, accepted_preds, average='macro', zero_division=0)
                f1 = f1_score(accepted_labels, accepted_preds, average='macro', zero_division=0)
            except:
                precision = recall = f1 = 0.0
            
            metrics.append({
                'threshold': threshold,
                'coverage': coverage,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'n_accepted': np.sum(accepted_mask),
                'n_rejected': np.sum(~accepted_mask)
            })
        
        self.metrics_by_threshold = {m['threshold']: m for m in metrics}
        
        # Select optimal threshold based on strategy
        if strategy == 'accuracy_coverage':
            # Maximize accuracy * coverage trade-off
            scores = [m['accuracy'] * m['coverage'] for m in metrics]
            best_idx = np.argmax(scores)
        elif strategy == 'target_coverage':
            # Find threshold closest to target coverage
            coverages = [m['coverage'] for m in metrics]
            best_idx = np.argmin(np.abs(np.array(coverages) - target_coverage))
        elif strategy == 'max_accuracy':
            # Maximize accuracy with coverage >= 0.7
            valid_metrics = [m for m in metrics if m['coverage'] >= 0.7]
            if valid_metrics:
                accuracies = [m['accuracy'] for m in valid_metrics]
                best_idx = np.argmax(accuracies)
                best_threshold = valid_metrics[best_idx]['threshold']
            else:
                best_idx = np.argmax([m['accuracy'] for m in metrics])
                best_threshold = metrics[best_idx]['threshold']
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        if strategy != 'max_accuracy':
            best_threshold = metrics[best_idx]['threshold']
        
        self.optimal_threshold = best_threshold
        return best_threshold
    
    def predict_with_rejection(self, 
                             probabilities: np.ndarray,
                             threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with rejection option.
        
        Args:
            probabilities: Predicted probabilities (n_samples, n_classes)
            threshold: Confidence threshold (uses optimal if None)
            
        Returns:
            predictions: Class predictions (-1 for rejected samples)
            accepted_mask: Boolean mask of accepted samples
        """
        if threshold is None:
            threshold = self.optimal_threshold
            
        if threshold is None:
            raise ValueError("No threshold set. Call fit_threshold first.")
        
        confidences = np.max(probabilities, axis=1)
        predictions = np.argmax(probabilities, axis=1)
        accepted_mask = confidences >= threshold
        
        # Set rejected samples to -1
        predictions[~accepted_mask] = -1
        
        return predictions, accepted_mask
    
    def evaluate_rejection(self, 
                          probabilities: np.ndarray,
                          true_labels: np.ndarray,
                          threshold: Optional[float] = None) -> Dict:
        """
        Evaluate rejection performance at given threshold.
        
        Args:
            probabilities: Predicted probabilities
            true_labels: True class labels
            threshold: Confidence threshold
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions, accepted_mask = self.predict_with_rejection(probabilities, threshold)
        
        if threshold is None:
            threshold = self.optimal_threshold
        
        # Overall statistics
        n_total = len(true_labels)
        n_accepted = np.sum(accepted_mask)
        n_rejected = n_total - n_accepted
        coverage = n_accepted / n_total
        
        if n_accepted == 0:
            return {
                'threshold': threshold,
                'coverage': 0.0,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'n_total': n_total,
                'n_accepted': 0,
                'n_rejected': n_total,
                'rejection_rate': 1.0
            }
        
        # Metrics on accepted samples
        accepted_preds = predictions[accepted_mask]
        accepted_labels = true_labels[accepted_mask]
        
        accuracy = accuracy_score(accepted_labels, accepted_preds)
        precision = precision_score(accepted_labels, accepted_preds, average='macro', zero_division=0)
        recall = recall_score(accepted_labels, accepted_preds, average='macro', zero_division=0)
        f1 = f1_score(accepted_labels, accepted_preds, average='macro', zero_division=0)
        
        return {
            'threshold': threshold,
            'coverage': coverage,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'n_total': n_total,
            'n_accepted': n_accepted,
            'n_rejected': n_rejected,
            'rejection_rate': n_rejected / n_total
        }


class ConformalPredictor:
    """
    Split Conformal Prediction implementation for prediction sets with coverage guarantees.
    
    Provides prediction sets that contain the true label with specified probability.
    """
    
    def __init__(self, alpha: float = 0.1):
        """
        Initialize conformal predictor.
        
        Args:
            alpha: Miscoverage level (1-alpha is target coverage)
        """
        self.alpha = alpha
        self.target_coverage = 1 - alpha
        self.quantile = None
        self.calibration_scores = None
        
    def calibrate(self, 
                 probabilities: np.ndarray, 
                 true_labels: np.ndarray) -> float:
        """
        Calibrate conformal predictor on calibration set.
        
        Args:
            probabilities: Predicted probabilities on calibration set
            true_labels: True labels on calibration set
            
        Returns:
            Calibrated quantile threshold
        """
        # Calculate conformity scores (1 - probability of true class)
        n_samples = len(true_labels)
        scores = []
        
        for i in range(n_samples):
            true_class = true_labels[i]
            true_class_prob = probabilities[i, true_class]
            score = 1 - true_class_prob
            scores.append(score)
        
        self.calibration_scores = np.array(scores)
        
        # Calculate quantile for desired coverage
        # Add 1 to numerator for exact finite-sample guarantee
        quantile_level = np.ceil((n_samples + 1) * (1 - self.alpha)) / n_samples
        quantile_level = min(quantile_level, 1.0)  # Cap at 1.0
        
        self.quantile = np.quantile(self.calibration_scores, quantile_level)
        
        return self.quantile
    
    def predict_sets(self, probabilities: np.ndarray) -> List[List[int]]:
        """
        Generate prediction sets for test samples.
        
        Args:
            probabilities: Predicted probabilities for test samples
            
        Returns:
            List of prediction sets (lists of class indices)
        """
        if self.quantile is None:
            raise ValueError("Conformal predictor not calibrated. Call calibrate() first.")
        
        prediction_sets = []
        n_samples, n_classes = probabilities.shape
        
        for i in range(n_samples):
            # Include classes where 1 - p(class) <= quantile
            # Equivalently: p(class) >= 1 - quantile
            threshold = 1 - self.quantile
            included_classes = []
            
            for class_idx in range(n_classes):
                if probabilities[i, class_idx] >= threshold:
                    included_classes.append(class_idx)
            
            # Ensure at least one class is included
            if len(included_classes) == 0:
                # Include class with highest probability
                best_class = np.argmax(probabilities[i])
                included_classes = [best_class]
            
            prediction_sets.append(included_classes)
        
        return prediction_sets
    
    def evaluate_coverage(self, 
                         probabilities: np.ndarray,
                         true_labels: np.ndarray) -> Dict:
        """
        Evaluate conformal prediction coverage and efficiency.
        
        Args:
            probabilities: Predicted probabilities
            true_labels: True class labels
            
        Returns:
            Dictionary of coverage metrics
        """
        prediction_sets = self.predict_sets(probabilities)
        
        # Calculate coverage (fraction of true labels in prediction sets)
        covered = 0
        set_sizes = []
        
        for i, (true_label, pred_set) in enumerate(zip(true_labels, prediction_sets)):
            if true_label in pred_set:
                covered += 1
            set_sizes.append(len(pred_set))
        
        empirical_coverage = covered / len(true_labels)
        avg_set_size = np.mean(set_sizes)
        
        # Calculate efficiency metrics
        singleton_rate = np.mean([size == 1 for size in set_sizes])
        empty_rate = np.mean([size == 0 for size in set_sizes])
        
        return {
            'target_coverage': self.target_coverage,
            'empirical_coverage': empirical_coverage,
            'coverage_gap': empirical_coverage - self.target_coverage,
            'average_set_size': avg_set_size,
            'singleton_rate': singleton_rate,
            'empty_rate': empty_rate,
            'quantile_threshold': self.quantile,
            'alpha': self.alpha,
            'n_samples': len(true_labels)
        }


class CalibrationAssessment:
    """
    Model calibration assessment and reliability analysis.
    
    Evaluates how well predicted probabilities match empirical frequencies.
    """
    
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        
    def compute_calibration_error(self, 
                                probabilities: np.ndarray,
                                true_labels: np.ndarray,
                                method: str = 'ECE') -> float:
        """
        Compute calibration error metrics.
        
        Args:
            probabilities: Predicted probabilities
            true_labels: True class labels
            method: Calibration error method ('ECE', 'MCE', 'ACE')
            
        Returns:
            Calibration error value
        """
        # Get confidence scores and predictions
        confidences = np.max(probabilities, axis=1)
        predictions = np.argmax(probabilities, axis=1)
        accuracies = (predictions == true_labels).astype(float)
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        errors = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                if method == 'ECE':
                    # Expected Calibration Error
                    error = torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                elif method == 'MCE':
                    # Maximum Calibration Error
                    error = torch.abs(avg_confidence_in_bin - accuracy_in_bin)
                elif method == 'ACE':
                    # Average Calibration Error
                    error = torch.abs(avg_confidence_in_bin - accuracy_in_bin)
                else:
                    raise ValueError(f"Unknown calibration error method: {method}")
                
                errors.append(error.item())
        
        if method == 'ECE' or method == 'ACE':
            return sum(errors)
        elif method == 'MCE':
            return max(errors) if errors else 0.0
    
    def reliability_diagram(self, 
                          probabilities: np.ndarray,
                          true_labels: np.ndarray,
                          save_path: Optional[str] = None) -> Dict:
        """
        Generate reliability diagram and compute calibration metrics.
        
        Args:
            probabilities: Predicted probabilities
            true_labels: True class labels
            save_path: Path to save plot (optional)
            
        Returns:
            Dictionary with calibration metrics and plot data
        """
        # Get confidence scores and predictions
        confidences = np.max(probabilities, axis=1)
        predictions = np.argmax(probabilities, axis=1)
        accuracies = (predictions == true_labels).astype(float)
        
        # Compute calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            accuracies, confidences, n_bins=self.n_bins
        )
        
        # Create reliability diagram
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Reliability diagram
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax1.plot(mean_predicted_value, fraction_of_positives, 'o-', label='Model')
        ax1.set_xlabel('Mean Predicted Probability')
        ax1.set_ylabel('Fraction of Positives')
        ax1.set_title('Reliability Diagram')
        ax1.legend()
        ax1.grid(True)
        
        # Confidence histogram
        ax2.hist(confidences, bins=self.n_bins, alpha=0.7, density=True)
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Density')
        ax2.set_title('Confidence Distribution')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        # Calculate calibration metrics
        ece = self.compute_calibration_error(probabilities, true_labels, 'ECE')
        mce = self.compute_calibration_error(probabilities, true_labels, 'MCE')
        ace = self.compute_calibration_error(probabilities, true_labels, 'ACE')
        
        return {
            'ECE': ece,
            'MCE': mce,
            'ACE': ace,
            'mean_predicted_values': mean_predicted_value.tolist(),
            'fraction_of_positives': fraction_of_positives.tolist(),
            'confidence_distribution': confidences.tolist()
        }


class UncertaintyAnalyzer:
    """
    Comprehensive uncertainty analysis combining multiple methods.
    
    Integrates ROC, Conformal Prediction, and calibration assessment.
    """
    
    def __init__(self, 
                 num_classes: int = 2,
                 alpha: float = 0.1,
                 n_bins: int = 10):
        self.num_classes = num_classes
        self.roc = RejectOptionClassifier(num_classes)
        self.conformal = ConformalPredictor(alpha)
        self.calibration = CalibrationAssessment(n_bins)
        
    def analyze_uncertainty(self, 
                          cal_probabilities: np.ndarray,
                          cal_labels: np.ndarray,
                          test_probabilities: np.ndarray,
                          test_labels: np.ndarray,
                          save_dir: Optional[str] = None) -> Dict:
        """
        Perform comprehensive uncertainty analysis.
        
        Args:
            cal_probabilities: Calibration set probabilities
            cal_labels: Calibration set labels
            test_probabilities: Test set probabilities  
            test_labels: Test set labels
            save_dir: Directory to save plots and results
            
        Returns:
            Comprehensive uncertainty analysis results
        """
        results = {}
        
        # 1. Reject Option Classification
        print("Performing Reject Option Classification analysis...")
        self.roc.fit_threshold(cal_probabilities, cal_labels, strategy='accuracy_coverage')
        roc_results = self.roc.evaluate_rejection(test_probabilities, test_labels)
        results['reject_option'] = roc_results
        
        # 2. Conformal Prediction
        print("Performing Conformal Prediction analysis...")
        self.conformal.calibrate(cal_probabilities, cal_labels)
        conformal_results = self.conformal.evaluate_coverage(test_probabilities, test_labels)
        results['conformal_prediction'] = conformal_results
        
        # 3. Calibration Assessment
        print("Performing Calibration Assessment...")
        calibration_plot_path = None
        if save_dir:
            calibration_plot_path = f"{save_dir}/calibration_reliability.png"
        
        calibration_results = self.calibration.reliability_diagram(
            test_probabilities, test_labels, calibration_plot_path
        )
        results['calibration'] = calibration_results
        
        # 4. Summary statistics
        confidences = np.max(test_probabilities, axis=1)
        predictions = np.argmax(test_probabilities, axis=1)
        
        results['summary'] = {
            'mean_confidence': float(np.mean(confidences)),
            'std_confidence': float(np.std(confidences)),
            'min_confidence': float(np.min(confidences)),
            'max_confidence': float(np.max(confidences)),
            'baseline_accuracy': float(accuracy_score(test_labels, predictions)),
            'n_test_samples': len(test_labels),
            'n_cal_samples': len(cal_labels)
        }
        
        return results
    
    def plot_uncertainty_analysis(self, 
                                results: Dict,
                                save_dir: str) -> None:
        """
        Create comprehensive uncertainty analysis plots.
        
        Args:
            results: Results from analyze_uncertainty
            save_dir: Directory to save plots
        """
        # ROC threshold analysis
        if hasattr(self.roc, 'metrics_by_threshold') and self.roc.metrics_by_threshold:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            thresholds = list(self.roc.metrics_by_threshold.keys())
            coverages = [self.roc.metrics_by_threshold[t]['coverage'] for t in thresholds]
            accuracies = [self.roc.metrics_by_threshold[t]['accuracy'] for t in thresholds]
            f1_scores = [self.roc.metrics_by_threshold[t]['f1'] for t in thresholds]
            
            # Coverage vs Accuracy
            axes[0, 0].plot(coverages, accuracies, 'o-')
            axes[0, 0].set_xlabel('Coverage')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].set_title('ROC: Coverage vs Accuracy')
            axes[0, 0].grid(True)
            
            # Threshold vs Coverage
            axes[0, 1].plot(thresholds, coverages, 'o-')
            axes[0, 1].set_xlabel('Confidence Threshold')
            axes[0, 1].set_ylabel('Coverage')
            axes[0, 1].set_title('ROC: Threshold vs Coverage')
            axes[0, 1].grid(True)
            
            # Threshold vs Accuracy
            axes[1, 0].plot(thresholds, accuracies, 'o-')
            axes[1, 0].set_xlabel('Confidence Threshold')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].set_title('ROC: Threshold vs Accuracy')
            axes[1, 0].grid(True)
            
            # Threshold vs F1
            axes[1, 1].plot(thresholds, f1_scores, 'o-')
            axes[1, 1].set_xlabel('Confidence Threshold')
            axes[1, 1].set_ylabel('F1 Score')
            axes[1, 1].set_title('ROC: Threshold vs F1 Score')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/roc_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Print summary
        print("\n=== UNCERTAINTY ANALYSIS SUMMARY ===")
        print(f"Reject Option Classification:")
        print(f"  Optimal threshold: {results['reject_option']['threshold']:.3f}")
        print(f"  Coverage: {results['reject_option']['coverage']:.3f}")
        print(f"  Accuracy on accepted: {results['reject_option']['accuracy']:.3f}")
        print(f"  Rejection rate: {results['reject_option']['rejection_rate']:.3f}")
        
        print(f"\nConformal Prediction:")
        print(f"  Target coverage: {results['conformal_prediction']['target_coverage']:.3f}")
        print(f"  Empirical coverage: {results['conformal_prediction']['empirical_coverage']:.3f}")
        print(f"  Average set size: {results['conformal_prediction']['average_set_size']:.3f}")
        print(f"  Singleton rate: {results['conformal_prediction']['singleton_rate']:.3f}")
        
        print(f"\nCalibration Assessment:")
        print(f"  ECE: {results['calibration']['ECE']:.4f}")
        print(f"  MCE: {results['calibration']['MCE']:.4f}")
        print(f"  ACE: {results['calibration']['ACE']:.4f}")
        
        print(f"\nSummary Statistics:")
        print(f"  Mean confidence: {results['summary']['mean_confidence']:.3f}")
        print(f"  Baseline accuracy: {results['summary']['baseline_accuracy']:.3f}")
        print(f"  Test samples: {results['summary']['n_test_samples']}")
