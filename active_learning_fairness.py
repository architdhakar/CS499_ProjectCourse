# CORRECTED VERSION OF active_learning_fairness.py
# This file contains the fixes for the algorithm


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Callable
from copy import deepcopy
import random

from fcg_algo_main import (
    calculate_fairness_metrics,
    predict_with_llm,
    stratify_by_groups,
    EPS
)
from dataset import get_llm_probabilities, MODEL_NAME

# ============================================================================
# PROMPT TRUNCATION UTILITIES
# ============================================================================

try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
except:
    tokenizer = None

MAX_PROMPT_LENGTH = 900

def truncate_prompt(prompt: str, max_length: int = MAX_PROMPT_LENGTH) -> str:
    """
    Truncate prompt to fit within model's max sequence length.
    Keeps the most recent demonstrations (end of prompt) which are more relevant.
    """
    if tokenizer is None:
        # Fallback: truncate by character count
        chars_per_token = 4  # Approximate
        max_chars = max_length * chars_per_token
        if len(prompt) > max_chars:
            return prompt[-max_chars:]
        return prompt
    
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if len(tokens) <= max_length:
        return prompt
    
    # Keep the most recent tokens
    truncated_tokens = tokens[-max_length:]
    truncated_prompt = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    return truncated_prompt


class FairnessAwareActiveLearning:
    """
    Iteratively select demonstrations using:
    1. Model uncertainty (entropy-based)
    2. Fairness gap (bias per sample)
    3. Feature diversity (maximize coverage)
    
    NOVELTY: Combines three objectives to find demonstrations that
    - Make the model most confused (high uncertainty)
    - Would reduce the largest demographic biases (high fairness gap)
    - Cover diverse regions of feature space (high diversity)
    """
    
    def __init__(self, 
                 formatter: Callable,
                 label_fn: Callable,
                 sensitive_attr_name: str = "sex",
                 dataset_name: str = "adult"):
        """
        Initialize Fairness-Aware Active Learning.
        
        Args:
            formatter: Function to format row as string
            label_fn: Function to extract label from row
            sensitive_attr_name: Key for sensitive attribute (e.g., "sex")
            dataset_name: Dataset name ("adult" or "credit")
        """
        self.formatter = formatter
        self.label_fn = label_fn
        self.sensitive_attr_name = sensitive_attr_name
        self.dataset_name = dataset_name
        
        self.selection_history = []  # Track all selections
        self.demo_history = []  # Track selected demos and their contributions
        
    def _extract_sensitive_attr(self, row: Dict) -> int:
        """Extract sensitive attribute (0=minority, 1=majority)"""
        if self.dataset_name == "adult":
            return 1 if row.get(self.sensitive_attr_name) == "Male" else 0
        elif self.dataset_name == "credit":
            return 1 if str(row.get(self.sensitive_attr_name, "")) == "1" else 0
        else:
            return 1
    
    def _compute_uncertainty_score(self, 
                                   row: Dict,
                                   demo_prompt: str) -> float:
        """
        Compute uncertainty using entropy of LLM predictions.
        Higher entropy = higher uncertainty (more confused model)
        
        Returns: Entropy value [0, 1]
        """
        try:
            # TRUNCATE demo_prompt to avoid sequence length issues
            demo_prompt = truncate_prompt(demo_prompt, max_length=MAX_PROMPT_LENGTH)
            full_input = demo_prompt + "\nInput: " + self.formatter(row) + "\nIncome:"
            probs = get_llm_probabilities(full_input)
            
            # Entropy = -sum(p * log(p))
            entropy = -sum(p * np.log(p + EPS) for p in probs)
            # Normalize to [0, 1]
            entropy = entropy / np.log(len(probs) + EPS)
            
            return float(entropy)
        except Exception as e:
            print(f"[WARNING] Uncertainty computation failed: {e}")
            return 0.5  # Return neutral uncertainty
    
    def _compute_fairness_gap_score(self,
                                     unlabeled_data: List[Dict],
                                     demo_prompt: str,
                                     sample_size: int = 15) -> np.ndarray:
        """
        Compute per-sample fairness gap score.
        
        Strategy: Ensure balanced representation of labels and demographics
        in the demonstration set.
        
        Returns: Array of fairness gap scores [0, 1]
        """
        fairness_gaps = np.zeros(len(unlabeled_data))
        
        # Count demographic and label representation
        sensitive_attrs = np.array([self._extract_sensitive_attr(row) for row in unlabeled_data])
        
        # Sample to evaluate (for efficiency)
        sample_indices = np.random.choice(
            len(unlabeled_data),
            size=min(sample_size, len(unlabeled_data)),
            replace=False
        )
        
        # Analyze existing demonstrations
        if demo_prompt and len(demo_prompt.strip()) > 10:
            # Count labels in existing demos
            pos_count = demo_prompt.count("Income: Positive")
            neg_count = demo_prompt.count("Income: Negative")
            total_demos = pos_count + neg_count
            
            if total_demos > 0:
                pos_ratio = pos_count / total_demos
                neg_ratio = neg_count / total_demos
            else:
                pos_ratio = 0.5
                neg_ratio = 0.5
        else:
            # No existing demos - start balanced
            pos_ratio = 0.5
            neg_ratio = 0.5
        
        # For each sample, compute fairness score
        for i in sample_indices:
            row = unlabeled_data[i]
            label = self.label_fn(row)
            demo_group = sensitive_attrs[i]
            
            # Score components:
            # 1. Demographic balance: Prefer minority group if underrepresented
            if demo_group == 0:  # Minority
                demographic_score = 1.0  # Always high to encourage minority
            else:
                demographic_score = 0.3  # Lower for majority
            
            # 2. Label balance: Prefer underrepresented labels strongly
            if label == "Positive" and pos_ratio < 0.5:
                label_score = 1.0  # Positive samples are valuable if underrepresented
            elif label == "Negative" and neg_ratio < 0.5:
                label_score = 1.0  # Negative samples are valuable if underrepresented
            else:
                label_score = 0.2  # Low score if label is already well represented
            
            # Combined fairness score (weighted average)
            # Weight label balance higher since it directly affects model accuracy
            fairness_gaps[i] = 0.4 * demographic_score + 0.6 * label_score
        
        return fairness_gaps
    
    def _compute_diversity_score(self,
                                  unlabeled_data: List[Dict],
                                  selected_indices: List[int],
                                  numeric_features: List[str] = None) -> np.ndarray:
        """
        FIX #2: Compute diversity score using proper numeric features.
        
        ORIGINAL BUG: Used hash-based features which aren't semantically meaningful
        FIXED: Extract numeric features properly (age, hours per week, etc.)
        
        For each unselected sample, compute distance to nearest selected sample.
        Higher diversity = larger distance from already selected samples.
        
        Returns: Array of diversity scores [0, 1]
        """
        diversity_scores = np.ones(len(unlabeled_data))
        
        if not selected_indices:
            # All samples equally diverse if nothing selected yet
            return diversity_scores
        
        # Define numeric features based on dataset
        if numeric_features is None:
            if self.dataset_name == "adult":
                numeric_features = ['age', 'hours.per.week']
            elif self.dataset_name == "credit":
                numeric_features = ['age', 'limit_bal']
            else:
                numeric_features = []
        
        def get_features(row):
            """Extract numeric features from row"""
            try:
                feat = []
                for col in numeric_features:
                    try:
                        feat.append(float(row.get(col, 0)))
                    except (ValueError, TypeError):
                        feat.append(0.0)
                return np.array(feat)
            except:
                return np.zeros(len(numeric_features))
        
        # Get features for selected samples
        selected_features = np.array([
            get_features(unlabeled_data[i])
            for i in selected_indices
        ])
        
        # Compute diversity for each sample
        for i in range(len(unlabeled_data)):
            if i in selected_indices:
                diversity_scores[i] = 0  # Already selected, not diverse
                continue
            
            sample_features = get_features(unlabeled_data[i])
            
            # Compute distance to nearest selected sample
            distances = np.linalg.norm(selected_features - sample_features, axis=1)
            min_distance = distances.min() if len(distances) > 0 else 1.0
            
            # Normalize distance to [0, 1]
            # Cap at 1.0 to avoid issues with very large distances
            diversity_scores[i] = min(min_distance / 100.0, 1.0)
        
        return diversity_scores
    
    def active_select(self,
                     unlabeled_data: List[Dict],
                     demo_budget: int = 10,
                     uncertainty_weight: float = 0.4,
                     fairness_weight: float = 0.4,
                     diversity_weight: float = 0.2) -> Tuple[List[Dict], Dict]:
        """
        Iteratively select demonstrations using fairness-aware active learning.
        
        At each iteration:
        1. Compute uncertainty (where is LLM most confused?)
        2. Compute fairness gap (where is LLM most biased?)
        3. Compute diversity (which samples maximize feature coverage?)
        4. Select sample maximizing weighted combination
        5. Add to demonstration set and repeat
        
        Args:
            unlabeled_data: List of candidate samples
            demo_budget: Number of demonstrations to select
            uncertainty_weight: Weight for uncertainty score [0, 1]
            fairness_weight: Weight for fairness gap score [0, 1]
            diversity_weight: Weight for diversity score [0, 1]
        
        Returns:
            (selected_demos, selection_info_dict)
        """
        selected_indices = []
        selected_demos = []
        
        # Normalize weights
        total_weight = uncertainty_weight + fairness_weight + diversity_weight
        uncertainty_weight /= total_weight
        fairness_weight /= total_weight
        diversity_weight /= total_weight
        
        print(f"\n=== FAIRNESS-AWARE ACTIVE LEARNING ===")
        print(f"Budget: {demo_budget} demonstrations")
        print(f"Weights: Uncertainty={uncertainty_weight:.2f}, "
              f"Fairness={fairness_weight:.2f}, Diversity={diversity_weight:.2f}")
        print()
        
        for iteration in range(demo_budget):
            # Build current demonstration prompt
            if selected_demos:
                demo_prompt = "\n".join([
                    self.formatter(demo) + " Income: " + self.label_fn(demo)
                    for demo in selected_demos
                ])
            else:
                demo_prompt = ""
            
            print(f"Iteration {iteration + 1}/{demo_budget}: Computing scores...")
            
            # FIX #3: Uncertainty sampling on subset + proper normalization
            # Subsample candidates to evaluate for uncertainty (cheaper than fairness)
            eval_size = min(15, len(unlabeled_data))
            
            # Only evaluate UNSELECTED samples
            unselected_indices = np.array([i for i in range(len(unlabeled_data)) 
                                          if i not in selected_indices])
            eval_indices = np.random.choice(unselected_indices,
                                           size=min(eval_size, len(unselected_indices)),
                                           replace=False)
            
            # Compute uncertainty scores
            uncertainty_scores = np.zeros(len(unlabeled_data))
            for i in eval_indices:
                uncertainty_scores[i] = self._compute_uncertainty_score(unlabeled_data[i], demo_prompt)
            
            # FIX #3b: Normalize uncertainty only on evaluated samples
            eval_uncertainty = uncertainty_scores[eval_indices]
            if eval_uncertainty.max() > EPS:
                norm_min = eval_uncertainty.min()
                norm_max = eval_uncertainty.max()
                for i in eval_indices:
                    uncertainty_scores[i] = (uncertainty_scores[i] - norm_min) / (norm_max - norm_min + EPS)
            
            # Compute fairness gap scores (all samples, but uses sample_size subset internally)
            fairness_scores = self._compute_fairness_gap_score(unlabeled_data, demo_prompt, sample_size=eval_size)
            
            # Compute diversity scores (all samples)
            diversity_scores = self._compute_diversity_score(unlabeled_data, selected_indices)
            
            # FIX #4: Normalize scores properly
            # Only normalize on unselected samples
            unselected_mask = np.ones(len(unlabeled_data), dtype=bool)
            unselected_mask[selected_indices] = False
            
            # Normalize fairness (only on unselected)
            fair_unselected = fairness_scores[unselected_mask]
            if fair_unselected.max() > EPS:
                norm_min = fair_unselected.min()
                norm_max = fair_unselected.max()
                fairness_scores[unselected_mask] = (fairness_scores[unselected_mask] - norm_min) / (norm_max - norm_min + EPS)
            
            # Normalize diversity (only on unselected)
            div_unselected = diversity_scores[unselected_mask]
            if div_unselected.max() > EPS:
                norm_min = div_unselected.min()
                norm_max = div_unselected.max()
                diversity_scores[unselected_mask] = (diversity_scores[unselected_mask] - norm_min) / (norm_max - norm_min + EPS)
            
            # Compute combined score (only on unselected samples)
            combined_scores = np.zeros(len(unlabeled_data))
            combined_scores[unselected_mask] = (
                uncertainty_weight * uncertainty_scores[unselected_mask] +
                fairness_weight * fairness_scores[unselected_mask] +
                diversity_weight * diversity_scores[unselected_mask]
            )
            
            # Select sample with highest combined score
            best_idx = np.argmax(combined_scores)
            
            if combined_scores[best_idx] < EPS:
                print(f"[WARNING] No valid samples remaining. Stopping early.")
                break
            
            selected_indices.append(best_idx)
            selected_demos.append(unlabeled_data[best_idx])
            
            # Record selection metadata
            selection_info = {
                'iteration': iteration,
                'selected_index': best_idx,
                'uncertainty_score': float(uncertainty_scores[best_idx]),
                'fairness_score': float(fairness_scores[best_idx]),
                'diversity_score': float(diversity_scores[best_idx]),
                'combined_score': float(combined_scores[best_idx]),
                'sensitive_attr': self._extract_sensitive_attr(unlabeled_data[best_idx]),
                'label': self.label_fn(unlabeled_data[best_idx])
            }
            
            self.demo_history.append(selection_info)
            
            print(f"  Selected: "
                  f"Uncertainty={selection_info['uncertainty_score']:.3f}, "
                  f"Fairness={selection_info['fairness_score']:.3f}, "
                  f"Diversity={selection_info['diversity_score']:.3f}, "
                  f"Combined={selection_info['combined_score']:.3f} "
                  f"[Z={selection_info['sensitive_attr']}, Y={selection_info['label']}]")
        
        # Compute summary statistics
        summary = {
            'total_selected': len(selected_demos),
            'iterations': self.demo_history,
            'avg_uncertainty': np.mean([h['uncertainty_score'] for h in self.demo_history]) if self.demo_history else 0,
            'avg_fairness': np.mean([h['fairness_score'] for h in self.demo_history]) if self.demo_history else 0,
            'avg_diversity': np.mean([h['diversity_score'] for h in self.demo_history]) if self.demo_history else 0,
            'minority_selected': sum(1 for h in self.demo_history if h['sensitive_attr'] == 0),
            'majority_selected': sum(1 for h in self.demo_history if h['sensitive_attr'] == 1),
            'positive_selected': sum(1 for h in self.demo_history if h['label'] == 'Positive'),
            'negative_selected': sum(1 for h in self.demo_history if h['label'] == 'Negative')
        }
        
        print(f"\n=== SELECTION SUMMARY ===")
        print(f"Total selected: {summary['total_selected']}")
        print(f"Minority/Majority: {summary['minority_selected']}/{summary['majority_selected']}")
        print(f"Positive/Negative: {summary['positive_selected']}/{summary['negative_selected']}")
        print(f"Avg Uncertainty: {summary['avg_uncertainty']:.3f}")
        print(f"Avg Fairness Impact: {summary['avg_fairness']:.3f}")
        print(f"Avg Diversity: {summary['avg_diversity']:.3f}")
        
        return selected_demos, summary
    
    def compare_with_static_selection(self,
                                      static_demos: List[Dict],
                                      test_data: List[Dict],
                                      active_demos: List[Dict]) -> Dict:
        """
        Compare active learning selection with static selection on test data.
        
        Returns: Dictionary with accuracy and fairness metrics for both
        """
        results = {}
        
        for method_name, demos in [("Static Selection", static_demos), 
                                    ("Active Learning", active_demos)]:
            
            if not demos:
                results[method_name] = None
                continue
            
            # Build prompt
            demo_prompt = "\n".join([
                self.formatter(demo) + " Income: " + self.label_fn(demo)
                for demo in demos
            ])
            
            # TRUNCATE PROMPT to avoid sequence length issues
            demo_prompt = truncate_prompt(demo_prompt, max_length=MAX_PROMPT_LENGTH)
            
            # Evaluate
            predictions = []
            true_labels = []
            sensitive_attrs = []
            
            for row in test_data[:min(300, len(test_data))]:
                try:
                    pred = predict_with_llm(demo_prompt, self.formatter, row)
                except:
                    pred = "Negative"
                
                predictions.append(pred)
                true_labels.append(self.label_fn(row))
                sensitive_attrs.append(self._extract_sensitive_attr(row))
            
            # Calculate metrics
            accuracy = sum(1 for p, t in zip(predictions, true_labels) if p == t) / len(predictions)
            fair_metrics = calculate_fairness_metrics(predictions, true_labels, sensitive_attrs, self.label_fn)
            
            results[method_name] = {
                'accuracy': accuracy,
                'fairness_metrics': fair_metrics,
                'num_demos': len(demos)
            }
        
        return results
    
    def plot_selection_trajectory(self, save_path: str = None):
        """
        Visualize the selection trajectory showing how uncertainty, fairness, 
        and diversity scores evolve across iterations.
        """
        if not self.demo_history:
            print("[WARNING] No selection history to plot")
            return
        
        iterations = [h['iteration'] for h in self.demo_history]
        uncertainty = [h['uncertainty_score'] for h in self.demo_history]
        fairness = [h['fairness_score'] for h in self.demo_history]
        diversity = [h['diversity_score'] for h in self.demo_history]
        combined = [h['combined_score'] for h in self.demo_history]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Individual score trajectories
        ax = axes[0, 0]
        ax.plot(iterations, uncertainty, marker='o', label='Uncertainty', linewidth=2)
        ax.plot(iterations, fairness, marker='s', label='Fairness Gap', linewidth=2)
        ax.plot(iterations, diversity, marker='^', label='Diversity', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Score')
        ax.set_title('Individual Scores Over Iterations')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Combined score
        ax = axes[0, 1]
        ax.plot(iterations, combined, marker='o', color='red', linewidth=2)
        ax.fill_between(iterations, 0, combined, alpha=0.3, color='red')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Combined Score')
        ax.set_title('Combined Score Trajectory')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Demographic distribution
        ax = axes[1, 0]
        minority_samples = [h for h in self.demo_history if h['sensitive_attr'] == 0]
        majority_samples = [h for h in self.demo_history if h['sensitive_attr'] == 1]
        ax.bar(['Minority', 'Majority'], [len(minority_samples), len(majority_samples)], color=['#FF6B6B', '#4ECDC4'])
        ax.set_ylabel('Count')
        ax.set_title('Demographic Distribution of Selected Samples')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Label distribution
        ax = axes[1, 1]
        positive_samples = [h for h in self.demo_history if h['label'] == 'Positive']
        negative_samples = [h for h in self.demo_history if h['label'] == 'Negative']
        ax.bar(['Positive', 'Negative'], [len(positive_samples), len(negative_samples)], color=['#95E1D3', '#F38181'])
        ax.set_ylabel('Count')
        ax.set_title('Label Distribution of Selected Samples')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[INFO] Plot saved to {save_path}")
        else:
            plt.show()
        
        return fig, axes
    
    def plot_comparison(self, comparison_results: Dict, save_path: str = None):
        """
        Visualize comparison between static and active learning selection.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        methods = list(comparison_results.keys())
        
        # Only include methods with results
        methods = [m for m in methods if comparison_results[m] is not None]
        
        if not methods:
            print("[WARNING] No valid results to plot")
            return
        
        # Plot 1: Accuracy comparison
        ax = axes[0, 0]
        accuracies = [comparison_results[m]['accuracy'] for m in methods]
        colors = ['#4ECDC4', '#FF6B6B']
        ax.bar(methods, accuracies, color=colors[:len(methods)])
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Comparison')
        ax.set_ylim([0, 1])
        for i, acc in enumerate(accuracies):
            ax.text(i, acc + 0.02, f'{acc:.3f}', ha='center')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Demographic Parity Ratio (higher is better)
        ax = axes[0, 1]
        dp_ratios = [comparison_results[m]['fairness_metrics']['dp_ratio'] for m in methods]
        ax.bar(methods, dp_ratios, color=colors[:len(methods)])
        ax.set_ylabel('DP Ratio')
        ax.set_title('Demographic Parity Ratio (Higher ↑)')
        ax.set_ylim([0, 1])
        ax.axhline(y=0.8, color='red', linestyle='--', label='Fair Threshold (0.8)')
        for i, dp in enumerate(dp_ratios):
            ax.text(i, dp + 0.02, f'{dp:.3f}', ha='center')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Equalized Odds Ratio (higher is better)
        ax = axes[1, 0]
        eo_ratios = [comparison_results[m]['fairness_metrics']['eo_ratio'] for m in methods]
        ax.bar(methods, eo_ratios, color=colors[:len(methods)])
        ax.set_ylabel('EO Ratio')
        ax.set_title('Equalized Odds Ratio (Higher ↑)')
        ax.set_ylim([0, 1])
        ax.axhline(y=0.8, color='red', linestyle='--', label='Fair Threshold (0.8)')
        for i, eo in enumerate(eo_ratios):
            ax.text(i, eo + 0.02, f'{eo:.3f}', ha='center')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Demographic Parity Difference (lower is better)
        ax = axes[1, 1]
        dp_diffs = [comparison_results[m]['fairness_metrics']['dp_diff'] for m in methods]
        ax.bar(methods, dp_diffs, color=colors[:len(methods)])
        ax.set_ylabel('DP Difference')
        ax.set_title('Demographic Parity Difference (Lower ↓)')
        ax.set_ylim([0, 1])
        for i, dp_diff in enumerate(dp_diffs):
            ax.text(i, dp_diff + 0.02, f'{dp_diff:.3f}', ha='center')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[INFO] Plot saved to {save_path}")
        else:
            plt.show()
        
        return fig, axes
