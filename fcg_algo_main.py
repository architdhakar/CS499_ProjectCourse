import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
import random
import math
from copy import deepcopy
from dataset import get_llm_probabilities, label_map, load_adult as load_adult_dataset


K = 4  # Number of demonstrations to select (reduced from 8)
CLUSTERS_PER_GROUP = 5  # Reduced from 10
GENETIC_ITER = 3  # Number of generations (DRASTICALLY reduced from 100 for testing)
POP_SIZE = 20  # Population size
MUTATION_RATE = 0.1  # Probability of mutation per individual
CROSSOVER_RATE = 0.8  # Probability of crossover
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# --- 1. DATA PREPARATION & STRATIFICATION ---

def prepare_data(dataset_name="adult"):
    """Load and prepare demographic/tabular dataset depending on name"""
    if dataset_name == "adult":
        from dataset import load_adult as load_dataset
        # For Adult: sensitive_attr_name = "sex", minority="Female", majority="Male"
    elif dataset_name == "credit":
        from dataset import load_credit as load_dataset
        # Credit uses "sex" assuming 2 as Female (minority typically for these datasets) and 1 as Male (majority)
    elif dataset_name == "civil":
        from dataset import load_civil_comments as load_dataset
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported or defined.")
        
    train_dataset, test_dataset, formatter, label_fn = load_dataset()
    
    # Convert datasets to lists
    train_data = [train_dataset[i] for i in range(len(train_dataset))]
    test_data = [test_dataset[i] for i in range(len(test_dataset))]
    
    return train_data, test_data, formatter, label_fn


def stratify_by_groups(data, label_fn, sensitive_attr_name="sex", dataset_name="adult"):
    """
    Stratify data into 4 subgroups based on sensitive attribute and label.
    
    Args:
        data: List of data rows
        label_fn: Function to extract label from row
        sensitive_attr_name: Key for sensitive attribute (default "sex")
    
    Returns dict with keys:
    - (0, 0): Minority (z=0), Negative label (y=0)
    - (0, 1): Minority (z=0), Positive label (y=1)
    - (1, 0): Majority (z=1), Negative label (y=0)
    - (1, 1): Majority (z=1), Positive label (y=1)
    """
    subgroups = {(0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []}
    
    for i, row in enumerate(data):
        # Get sensitive attribute (0=minority, 1=majority)
        if dataset_name == "adult":
            z = 1 if row.get("sex") == "Male" else 0
        elif dataset_name == "credit":
             # Assuming 1 is male, 2 is female in credit dataset 
             sex_val = str(row.get("sex", row.get("SEX", "")))
             z = 1 if sex_val == "1" or sex_val == "1.0" else 0
        elif dataset_name == "civil":
             z = 1 if float(row.get("identity_attack", 0.0)) > 0.3 else 0
        else:
             z = 1 # default
        
        # Get label using label_fn (0=negative, 1=positive)
        label = label_fn(row)
        y = 1 if label == "Positive" else 0
        
        subgroups[(z, y)].append(row)
        
        # Add a debug print for the first few samples to cross verify
        if i < 5:
            print(f"[DEBUG Stratification] Row {i}: original sensitive='{row.get(sensitive_attr_name)}' -> z={z}, original label_fn='{label}' -> y={y}")
    
    return subgroups


# --- 2. FAIRNESS METRICS CALCULATION ---

EPS = 1e-12

def calculate_fairness_metrics(predictions, true_labels, sensitive_attrs, label_fn):
    """
    Calculate fairness metrics: demographic parity and equalized odds.
    
    Returns dict with:
    - dp_ratio (Rdp): Higher is fairer [0, 1]
    - eo_ratio (Reo): Higher is fairer [0, 1]
    - dp_diff (Δdp): Lower is fairer [0, 1]
    - eo_diff (Δeo): Lower is fairer [0, 1]
    """
    # Convert to numpy arrays for easier manipulation
    preds = np.array([1 if p == "Positive" else 0 for p in predictions])
    labels = np.array([1 if l == "Positive" else 0 for l in true_labels])
    sens = np.array(sensitive_attrs)
    
    # Separate by sensitive attribute
    minority_mask = (sens == 0)
    majority_mask = (sens == 1)
    
    # Demographic Parity
    if minority_mask.sum() > 0:
        dp_minority = float((preds[minority_mask] == 1).sum()) / float(minority_mask.sum())
    else:
        dp_minority = 0.0
        
    if majority_mask.sum() > 0:
        dp_majority = float((preds[majority_mask] == 1).sum()) / float(majority_mask.sum())
    else:
        dp_majority = 0.0
    
    dp_diff = abs(float(dp_majority) - float(dp_minority))
    
    # Avoid division by zero in ratio
    if max(dp_minority, dp_majority) < EPS:
        dp_ratio = 1.0  # Perfect fairness if both groups have no positive predictions
    else:
        dp_ratio = min(dp_minority, dp_majority) / (max(dp_minority, dp_majority) + EPS)
    
    # Equalized Odds (TPR and FPR)
    positive_mask = (labels == 1)
    negative_mask = (labels == 0)
    
    # TPR for minority
    minor_pos_mask = minority_mask & positive_mask
    if minor_pos_mask.sum() > 0:
        tpr_minority = float((preds[minor_pos_mask] == 1).sum()) / float(minor_pos_mask.sum())
    else:
        tpr_minority = 0.0
    
    # TPR for majority
    major_pos_mask = majority_mask & positive_mask
    if major_pos_mask.sum() > 0:
        tpr_majority = float((preds[major_pos_mask] == 1).sum()) / float(major_pos_mask.sum())
    else:
        tpr_majority = 0.0
    
    # FPR for minority
    minor_neg_mask = minority_mask & negative_mask
    if minor_neg_mask.sum() > 0:
        fpr_minority = float((preds[minor_neg_mask] == 1).sum()) / float(minor_neg_mask.sum())
    else:
        fpr_minority = 0.0
    
    # FPR for majority
    major_neg_mask = majority_mask & negative_mask
    if major_neg_mask.sum() > 0:
        fpr_majority = float((preds[major_neg_mask] == 1).sum()) / float(major_neg_mask.sum())
    else:
        fpr_majority = 0.0
    
    tpr_diff = abs(float(tpr_majority) - float(tpr_minority))
    fpr_diff = abs(float(fpr_majority) - float(fpr_minority))
    eo_diff = max(tpr_diff, fpr_diff)
    
    # Compute EO ratio carefully
    if (tpr_majority + EPS) < EPS and (fpr_majority + EPS) < EPS:
        eo_ratio = 1.0  # Perfect fairness if denominator is 0
    else:
        tpr_ratio = tpr_minority / (tpr_majority + EPS) if (tpr_majority + EPS) > EPS else 1.0
        fpr_ratio = fpr_minority / (fpr_majority + EPS) if (fpr_majority + EPS) > EPS else 1.0
        eo_ratio = min(tpr_ratio, fpr_ratio)
    
    return {
        'dp_ratio': float(dp_ratio),
        'eo_ratio': float(eo_ratio),
        'dp_diff': float(dp_diff),
        'eo_diff': float(eo_diff),
        'dp_minority': float(dp_minority),
        'dp_majority': float(dp_majority),
        'tpr_minority': float(tpr_minority),
        'tpr_majority': float(tpr_majority),
        'fpr_minority': float(fpr_minority),
        'fpr_majority': float(fpr_majority)
    }


def predict_with_llm(prompt, formatter, row):
    """
    Get LLM prediction for a single row.
    
    IMPORTANT: get_llm_probabilities returns probs in the order of label_map.keys()
    (insertion order: Negative, Positive), NOT in sorted order by token ID.
    
    Args:
        prompt: Demonstration prompt (e.g., "formatted_sample1 Positive\nformatted_sample2 Negative\n...")
        formatter: Function to format row
        row: Data row to predict for
    """
    # Build the full input: demonstrations + test sample + label prompt
    if prompt:
        full_input = prompt + "\n" + formatter(row) + "\nIncome:"
    else:
        full_input = formatter(row) + "\nIncome:"
    
    probs = get_llm_probabilities(full_input)
    
    # Probs are in the order of label_map dictionary keys (Negative, Positive)
    label_order = list(label_map.keys())  # This is: ['Negative', 'Positive']
    
    best_idx = probs.index(max(probs))
    return label_order[best_idx]


def evaluate_demos(demo_indices, demo_data, val_data, formatter, label_fn, alpha=0.5):
    """
    Evaluate a set of demonstrations on validation data.
    
    Returns:
    - pred_score: Prediction accuracy
    - fair_score: Fairness score (Rdp)
    - combined_score: α·ΔPred + (1-α)·ΔFair
    """
    # Check if we have valid demonstrations
    # Handle both list and numpy array types
    demo_count = len(demo_indices) if hasattr(demo_indices, '__len__') else 0
    if demo_count == 0:
        print("  WARNING: No demonstrations to evaluate!")
        return 0.0, 0.0, 0.05
    
    # Build prompt from demonstrations
    prompt_parts = []
    for idx in demo_indices:
        if idx >= len(demo_data):
            print(f"  WARNING: Index {idx} out of range for demo_data (len={len(demo_data)})")
            continue
        row = demo_data[idx]
        label = label_fn(row)
        demo_text = formatter(row) + " " + label
        prompt_parts.append(demo_text)
    
    if not prompt_parts:
        print("  WARNING: Could not build any demonstrations from indices!")
        return 0.0, 0.0, 0.05
    
    demo_prompt = "\n".join(prompt_parts)
    
    # Evaluate on validation data
    predictions = []
    true_labels = []
    sensitive_attrs = []
    
    num_eval = min(20, len(val_data))  # Evaluate on up to 20 samples (reduced from 100)
    
    for i in range(num_eval):
        row = val_data[i]
        true_label = label_fn(row)
        try:
            pred = predict_with_llm(demo_prompt, formatter, row)
        except Exception as e:
            print(f"  WARNING: Prediction error at sample {i}: {e}")
            pred = "Negative"  # Default
        
        predictions.append(pred)
        true_labels.append(true_label)
        # Extract sensitive attribute (gender)
        z = 1 if row.get("sex") == "Male" else 0
        sensitive_attrs.append(z)
    
    # Calculate metrics
    correct = sum(1 for p, t in zip(predictions, true_labels) if p == t)
    pred_score = correct / num_eval if num_eval > 0 else 0.0
    
    fair_metrics = calculate_fairness_metrics(predictions, true_labels, sensitive_attrs, label_fn)
    fair_score = fair_metrics['eo_ratio']  # Use Reo as fairness metric
    
    # Baseline (zero-shot)
    baseline_pred = 0.5
    baseline_fair = 0.2
    
    delta_pred = max((pred_score - baseline_pred), 0.05)
    delta_fair = max((fair_score - baseline_fair), 0.05)
    
    combined_score = alpha * delta_pred + (1 - alpha) * delta_fair
    
    return pred_score, fair_score, combined_score


# --- 3. FCG ALGORITHM ---

class FCGAlgorithm:
    def __init__(self, train_data, val_data, formatter, label_fn, 
                 n_clusters=8, m_neighbors=5, iters=10, k=K, alpha=0.5, initial_score=0.05):
        """
        Initialize FCG algorithm parameters.
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset
            formatter: Function to format row as string
            label_fn: Function to extract label from row
            n_clusters: Number of K-means clusters per subgroup
            m_neighbors: Number of neighbors to select per cluster center
            iters: Number of genetic algorithm iterations
            k: Number of demonstrations to select
            alpha: Trade-off between prediction and fairness
            initial_score: Initial score for new samples
        """
        self.train_data = train_data
        self.val_data = val_data
        self.formatter = formatter
        self.label_fn = label_fn
        self.n_clusters = n_clusters
        self.m_neighbors = m_neighbors
        self.iters = iters
        self.k = k
        self.alpha = alpha
        self.initial_score = initial_score
        
        # Store subgroups and their scores
        self.subgroups = {}
        self.subgroup_scores = {}
        self.selected_demos = {}
    
    def step1_clustering(self, dataset_name="adult"):
        """
        Step 1: Diverse Clustering
        Reduce candidate pool while maintaining diversity.
        """
        print("\n=== STEP 1: DIVERSE CLUSTERING ===")
        
        # Stratify data using label_fn
        sg = stratify_by_groups(self.train_data, self.label_fn, dataset_name=dataset_name)
        
        # Cluster each subgroup
        for (z, y), subgroup in sg.items():
            print(f"\nProcessing subgroup (z={z}, y={y}): {len(subgroup)} samples")
            
            if len(subgroup) == 0:
                self.subgroups[(z, y)] = []
                continue
            
            # Print sample to verify the content in subgroup
            print(f"[DEBUG] Sample 1 from subgroup (z={z}, y={y}):")
            print(f"        {self.formatter(subgroup[0]).replace(chr(10), ' | ')}")
            
            # Convert to feature vectors for clustering
            # Use embedding-based or simple feature extraction
            # For now, use a simple approach: represent each row by its numeric features
            try:
                X = np.array([
                    [row.get('age', 0), 
                     hash(row.get('workclass', '')) % 100,
                     hash(row.get('education', '')) % 100]
                    for row in subgroup
                ])
            except:
                # Fallback: random features
                X = np.random.randn(len(subgroup), 3)
            
            # K-means clustering
            n_clusters = min(self.n_clusters, len(subgroup))
            kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
            kmeans.fit(X)
            
            # Select m nearest neighbors for each cluster center
            selected_indices = []
            for center_idx in range(n_clusters):
                distances = np.linalg.norm(X - kmeans.cluster_centers_[center_idx], axis=1)
                nearest_indices = np.argsort(distances)[:self.m_neighbors]
                selected_indices.extend(nearest_indices)
            
            # Remove duplicates while preserving order
            selected_indices = list(dict.fromkeys(selected_indices))
            selected_subgroup = [subgroup[i] for i in selected_indices]
            
            self.subgroups[(z, y)] = selected_subgroup
            self.subgroup_scores[(z, y)] = [self.initial_score] * len(selected_subgroup)
            
            print(f"  → Reduced to {len(selected_subgroup)} candidates (n={n_clusters}, m={self.m_neighbors})")
    
    def step2_genetic_evolution(self):
        """
        Step 2: Genetic Evolution with Score Updates
        Iteratively select and score demonstrations using roulette wheel selection.
        """
        print("\n=== STEP 2: GENETIC EVOLUTION ===")
        
        for (z, y), subgroup in self.subgroups.items():
            if len(subgroup) == 0:
                continue
            
            print(f"\nEvolving subgroup (z={z}, y={y})...")
            scores = np.array(self.subgroup_scores[(z, y)], dtype=float)
            
            for iteration in range(self.iters):
                # Roulette wheel selection: select k samples with probability based on scores
                prob = scores / scores.sum()
                try:
                    selected_indices = np.random.choice(
                        len(subgroup), 
                        size=min(self.k, len(subgroup)), 
                        replace=False, 
                        p=prob
                    )
                except:
                    selected_indices = np.arange(min(self.k, len(subgroup)))
                
                demo_indices = [selected_indices]  # For evaluation in step 2, indices relative to subgroup
                
                # Evaluate these demonstrations
                try:
                    pred_score, fair_score, evo_score = evaluate_demos(
                        selected_indices, 
                        subgroup, 
                        self.val_data, 
                        self.formatter, 
                        self.label_fn,
                        alpha=self.alpha
                    )
                    
                    # Update scores of selected samples
                    for idx in selected_indices:
                        scores[idx] = (scores[idx] + evo_score) / 2  # Average with previous
                    
                    if iteration % 2 == 0:
                        print(f"  Iteration {iteration}: pred={pred_score:.3f}, fair={fair_score:.3f}, evo_score={evo_score:.3f}")
                
                except Exception as e:
                    import traceback
                    print(f"  Error during evaluation: {e}")
                    if iteration == 0:  # Only print traceback for first error
                        traceback.print_exc()
                    continue
            
            # Normalize scores
            scores = scores / (scores.max() + EPS)
            self.subgroup_scores[(z, y)] = scores.tolist()
            print(f"  → Evolution complete. Final scores: min={min(scores):.3f}, max={max(scores):.3f}")
    
    def step3_select_demonstrations(self, strategy="s2"):
        """
        Step 3: Select Demonstrations using Strategy
        
        Strategies:
        - s1: Balanced samples with balanced labels (rz=0.5, ry=0.5)
        - s2: Minority samples with balanced labels (rz=1, ry=0.5)
        - s3: Minority samples with unbalanced labels (rz=1, ry≠0.5)
        """
        print(f"\n=== STEP 3: SELECT DEMONSTRATIONS ({strategy.upper()}) ===")
        
        selected = []
        
        if strategy.lower() == "s1":
            # Balanced: select from all subgroups
            for (z, y), subgroup in self.subgroups.items():
                if len(subgroup) == 0:
                    continue
                scores = np.array(self.subgroup_scores[(z, y)])
                top_indices = np.argsort(scores)[-self.k//4:]
                selected.extend([subgroup[i] for i in top_indices])
        
        elif strategy.lower() == "s2":
            # Minority with balanced labels
            for y in [0, 1]:
                subgroup = self.subgroups.get((0, y), [])
                if len(subgroup) == 0:
                    continue
                scores = np.array(self.subgroup_scores[(0, y)])
                top_indices = np.argsort(scores)[-self.k//2:]
                selected.extend([subgroup[i] for i in top_indices])
        
        elif strategy.lower() == "s3":
            # Minority with positive labels
            subgroup = self.subgroups.get((0, 1), [])
            if len(subgroup) > 0:
                scores = np.array(self.subgroup_scores[(0, 1)])
                top_indices = np.argsort(scores)[-self.k:]
                selected.extend([subgroup[i] for i in top_indices])
        
        self.selected_demos[strategy] = selected
        print(f"Selected {len(selected)} demonstrations")
        return selected

# --- 4. MAIN EXECUTION ---

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FCG Algorithm Tests")
    parser.add_argument("--dataset", type=str, default="adult", choices=["adult", "credit", "civil"], help="Dataset to use")
    args = parser.parse_args()
    
    dataset_to_use = args.dataset
    
    print("="*60)
    print(f"FCG ALGORITHM - Fairness via Clustering-Genetic [{dataset_to_use.upper()}]")
    print("="*60)
    
    print("\nLoading data...")
    train_data, test_data, formatter, label_fn = prepare_data(dataset_to_use)
    
    # Split train data into train/val (30% train, 70% val from labeled data)
    split_idx = int(len(train_data) * 0.3)
    random.shuffle(train_data)
    val_data = train_data[split_idx:]
    train_data = train_data[:split_idx]

    print(f"Train set size: {len(train_data)}")
    print(f"Val set size: {len(val_data)}")
    print(f"Test set size: {len(test_data)}")

    # Initialize FCG Algorithm
    print("\nInitializing FCG Algorithm...")
    fcg = FCGAlgorithm(
        train_data=train_data,
        val_data=val_data,
        formatter=formatter,
        label_fn=label_fn,
        n_clusters=CLUSTERS_PER_GROUP,
        m_neighbors=K // 4,  # Neighbors per cluster
        iters=GENETIC_ITER,
        k=K,
        alpha=0.5  # 50% accuracy, 50% fairness
    )

    # Run FCG Algorithm
    print("\n" + "="*60)
    print("RUNNING FCG ALGORITHM")
    print("="*60)
    
    fcg.step1_clustering(dataset_name=dataset_to_use)
    fcg.step2_genetic_evolution()
    
    # Evaluate with different strategies
    strategies = ["s1", "s2", "s3"]
    results = {}
    
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    
    for strategy in strategies:
        print(f"\n--- Strategy {strategy.upper()} ---")
        selected_demos = fcg.step3_select_demonstrations(strategy)
        
        if not selected_demos:
            print(f"No demonstrations selected for {strategy}")
            continue
        
        # Build prompt
        prompt_parts = []
        for demo in selected_demos:
            label = label_fn(demo)
            demo_text = formatter(demo) + " " + label
            prompt_parts.append(demo_text)
        
        demo_prompt = "\n".join(prompt_parts)
        
        # Evaluate on test set
        predictions = []
        true_labels = []
        sensitive_attrs = []
        
        for i in range(min(300, len(test_data))):
            row = test_data[i]
            true_label = label_fn(row)
            pred = predict_with_llm(demo_prompt, formatter, row)
            
            predictions.append(pred)
            true_labels.append(true_label)
            
            if dataset_to_use == "adult":
                z = 1 if row.get("sex") == "Male" else 0
            elif dataset_to_use == "credit":
                sex_val = 1 if row.get("sex:1", 0) == 1 else (2 if row.get("sex:2", 0) == 1 else 0)
                z = 1 if sex_val == 1 else 0
            elif dataset_to_use == "civil":
                z = 1 if float(row.get("identity_attack", 0.0)) > 0.3 else 0
            else:
                z = 1
                
            sensitive_attrs.append(z)
            
            # Print debug info for the first 5 test samples to ensure predictions look solid
            if i < 5:
                print(f"[DEBUG TEST {i}] Ground Truth={true_label}, LLM Pred={pred}, Z={z}")
        
        # Calculate accuracy
        accuracy = sum(1 for p, t in zip(predictions, true_labels) if p == t) / len(predictions)
        
        # Calculate fairness metrics
        fair_metrics = calculate_fairness_metrics(predictions, true_labels, sensitive_attrs, label_fn)
        
        results[strategy] = {
            'accuracy': accuracy,
            'fairness_metrics': fair_metrics
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Rdp (demographic parity ratio): {fair_metrics['dp_ratio']:.4f} ↑")
        print(f"Reo (equalized odds ratio): {fair_metrics['eo_ratio']:.4f} ↑")
        print(f"Δdp (demographic parity diff): {fair_metrics['dp_diff']:.4f} ↓")
        print(f"Δeo (equalized odds diff): {fair_metrics['eo_diff']:.4f} ↓")
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)