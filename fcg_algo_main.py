import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
import random
import math
from dataset import get_llm_probabilities, label_map, load_adult as load_adult_dataset


K = 8  # Number of demonstrations to select
CLUSTERS_PER_GROUP = 10
GENETIC_ITER = 10
POP_SIZE = 10
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# --- 1. DATA PREPARATION ---

def prepare_data():
    """Load and prepare adult income dataset"""
    train_dataset, test_dataset, formatter, label_fn = load_adult_dataset()
    
    # Convert datasets to dataframes for easier handling
    train_data = []
    for i in range(len(train_dataset)):
        row = train_dataset[i]
        train_data.append(row)
    
    test_data = []
    for i in range(len(test_dataset)):
        row = test_dataset[i]
        test_data.append(row)
    
    return train_data, test_data, formatter, label_fn

# --- 2. REAL LLM (GPT-2 via dataset.py) ---

EPS = 1e-12

class RealLLM:
    def __init__(self, formatter, label_fn):
        # Store formatter and label function
        self.formatter = formatter
        self.label_fn = label_fn
    
    def format_example(self, row, include_label=False):
        base = self.formatter(row)
        if include_label:
            label = self.label_fn(row)
            return base + " " + label
        return base
    
    def predict(self, prompt, row):
        # Use the real LLM to get probabilities
        full_input = prompt + "\nInput: " + self.formatter(row)
        probs = get_llm_probabilities(full_input)
        # probs are in the order of label_map.values(), so we need to match that order
        label_order = [word for word, _ in sorted(label_map.items(), key=lambda x: x[1])]
        best_idx = probs.index(max(probs))
        return label_order[best_idx]

def prompt_formatter(demo_set, llm):
    # Format demonstrations as text prompt
    demos = []
    for row in demo_set:
        demo_text = llm.format_example(row, include_label=True)
        demos.append(demo_text)
    return "\n".join(demos)

def evaluate_prompt(demo_set, val_data, llm, prompt_formatter, label_fn, formatter):
    prompt = prompt_formatter(demo_set, llm)
    correct = 0
    total = len(val_data)
    for i in range(min(total, 100)):  # Evaluate on first 100 samples
        row = val_data[i]
        true_label = label_fn(row)
        full_input = prompt + "\nInput: " + formatter(row)
        probs = get_llm_probabilities(full_input)
        label_order = [word for word, _ in sorted(label_map.items(), key=lambda x: x[1])]
        best_idx = probs.index(max(probs))
        pred = label_order[best_idx]
        if pred == true_label:
            correct += 1
    
    accuracy = correct / min(total, 100)
    fairness = 0.5  # Placeholder for fairness metric
    return accuracy, fairness

def roulette_wheel_selection(pool, scores, k):
    probs = np.array(scores) / np.sum(scores)
    indices = np.random.choice(len(pool), size=k, replace=False, p=probs)
    return [pool[i] for i in indices]

def fcg_algorithm(pool, val_data, llm, prompt_formatter, label_fn, formatter, k=K, iters=GENETIC_ITER, pop_size=POP_SIZE):
    scores = np.ones(len(pool))
    best_set, best_score = None, -np.inf
    for gen in range(iters):
        for _ in range(pop_size):
            demo_set = roulette_wheel_selection(pool, scores, k)
            accuracy, fairness = evaluate_prompt(demo_set, val_data, llm, prompt_formatter, label_fn, formatter)
            evol_score = 0.5 * accuracy + 0.5 * fairness
            # Update scores for selected indices
            demo_indices = [pool.index(row) for row in demo_set if row in pool]
            for idx in demo_indices:
                scores[idx] += evol_score
            if evol_score > best_score:
                best_score = evol_score
                best_set = demo_set
        scores = scores / np.max(scores)
    return best_set

# --- 3. MAIN EXECUTION ---

if __name__ == "__main__":
    print("Loading data...")
    train_data, test_data, formatter, label_fn = prepare_data()
    
    # Split train data into train/val
    val_data = train_data[3000:]
    train_data = train_data[:3000]

    print(f"Train set size: {len(train_data)}")
    print(f"Val set size: {len(val_data)}")
    print(f"Test set size: {len(test_data)}")

    # Create candidate pool (just use sample of training data for diversity)
    random.shuffle(train_data)
    pool = train_data[:CLUSTERS_PER_GROUP * 10]  # Use first candidates for pool
    print(f"Candidate pool size: {len(pool)}")

    print("Initializing real LLM...")
    llm = RealLLM(formatter, label_fn)

    print("Running FCG algorithm...")
    best_demos = fcg_algorithm(pool, val_data, llm, prompt_formatter, label_fn, formatter, k=K)
    print(f"\nSelected {len(best_demos)} demonstrations")

    print("\nEvaluating on test set...")
    accuracy, fairness = evaluate_prompt(best_demos, test_data, llm, prompt_formatter, label_fn, formatter)
    print(f"Test Accuracy: {accuracy:.3f}")
    print(f"Test Fairness: {fairness:.3f}")