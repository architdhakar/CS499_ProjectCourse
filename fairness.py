import random
import math
import numpy as np
import matplotlib.pyplot as plt

from dataset import (
    get_llm_probabilities,
    label_map,
    load_sst2,
    load_agnews_binary,
    load_trec_binary,
    load_rte
)

# ----------------------------
# Config
# ----------------------------
random.seed(42)
np.random.seed(42)

K = 3
NUM_EVAL = 200
EPS = 1e-12

DATASETS = ["sst2", "agnews", "trec", "rte"]

all_dataset_results = {}

# ----------------------------
# Core Functions
# ----------------------------
def fairnessScore(prompt):
    full_input = f"{prompt}\nInput: N/A\nLabel:"
    probs = get_llm_probabilities(full_input)
    return -sum(p * math.log(p + EPS) for p in probs)

def build_prompt(demos, row, formatter):
    return "\n".join(demos) + "\n" + formatter(row)

def predict_label(prompt):
    probs = get_llm_probabilities(prompt)
    label_order = list(label_map.keys())
    return label_order[probs.index(max(probs))]

def evaluate_prompt(demos, data, formatter, label_fn):
    correct = 0
    for i in range(min(NUM_EVAL, len(data))):
        row = data[i]
        true_label = label_fn(row)

        prompt = build_prompt(demos, row, formatter)
        pred = predict_label(prompt)

        if pred == true_label:
            correct += 1

    return correct / NUM_EVAL

# ----------------------------
# Individual Fairness (TEXT)
# ----------------------------
def compute_individual_fairness(demos, data, formatter, label_fn, num_pairs=100):
    diffs = []

    def similarity(r1, r2):
        # simple proxy: same label
        return label_fn(r1) == label_fn(r2)

    for _ in range(num_pairs):
        i, j = random.sample(range(len(data)), 2)
        r1, r2 = data[i], data[j]

        if not similarity(r1, r2):
            continue

        p1 = predict_label(build_prompt(demos, r1, formatter))
        p2 = predict_label(build_prompt(demos, r2, formatter))

        diffs.append(abs((p1 == "Positive") - (p2 == "Positive")))

    if len(diffs) == 0:
        return 1.0

    return 1 - np.mean(diffs)

# ----------------------------
# Methods
# ----------------------------
def random_prompt(examples, k):
    return random.sample(examples, k)

def TFairPrompting(examples, k):
    scored = [(fairnessScore(ex), ex) for ex in examples]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ex for _, ex in scored[:k]]

def GFairPrompting(examples):
    remaining = examples.copy()
    selected = []
    current_prompt = ""
    current_score = -float("inf")

    while remaining:
        best = None
        best_score = current_score

        for ex in remaining:
            temp_prompt = ex if current_prompt == "" else ex + "\n" + current_prompt
            score = fairnessScore(temp_prompt)

            if score > best_score:
                best_score = score
                best = ex

        if best is None:
            break

        selected.append(best)
        remaining.remove(best)
        current_prompt = (
            best if current_prompt == ""
            else best + "\n" + current_prompt
        )
        current_score = best_score

    return selected[:K]

# ----------------------------
# MAIN LOOP
# ----------------------------
for DATASET in DATASETS:

    print("\n" + "="*60)
    print(f"DATASET: {DATASET.upper()}")
    print("="*60)

    # Load dataset
    if DATASET == "sst2":
        train_data, test_data, formatter, label_fn = load_sst2()
    elif DATASET == "agnews":
        train_data, test_data, formatter, label_fn = load_agnews_binary()
    elif DATASET == "trec":
        train_data, test_data, formatter, label_fn = load_trec_binary()
    elif DATASET == "rte":
        train_data, test_data, formatter, label_fn = load_rte()

    # Convert to list
    train_data = [train_data[i] for i in range(len(train_data))]
    test_data = [test_data[i] for i in range(len(test_data))]

    # Build candidates
    candidates = [
        formatter(train_data[i]) + " " + label_fn(train_data[i])
        for i in range(min(100, len(train_data)))
    ]

    dataset_results = []

    # Methods
    methods = [
        ("Random", random_prompt(candidates, K)),
        ("T-Fair", TFairPrompting(candidates, K)),
        ("G-Fair", GFairPrompting(candidates)),
    ]

    for name, demos in methods:
        entropy = fairnessScore("\n".join(demos))
        acc = evaluate_prompt(demos, test_data, formatter, label_fn)
        indiv = compute_individual_fairness(demos, test_data, formatter, label_fn)

        print(f"\n{name}:")
        print(f"  Accuracy = {acc:.3f}")
        print(f"  Entropy Fairness = {entropy:.4f}")
        print(f"  Individual Fairness = {indiv:.4f}")

        dataset_results.append({
            "method": name,
            "accuracy": acc,
            "fairness": entropy,
            "individual": indiv
        })

    all_dataset_results[DATASET] = dataset_results

    # ----------------------------
    # Table per dataset
    # ----------------------------
    print("\nTABLE:")
    print("Method\t\tAccuracy\tFairness\tIndividual")

    for r in dataset_results:
        print(f"{r['method']}\t\t{r['accuracy']:.3f}\t\t{r['fairness']:.3f}\t\t{r['individual']:.3f}")

# ----------------------------
# CROSS DATASET TABLE
# ----------------------------
print("\n" + "="*60)
print("CROSS-DATASET ACCURACY TABLE")
print("="*60)

methods = ["Random", "T-Fair", "G-Fair"]

print("Dataset\t\t" + "\t".join(methods))

for ds, results in all_dataset_results.items():
    row = [ds.upper()]
    for m in methods:
        val = next(r["accuracy"] for r in results if r["method"] == m)
        row.append(f"{val:.3f}")
    print("\t\t".join(row))

# ----------------------------
# PLOTS
# ----------------------------

# Accuracy Bar
for ds, results in all_dataset_results.items():
    plt.figure()
    names = [r["method"] for r in results]
    accs = [r["accuracy"] for r in results]

    plt.bar(names, accs)
    plt.title(f"{ds.upper()} - Accuracy")
    plt.ylabel("Accuracy")
    plt.show()

# Fairness vs Accuracy Scatter
for ds, results in all_dataset_results.items():
    plt.figure()

    x = [r["fairness"] for r in results]
    y = [r["accuracy"] for r in results]
    labels = [r["method"] for r in results]

    plt.scatter(x, y)

    for i, label in enumerate(labels):
        plt.annotate(label, (x[i], y[i]))

    plt.xlabel("Fairness")
    plt.ylabel("Accuracy")
    plt.title(f"{ds.upper()} Fairness vs Accuracy")
    plt.show()

# Cross dataset comparison
for m in methods:
    vals = []
    for ds in all_dataset_results:
        val = next(r["accuracy"] for r in all_dataset_results[ds] if r["method"] == m)
        vals.append(val)

    plt.plot(all_dataset_results.keys(), vals, marker='o', label=m)

plt.title("Method Comparison Across Datasets")
plt.ylabel("Accuracy")
plt.legend()
plt.show()