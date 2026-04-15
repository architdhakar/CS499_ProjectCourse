import math
import random
import matplotlib.pyplot as plt
from datasets import load_dataset
from dataset import get_llm_probabilities, label_map
from sklearn.cluster import KMeans
import numpy as np

random.seed(42)
np.random.seed(42)

EPS = 1e-12
K = 3
NUM_EVAL = 200

# ----------------------------
# Load Adult Income dataset
# ----------------------------
dataset = load_dataset("scikit-learn/adult-census-income")

full_data = dataset["train"]

# Create train/test split manually
split = full_data.train_test_split(test_size=0.2, seed=42)

train_data = split["train"]
test_data = split["test"]

# print(train_data.column_names)

# ----------------------------
# Convert structured data to text
# ----------------------------
def format_example(row, include_label=False):
    base = (
        f"Age: {row['age']}, "
        f"Workclass: {row['workclass']}, "
        f"Education: {row['education']}, "
        f"Marital status: {row['marital.status']}, "
        f"Occupation: {row['occupation']}, "
        f"Relationship: {row['relationship']}, "
        f"Race: {row['race']}, "
        f"Sex: {row['sex']}, "
        f"Hours per week: {row['hours.per.week']}\n"
        "Income:"
    )
    if include_label:
        label = row["income"]
        return base + " " + label
    return base

def label_to_word(label):
    # label is either '>50K' or '<=50K'
    return "Positive" if label == ">50K" else "Negative"

# ----------------------------
# Build candidate demonstrations
# ----------------------------
candidates = []
for i in range(100):
    row = train_data[i]
    text = format_example(row, include_label=True)
    candidates.append(text)

# ----------------------------
# Fairness (Entropy on meaningless input)
# ----------------------------
def fairnessScore(prompt_example):
    dummy_input = "N/A"
    full_input = f"{prompt_example}\nInput: {dummy_input}\nIncome:"
    probs = get_llm_probabilities(full_input)
    return -sum(p * math.log(p + EPS) for p in probs)

# ----------------------------
# DP + EO Fairness Metrics
# ----------------------------
def compute_fairness_metrics(demos, data):
    preds = []
    labels = []
    sens = []

    for i in range(NUM_EVAL):
        row = data[i]

        # Build prompt and predict
        prompt = build_prompt(demos, row)
        pred = predict_label(prompt)
        true_label = label_to_word(row["income"])

        preds.append(1 if pred == "Positive" else 0)
        labels.append(1 if true_label == "Positive" else 0)
        sens.append(1 if row["sex"] == "Male" else 0)

    import numpy as np
    preds = np.array(preds)
    labels = np.array(labels)
    sens = np.array(sens)

    minority = (sens == 0)
    majority = (sens == 1)

    # ---------------- DP ----------------
    dp_min = preds[minority].mean() if minority.sum() > 0 else 0
    dp_maj = preds[majority].mean() if majority.sum() > 0 else 0

    dp_diff = abs(dp_maj - dp_min)
    dp_ratio = min(dp_min, dp_maj) / (max(dp_min, dp_maj) + EPS)

    # ---------------- EO ----------------
    def tpr(mask):
        pos = (labels == 1) & mask
        return (preds[pos] == 1).mean() if pos.sum() > 0 else 0

    def fpr(mask):
        neg = (labels == 0) & mask
        return (preds[neg] == 1).mean() if neg.sum() > 0 else 0

    tpr_min = tpr(minority)
    tpr_maj = tpr(majority)
    fpr_min = fpr(minority)
    fpr_maj = fpr(majority)

    eo_diff = max(abs(tpr_min - tpr_maj), abs(fpr_min - fpr_maj))

    tpr_ratio = tpr_min / (tpr_maj + EPS)
    fpr_ratio = fpr_min / (fpr_maj + EPS)
    eo_ratio = min(tpr_ratio, fpr_ratio)

    return {
        "dp_diff": dp_diff,
        "dp_ratio": dp_ratio,
        "eo_diff": eo_diff,
        "eo_ratio": eo_ratio
    }

# ----------------------------
# Individual Fairness (Consistency)
# ----------------------------
def compute_individual_fairness(demos, data, num_pairs=100):
    import numpy as np
    import random

    def similarity(row1, row2):
        score = 0
        keys = ["education", "occupation", "hours.per.week"]

        for k in keys:
            if str(row1[k]) == str(row2[k]):
                score += 1

        # age: allow small tolerance
        if abs(row1["age"] - row2["age"]) <= 5:
            score += 1

        return score / (len(keys) + 1)

    diffs = []

    for _ in range(num_pairs):
        i, j = random.sample(range(len(data)), 2)
        row1, row2 = data[i], data[j]

        sim = similarity(row1, row2)

        if sim < 0.75:
            continue

        prompt1 = build_prompt(demos, row1)
        prompt2 = build_prompt(demos, row2)

        pred1 = predict_label(prompt1)
        pred2 = predict_label(prompt2)

        p1 = 1 if pred1 == "Positive" else 0
        p2 = 1 if pred2 == "Positive" else 0

        diffs.append(abs(p1 - p2))

    if len(diffs) == 0:
        return 1.0

    return 1 - np.mean(diffs)

# ----------------------------
# T-fair
# ----------------------------
def TFairPrompting(training_examples, k):
    scored = [(fairnessScore(ex), ex) for ex in training_examples]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ex for _, ex in scored[:k]]

# ----------------------------
# G-fair
# ----------------------------
def GFairPrompting(training_examples):
    remaining = training_examples.copy()
    selected = []
    current_prompt = ""
    current_fairness = -float("inf")

    while remaining:
        best_candidate = None
        best_fairness = current_fairness

        for ex in remaining:
            temp_prompt = ex if current_prompt == "" else ex + "\n" + current_prompt
            score = fairnessScore(temp_prompt)

            if score > best_fairness:
                best_fairness = score
                best_candidate = ex

        if best_candidate is None:
            break

        selected.append(best_candidate)
        remaining.remove(best_candidate)
        current_prompt = (
            best_candidate if current_prompt == ""
            else best_candidate + "\n" + current_prompt
        )
        current_fairness = best_fairness

    return selected

def diversity_prompt(data, k):
    data_list = [data[i] for i in range(min(100, len(data)))]  # FIX

    X = np.array([
        [row['age'], hash(row['education']) % 100, row['hours.per.week']]
        for row in data_list
    ])

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)

    selected = []
    for center in kmeans.cluster_centers_:
        distances = np.linalg.norm(X - center, axis=1)
        idx = np.argmin(distances)
        selected.append(format_example(data_list[idx], include_label=True))

    return selected

def similarity_prompt(data, k):
    data_list = [data[i] for i in range(min(100, len(data)))]  # FIX

    base = data_list[0]

    def dist(row):
        return abs(row['age'] - base['age'])

    sorted_data = sorted(data_list, key=dist)

    return [
        format_example(row, include_label=True)
        for row in sorted_data[:k]
    ]

# ----------------------------
# Evaluation
# ----------------------------
def build_prompt(demos, row):
    base = "\n".join(demos)
    return base + "\n" + format_example(row, include_label=False)

def predict_label(prompt):
    probs = get_llm_probabilities(prompt)
    # probs are in the order of label_map.values(), so we need to match that order
    label_order = [word for word, _ in sorted(label_map.items(), key=lambda x: x[1])]
    best_idx = probs.index(max(probs))
    return label_order[best_idx]

def evaluate_prompt(demos, data):
    correct = 0
    for i in range(NUM_EVAL):
        row = data[i]
        true_label = label_to_word(row["income"])
        prompt = build_prompt(demos, row)
        pred = predict_label(prompt)
        if pred == true_label:
            correct += 1
    return correct / NUM_EVAL

def random_prompt(examples, k):
    return random.sample(examples, k)

# ----------------------------
# Run experiment
# ----------------------------
random_demos = random_prompt(candidates, K)
t_fair_demos = TFairPrompting(candidates, K)
g_fair_demos = GFairPrompting(candidates)[:K]

results = []

rand_accuracy = evaluate_prompt(random_demos, test_data)
rand_entropy = fairnessScore("\n".join(random_demos))
rand_metrics = compute_fairness_metrics(random_demos, test_data)
rand_if = compute_individual_fairness(random_demos, test_data)

results.append(("Random", rand_entropy, rand_accuracy))

print(f"\nRandom:")
print(f"  Accuracy = {rand_accuracy:.3f}")
print(f"  Entropy Fairness = {rand_entropy:.4f}")
print(f"  DP Ratio = {rand_metrics['dp_ratio']:.4f}")
print(f"  EO Ratio = {rand_metrics['eo_ratio']:.4f}")
print(f"  Individual Fairness = {rand_if:.4f}")

# Other methods
methods = [
    ("T-Fair", TFairPrompting(candidates, K)),
    ("G-Fair", GFairPrompting(candidates)[:K]),
    ("Diversity", diversity_prompt(train_data, K)),
    ("Similarity", similarity_prompt(train_data, K)),
]

for name, demos in methods:
    entropy_fairness = fairnessScore("\n".join(demos))
    accuracy = evaluate_prompt(demos, test_data)

    metrics = compute_fairness_metrics(demos, test_data)
    indiv_fair = compute_individual_fairness(demos, test_data)

    results.append((name, entropy_fairness, accuracy))

    print(f"\n{name}:")
    print(f"  Accuracy = {accuracy:.3f}")
    print(f"  Entropy Fairness = {entropy_fairness:.4f}")
    print(f"  DP Ratio = {metrics['dp_ratio']:.4f}")
    print(f"  EO Ratio = {metrics['eo_ratio']:.4f}")
    print(f"  Individual Fairness = {indiv_fair:.4f}")

# ----------------------------
# Plot
# ----------------------------
x = [r[1] for r in results]
y = [r[2] for r in results]
labels = [r[0] for r in results]

plt.figure()
plt.scatter(x, y)
for i, label in enumerate(labels):
    plt.annotate(label, (x[i], y[i]))

plt.xlabel("Fairness (Entropy)")
plt.ylabel("Accuracy")
plt.title("Fairness vs Accuracy on Adult Income")
plt.show()
