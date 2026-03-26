import math
import random
import matplotlib.pyplot as plt
from datasets import load_dataset
from dataset import get_llm_probabilities, label_map

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
g_fair_demos = GFairPrompting(candidates)

results = []

for name, demos in [
    ("Random", random_demos),
    ("T-Fair", t_fair_demos),
    ("G-Fair", g_fair_demos),
]:
    fairness = fairnessScore("\n".join(demos))
    accuracy = evaluate_prompt(demos, test_data)
    results.append((name, fairness, accuracy))
    print(f"{name}: fairness={fairness:.4f}, accuracy={accuracy:.3f}")

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
