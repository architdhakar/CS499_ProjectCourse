import random
from dataset import load_sst2, get_llm_probabilities, label_words
import matplotlib.pyplot as plt
import math

EPS = 1e-12
CONTENT_FREE = ["", "N/A", " "]

def entropy(probs):
    return -sum(p * math.log(p + EPS) for p in probs)

# Paper 1: Fairness entropy
def fairness_score(example, get_probs):
    entropies = []
    for dummy in CONTENT_FREE:
        prompt = f"{example}\nInput: {dummy}\nIncome:"
        probs = get_probs(prompt)
        entropies.append(entropy(probs))
    return sum(entropies) / len(entropies)

# Paper 2: Real input uncertainty
def uncertainty_score(example, get_probs):
    probs = get_probs(example)
    return entropy(probs)

def T_fair(examples, k, get_probs):
    scored = [(fairness_score(e, get_probs), e) for e in examples]
    scored.sort(reverse=True)
    return [e for _, e in scored[:k]]

def G_fair(examples, get_probs):
    remaining = examples.copy()
    selected = []
    current_prompt = ""
    current_score = -1

    while remaining:
        best = None
        best_score = current_score

        for ex in remaining:
            temp = ex if not current_prompt else ex + "\n" + current_prompt
            score = fairness_score(temp, get_probs)

            if score > best_score:
                best_score = score
                best = ex

        if best is None:
            break

        selected.append(best)
        remaining.remove(best)
        current_prompt = best if not current_prompt else best + "\n" + current_prompt
        current_score = best_score

    return selected

def uncertainty_prompting(examples, k, get_probs):
    scored = [(uncertainty_score(e, get_probs), e) for e in examples]
    scored.sort(reverse=True)
    return [e for _, e in scored[:k]]

def hybrid_prompting(examples, k, get_probs, alpha=0.5):
    scored = []
    for e in examples:
        score = alpha * uncertainty_score(e, get_probs) + \
                (1-alpha) * fairness_score(e, get_probs)
        scored.append((score, e))

    scored.sort(reverse=True)
    return [e for _, e in scored[:k]]

def plot_results(results):
    names = [r["name"] for r in results]
    fairness = [r["fairness"] for r in results]
    accuracy = [r["accuracy"] for r in results]
    gap = [r["gap"] for r in results]

    # Fairness vs Accuracy
    plt.figure(figsize=(6,5))
    plt.scatter(fairness, accuracy)
    for i, name in enumerate(names):
        plt.annotate(name, (fairness[i], accuracy[i]))
    plt.xlabel("Fairness (Entropy)")
    plt.ylabel("Accuracy")
    plt.title("Fairness vs Accuracy")
    plt.show()

    # Accuracy vs Demographic Gap
    plt.figure(figsize=(6,5))
    plt.scatter(accuracy, gap)
    for i, name in enumerate(names):
        plt.annotate(name, (accuracy[i], gap[i]))
    plt.xlabel("Accuracy")
    plt.ylabel("Gender Accuracy Gap")
    plt.title("Accuracy vs Fairness Gap")
    plt.show()

    # Bar plot comparison
    plt.figure(figsize=(8,4))
    plt.bar(names, accuracy)
    plt.title("Accuracy Comparison")
    plt.show()

def build_prompt(demos, row, formatter):
    return "\n".join(demos) + "\n" + formatter(row)


def predict(prompt, get_probs, labels):
    probs = get_probs(prompt)
    return labels[probs.index(max(probs))]


def evaluate(
    demos,
    test_data,
    formatter,
    get_probs,
    labels,
    label_fn,
    group_key=None,          # e.g. "sex" for Adult, None for SST2
    num_samples=300
):
    correct = 0
    group_stats = {}

    # If demographic group exists, initialize counters dynamically
    if group_key is not None and group_key in test_data.column_names:
        unique_groups = set(test_data[i][group_key] for i in range(min(num_samples, len(test_data))))
        for g in unique_groups:
            group_stats[g] = [0, 0]   # [correct, total]

    for i in range(min(num_samples, len(test_data))):
        row = test_data[i]
        true_label = label_fn(row)

        prompt = build_prompt(demos, row, formatter)
        pred = predict(prompt, get_probs, labels)

        if pred == true_label:
            correct += 1

        # If group fairness requested
        if group_key is not None and group_key in row:
            g = row[group_key]
            group_stats[g][1] += 1
            if pred == true_label:
                group_stats[g][0] += 1

    acc = correct / min(num_samples, len(test_data))

    gap = 0
    if group_stats:
        accs = []
        for g in group_stats:
            group_acc = group_stats[g][0] / max(group_stats[g][1], 1)
            accs.append(group_acc)
        gap = max(accs) - min(accs)

    return acc, gap

# Load Adult
train_data, test_data, formatter, label_fn = load_sst2()

# Build candidates
NUM_CANDIDATES = 150
candidates = [formatter(train_data[i]) for i in range(NUM_CANDIDATES)]

USE_DATASET = "sst2"
if USE_DATASET == "adult":
    group_key = "sex"
else:
    group_key = None

K = 3

strategies = [
    ("Random", lambda: random.sample(candidates, K)),
    ("T-Fair", lambda: T_fair(candidates, K, get_llm_probabilities)),
    ("G-Fair", lambda: G_fair(candidates, get_llm_probabilities)),
    ("Uncertainty", lambda: uncertainty_prompting(candidates, K, get_llm_probabilities)),
    ("Hybrid", lambda: hybrid_prompting(candidates, K, get_llm_probabilities))
]

results = []

for name, func in strategies:
    demos = func()

    fairness = fairness_score("\n".join(demos), get_llm_probabilities)

    acc, gap = evaluate(
        demos,
        test_data,
        formatter,
        get_llm_probabilities,
        label_words,
        label_fn=label_fn,
        group_key=group_key
    )

    print(f"{name}: accuracy={acc:.3f}, fairness={fairness:.3f}, gap={gap:.3f}")

    results.append({
        "name": name,
        "accuracy": acc,
        "fairness": fairness,
        "gap": gap
    })

plot_results(results)
