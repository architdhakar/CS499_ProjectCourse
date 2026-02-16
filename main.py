import random
from dataset import load_adult, get_llm_probabilities, label_words
from selection import *
from evaluation import evaluate
from plotting import plot_results

# Load Adult
train_data, test_data, formatter, label_fn = load_adult()

# Build candidates
NUM_CANDIDATES = 150
candidates = [formatter(train_data[i]) for i in range(NUM_CANDIDATES)]

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
        label_fn=label_fn
    )

    print(f"{name}: accuracy={acc:.3f}, fairness={fairness:.3f}, gap={gap:.3f}")

    results.append({
        "name": name,
        "accuracy": acc,
        "fairness": fairness,
        "gap": gap
    })

plot_results(results)
