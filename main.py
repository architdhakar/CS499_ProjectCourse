import math
from dataset import get_llm_probabilities,label_map

def fairnessScore(prompt_example):
    dummy_input = "N/A" 

    full_input = f"{prompt_example}\nInput: {dummy_input}\nLabel:"
    probs = get_llm_probabilities(full_input,label_map) 
    entropy = 0
    for p in probs:
        if p > 0: # Avoid log(0) error
            entropy += -1 * (p * math.log(p))
    return entropy

def TFairPrompting(training_examples, k=3):
    scored_candidates = []

    for ex in training_examples:
        score = fairnessScore(ex)
        scored_candidates.append((score, ex))
    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    final_prompt_examples = []
    for i in range(k):
        final_prompt_examples.append(scored_candidates[i][1])

    return final_prompt_examples

def GFairPrompting(training_examples):
    """
    Greedy fairness-guided prompting (G-fair)
    Automatically decides how many examples to include.
    """

    remaining = training_examples.copy()
    selected = []

    # Start with empty prompt
    current_prompt = ""
    current_fairness = -float("inf")

    while len(remaining) > 0:
        best_candidate = None
        best_fairness = current_fairness

        for ex in remaining:
            # Build temporary prompt by prepending candidate
            if current_prompt == "":
                temp_prompt = ex
            else:
                temp_prompt = ex + "\n" + current_prompt

            score = fairnessScore(temp_prompt)

            if score > best_fairness:
                best_fairness = score
                best_candidate = ex

        # Stop if no improvement
        if best_candidate is None:
            break

        # Update state
        selected.append(best_candidate)
        remaining.remove(best_candidate)
        current_prompt = best_candidate + "\n" + current_prompt
        current_fairness = best_fairness

    return selected

examples = [
    "Review: This movie was terrible.\nLabel:",
    "Review: I absolutely loved this film.\nLabel:",
    "Review: It was okay, not great.\nLabel:"
]

print("Running T-Fair Selection...")
t_selected = TFairPrompting(examples, k=2)
print("T-Fair:", t_selected)

print("\nRunning G-Fair Selection...")
g_selected = GFairPrompting(examples)
print("G-Fair:", g_selected)