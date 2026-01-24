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

examples = [
    "Review: This movie was terrible.\nLabel:",
    "Review: I absolutely loved this film.\nLabel:",
    "Review: It was okay, not great.\nLabel:"
]

print("Running T-Fair Selection...")
selected = TFairPrompting(examples)
print(selected)