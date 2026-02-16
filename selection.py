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
