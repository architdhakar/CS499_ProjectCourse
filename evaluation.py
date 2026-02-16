def build_prompt(demos, row, formatter):
    return "\n".join(demos) + "\n" + formatter(row)

def predict(prompt, get_probs, labels):
    probs = get_probs(prompt)
    return labels[probs.index(max(probs))]

def evaluate(demos, test_data, formatter, get_probs, labels, num_samples=300):
    correct = 0
    group_stats = {"Male": [0,0], "Female": [0,0]}

    for i in range(num_samples):
        row = test_data[i]
        true_label = "Positive" if row["income"] == 1 else "Negative"

        prompt = build_prompt(demos, row, formatter)
        pred = predict(prompt, get_probs, labels)

        if pred == true_label:
            correct += 1

        sex = row["sex"]
        group_stats[sex][1] += 1
        if pred == true_label:
            group_stats[sex][0] += 1

    acc = correct / num_samples
    male_acc = group_stats["Male"][0] / max(group_stats["Male"][1],1)
    female_acc = group_stats["Female"][0] / max(group_stats["Female"][1],1)
    gap = abs(male_acc - female_acc)

    return acc, gap
