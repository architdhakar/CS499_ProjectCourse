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
