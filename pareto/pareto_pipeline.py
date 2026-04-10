import numpy as np
from pareto.pareto_engine import ParetoEngine
from dataset import get_llm_probabilities, label_map
from copy import deepcopy


class ParetoPipeline:
    def __init__(self, train_data, val_data, formatter, label_fn, k=4):
        self.train_data = train_data
        self.val_data = val_data
        self.formatter = formatter
        self.label_fn = label_fn
        self.k = k

        self.subgroups = {}
        self.pareto_fronts = {}

    # ---------------------------
    # reuse same subgroup logic
    # ---------------------------
    def stratify(self):
        subgroups = {(0,0):[], (0,1):[], (1,0):[], (1,1):[]}

        for row in self.train_data:
            z = 1 if row.get("sex") == "Male" else 0
            y = 1 if self.label_fn(row) == "Positive" else 0
            subgroups[(z,y)].append(row)

        self.subgroups = subgroups

    # ---------------------------
    # LLM prediction
    # ---------------------------
    def predict(self, prompt, row):
        full_input = prompt + "\nInput: " + self.formatter(row) + "\nIncome:"
        probs = get_llm_probabilities(full_input)
        label_order = list(label_map.keys())
        return label_order[probs.index(max(probs))]

    # ---------------------------
    # evaluation (NO scalarization)
    # ---------------------------
    def evaluate(self, indices, subgroup):
        prompt_parts = []

        for idx in indices:
            row = subgroup[idx]
            label = self.label_fn(row)
            prompt_parts.append(self.formatter(row) + " " + label)

        prompt = "\n".join(prompt_parts)

        preds, labels, sens = [], [], []

        for row in self.val_data[:20]:
            pred = self.predict(prompt, row)
            true = self.label_fn(row)

            preds.append(pred)
            labels.append(true)
            sens.append(1 if row.get("sex") == "Male" else 0)

        # accuracy
        acc = sum(p == t for p, t in zip(preds, labels)) / len(preds)

        # fairness (EO ratio)
        preds_bin = np.array([1 if p == "Positive" else 0 for p in preds])
        labels_bin = np.array([1 if l == "Positive" else 0 for l in labels])
        sens = np.array(sens)

        def tpr(mask):
            pos = (labels_bin == 1) & mask
            return (preds_bin[pos] == 1).mean() if pos.sum() > 0 else 0

        def fpr(mask):
            neg = (labels_bin == 0) & mask
            return (preds_bin[neg] == 1).mean() if neg.sum() > 0 else 0

        minor = sens == 0
        major = sens == 1

        tpr_ratio = tpr(minor) / (tpr(major) + 1e-12)
        fpr_ratio = fpr(minor) / (fpr(major) + 1e-12)

        fairness = min(tpr_ratio, fpr_ratio)

        return acc, fairness

    # ---------------------------
    # run pareto optimisation
    # ---------------------------
    def run(self):
        self.stratify()

        for key, subgroup in self.subgroups.items():
            print(f"\nRunning Pareto for subgroup {key}")

            engine = ParetoEngine(
                evaluate_fn=self.evaluate,
                k=self.k,
                pop_size=20,
                iters=3
            )

            front = engine.evolve(subgroup)
            self.pareto_fronts[key] = front

    # ---------------------------
    # final demo selection
    # ---------------------------
    def select(self):
        selected = []

        for key, front in self.pareto_fronts.items():
            if not front:
                continue

            # choose best tradeoff (you can change this)
            best = max(front, key=lambda x: x[0] + x[1])

            subgroup = self.subgroups[key]
            indices = best[2]

            selected.extend([subgroup[i] for i in indices])

        return selected[:self.k]