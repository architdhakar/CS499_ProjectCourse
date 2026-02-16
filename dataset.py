import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

MODEL_NAME = "gpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# Label tokens
label_words = ["Negative", "Positive"]
label_map = {}

for word in label_words:
    token_id = tokenizer.encode(" " + word, add_special_tokens=False)
    assert len(token_id) == 1
    label_map[word] = token_id[0]

def get_llm_probabilities(text_input):
    inputs = tokenizer(text_input, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1, :]
    scores = torch.tensor([logits[t] for t in label_map.values()])
    probs = F.softmax(scores, dim=0)
    return probs.tolist()

# -----------------------------------------------------
# 1️⃣ Adult Income (structured fairness dataset)
# -----------------------------------------------------
def load_adult():
    dataset = load_dataset("scikit-learn/adult-census-income")
    full = dataset["train"]
    split = full.train_test_split(test_size=0.2, seed=42)

    def formatter(row):
        return (
            f"Age: {row['age']}, "
            f"Workclass: {row['workclass']}, "
            f"Education: {row['education']}, "
            f"Marital status: {row['marital.status']}, "
            f"Occupation: {row['occupation']}, "
            f"Relationship: {row['relationship']}, "
            f"Race: {row['race']}, "
            f"Sex: {row['sex']}, "
            f"Hours per week: {row['hours.per.week']}\nIncome:"
        )

    def label_fn(row):
        return "Positive" if row["income"] == 1 else "Negative"

    return split["train"], split["test"], formatter, label_fn


# -----------------------------------------------------
# 2️⃣ SST-2 (Sentiment, used in both papers)
# -----------------------------------------------------
def load_sst2():
    dataset = load_dataset("glue", "sst2")

    def formatter(row):
        return f"Review: {row['sentence']}\nLabel:"

    def label_fn(row):
        return "Positive" if row["label"] == 1 else "Negative"

    return dataset["train"], dataset["validation"], formatter, label_fn


# -----------------------------------------------------
# 3️⃣ AG News (4-class → mapped to binary)
# -----------------------------------------------------
def load_agnews_binary():
    dataset = load_dataset("ag_news")

    def formatter(row):
        return f"News: {row['text']}\nLabel:"

    def label_fn(row):
        # Map 4 classes into binary:
        # 0,1 -> Negative
        # 2,3 -> Positive
        return "Positive" if row["label"] >= 2 else "Negative"

    return dataset["train"], dataset["test"], formatter, label_fn


# -----------------------------------------------------
# 4️⃣ TREC Question Classification (binary mapped)
# -----------------------------------------------------
def load_trec_binary():
    dataset = load_dataset("trec")

    def formatter(row):
        return f"Question: {row['text']}\nLabel:"

    def label_fn(row):
        # Binary mapping:
        # numeric type = Positive
        # others = Negative
        return "Positive" if row["coarse_label"] == 2 else "Negative"

    return dataset["train"], dataset["test"], formatter, label_fn


# -----------------------------------------------------
# 5️⃣ RTE (Recognizing Textual Entailment)
# -----------------------------------------------------
def load_rte():
    dataset = load_dataset("glue", "rte")

    def formatter(row):
        return (
            f"Premise: {row['sentence1']}\n"
            f"Hypothesis: {row['sentence2']}\nLabel:"
        )

    def label_fn(row):
        return "Positive" if row["label"] == 1 else "Negative"

    return dataset["train"], dataset["validation"], formatter, label_fn