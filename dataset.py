import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

MODEL_NAME = "gpt2"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    # Use Apple Silicon GPU (Metal Performance Shaders)
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")

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
# 1.5 Credit Card Default Dataset
# -----------------------------------------------------
def load_credit():
    # Use UCI credit card default dataset. Since huggingface datasets has default of credit card clients.
    # Provide a simple local mock/loader or load from huggingface if available. 
    # For now, let's load it from a known hub path or simulate if it cannot be loaded.
    try:
        from datasets import load_dataset
        dataset = load_dataset("imodels/credit-card")
        full = dataset["train"]
        split = full.train_test_split(test_size=0.2, seed=42)
        
        def formatter(row):
            return (
                f"Limit Balance: {row['limit_bal']}, "
                f"Sex: {row['sex'] if 'sex' in row else row.get('SEX', 'Unknown')}, "
                f"Education: {row['education'] if 'education' in row else row.get('EDUCATION', 'Unknown')}, "
                f"Marriage: {row['marriage'] if 'marriage' in row else row.get('MARRIAGE', 'Unknown')}, "
                f"Age: {row['age'] if 'age' in row else row.get('AGE', 'Unknown')}, "
                f"History 1: {row['pay_0'] if 'pay_0' in row else row.get('PAY_0', 'Unknown')}, "
                f"Bill Amount 1: {row['bill_amt1'] if 'bill_amt1' in row else row.get('BILL_AMT1', 'Unknown')}, "
                f"Pay Amount 1: {row['pay_amt1'] if 'pay_amt1' in row else row.get('PAY_AMT1', 'Unknown')}\nDefault:"
            )

        def label_fn(row):
            # Target is `default.payment.next.month`
            return "Positive" if float(row.get("default.payment.next.month", 0)) == 1.0 else "Negative"

        return split["train"], split["test"], formatter, label_fn

    except Exception as e:
        print(f"Failed to load credit card dataset: {e}")
        # Return empty lists as fallback, preventing hard crash
        return [], [], lambda x: "", lambda x: "Negative"

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
        # Income is a string: ">50K" or "<=50K"
        return "Positive" if row["income"] == ">50K" else "Negative"

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