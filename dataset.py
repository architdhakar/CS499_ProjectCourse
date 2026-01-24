from datasets import load_dataset
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


model_name = "gpt2" # Or "bigscience/bloom-560m" or "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval() 

if torch.cuda.is_available():
    model = model.cuda()

def get_llm_probabilities(text_input, label_map):
    """
    text_input: The full prompt ending in "Label:"
    label_map: Dictionary mapping label names to their token IDs.
               e.g., {"Positive": 3893, "Negative": 4393}
    """
    inputs = tokenizer(text_input, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")

    with torch.no_grad():
        outputs = model(**inputs)
        
    # Get logits of the last token
    next_token_logits = outputs.logits[0, -1, :]
    
    # We only care about the probabilities of our specific label tokens
    # e.g., P("Positive") and P("Negative")
    label_probs = []
    for label_word, token_id in label_map.items():
        # Get raw score for this specific token
        score = next_token_logits[token_id].item()
        label_probs.append(score)
    
    # Softmax ONLY over our candidate labels to normalize them to sum to 1
    # This effectively asks: "Given it must be one of these two, which is it?"
    probs_tensor = F.softmax(torch.tensor(label_probs), dim=0)
    
    return probs_tensor.tolist() # Returns [0.7, 0.3]


# Dataset loading

dataset = load_dataset("glue", "sst2")
train_data = dataset['train']

label_words = ["Negative", "Positive"]
label_map = {}

for word in label_words:
    # Note: We add a space " " before the word because 
    # models usually predict " Positive" (with leading space)
    token_id = tokenizer.encode(" " + word, add_special_tokens=False)[0]
    label_map[word] = token_id

print(f"Token IDs: {label_map}")


candidates = []
for i in range(100):
    text = train_data[i]['sentence']
    label = train_data[i]['label'] # 0 or 1
    
    # Convert numeric label to word
    label_word = label_words[label] 
    
    # Format as a few-shot example string
    formatted_example = f"Review: {text}\nLabel: {label_word}"
    candidates.append(formatted_example)