# recipe_query.py

import json
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

MODEL_DIR = "saved_model"
RECIPE_INPUTS = "bert_recipe_inputs.json"
MAX_LEN   = 64
TOP_K     = 5

def load_model_and_tokenizer(model_dir):
    model = BertModel.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model.eval()
    return model, tokenizer

def load_recipes(recipe_file):
    with open(recipe_file, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_recipe_vectors(model, tokenizer, recipes, max_len=MAX_LEN):
    vectors = []
    names = []
    with torch.no_grad():
        for entry in recipes:
            enc = tokenizer(
                entry["input_text"],
                truncation=True,
                padding="max_length",
                max_length=max_len,
                return_tensors="pt"
            )
            outputs = model(**{k: v for k, v in enc.items()})
            vec = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            vectors.append(vec.squeeze())
            names.append(entry["name"])
    return np.vstack(vectors), names

def query_recipes(model, tokenizer, recipe_vectors, recipe_names, tags, top_k=TOP_K):
    query_text = "[CLS] " + " ".join(tags).lower() + " [SEP] [SEP]"
    enc = tokenizer(
        query_text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**{k: v for k, v in enc.items()})
        qvec = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    sims = cosine_similarity(qvec, recipe_vectors)[0]
    top_idx = sims.argsort()[::-1][:top_k]
    return [(recipe_names[i], float(sims[i])) for i in top_idx]

def main():
    model, tokenizer = load_model_and_tokenizer(MODEL_DIR)
    recipes = load_recipes(RECIPE_INPUTS)
    recipe_vectors, recipe_names = compute_recipe_vectors(model, tokenizer, recipes)

    test_tags = ["healthy", "banana", "quick"]
    print(f"Query Tags: {test_tags}\nTop {TOP_K} Recommendations:")
    for name, score in query_recipes(model, tokenizer, recipe_vectors, recipe_names, test_tags):
        print(f"- {name} (score: {score:.4f})")

if __name__ == "__main__":
    main()
