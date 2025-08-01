import pandas as pd
import json

def main():
    df = pd.read_csv("RAW_recipes.csv")

    processed = []
    for _, row in df.iterrows():
        tags = str(row.get("tags", "")).lower().replace(",", " ")
        ingredients = str(row.get("ingredients", "")).lower().replace(",", " ")
        text = f"[CLS] {tags} [SEP] {ingredients} [SEP]"
        processed.append({
            "name": row.get("name", ""),
            "input_text": text
        })

    with open("bert_recipe_inputs.json", "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    print("Generated bert_recipe_inputs.json with", len(processed), "entries.")

if __name__ == "__main__":
    main()
