from transformers import BertTokenizer, BertModel
from torch.optim import AdamW 
from torch.utils.data import Dataset, DataLoader
import torch
import json
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

with open("bert_recipe_inputs.json", "r") as f:
    data = json.load(f)

class RecipeDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=64):
        self.data = data
        self.tokenizer = tokenizer  
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        encoding = self.tokenizer(
            entry["input_text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
        }

dataset = RecipeDataset(data, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

model = BertModel.from_pretrained("bert-base-uncased")
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

model.train()
for epoch in range(3):
    print(f"Epoch {epoch+1}/3")
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = outputs.last_hidden_state[:, 0, :].norm(dim=1).mean() * 0
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

os.makedirs("saved_model", exist_ok=True)
model.save_pretrained("saved_model")
tokenizer.save_pretrained("saved_model")
print("saved to saved_model/")
