import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from tqdm import tqdm

# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "bert-base-uncased"   # can change to "roberta-base"
MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 3
LR = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_PATH = "data/aes_dataset/training_set_rel3.tsv"
MODEL_SAVE_PATH = "models/bert_regressor.pt"

# -----------------------------
# Dataset Class
# -----------------------------
class EssayDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_len):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        target = self.targets[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "target": torch.tensor(target, dtype=torch.float)
        }

# -----------------------------
# Model Class
# -----------------------------
class BertRegressor(nn.Module):
    def __init__(self, model_name):
        super(BertRegressor, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        return self.out(x)

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv(DATA_PATH, sep="\t", encoding="latin1")

X = df["essay"].values
y = df["domain1_score"].values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Tokenizer + DataLoader
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = EssayDataset(X_train, y_train, tokenizer, MAX_LEN)
val_dataset = EssayDataset(X_val, y_val, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# -----------------------------
# Init Model
# -----------------------------
model = BertRegressor(MODEL_NAME).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(EPOCHS):
    model.train()
    train_losses = []
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        targets = batch["target"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    avg_train_loss = np.mean(train_losses)

    # -----------------------------
    # Validation
    # -----------------------------
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            targets = batch["target"].to(DEVICE)

            outputs = model(input_ids, attention_mask)
            val_preds.extend(outputs.squeeze().cpu().numpy())
            val_targets.extend(targets.cpu().numpy())

    rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val RMSE: {rmse:.4f}")

# -----------------------------
# Save Model
# -----------------------------
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"✅ Transformer model saved to {MODEL_SAVE_PATH}")
