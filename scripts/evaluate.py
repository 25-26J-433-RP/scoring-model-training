import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import os

# -------------------------
# Paths & Setup
# -------------------------
MODEL_PATH = "models/bert_regressor.pt"
DATA_PATH = "data/aes_dataset/training_set_rel3.tsv"  # update if needed
OUTPUT_PATH = "outputs/transformer_predictions.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load Tokenizer & Model
# -------------------------
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

class BertRegressor(torch.nn.Module):
    def __init__(self, bert_model):
        super(BertRegressor, self).__init__()
        self.bert = bert_model
        self.out = torch.nn.Linear(768, 1)  # match training

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token
        score = self.out(pooled_output)
        return score

bert = AutoModel.from_pretrained("bert-base-uncased")
model = BertRegressor(bert).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -------------------------
# Load Dataset
# -------------------------
df = pd.read_csv(DATA_PATH, sep="\t", encoding="latin-1")
essays = df["essay"].astype(str).tolist()
scores = df["domain1_score"].tolist()

# Train/val split (use same style as training)
_, X_val, _, y_val = train_test_split(essays, scores, test_size=0.2, random_state=42)

# -------------------------
# Prediction Loop
# -------------------------
predictions = []
with torch.no_grad():
    for essay in X_val:
        tokens = tokenizer(
            essay,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length"
        )
        input_ids = tokens["input_ids"].to(DEVICE)
        attention_mask = tokens["attention_mask"].to(DEVICE)

        score = model(input_ids, attention_mask)
        predictions.append(score.item())

# -------------------------
# Evaluate
# -------------------------
rmse = np.sqrt(mean_squared_error(y_val, predictions))
print(f"✅ Transformer Validation RMSE: {rmse:.4f}")

# -------------------------
# Save CSV
# -------------------------
os.makedirs("outputs", exist_ok=True)
results_df = pd.DataFrame({
    "essay": X_val,
    "teacher_score": y_val,
    "predicted_score": np.round(predictions, 2)
})
results_df.to_csv(OUTPUT_PATH, index=False)
print(f"📂 Predictions saved to {OUTPUT_PATH}")
