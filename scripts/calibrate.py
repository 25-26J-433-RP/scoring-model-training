# scripts/calibrate.py
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
import torch
from transformers import AutoTokenizer, AutoModel
import sys, os

DATA_PATH = "data/aes_dataset/training_set_rel3.tsv"
MODEL_PATH = "models/bert_regressor.pt"
CALIBRATION_PATH = "models/calibration_set{}.pkl"  # <-- per set

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Model ----------------
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

class BertRegressor(torch.nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        self.out = torch.nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        return self.out(pooled_output)

bert = AutoModel.from_pretrained("bert-base-uncased")
model = BertRegressor(bert).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

def get_score(essay):
    tokens = tokenizer(essay, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    with torch.no_grad():
        return model(tokens["input_ids"].to(DEVICE), tokens["attention_mask"].to(DEVICE)).item()

# ---------------- Main ----------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Usage: python scripts/calibrate.py <essay_set>")
        sys.exit(1)

    essay_set = int(sys.argv[1])
    print(f"📂 Calibrating for essay_set = {essay_set}")

    df = pd.read_csv(DATA_PATH, sep="\t", encoding="ISO-8859-1")
    df = df[df["essay_set"] == essay_set]  # filter only one set
    essays = df["essay"].astype(str).tolist()
    teacher_scores = df["domain1_score"].tolist()

    raw_preds = [get_score(e) for e in essays]

    # Train calibration mapping
    calibrator = LinearRegression().fit([[r] for r in raw_preds], teacher_scores)

    score_range = (min(teacher_scores), max(teacher_scores))
    out_path = CALIBRATION_PATH.format(essay_set)
    os.makedirs("models", exist_ok=True)
    joblib.dump({"calibrator": calibrator, "score_range": score_range}, out_path)

    print(f"✅ Calibration trained for set {essay_set}")
    print(f"   Teacher score range = {score_range}")
    print(f"💾 Saved -> {out_path}")
