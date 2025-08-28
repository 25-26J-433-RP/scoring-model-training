import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LinearRegression
import joblib

# -------------------------
# Paths & Setup
# -------------------------
TRAIN_PATH = "data/aes_dataset/training_set_rel3.tsv"
MODEL_PATH = "models/bert_regressor.pt"
CALIBRATION_PATH = "models/calibration.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load Tokenizer & Model
# -------------------------
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

class BertRegressor(torch.nn.Module):
    def __init__(self, bert_model):
        super(BertRegressor, self).__init__()
        self.bert = bert_model
        self.out = torch.nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token
        score = self.out(pooled_output)
        return score

# Load trained model
bert = AutoModel.from_pretrained("bert-base-uncased")
model = BertRegressor(bert).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv(TRAIN_PATH, sep="\t", encoding="ISO-8859-1")
essays = df["essay"].astype(str).tolist()
teacher_scores = df["domain1_score"].tolist()

print(f"📊 Loaded {len(essays)} essays for calibration")

# -------------------------
# Get raw model predictions
# -------------------------
raw_preds = []
with torch.no_grad():
    for essay in essays:
        tokens = tokenizer(
            essay,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length"
        )
        input_ids = tokens["input_ids"].to(DEVICE)
        attention_mask = tokens["attention_mask"].to(DEVICE)

        raw_score = model(input_ids, attention_mask).item()
        raw_preds.append(raw_score)

print("✅ Raw predictions collected")

# -------------------------
# Fit calibration model
# -------------------------
calibrator = LinearRegression()
calibrator.fit([[p] for p in raw_preds], teacher_scores)

print("✅ Calibration model trained")

# -------------------------
# Save calibrator + score range
# -------------------------
score_range = (min(teacher_scores), max(teacher_scores))  # actual teacher min/max
joblib.dump({"calibrator": calibrator, "score_range": score_range}, CALIBRATION_PATH)

print(f"💾 Saved calibration model + score range → {CALIBRATION_PATH}")
