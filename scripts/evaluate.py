import torch
from transformers import AutoTokenizer, AutoModel
import joblib
import sys
import os
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np

# -------------------------
# Paths & Setup
# -------------------------
MODEL_PATH = "models/bert_regressor.pt"
CALIBRATION_PATH = "models/calibration.pkl"
DATA_PATH = "data/aes_dataset/training_set_rel3.tsv"   # used for bulk
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
        self.out = torch.nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token
        score = self.out(pooled_output)
        return score

bert = AutoModel.from_pretrained("bert-base-uncased")
model = BertRegressor(bert).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Load calibration model + teacher score range
calib_data = joblib.load(CALIBRATION_PATH)
calibrator = calib_data["calibrator"]
score_range = calib_data["score_range"]  # (min_score, max_score)

# -------------------------
# Prediction Function
# -------------------------
def predict_score(essay_text):
    tokens = tokenizer(
        essay_text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding="max_length"
    )
    input_ids = tokens["input_ids"].to(DEVICE)
    attention_mask = tokens["attention_mask"].to(DEVICE)

    with torch.no_grad():
        raw_score = model(input_ids, attention_mask).item()

    # Apply calibration mapping
    calibrated_score = calibrator.predict([[raw_score]])[0]

    # Scale into 0–100
    min_score, max_score = score_range
    scaled_score = 100 * (calibrated_score - min_score) / (max_score - min_score)
    scaled_score = max(0, min(100, scaled_score))  # clip

    return round(scaled_score, 2), round(calibrated_score, 2), round(raw_score, 2)

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Usage:")
        print("   python scripts/evaluate.py my_essay.txt   # single essay")
        print("   python scripts/evaluate.py --bulk         # bulk validation")
        sys.exit(1)

    arg = sys.argv[1]

    # -------- Option 2: Single Essay --------
    if arg.endswith(".txt"):
        essay_file = arg
        with open(essay_file, "r", encoding="utf-8") as f:
            essay_text = f.read()

        scaled_score, calibrated_score, raw_score = predict_score(essay_text)
        print(f"📄 Essay: {essay_file}")
        print(f"   🔹 Raw Model Score: {raw_score}")
        print(f"   🔹 Calibrated Score (teacher scale): {calibrated_score}")
        print(f"   🔹 Normalized Score (0–100): {scaled_score}")

    # -------- Option 1: Bulk Validation --------
    elif arg == "--bulk":
        df = pd.read_csv(DATA_PATH, sep="\t", encoding="ISO-8859-1")
        essays = df["essay"].astype(str).tolist()
        teacher_scores = df["domain1_score"].tolist()

        predictions = []
        raw_scores = []
        calib_scores = []

        for essay in essays:
            scaled, calib, raw = predict_score(essay)
            predictions.append(scaled)
            calib_scores.append(calib)
            raw_scores.append(raw)

        rmse = np.sqrt(mean_squared_error(teacher_scores, calib_scores))
        print(f"✅ Bulk evaluation done | RMSE (calibrated): {rmse:.4f}")

        os.makedirs("outputs", exist_ok=True)
        out_df = pd.DataFrame({
            "essay": essays,
            "teacher_score": teacher_scores,
            "raw_score": raw_scores,
            "calibrated_score": calib_scores,
            "scaled_score_0_100": predictions
        })
        out_df.to_csv(OUTPUT_PATH, index=False)
        print(f"📂 Saved results to {OUTPUT_PATH}")

    else:
        print("❌ Invalid argument. Use a .txt file or --bulk")
