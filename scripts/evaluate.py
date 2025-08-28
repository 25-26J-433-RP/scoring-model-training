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
CALIBRATION_PATH = "models/calibration_set{}.pkl"   # per essay set
DATA_PATH = "data/aes_dataset/training_set_rel3.tsv"
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

# -------------------------
# Prediction Function
# -------------------------
def predict_score(essay_text, essay_set: int):
    # Load calibration for this essay set
    calib_file = CALIBRATION_PATH.format(essay_set)
    if not os.path.exists(calib_file):
        raise FileNotFoundError(f"❌ Calibration file missing for set {essay_set}: {calib_file}")
    calib_data = joblib.load(calib_file)
    calibrator = calib_data["calibrator"]
    score_range = calib_data["score_range"]  # (min, max teacher score)

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

    calibrated_score = calibrator.predict([[raw_score]])[0]

    # Normalize into 0–100 scale
    min_score, max_score = score_range
    scaled_score = 100 * (calibrated_score - min_score) / (max_score - min_score)
    scaled_score = max(0, min(100, scaled_score))  # clip

    return round(scaled_score, 2), round(calibrated_score, 2), round(raw_score, 2)

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("❌ Usage:")
        print("   python scripts/evaluate.py my_essay.txt <essay_set>")
        print("   python scripts/evaluate.py --bulk <essay_set>")
        sys.exit(1)

    arg = sys.argv[1]
    essay_set = int(sys.argv[2])  # user specifies which set to use

    # -------- Option 2: Single Essay --------
    if arg.endswith(".txt"):
        essay_file = arg
        with open(essay_file, "r", encoding="utf-8") as f:
            essay_text = f.read()

        scaled_score, calibrated_score, raw_score = predict_score(essay_text, essay_set)
        print(f"📄 Essay: {essay_file} | Essay Set: {essay_set}")
        print(f"   🔹 Raw Model Score: {raw_score}")
        print(f"   🔹 Calibrated Score (teacher scale): {calibrated_score}")
        print(f"   🔹 Normalized Score (0–100): {scaled_score}")

    # -------- Option 1: Bulk Validation --------
    elif arg == "--bulk":
        df = pd.read_csv(DATA_PATH, sep="\t", encoding="ISO-8859-1")
        df = df[df["essay_set"] == essay_set]  # only essays from this set
        essays = df["essay"].astype(str).tolist()
        teacher_scores = df["domain1_score"].tolist()

        predictions, raw_scores, calib_scores = [], [], []

        for essay in essays:
            scaled, calib, raw = predict_score(essay, essay_set)
            predictions.append(scaled)
            calib_scores.append(calib)
            raw_scores.append(raw)

        rmse = np.sqrt(mean_squared_error(teacher_scores, calib_scores))
        print(f"✅ Bulk evaluation done | Essay Set {essay_set} | RMSE (calibrated): {rmse:.4f}")

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
