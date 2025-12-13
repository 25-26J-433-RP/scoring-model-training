import torch
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from transformers import AutoTokenizer

from .dataset_loader import load_sinhala_dataset, build_splits
from .model_multitask_xlmr import SinhalaMultiHeadRegressor


# =============================
# DEVICE
# =============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(
    model_path: str,
    csv_path: str = "sinhala_dataset_final.csv"
):
    print(f"\n🔍 Evaluating model from: {model_path}")

    # ---- Load dataset
    df = load_sinhala_dataset(csv_path)
    train_df, val_df = build_splits(df)

    # ---- HF-CORRECT LOADING (SAME AS BACKEND)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = SinhalaMultiHeadRegressor.from_pretrained(model_path)
    model.to(DEVICE)
    model.eval()

    y_true = []
    y_pred = []

    print("\n🚀 Running inference on validation set...")

    for _, row in val_df.iterrows():

        enc = tokenizer(
            row["input_text"],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512
        )

        input_ids = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)

        grade_id = torch.tensor(
            [int(row["grade"])],
            dtype=torch.long,
            device=DEVICE
        )

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                grade_id=grade_id
            )

        pred_richness = outputs["richness_5"].item()
        pred_org = outputs["organization_6"].item()
        pred_tech = outputs["technical_3"].item()

        pred_total = pred_richness + pred_org + pred_tech


        y_true.append(float(row["total_14"]))
        y_pred.append(pred_total)

    # =============================
    # METRICS
    # =============================
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    r, _ = pearsonr(y_true, y_pred)

    print("\n📊 === Evaluation Results ===")
    print(f"MAE:       {mae:.3f}")
    print(f"RMSE:      {rmse:.3f}")
    print(f"Pearson r: {r:.3f}")

    print("\nTrue Scores:", y_true)
    print("Predicted :", [round(x, 2) for x in y_pred])


# =============================
# ENTRY POINT
# =============================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise ValueError(
            "Usage: python -m training.eval_model <MODEL_DIR>"
        )

    model_path = sys.argv[1]
    evaluate_model(model_path)
