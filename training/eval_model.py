# training/eval_model.py

import torch
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from transformers import AutoTokenizer

from .dataset_loader import load_sinhala_dataset, build_splits
from .model_multitask_xlmr import SinhalaMultiHeadRegressor


def evaluate_model(model_path, csv_path="sinhala_dataset_final.csv"):
    print(f"\n🔍 Evaluating model: {model_path}")

    df = load_sinhala_dataset(csv_path)
    train_df, val_df = build_splits(df)

    base_model = "xlm-roberta-large"

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = SinhalaMultiHeadRegressor(base_model)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
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

        # grade embedding input
        grade_id = torch.tensor([int(row["grade"])], dtype=torch.long)

        with torch.no_grad():
            out = model(
                enc["input_ids"],
                enc["attention_mask"],
                grade_id
            )

        pred_total = out["total_14"].item()

        y_true.append(row["total_14"])
        y_pred.append(pred_total)

    # ---- METRICS ----
    mae = mean_absolute_error(y_true, y_pred)

    # FIXED: no squared parameter
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5

    r, _ = pearsonr(y_true, y_pred)

    print("\n📊 === Evaluation Results ===")
    print(f"MAE:       {mae:.3f}")
    print(f"RMSE:      {rmse:.3f}")
    print(f"Pearson r: {r:.3f}")

    print("\nTrue Scores:", y_true)
    print("Predicted :", [round(x, 2) for x in y_pred])


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1]
    evaluate_model(model_path)
