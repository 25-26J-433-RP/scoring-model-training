# training/dataset_loader.py

import pandas as pd
from sklearn.model_selection import train_test_split


def load_sinhala_dataset(csv_path: str):
    """
    Loads the labeled Sinhala dataset.
    This file MUST contain:
        - essay_text (string)
        - grade (int)
        - richness_5 (float)
        - organization_6 (float)
        - technical_3 (float)
        - total_14 (float)
    """

    df = pd.read_csv(csv_path)

    required_cols = [
        "essay_text", "grade",
        "richness_5", "organization_6", "technical_3", "total_14"
    ]

    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # Combine text + grade for model input representation
    df["input_text"] = df.apply(lambda r: f"<GRADE={r['grade']}> {r['essay_text']}", axis=1)

    print("✅ Sinhala dataset loaded. Rows:", len(df))

    return df


def build_splits(df, test_size=0.15, seed=42):
    """
    Splits into train/validation sets.
    For Phase 6: placeholders (no training yet).
    """

    train_df, val_df = train_test_split(df, test_size=test_size, random_state=seed)

    print("🔹 Train size:", len(train_df))
    print("🔹 Val size:  ", len(val_df))

    return train_df, val_df
