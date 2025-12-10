import pandas as pd
from sklearn.model_selection import train_test_split


def load_sinhala_dataset(path):
    df = pd.read_csv(path)

    required = [
        "input_text",
        "grade",            
        "richness_5",
        "organization_6",
        "technical_3",
        "total_14"
    ]

    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    return df


def build_splits(df, test_size=0.15):
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42
    )
    return train_df, val_df
