import pandas as pd
from sklearn.model_selection import train_test_split


def load_sinhala_dataset(path):
    df = pd.read_csv(path)
    
    # Handle different column names for essay text
    if "essay_text" in df.columns and "input_text" not in df.columns:
        df = df.rename(columns={"essay_text": "input_text"})
    
    # Drop rows with missing values in critical columns
    required_cols = ["grade", "richness_5", "organization_6", "technical_3", "total_14", "input_text"]
    df = df.dropna(subset=required_cols)
    
    # Convert grade to integer (in case it's float like 5.0)
    df["grade"] = df["grade"].astype(int)
    
    # Reset index after dropping rows
    df = df.reset_index(drop=True)

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
