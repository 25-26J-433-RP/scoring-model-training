import pandas as pd

# Load training data with encoding fix
train_df = pd.read_csv(
    "data/aes_dataset/training_set_rel3.tsv",
    sep="\t",
    encoding="latin1",   # <-- FIX HERE
    on_bad_lines="skip"  # (optional) skips problematic lines if any
)

print("✅ Training Data Loaded")
print("Shape:", train_df.shape)
print("Columns:", train_df.columns.tolist())
print(train_df.head())

# Check score distribution
print("\nScore distribution:")
print(train_df['domain1_score'].value_counts().sort_index())
