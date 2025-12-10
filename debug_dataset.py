import pandas as pd

df = pd.read_csv("sinhala_dataset_final.csv")

print("\n=== COLUMN TYPES ===")
print(df[["richness_5", "organization_6", "technical_3", "total_14"]].dtypes)

print("\n=== NaN CHECK ===")
print(df.isna().sum())

print("\n=== UNIQUE VALUES ===")
print("Richness:", df["richness_5"].unique())
print("Organization:", df["organization_6"].unique())
print("Technical:", df["technical_3"].unique())
print("Total:", df["total_14"].unique())
