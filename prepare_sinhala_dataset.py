import pandas as pd

INPUT = "akura_dataset.csv"
OUTPUT = "sinhala_dataset_final.csv"

df = pd.read_csv(INPUT)

required = [
    "essay_text",
    "grade",
    "richness_5",
    "organization_6",
    "technical_3",
    "total_14"
]

# Validate
for c in required:
    if c not in df.columns:
        raise ValueError(f"Missing column: {c}")

# Clean numerics
numeric_cols = ["grade", "richness_5", "organization_6", "technical_3", "total_14"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=numeric_cols)

# Build conditioning input text
df["input_text"] = df.apply(
    lambda r: f"<GRADE={int(r['grade'])}> {str(r['essay_text'])}",
    axis=1
)

# Final dataset
df = df[[
    "input_text",
    "grade",
    "richness_5",
    "organization_6",
    "technical_3",
    "total_14"
]]

df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
print("✔ CLEAN dataset saved:", OUTPUT)
print("✔ Total rows:", len(df))
