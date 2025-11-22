import pandas as pd

df = pd.read_csv("Phase5 data training.csv")

clean_df = pd.DataFrame({
    "essay_text": df["essay_text"].astype(str),
    "grade": df["grade"].astype(int),
    "topic": df["essay_topic"].astype(str),
    "teacher_score": df["normalized_score"].astype(float)
})

clean_df.to_csv("sinhala_dataset_for_training.csv", index=False)

print("✅ Sinhala dataset prepared!")
print(clean_df.head())
