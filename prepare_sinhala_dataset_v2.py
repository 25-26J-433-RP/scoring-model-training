import pandas as pd

INPUT_FILE = "Phase5 data training.csv"
OUTPUT_FILE = "sinhala_dataset_v2.csv"

df = pd.read_csv(INPUT_FILE)

def build_combined(row):
    grade = row["grade"]
    topic = row["essay_topic"]
    lang = row["rubric_score_language"]
    content = row["rubric_score_content"]
    relevance = row["rubric_score_relevance"]
    essay = row["essay_text"]

    combined = (
        f"[GRADE={grade}] "
        f"[TOPIC={topic}] "
        f"[RUBRIC L={lang} C={content} R={relevance}] "
        f"[ESSAY] {essay}"
    )

    return combined

print("🔄 Building V2 Sinhala training dataset...")

df["combined_input"] = df.apply(build_combined, axis=1)

clean = df[["combined_input", "normalized_score"]]
clean.columns = ["text", "label"]

clean.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("✅ DONE! Saved to:", OUTPUT_FILE)
print(clean.head())
