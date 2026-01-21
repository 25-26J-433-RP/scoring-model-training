"""
Merge real dyslexic essays with synthetically generated ones
"""

import pandas as pd

print("[*] Merging Real and Synthetic Dyslexic Essays")
print("="*60)

# Load the original dataset with your manually added dyslexic essays
print("\n[1] Loading your manually curated dataset...")
real_df = pd.read_csv('../Akura dataset(Sheet1) (3).csv')
print(f"   Total essays: {len(real_df)}")
print(f"   Dyslexic essays: {real_df['dyslexic_flag'].sum()}")
print(f"   Non-dyslexic essays: {(~real_df['dyslexic_flag']).sum()}")

# Load the synthetically generated dataset
print("\n[2] Loading synthetically generated dataset...")
try:
    synthetic_df = pd.read_csv('sinhala_dataset_with_dyslexic.csv')
    print(f"   Total essays: {len(synthetic_df)}")
    print(f"   Dyslexic essays: {synthetic_df['dyslexic_flag'].sum()}")
    print(f"   Non-dyslexic essays: {(~synthetic_df['dyslexic_flag']).sum()}")
except FileNotFoundError:
    print("   [ERROR] Synthetic dataset not found!")
    print("   Run: python generate_dyslexic_essays.py first")
    exit(1)

# Extract only the dyslexic essays from synthetic dataset
print("\n[3] Extracting synthetic dyslexic essays...")
synthetic_dyslexic = synthetic_df[synthetic_df['dyslexic_flag'] == True].copy()
print(f"   Synthetic dyslexic essays: {len(synthetic_dyslexic)}")

# Extract only non-dyslexic essays from real dataset
print("\n[4] Extracting non-dyslexic essays from real dataset...")
real_non_dyslexic = real_df[real_df['dyslexic_flag'] == False].copy()
print(f"   Real non-dyslexic essays: {len(real_non_dyslexic)}")

# Extract dyslexic essays from real dataset
print("\n[5] Extracting real dyslexic essays...")
real_dyslexic = real_df[real_df['dyslexic_flag'] == True].copy()
print(f"   Real dyslexic essays: {len(real_dyslexic)}")

# Combine all essays
print("\n[6] Combining datasets...")
combined_df = pd.concat([
    real_non_dyslexic,      # All non-dyslexic from original
    real_dyslexic,          # Your manually added dyslexic essays
    synthetic_dyslexic      # Synthetically generated dyslexic essays
], ignore_index=True)

# Shuffle the dataset
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the combined dataset
output_file = 'sinhala_dataset_final_with_dyslexic.csv'
combined_df.to_csv(output_file, index=False, encoding='utf-8')

print(f"\n[SUCCESS] Combined dataset saved to: {output_file}")
print("="*60)
print("\n[*] Final Dataset Statistics:")
print(f"   Total essays: {len(combined_df)}")
print(f"   Non-dyslexic: {(~combined_df['dyslexic_flag']).sum()} ({(~combined_df['dyslexic_flag']).sum()/len(combined_df)*100:.1f}%)")
print(f"   Dyslexic (Real): {len(real_dyslexic)} ({len(real_dyslexic)/len(combined_df)*100:.1f}%)")
print(f"   Dyslexic (Synthetic): {len(synthetic_dyslexic)} ({len(synthetic_dyslexic)/len(combined_df)*100:.1f}%)")
print(f"   Dyslexic (Total): {combined_df['dyslexic_flag'].sum()} ({combined_df['dyslexic_flag'].sum()/len(combined_df)*100:.1f}%)")

print("\n[*] Score Comparison:")
print("\nNon-Dyslexic Essays:")
print(combined_df[~combined_df['dyslexic_flag']][['richness_5', 'organization_6', 'technical_3', 'total_14']].describe())

print("\nReal Dyslexic Essays:")
real_dyslexic_ids = real_dyslexic['essay_id'].tolist()
real_dyslexic_in_combined = combined_df[combined_df['essay_id'].isin(real_dyslexic_ids)]
print(real_dyslexic_in_combined[['richness_5', 'organization_6', 'technical_3', 'total_14']].describe())

print("\nSynthetic Dyslexic Essays:")
synthetic_dyslexic_ids = synthetic_dyslexic['essay_id'].tolist()
synthetic_dyslexic_in_combined = combined_df[combined_df['essay_id'].isin(synthetic_dyslexic_ids)]
print(synthetic_dyslexic_in_combined[['richness_5', 'organization_6', 'technical_3', 'total_14']].describe())

print("\n" + "="*60)
print("[SUCCESS] Dataset merge complete!")
print("="*60)
print("\nNext steps:")
print("1. Review the combined dataset: sinhala_dataset_final_with_dyslexic.csv")
print("2. Retrain your model with this balanced dataset")
print("3. Re-evaluate fairness metrics")
