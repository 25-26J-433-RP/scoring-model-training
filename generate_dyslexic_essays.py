"""
Synthetic Dyslexic Essay Generator for Sinhala Essays

This script generates synthetic dyslexic versions of existing essays by:
1. Introducing spelling variations
2. Simplifying sentence structures
3. Reducing technical scores
4. Maintaining content quality (richness and organization)
"""

import pandas as pd
import random
import re
from typing import Dict, List

# Sinhala character mappings for common dyslexic patterns
SINHALA_REVERSALS = {
    'ප': 'බ', 'බ': 'ප',  # p/b reversal
    'ද': 'ත', 'ත': 'ද',  # d/t reversal
    'ග': 'ක', 'ක': 'ග',  # g/k reversal
}

# Common Sinhala spelling errors (phonetic variations)
SINHALA_SPELLING_ERRORS = {
    'ඇති': ['ඇත', 'ඇතී'],
    'වෙයි': ['වෙයී', 'වෙය'],
    'නිසා': ['නිසා', 'නිසාා'],
    'කියා': ['කියා', 'කියාා'],
    'ගැන': ['ගැන', 'ගැනා'],
}


def apply_letter_reversals(text: str, error_rate: float = 0.05) -> str:
    """
    Apply letter reversals to simulate dyslexic writing.
    
    Args:
        text: Original Sinhala text
        error_rate: Probability of reversing each character (default 5%)
    
    Returns:
        Text with some letters reversed
    """
    chars = list(text)
    for i, char in enumerate(chars):
        if char in SINHALA_REVERSALS and random.random() < error_rate:
            chars[i] = SINHALA_REVERSALS[char]
    return ''.join(chars)


def apply_spelling_errors(text: str, error_rate: float = 0.10) -> str:
    """
    Apply common spelling errors to simulate dyslexic writing.
    
    Args:
        text: Original Sinhala text
        error_rate: Probability of introducing spelling error (default 10%)
    
    Returns:
        Text with some spelling errors
    """
    for correct, errors in SINHALA_SPELLING_ERRORS.items():
        if correct in text and random.random() < error_rate:
            replacement = random.choice(errors)
            # Replace only some occurrences
            text = text.replace(correct, replacement, random.randint(1, 3))
    return text


def remove_some_punctuation(text: str, removal_rate: float = 0.30) -> str:
    """
    Remove some punctuation marks to simulate dyslexic writing.
    
    Args:
        text: Original text
        removal_rate: Probability of removing each punctuation mark
    
    Returns:
        Text with some punctuation removed
    """
    punctuation = ['.', ',', '!', '?', ':', ';']
    for punct in punctuation:
        if random.random() < removal_rate:
            text = text.replace(punct, '', random.randint(1, 2))
    return text


def simplify_sentences(text: str) -> str:
    """
    Simplify sentence structure by breaking long sentences.
    
    Args:
        text: Original text
    
    Returns:
        Text with simplified sentences
    """
    # Split on periods and rejoin with more breaks
    sentences = text.split('.')
    simplified = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) > 100:  # Long sentence
            # Try to break at conjunctions
            sent = sent.replace(' සහ ', '. ')
            sent = sent.replace(' හා ', '. ')
            sent = sent.replace(' ද ', '. ')
        simplified.append(sent)
    return '. '.join(simplified)


def adjust_scores_for_dyslexic(scores: Dict[str, float]) -> Dict[str, float]:
    """
    Adjust rubric scores to reflect dyslexic writing patterns.
    
    Dyslexic essays typically:
    - Have similar content richness (ideas are good)
    - Have similar organization (structure is maintained)
    - Have LOWER technical scores (spelling/grammar errors)
    
    Args:
        scores: Original scores dict with richness_5, organization_6, technical_3
    
    Returns:
        Adjusted scores for dyslexic version
    """
    adjusted = scores.copy()
    
    # Richness: Slight decrease (0-1 points) - ideas still present
    adjusted['richness_5'] = max(1, scores['richness_5'] - random.choice([0, 1]))
    
    # Organization: Slight decrease (0-1 points) - structure mostly maintained
    adjusted['organization_6'] = max(1, scores['organization_6'] - random.choice([0, 1]))
    
    # Technical: Significant decrease (1-2 points) - spelling/grammar issues
    adjusted['technical_3'] = max(1, scores['technical_3'] - random.choice([1, 2]))
    
    # Recalculate total
    adjusted['total_14'] = (
        adjusted['richness_5'] + 
        adjusted['organization_6'] + 
        adjusted['technical_3']
    )
    
    return adjusted


def generate_dyslexic_version(essay_row: pd.Series, severity: str = 'moderate') -> pd.Series:
    """
    Generate a dyslexic version of an essay.
    
    Args:
        essay_row: Original essay row from DataFrame
        severity: 'mild', 'moderate', or 'severe'
    
    Returns:
        New row with dyslexic version
    """
    # Set error rates based on severity
    error_rates = {
        'mild': {'reversal': 0.03, 'spelling': 0.05, 'punctuation': 0.20},
        'moderate': {'reversal': 0.05, 'spelling': 0.10, 'punctuation': 0.30},
        'severe': {'reversal': 0.08, 'spelling': 0.15, 'punctuation': 0.40},
    }
    rates = error_rates[severity]
    
    # Apply transformations to text
    text = essay_row['essay_text']
    text = apply_letter_reversals(text, rates['reversal'])
    text = apply_spelling_errors(text, rates['spelling'])
    text = remove_some_punctuation(text, rates['punctuation'])
    text = simplify_sentences(text)
    
    # Create new row
    dyslexic_row = essay_row.copy()
    dyslexic_row['essay_id'] = f"{essay_row['essay_id']}_DYS"
    dyslexic_row['essay_text'] = text
    dyslexic_row['dyslexic_flag'] = True
    
    # Adjust scores
    original_scores = {
        'richness_5': essay_row['richness_5'],
        'organization_6': essay_row['organization_6'],
        'technical_3': essay_row['technical_3'],
        'total_14': essay_row['total_14'],
    }
    adjusted_scores = adjust_scores_for_dyslexic(original_scores)
    
    dyslexic_row['richness_5'] = adjusted_scores['richness_5']
    dyslexic_row['organization_6'] = adjusted_scores['organization_6']
    dyslexic_row['technical_3'] = adjusted_scores['technical_3']
    dyslexic_row['total_14'] = adjusted_scores['total_14']
    
    return dyslexic_row


def generate_balanced_dataset(
    input_csv: str = 'akura_dataset.csv',
    output_csv: str = 'sinhala_dataset_with_dyslexic.csv',
    dyslexic_ratio: float = 0.35,
    severity_distribution: Dict[str, float] = None
):
    """
    Generate a balanced dataset with dyslexic essays.
    
    Args:
        input_csv: Path to original dataset
        output_csv: Path to save balanced dataset
        dyslexic_ratio: Proportion of dyslexic essays (default 35%)
        severity_distribution: Distribution of severity levels
    """
    if severity_distribution is None:
        severity_distribution = {
            'mild': 0.40,      # 40% mild
            'moderate': 0.45,  # 45% moderate
            'severe': 0.15,    # 15% severe
        }
    
    print(f"[*] Loading dataset from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    print(f"[*] Original dataset: {len(df)} essays")
    print(f"   - All dyslexic_flag=FALSE")
    
    # Calculate how many dyslexic essays to generate
    num_dyslexic = int(len(df) * dyslexic_ratio / (1 - dyslexic_ratio))
    print(f"\n[*] Generating {num_dyslexic} dyslexic essays ({dyslexic_ratio*100:.1f}% of final dataset)")
    
    # Randomly select essays to create dyslexic versions
    selected_indices = random.sample(range(len(df)), num_dyslexic)
    
    dyslexic_essays = []
    severity_counts = {'mild': 0, 'moderate': 0, 'severe': 0}
    
    for idx in selected_indices:
        # Choose severity based on distribution
        severity = random.choices(
            list(severity_distribution.keys()),
            weights=list(severity_distribution.values())
        )[0]
        severity_counts[severity] += 1
        
        dyslexic_row = generate_dyslexic_version(df.iloc[idx], severity=severity)
        dyslexic_essays.append(dyslexic_row)
    
    # Combine original and dyslexic essays
    dyslexic_df = pd.DataFrame(dyslexic_essays)
    balanced_df = pd.concat([df, dyslexic_df], ignore_index=True)
    
    # Shuffle the dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to CSV
    balanced_df.to_csv(output_csv, index=False, encoding='utf-8')
    
    print(f"\n[SUCCESS] Balanced dataset saved to {output_csv}")
    print(f"   - Total essays: {len(balanced_df)}")
    print(f"   - Non-dyslexic: {len(df)} ({len(df)/len(balanced_df)*100:.1f}%)")
    print(f"   - Dyslexic: {num_dyslexic} ({num_dyslexic/len(balanced_df)*100:.1f}%)")
    print(f"\n[*] Severity distribution:")
    for severity, count in severity_counts.items():
        print(f"   - {severity.capitalize()}: {count} ({count/num_dyslexic*100:.1f}%)")
    
    # Show sample statistics
    print(f"\n[*] Score statistics:")
    print("\nNon-dyslexic essays:")
    print(df[['richness_5', 'organization_6', 'technical_3', 'total_14']].describe())
    print("\nDyslexic essays:")
    print(dyslexic_df[['richness_5', 'organization_6', 'technical_3', 'total_14']].describe())


if __name__ == "__main__":
    # Generate balanced dataset
    generate_balanced_dataset(
        input_csv='akura_dataset.csv',
        output_csv='sinhala_dataset_with_dyslexic.csv',
        dyslexic_ratio=0.35,  # 35% dyslexic essays
    )
    
    print("\n" + "="*60)
    print("[SUCCESS] Dataset generation complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the generated dataset: sinhala_dataset_with_dyslexic.csv")
    print("2. Manually inspect some dyslexic essays for quality")
    print("3. Retrain your model with the balanced dataset")
    print("4. Re-evaluate fairness metrics")
