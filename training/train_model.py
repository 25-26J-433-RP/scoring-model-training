# training/train_model.py

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AdamW
from tqdm import tqdm

from .dataset_loader import load_sinhala_dataset, build_splits
from .model_multitask_xlmr import SinhalaMultiHeadRegressor


class SinhalaDataset(Dataset):
    """
    Dataset class for multi-head Sinhala scoring.
    """

    def __init__(self, df, tokenizer, max_len=512):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        encoded = self.tokenizer(
            row["input_text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        labels = {
            "richness_5": torch.tensor(row["richness_5"], dtype=torch.float),
            "organization_6": torch.tensor(row["organization_6"], dtype=torch.float),
            "technical_3": torch.tensor(row["technical_3"], dtype=torch.float),
            "total_14": torch.tensor(row["total_14"], dtype=torch.float),
        }

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": labels,
        }


def train_phase():
    """
    DO NOT RUN NOW.
    This function will activate only when teacher scores arrive.
    """

    raise RuntimeError("❗ Training cannot start yet — teacher rubric scores are required.")

    # Placeholder for future:
    """
    df = load_sinhala_dataset("sinhala_dataset_v2.csv")
    train_df, val_df = build_splits(df)
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = SinhalaMultiHeadRegressor()

    # Prepare DataLoaders, optimizer, loss etc.
    # And start training with MSE loss per head.
    """

