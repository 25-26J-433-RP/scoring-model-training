import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer

from .dataset_loader import load_sinhala_dataset, build_splits
from .model_multitask_xlmr import SinhalaMultiHeadRegressor


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔥 Using device:", DEVICE)


class SinhalaDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        enc = self.tokenizer(
            row["input_text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "grade_ids": torch.tensor(row["grade"], dtype=torch.long),
            "richness_5": torch.tensor(row["richness_5"], dtype=torch.float),
            "organization_6": torch.tensor(row["organization_6"], dtype=torch.float),
            "technical_3": torch.tensor(row["technical_3"], dtype=torch.float),
            "total_14": torch.tensor(row["total_14"], dtype=torch.float),
        }


def train_model(model_name, csv_path="sinhala_dataset_final.csv",
                epochs=20, batch_size=4, lr=3e-5):

    print(f"\n🚀 Training Multi-Head Model using: {model_name}")

    df = load_sinhala_dataset(csv_path)
    train_df, val_df = build_splits(df)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = SinhalaDataset(train_df, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = SinhalaMultiHeadRegressor(model_name).to(DEVICE)

    loss_fn = torch.nn.SmoothL1Loss()
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader):

            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            grade_ids = batch["grade_ids"].to(DEVICE)

            out = model(input_ids, attention_mask, grade_ids)

            # Weighted loss
            loss = (
                0.2 * loss_fn(out["richness_5"], batch["richness_5"].to(DEVICE)) +
                0.2 * loss_fn(out["organization_6"], batch["organization_6"].to(DEVICE)) +
                0.2 * loss_fn(out["technical_3"], batch["technical_3"].to(DEVICE)) +
                0.4 * loss_fn(out["total_14"], batch["total_14"].to(DEVICE))
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss / len(train_loader)}")

    save_path = f"{model_name.replace('/', '_')}_sinhala_multihead.pt"
    torch.save(model.state_dict(), save_path)
    print("✔ Model saved:", save_path)


if __name__ == "__main__":
    import sys
    print("🟡 MAIN BLOCK RUNNING")
    model_name = sys.argv[1]
    train_model(model_name)
