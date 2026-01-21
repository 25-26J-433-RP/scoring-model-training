import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from transformers import AutoTokenizer, AutoConfig

from .dataset_loader import load_sinhala_dataset, build_splits
from .model_multitask_xlmr import SinhalaMultiHeadRegressor


# =============================
# DEVICE
# =============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔥 Using device:", DEVICE)


# =============================
# DATASET
# =============================
class SinhalaDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512):
        self.df = df.reset_index(drop=True)
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
            "grade_id": torch.tensor(row["grade"], dtype=torch.long),

            # targets
            "richness": torch.tensor(row["richness_5"], dtype=torch.float),
            "organization": torch.tensor(row["organization_6"], dtype=torch.float),
            "technical": torch.tensor(row["technical_3"], dtype=torch.float),
            "total": torch.tensor(row["total_14"], dtype=torch.float),
        }


# =============================
# TRAINING LOOP
# =============================
def train_model(
    model_name: str,
    csv_path: str = "sinhala_dataset_final_with_dyslexic.csv",
    epochs: int = 20,
    batch_size: int = 4,
    lr: float = 1e-5,  # Reduced learning rate to prevent NaN
):
    print(f"\n🚀 Training Sinhala Multi-Head Model using: {model_name}")

    # ---- Load dataset
    df = load_sinhala_dataset(csv_path)
    train_df, val_df = build_splits(df)
    
    print(f"📊 Dataset loaded: {len(df)} total essays")
    print(f"   Training: {len(train_df)} essays")
    print(f"   Validation: {len(val_df)} essays")

    # ---- Tokenizer (SOURCE OF TRUTH)
    tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=True
)


    train_ds = SinhalaDataset(train_df, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # ---- Config + Model (HF-SAFE)
    config = AutoConfig.from_pretrained(model_name)
    model = SinhalaMultiHeadRegressor(config).to(DEVICE)


    # =============================
    # PARTIAL FINE-TUNING (CRITICAL)
    # =============================

    # Freeze entire encoder
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Unfreeze last 4 transformer layers
    for layer in model.encoder.encoder.layer[-4:]:
        for param in layer.parameters():
            param.requires_grad = True

    # =============================
    # MODEL IDENTITY (NICE-TO-HAVE)
    # =============================
    model.config.architectures = ["SinhalaMultiHeadRegressor"]

    # ---- Optim (ONLY trainable params)
    loss_fn = torch.nn.SmoothL1Loss()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    # =============================
    # EPOCHS
    # =============================
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):

            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            grade_id = batch["grade_id"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                grade_id=grade_id
            )

            loss = (
                0.2 * loss_fn(outputs["richness_5"], batch["richness"].to(DEVICE)) +
                0.2 * loss_fn(outputs["organization_6"], batch["organization"].to(DEVICE)) +
                0.2 * loss_fn(outputs["technical_3"], batch["technical"].to(DEVICE)) +
                0.4 * loss_fn(outputs["total_14"], batch["total"].to(DEVICE))
            )

            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        print(f"✅ Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

    # =============================
    # SAVE (HF STANDARD — FINAL)
    # =============================
    OUTPUT_DIR = "xlm-roberta-large-sinhala-multihead"

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n🎉 TRAINING COMPLETE")
    print("📦 Model saved to:", OUTPUT_DIR)
    print("✅ Ready for Hugging Face & backend loading")


# =============================
# ENTRY POINT
# =============================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise ValueError(
            "Usage: python -m training.train_model xlm-roberta-large"
        )

    model_name = sys.argv[1]
    train_model(model_name)
