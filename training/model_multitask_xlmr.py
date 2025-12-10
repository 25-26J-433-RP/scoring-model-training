import torch
import torch.nn as nn
from transformers import AutoModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SinhalaMultiHeadRegressor(nn.Module):
    def __init__(self, model_name):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)

        hidden = self.encoder.config.hidden_size

        # ---- Freeze bottom layers (prevent underfitting) ----
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Unfreeze last 4 layers
        for layer in self.encoder.encoder.layer[-4:]:
            for param in layer.parameters():
                param.requires_grad = True

        # ---- Grade embedding ----
        self.grade_embed = nn.Embedding(10, hidden)  # grades 0–9 mapped to vector

        # ---- Dropout ----
        self.dropout = nn.Dropout(0.2)

        # ---- Regression heads ----
        self.head_richness = nn.Linear(hidden, 1)
        self.head_org = nn.Linear(hidden, 1)
        self.head_tech = nn.Linear(hidden, 1)
        self.head_total = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask, grade_ids):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls = out.last_hidden_state[:, 0, :]

        # Merge grade embedding
        grade_vec = self.grade_embed(grade_ids)
        cls = cls + grade_vec

        cls = self.dropout(cls)

        return {
            "richness_5": self.head_richness(cls).squeeze(-1),
            "organization_6": self.head_org(cls).squeeze(-1),
            "technical_3": self.head_tech(cls).squeeze(-1),
            "total_14": self.head_total(cls).squeeze(-1),
        }
