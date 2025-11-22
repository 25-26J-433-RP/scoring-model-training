# training/model_multitask_xlmr.py

import torch
import torch.nn as nn
from transformers import AutoModel


class SinhalaMultiHeadRegressor(nn.Module):
    """
    Multi-task head for Sinhala scoring:
        - richness_5
        - organization_6
        - technical_3
        - total_14
    """

    def __init__(self, model_name="xlm-roberta-base"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

        hidden = self.encoder.config.hidden_size

        # Four regression heads
        self.richness_head = nn.Linear(hidden, 1)
        self.organization_head = nn.Linear(hidden, 1)
        self.technical_head = nn.Linear(hidden, 1)
        self.total_head = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]

        return {
            "richness_5": self.richness_head(cls).squeeze(-1),
            "organization_6": self.organization_head(cls).squeeze(-1),
            "technical_3": self.technical_head(cls).squeeze(-1),
            "total_14": self.total_head(cls).squeeze(-1),
        }
