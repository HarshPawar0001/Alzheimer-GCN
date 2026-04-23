"""Clinical data branch: small MLP to produce a 16-dim embedding."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn


class ClinicalBranch(nn.Module):
    """Multilayer perceptron for clinical feature embeddings."""

    def __init__(self, config: Dict) -> None:
        super().__init__()

        # 🔥 IMPORTANT FIX: config passed is already model config
        model_cfg = config

        in_dim: int = int(model_cfg.get("clinical_input_dim", 2))
        hidden_dim: int = int(model_cfg.get("clinical_hidden_dim", 32))
        emb_dim: int = int(model_cfg.get("clinical_embedding_dim", 16))

        print(f"[DEBUG] ClinicalBranch input dim = {in_dim}")

        self.net: nn.Sequential = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, emb_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
