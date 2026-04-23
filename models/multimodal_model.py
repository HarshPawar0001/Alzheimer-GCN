"""Multimodal Alzheimer’s disease classification model.

This module defines `MultiModalADModel`, which:
  - Processes DTI brain networks via the `DTIBranch` (GCN + self-attention).
  - Processes clinical features via the `ClinicalBranch` MLP.
  - Fuses the resulting embeddings with late fusion (concatenation).
  - Classifies subjects into AD vs NC using a small MLP classifier.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from .clinical_branch import ClinicalBranch
from .dti_branch import DTIBranch


class MultiModalADModel(nn.Module):
    """Full multimodal model combining DTI graphs and clinical data."""

    def __init__(self, config: Dict) -> None:
        """Initialize the multimodal model using configuration.

        Args:
            config: Parsed configuration dictionary (from `config.yaml`).
                Expected keys under `model`:
                    - dti_embedding_dim
                    - clinical_embedding_dim
                    - fused_dim
                    - classifier_hidden
                    - num_classes
                    - dropout
        """
        super().__init__()
        model_cfg = config.get("model", {})

        dti_emb_dim: int = int(model_cfg.get("dti_embedding_dim", 32))
        clinical_emb_dim: int = int(model_cfg.get("clinical_embedding_dim", 16))
        fused_dim: int = int(model_cfg.get("fused_dim", dti_emb_dim + clinical_emb_dim))
        classifier_hidden: int = int(model_cfg.get("classifier_hidden", 32))
        num_classes: int = int(model_cfg.get("num_classes", 2))
        dropout_p: float = float(model_cfg.get("dropout", 0.5))

        self.dti_branch: DTIBranch = DTIBranch(config)
        self.clinical_branch: ClinicalBranch = ClinicalBranch(config)

        self.classifier: nn.Sequential = nn.Sequential(
            nn.Linear(dti_emb_dim + clinical_emb_dim, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, graph_data: Data, clinical_features: Tensor) -> Tensor:
        """Forward pass of the multimodal model.

        Args:
            graph_data: Batched PyG `Data` or `Batch` object representing
                DTI brain networks.
            clinical_features: Tensor of shape (batch_size, 6) containing
                clinical feature vectors.

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        dti_emb: Tensor = self.dti_branch(graph_data)  # (B, 32)
        clin_emb: Tensor = self.clinical_branch(clinical_features)  # (B, 16)

        fused: Tensor = torch.cat([dti_emb, clin_emb], dim=1)  # (B, 48)
        logits: Tensor = self.classifier(fused)
        return logits

    def get_attention_scores(self, graph_data: Data) -> Tensor:
        """Return self-attention scores from the DTI branch for visualization.

        Args:
            graph_data: Batched PyG `Data` or `Batch` object.

        Returns:
            Tensor of shape (N,) with attention scores for each node across
            all graphs in the batch.
        """
        return self.dti_branch.get_attention_scores(graph_data)

