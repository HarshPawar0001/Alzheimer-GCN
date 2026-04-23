"""Loss utilities for Alzheimer’s multimodal classification.

This module computes class-balanced weights and constructs a weighted
cross-entropy loss function based on training labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor, nn
import yaml


def load_config(config_path: Path) -> Dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to the `config.yaml` file.

    Returns:
        Parsed configuration dictionary.
    """
    try:
        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except OSError as exc:
        raise RuntimeError(f"Failed to read config file at '{config_path}'.") from exc
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Failed to parse YAML config at '{config_path}'.") from exc


def compute_class_weights(labels: Tensor) -> Tensor:
    """Compute class weights for binary classification (AD vs NC).

    The weights follow:
        weight_AD = total_samples / (2 * num_AD_samples)
        weight_NC = total_samples / (2 * num_NC_samples)

    Args:
        labels: 1D tensor of integer class labels (0 or 1) for training data.

    Returns:
        Tensor of shape (2,) containing weights for classes [0 (AD), 1 (NC)].
    """
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D tensor.")

    total: int = int(labels.numel())
    if total == 0:
        raise ValueError("labels tensor is empty; cannot compute class weights.")

    num_ad: int = int((labels == 0).sum().item())
    num_nc: int = int((labels == 1).sum().item())

    if num_ad == 0 or num_nc == 0:
        raise ValueError(
            "Both classes must be present in training labels to compute weights."
        )

    weight_ad = total / (2.0 * float(num_ad))
    weight_nc = total / (2.0 * float(num_nc))

    weights = torch.tensor([weight_ad, weight_nc], dtype=torch.float32)
    return weights


def build_loss_fn(train_labels: Tensor, device: torch.device) -> nn.Module:
    """Create a class-weighted CrossEntropyLoss instance.

    Args:
        train_labels: 1D tensor of integer class labels (0 or 1) from the
            training set.
        device: Torch device on which the model and loss run.

    Returns:
        An initialized `torch.nn.CrossEntropyLoss` with class weights.
    """
    class_weights = compute_class_weights(train_labels).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    return loss_fn

