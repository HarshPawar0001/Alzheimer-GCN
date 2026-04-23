"""Graph construction helper utilities for DTI brain networks.

This module provides functions to convert dense adjacency matrices and node
features into PyTorch Geometric graph objects, which are later consumed by
the GCN-based DTI branch of the model.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data


def adjacency_to_edge_index(adj_matrix: np.ndarray, threshold: float = 0.0) -> Tuple[Tensor, Tensor]:
    """Convert a dense adjacency matrix into PyG `edge_index` and `edge_attr`.

    All entries strictly greater than the provided threshold are treated as
    edges. Zero (or below-threshold) entries are considered absent edges.

    Args:
        adj_matrix: Square numpy array of shape (N, N) representing the
            weighted adjacency matrix of the brain network.
        threshold: Minimum value for an entry to be treated as an edge.

    Returns:
        edge_index: LongTensor of shape (2, E) with COO indices.
        edge_attr: FloatTensor of shape (E,) containing edge weights.

    Raises:
        ValueError: If the adjacency matrix is not a 2D square array.
    """
    try:
        if adj_matrix.ndim != 2 or adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError(
                f"Adjacency matrix must be square, got shape {adj_matrix.shape}."
            )

        # Ensure we are working with float values.
        adj = adj_matrix.astype(np.float32)

        # Mask of valid (non-zero and above-threshold) edges.
        mask = adj > threshold
        row, col = np.nonzero(mask)
        weights = adj[mask]

        edge_index = torch.from_numpy(
            np.stack([row.astype(np.int64), col.astype(np.int64)], axis=0)
        )
        edge_attr = torch.from_numpy(weights)

        return edge_index, edge_attr
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError("Failed to convert adjacency matrix to edge index.") from exc


def build_graph_data(
    adj_matrix: np.ndarray,
    node_features: np.ndarray,
    label: int,
) -> Data:
    """Create a PyTorch Geometric `Data` object from adjacency and node features.

    Args:
        adj_matrix: Numpy array of shape (N, N) representing the adjacency
            matrix of the brain network.
        node_features: Numpy array of shape (N, F) representing node features.
        label: Integer class label (0 for AD, 1 for NC).

    Returns:
        A `torch_geometric.data.Data` instance containing:
        - x: Node features (N, F)
        - edge_index: COO indices of edges (2, E)
        - edge_attr: Edge weights (E,)
        - y: Graph label tensor of shape (1,)

    Raises:
        RuntimeError: If graph construction fails.
    """
    try:
        edge_index, edge_attr = adjacency_to_edge_index(adj_matrix)

        x = torch.from_numpy(node_features.astype(np.float32))
        y = torch.tensor([int(label)], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        return data
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError("Failed to build PyG Data object from inputs.") from exc

