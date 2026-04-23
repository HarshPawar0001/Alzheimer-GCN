"""DTI branch: three-layer GCN with self-attention pooling (Figure 2).

This module implements the DTI branch described in the paper:
  - Three stacked modules of:
        GCNLayer -> SelfAttentionPool(k=0.8) -> global_add_pool (readout)
  - The readouts from each module are summed and passed through a linear
    layer with ReLU to obtain a 32-dimensional DTI embedding.

It also exposes a `get_attention_scores` method that computes self-attention
scores for all nodes in a given batched graph, suitable for later
visualization of important brain regions.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import add_self_loops, degree

from .gcn_layer import GCNLayer
from .self_attention_pool import SelfAttentionPool


class DTIBranch(nn.Module):
    """Three-layer GCN with self-attention pooling for DTI brain networks."""

    def __init__(self, config: Dict) -> None:
        """Initialize the DTIBranch from configuration.

        Args:
            config: Parsed configuration dictionary (from `config.yaml`).
                Expected keys under `model`:
                    - num_nodes
                    - node_feature_dim
                    - gcn_layer1_out
                    - gcn_layer2_out
                    - gcn_layer3_out
                    - pooling_ratio_k
                    - dti_embedding_dim
        """
        super().__init__()
        model_cfg = config.get("model", {})

        in_dim: int = int(model_cfg.get("node_feature_dim", 90))
        gcn1_out: int = int(model_cfg.get("gcn_layer1_out", 64))
        gcn2_out: int = int(model_cfg.get("gcn_layer2_out", 32))
        gcn3_out: int = int(model_cfg.get("gcn_layer3_out", 16))
        pooling_ratio: float = float(model_cfg.get("pooling_ratio_k", 0.8))
        dti_emb_dim: int = int(model_cfg.get("dti_embedding_dim", 32))

        # Graph convolutional layers (no extra dropout beyond this module).
        self.gcn1: GCNLayer = GCNLayer(in_dim, gcn1_out, dropout=0.0)
        self.gcn2: GCNLayer = GCNLayer(gcn1_out, gcn2_out, dropout=0.0)
        self.gcn3: GCNLayer = GCNLayer(gcn2_out, gcn3_out, dropout=0.0)

        # Self-attention pooling layers.
        self.pool1: SelfAttentionPool = SelfAttentionPool(gcn1_out, k=pooling_ratio)
        self.pool2: SelfAttentionPool = SelfAttentionPool(gcn2_out, k=pooling_ratio)
        self.pool3: SelfAttentionPool = SelfAttentionPool(gcn3_out, k=pooling_ratio)

        # Final readout aggregation and projection to DTI embedding.
        # Each readout vector has size equal to the GCN output dimension at
        # that layer; we sum three readouts of potentially different sizes by
        # first projecting each to the same dimension (gcn3_out), then summing.
        self.readout_proj1: nn.Linear = nn.Linear(gcn1_out, gcn3_out)
        self.readout_proj2: nn.Linear = nn.Linear(gcn2_out, gcn3_out)
        self.readout_proj3: nn.Linear = nn.Linear(gcn3_out, gcn3_out)

        self.fc_out: nn.Linear = nn.Linear(gcn3_out, dti_emb_dim)
        self.relu: nn.ReLU = nn.ReLU(inplace=True)

    def forward(self, graph_data: Data) -> Tensor:
        """Forward pass through the DTI branch.

        Args:
            graph_data: Batched PyG `Data` or `Batch` object containing:
                - x: Node features (N, F).
                - edge_index: Edge indices (2, E).
                - edge_attr: Optional edge weights (E,).
                - batch: Batch vector (N,) indicating graph membership.

        Returns:
            Tensor of shape (batch_size, dti_embedding_dim) representing
            the DTI graph embedding for each subject.
        """
        x: Tensor = graph_data.x
        edge_index: Tensor = graph_data.edge_index
        batch: Tensor = getattr(
            graph_data, "batch", torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        )
        edge_weight: Optional[Tensor] = getattr(graph_data, "edge_attr", None)

        # Module 1: GCN -> SelfAttentionPool -> global_add_pool
        x1 = self.gcn1(x, edge_index, edge_weight)
        x1_pooled, edge_index1, batch1 = self.pool1(x1, edge_index, batch, edge_weight)
        readout1 = global_add_pool(x1_pooled, batch1)  # (B, gcn1_out)

        # Module 2: GCN -> SelfAttentionPool -> global_add_pool
        x2 = self.gcn2(x1_pooled, edge_index1, None)
        x2_pooled, edge_index2, batch2 = self.pool2(x2, edge_index1, batch1, None)
        readout2 = global_add_pool(x2_pooled, batch2)  # (B, gcn2_out)

        # Module 3: GCN -> SelfAttentionPool -> global_add_pool
        x3 = self.gcn3(x2_pooled, edge_index2, None)
        x3_pooled, edge_index3, batch3 = self.pool3(x3, edge_index2, batch2, None)
        readout3 = global_add_pool(x3_pooled, batch3)  # (B, gcn3_out)

        # Project each readout to a common dimension and aggregate.
        r1 = self.readout_proj1(readout1)
        r2 = self.readout_proj2(readout2)
        r3 = self.readout_proj3(readout3)

        aggregated = r1 + r2 + r3
        out = self.fc_out(self.relu(aggregated))
        return out

    def get_attention_scores(self, graph_data: Data) -> Tensor:
        """Compute self-attention scores for all nodes in the input graphs.

        This uses the parameters of the first self-attention pooling layer
        (`pool1`) and follows Equation (2):
            Z = D^{-1/2} (A + I) D^{-1/2} X θ_att

        Args:
            graph_data: Batched PyG `Data` or `Batch` object.

        Returns:
            Tensor `Z` of shape (N,) containing attention scores for each node
            across the batch (N total nodes in all graphs).
        """
        x: Tensor = graph_data.x
        edge_index: Tensor = graph_data.edge_index
        batch: Tensor = getattr(
            graph_data, "batch", torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        )
        edge_weight: Optional[Tensor] = getattr(graph_data, "edge_attr", None)

        device = x.device
        num_nodes: int = x.size(0)

        # Add self-loops with weight 1.0 (A + I).
        edge_index_hat, edge_weight_hat = add_self_loops(
            edge_index,
            edge_weight,
            fill_value=1.0,
            num_nodes=num_nodes,
        )

        if edge_weight_hat is None:
            edge_weight_hat = torch.ones(
                edge_index_hat.size(1), device=device, dtype=x.dtype
            )

        # Degree and symmetric normalization D^{-1/2} (A+I) D^{-1/2}.
        row, col = edge_index_hat
        deg = degree(row, num_nodes=num_nodes, dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        norm_weight = deg_inv_sqrt[row] * edge_weight_hat * deg_inv_sqrt[col]

        # Aggregate features using normalized adjacency.
        agg = torch.zeros_like(x)
        src, dst = row, col
        agg.index_add_(0, dst, x[src] * norm_weight.unsqueeze(-1))

        # Apply attention parameters θ_att from the first pooling layer.
        z_raw = agg @ self.pool1.theta_att  # (N, 1)
        z = z_raw.squeeze(-1)  # (N,)

        # The `batch` vector is returned along with scores via graph_data if needed
        # by downstream visualization utilities.
        _ = batch  # unused here but kept for API symmetry.
        return z

