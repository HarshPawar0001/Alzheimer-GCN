"""Self-attention pooling layer implementing Equations (2) and (3) from the paper.

Equations:
    Z = D^{-1/2} (A + I) D^{-1/2} X θ_att                (2)
    idx = top_rank(Z, ⌊k * N⌋)                           (3)

This module defines a pooling layer that:
  - Computes self-attention scores for each node using normalized adjacency.
  - Ranks nodes within each graph in the batch by their attention score.
  - Keeps the top k * N nodes (per graph) and discards the rest.
  - Returns a new graph with updated node features, adjacency, and batch vector.

The retained node features are scaled by a sigmoid of their attention scores,
following the description in the project specification.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch_geometric.utils import add_self_loops, degree, subgraph


class SelfAttentionPool(nn.Module):
    """Self-attention graph pooling with a learnable attention vector.

    The layer follows Equations (2) and (3) from the paper, operating on
    batched graphs represented by `edge_index` and `batch`.
    """

    def __init__(self, in_features: int, k: float = 0.8) -> None:
        """Initialize the SelfAttentionPool layer.

        Args:
            in_features: Dimensionality of input node features.
            k: Pooling ratio in (0, 1], fraction of nodes to retain per graph.
        """
        super().__init__()
        if not (0.0 < k <= 1.0):
            raise ValueError(f"Pooling ratio k must be in (0, 1], got {k}.")

        self.k: float = k
        # θ_att: learnable attention parameter vector.
        self.theta_att: nn.Parameter = nn.Parameter(
            torch.empty(in_features, 1, dtype=torch.float32)
        )
        nn.init.xavier_uniform_(self.theta_att)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Apply self-attention pooling to a batch of graphs.

        Args:
            x: Node feature matrix X of shape (N, F).
            edge_index: Edge indices of shape (2, E) in COO format.
            batch: Batch vector of shape (N,) mapping each node to a graph ID.
            edge_weight: Optional edge weights of shape (E,). If `None`,
                all edges are assigned weight 1.

        Returns:
            A tuple `(x_new, edge_index_new, batch_new)` where:
                - `x_new` has shape (N_new, F),
                - `edge_index_new` has shape (2, E_new),
                - `batch_new` has shape (N_new,).
        """
        device = x.device
        num_nodes: int = x.size(0)

        # Step 1: Add self-loops: A_hat = A + I.
        edge_index_hat, edge_weight_hat = add_self_loops(
            edge_index,
            edge_weight,
            fill_value=1.0,
            num_nodes=x.size(0),
        )
        if edge_weight_hat is None:
            edge_weight_hat = torch.ones(
                edge_index_hat.size(1), device=device, dtype=x.dtype
            )

        # Step 2: Compute normalized adjacency weights D^{-1/2} (A+I) D^{-1/2}.
        row, col = edge_index_hat
        deg = degree(row, num_nodes=num_nodes, dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        norm_weight = deg_inv_sqrt[row] * edge_weight_hat * deg_inv_sqrt[col]

        # Step 3: Aggregate features using normalized adjacency:
        #         Z_raw = (D^{-1/2} (A+I) D^{-1/2} X) θ_att
        agg = torch.zeros_like(x)
        src, dst = row, col
        agg.index_add_(0, dst, x[src] * norm_weight.unsqueeze(-1))
        z_raw = agg @ self.theta_att  # (N, 1)
        z = z_raw.squeeze(-1)  # (N,)

        # Step 4: For each graph in the batch, keep top ⌊k * N_g⌋ nodes.
        unique_batches = batch.unique(sorted=True)
        keep_indices: List[Tensor] = []

        for b in unique_batches:
            mask = batch == b
            idx_in_graph = mask.nonzero(as_tuple=False).view(-1)
            if idx_in_graph.numel() == 0:
                continue

            scores_graph = z[idx_in_graph]
            num_nodes_g = scores_graph.size(0)
            k_num = max(1, int(self.k * float(num_nodes_g)))

            topk_vals, topk_pos = torch.topk(
                scores_graph, k_num, largest=True, sorted=True
            )
            selected_global = idx_in_graph[topk_pos]
            keep_indices.append(selected_global)

        if not keep_indices:
            raise RuntimeError(
                "SelfAttentionPool resulted in no nodes being selected; "
                "check pooling ratio k and input graph sizes."
            )

        keep_idx = torch.cat(keep_indices, dim=0)
        keep_idx, _ = torch.sort(keep_idx)

        # Step 5: Induce the pooled subgraph.
        x_new = x[keep_idx]
        batch_new = batch[keep_idx]

        # Scale retained features by sigmoid of their attention scores.
        z_kept = z[keep_idx]
        scale = torch.sigmoid(z_kept).unsqueeze(-1)
        x_new = x_new * scale

        # Use PyG subgraph utility to obtain induced edges; reindex nodes.
        edge_index_new, _ = subgraph(
            keep_idx, edge_index, relabel_nodes=True, num_nodes=num_nodes
        )

        return x_new, edge_index_new, batch_new

