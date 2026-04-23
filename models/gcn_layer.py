from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
from torch_geometric.utils import add_self_loops, degree


class GCNLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.linear: nn.Linear = nn.Linear(in_features, out_features, bias=False)
        self.batch_norm: nn.BatchNorm1d = nn.BatchNorm1d(out_features)
        self.relu: nn.ReLU = nn.ReLU(inplace=True)
        self.dropout: nn.Dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:

        # Linear transform
        h = self.linear(x)

        num_nodes = h.size(0)

        # ✅ FIXED: positional argument instead of keyword
        edge_index_hat, edge_weight_hat = add_self_loops(
            edge_index,
            edge_weight,
            fill_value=1.0,
            num_nodes=num_nodes,
        )

        if edge_weight_hat is None:
            edge_weight_hat = torch.ones(
                edge_index_hat.size(1), device=h.device, dtype=h.dtype
            )

        # Degree
        row = edge_index_hat[0]
        deg = degree(row, num_nodes=num_nodes, dtype=h.dtype)

        # Normalize
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

        norm_weight = deg_inv_sqrt[row] * edge_weight_hat * deg_inv_sqrt[
            edge_index_hat[1]
        ]

        # Message passing
        out = torch.zeros_like(h)
        src, dst = edge_index_hat
        out.index_add_(0, dst, h[src] * norm_weight.unsqueeze(-1))

        # Activation + BN + Dropout
        out = self.relu(out)
        out = self.batch_norm(out)
        out = self.dropout(out)

        return out
