"""PyTorch Dataset for multimodal Alzheimer’s disease classification."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Data
import yaml

from .graph_utils import adjacency_to_edge_index as _adjacency_to_edge_index_core


def load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def adjacency_to_edge_index(adj_matrix: np.ndarray) -> Tuple[Tensor, Tensor]:
    return _adjacency_to_edge_index_core(adj_matrix)


class AlzheimerDataset(Dataset):

    def __init__(
        self,
        subject_ids: Optional[List[str]],
        config: Dict,
        split: str = "train",
    ) -> None:
        super().__init__()

        dataset_cfg = config.get("dataset", {})

        clinical_csv = Path(dataset_cfg.get("clinical_csv", ""))
        networks_dir = Path(dataset_cfg.get("networks_dir", ""))

        self.networks_dir = networks_dir
        self.clinical_dir = clinical_csv.parent

        # ---- SUBJECT IDS ----
        if subject_ids is None:
            raise RuntimeError("KFold should always pass subject_ids")

        self.subject_ids = [str(sid) for sid in subject_ids]

        # ---- LABELS ----
        import pandas as pd

        labels_df = pd.read_csv(self.networks_dir / "labels.csv")

        # 🔥 SAFE label encoding (correct order)
        label_encoding = {"CN": 0, "MCI": 1, "AD": 2}

        label_map: Dict[str, int] = {}

        for _, row in labels_df.iterrows():
            sid = str(row["SubjectID"])
            label = str(row["Label"])

            if label not in label_encoding:
                continue

            label_map[sid] = label_encoding[label]

        # ---- CLINICAL ----
        clinical_features = np.load(
            self.clinical_dir / "clinical_features.npy", allow_pickle=False
        )
        clinical_ids = np.load(
            self.clinical_dir / "clinical_subject_ids.npy", allow_pickle=True
        ).astype(str)

        clinical_index = {sid: i for i, sid in enumerate(clinical_ids)}

        self.graphs: List[Data] = []
        self.clinical_tensors: List[Tensor] = []
        self.labels: List[int] = []

        # ---- LOAD DATA ----
        for sid in self.subject_ids:

            # skip invalid
            if sid not in label_map:
                continue

            if sid not in clinical_index:
                continue

            network_path = self.networks_dir / f"{sid}.npy"

            if not network_path.exists():
                continue

            adj = np.load(network_path, allow_pickle=False)

            # 🔥 identity node features
            num_nodes = adj.shape[0]
            node_features = np.eye(num_nodes)

            edge_index, edge_attr = adjacency_to_edge_index(adj)

            label_val = label_map[sid]

            # 🔥 FINAL SAFETY CHECK (this fixes your crash)
            if label_val not in [0, 1, 2]:
                print(f"[SKIP BAD LABEL] {sid} -> {label_val}")
                continue

            graph = Data(
                x=torch.tensor(node_features, dtype=torch.float32),
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor([label_val], dtype=torch.long),
            )

            clinical_tensor = torch.tensor(
                clinical_features[clinical_index[sid]], dtype=torch.float32
            )

            self.graphs.append(graph)
            self.clinical_tensors.append(clinical_tensor)
            self.labels.append(label_val)

        if len(self.graphs) == 0:
            raise RuntimeError("No valid subjects loaded — check dataset")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return (
            self.graphs[idx],
            self.clinical_tensors[idx],
            torch.tensor(self.labels[idx], dtype=torch.long),
        )