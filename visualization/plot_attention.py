"""Visualize top brain regions by self-attention score."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm

from models.multimodal_model import MultiModalADModel
from utils.aal_labels import AAL_REGIONS
from utils.dataset import AlzheimerDataset


def compute_region_attention(
    model: MultiModalADModel,
    dataset: AlzheimerDataset,
    device: torch.device,
    max_subjects: int = 16,
) -> np.ndarray:
    """Compute mean attention score per AAL region over a subset of subjects.

    Args:
        model: Trained multimodal model.
        dataset: AlzheimerDataset providing graph data.
        device: Torch device.
        max_subjects: Maximum number of subjects to use for averaging.

    Returns:
        Numpy array of shape (90,) with mean attention scores per region.
    """
    model.eval()

    num_regions = len(AAL_REGIONS)
    scores_sum = np.zeros(num_regions, dtype=np.float64)
    count = 0

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
    )

    with torch.no_grad():
        for batch_idx, (graph_data, clinical_tensor, labels) in enumerate(
            tqdm(loader, desc="Attention", unit="batch")
        ):
            graph_data = graph_data.to(device)
            # clinical_tensor and labels are unused for attention computation.

            scores_t = model.get_attention_scores(graph_data).detach().cpu()
            scores_np = scores_t.numpy()

            # Each graph is assumed to have exactly 90 nodes corresponding to AAL regions.
            batch_obj: Batch = graph_data  # type: ignore[assignment]
            batch_vec = batch_obj.batch.detach().cpu().numpy()
            num_graphs = int(batch_vec.max() + 1) if batch_vec.size > 0 else 0

            for g in range(num_graphs):
                idx = np.where(batch_vec == g)[0]
                if idx.size != num_regions:
                    # Skip graphs that do not have exactly 90 nodes.
                    continue
                region_scores = scores_np[idx]
                scores_sum += region_scores
                count += 1
                if count >= max_subjects:
                    break

            if count >= max_subjects:
                break

    if count == 0:
        raise RuntimeError(
            "No graphs with exactly 90 nodes were found for attention analysis."
        )

    return scores_sum / float(count)


def main(config_path: str) -> None:
    """Generate a bar chart of top brain regions ranked by attention score."""
    config = load_config(Path(config_path))
    training_cfg = config.get("training", {})
    paths_cfg = config.get("paths", {})

    seed: int = int(training_cfg.get("seed", 42))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = Path(paths_cfg.get("checkpoint_dir", "results/checkpoints/"))
    plot_dir = Path(paths_cfg.get("plot_dir", "results/plots/"))
    plot_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt_path = checkpoint_dir / "best_model.pth"
    if not best_ckpt_path.is_file():
        raise RuntimeError(
            f"Best checkpoint not found at '{best_ckpt_path}'. "
            "Run training before plotting attention."
        )

    # Load model and weights.
    model = MultiModalADModel(config).to(device)
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Use the test split for attention visualization.
    dataset = AlzheimerDataset(subject_ids=None, config=config, split="test")

    mean_scores = compute_region_attention(model, dataset, device=device)

    # Get top 15 regions.
    top_k = 15
    indices = np.argsort(mean_scores)[-top_k:][::-1]
    top_regions: List[str] = [AAL_REGIONS[i] for i in indices]
    top_values = mean_scores[indices]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(range(top_k), top_values, tick_label=top_regions)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Attention Score")
    plt.title("Top Brain Regions by Self-Attention Score")
    for bar, val in zip(bars, top_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.tight_layout()

    out_path = plot_dir / "brain_attention.png"
    plt.savefig(out_path)
    plt.close()

    print(f"Saved brain attention bar plot to '{out_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot top brain regions by self-attention score."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()
    main(args.config)

