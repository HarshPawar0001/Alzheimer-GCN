"""Plot a normalized confusion matrix heatmap from saved data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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


def main(config_path: str) -> None:
    """Load a confusion matrix and plot a normalized heatmap."""
    config = load_config(Path(config_path))
    paths_cfg = config.get("paths", {})

    plot_dir = Path(paths_cfg.get("plot_dir", "results/plots/"))
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Expect confusion matrix values saved as a NumPy array.
    cm_npy_path = plot_dir / "confusion_matrix.npy"
    if not cm_npy_path.is_file():
        raise RuntimeError(
            f"Confusion matrix file '{cm_npy_path}' not found. "
            "Run evaluation first to generate it."
        )

    try:
        cm = np.load(cm_npy_path)
    except OSError as exc:
        raise RuntimeError(f"Failed to load confusion matrix from '{cm_npy_path}'.") from exc

    if cm.shape != (2, 2):
        raise ValueError(
            f"Expected a 2x2 confusion matrix for AD/NC, got shape {cm.shape}."
        )

    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    labels = ["AD", "NC"]

    plt.figure(figsize=(4, 4))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix")
    out_path = plot_dir / "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    print(f"Saved confusion matrix heatmap to '{out_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot a normalized confusion matrix heatmap."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()
    main(args.config)

