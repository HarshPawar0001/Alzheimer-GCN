"""Plot ROC curve with AUC value from saved evaluation data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.metrics import roc_auc_score, roc_curve


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
    """Load saved ROC data and generate ROC curve plot."""
    config = load_config(Path(config_path))
    paths_cfg = config.get("paths", {})

    plot_dir = Path(paths_cfg.get("plot_dir", "results/plots/"))
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Expect saved true labels and probabilities in an .npz archive.
    roc_data_path = plot_dir / "roc_data.npz"
    if not roc_data_path.is_file():
        raise RuntimeError(
            f"ROC data file '{roc_data_path}' not found. "
            "Run evaluation first to generate it."
        )

    try:
        data = np.load(roc_data_path)
        y_true = data["y_true"]
        y_prob = data["y_prob"]
    except Exception as exc:
        raise RuntimeError(f"Failed to load ROC data from '{roc_data_path}'.") from exc

    if y_true.ndim != 1 or y_prob.ndim != 1 or y_true.shape[0] != y_prob.shape[0]:
        raise ValueError("ROC data arrays must be 1D and of equal length.")

    if len(np.unique(y_true)) < 2:
        raise RuntimeError("ROC curve requires at least two classes in y_true.")

    auc = float(roc_auc_score(y_true, y_prob))
    fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=1)

    plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    out_path = plot_dir / "roc_curve.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    print(f"Saved ROC curve plot to '{out_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot ROC curve from saved evaluation data."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()
    main(args.config)

