"""Plot training curves (loss, accuracy, AUC) from CSV logs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
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
    """Load training log CSV and generate training curves."""
    config = load_config(Path(config_path))
    paths_cfg = config.get("paths", {})

    log_dir = Path(paths_cfg.get("log_dir", "results/logs/"))
    plot_dir = Path(paths_cfg.get("plot_dir", "results/plots/"))
    plot_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / "train_log.csv"
    if not log_path.is_file():
        raise RuntimeError(
            f"Training log CSV not found at '{log_path}'. "
            "Ensure training has been run first."
        )

    try:
        df = pd.read_csv(log_path)
    except OSError as exc:
        raise RuntimeError(f"Failed to read training log CSV at '{log_path}'.") from exc

    required_cols = {"epoch", "train_loss", "val_loss", "val_acc", "val_auc"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Training log CSV is missing columns: {', '.join(sorted(missing))}."
        )

    epochs = df["epoch"].to_numpy()

    # Plot 1: Train Loss vs Val Loss
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, df["train_loss"], label="Train Loss")
    plt.plot(epochs, df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Plot 2: Val Accuracy
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, df["val_acc"], label="Val Accuracy", color="tab:green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Plot 3: Val AUC
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, df["val_auc"], label="Val AUC", color="tab:red")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("Validation AUC-ROC")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = plot_dir / "training_curves.png"
    plt.savefig(out_path)
    plt.close("all")

    print(f"Saved training curves to '{out_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot training curves from training log CSV."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()
    main(args.config)

