"""Main entry point for the AlzheimerGCN project.

Available modes:
    --mode preprocess  : Run all preprocessing steps in order.
    --mode train       : Train the multimodal model.
    --mode evaluate    : Evaluate the best checkpoint on the test set.
    --mode visualize   : Generate all visualization plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

from preprocessing.build_brain_networks import main as build_brain_networks_main
from preprocessing.extract_node_features import main as extract_node_features_main
from preprocessing.prepare_clinical import main as prepare_clinical_main
from preprocessing.prepare_dataset import main as prepare_dataset_main
from training.train import train as train_main
from training.evaluate import main as evaluate_main
from visualization.plot_training import main as plot_training_main
from visualization.plot_confusion import main as plot_confusion_main
from visualization.plot_roc import main as plot_roc_main
from visualization.plot_attention import main as plot_attention_main


def run_preprocess(config_path: str) -> None:
    """Run the full preprocessing pipeline in order."""
    print("=== Starting: preprocess ===")
    build_brain_networks_main(config_path)
    extract_node_features_main(config_path)
    prepare_clinical_main(config_path)
    prepare_dataset_main(config_path)
    print("=== Done: preprocess ===")


def run_train(config_path: str) -> None:
    """Run the training procedure."""
    print("=== Starting: train ===")
    train_main(config_path)
    print("=== Done: train ===")


def run_evaluate(config_path: str) -> None:
    """Run evaluation on the test set using the best checkpoint."""
    print("=== Starting: evaluate ===")
    evaluate_main(config_path)
    print("=== Done: evaluate ===")


def run_visualize(config_path: str) -> None:
    """Run all visualization scripts to generate plots."""
    print("=== Starting: visualize ===")
    plot_training_main(config_path)
    # The evaluation script already saves confusion matrix and ROC plots.
    # The standalone scripts can be used if separate generation is needed and
    # supporting data files are present.
    try:
        plot_confusion_main(config_path)
    except Exception as exc:
        print(f"Skipping standalone confusion matrix plot: {exc}")
    try:
        plot_roc_main(config_path)
    except Exception as exc:
        print(f"Skipping standalone ROC plot: {exc}")
    plot_attention_main(config_path)
    print("=== Done: visualize ===")


def main() -> None:
    """Parse CLI arguments and dispatch to the selected mode."""
    parser = argparse.ArgumentParser(
        description="AlzheimerGCN project entry point."
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["preprocess", "train", "evaluate", "visualize"],
        help="Execution mode.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the YAML configuration file.",
    )

    args = parser.parse_args()
    mode: Literal["preprocess", "train", "evaluate", "visualize"] = args.mode  # type: ignore[assignment]
    config_path: str = args.config

    if mode == "preprocess":
        run_preprocess(config_path)
    elif mode == "train":
        run_train(config_path)
    elif mode == "evaluate":
        run_evaluate(config_path)
    elif mode == "visualize":
        run_visualize(config_path)
    else:
        raise ValueError(f"Unsupported mode '{mode}'.")


if __name__ == "__main__":
    main()

