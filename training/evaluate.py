
"""Evaluation utilities for the multimodal Alzheimer’s classification model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm
import yaml
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from models.multimodal_model import MultiModalADModel
from utils.dataset import AlzheimerDataset


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    plot_dir: Path,
) -> Dict[str, float]:
    """Evaluate a trained model on the test set and save plots.

    Args:
        model: Trained multimodal model.
        test_loader: DataLoader for the test set.
        device: Torch device.
        plot_dir: Directory where evaluation plots will be saved.

    Returns:
        Dictionary of computed metrics: accuracy, sensitivity, specificity,
        precision, f1, auc.
    """
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[float] = []

    try:
        with torch.no_grad():
            for graph_data, clinical_tensor, labels in tqdm(
                test_loader, desc="Test", unit="batch"
            ):
                graph_data = graph_data.to(device)
                clinical_tensor = clinical_tensor.to(device)
                labels = labels.to(device)

                logits: Tensor = model(graph_data, clinical_tensor)
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = torch.argmax(logits, dim=1)

                all_labels.extend(labels.detach().cpu().tolist())
                all_preds.extend(preds.detach().cpu().tolist())
                all_probs.extend(probs.detach().cpu().tolist())
    except Exception as exc:
        raise RuntimeError("Error during model evaluation.") from exc

    y_true = np.array(all_labels, dtype=int)
    y_pred = np.array(all_preds, dtype=int)
    y_prob = np.array(all_probs, dtype=float)

    acc = float(accuracy_score(y_true, y_pred))
    # Sensitivity: recall for AD class (0)
    sens = float(recall_score(y_true, y_pred, pos_label=0))
    # Specificity: recall for NC class (1)
    spec = float(recall_score(y_true, y_pred, pos_label=1))
    prec = float(precision_score(y_true, y_pred, pos_label=1))
    f1 = float(f1_score(y_true, y_pred, pos_label=1))

    if len(np.unique(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = 0.0
    else:
        auc = 0.0

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # Print results in a simple table format.
    print("=== Test Metrics ===")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Sensitivity : {sens:.4f} (Recall for AD)")
    print(f"Specificity : {spec:.4f} (Recall for NC)")
    print(f"Precision   : {prec:.4f} (Positive class = NC)")
    print(f"F1-Score    : {f1:.4f}")
    print(f"AUC-ROC     : {auc:.4f}")
    print("Confusion Matrix (rows=true, cols=pred; [AD, NC]):")
    print(cm)

    # Save confusion matrix heatmap.
    plot_dir.mkdir(parents=True, exist_ok=True)
    labels_str = ["AD", "NC"]

    plt.figure(figsize=(4, 4))
    sns.heatmap(
        cm.astype(float) / cm.sum(axis=1, keepdims=True),
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels_str,
        yticklabels=labels_str,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix")
    cm_path = plot_dir / "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    # Save ROC curve.
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=1)
        plt.figure(figsize=(4, 4))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], "k--", label="Chance")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        roc_path = plot_dir / "roc_curve.png"
        plt.tight_layout()
        plt.savefig(roc_path)
        plt.close()

    return {
        "accuracy": acc,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1": f1,
        "auc": auc,
    }


def main(config_path: str) -> None:
    """Load the best checkpoint and evaluate on the test set.

    Args:
        config_path: Path to the YAML configuration file.
    """
    try:
        config = load_config(Path(config_path))
        training_cfg = config.get("training", {})
        paths_cfg = config.get("paths", {})

        seed: int = int(training_cfg.get("seed", 42))
        set_seed(seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint_dir = Path(paths_cfg.get("checkpoint_dir", "results/checkpoints/"))
        plot_dir = Path(paths_cfg.get("plot_dir", "results/plots/"))

        best_ckpt_path = checkpoint_dir / "best_model.pth"
        if not best_ckpt_path.is_file():
            raise RuntimeError(
                f"Best checkpoint not found at '{best_ckpt_path}'. "
                "Ensure training has been run first."
            )

        # Build model and load weights.
        model = MultiModalADModel(config).to(device)
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Build test dataset/loader.
        test_dataset = AlzheimerDataset(
            subject_ids=None,
            config=config,
            split="test",
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=8,
            shuffle=False,
            collate_fn=collate_fn,
        )

        metrics = evaluate(model, test_loader, device, plot_dir)

        print("=== Done: evaluate ===")
    except Exception as exc:
        raise RuntimeError("Error in main() evaluation function.") from exc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the multimodal AD classification model on the test set."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()
    main(args.config)

