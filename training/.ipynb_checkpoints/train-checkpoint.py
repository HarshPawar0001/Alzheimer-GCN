from __future__ import annotations

import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from utils.dataset import AlzheimerDataset
from models.multimodal_model import MultiModalADModel
import yaml


def train(config_path: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load config ----
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("=== Starting: train ===")

    # ---- Dataset ----
    dataset = AlzheimerDataset(None, config)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=8)

    # ---- Model ----
    model = MultiModalADModel(config["model"]).to(device)

    optimizer = Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    epochs = config["training"]["epochs"]

    print("=== Training Started ===")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for graph_batch, clinical_batch, labels in train_loader:
            graph_batch = graph_batch.to(device)
            clinical_batch = clinical_batch.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(graph_batch, clinical_batch)

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # ---- Eval ----
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for graph_batch, clinical_batch, labels in test_loader:
                graph_batch = graph_batch.to(device)
                clinical_batch = clinical_batch.to(device)
                labels = labels.to(device)

                outputs = model(graph_batch, clinical_batch)

                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total

        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Test Acc: {acc:.4f}")

    print("=== Training Complete ===")