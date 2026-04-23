import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader

from utils.dataset import AlzheimerDataset
from models.multimodal_model import MultiModalADModel


# ---- Load config ----
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# ---- Seed ----
torch.manual_seed(config["training"]["seed"])
np.random.seed(config["training"]["seed"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ---- Load labels ----
labels_df = pd.read_csv("data/processed/networks/labels.csv")

label_map = {"AD": 0, "CN": 1, "MCI": 2}
labels_df["LabelNum"] = labels_df["Label"].apply(
    lambda x: int(x) if str(x).isdigit() else label_map[x]
)

subjects = labels_df["SubjectID"].astype(str).values
labels_arr = labels_df["LabelNum"].values


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_accs = []


# ---- K-Fold loop ----
for fold, (train_idx, test_idx) in enumerate(skf.split(subjects, labels_arr)):

    print(f"\n{'='*40}")
    print(f"FOLD {fold+1}/5")

    train_ids = subjects[train_idx].tolist()
    test_ids  = subjects[test_idx].tolist()

    train_ds = AlzheimerDataset(train_ids, config)
    test_ds  = AlzheimerDataset(test_ids, config)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=8, shuffle=False)

    model = MultiModalADModel(config["model"]).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )

    criterion = nn.CrossEntropyLoss()

    best_acc = 0

    # ---- Training loop ----
    for epoch in range(config["training"]["epochs"]):

        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            batch = [b.to(device) if hasattr(b, 'to') else b for b in batch]

            optimizer.zero_grad()
            outputs = model(*batch[:-1])
            labels = batch[-1]

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss /= len(train_loader)
        train_acc = correct / total

        # ---- Validation ----
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_loader:
                batch = [b.to(device) if hasattr(b, 'to') else b for b in batch]

                outputs = model(*batch[:-1])
                labels = batch[-1]

                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(test_loader)
        val_acc = correct / total

        # ---- Print logs ----
        print(f"Epoch {epoch+1}/{config['training']['epochs']}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc

    print(f"Best Accuracy Fold {fold+1}: {best_acc:.4f}")
    fold_accs.append(best_acc)


# ---- Final result ----
print("\nFINAL RESULTS")
print(f"Mean Accuracy: {np.mean(fold_accs):.4f}")
print(f"Std Dev: {np.std(fold_accs):.4f}")
