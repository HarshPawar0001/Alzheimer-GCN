import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def main():
    # Path setup
    networks_dir = Path("data/processed/networks")
    labels_csv = networks_dir / "labels.csv"

    # Check labels file
    if not labels_csv.exists():
        raise RuntimeError(f"labels.csv not found at {labels_csv}")

    df = pd.read_csv(labels_csv)

    print(f"[INFO] Loaded {len(df)} labels")

    X = []
    y = []

    print("[INFO] Matching networks with labels...")

    # Match labels with actual .npy files
    for _, row in df.iterrows():
        subject_id = str(row["SubjectID"])
        label = int(row["Label"])

        file_path = networks_dir / f"{subject_id}.npy"

        if not file_path.exists():
            continue  # skip missing files

        try:
            data = np.load(file_path)
            X.append(data)
            y.append(label)
        except Exception as e:
            print(f"[WARN] Error loading {file_path}: {e}")

    print(f"[INFO] Total matched samples: {len(X)}")

    if len(X) == 0:
        raise RuntimeError("No matching samples found. Check file names and labels.")

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"[INFO] Train samples: {len(X_train)}")
    print(f"[INFO] Test samples: {len(X_test)}")

    # Save dataset
    output_dir = Path("data/processed/dataset")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "y_test.npy", y_test)

    print("[OK] Dataset prepared successfully!")


if __name__ == "__main__":
    main()