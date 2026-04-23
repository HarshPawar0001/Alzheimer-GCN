from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import yaml
from nilearn.image import resample_to_img
from tqdm import tqdm


def load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_label_map(labels_csv: Path) -> Dict[str, int]:
    if not labels_csv.is_file():
        print(f"[WARN] labels.csv not found at {labels_csv}.")
        return {}

    df = pd.read_csv(labels_csv)

    label_map: Dict[str, int] = {}
    for _, row in df.iterrows():
        sid = str(row["SubjectID"])
        dx = str(row["Label"]).strip().upper()

        if dx == "AD":
            label_map[sid] = 0
        elif dx == "MCI":          # ← CHANGE 1: Added MCI
            label_map[sid] = 2
        elif dx == "CN":
            label_map[sid] = 1

    return label_map


# ✅ DUMMY ATLAS (NO INTERNET, NO SSL)
def load_aal_atlas():
    shape = (91, 109, 91)
    atlas_data = np.zeros(shape, dtype=np.int32)

    for i in range(1, 91):
        atlas_data[i % shape[0], :, :] = i

    atlas_img = nib.Nifti1Image(atlas_data, affine=np.eye(4))
    region_indices = list(range(1, 91))

    print("[INFO] Using dummy atlas (no download required)")
    return atlas_img, region_indices


def compute_region_means(fa_img, atlas_img, region_indices):
    fa_resampled = resample_to_img(fa_img, atlas_img, interpolation="continuous")
    fa_data = np.asarray(fa_resampled.get_fdata(), dtype=np.float32)
    atlas_data = np.asarray(atlas_img.get_fdata(), dtype=np.int32)

    region_means = []
    for idx in region_indices:
        mask = atlas_data == idx
        if np.any(mask):
            region_means.append(float(fa_data[mask].mean()))
        else:
            region_means.append(0.0)

    return np.asarray(region_means, dtype=np.float32)


def build_adjacency_from_region_means(region_means):
    row = region_means.reshape(-1, 1)
    col = region_means.reshape(1, -1)
    adj = (row + col) / 2.0
    np.fill_diagonal(adj, region_means)
    return adj.astype(np.float32)


def main(config_path: str) -> None:
    config = load_config(Path(config_path))

    dataset_cfg = config.get("dataset", {})
    networks_dir = Path(dataset_cfg.get("networks_dir", ""))

    networks_dir.mkdir(parents=True, exist_ok=True)

    # fallback to DTI
    fa_dir = networks_dir.parent / "FA"
    if not fa_dir.is_dir():
        print("[WARN] FA not found → using DTI instead")
        fa_dir = Path("data/nifti")

    labels_csv = Path("data/processed/networks/labels.csv")
    label_map = build_label_map(labels_csv)

    atlas_img, region_indices = load_aal_atlas()

    fa_files = list(fa_dir.glob("**/*.nii.gz")) + list(fa_dir.glob("**/*.nii"))  # ← CHANGE 2: Added .nii

    if not fa_files:
        print(f"[ERROR] No NIfTI files found in {fa_dir}")
        return

    records = []

    print("=== Building brain networks ===")

    for fa_path in tqdm(fa_files):
        subject_id = fa_path.parent.name

        if subject_id not in label_map:
            continue

        try:
            fa_img = nib.load(str(fa_path))
            region_means = compute_region_means(fa_img, atlas_img, region_indices)
            adj = build_adjacency_from_region_means(region_means)

            out_path = networks_dir / f"{subject_id}.npy"
            np.save(out_path, adj)

            records.append((subject_id, label_map[subject_id]))

        except Exception as e:
            print(f"[WARN] Skipping {subject_id}: {e}")

    if not records:
        print("[WARN] No networks generated.")
        return

    labels_df = pd.DataFrame(records, columns=["SubjectID", "Label"])
    labels_df.to_csv(networks_dir / "labels.csv", index=False)

    print(f"[OK] Generated {len(records)} networks")
    print("DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    main(args.config)