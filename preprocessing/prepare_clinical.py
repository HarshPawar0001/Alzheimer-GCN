"""Prepare and encode ADNI clinical data for multimodal modeling.

This script:
1. Loads configuration from `configs/config.yaml`.
2. Reads the raw ADNI clinical CSV.
3. Keeps only the required columns:
   - SubjectID (or RID), DX_bl (or DX_BL), AGE, PTGENDER, PTEDUCAT, MMSE, CDRSB, APOE4.
4. Standardizes diagnosis labels and filters to AD/CN subjects.
5. Matches subject IDs with DTI subjects that have networks (inner join).
6. Encodes categorical variables:
   - PTGENDER: Male=0, Female=1.
   - APOE4: assumed numeric (0, 1, 2) so kept as-is.
7. Normalizes AGE, MMSE, CDRSB, PTEDUCAT using StandardScaler.
   The scaler is fit on train subjects only if a split file already exists,
   otherwise it is fit on all available subjects.
8. Saves:
   - clinical_features.npy: feature matrix in SubjectID order.
   - clinical_subject_ids.npy: ordered SubjectID array matching features.
   - clinical_scaler.pkl: fitted StandardScaler instance for reuse.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _infer_subject_id_from_nifti_path(p: Path) -> str:
    """Infer a subject ID from a NIfTI path (file or subject folder layout)."""
    # Common layouts:
    # - data/nifti/AD/{subject_id}.nii.gz
    # - data/nifti/AD/{subject_id}/DTI.nii.gz
    if p.is_dir():
        return p.name
    name = p.name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    return p.stem


def build_labels_from_nifti(nifti_root: Path, out_csv: Path) -> pd.DataFrame:
    """Create a basic labels.csv by scanning `data/nifti/AD` and `data/nifti/CN`.

    Labels:
      - AD = 1
      - CN = 0
    """
    ad_dir = nifti_root / "AD"
    cn_dir = nifti_root / "CN"

    if not ad_dir.exists() and not cn_dir.exists():
        raise RuntimeError(
            "labels.csv was missing and no NIfTI folders were found at:\n"
            f"  {ad_dir}\n"
            f"  {cn_dir}\n"
            "Create these folders or provide `data/processed/networks/labels.csv`."
        )

    def collect_subject_ids(cls_dir: Path) -> List[str]:
        if not cls_dir.exists():
            return []
        # Accept either subject folders or direct NIfTI files.
        subj_ids: List[str] = []
        for d in cls_dir.iterdir():
            if d.is_dir():
                subj_ids.append(_infer_subject_id_from_nifti_path(d))
        for ext in ("*.nii.gz", "*.nii"):
            for f in cls_dir.glob(ext):
                subj_ids.append(_infer_subject_id_from_nifti_path(f))
        # Unique + stable
        return sorted(set(s.strip() for s in subj_ids if str(s).strip()))

    ad_ids = collect_subject_ids(ad_dir)
    cn_ids = collect_subject_ids(cn_dir)

    rows: List[Dict[str, object]] = []
    rows += [{"SubjectID": sid, "Label": 1} for sid in ad_ids]
    rows += [{"SubjectID": sid, "Label": 0} for sid in cn_ids]

    if not rows:
        raise RuntimeError(
            "Found AD/CN NIfTI folders but could not infer any subject IDs.\n"
            f"AD dir: {ad_dir}\n"
            f"CN dir: {cn_dir}\n"
            "Expected either subject subfolders or .nii/.nii.gz files."
        )

    df = pd.DataFrame(rows).drop_duplicates(subset=["SubjectID"], keep="first")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


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


def load_dti_subjects(networks_dir: Path) -> pd.DataFrame:
    """Load DTI subject IDs and labels from `labels.csv`.

    Args:
        networks_dir: Directory containing the brain networks and `labels.csv`.

    Returns:
        DataFrame with at least columns: SubjectID, Label.
    """
    labels_csv = networks_dir / "labels.csv"
    if not labels_csv.is_file():
        # Fallback: build a basic labels.csv from NIfTI folder structure.
        nifti_root = PROJECT_ROOT / "data" / "nifti"
        try:
            df = build_labels_from_nifti(nifti_root=nifti_root, out_csv=labels_csv)
            print(
                f"labels.csv not found in '{networks_dir}'. "
                f"Created '{labels_csv}' from NIfTI folders (AD=1, CN=0). "
                f"Subjects: {len(df)}"
            )
            return df
        except Exception as exc:
            raise RuntimeError(
                f"Expected labels.csv in '{networks_dir}', but it was not found, "
                "and automatic label generation from NIfTI folders failed."
            ) from exc
    try:
        df = pd.read_csv(labels_csv)
    except OSError as exc:
        raise RuntimeError(f"Failed to read labels.csv at '{labels_csv}'.") from exc

    required_cols = {"SubjectID", "Label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(
            f"labels.csv is missing required columns: {', '.join(sorted(missing))}."
        )
    return df


def load_clinical_csv(clinical_csv: Path) -> pd.DataFrame:
    """Load and filter the ADNI clinical CSV.

    Args:
        clinical_csv: Path to the ADNI clinical CSV file.

    Returns:
        Filtered DataFrame containing only required columns and AD/CN rows.

        Notes:
        - Supports both "ADNIMERGE.csv"-style files with `DX_bl`
          and the generated file built from RDA tables which uses `DX_BL` and `RID`.
    """
    try:
        df = pd.read_csv(clinical_csv)
    except OSError as exc:
        raise RuntimeError(f"Failed to read clinical CSV at '{clinical_csv}'.") from exc

    # Accept either SubjectID or RID as the subject identifier.
    if "SubjectID" not in df.columns:
        if "RID" in df.columns:
            df = df.rename(columns={"RID": "SubjectID"})
        else:
            raise RuntimeError(
                "Clinical CSV must contain either 'SubjectID' or 'RID' column."
            )

    # Accept either DX_bl or DX_BL (generated file uses DX_BL).
    if "DX_bl" not in df.columns and "DX_BL" in df.columns:
        df = df.rename(columns={"DX_BL": "DX_bl"})

    required_cols = {
        "SubjectID",
        "DX_bl",
        "AGE",
        "PTGENDER",
        "PTEDUCAT",
        "MMSE",
        "CDRSB",
        "APOE4",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Clinical CSV is missing required columns: {', '.join(sorted(missing))}."
        )

    df = df[list(required_cols)]

    # Normalize diagnosis values to {AD, CN}.
    dx = df["DX_bl"].astype(str).str.strip().str.upper()
    dx_map = {
        "AD": "AD",
        "ALZHEIMER'S DISEASE": "AD",
        "ALZHEIMERS DISEASE": "AD",
        "DEMENTIA": "AD",
        "DAT": "AD",
        "CN": "CN",
        "NC": "CN",
        "NORMAL": "CN",
        "CONTROL": "CN",
    }
    df["DX_bl"] = dx.map(dx_map).fillna(dx)
    df = df[df["DX_bl"].isin(["AD", "CN"])]
    return df


def encode_and_normalize(
    df_merged: pd.DataFrame, splits_dir: Path
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Encode categorical features and normalize numerical clinical features.

    Args:
        df_merged: DataFrame after inner join between clinical and DTI subjects.
        splits_dir: Directory where train/test split files may be located.

    Returns:
        A tuple of (features, labels, scaler):
        - features: numpy array of shape (N, 6)
        - labels: numpy array of shape (N,)
        - scaler: fitted StandardScaler instance.
    """
    # Map diagnosis to label for training: CN=0, AD=1
    label_map = {"CN": 0, "AD": 1}
    labels = df_merged["DX_bl"].map(label_map).astype(int).to_numpy()

    # Encode PTGENDER: Male=0, Female=1
    gender_map = {"Male": 0, "Female": 1}
    df_merged["PTGENDER"] = df_merged["PTGENDER"].map(gender_map)

    if df_merged["PTGENDER"].isnull().any():
        raise RuntimeError("Encountered unknown values in PTGENDER after mapping.")

    # APOE4 is assumed numeric (0, 1, 2); coerce errors to NaN and drop if needed.
    df_merged["APOE4"] = pd.to_numeric(df_merged["APOE4"], errors="coerce")
    if df_merged["APOE4"].isnull().any():
        # For simplicity, drop rows with invalid APOE4.
        df_merged = df_merged.dropna(subset=["APOE4"])
        labels = df_merged["DX_bl"].map(label_map).astype(int).to_numpy()

    # Numerical columns to normalize.
    numeric_cols = ["AGE", "MMSE", "CDRSB", "PTEDUCAT"]

    # Determine which subjects to use for fitting the scaler.
    train_ids_path = splits_dir / "train_ids.txt"
    if train_ids_path.is_file():
        try:
            train_ids = (
                train_ids_path.read_text(encoding="utf-8").strip().splitlines()
            )
            train_ids = [tid.strip() for tid in train_ids if tid.strip()]
        except OSError as exc:
            raise RuntimeError(
                f"Failed to read train_ids from '{train_ids_path}'."
            ) from exc

        df_train = df_merged[df_merged["SubjectID"].astype(str).isin(train_ids)]
        if df_train.empty:
            # Fallback: fit on all data if no matches.
            fit_data = df_merged[numeric_cols].to_numpy(dtype=float)
        else:
            fit_data = df_train[numeric_cols].to_numpy(dtype=float)
    else:
        # If split not created yet, fit on all available subjects.
        fit_data = df_merged[numeric_cols].to_numpy(dtype=float)

    scaler = StandardScaler()
    scaler.fit(fit_data)

    normalized_numeric = scaler.transform(df_merged[numeric_cols].to_numpy(dtype=float))

    # Final feature matrix: [AGE, GENDER, MMSE, CDRSB, APOE4, PTEDUCAT]
    features = np.stack(
        [
            normalized_numeric[:, numeric_cols.index("AGE")],
            df_merged["PTGENDER"].to_numpy(dtype=float),
            normalized_numeric[:, numeric_cols.index("MMSE")],
            normalized_numeric[:, numeric_cols.index("CDRSB")],
            df_merged["APOE4"].to_numpy(dtype=float),
            normalized_numeric[:, numeric_cols.index("PTEDUCAT")],
        ],
        axis=1,
    ).astype(np.float32)

    return features, labels, scaler


def main(config_path: str) -> None:
    """Entry point to prepare and encode ADNI clinical data.

    Args:
        config_path: Path to the YAML configuration file.
    """
    try:
        config = load_config(Path(config_path))
        dataset_cfg = config.get("dataset", {})

        clinical_csv_path = Path(dataset_cfg.get("clinical_csv", ""))
        networks_dir = Path(dataset_cfg.get("networks_dir", ""))
        splits_dir = Path(dataset_cfg.get("splits_dir", ""))

        if not clinical_csv_path:
            raise RuntimeError("Missing 'dataset.clinical_csv' in configuration.")
        if not networks_dir:
            raise RuntimeError("Missing 'dataset.networks_dir' in configuration.")
        if not splits_dir:
            raise RuntimeError("Missing 'dataset.splits_dir' in configuration.")

        splits_dir.mkdir(parents=True, exist_ok=True)

        dti_df = load_dti_subjects(networks_dir)
        clinical_df = load_clinical_csv(clinical_csv_path)

        # Inner join on SubjectID between DTI and clinical data.
        dti_df["SubjectID"] = dti_df["SubjectID"].astype(str)
        clinical_df["SubjectID"] = clinical_df["SubjectID"].astype(str)

        df_merged = clinical_df.merge(dti_df[["SubjectID", "Label"]], on="SubjectID")
        if df_merged.empty:
            raise RuntimeError(
                "No overlapping subjects between clinical CSV and DTI networks."
            )

        features, labels, scaler = encode_and_normalize(df_merged, splits_dir)

        # Ensure output directory exists (use the directory containing the clinical CSV).
        clinical_out_dir = clinical_csv_path.parent
        clinical_out_dir.mkdir(parents=True, exist_ok=True)

        # Save in SubjectID order matching `df_merged`.
        subject_ids = df_merged["SubjectID"].astype(str).to_numpy()

        features_path = clinical_out_dir / "clinical_features.npy"
        subject_ids_path = clinical_out_dir / "clinical_subject_ids.npy"
        scaler_path = clinical_out_dir / "clinical_scaler.pkl"

        np.save(features_path, features)
        np.save(subject_ids_path, subject_ids)
        joblib.dump(scaler, scaler_path)

        print(
            f"Saved clinical features for {features.shape[0]} subjects to '{features_path}'."
        )
        print(f"Saved subject ID mapping to '{subject_ids_path}'.")
        print(f"Saved fitted scaler to '{scaler_path}'.")
        print("=== Done: prepare_clinical ===")
    except Exception as exc:
        raise RuntimeError("Error while preparing clinical data.") from exc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare and encode ADNI clinical CSV data."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()
    main(args.config)

