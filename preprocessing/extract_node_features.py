"""Extract ROIS node features (voxel counts) for each AAL region.

This script:
1. Loads configuration from `configs/config.yaml`.
2. Reads FA maps in MNI space for each subject.
3. Uses the AAL atlas to define 90 brain regions.
4. For each region, counts the number of voxels with non-zero FA within that
   region as an approximation of the number of voxels traversed by fibers
   (ROIS feature in the paper).
5. Saves a 90-dimensional ROIS vector per subject into the configured
   `node_features_dir` as `{subject_id}_ROIS.npy`.

Note:
    The original paper uses deterministic fiber tracking to obtain exactly the
    number of voxels that fiber tracts pass through (ROISurfaceSize). Here we
    approximate this using non-zero FA voxels per region, which is practical
    when only the FA maps are available.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import nibabel as nib
import numpy as np
import yaml
from nilearn.datasets import fetch_atlas_aal
from nilearn.image import resample_to_img
from tqdm import tqdm


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


def load_aal_atlas() -> tuple[nib.Nifti1Image, List[int]]:
    """Load the AAL atlas and return the atlas image and first 90 region labels.

    Returns:
        A tuple of (atlas_img, region_indices) for the first 90 non-zero labels.
    """
    try:
        atlas = fetch_atlas_aal()
        atlas_img = nib.load(atlas.maps)
        atlas_data = atlas_img.get_fdata().astype(int)
        unique_labels = sorted(int(v) for v in np.unique(atlas_data) if v > 0)
        if len(unique_labels) < 90:
            raise RuntimeError(
                f"AAL atlas has fewer than 90 non-zero labels (found {len(unique_labels)})."
            )
        region_indices = unique_labels[:90]
        return atlas_img, region_indices
    except Exception as exc:
        raise RuntimeError("Failed to load or process AAL atlas.") from exc


def compute_rois_features(
    fa_img: nib.Nifti1Image,
    atlas_img: nib.Nifti1Image,
    region_indices: List[int],
    fa_threshold: float = 0.0,
) -> np.ndarray:
    """Compute ROIS features: count of non-zero FA voxels per AAL region.

    Args:
        fa_img: FA NIfTI image in MNI space.
        atlas_img: AAL atlas NIfTI image in MNI space.
        region_indices: List of atlas integer labels (length 90).
        fa_threshold: Minimum FA value to be considered as a valid voxel.

    Returns:
        Numpy array of shape (90,) with voxel counts per region.
    """
    fa_resampled = resample_to_img(fa_img, atlas_img, interpolation="continuous")
    fa_data = np.asarray(fa_resampled.get_fdata(), dtype=np.float32)
    atlas_data = np.asarray(atlas_img.get_fdata(), dtype=np.int32)

    rois_counts: List[int] = []
    for idx in region_indices:
        region_mask = atlas_data == idx
        if not np.any(region_mask):
            rois_counts.append(0)
            continue
        region_fa = fa_data[region_mask]
        count = int(np.sum(region_fa > fa_threshold))
        rois_counts.append(count)

    return np.asarray(rois_counts, dtype=np.int32)


def main(config_path: str) -> None:
    """Entry point to extract ROIS node features for all subjects.

    Args:
        config_path: Path to the YAML configuration file.
    """
    try:
        config = load_config(Path(config_path))
        dataset_cfg = config.get("dataset", {})

        networks_dir = Path(dataset_cfg.get("networks_dir", ""))
        node_features_dir = Path(dataset_cfg.get("node_features_dir", ""))

        if not networks_dir:
            raise RuntimeError("Missing 'dataset.networks_dir' in configuration.")
        if not node_features_dir:
            raise RuntimeError("Missing 'dataset.node_features_dir' in configuration.")

        # FA directory is assumed to be sibling of networks_dir (created by FSL pipeline).
        fa_dir = networks_dir.parent / "FA"
        if not fa_dir.is_dir():
            raise RuntimeError(
                f"FA directory '{fa_dir}' not found. Ensure FSL preprocessing has run."
            )

        node_features_dir.mkdir(parents=True, exist_ok=True)

        atlas_img, region_indices = load_aal_atlas()
        fa_files = sorted(fa_dir.glob("*_dti_FA_MNI.nii.gz"))

        if not fa_files:
            raise RuntimeError(
                f"No FA MNI files found in '{fa_dir}'. Expected files like "
                "'<subject_id>_dti_FA_MNI.nii.gz'."
            )

        print("=== Extracting ROIS node features ===")
        num_saved = 0
        for fa_path in tqdm(fa_files, desc="Subjects", unit="subject"):
            subject_id = fa_path.name.split("_dti_FA_MNI.nii.gz")[0]
            try:
                fa_img = nib.load(str(fa_path))
                rois_vec = compute_rois_features(fa_img, atlas_img, region_indices)
                out_path = node_features_dir / f"{subject_id}_ROIS.npy"
                np.save(out_path, rois_vec)
                num_saved += 1
            except Exception as exc:
                print(
                    f"WARNING: Failed to extract ROIS for subject '{subject_id}' "
                    f"from '{fa_path}': {exc}"
                )

        if num_saved == 0:
            raise RuntimeError(
                "No ROIS node feature vectors were generated. Check FA files and atlas."
            )

        print(f"Saved ROIS node features for {num_saved} subjects to '{node_features_dir}'.")
        print("=== Done: extract_node_features ===")
    except Exception as exc:
        raise RuntimeError("Error while extracting ROIS node features.") from exc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract ROIS node features (voxel counts) for each AAL region."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()
    main(args.config)

