"""
End-to-end fixer for ADNI DTI data layout in D:\\AlzheimerGCN.

What it does (TASK 1):
- Reads `data\\clinical\\adni_clinical.csv` to get RID -> diagnosis (AD/CN).
- Parses RID from raw DICOM subject folder names like "003_S_4136" (RID=4136).
- Copies those folders into:
    - data\\raw\\AD\\{SubjectID}\\
    - data\\raw\\CN\\{SubjectID}\\
- Converts each subject's DICOM folder to NIfTI using dcm2niix.exe into:
    - data\\nifti\\AD\\{SubjectID}\\
    - data\\nifti\\CN\\{SubjectID}\\
- Creates `data\\processed\\networks\\labels.csv` with:
    SubjectID,Label,DX   where AD=1 and CN=0
- Prints a summary of counts.

Run with your conda python:
  C:\\Users\\abhishek\\miniconda3\\envs\\alzgcn\\python.exe D:\\AlzheimerGCN\\fix_all.py
"""

from __future__ import annotations

import csv
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class Paths:
    project_root: Path
    raw_dti_mixed: Path
    raw_ad: Path
    raw_cn: Path
    nifti_ad: Path
    nifti_cn: Path
    clinical_csv: Path
    dcm2niix_exe: Path
    labels_csv: Path


RID_PATTERN = re.compile(r"_S_(\d+)", flags=re.IGNORECASE)


def get_paths() -> Paths:
    project_root = Path(r"D:\\AlzheimerGCN")
    return Paths(
        project_root=project_root,
        raw_dti_mixed=project_root / "data" / "raw" / "DTI",
        raw_ad=project_root / "data" / "raw" / "AD",
        raw_cn=project_root / "data" / "raw" / "CN",
        nifti_ad=project_root / "data" / "nifti" / "AD",
        nifti_cn=project_root / "data" / "nifti" / "CN",
        clinical_csv=project_root / "data" / "clinical" / "adni_clinical.csv",
        dcm2niix_exe=project_root / "dcm2niix.exe",
        labels_csv=project_root / "data" / "processed" / "networks" / "labels.csv",
    )


def parse_rid(subject_folder_name: str) -> Optional[int]:
    m = RID_PATTERN.search(subject_folder_name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def load_rid_to_dx(clinical_csv: Path) -> Dict[int, str]:
    if not clinical_csv.is_file():
        raise FileNotFoundError(f"Clinical CSV not found: {clinical_csv}")

    df = pd.read_csv(clinical_csv)
    cols = {c.upper(): c for c in df.columns}

    # Prefer RID column if present; otherwise attempt to use SubjectID as RID.
    if "RID" in cols:
        rid_col = cols["RID"]
        df["__RID__"] = pd.to_numeric(df[rid_col], errors="coerce").astype("Int64")
    elif "SUBJECTID" in cols:
        sid_col = cols["SUBJECTID"]
        df["__RID__"] = pd.to_numeric(df[sid_col], errors="coerce").astype("Int64")
    else:
        raise RuntimeError(
            "Clinical CSV must contain a RID column (preferred) or SubjectID that is RID."
        )

    # DX can be DX_bl or DX_BL or DX or DIAGNOSIS
    dx_col = None
    for cand in ("DX_BL", "DX_BL", "DX_BL", "DX_BL"):
        _ = cand
    for cand in ("DX_BL", "DX_BL"):
        _ = cand
    for cand in ("DX_BL", "DX_BL"):
        _ = cand
    for cand in ("DX_BL", "DX_BL"):
        _ = cand
    for cand in ("DX_BL", "DX_bl", "DX", "DIAGNOSIS"):
        if cand.upper() in cols:
            dx_col = cols[cand.upper()]
            break
    if dx_col is None:
        raise RuntimeError(
            f"Could not find diagnosis column in clinical CSV. Columns: {list(df.columns)}"
        )

    rid_to_dx: Dict[int, str] = {}
    for _, row in df.iterrows():
        rid = row["__RID__"]
        if pd.isna(rid):
            continue
        dx_raw = str(row[dx_col]).strip().upper()
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
        dx = dx_map.get(dx_raw, dx_raw)
        if dx in {"AD", "CN"}:
            rid_to_dx[int(rid)] = dx
    return rid_to_dx


def _ensure_pyreadr_available() -> None:
    """Install pyreadr into the current python if missing."""
    try:
        import pyreadr  # noqa: F401

        return
    except Exception:
        pass

    print("[INFO] pyreadr not found. Installing pyreadr to current environment...")
    cmd = [sys.executable, "-m", "pip", "install", "pyreadr"]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "Failed to install pyreadr.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}\n"
        )


def load_rid_to_dx_from_dxsum_rda(project_root: Path) -> Dict[int, str]:
    """
    Fallback RID->DX mapping using the ADNIMERGE2 R package tables (DXSUM.rda).

    This helps when your generated `adni_clinical.csv` doesn't include some RIDs.
    """
    _ensure_pyreadr_available()
    import glob
    import pyreadr

    base = project_root / "data" / "clinical"
    matches = glob.glob(str(base / "**" / "DXSUM*.rda"), recursive=True)
    if not matches:
        return {}

    dxsum_path = Path(matches[0])
    result = pyreadr.read_r(str(dxsum_path))
    df = list(result.values())[0].copy()
    df.columns = [str(c).upper() for c in df.columns]
    if "RID" not in df.columns:
        return {}

    # Prefer DIAGNOSIS column if present, else DX-like columns.
    dx_col = None
    for cand in ("DIAGNOSIS", "DX", "DX_BL", "DX_BL"):
        if cand in df.columns:
            dx_col = cand
            break
    if dx_col is None:
        return {}

    df["RID"] = pd.to_numeric(df["RID"], errors="coerce")
    df = df.dropna(subset=["RID"])
    df[dx_col] = df[dx_col].astype(str).str.strip().str.upper()

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

    rid_to_dx: Dict[int, str] = {}
    for _, row in df.iterrows():
        rid = int(row["RID"])
        dx_raw = str(row[dx_col]).strip().upper()
        dx = dx_map.get(dx_raw, dx_raw)
        if dx in {"AD", "CN"} and rid not in rid_to_dx:
            rid_to_dx[rid] = dx
    return rid_to_dx


def ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)


def copy_subject_folder(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    shutil.copytree(src, dst)


def run_dcm2niix(dcm2niix_exe: Path, dicom_dir: Path, out_dir: Path) -> None:
    ensure_dir(out_dir)
    cmd = [
        str(dcm2niix_exe),
        "-z",
        "y",
        "-o",
        str(out_dir),
        "-f",
        "%p_%s",
        str(dicom_dir),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "dcm2niix failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Return code: {completed.returncode}\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}\n"
        )


def write_labels_csv(rows: List[Tuple[str, int, str]], out_csv: Path) -> None:
    ensure_dir(out_csv.parent)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["SubjectID", "Label", "DX"])
        writer.writerows(rows)


def main() -> None:
    p = get_paths()
    print("=== fix_all.py (TASK 1) ===")
    print(f"Project root: {p.project_root}")
    print(f"Mixed raw DTI dir: {p.raw_dti_mixed}")
    print(f"Clinical CSV: {p.clinical_csv}")
    print(f"dcm2niix: {p.dcm2niix_exe}")

    if not p.raw_dti_mixed.exists():
        raise FileNotFoundError(f"Raw DTI folder not found: {p.raw_dti_mixed}")
    if not p.dcm2niix_exe.is_file():
        raise FileNotFoundError(f"dcm2niix.exe not found: {p.dcm2niix_exe}")

    rid_to_dx = load_rid_to_dx(p.clinical_csv)
    # If clinical CSV is missing many RIDs, augment from DXSUM.rda if available.
    try:
        rid_to_dx_dxsum = load_rid_to_dx_from_dxsum_rda(p.project_root)
        added = 0
        for rid, dx in rid_to_dx_dxsum.items():
            if rid not in rid_to_dx:
                rid_to_dx[rid] = dx
                added += 1
        if added > 0:
            print(f"[INFO] Added {added} RID->DX mappings from DXSUM.rda fallback.")
    except Exception as exc:
        print(f"[WARN] DXSUM.rda fallback mapping failed: {exc}")
    if not rid_to_dx:
        raise RuntimeError("No RID->DX mappings found for AD/CN in clinical CSV.")

    for d in (p.raw_ad, p.raw_cn, p.nifti_ad, p.nifti_cn, p.labels_csv.parent):
        ensure_dir(d)

    labels_rows: List[Tuple[str, int, str]] = []
    ad_count = 0
    cn_count = 0
    skipped = 0
    converted = 0

    subject_dirs = sorted([d for d in p.raw_dti_mixed.iterdir() if d.is_dir()], key=lambda x: x.name)

    for subj_dir in subject_dirs:
        subject_id = subj_dir.name
        rid = parse_rid(subject_id)
        if rid is None:
            skipped += 1
            print(f"[SKIP] Could not parse RID from folder: {subject_id}")
            continue

        dx = rid_to_dx.get(rid)
        if dx not in {"AD", "CN"}:
            skipped += 1
            print(f"[SKIP] RID {rid} not labeled AD/CN in clinical CSV (folder {subject_id})")
            continue

        if dx == "AD":
            label = 1
            dst_raw = p.raw_ad / subject_id
            dst_nifti = p.nifti_ad / subject_id
            ad_count += 1
        else:
            label = 0
            dst_raw = p.raw_cn / subject_id
            dst_nifti = p.nifti_cn / subject_id
            cn_count += 1

        copy_subject_folder(subj_dir, dst_raw)
        try:
            run_dcm2niix(p.dcm2niix_exe, dst_raw, dst_nifti)
        except Exception as exc:
            print(f"[ERROR] dcm2niix failed for {subject_id}: {exc}")
            continue

        labels_rows.append((subject_id, label, dx))
        converted += 1
        print(f"[OK] {subject_id} -> {dx} (label={label}) converted to NIfTI")

    write_labels_csv(labels_rows, p.labels_csv)

    print("\n=== SUMMARY ===")
    print(f"AD subjects matched: {ad_count}")
    print(f"CN subjects matched: {cn_count}")
    print(f"Subjects converted to NIfTI: {converted}")
    print(f"labels.csv rows: {len(labels_rows)} -> {p.labels_csv}")
    print(f"Skipped folders: {skipped}")
    print("=== Done: fix_all.py ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)

