"""
Scan all .rda/.RData files under data/clinical and print table names + columns.

This is a direct runnable version of the snippet you provided.

Run:
  python preprocessing/scan_rda_columns.py
"""

from __future__ import annotations

import glob
import os

import pandas as pd  # noqa: F401
import pyreadr


def main() -> None:
    base = r"D:\AlzheimerGCN\data\clinical"
    out = r"D:\AlzheimerGCN\data\clinical\adni_clinical.csv"  # noqa: F841

    # Step 1: Find all .rda files and print their table names + columns
    print("=== SCANNING YOUR .rda FILES ===\n")
    rda_files = glob.glob(base + "\\**\\*.rda", recursive=True)
    rda_files += glob.glob(base + "\\**\\*.RData", recursive=True)

    summary: dict[str, dict[str, object]] = {}
    for f in rda_files:
        try:
            result = pyreadr.read_r(f)
            for key, df in result.items():
                cols = [str(c).upper() for c in df.columns]
                fname = os.path.basename(f)
                summary[fname] = {"key": key, "cols": cols, "path": f, "shape": df.shape}
                # Print files that have diagnosis/demographic info
                keywords = ["RID", "DX", "AGE", "GENDER", "MMSE", "CDR", "APOE"]
                found = [k for k in keywords if any(k in c for c in cols)]
                if len(found) >= 2:
                    print(f"FILE: {fname}")
                    print(f"  Shape: {df.shape}, Key cols: {found}")
                    print(f"  All cols: {cols[:15]}\n")
        except Exception:
            pass


if __name__ == "__main__":
    main()

