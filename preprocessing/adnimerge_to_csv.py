"""
Utilities for inspecting ADNI clinical downloads and exporting ADNIMERGE to CSV.

This script does three things in order:
1) Recursively lists all files under data/clinical with full path + size.
2) Finds an ADNIMERGE .rda/.RData file and converts it to adni_clinical.csv.
3) Verifies the saved CSV and checks required columns exist.

Run (from project root):
  python preprocessing/adnimerge_to_csv.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLINICAL_DIR = PROJECT_ROOT / "data" / "clinical"
OUT_CSV = CLINICAL_DIR / "adni_clinical.csv"
REQUIRED_COLUMNS = ("RID", "DX_bl", "AGE", "PTGENDER", "MMSE", "CDRSB", "APOE4")


@dataclass(frozen=True)
class FileInfo:
    """Container for file listing output."""

    path: Path
    size_bytes: int


def iter_files_recursively(root: Path) -> Iterable[FileInfo]:
    """Yield all files under root recursively with their sizes."""
    for p in root.rglob("*"):
        if p.is_file():
            try:
                size = p.stat().st_size
            except OSError:
                size = -1
            yield FileInfo(path=p, size_bytes=size)


def print_file_listing(root: Path) -> None:
    """Print every file under root with full path and size."""
    if not root.exists():
        raise FileNotFoundError(f"Clinical directory not found: {root}")

    files = sorted(iter_files_recursively(root), key=lambda x: str(x.path).lower())
    print(f"=== TASK 1: Listing all files under: {root} ===")
    if not files:
        print("(No files found.)")
        return

    for fi in files:
        print(f"{fi.path} | {fi.size_bytes} bytes")
    print(f"Total files: {len(files)}")


def pick_adnimerge_rda(clinical_root: Path) -> Path:
    """Find the most likely ADNIMERGE .rda/.RData file under clinical_root."""
    candidates = []
    for ext in ("*.rda", "*.RDA", "*.RData", "*.rdata"):
        candidates.extend(clinical_root.rglob(ext))

    if not candidates:
        raise FileNotFoundError(
            f"No .rda/.RData files found under: {clinical_root}\n"
            "If your download is different, place ADNIMERGE .rda in data/clinical/."
        )

    # Prefer files containing 'adnimerge' in the name/path.
    scored = []
    for p in candidates:
        s = str(p).lower()
        score = 0
        if "adnimerge" in s:
            score += 100
        if "merge" in s:
            score += 10
        # Larger files are more likely to contain the full merged table.
        try:
            score += min(int(p.stat().st_size / (1024 * 1024)), 200)
        except OSError:
            pass
        scored.append((score, p))

    scored.sort(key=lambda t: (-t[0], str(t[1]).lower()))
    return scored[0][1]


def sort_rda_candidates_for_trying(candidates: Sequence[Path]) -> list[Path]:
    """Sort candidates so likely ADNIMERGE-like tables are tried first."""

    def score_path(p: Path) -> int:
        s = str(p).lower()
        score = 0
        if "adnimerge" in s:
            score += 10_000
        if "merge" in s:
            score += 200
        if "demog" in s:
            score += 100
        if "dx" in s:
            score += 100
        if p.name.lower() in {"adsl.rda", "dm.rda", "ptdemog.rda", "dxsum.rda"}:
            score += 2_000
        try:
            score += min(int(p.stat().st_size / (1024 * 1024)), 500)
        except OSError:
            pass
        return score

    return sorted(candidates, key=lambda p: (-score_path(p), str(p).lower()))


def ensure_pyreadr_installed() -> None:
    """Install pyreadr via pip if it's not importable."""
    try:
        import pyreadr  # noqa: F401

        return
    except Exception:
        pass

    print("pyreadr not found. Installing with pip...")
    cmd = f'"{sys.executable}" -m pip install pyreadr'
    rc = os.system(cmd)
    if rc != 0:
        raise RuntimeError(
            "Failed to install pyreadr. Please activate your conda env and run:\n"
            f"  {cmd}"
        )


def convert_rda_to_csv(rda_path: Path, out_csv: Path) -> None:
    """Read an R .rda/.RData file using pyreadr and export the first table to CSV."""
    ensure_pyreadr_installed()
    import pandas as pd
    import pyreadr

    print(f"=== TASK 2: Converting R data to CSV ===")
    print(f"Using R data file: {rda_path}")

    try:
        result = pyreadr.read_r(str(rda_path))
    except Exception as e:
        raise RuntimeError(f"pyreadr failed to read {rda_path}: {e}") from e

    if not result:
        raise RuntimeError(f"No objects found inside R data file: {rda_path}")

    # Pick the first DataFrame-like object.
    df: pd.DataFrame | None = None
    chosen_key: str | None = None
    for k, v in result.items():
        if hasattr(v, "shape") and hasattr(v, "columns"):
            df = v
            chosen_key = k
            break

    if df is None:
        keys = ", ".join(result.keys())
        raise RuntimeError(
            f"Found objects in {rda_path} but none looked like a table. Keys: {keys}"
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    first_cols = list(df.columns[:5])
    print(
        f"Saved CSV: {out_csv}\n"
        f"Loaded object: {chosen_key}\n"
        f"Shape: {df.shape}\n"
        f"First 5 columns: {first_cols}"
    )


def try_find_rda_with_required_columns(clinical_root: Path) -> tuple[Path, str]:
    """
    Search .rda/.RData files and return (rda_path, object_key) where the table
    contains REQUIRED_COLUMNS.
    """
    ensure_pyreadr_installed()
    import pyreadr

    candidates: list[Path] = []
    for ext in ("*.rda", "*.RDA", "*.RData", "*.rdata"):
        candidates.extend(clinical_root.rglob(ext))
    if not candidates:
        raise FileNotFoundError(f"No .rda/.RData files found under: {clinical_root}")

    candidates = sort_rda_candidates_for_trying(candidates)

    print("=== TASK 2: Searching for an R data table with required columns ===")
    print(f"Required columns: {list(REQUIRED_COLUMNS)}")
    print(f"Total .rda/.RData candidates: {len(candidates)}")

    for i, rda_path in enumerate(candidates, start=1):
        try:
            result = pyreadr.read_r(str(rda_path))
        except Exception:
            continue

        for k, v in result.items():
            cols = getattr(v, "columns", None)
            if cols is None:
                continue
            colset = set(map(str, cols))
            if all(c in colset for c in REQUIRED_COLUMNS):
                print(f"Selected candidate #{i}: {rda_path}")
                print(f"Selected object key: {k}")
                return rda_path, k

    fallback = pick_adnimerge_rda(clinical_root)
    raise RuntimeError(
        "Could not find any .rda/.RData table containing the required columns.\n"
        f"Fallback heuristic candidate was: {fallback}\n"
        "This likely means your download does not include the ADNIMERGE-style merged table.\n"
        "If you can download `ADNIMERGE.csv` (or `ADNIMERGE2.csv`) from ADNI, place it at:\n"
        f"  {OUT_CSV}"
    )


def convert_best_rda_to_csv(clinical_root: Path, out_csv: Path) -> None:
    """Find a suitable .rda with REQUIRED_COLUMNS and convert it to CSV."""
    ensure_pyreadr_installed()
    import pyreadr

    rda_path, object_key = try_find_rda_with_required_columns(clinical_root)
    print("=== TASK 2: Converting selected R data to CSV ===")
    result = pyreadr.read_r(str(rda_path))
    df = result[object_key]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    first_cols = list(df.columns[:5])
    print(
        f"Saved CSV: {out_csv}\n"
        f"Source R data file: {rda_path}\n"
        f"Loaded object: {object_key}\n"
        f"Shape: {df.shape}\n"
        f"First 5 columns: {first_cols}"
    )


def verify_csv(csv_path: Path) -> None:
    """Verify CSV exists and print rows/columns/head + required column check."""
    import pandas as pd

    print("=== TASK 3: Verifying saved CSV ===")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"CSV path: {csv_path}")
    print(f"Total rows: {df.shape[0]}")
    print(f"Total columns: {df.shape[1]}")
    print("Column names:")
    print(list(df.columns))
    print("First 3 rows:")
    print(df.head(3).to_string(index=False))

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(
            "Missing required columns: "
            + ", ".join(missing)
            + "\n"
            + "Your ADNIMERGE variant might use different names. "
            + "We can map them once we see the available columns."
        )

    print(f"Required columns check: OK ({', '.join(REQUIRED_COLUMNS)})")


def main() -> None:
    """Run Tasks 1-3 sequentially."""
    try:
        print_file_listing(CLINICAL_DIR)
        convert_best_rda_to_csv(CLINICAL_DIR, OUT_CSV)
        verify_csv(OUT_CSV)
        print("=== Done: clinical listing + ADNIMERGE CSV export + verification ===")
    except Exception as e:
        print(f"[ERROR] {e}")
        raise


if __name__ == "__main__":
    main()

