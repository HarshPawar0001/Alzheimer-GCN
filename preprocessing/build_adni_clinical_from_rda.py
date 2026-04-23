"""
Build `data/clinical/adni_clinical.csv` by merging ADNI RDA tables.

Why this exists:
- Your "ADNIMERGE2" download is an R package-style folder with many `.rda` tables.
- There isn't a single ADNIMERGE.csv inside, so we construct the minimal clinical CSV
  needed by this project by merging multiple tables on RID (and filtering to baseline).

Output columns (minimum required by pipeline):
  RID, DX_bl, AGE, PTGENDER, MMSE, CDRSB, APOE4

Run:
  python preprocessing/build_adni_clinical_from_rda.py
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyreadr


BASE = Path(r"D:\AlzheimerGCN\data\clinical")
OUT = BASE / "adni_clinical.csv"

REQUIRED = ["RID", "DX_BL", "AGE", "PTGENDER", "MMSE", "CDRSB", "APOE4"]


def _glob_one(pattern: str) -> Path | None:
    files = glob.glob(str(BASE / "**" / pattern), recursive=True)
    if not files:
        return None
    return Path(files[0])


def _load_rda_first_table(path: Path) -> pd.DataFrame:
    result = pyreadr.read_r(str(path))
    df = list(result.values())[0]
    df = df.copy()
    df.columns = [str(c).upper() for c in df.columns]
    return df


def load_table(patterns: Iterable[str], name: str) -> pd.DataFrame | None:
    """Try multiple glob patterns; load first match."""
    for pat in patterns:
        p = _glob_one(pat)
        if p is None:
            continue
        df = _load_rda_first_table(p)
        print(f"[OK] Loaded {name}: {p.name} -> shape {df.shape}")
        return df
    print(f"[MISSING] {name}: tried {list(patterns)}")
    return None


def pick_baseline(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Attempt to keep baseline rows using VISCODE/VISCODE2 when available."""
    if df is None:
        return None

    df2 = df.copy()
    for vc in ("VISCODE", "VISCODE2"):
        if vc in df2.columns:
            vals = df2[vc].astype(str).str.upper()
            # ADNI baseline commonly "BL"; some tables have "SC" (screening) or "M00".
            mask = vals.isin(["BL", "SC", "M00", "ADNI1"])
            if mask.any():
                return df2.loc[mask].copy()
    return df2


def safe_select(df: pd.DataFrame | None, cols: list[str], name: str) -> pd.DataFrame | None:
    """Select available columns; require RID."""
    if df is None:
        return None
    cols_up = [c.upper() for c in cols]
    available = [c for c in cols_up if c in df.columns]
    if "RID" not in available:
        print(f"[WARN] {name}: RID not found; skipping")
        return None
    out = df[available].copy()
    out = out.drop_duplicates(subset=["RID"])
    return out


def compute_age_from_dates(demo: pd.DataFrame | None) -> pd.DataFrame | None:
    """
    Compute AGE (years) if missing using PTDOB and VISDATE.

    PTDEMOG in your download does not include an AGE column, but it includes:
    - PTDOB / PTDOBYY (birth date/year)
    - VISDATE (visit date)
    """
    if demo is None:
        return None

    demo = demo.copy()
    if "AGE" in demo.columns:
        return demo

    if "PTDOB" in demo.columns and "VISDATE" in demo.columns:
        dob = pd.to_datetime(demo["PTDOB"], errors="coerce")
        vis = pd.to_datetime(demo["VISDATE"], errors="coerce")
        age = (vis - dob).dt.days / 365.25
        demo["AGE"] = age
        return demo

    if "PTDOBYY" in demo.columns and "VISDATE" in demo.columns:
        vis = pd.to_datetime(demo["VISDATE"], errors="coerce")
        birth_year = pd.to_numeric(demo["PTDOBYY"], errors="coerce")
        demo["AGE"] = vis.dt.year - birth_year
        return demo

    return demo


def standardize_columns(
    dx: pd.DataFrame | None,
    mmse: pd.DataFrame | None,
    cdr: pd.DataFrame | None,
    apoe: pd.DataFrame | None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """Rename common variants to standard column names."""
    if dx is not None:
        # Try to map a diagnosis column to DX_BL.
        for src in ["DX_BL", "DIAGNOSIS", "DX", "DXCURREN", "DXCHANGE"]:
            if src in dx.columns and "DX_BL" not in dx.columns:
                dx = dx.rename(columns={src: "DX_BL"})
        if "DX" in dx.columns and "DX_BL" not in dx.columns:
            dx = dx.rename(columns={"DX": "DX_BL"})

    if mmse is not None:
        for src in ["MMSE", "MMSCORE", "MMSESTOT", "MMSCORE_TOTAL"]:
            if src in mmse.columns and "MMSE" not in mmse.columns:
                mmse = mmse.rename(columns={src: "MMSE"})

    if cdr is not None:
        if "CDRSUM" in cdr.columns and "CDRSB" not in cdr.columns:
            cdr = cdr.rename(columns={"CDRSUM": "CDRSB"})
        if "CDRSB" not in cdr.columns and "CDR" in cdr.columns:
            cdr = cdr.rename(columns={"CDR": "CDRSB"})

    if apoe is not None:
        # APOE4 is often already present. If not, try to derive from APGEN1/APGEN2.
        if "APOE4" not in apoe.columns and "APGEN1" in apoe.columns and "APGEN2" in apoe.columns:
            # Convert allele strings to counts of "4" (works if alleles are 2/3/4 numeric or strings).
            a1 = apoe["APGEN1"].astype(str)
            a2 = apoe["APGEN2"].astype(str)
            apoe["APOE4"] = (a1 == "4").astype(int) + (a2 == "4").astype(int)
        # APOERES.rda in your download has a GENOTYPE column; derive APOE4 count from it.
        if "APOE4" not in apoe.columns and "GENOTYPE" in apoe.columns:
            g = apoe["GENOTYPE"].astype(str).str.upper()
            # Examples seen in ADNI exports: "3/4", "2/3", "E3/E4", "3|4"
            g = g.str.replace("E", "", regex=False)
            g = g.str.replace("|", "/", regex=False)
            parts = g.str.split("/", expand=True)
            if parts.shape[1] >= 2:
                a1 = parts[0].str.extract(r"(\d)", expand=False)
                a2 = parts[1].str.extract(r"(\d)", expand=False)
                apoe["APOE4"] = (a1 == "4").astype(int) + (a2 == "4").astype(int)

    return dx, mmse, cdr, apoe


def verify_and_save(final: pd.DataFrame) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUT, index=False)

    print("\n=== SAVED CSV ===")
    print(f"Path: {OUT}")
    print(f"Rows: {final.shape[0]} | Cols: {final.shape[1]}")
    print("Columns:", list(final.columns))
    print("\nFirst 3 rows:")
    print(final.head(3).to_string(index=False))

    missing = [c for c in REQUIRED if c not in final.columns]
    if missing:
        raise RuntimeError(f"Missing required columns after merge: {missing}")

    print("\n[OK] Required columns present:", REQUIRED)


def main() -> None:
    print("=== Building adni_clinical.csv from RDA tables ===")
    print(f"Base: {BASE}")

    dx = load_table(["DXSUM*.rda", "DXSUM*.RData"], "DX")
    demo = load_table(["PTDEMOG*.rda", "PTDEMOG*.RData", "DM*.rda", "DM*.RData"], "DEMOGRAPHICS")
    mmse = load_table(["MMSE*.rda", "MMSE*.RData"], "MMSE")
    cdr = load_table(["CDR*.rda", "CDR*.RData"], "CDR")
    apoe = load_table(["APOERES*.rda", "APOERES*.RData"], "APOE")

    # Fallbacks if some key tables are missing.
    if dx is None:
        dx = load_table(["NEUROPATH*.rda", "NEUROPATH*.RData"], "DX (fallback NEUROPATH)")
        if dx is not None and "DX" in dx.columns:
            dx = dx.rename(columns={"DX": "DX_BL"})

    if mmse is None:
        pacc = load_table(["PACC*.rda", "PACC*.RData"], "MMSE (fallback PACC)")
        if pacc is not None and "MMSE" in pacc.columns and "RID" in pacc.columns:
            mmse = pacc[["RID", "MMSE"]].copy()

    # Prefer baseline rows if possible.
    dx = pick_baseline(dx)
    demo = pick_baseline(demo)
    mmse = pick_baseline(mmse)
    cdr = pick_baseline(cdr)
    apoe = pick_baseline(apoe)

    # Select needed columns from each.
    dx = safe_select(dx, ["RID", "DX_BL", "DIAGNOSIS", "DX", "DXCURREN", "DXCHANGE"], "DX")
    demo = safe_select(demo, ["RID", "PTGENDER", "PTEDUCAT", "PTDOB", "PTDOBYY", "VISDATE", "AGE"], "DEMOGRAPHICS")
    mmse = safe_select(mmse, ["RID", "MMSE", "MMSCORE", "MMSESTOT", "MMSCORE_TOTAL"], "MMSE")
    cdr = safe_select(cdr, ["RID", "CDRSB", "CDRSUM", "CDR"], "CDR")
    apoe = safe_select(apoe, ["RID", "APOE4", "APGEN1", "APGEN2", "GENOTYPE"], "APOE")

    demo = compute_age_from_dates(demo)

    dx, mmse, cdr, apoe = standardize_columns(dx, mmse, cdr, apoe)

    # Merge (start from the table that is present).
    base_df = dx if dx is not None else demo
    if base_df is None:
        raise RuntimeError("Could not load any base table (DX or DEMOGRAPHICS).")

    final = base_df.copy()
    for t in [demo, mmse, cdr, apoe]:
        if t is not None:
            final = final.merge(t, on="RID", how="left")

    # Final standard column casing.
    final.columns = [str(c) for c in final.columns]

    # Ensure required columns exist (some may be missing if a table wasn't found).
    verify_and_save(final)


if __name__ == "__main__":
    main()

