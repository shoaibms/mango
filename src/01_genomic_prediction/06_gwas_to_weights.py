# -*- coding: utf-8 -*-
"""
06_gwas_to_weights.py

Use trait-specific GWAS summary statistics to build:
  1. SNP weight vectors aligned to geno_core.npz (for weighted GRM / ridge).
  2. "Major QTL" SNP lists (for fixed-effect modelling in GS).

Inputs
------
- geno_core.npz  (from 01_build_core_matrices.py)
- GWAS summary files per trait, configured in config_idea1.GWAS_SUMMARY_CONFIG

Outputs (in cfg.GWAS_WEIGHTS_DIR)
---------------------------------
- snp_weights_<trait>.npz
    variant_ids : array of variant IDs (exactly the same as in geno_core.npz)
    weights     : array of non-negative weights, same length as variant_ids

- major_qtl_snps_<trait>.csv
    variant_id, p_value, rank
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np

try:
    import pandas as pd
except ImportError as e:
    raise SystemExit("pandas is required. Install with: pip install pandas") from e

import config_idea1 as cfg


# =========================
# Helpers
# =========================

def load_gwas_for_trait(
    trait: str,
    variant_ids_core: np.ndarray,
) -> pd.DataFrame:
    """
    Load GWAS summary for a trait and harmonise with core variant_ids.

    Returns
    -------
    df_merged : DataFrame with columns:
        - variant_id  (matching geno_core.npz)
        - p_value
        - [optional] beta

    Notes
    -----
    - Matching is done using either:
        (a) a direct SNP ID column (snp_id_col), or
        (b) concatenated 'CHR:POS' from chr_col and pos_col.
    - Only SNPs whose IDs match variant_ids_core are kept.
    """
    if trait not in cfg.GWAS_SUMMARY_CONFIG:
        raise KeyError(f"Trait '{trait}' not in GWAS_SUMMARY_CONFIG")

    cfg_trait = cfg.GWAS_SUMMARY_CONFIG[trait]
    path = cfg_trait["path"]
    snp_id_col = cfg_trait.get("snp_id_col")
    chr_col = cfg_trait.get("chr_col")
    pos_col = cfg_trait.get("pos_col")
    p_col = cfg_trait.get("p_col")
    beta_col = cfg_trait.get("beta_col")

    path_str = str(path)
    if not os.path.exists(path_str):
        raise FileNotFoundError(
            f"GWAS summary file for trait '{trait}' not found at: {path_str}"
        )

    print(f"[INFO] Loading GWAS for {trait}: {path_str}")
    df = pd.read_csv(path_str, sep=None, engine="python")

    if p_col not in df.columns:
        raise KeyError(
            f"p-value column '{p_col}' not found in GWAS file for trait '{trait}'. "
            f"Available columns: {list(df.columns)}"
        )

    # Build a 'variant_id' column that matches geno_core's variant IDs
    if snp_id_col is not None and snp_id_col in df.columns:
        df["variant_id"] = df[snp_id_col].astype(str)
    elif chr_col is not None and pos_col is not None:
        if chr_col not in df.columns or pos_col not in df.columns:
            raise KeyError(
                f"chr_col '{chr_col}' or pos_col '{pos_col}' not found in GWAS file for '{trait}'."
            )
        df["variant_id"] = (
            df[chr_col].astype(str).str.strip()
            + ":" +
            df[pos_col].astype(str).str.strip()
        )
    else:
        raise RuntimeError(
            f"GWAS config for trait '{trait}' must define either 'snp_id_col' or (chr_col + pos_col)."
        )

    df["p_value"] = df[p_col].astype(float)

    # Optional effect size
    if beta_col is not None:
        if beta_col not in df.columns:
            print(
                f"[WARN] beta_col '{beta_col}' not found for trait '{trait}'; "
                "continuing without effect sizes."
            )
            df["beta"] = np.nan
        else:
            df["beta"] = df[beta_col].astype(float)
    else:
        df["beta"] = np.nan

    # Restrict to SNPs present in core variant IDs
    core_set = set(variant_ids_core.astype(str))
    df_in_core = df[df["variant_id"].isin(core_set)].copy()

    print(
        f"[INFO] Trait {trait}: GWAS rows = {len(df)}, "
        f"matched to core SNPs = {len(df_in_core)}"
    )

    if len(df_in_core) == 0:
        raise RuntimeError(
            f"No GWAS SNPs for trait '{trait}' matched variant_ids in geno_core.npz. "
            "Check ID format and GWAS_SUMMARY_CONFIG."
        )

    # Drop duplicates (in case of multiple models, etc.), keeping best p-value per variant
    df_in_core.sort_values("p_value", inplace=True)
    df_in_core = df_in_core.drop_duplicates(subset=["variant_id"], keep="first")

    return df_in_core


def compute_weights_from_pvalues(
    variant_ids_core: np.ndarray,
    df_gwas: pd.DataFrame,
    p_floor: float = 1e-300,
) -> np.ndarray:
    """
    Compute a non-negative weight vector aligned to variant_ids_core
    based on GWAS p-values.

    Strategy:
      - w_raw = -log10(p)
      - Clip extreme values.
      - Normalise to [0, 1], then add a small epsilon so that even
        low-weight SNPs are not exactly zero.

    SNPs not present in df_gwas get weight = median(weight_observed).

    Returns
    -------
    weights : np.ndarray, shape (n_snps,)
    """
    variant_ids_core = variant_ids_core.astype(str)
    df = df_gwas.copy()

    p = df["p_value"].values.astype(float)
    p = np.clip(p, p_floor, 1.0)
    logp = -np.log10(p)

    # Avoid all-zero / all-NaN
    if np.all(~np.isfinite(logp)) or np.nanmax(logp) <= 0:
        print("[WARN] All log10(p) values are non-positive or non-finite; using uniform weights.")
        w_obs = np.ones_like(logp)
    else:
        # Normalise logp to [0,1]
        max_logp = np.nanpercentile(logp, 99)  # robust cap
        if max_logp <= 0:
            max_logp = np.nanmax(logp)
        w_obs = np.clip(logp / max_logp, 0.0, 1.0)

    # Map variant_id -> weight
    weight_map: Dict[str, float] = dict(zip(df["variant_id"].astype(str), w_obs))

    weights = np.empty(len(variant_ids_core), dtype=float)
    median_w = float(np.median(w_obs))

    for i, vid in enumerate(variant_ids_core):
        weights[i] = weight_map.get(vid, median_w)

    # Add small epsilon so no weight is exactly zero
    eps = 1e-3
    weights = np.maximum(weights, eps)

    return weights.astype(np.float32)


def select_major_qtl(
    df_gwas: pd.DataFrame,
    p_threshold: float,
    n_top: int,
) -> pd.DataFrame:
    """
    Select a "major QTL" SNP set for a trait, based on GWAS p-values.

    Criteria:
      - p <= p_threshold OR in the top n_top SNPs (whichever is less strict).

    Returns
    -------
    df_major : DataFrame with variant_id, p_value, rank.
    """
    df = df_gwas.copy()
    df.sort_values("p_value", inplace=True)

    df["rank"] = np.arange(1, len(df) + 1)
    mask = (df["p_value"] <= p_threshold) | (df["rank"] <= n_top)
    df_major = df.loc[mask, ["variant_id", "p_value", "rank"]].copy()

    print(
        f"[INFO] Selected {len(df_major)} major QTL SNPs "
        f"(p <= {p_threshold} OR top {n_top})"
    )
    return df_major


# =========================
# Main
# =========================

def main() -> None:
    cfg.ensure_output_dirs()
    out_dir = cfg.GWAS_WEIGHTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    geno_path = os.path.join(cfg.CORE_DATA_DIR, "geno_core.npz")
    if not os.path.exists(geno_path):
        raise FileNotFoundError(
            f"geno_core.npz not found at {geno_path}. "
            "Run 01_build_core_matrices.py first."
        )

    print("=" * 72)
    print("Mango GS – Idea 1: GWAS → SNP weights and major QTL sets")
    print("=" * 72)
    print(f"[INFO] Geno core: {geno_path}")
    print(f"[INFO] Output dir: {out_dir}")
    print()

    npz = np.load(geno_path, allow_pickle=True)
    variant_ids_core = npz["variant_ids"]
    n_snps = len(variant_ids_core)
    print(f"[INFO] Core variant IDs: {n_snps} SNPs")

    traits = [t for t in cfg.TRAITS_DEFAULT if t in cfg.GWAS_SUMMARY_CONFIG]
    print("[INFO] Traits with GWAS config:", ", ".join(traits))
    print()

    summary_records: List[Dict[str, object]] = []

    for trait in traits:
        print("-" * 72)
        print(f"[TRAIT] {trait}")

        try:
            df_gwas = load_gwas_for_trait(
                trait=trait,
                variant_ids_core=variant_ids_core,
            )
        except Exception as exc:
            print(f"[WARN] Skipping trait '{trait}' due to error: {exc}")
            continue

        # 1) Compute weights
        weights = compute_weights_from_pvalues(
            variant_ids_core=variant_ids_core,
            df_gwas=df_gwas,
        )
        weights_out_path = os.path.join(out_dir, f"snp_weights_{trait}.npz")
        print(f"[INFO] Saving SNP weights -> {weights_out_path}")
        np.savez_compressed(
            weights_out_path,
            variant_ids=variant_ids_core,
            weights=weights,
        )

        # 2) Major QTL set
        df_major = select_major_qtl(
            df_gwas=df_gwas,
            p_threshold=cfg.GWAS_P_THRESHOLD_MAJOR,
            n_top=cfg.GWAS_N_TOP_MAJOR,
        )
        major_out_path = os.path.join(out_dir, f"major_qtl_snps_{trait}.csv")
        print(f"[INFO] Saving major QTL SNPs -> {major_out_path}")
        df_major.to_csv(major_out_path, index=False)

        summary_records.append(
            {
                "trait": trait,
                "n_core_snps": n_snps,
                "n_gwas_snps": len(df_gwas),
                "n_major_qtl": len(df_major),
            }
        )

    if summary_records:
        summary_df = pd.DataFrame.from_records(summary_records)
        summary_path = os.path.join(out_dir, "gwas_weights_summary.csv")
        print(f"[INFO] Saving summary -> {summary_path}")
        summary_df.to_csv(summary_path, index=False)
        print()
        print(summary_df.to_string(index=False))
    else:
        print("[WARN] No traits successfully processed; nothing to summarise.")

    print()
    print("[DONE] GWAS → SNP weights step complete.")


if __name__ == "__main__":
    main()
