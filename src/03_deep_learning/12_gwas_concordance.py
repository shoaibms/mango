import os
import json
import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================

BASE_DIR = r"C:\Users\ms\Desktop\mango\output\idea_3"

TENSOR_DIR = os.path.join(BASE_DIR, "tensors")
INTERP_DIR = os.path.join(BASE_DIR, "interpretation")
SAL_DIR = os.path.join(INTERP_DIR, "saliency")
OUT_DIR = os.path.join(INTERP_DIR, "ai_vs_gwas")

# Saliency matrix (from 06_ai_saliency_multitrait.py)
SAL_MATRIX_PATH = os.path.join(SAL_DIR, "saliency_matrix_block-raw.csv")

# Feature map (from 02_cnn_tensor_builder.py)
FEATURE_MAP_PATH = os.path.join(TENSOR_DIR, "feature_map.tsv")

# Trait names (for 'raw' block)
Y_RAW_TRAITS_PATH = os.path.join(TENSOR_DIR, "y_raw_traits.json")

# ======= GWAS CONFIG (ADAPT THESE TO YOUR FILE) =======
# Expected: one row per SNP, with:
#   - a SNP identifier column, e.g. 'snp_id'
#   - p-value columns per trait, e.g. 'p_FBC', 'p_AFW', ...
#   - optional effect size columns, e.g. 'beta_FBC', 'beta_AFW', ...
GWAS_SUMMARY_PATH = r"C:\Users\ms\Desktop\mango\output\idea_1\gwas\gwas_summary_by_trait.csv"
GWAS_SNP_COL = "snp_id"
GWAS_P_PREFIX = "p_"       # e.g. p_FBC, p_AFW, ...
GWAS_BETA_PREFIX = "beta_" # optional: beta_FBC, beta_AFW, ...

# Minimum number of overlapping SNPs required to compute correlations
MIN_SNPS_FOR_STATS = 50


# =========================
# UTILITIES
# =========================

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_saliency_and_traits():
    """Load multi-trait saliency matrix and raw trait names."""
    if not os.path.exists(SAL_MATRIX_PATH):
        raise FileNotFoundError(f"Saliency matrix not found at:\n  {SAL_MATRIX_PATH}")
    sal = pd.read_csv(SAL_MATRIX_PATH)
    print(f"[INFO] Loaded saliency matrix: {sal.shape[0]} SNPs x {sal.shape[1]} cols")

    if not os.path.exists(Y_RAW_TRAITS_PATH):
        raise FileNotFoundError(f"y_raw_traits.json not found at:\n  {Y_RAW_TRAITS_PATH}")
    with open(Y_RAW_TRAITS_PATH, "r", encoding="utf-8") as f:
        raw_traits = json.load(f)
    print(f"[INFO] Raw traits: {raw_traits}")

    return sal, raw_traits


def load_feature_map():
    """Load feature_map.tsv (if present) to get SNP IDs / coordinates."""
    if not os.path.exists(FEATURE_MAP_PATH):
        print("[WARN] feature_map.tsv not found; SNP IDs will be from saliency matrix only.")
        return None

    fmap = pd.read_csv(FEATURE_MAP_PATH, sep="\t")
    print(f"[INFO] Loaded feature_map: {fmap.shape[0]} rows x {fmap.shape[1]} cols")
    return fmap


def attach_snp_ids(sal: pd.DataFrame, fmap: pd.DataFrame | None) -> pd.DataFrame:
    """
    Ensure saliency matrix has an 'snp_id' column.
    We rely on 'feature_index' aligning with feature_map row order.
    """
    if "feature_index" not in sal.columns:
        raise ValueError("Saliency matrix must contain a 'feature_index' column.")

    df = sal.copy()

    if fmap is not None and "snp_id" in fmap.columns:
        # Align by row index; assume feature_index is 0..n-1 in the same order
        # Drop any existing 'snp_id' in sal to avoid confusion
        if "snp_id" in df.columns:
            df = df.drop(columns=["snp_id"])
        # Just attach fmap['snp_id'] as a column, relying on index alignment
        df["snp_id"] = fmap["snp_id"].values
        print("[INFO] Attached 'snp_id' from feature_map to saliency matrix.")
    else:
        if "snp_id" not in df.columns:
            df["snp_id"] = df["feature_index"].apply(lambda i: f"idx_{i}")
        print("[INFO] Using synthetic SNP IDs from feature_index (no feature_map snp_id).")

    return df


def load_gwas_summary():
    """Load GWAS summary table."""
    if not os.path.exists(GWAS_SUMMARY_PATH):
        raise FileNotFoundError(f"GWAS summary not found at:\n  {GWAS_SUMMARY_PATH}")
    gwas = pd.read_csv(GWAS_SUMMARY_PATH)
    print(f"[INFO] Loaded GWAS summary: {gwas.shape[0]} SNPs x {gwas.shape[1]} cols")

    if GWAS_SNP_COL not in gwas.columns:
        raise ValueError(
            f"GWAS summary is missing SNP ID column '{GWAS_SNP_COL}'. "
            f"Available columns: {list(gwas.columns)}"
        )
    return gwas


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    """NaN-safe Pearson correlation."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    x_m = x[mask]
    y_m = y[mask]
    if np.std(x_m) == 0 or np.std(y_m) == 0:
        return np.nan
    return float(np.corrcoef(x_m, y_m)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """
    Spearman correlation via rank transform (using pandas rank, no SciPy).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    x_m = x[mask]
    y_m = y[mask]
    x_rank = pd.Series(x_m).rank(method="average").to_numpy()
    y_rank = pd.Series(y_m).rank(method="average").to_numpy()
    if np.std(x_rank) == 0 or np.std(y_rank) == 0:
        return np.nan
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


# =========================
# MAIN
# =========================

def main():
    print("=" * 72)
    print("Mango GS – Idea 3: AI vs GWAS concordance")
    print(" (09_ai_vs_gwas_concordance.py)")
    print("=" * 72)

    safe_mkdir(OUT_DIR)

    # 1) Load saliency + traits
    sal_raw, raw_traits = load_saliency_and_traits()

    # 2) Load feature_map and attach SNP IDs to saliency
    fmap = load_feature_map()
    sal = attach_snp_ids(sal_raw, fmap)

    # 3) Load GWAS summary and merge
    gwas = load_gwas_summary()

    # Keep only relevant columns from saliency side
    # (feature_index, snp_id, saliency_*_norm/raw, plus any coords)
    merge_cols = [c for c in sal.columns if c in ("feature_index", "snp_id") or c.startswith("saliency_")]
    # If feature_map had extra positional info and got carried into sal, they’re already in sal columns.

    sal_for_merge = sal[merge_cols].copy()

    merged = pd.merge(
        sal_for_merge,
        gwas,
        how="inner",
        left_on="snp_id",
        right_on=GWAS_SNP_COL,
    )

    print(
        f"[INFO] Merged saliency + GWAS on '{GWAS_SNP_COL}': "
        f"{merged.shape[0]} SNPs overlap."
    )
    if merged.shape[0] < MIN_SNPS_FOR_STATS:
        print(
            f"[WARN] Overlap < MIN_SNPS_FOR_STATS ({MIN_SNPS_FOR_STATS}). "
            f"Correlations may be unstable."
        )

    # 4) For each trait, compute concordance between AI saliency and GWAS stats
    summary_records = []

    for trait in raw_traits:
        sal_col_norm = f"saliency_{trait}_norm"
        sal_col_raw = f"saliency_{trait}_raw"

        if sal_col_norm in merged.columns:
            sal_col = sal_col_norm
        elif sal_col_raw in merged.columns:
            sal_col = sal_col_raw
        else:
            print(f"[WARN] No saliency column for trait '{trait}'. Skipping.")
            continue

        p_col = GWAS_P_PREFIX + trait  # e.g. p_FBC
        beta_col = GWAS_BETA_PREFIX + trait  # e.g. beta_FBC

        if p_col not in merged.columns:
            print(
                f"[WARN] GWAS summary has no p-value column '{p_col}' for trait '{trait}'. Skipping."
            )
            continue

        df_trait = merged[[sal_col, p_col]].copy()

        # Filter out invalid p-values
        df_trait = df_trait.replace([np.inf, -np.inf], np.nan)
        df_trait = df_trait.dropna(subset=[sal_col, p_col])
        df_trait = df_trait[df_trait[p_col] > 0]

        n_used = df_trait.shape[0]
        if n_used < MIN_SNPS_FOR_STATS:
            print(
                f"[WARN] Trait '{trait}': only {n_used} SNPs with valid saliency + p; skipping stats."
            )
            continue

        sal_vals = df_trait[sal_col].to_numpy()
        p_vals = df_trait[p_col].to_numpy()

        # Clip to avoid log(0) and crazy underflow
        p_vals = np.clip(p_vals, 1e-300, 1.0)
        neglog10p = -np.log10(p_vals)

        pearson_p = pearson_corr(sal_vals, neglog10p)
        spearman_p = spearman_corr(sal_vals, neglog10p)

        # Effect-size concordance if available
        pearson_beta = np.nan
        spearman_beta = np.nan
        if beta_col in merged.columns:
            df_beta = merged[[sal_col, beta_col]].copy()
            df_beta = df_beta.replace([np.inf, -np.inf], np.nan)
            df_beta = df_beta.dropna(subset=[sal_col, beta_col])

            if df_beta.shape[0] >= MIN_SNPS_FOR_STATS:
                sal_vals_beta = df_beta[sal_col].to_numpy()
                abs_beta = np.abs(df_beta[beta_col].to_numpy())
                pearson_beta = pearson_corr(sal_vals_beta, abs_beta)
                spearman_beta = spearman_corr(sal_vals_beta, abs_beta)
            else:
                print(
                    f"[INFO] Trait '{trait}': insufficient SNPs ({df_beta.shape[0]}) "
                    f"with valid saliency + beta for effect-size concordance."
                )

        summary_records.append(
            {
                "trait": trait,
                "saliency_col_used": sal_col,
                "gwas_p_col": p_col,
                "gwas_beta_col": beta_col if beta_col in merged.columns else None,
                "n_overlap_snps": int(merged.shape[0]),
                "n_used_for_p": int(n_used),
                "pearson_sal_vs_neglog10p": pearson_p,
                "spearman_sal_vs_neglog10p": spearman_p,
                "pearson_sal_vs_absbeta": pearson_beta,
                "spearman_sal_vs_absbeta": spearman_beta,
            }
        )

        # 5) Also write a trait-specific merged table for plotting
        trait_keep_cols = [
            "feature_index",
            "snp_id",
            sal_col,
            p_col,
        ]
        if beta_col in merged.columns:
            trait_keep_cols.append(beta_col)

        # Keep any positional columns if present (chr/pos)
        for extra in ("chr", "chrom", "pos", "position"):
            if extra in merged.columns and extra not in trait_keep_cols:
                trait_keep_cols.append(extra)

        trait_df = merged[trait_keep_cols].copy()
        trait_df["neglog10p"] = -np.log10(trait_df[p_col].replace(0, np.nan))

        out_trait = os.path.join(
            OUT_DIR,
            f"ai_gwas_merged_trait-{trait}.csv",
        )
        trait_df.to_csv(out_trait, index=False)
        print(f"[OK] Saved AI+GWAS merged table for trait '{trait}' to:\n  {out_trait}")

    # 6) Save summary table
    if summary_records:
        df_summary = pd.DataFrame(summary_records)
        out_summary = os.path.join(OUT_DIR, "ai_gwas_concordance_summary.csv")
        df_summary.to_csv(out_summary, index=False)
        print(f"[OK] Saved AI vs GWAS concordance summary to:\n  {out_summary}")
    else:
        print("[WARN] No summary records generated; check column names and GWAS config.")

    print("[DONE] 09_ai_vs_gwas_concordance.py complete.")


if __name__ == "__main__":
    main()
