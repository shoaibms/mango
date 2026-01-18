#!/usr/bin/env python
r"""
06_feature_importance.py

GWAS + ML feature importance + SNP->gene mapping for Mango GS - Idea 2.

This script assumes that Idea 1 and Idea 2 core scripts have already run and that
the following files exist:

Core data (Idea 2):
  - X_full.npy          (genotype matrix, samples x SNPs)
  - y_traits.csv        (phenotypes, one row per sample)
  - samples.csv         (sample_id in the same row order as X_full)
  - pcs.csv             (PC1..PCn for each sample_id)

Core data (Idea 1):
  - geno_core.npz       (contains at least 'G', 'sample_ids', 'variant_ids')

Reference annotation:
  - genomic.gff         (NCBI mango genome annotation; gene features)
                        NOTE: Must use NCBI GFF to match VCF chromosome IDs (NC_058xxx)

Outputs:
  - gwas/gwas_results_<trait>.csv
  - gwas/gwas_topk_<trait>.csv
  - importance/xgb_importance_<trait>.csv
  - importance/rf_importance_<trait>.csv
  - importance/gwas_ml_rank_comparison_<trait>.csv
  - genes/candidate_snps_<trait>.csv
  - genes/candidate_snps_with_genes_<trait>.csv
"""

import argparse
import gzip
import math
import os
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import pandas as pd
except ImportError as e:
    raise SystemExit("pandas is required. Install with: pip install pandas") from e

# Optional: t-distribution (for p-values). Fallback to normal if SciPy missing.
try:
    from scipy import stats as sp_stats
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# Tree models for importance
try:
    from xgboost import XGBRegressor
except ImportError as e:
    raise SystemExit("xgboost is required. Install with: pip install xgboost") from e

try:
    from sklearn.ensemble import RandomForestRegressor
except ImportError as e:
    raise SystemExit(
        "scikit-learn is required. Install with: pip install scikit-learn"
    ) from e


# =========================
# DEFAULT PATHS / PARAMS
# =========================

# Core Idea 2 outputs
DEFAULT_X_PATH = r"C:\Users\ms\Desktop\mango\output\idea_2\core_ml\X_full.npy"
DEFAULT_Y_PATH = r"C:\Users\ms\Desktop\mango\output\idea_2\core_ml\y_traits.csv"
DEFAULT_SAMPLES_PATH = r"C:\Users\ms\Desktop\mango\output\idea_2\core_ml\samples.csv"
DEFAULT_PCS_PATH = r"C:\Users\ms\Desktop\mango\output\idea_2\core_ml\pcs.csv"

# Core Idea 1 geno data (for variant_ids)
DEFAULT_GENO_CORE_PATH = r"C:\Users\ms\Desktop\mango\output\idea_1\core_data\geno_core.npz"

# Annotation
DEFAULT_GFF_PATH = r"C:\Users\ms\Desktop\mango\data\ncbi\genomic.gff"

# Output dirs
DEFAULT_GWAS_OUTDIR = r"C:\Users\ms\Desktop\mango\output\idea_2\gwas"
DEFAULT_IMPORTANCE_OUTDIR = r"C:\Users\ms\Desktop\mango\output\idea_2\importance"
DEFAULT_GENES_OUTDIR = r"C:\Users\ms\Desktop\mango\output\idea_2\genes"

RANDOM_STATE = 42


# =========================
# UTILS
# =========================

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_core_matrices(
    X_path: str,
    y_path: str,
    samples_path: str,
    pcs_path: str,
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Load X_full, y_traits, samples, pcs and ensure alignment.
    """
    if not os.path.isfile(X_path):
        raise FileNotFoundError(f"X_full.npy not found: {X_path}")
    if not os.path.isfile(y_path):
        raise FileNotFoundError(f"y_traits.csv not found: {y_path}")
    if not os.path.isfile(samples_path):
        raise FileNotFoundError(f"samples.csv not found: {samples_path}")

    X = np.load(X_path)
    y_df = pd.read_csv(y_path, index_col=0)
    samples_df = pd.read_csv(samples_path)

    if "sample_id" not in samples_df.columns:
        raise RuntimeError("samples.csv must have a 'sample_id' column.")

    sample_ids = samples_df["sample_id"].astype(str).tolist()

    # PCs (optional but expected)
    if os.path.isfile(pcs_path):
        pcs_df = pd.read_csv(pcs_path, index_col=0)
        pcs_df.index = pcs_df.index.astype(str)
        pcs_df = pcs_df.reindex(sample_ids)
    else:
        pcs_df = pd.DataFrame(index=sample_ids)

    # Reindex y_df to sample_ids order
    y_df.index = y_df.index.astype(str)
    y_df = y_df.reindex(sample_ids)

    print(f"[INFO] Loaded X: {X.shape}")
    print(f"[INFO] Loaded y_traits: {y_df.shape}")
    print(f"[INFO] Loaded PCs: {pcs_df.shape}")
    print(f"[INFO] Samples: {len(sample_ids)}")

    return X, y_df, pcs_df, sample_ids


def load_variant_ids(geno_core_path: str) -> np.ndarray:
    """
    Load variant_ids from geno_core.npz (from Idea 1).
    """
    if not os.path.isfile(geno_core_path):
        raise FileNotFoundError(f"geno_core.npz not found: {geno_core_path}")

    npz = np.load(geno_core_path, allow_pickle=True)
    variant_ids = npz["variant_ids"].astype(str)
    print(f"[INFO] Loaded {len(variant_ids)} variant IDs from geno_core.npz")
    return variant_ids


def parse_variant_id(vid: str) -> Tuple[str, int]:
    """
    Parse a variant ID like 'NC_058137.1:12345' into (chrom, pos).
    Returns (chrom, pos) or (vid, -1) if parsing fails.
    """
    if ":" in vid:
        parts = vid.rsplit(":", 1)
        try:
            return parts[0], int(parts[1])
        except ValueError:
            return vid, -1
    return vid, -1


# =========================
# GWAS (simple univariate)
# =========================

def run_gwas_univariate(
    X: np.ndarray,
    y: np.ndarray,
    variant_ids: np.ndarray,
) -> pd.DataFrame:
    """
    Simple univariate GWAS: for each SNP, regress y ~ g.

    Returns a DataFrame with columns:
      snp_id, chrom, pos, beta, se, t, p, log10p
    """
    n_samples, n_snps = X.shape
    y = y.astype(float)

    # Center y
    y_mean = np.mean(y)
    y_centered = y - y_mean
    var_y = np.sum(y_centered ** 2)

    # Center X
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    var_X = np.sum(X_centered ** 2, axis=0)

    # Covariance
    cov = X_centered.T @ y_centered

    # Beta = cov / var(X)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = cov / var_X

    # Correlation
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.sqrt(var_X * var_y)
        r = cov / denom

    # t-statistic
    df = n_samples - 2
    with np.errstate(divide="ignore", invalid="ignore"):
        t = r * np.sqrt(df / np.maximum(1.0 - r ** 2, 1e-12))

    # p-value
    if HAVE_SCIPY:
        p = 2.0 * sp_stats.t.sf(np.abs(t), df=df)
    else:
        # Fallback: normal approximation
        p = 2.0 * (1.0 - 0.5 * (1.0 + np.sign(t) * (1.0 - np.exp(-0.5 * t ** 2))))

    # Handle monomorphic or degenerate SNPs
    bad = (var_X <= 0) | (~np.isfinite(t)) | (df <= 0)
    beta[bad] = 0.0
    r[bad] = 0.0
    p[bad] = 1.0

    # Standard error
    with np.errstate(divide="ignore", invalid="ignore"):
        se = beta / t
    se[bad] = np.nan

    # log10(p)
    with np.errstate(divide="ignore", invalid="ignore"):
        log10p = -np.log10(np.clip(p, 1e-300, 1.0))

    # Parse chrom/pos from variant_ids
    chroms = []
    poses = []
    for vid in variant_ids:
        c, pos = parse_variant_id(vid)
        chroms.append(c)
        poses.append(pos)

    df_out = pd.DataFrame(
        {
            "snp_id": variant_ids,
            "chrom": chroms,
            "pos": poses,
            "beta": beta,
            "se": se,
            "t": t,
            "p": p,
            "log10p": log10p,
        }
    )

    return df_out


# =========================
# Tree-based feature importance
# =========================

def compute_tree_importance(
    model,
    variant_ids: np.ndarray,
    chroms: List[str],
    poses: List[int],
) -> pd.DataFrame:
    """
    Extract feature importances from a fitted tree model.
    """
    importances = model.feature_importances_

    df = pd.DataFrame(
        {
            "snp_id": variant_ids,
            "chrom": chroms,
            "pos": poses,
            "importance": importances,
        }
    )
    return df


# =========================
# Gene mapping (simple GFF parsing)
# =========================

def load_gene_annotation(gff_path: str) -> pd.DataFrame:
    """
    Load gene features from a GFF3 file.

    Returns a DataFrame with columns: chrom, start, end, gene_id, gene_name
    """
    if not os.path.isfile(gff_path):
        print(f"[WARN] GFF file not found: {gff_path}. Skipping gene mapping.")
        return pd.DataFrame()

    # Check if gzipped
    opener = gzip.open if gff_path.endswith(".gz") else open
    mode = "rt" if gff_path.endswith(".gz") else "r"

    records = []

    with opener(gff_path, mode) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue

            ftype = parts[2]
            if ftype != "gene":
                continue

            chrom = parts[0]
            start = int(parts[3])
            end = int(parts[4])
            attrs = parts[8]

            # Parse attributes
            gene_id = None
            gene_name = None
            for attr in attrs.split(";"):
                if "=" in attr:
                    key, val = attr.split("=", 1)
                    key = key.strip()
                    val = val.strip()
                    if key == "ID":
                        gene_id = val
                    elif key == "Name":
                        gene_name = val

            if gene_id is None:
                gene_id = f"{chrom}:{start}-{end}"

            records.append(
                {
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                    "gene_id": gene_id,
                    "gene_name": gene_name if gene_name else gene_id,
                }
            )

    df = pd.DataFrame(records)
    print(f"[INFO] Loaded {len(df)} genes from GFF")
    return df


def map_variant_to_nearest_gene(
    chrom: str,
    pos: int,
    gene_df: pd.DataFrame,
) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """
    Find the nearest gene to a given position on a chromosome.

    Returns (gene_id, gene_name, distance) or (None, None, None) if no genes on chrom.
    """
    if gene_df.empty:
        return None, None, None

    chrom_genes = gene_df[gene_df["chrom"] == chrom]
    if chrom_genes.empty:
        return None, None, None

    # Check overlap first
    overlap = chrom_genes[(chrom_genes["start"] <= pos) & (chrom_genes["end"] >= pos)]
    if not overlap.empty:
        g = overlap.iloc[0]
        return g["gene_id"], g["gene_name"], 0

    # Find nearest
    starts = chrom_genes["start"].values
    ends = chrom_genes["end"].values

    dist_to_start = np.abs(pos - starts)
    dist_to_end = np.abs(pos - ends)
    min_dist = np.minimum(dist_to_start, dist_to_end)

    idx = np.argmin(min_dist)
    g = chrom_genes.iloc[idx]

    return g["gene_id"], g["gene_name"], int(min_dist[idx])


# =========================
# MAIN
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GWAS + ML feature importance + gene mapping for Mango GS Idea 2."
    )
    parser.add_argument(
        "--X-path",
        type=str,
        default=DEFAULT_X_PATH,
        help=f"Path to X_full.npy (default: {DEFAULT_X_PATH})",
    )
    parser.add_argument(
        "--y-path",
        type=str,
        default=DEFAULT_Y_PATH,
        help=f"Path to y_traits.csv (default: {DEFAULT_Y_PATH})",
    )
    parser.add_argument(
        "--samples-path",
        type=str,
        default=DEFAULT_SAMPLES_PATH,
        help=f"Path to samples.csv (default: {DEFAULT_SAMPLES_PATH})",
    )
    parser.add_argument(
        "--pcs-path",
        type=str,
        default=DEFAULT_PCS_PATH,
        help=f"Path to pcs.csv (default: {DEFAULT_PCS_PATH})",
    )
    parser.add_argument(
        "--geno-core-path",
        type=str,
        default=DEFAULT_GENO_CORE_PATH,
        help=f"Path to geno_core.npz (default: {DEFAULT_GENO_CORE_PATH})",
    )
    parser.add_argument(
        "--gff-path",
        type=str,
        default=DEFAULT_GFF_PATH,
        help=f"Path to GFF annotation (default: {DEFAULT_GFF_PATH})",
    )
    parser.add_argument(
        "--gwas-outdir",
        type=str,
        default=DEFAULT_GWAS_OUTDIR,
        help=f"Output dir for GWAS results (default: {DEFAULT_GWAS_OUTDIR})",
    )
    parser.add_argument(
        "--importance-outdir",
        type=str,
        default=DEFAULT_IMPORTANCE_OUTDIR,
        help=f"Output dir for feature importances (default: {DEFAULT_IMPORTANCE_OUTDIR})",
    )
    parser.add_argument(
        "--genes-outdir",
        type=str,
        default=DEFAULT_GENES_OUTDIR,
        help=f"Output dir for gene mappings (default: {DEFAULT_GENES_OUTDIR})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of top SNPs to keep per trait (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_STATE,
        help=f"Random seed (default: {RANDOM_STATE})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 72)
    print("Mango GS - Idea 2: GWAS + ML feature importance + gene mapping")
    print("=" * 72)
    print(f"[INFO] X_full:       {args.X_path}")
    print(f"[INFO] y_traits:     {args.y_path}")
    print(f"[INFO] samples:      {args.samples_path}")
    print(f"[INFO] geno_core:    {args.geno_core_path}")
    print(f"[INFO] GFF:          {args.gff_path}")
    print(f"[INFO] Top-K:        {args.top_k}")
    print("")

    safe_mkdir(args.gwas_outdir)
    safe_mkdir(args.importance_outdir)
    safe_mkdir(args.genes_outdir)

    # 1) Load data
    X, y_df, pcs_df, sample_ids = load_core_matrices(
        X_path=args.X_path,
        y_path=args.y_path,
        samples_path=args.samples_path,
        pcs_path=args.pcs_path,
    )

    variant_ids = load_variant_ids(args.geno_core_path)

    if X.shape[1] != len(variant_ids):
        raise RuntimeError(
            f"Number of SNPs in X ({X.shape[1]}) does not match variant_ids ({len(variant_ids)})"
        )

    # Parse chrom/pos once
    chroms = []
    poses = []
    for vid in variant_ids:
        c, pos = parse_variant_id(vid)
        chroms.append(c)
        poses.append(pos)

    # 2) Load gene annotation
    gene_df = load_gene_annotation(args.gff_path)

    traits = list(y_df.columns)
    print(f"[INFO] Traits: {traits}")
    print("")

    # 3) Loop over traits
    for trait in traits:
        print(f"[TRAIT] {trait}")

        y_full = y_df[trait].to_numpy(dtype=float)
        mask = ~np.isnan(y_full)
        n_used = int(mask.sum())

        if n_used < 10:
            print(f"  [WARN] Only {n_used} non-NaN samples for trait '{trait}'. Skipping.")
            continue

        X_t = X[mask, :]
        y_t = y_full[mask]

        # A. GWAS
        print("  [Step 1] Running univariate GWAS...")
        gwas_res = run_gwas_univariate(X_t, y_t, variant_ids)
        out_gwas = os.path.join(args.gwas_outdir, f"gwas_results_{trait}.csv")
        gwas_res.to_csv(out_gwas, index=False)

        # Top-K by p-value
        topk = gwas_res.nsmallest(args.top_k, "p")
        out_topk = os.path.join(args.gwas_outdir, f"gwas_topk_{trait}.csv")
        topk.to_csv(out_topk, index=False)

        # B. Tree-based importance
        print("  [Step 2] Training XGBoost and RF for feature importance...")

        xgb_model = XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.7,
            colsample_bytree=0.3,
            reg_lambda=1.0,
            random_state=args.seed,
            n_jobs=-1,
            tree_method="hist",
        )
        xgb_model.fit(X_t, y_t)
        xgb_imp = compute_tree_importance(xgb_model, variant_ids, chroms, poses)
        out_xgb = os.path.join(args.importance_outdir, f"xgb_importance_{trait}.csv")
        xgb_imp.sort_values("importance", ascending=False).to_csv(out_xgb, index=False)

        rf_model = RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            max_features="sqrt",
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=args.seed,
            n_jobs=-1,
        )
        rf_model.fit(X_t, y_t)
        rf_imp = compute_tree_importance(rf_model, variant_ids, chroms, poses)
        out_rf = os.path.join(args.importance_outdir, f"rf_importance_{trait}.csv")
        rf_imp.sort_values("importance", ascending=False).to_csv(out_rf, index=False)

        # C. Compare Ranks
        # Merge on snp_id
        rank_df = gwas_res[["snp_id", "p"]].copy()
        rank_df["gwas_rank"] = rank_df["p"].rank()

        x_df = xgb_imp[["snp_id", "importance"]].rename(columns={"importance": "xgb_imp"})
        x_df["xgb_rank"] = x_df["xgb_imp"].rank(ascending=False)

        r_df = rf_imp[["snp_id", "importance"]].rename(columns={"importance": "rf_imp"})
        r_df["rf_rank"] = r_df["rf_imp"].rank(ascending=False)

        merged = rank_df.merge(x_df, on="snp_id").merge(r_df, on="snp_id")
        # Combined rank (average of ranks)
        merged["avg_rank"] = (merged["gwas_rank"] + merged["xgb_rank"] + merged["rf_rank"]) / 3.0

        out_rank = os.path.join(args.importance_outdir, f"gwas_ml_rank_comparison_{trait}.csv")
        merged.sort_values("avg_rank").to_csv(out_rank, index=False)

        # D. Gene Mapping (Top candidates)
        if not gene_df.empty:
            print("  [Step 3] Mapping top variants to genes...")
            # Take top 100 by avg rank
            top_candidates = merged.sort_values("avg_rank").head(100)

            res_genes = []
            for _, row in top_candidates.iterrows():
                sid = row["snp_id"]
                # Look up chrom/pos from gwas_res (it's already there)
                g_row = gwas_res.loc[gwas_res["snp_id"] == sid].iloc[0]
                chrom_ = g_row["chrom"]
                pos_ = g_row["pos"]

                gid, gname, dist = map_variant_to_nearest_gene(chrom_, pos_, gene_df)
                res_genes.append({
                    "snp_id": sid,
                    "chrom": chrom_,
                    "pos": pos_,
                    "gwas_p": row["p"],
                    "xgb_imp": row["xgb_imp"],
                    "rf_imp": row["rf_imp"],
                    "avg_rank": row["avg_rank"],
                    "gene_id": gid,
                    "gene_name": gname,
                    "dist_to_gene": dist
                })

            cand_df = pd.DataFrame(res_genes)
            out_genes = os.path.join(args.genes_outdir, f"candidate_snps_with_genes_{trait}.csv")
            cand_df.to_csv(out_genes, index=False)

    print("")
    print("[OK] GWAS + feature importance + gene mapping completed.")


if __name__ == "__main__":
    main()
