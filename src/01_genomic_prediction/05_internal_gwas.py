# -*- coding: utf-8 -*-
"""
05_internal_gwas.py

Option B: run per-SNP GWAS directly on geno_core.npz / pheno_core.csv and
derive SNP weights and "major QTL" sets for each trait.

For each trait:
  - Simple univariate GWAS (no covariates): y ~ g
  - PC-corrected GWAS: y_res ~ g, where y_res are residuals after regressing y on
    global genotype PCs (using cfg.N_PCS).
  - We store both in an internal GWAS table, but build SNP weights from the
    PC-corrected p-values to reduce confounding by structure.

Outputs (in cfg.GWAS_WEIGHTS_DIR)
---------------------------------
  - internal_gwas_<trait>.csv
      variant_id, beta_raw, r_raw, p_raw, beta_pc, r_pc, p_pc

  - snp_weights_<trait>.npz
      variant_ids : array of variant IDs (same order as geno_core.npz)
      weights     : non-negative weights (float32), same length as variant_ids

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

try:
    from scipy import stats
except ImportError as e:
    raise SystemExit("scipy is required. Install with: pip install scipy") from e

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
except ImportError as e:
    raise SystemExit(
        "scikit-learn is required. Install with: pip install scikit-learn"
    ) from e

import config_idea1 as cfg


# =========================
# Core GWAS helpers
# =========================

def run_univariate_gwas(
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorised simple linear regression GWAS: y ~ g (no other covariates).

    Parameters
    ----------
    X : array-like, shape (n_samples, n_snps)
        Genotype matrix.
    y : array-like, shape (n_samples,)
        Phenotype vector.

    Returns
    -------
    beta : np.ndarray, shape (n_snps,)
        Regression slope for each SNP.
    r : np.ndarray, shape (n_snps,)
        Pearson correlation between SNP and phenotype.
    p : np.ndarray, shape (n_snps,)
        Two-sided p-value for association.
    """
    y = y.astype(float)
    n, m = X.shape

    # Center y
    y_centered = y - np.mean(y)
    var_y = np.sum(y_centered**2)

    # Center X along samples
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean  # (n, m)
    var_X = np.sum(X_centered**2, axis=0)  # (m,)

    # Covariance (without dividing by n-1)
    cov = X_centered.T @ y_centered  # (m,)

    # Slope beta = cov / var(X)
    beta = np.zeros(m, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = cov / var_X

    # Correlation r = cov / sqrt(var_X * var_y)
    r = np.zeros(m, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.sqrt(var_X * var_y)
        r = cov / denom

    # t-statistic for slope in simple regression: t = r * sqrt((n-2)/(1-r^2))
    df = n - 2
    t = np.zeros(m, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = r * np.sqrt(df / np.maximum(1.0 - r**2, 1e-12))

    # Two-sided p-values from t distribution
    p = 2.0 * stats.t.sf(np.abs(t), df=df)

    # Handle monomorphic or degenerate SNPs: var_X == 0 or df <= 0
    bad = (var_X <= 0) | (~np.isfinite(t)) | (df <= 0)
    if np.any(bad):
        beta[bad] = 0.0
        r[bad] = 0.0
        p[bad] = 1.0

    return beta, r, p


def compute_pc_corrected_residuals(
    X: np.ndarray,
    y: np.ndarray,
    n_pcs: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute global genotype PCs and residualise y on those PCs.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_snps)
    y : array-like, shape (n_samples,)
    n_pcs : int
        Number of PCs to use (capped internally).
    random_state : int

    Returns
    -------
    y_res : np.ndarray, shape (n_samples,)
        Residuals after regressing y on PCs.
    PCs : np.ndarray, shape (n_samples, n_pcs_used)
        PC scores (for diagnostics if needed).
    """
    n, m = X.shape
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X)

    n_pcs_use = min(n_pcs, n - 1, m)
    if n_pcs_use <= 0:
        raise RuntimeError(
            f"Cannot compute PCs for PC-correction (n_pcs_use={n_pcs_use})."
        )

    pca = PCA(n_components=n_pcs_use, random_state=random_state)
    PCs = pca.fit_transform(X_scaled)

    lr = LinearRegression()
    lr.fit(PCs, y)
    y_hat = lr.predict(PCs)
    y_res = y - y_hat

    return y_res.astype(float), PCs.astype(float)


def compute_weights_from_pvalues(
    variant_ids_core: np.ndarray,
    p_values: np.ndarray,
    p_floor: float = 1e-300,
) -> np.ndarray:
    """
    Compute non-negative SNP weights from p-values.

    Strategy:
      - w_raw = -log10(p)
      - Clip extremes and normalise to [0, 1]
      - Replace missing with median
      - Ensure all weights >= epsilon
    """
    p = np.array(p_values, dtype=float)
    p = np.clip(p, p_floor, 1.0)
    logp = -np.log10(p)

    if np.all(~np.isfinite(logp)) or np.nanmax(logp) <= 0:
        print("[WARN] All log10(p) values are non-positive or non-finite; using uniform weights.")
        w_obs = np.ones_like(logp)
    else:
        max_logp = np.nanpercentile(logp, 99)
        if max_logp <= 0:
            max_logp = np.nanmax(logp)
        w_obs = np.clip(logp / max_logp, 0.0, 1.0)

    # Replace NaNs with median
    median_w = float(np.nanmedian(w_obs))
    w_obs_clean = np.where(np.isfinite(w_obs), w_obs, median_w)

    # Add epsilon
    eps = 1e-3
    weights = np.maximum(w_obs_clean, eps).astype(np.float32)
    return weights


def select_major_qtl(
    variant_ids: np.ndarray,
    p_values: np.ndarray,
    p_threshold: float,
    n_top: int,
) -> pd.DataFrame:
    """
    Select "major QTL" SNPs based on p-values.

    Criteria:
      - p <= p_threshold OR
      - rank <= n_top (sorted by p)
    """
    df = pd.DataFrame(
        {
            "variant_id": variant_ids.astype(str),
            "p_value": p_values.astype(float),
        }
    )
    df = df.sort_values("p_value", ascending=True).reset_index(drop=True)
    df["rank"] = df.index + 1

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
    pheno_path = os.path.join(cfg.CORE_DATA_DIR, "pheno_core.csv")

    print("=" * 72)
    print("Mango GS – Idea 1: Internal GWAS + SNP weights (Option B)")
    print("=" * 72)
    print(f"[INFO] Geno core:  {geno_path}")
    print(f"[INFO] Pheno core: {pheno_path}")
    print(f"[INFO] Output dir: {out_dir}")
    print()

    if not os.path.exists(geno_path):
        raise FileNotFoundError(
            f"geno_core.npz not found at {geno_path}. "
            "Run 01_build_core_matrices.py first."
        )
    if not os.path.exists(pheno_path):
        raise FileNotFoundError(
            f"pheno_core.csv not found at {pheno_path}. "
            "Run 01_build_core_matrices.py first."
        )

    # Load core data
    npz = np.load(geno_path, allow_pickle=True)
    G = npz["G"]  # (n_samples, n_snps)
    sample_ids = npz["sample_ids"].astype(str)
    variant_ids = npz["variant_ids"].astype(str)
    n_samples, n_snps = G.shape

    pheno_df = pd.read_csv(pheno_path, index_col=0)
    pheno_df.index = pheno_df.index.astype(str)

    print(f"[INFO] Genotype matrix shape: {G.shape} (samples x SNPs)")
    print(f"[INFO] Phenotype table shape: {pheno_df.shape}")
    print()

    # Align phenotype to genotype samples
    common_samples = [sid for sid in sample_ids if sid in pheno_df.index]
    if len(common_samples) == 0:
        raise RuntimeError(
            "No overlapping samples between geno_core and pheno_core."
        )

    sample_idx_map = {sid: i for i, sid in enumerate(sample_ids)}
    geno_idx = [sample_idx_map[sid] for sid in common_samples]

    G_aligned = G[geno_idx, :]
    pheno_aligned = pheno_df.loc[common_samples].copy()

    print(
        f"[INFO] Overlapping samples (geno and pheno): "
        f"{len(common_samples)} / {len(sample_ids)}"
    )
    print()

    traits = [t for t in cfg.TRAITS_DEFAULT if t in pheno_aligned.columns]
    if not traits:
        raise RuntimeError(
            "None of the traits in TRAITS_DEFAULT are present in pheno_core.csv."
        )

    print("[INFO] Traits for internal GWAS:", ", ".join(traits))
    print()

    summary_records: List[Dict[str, object]] = []

    for trait in traits:
        print("-" * 72)
        print(f"[TRAIT] {trait}")

        y_full = pheno_aligned[trait].values.astype(float)
        mask = ~np.isnan(y_full)

        X_trait = G_aligned[mask, :]  # (n_used, n_snps)
        y_trait = y_full[mask]
        n_used = X_trait.shape[0]

        print(f"[INFO] Trait {trait}: using {n_used} samples with non-missing phenotype")

        # Raw GWAS: y ~ g
        print("[STEP] Raw GWAS (no covariates)")
        beta_raw, r_raw, p_raw = run_univariate_gwas(X_trait, y_trait)

        # PC-corrected GWAS: y_res ~ g, where y_res = y - PCs * gamma
        print("[STEP] PC-corrected GWAS (global PCs)")
        try:
            y_res, PCs = compute_pc_corrected_residuals(
                X=X_trait,
                y=y_trait,
                n_pcs=cfg.N_PCS,
                random_state=cfg.RANDOM_STATE,
            )
            beta_pc, r_pc, p_pc = run_univariate_gwas(X_trait, y_res)
        except Exception as exc:
            print(
                f"[WARN] PC-corrected GWAS failed for trait {trait}: {exc}. "
                "Falling back to raw GWAS p-values only."
            )
            beta_pc = np.full_like(beta_raw, np.nan, dtype=float)
            r_pc = np.full_like(r_raw, np.nan, dtype=float)
            p_pc = np.full_like(p_raw, np.nan, dtype=float)

        # Build GWAS table (internal)
        gwas_df = pd.DataFrame(
            {
                "variant_id": variant_ids,
                "beta_raw": beta_raw,
                "r_raw": r_raw,
                "p_raw": p_raw,
                "beta_pc": beta_pc,
                "r_pc": r_pc,
                "p_pc": p_pc,
            }
        )

        gwas_out_path = os.path.join(out_dir, f"internal_gwas_{trait}.csv")
        print(f"[INFO] Saving internal GWAS -> {gwas_out_path}")
        gwas_df.to_csv(gwas_out_path, index=False)

        # Decide which p-values to use for weights: prefer PC-corrected if available
        if np.all(np.isnan(p_pc)):
            print("[WARN] Using raw p-values for weights (PC-corrected p's are all NaN).")
            p_for_weights = p_raw
        else:
            p_for_weights = p_pc

        # SNP weights
        print("[STEP] Computing SNP weights from p-values")
        weights = compute_weights_from_pvalues(
            variant_ids_core=variant_ids,
            p_values=p_for_weights,
        )
        weights_out_path = os.path.join(out_dir, f"snp_weights_{trait}.npz")
        print(f"[INFO] Saving SNP weights -> {weights_out_path}")
        np.savez_compressed(
            weights_out_path,
            variant_ids=variant_ids,
            weights=weights,
        )

        # Major QTL selection
        print("[STEP] Selecting major QTL SNPs")
        df_major = select_major_qtl(
            variant_ids=variant_ids,
            p_values=p_for_weights,
            p_threshold=cfg.GWAS_P_THRESHOLD_MAJOR,
            n_top=cfg.GWAS_N_TOP_MAJOR,
        )
        major_out_path = os.path.join(out_dir, f"major_qtl_snps_{trait}.csv")
        print(f"[INFO] Saving major QTL SNPs -> {major_out_path}")
        df_major.to_csv(major_out_path, index=False)

        summary_records.append(
            {
                "trait": trait,
                "n_snps": n_snps,
                "n_samples_used": n_used,
                "n_major_qtl": len(df_major),
            }
        )

    if summary_records:
        summary_df = pd.DataFrame.from_records(summary_records)
        summary_path = os.path.join(out_dir, "internal_gwas_summary.csv")
        print(f"[INFO] Saving summary -> {summary_path}")
        summary_df.to_csv(summary_path, index=False)
        print()
        print(summary_df.to_string(index=False))
    else:
        print("[WARN] No traits processed; nothing to summarise.")

    print()
    print("[DONE] Internal GWAS + SNP weights complete.")


if __name__ == "__main__":
    main()
