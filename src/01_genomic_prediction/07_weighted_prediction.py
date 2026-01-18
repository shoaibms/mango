# -*- coding: utf-8 -*-
"""
07_weighted_prediction.py

Mango GS – Idea 1: Compare baseline ridge, GWAS-weighted ridge, and
"major QTL + background ridge" models under standard K-fold CV.

Models
------
For each trait:

  1. baseline
       - Ridge regression on standardised whole-genome SNPs.
       - Scenarios:
           * no_pc        : y ~ polygenic (ridge)
           * pc_corrected : y_res ~ polygenic (ridge), where y_res is residual
                            after regressing y on global PCs.

  2. weighted
       - Same as baseline, but SNPs are re-weighted using GWAS-derived weights
         from 04_internal_gwas_and_weights.py.
       - Implementation: after standardisation, each SNP column is multiplied
         by sqrt(weight_i) (equivalent to a weighted GRM).

  3. major_plus_bg
       - Two-step model separating large-effect and polygenic background:
           * Step 1: y ~ PCs (+ major SNPs) by OLS
           * Step 2: residuals are modelled by ridge on background SNPs.
         Prediction on test = fixed part (PCs + major SNPs) + predicted residuals.
       - Scenarios:
           * no_pc        : fixed effects = major SNPs only
           * pc_corrected : fixed effects = PCs + major SNPs

Inputs
------
From previous steps:

  - cfg.CORE_DATA_DIR / "geno_core.npz"
      G           : (n_samples, n_snps) float32
      sample_ids  : array of sample IDs
      variant_ids : array of variant IDs (same as internal GWAS)

  - cfg.CORE_DATA_DIR / "pheno_core.csv"
      Index: sample ID
      Columns: traits (FBC, FF, AFW, TSS, TC)

  - cfg.GWAS_WEIGHTS_DIR / "snp_weights_<trait>.npz"
      variant_ids : must match geno_core.npz order
      weights     : non-negative weights

  - cfg.GWAS_WEIGHTS_DIR / "major_qtl_snps_<trait>.csv"
      variant_id, p_value, rank

Outputs
-------
In cfg.CV_GWAS_INTEGRATION_DIR:

  - cv_gwas_integration_results.csv
      trait,model_type,scenario,fold,r

  - cv_gwas_integration_summary.csv
      trait,model_type,scenario,mean_r,n_folds

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
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge, LinearRegression
    from sklearn.decomposition import PCA
except ImportError as e:
    raise SystemExit(
        "scikit-learn is required. Install with: pip install scikit-learn"
    ) from e

import config_idea1 as cfg


# =========================
# Shared CV helpers
# =========================

def get_kfold_splits(n_samples: int, n_splits: int, random_state: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate K-fold train/test splits for a given number of samples.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(kf.split(np.arange(n_samples)))


def ridge_cv_baseline_or_weighted(
    X: np.ndarray,
    y: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    weights_sqrt: np.ndarray | None,
    use_pc_correction: bool,
    n_pcs: int,
    random_state: int,
) -> List[float]:
    """
    Ridge CV for baseline (weights_sqrt=None) or GWAS-weighted model.

    Parameters
    ----------
    X : (n_samples, n_snps)
    y : (n_samples,)
    splits : list of (train_idx, test_idx)
    weights_sqrt : None or (n_snps,)
        If None, unweighted. Otherwise each SNP column is multiplied by sqrt(weight_i)
        after standardisation.
    use_pc_correction : bool
    n_pcs : int
    random_state : int

    Returns
    -------
    r_values : list of float (per fold)
    """
    r_values: List[float] = []

    model_label = "weighted" if weights_sqrt is not None else "baseline"
    if use_pc_correction:
        print(f"[INFO] {model_label} ridge with PC correction (n_pcs={n_pcs})")
    else:
        print(f"[INFO] {model_label} ridge without PC correction")

    n_samples, n_snps = X.shape

    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train = X[train_idx, :]
        X_test = X[test_idx, :]
        y_train = y[train_idx]
        y_test = y[test_idx]

        if np.nanstd(y_train) == 0.0:
            print(f"  [WARN] Fold {fold_idx}: y_train has zero variance; skipping.")
            r_values.append(np.nan)
            continue

        if not use_pc_correction:
            # Standardisation
            scaler_X = StandardScaler(with_mean=True, with_std=True)
            Z_train = scaler_X.fit_transform(X_train)
            Z_test = scaler_X.transform(X_test)

            # Apply weights if provided
            if weights_sqrt is not None:
                Z_train = Z_train * weights_sqrt
                Z_test = Z_test * weights_sqrt

            model = Ridge(alpha=1.0, random_state=random_state)
            model.fit(Z_train, y_train)
            y_pred = model.predict(Z_test)

            if np.nanstd(y_test) == 0.0 or np.nanstd(y_pred) == 0.0:
                r = np.nan
            else:
                r = np.corrcoef(y_test, y_pred)[0, 1]

            print(f"  Fold {fold_idx}: r = {r:.3f} ({model_label}, no PC)")

        else:
            # 1) Compute PCs for PC correction (on whole X)
            scaler_pc = StandardScaler(with_mean=True, with_std=True)
            X_train_pc = scaler_pc.fit_transform(X_train)
            X_test_pc = scaler_pc.transform(X_test)

            max_pcs = min(n_pcs, X_train_pc.shape[0] - 1, X_train_pc.shape[1])
            if max_pcs <= 0:
                print(
                    f"  [WARN] Fold {fold_idx}: cannot compute PCs (max_pcs={max_pcs}); skipping."
                )
                r_values.append(np.nan)
                continue

            pca = PCA(n_components=max_pcs, random_state=random_state)
            PCs_train = pca.fit_transform(X_train_pc)
            PCs_test = pca.transform(X_test_pc)

            lr_pc = LinearRegression()
            lr_pc.fit(PCs_train, y_train)
            y_hat_train_pc = lr_pc.predict(PCs_train)
            y_hat_test_pc = lr_pc.predict(PCs_test)

            y_train_res = y_train - y_hat_train_pc
            y_test_res = y_test - y_hat_test_pc

            # 2) Ridge on residuals, with possible SNP weights
            scaler_X2 = StandardScaler(with_mean=True, with_std=True)
            Z_train = scaler_X2.fit_transform(X_train)
            Z_test = scaler_X2.transform(X_test)

            if weights_sqrt is not None:
                Z_train = Z_train * weights_sqrt
                Z_test = Z_test * weights_sqrt

            model = Ridge(alpha=1.0, random_state=random_state)
            model.fit(Z_train, y_train_res)
            y_pred_res = model.predict(Z_test)

            # Reconstruction: y_pred = y_hat_test_pc + y_pred_res
            y_pred = y_hat_test_pc + y_pred_res

            if np.nanstd(y_test) == 0.0 or np.nanstd(y_pred) == 0.0:
                r = np.nan
            else:
                r = np.corrcoef(y_test, y_pred)[0, 1]

            print(f"  Fold {fold_idx}: r = {r:.3f} ({model_label}, PC-corrected)")

        r_values.append(float(r))

    return r_values


def ridge_cv_major_plus_bg(
    X: np.ndarray,
    y: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    major_idx: np.ndarray,
    use_pc_correction: bool,
    n_pcs: int,
    random_state: int,
) -> List[float]:
    """
    Ridge CV for "major QTL + background" model.

    Parameters
    ----------
    X : (n_samples, n_snps)
    y : (n_samples,)
    splits : list of (train_idx, test_idx)
    major_idx : array of column indices for major SNPs (can be empty)
    use_pc_correction : bool
    n_pcs : int
    random_state : int

    Returns
    -------
    r_values : list of float
    """
    r_values: List[float] = []

    if len(major_idx) == 0:
        print("[WARN] No major SNPs available; major_plus_bg reduces to baseline.")
    else:
        print(f"[INFO] major_plus_bg with {len(major_idx)} major SNPs "
              f"and {X.shape[1] - len(major_idx)} background SNPs.")

    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train_full = X[train_idx, :]
        X_test_full = X[test_idx, :]
        y_train = y[train_idx]
        y_test = y[test_idx]

        if np.nanstd(y_train) == 0.0:
            print(f"  [WARN] Fold {fold_idx}: y_train has zero variance; skipping.")
            r_values.append(np.nan)
            continue

        # Partition into major and background sets
        if len(major_idx) > 0:
            major_idx_sorted = np.sort(major_idx)
            bg_mask = np.ones(X.shape[1], dtype=bool)
            bg_mask[major_idx_sorted] = False

            X_train_maj = X_train_full[:, major_idx_sorted]
            X_test_maj = X_test_full[:, major_idx_sorted]
            X_train_bg = X_train_full[:, bg_mask]
            X_test_bg = X_test_full[:, bg_mask]
        else:
            # No major SNPs; all go to background
            X_train_maj = None
            X_test_maj = None
            X_train_bg = X_train_full
            X_test_bg = X_test_full

        # Fixed-effects part: PCs +/- major SNPs
        if not use_pc_correction:
            # Fixed effects: major SNPs only (if any)
            if X_train_maj is None:
                # No fixed covariates; this degenerates to baseline ridge on background
                y_train_res = y_train.copy()
                y_hat_test_fixed = np.zeros_like(y_test)
            else:
                scaler_fixed = StandardScaler(with_mean=True, with_std=True)
                Z_train_maj = scaler_fixed.fit_transform(X_train_maj)
                Z_test_maj = scaler_fixed.transform(X_test_maj)

                lr_fixed = LinearRegression()
                lr_fixed.fit(Z_train_maj, y_train)
                y_hat_train_fixed = lr_fixed.predict(Z_train_maj)
                y_hat_test_fixed = lr_fixed.predict(Z_test_maj)

                y_train_res = y_train - y_hat_train_fixed
        else:
            # Fixed effects: PCs + major SNPs
            # 1) PCs from background + major combined (i.e. whole X)
            scaler_pc = StandardScaler(with_mean=True, with_std=True)
            X_train_pc = scaler_pc.fit_transform(X_train_full)
            X_test_pc = scaler_pc.transform(X_test_full)

            max_pcs = min(n_pcs, X_train_pc.shape[0] - 1, X_train_pc.shape[1])
            if max_pcs <= 0:
                print(
                    f"  [WARN] Fold {fold_idx}: cannot compute PCs (max_pcs={max_pcs}); skipping."
                )
                r_values.append(np.nan)
                continue

            pca = PCA(n_components=max_pcs, random_state=random_state)
            PCs_train = pca.fit_transform(X_train_pc)
            PCs_test = pca.transform(X_test_pc)

            # 2) Build fixed-effects design: [PCs, major SNPs] (if any)
            if X_train_maj is not None:
                scaler_maj = StandardScaler(with_mean=True, with_std=True)
                Z_train_maj = scaler_maj.fit_transform(X_train_maj)
                Z_test_maj = scaler_maj.transform(X_test_maj)

                X_train_fixed = np.hstack([PCs_train, Z_train_maj])
                X_test_fixed = np.hstack([PCs_test, Z_test_maj])
            else:
                X_train_fixed = PCs_train
                X_test_fixed = PCs_test

            lr_fixed = LinearRegression()
            lr_fixed.fit(X_train_fixed, y_train)
            y_hat_train_fixed = lr_fixed.predict(X_train_fixed)
            y_hat_test_fixed = lr_fixed.predict(X_test_fixed)

            y_train_res = y_train - y_hat_train_fixed

        # Polygenic background via ridge
        scaler_bg = StandardScaler(with_mean=True, with_std=True)
        Z_train_bg = scaler_bg.fit_transform(X_train_bg)
        Z_test_bg = scaler_bg.transform(X_test_bg)

        model_bg = Ridge(alpha=1.0, random_state=random_state)
        model_bg.fit(Z_train_bg, y_train_res)
        y_pred_res = model_bg.predict(Z_test_bg)

        # Full prediction = fixed + residual prediction
        y_pred = y_hat_test_fixed + y_pred_res

        if np.nanstd(y_test) == 0.0 or np.nanstd(y_pred) == 0.0:
            r = np.nan
        else:
            r = np.corrcoef(y_test, y_pred)[0, 1]

        label_pc = "PC-corrected" if use_pc_correction else "no PC"
        print(f"  Fold {fold_idx}: r = {r:.3f} (major_plus_bg, {label_pc})")

        r_values.append(float(r))

    return r_values


# =========================
# Main
# =========================

def main() -> None:
    cfg.ensure_output_dirs()
    out_dir = cfg.CV_GWAS_INTEGRATION_DIR
    os.makedirs(out_dir, exist_ok=True)

    geno_path = os.path.join(cfg.CORE_DATA_DIR, "geno_core.npz")
    pheno_path = os.path.join(cfg.CORE_DATA_DIR, "pheno_core.csv")

    print("=" * 72)
    print("Mango GS – Idea 1: GWAS-integrated GS (baseline vs weighted vs major+bg)")
    print("=" * 72)
    print(f"[INFO] Geno core:  {geno_path}")
    print(f"[INFO] Pheno core: {pheno_path}")
    print(f"[INFO] Output dir: {out_dir}")
    print()

    if not os.path.exists(geno_path):
        raise FileNotFoundError(
            f"geno_core.npz not found at {geno_path}. Run 01_build_core_matrices.py first."
        )
    if not os.path.exists(pheno_path):
        raise FileNotFoundError(
            f"pheno_core.csv not found at {pheno_path}. Run 01_build_core_matrices.py first."
        )

    # Load core data
    npz = np.load(geno_path, allow_pickle=True)
    G = npz["G"]  # (n_samples, n_snps)
    sample_ids = npz["sample_ids"].astype(str)
    variant_ids = npz["variant_ids"].astype(str)

    pheno_df = pd.read_csv(pheno_path, index_col=0)
    pheno_df.index = pheno_df.index.astype(str)

    print(f"[INFO] Genotype matrix shape: {G.shape} (samples x SNPs)")
    print(f"[INFO] Phenotype table shape: {pheno_df.shape}")
    print()

    # Align samples
    common_samples = [sid for sid in sample_ids if sid in pheno_df.index]
    if len(common_samples) == 0:
        raise RuntimeError("No overlapping samples between geno_core and pheno_core.")

    sample_idx_map = {sid: i for i, sid in enumerate(sample_ids)}
    geno_idx = [sample_idx_map[sid] for sid in common_samples]

    G_aligned = G[geno_idx, :]
    pheno_aligned = pheno_df.loc[common_samples].copy()

    print(
        f"[INFO] Overlapping samples (geno & pheno): "
        f"{len(common_samples)} / {len(sample_ids)}"
    )
    print()

    traits = [t for t in cfg.TRAITS_DEFAULT if t in pheno_aligned.columns]
    if not traits:
        raise RuntimeError(
            "None of the traits in TRAITS_DEFAULT are present in pheno_core.csv."
        )

    print("[INFO] Traits for GWAS-integrated GS:", ", ".join(traits))
    print()

    n_samples, n_snps = G_aligned.shape
    splits = get_kfold_splits(
        n_samples=n_samples,
        n_splits=cfg.N_SPLITS,
        random_state=cfg.RANDOM_STATE,
    )

    records: List[Dict[str, object]] = []

    for trait in traits:
        print("-" * 72)
        print(f"[TRAIT] {trait}")

        y_full = pheno_aligned[trait].values.astype(float)
        mask = ~np.isnan(y_full)

        X_trait = G_aligned[mask, :]
        y_trait = y_full[mask]

        n_used = X_trait.shape[0]
        print(
            f"[INFO] Trait {trait}: using {n_used} samples with non-missing phenotype"
        )

        # Rebuild splits for this trait subset (since mask may drop some indices)
        # Map global indices -> local indices
        mask_indices = np.where(mask)[0]
        splits_trait: List[Tuple[np.ndarray, np.ndarray]] = []
        for train_global, test_global in splits:
            # Intersect global splits with mask
            train_local = np.intersect1d(
                np.where(mask)[0], train_global, assume_unique=True
            )
            test_local = np.intersect1d(
                np.where(mask)[0], test_global, assume_unique=True
            )
            # Convert local global indices to 0..n_used-1
            train_local = np.searchsorted(mask_indices, train_local)
            test_local = np.searchsorted(mask_indices, test_local)

            if len(train_local) >= 10 and len(test_local) >= 5:
                splits_trait.append((train_local, test_local))

        if not splits_trait:
            print(
                f"[WARN] No valid K-fold splits for trait {trait} after masking; skipping."
            )
            continue

        print(f"[INFO] Using {len(splits_trait)} K-fold splits for trait {trait}")

        # ---------- Load weights and major QTL for this trait ----------
        weights_path = os.path.join(
            cfg.GWAS_WEIGHTS_DIR, f"snp_weights_{trait}.npz"
        )
        major_path = os.path.join(
            cfg.GWAS_WEIGHTS_DIR, f"major_qtl_snps_{trait}.csv"
        )

        if not os.path.exists(weights_path) or not os.path.exists(major_path):
            print(
                f"[WARN] Missing weights or major QTL file for {trait}; "
                "skipping GWAS-integrated models for this trait."
            )
            continue

        w_npz = np.load(weights_path, allow_pickle=True)
        w_variant_ids = w_npz["variant_ids"].astype(str)
        weights = w_npz["weights"].astype(float)

        if len(w_variant_ids) != n_snps or not np.all(w_variant_ids == variant_ids):
            raise RuntimeError(
                f"Variant IDs in {weights_path} do not match geno_core.npz"
            )

        weights_sqrt = np.sqrt(weights).astype(np.float32)

        major_df = pd.read_csv(major_path)
        if "variant_id" not in major_df.columns:
            raise KeyError(
                f"'variant_id' column not found in {major_path}. "
                "Check output from 04_internal_gwas_and_weights.py."
            )

        major_ids = major_df["variant_id"].astype(str).tolist()
        var_to_idx = {vid: i for i, vid in enumerate(variant_ids)}
        major_idx = np.array(
            [var_to_idx[vid] for vid in major_ids if vid in var_to_idx],
            dtype=int,
        )

        print(
            f"[INFO] Trait {trait}: loaded weights for {len(weights)} SNPs "
            f"and {len(major_idx)} mapped major SNP indices."
        )

        # ---------- (1) baseline model ----------
        for scenario_name, use_pc in [
            ("no_pc", False),
            ("pc_corrected", True),
        ]:
            r_vals = ridge_cv_baseline_or_weighted(
                X=X_trait,
                y=y_trait,
                splits=splits_trait,
                weights_sqrt=None,
                use_pc_correction=use_pc,
                n_pcs=cfg.N_PCS,
                random_state=cfg.RANDOM_STATE,
            )
            for fold_idx, r in enumerate(r_vals, start=1):
                records.append(
                    {
                        "trait": trait,
                        "model_type": "baseline",
                        "scenario": scenario_name,
                        "fold": fold_idx,
                        "r": r,
                    }
                )

        print()

        # ---------- (2) weighted model ----------
        for scenario_name, use_pc in [
            ("no_pc", False),
            ("pc_corrected", True),
        ]:
            r_vals = ridge_cv_baseline_or_weighted(
                X=X_trait,
                y=y_trait,
                splits=splits_trait,
                weights_sqrt=weights_sqrt,
                use_pc_correction=use_pc,
                n_pcs=cfg.N_PCS,
                random_state=cfg.RANDOM_STATE,
            )
            for fold_idx, r in enumerate(r_vals, start=1):
                records.append(
                    {
                        "trait": trait,
                        "model_type": "weighted",
                        "scenario": scenario_name,
                        "fold": fold_idx,
                        "r": r,
                    }
                )

        print()

        # ---------- (3) major_plus_bg model ----------
        for scenario_name, use_pc in [
            ("no_pc", False),
            ("pc_corrected", True),
        ]:
            r_vals = ridge_cv_major_plus_bg(
                X=X_trait,
                y=y_trait,
                splits=splits_trait,
                major_idx=major_idx,
                use_pc_correction=use_pc,
                n_pcs=cfg.N_PCS,
                random_state=cfg.RANDOM_STATE,
            )
            for fold_idx, r in enumerate(r_vals, start=1):
                records.append(
                    {
                        "trait": trait,
                        "model_type": "major_plus_bg",
                        "scenario": scenario_name,
                        "fold": fold_idx,
                        "r": r,
                    }
                )

        print()

    # ---------- Save results ----------
    if not records:
        print("[WARN] No CV records generated; nothing to save.")
        return

    results_df = pd.DataFrame.from_records(records)
    results_path = os.path.join(out_dir, "cv_gwas_integration_results.csv")
    print(f"[INFO] Saving detailed results -> {results_path}")
    results_df.to_csv(results_path, index=False)

    summary_df = (
        results_df.groupby(["trait", "model_type", "scenario"], dropna=False)["r"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_r", "count": "n_folds"})
    )
    summary_path = os.path.join(out_dir, "cv_gwas_integration_summary.csv")
    print(f"[INFO] Saving summary -> {summary_path}")
    summary_df.to_csv(summary_path, index=False)

    print()
    print("[DONE] GWAS-integrated GS (baseline vs weighted vs major+bg) complete.")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
