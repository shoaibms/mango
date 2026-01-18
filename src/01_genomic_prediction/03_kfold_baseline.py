# -*- coding: utf-8 -*-
"""
03_kfold_baseline.py

Baseline genomic prediction for Mango GS (Idea 1) using ridge regression:

  - Scenario A: no PC correction
  - Scenario B: PC-corrected phenotype (y residualised on PCs of X)

Inputs
------
Uses core matrices produced by 01_build_core_matrices.py:
  - geno_core.npz  (G, sample_ids, variant_ids)
  - pheno_core.csv (traits: FBC, FF, AFW, TSS, TC)

Outputs
-------
In cfg.CV_BASELINE_DIR:
  - cv_baseline_results.csv
      trait,scenario,fold,r

  - cv_baseline_summary.csv
      trait,scenario,mean_r,n_folds
"""

from __future__ import annotations

import os
import argparse
from typing import Dict, List, Tuple, Optional

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
# Core CV utility
# =========================

def load_cv_design_csv(cv_path: str, sample_ids: List[str]) -> Tuple[np.ndarray, str]:
    """
    Load Idea2 CV design CSV with columns: sample_id, fold (fold is int-like, 1..K).
    Returns:
      fold_ids aligned to sample_ids (len == len(sample_ids))
      scheme_name inferred from filename (e.g., cv_random_k5)
    """
    df = pd.read_csv(cv_path)
    if not {"sample_id", "fold"}.issubset(df.columns):
        raise ValueError(f"cv_design must have columns sample_id, fold. Found: {list(df.columns)}")

    scheme_name = os.path.splitext(os.path.basename(cv_path))[0]
    fold_map = df.set_index("sample_id")["fold"]

    missing = [sid for sid in sample_ids if sid not in fold_map.index]
    if missing:
        raise ValueError(f"{len(missing)} sample_ids not found in cv_design. Example: {missing[:5]}")

    fold_ids = np.array([int(fold_map.loc[sid]) for sid in sample_ids], dtype=int)
    return fold_ids, scheme_name


def crossval_ridge(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    use_pc_correction: bool,
    n_pcs: int,
    random_state: int,
    sample_ids: Optional[np.ndarray] = None,
    trait: str = "",
    scenario: str = "",
    scheme: str = "kfold_random_5",
    feature_set: str = "SNP",
    model_name: str = "ridge",
    oof_rows: Optional[List[Dict[str, object]]] = None,
    fold_ids: Optional[np.ndarray] = None,
) -> Tuple[List[float], float]:
    """
    Ridge regression K-fold CV.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_snps)
    y : array-like, shape (n_samples,)
    n_splits : int
        Number of CV folds.
    use_pc_correction : bool
        If True, residualise y on PCs of X within each training fold.
    n_pcs : int
        Number of PCs to use when residualising (capped at min(n_samples-1, n_snps)).
    random_state : int
        Seed for KFold.
    fold_ids : array-like, optional
        If provided, use these fold assignments (ints) instead of KFold.
        Must be aligned with X and y.

    Returns
    -------
    r_values : list of float
        Pearson r for each fold.
    r_mean : float
        Mean r across folds (ignoring NaN).
    """
    n_samples, n_features = X.shape

    r_values: List[float] = []

    if use_pc_correction:
        print(f"[INFO] Running ridge CV with PC correction (n_pcs={n_pcs})")
    else:
        print("[INFO] Running ridge CV without PC correction")

    if fold_ids is None:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iter = [(i, tr, te) for i, (tr, te) in enumerate(kf.split(X), start=1)]
    else:
        uniq = sorted(set(int(x) for x in np.unique(fold_ids)))
        split_iter = []
        for f in uniq:
            te = np.where(fold_ids == f)[0]
            tr = np.where(fold_ids != f)[0]
            if len(te) == 0 or len(tr) == 0:
                continue
            split_iter.append((f, tr, te))

    for fold_idx, train_idx, test_idx in split_iter:
        X_train = X[train_idx, :]
        X_test = X[test_idx, :]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # Skip folds with constant y values
        if np.nanstd(y_train) == 0.0:
            print(f"  [WARN] Fold {fold_idx}: y_train has zero variance; skipping.")
            r_values.append(np.nan)
            continue

        if not use_pc_correction:
            # Standard ridge regression on raw y
            scaler_X = StandardScaler(with_mean=True, with_std=True)
            X_train_scaled = scaler_X.fit_transform(X_train)
            X_test_scaled = scaler_X.transform(X_test)

            model = Ridge(alpha=1.0, random_state=random_state)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            if oof_rows is not None and sample_ids is not None:
                for sid, yt, yp in zip(sample_ids[test_idx], y_test, y_pred):
                    oof_rows.append(
                        {
                            "source": "idea1",
                            "feature_set": feature_set,
                            "model": model_name,
                            "scheme": scheme,
                            "scenario": scenario,
                            "trait": trait,
                            "fold": fold_idx,
                            "sample_id": str(sid),
                            "y_true": float(yt),
                            "y_pred": float(yp),
                            "y_true_resid": np.nan,
                            "y_pred_resid": np.nan,
                            "y_fixed_pred": np.nan,
                        }
                    )

            # Corr(y_test, y_pred)
            if np.nanstd(y_test) == 0.0 or np.nanstd(y_pred) == 0.0:
                r = np.nan
            else:
                r = np.corrcoef(y_test, y_pred)[0, 1]

            print(f"  Fold {fold_idx}: r = {r:.3f} (no PC correction)")

        else:
            # 1) PCs of X (training only)
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

            pca = PCA(n_components=max_pcs)
            PCs_train = pca.fit_transform(X_train_pc)
            PCs_test = pca.transform(X_test_pc)

            # 2) Regress y on PCs in training, get residuals
            lr = LinearRegression()
            lr.fit(PCs_train, y_train)
            y_hat_train = lr.predict(PCs_train)
            y_hat_test = lr.predict(PCs_test)

            y_train_res = y_train - y_hat_train
            y_test_res = y_test - y_hat_test

            # 3) Ridge on residuals (with fresh scaling of X)
            scaler_X2 = StandardScaler(with_mean=True, with_std=True)
            X_train_scaled2 = scaler_X2.fit_transform(X_train)
            X_test_scaled2 = scaler_X2.transform(X_test)

            model = Ridge(alpha=1.0, random_state=random_state)
            model.fit(X_train_scaled2, y_train_res)
            y_pred_res = model.predict(X_test_scaled2)

            y_pred_pheno = y_hat_test + y_pred_res

            if oof_rows is not None and sample_ids is not None:
                for sid, yt, yp, ytr, ypr, yfix in zip(
                    sample_ids[test_idx],
                    y_test,
                    y_pred_pheno,
                    y_test_res,
                    y_pred_res,
                    y_hat_test,
                ):
                    oof_rows.append(
                        {
                            "source": "idea1",
                            "feature_set": feature_set,
                            "model": model_name,
                            "scheme": scheme,
                            "scenario": scenario,
                            "trait": trait,
                            "fold": fold_idx,
                            "sample_id": str(sid),
                            "y_true": float(yt),
                            "y_pred": float(yp),
                            "y_true_resid": float(ytr),
                            "y_pred_resid": float(ypr),
                            "y_fixed_pred": float(yfix),
                        }
                    )

            if np.nanstd(y_test_res) == 0.0 or np.nanstd(y_pred_res) == 0.0:
                r = np.nan
            else:
                r = np.corrcoef(y_test_res, y_pred_res)[0, 1]

            print(f"  Fold {fold_idx}: r = {r:.3f} (PC-corrected)")

        r_values.append(float(r))

    r_mean = float(np.nanmean(r_values))
    return r_values, r_mean


# =========================
# Main
# =========================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv_design", type=str, default="", help="Optional Idea2 cv_design CSV (sample_id, fold).")
    args = parser.parse_args()

    cfg.ensure_output_dirs()
    out_dir = cfg.CV_BASELINE_DIR
    os.makedirs(out_dir, exist_ok=True)

    geno_path = os.path.join(cfg.CORE_DATA_DIR, "geno_core.npz")
    pheno_path = os.path.join(cfg.CORE_DATA_DIR, "pheno_core.csv")

    print("=" * 72)
    print("Mango GS – Idea 1: Baseline K-fold ridge CV")
    print("=" * 72)
    print(f"[INFO] Geno core:   {geno_path}")
    print(f"[INFO] Pheno core:  {pheno_path}")
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

    # 1) Load genotype core
    npz = np.load(geno_path, allow_pickle=True)
    G = npz["G"]  # shape: (n_samples, n_snps)
    sample_ids = npz["sample_ids"].astype(str)
    variant_ids = npz["variant_ids"]

    print(f"[INFO] Genotype matrix shape: {G.shape} (samples x SNPs)")

    # 2) Load phenotype core
    pheno_df = pd.read_csv(pheno_path, index_col=0)
    pheno_df.index = pheno_df.index.astype(str)

    print(f"[INFO] Phenotype table shape: {pheno_df.shape}")

    # 3) Align sample order
    common_samples = [sid for sid in sample_ids if sid in pheno_df.index]
    if len(common_samples) == 0:
        raise RuntimeError(
            "No overlapping samples between geno_core and pheno_core."
        )

    print(
        f"[INFO] Overlapping samples (geno ↔ pheno): "
        f"{len(common_samples)} / {len(sample_ids)}"
    )

    sample_ids_aligned = np.array(common_samples, dtype=str)

    sample_idx_map = {sid: i for i, sid in enumerate(sample_ids)}
    geno_idx = [sample_idx_map[sid] for sid in common_samples]

    G_aligned = G[geno_idx, :]
    pheno_aligned = pheno_df.loc[common_samples].copy()

    fold_ids_aligned = None
    scheme_name = f"kfold_random_{cfg.N_SPLITS}"
    if args.cv_design:
        fold_ids_aligned, scheme_name = load_cv_design_csv(args.cv_design, common_samples)
        print(f"[INFO] Using external CV design: {args.cv_design}")
        print(f"[INFO] CV scheme_name: {scheme_name}")

    # 4) Traits to analyse
    traits_available = [t for t in cfg.TRAITS_DEFAULT if t in pheno_aligned.columns]
    if not traits_available:
        raise RuntimeError(
            "None of the traits in TRAITS_DEFAULT are columns of pheno_core.csv"
        )

    print("[INFO] Traits to analyse:", ", ".join(traits_available))
    print()

    # 5) Run CV per trait and scenario
    records: List[Dict[str, object]] = []
    oof_rows: List[Dict[str, object]] = []

    for trait in traits_available:
        print("-" * 72)
        print(f"[TRAIT] {trait}")
        y_full = pheno_aligned[trait].values.astype(float)

        # Mask out missing phenotypes
        mask = ~np.isnan(y_full)
        X_trait = G_aligned[mask, :]
        y_trait = y_full[mask]
        ids_trait = sample_ids_aligned[mask]

        print(
            f"[INFO] Trait {trait}: {np.sum(mask)} samples with non-missing phenotype"
        )

        fold_ids_trait = None
        if fold_ids_aligned is not None:
            fold_ids_trait = fold_ids_aligned[mask]

        # Scenario A: no PC correction
        r_values_no_pc, r_mean_no_pc = crossval_ridge(
            X=X_trait,
            y=y_trait,
            n_splits=cfg.N_SPLITS,
            use_pc_correction=False,
            n_pcs=cfg.N_PCS,
            random_state=cfg.RANDOM_STATE,
            sample_ids=ids_trait,
            trait=trait,
            scenario="no_pc",
            scheme=scheme_name,
            feature_set="SNP",
            model_name="ridge",
            oof_rows=oof_rows,
            fold_ids=fold_ids_trait,
        )

        for fold_idx, r in enumerate(r_values_no_pc, start=1):
            records.append(
                {
                    "trait": trait,
                    "scenario": "no_pc",
                    "fold": fold_idx,
                    "r": r,
                }
            )

        print(f"[RESULT] {trait} – no PC: mean r = {r_mean_no_pc:.3f}")
        print()

        # Scenario B: PC-corrected phenotype
        r_values_pc, r_mean_pc = crossval_ridge(
            X=X_trait,
            y=y_trait,
            n_splits=cfg.N_SPLITS,
            use_pc_correction=True,
            n_pcs=cfg.N_PCS,
            random_state=cfg.RANDOM_STATE,
            sample_ids=ids_trait,
            trait=trait,
            scenario="pc_corrected",
            scheme=scheme_name,
            feature_set="SNP",
            model_name="ridge",
            oof_rows=oof_rows,
            fold_ids=fold_ids_trait,
        )

        for fold_idx, r in enumerate(r_values_pc, start=1):
            records.append(
                {
                    "trait": trait,
                    "scenario": "pc_corrected",
                    "fold": fold_idx,
                    "r": r,
                }
            )

        print(f"[RESULT] {trait} – PC-corrected: mean r = {r_mean_pc:.3f}")
        print()

    # 6) Save results
    results_df = pd.DataFrame.from_records(records)
    results_path = os.path.join(out_dir, "cv_baseline_results.csv")
    print(f"[INFO] Saving detailed results -> {results_path}")
    results_df.to_csv(results_path, index=False)

    # Summary
    summary_df = (
        results_df.groupby(["trait", "scenario"], dropna=False)["r"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_r", "count": "n_folds"})
    )
    summary_path = os.path.join(out_dir, "cv_baseline_summary.csv")
    print(f"[INFO] Saving summary -> {summary_path}")
    summary_df.to_csv(summary_path, index=False)

    if oof_rows:
        oof_df = pd.DataFrame(oof_rows)
        oof_path = os.path.join(out_dir, "cv_baseline_oof_predictions.csv")
        print(f"[INFO] Saving OOF predictions -> {oof_path}")
        oof_df.to_csv(oof_path, index=False)

    print()
    print("[DONE] Baseline K-fold CV complete.")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
