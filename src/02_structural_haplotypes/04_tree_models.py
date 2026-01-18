#!/usr/bin/env python
r"""
04_tree_models.py

Tree-based GS for Mango GS – Idea 2.

This script:
  * Loads ML-ready genotype + phenotype from Idea 2 core step:
      - output\idea_2\core_ml\X_full.npy
      - output\idea_2\core_ml\y_traits.csv
      - output\idea_2\core_ml\samples.csv
  * Loads cross-validation designs from:
      - output\idea_2\cv_design\cv_random_k5.csv
      - output\idea_2\cv_design\cv_cluster_kmeans.csv
      - (optionally) cv_ancestry.csv, if present
  * For each trait and each CV scheme:
      - Drops samples with NaN phenotype (trait-specific mask)
      - Runs:
          - XGBoostRegressor
          - RandomForestRegressor
  * Writes:
      - output\idea_2\results_xgb_rf\results_xgb_rf_perfold.csv
      - output\idea_2\results_xgb_rf\results_xgb_rf_summary.csv
      - output\idea_2\results_xgb_rf\predictions_xgb_rf.csv
"""

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np

try:
    import pandas as pd
except ImportError as e:
    raise SystemExit("pandas is required. Install with: pip install pandas") from e

try:
    from sklearn.ensemble import RandomForestRegressor
except ImportError as e:
    raise SystemExit(
        "scikit-learn is required for RandomForestRegressor. Install with:\n\n  pip install scikit-learn\n"
    ) from e

# XGBoost
try:
    from xgboost import XGBRegressor
except ImportError as e:
    raise SystemExit(
        "xgboost is required. Install with:\n\n  pip install xgboost\n"
    ) from e


# =========================
# DEFAULT PATHS / PARAMS
# =========================

DEFAULT_X_PATH = r"C:\Users\ms\Desktop\mango\output\idea_2\core_ml\X_full.npy"
DEFAULT_Y_PATH = r"C:\Users\ms\Desktop\mango\output\idea_2\core_ml\y_traits.csv"
DEFAULT_SAMPLES_PATH = r"C:\Users\ms\Desktop\mango\output\idea_2\core_ml\samples.csv"
DEFAULT_CV_DIR = r"C:\Users\ms\Desktop\mango\output\idea_2\cv_design"
DEFAULT_OUTDIR = r"C:\Users\ms\Desktop\mango\output\idea_2\results_xgb_rf"

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
) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    """
    Load X_full, y_traits and samples, and ensure alignment.

    Returns:
        X           : numpy array (n_samples x n_snps)
        y_df        : DataFrame (n_samples x n_traits), index=sample_id
        sample_ids  : list of sample IDs in row-order of X
    """
    if not os.path.isfile(X_path):
        raise FileNotFoundError(f"Genotype matrix not found: {X_path}")
    if not os.path.isfile(y_path):
        raise FileNotFoundError(f"Phenotype file not found: {y_path}")
    if not os.path.isfile(samples_path):
        raise FileNotFoundError(f"samples.csv not found: {samples_path}")

    print(f"[LOAD] X_full -> {X_path}")
    X = np.load(X_path)
    if X.ndim != 2:
        raise RuntimeError(f"Expected 2D array for X_full, got shape={X.shape}")

    print(f"[LOAD] y_traits -> {y_path}")
    y_df = pd.read_csv(y_path)
    if "sample_id" in y_df.columns:
        y_df = y_df.set_index("sample_id")
    else:
        # maybe first col is index
        first = y_df.columns[0]
        if first.lower().startswith("sample"):
            y_df = y_df.set_index(first)
        else:
            raise RuntimeError(
                f"Expected 'sample_id' column in {y_path}, got columns={list(y_df.columns)}"
            )

    y_df.index = y_df.index.astype(str)

    print(f"[LOAD] samples -> {samples_path}")
    s_df = pd.read_csv(samples_path)
    if "sample_id" not in s_df.columns:
        raise RuntimeError(
            f"Expected 'sample_id' column in {samples_path}, got columns={list(s_df.columns)}"
        )
    sample_ids = s_df["sample_id"].astype(str).tolist()

    # Align y_df to sample_ids
    missing_y = [sid for sid in sample_ids if sid not in y_df.index]
    if missing_y:
        raise RuntimeError(
            f"{len(missing_y)} samples from samples.csv missing in y_traits.csv.\n"
            f"Example missing IDs: {missing_y[:5]}"
        )

    y_df = y_df.loc[sample_ids]
    if X.shape[0] != len(sample_ids):
        raise RuntimeError(
            f"Row mismatch: X has {X.shape[0]} rows, but samples.csv has {len(sample_ids)} entries."
        )

    print(
        f"[INFO] Loaded core data: X shape = {X.shape} (samples x SNPs), "
        f"y shape = {y_df.shape} (samples x traits)"
    )
    return X, y_df, sample_ids


def load_cv_design(cv_dir: str) -> Dict[str, np.ndarray]:
    """
    Load all CV design files from cv_dir.

    Each file is expected to have columns:
      - sample_id
      - fold (integer labels 1..k)

    Returns:
      dict mapping scheme_name -> fold_ids (np.ndarray, length = n_samples)
    """
    if not os.path.isdir(cv_dir):
        raise FileNotFoundError(f"CV directory not found: {cv_dir}")

    cv_files = [
        f for f in os.listdir(cv_dir)
        if f.lower().endswith(".csv") and f.lower().startswith("cv_")
    ]
    if not cv_files:
        raise RuntimeError(f"No CV design files found in {cv_dir} (expected files starting with 'cv_').")

    print(f"[INFO] Found CV design files: {cv_files}")

    cv_designs: Dict[str, np.ndarray] = {}
    ref_sample_ids: List[str] = []

    for fname in cv_files:
        scheme_name = os.path.splitext(fname)[0]  # e.g. cv_random_k5
        path = os.path.join(cv_dir, fname)

        df = pd.read_csv(path)
        if "sample_id" not in df.columns or "fold" not in df.columns:
            raise RuntimeError(
                f"CV file {fname} must have 'sample_id' and 'fold' columns. Got {list(df.columns)}"
            )

        df["sample_id"] = df["sample_id"].astype(str)

        if not ref_sample_ids:
            ref_sample_ids = df["sample_id"].tolist()
        else:
            ids = df["sample_id"].tolist()
            if set(ids) != set(ref_sample_ids):
                raise RuntimeError(
                    f"Sample ID mismatch between CV files.\n"
                    f"  Reference (from {cv_files[0]}): {len(ref_sample_ids)} samples\n"
                    f"  Current ({fname}): {len(ids)} samples"
                )
            df = df.set_index("sample_id").loc[ref_sample_ids].reset_index()

        folds = df["fold"].to_numpy()
        cv_designs[scheme_name] = folds.astype(int)

    return cv_designs


# =========================
# MODEL HELPERS
# =========================

def xgb_cv_with_fixed_folds(
    X: np.ndarray,
    y: np.ndarray,
    fold_ids: np.ndarray,
    random_state: int = RANDOM_STATE,
) -> Tuple[List[float], float]:
    """
    XGBoostRegressor with pre-defined folds.

    Returns:
        r_values : list of Pearson r per fold
        r_mean   : mean r across folds
    """
    from numpy import corrcoef

    unique_folds = sorted(set(int(f) for f in fold_ids))
    r_values: List[float] = []

    for fold in unique_folds:
        test_mask = (fold_ids == fold)
        train_mask = ~test_mask

        X_train = X[train_mask, :]
        X_test = X[test_mask, :]
        y_train = y[train_mask]
        y_test = y[test_mask]

        # XGBoost regressor with moderate depth and strong subsampling
        model = XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.7,
            colsample_bytree=0.3,
            reg_lambda=1.0,
            reg_alpha=0.0,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if np.all(np.isfinite(y_test)) and np.all(np.isfinite(y_pred)):
            r = corrcoef(y_test, y_pred)[0, 1]
        else:
            r = np.nan
        r_values.append(r)
        print(f"    Fold {fold}: r = {r:.3f} (XGB)")

    r_mean = float(np.nanmean(r_values)) if r_values else float("nan")
    return r_values, r_mean


def rf_cv_with_fixed_folds(
    X: np.ndarray,
    y: np.ndarray,
    fold_ids: np.ndarray,
    random_state: int = RANDOM_STATE,
) -> Tuple[List[float], float]:
    """
    RandomForestRegressor with pre-defined folds.

    Returns:
        r_values : list of Pearson r per fold
        r_mean   : mean r across folds
    """
    from numpy import corrcoef

    unique_folds = sorted(set(int(f) for f in fold_ids))
    r_values: List[float] = []

    for fold in unique_folds:
        test_mask = (fold_ids == fold)
        train_mask = ~test_mask

        X_train = X[train_mask, :]
        X_test = X[test_mask, :]
        y_train = y[train_mask]
        y_test = y[test_mask]

        model = RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            max_features="sqrt",
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=random_state,
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if np.all(np.isfinite(y_test)) and np.all(np.isfinite(y_pred)):
            r = corrcoef(y_test, y_pred)[0, 1]
        else:
            r = np.nan
        r_values.append(r)
        print(f"    Fold {fold}: r = {r:.3f} (RF)")

    r_mean = float(np.nanmean(r_values)) if r_values else float("nan")
    return r_values, r_mean


# =========================
# MAIN
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run XGBoost and RandomForest GS models for Mango GS Idea 2."
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
        "--cv-dir",
        type=str,
        default=DEFAULT_CV_DIR,
        help=f"Directory with CV design files (default: {DEFAULT_CV_DIR})",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=DEFAULT_OUTDIR,
        help=f"Output directory for XGB/RF results (default: {DEFAULT_OUTDIR})",
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
    print("Mango GS – Idea 2: XGBoost / Random Forest models")
    print("=" * 72)
    print(f"[INFO] X_full:       {args.X_path}")
    print(f"[INFO] y_traits:     {args.y_path}")
    print(f"[INFO] samples:      {args.samples_path}")
    print(f"[INFO] CV dir:       {args.cv_dir}")
    print(f"[INFO] Output dir:   {args.outdir}")
    print(f"[INFO] Seed:         {args.seed}")
    print("")

    safe_mkdir(args.outdir)

    # 1) Core data
    X, y_df, sample_ids = load_core_matrices(
        X_path=args.X_path,
        y_path=args.y_path,
        samples_path=args.samples_path,
    )

    # 2) CV designs
    cv_designs = load_cv_design(args.cv_dir)

    traits = list(y_df.columns)
    print(f"[INFO] Traits: {traits}")
    print(f"[INFO] CV schemes: {list(cv_designs.keys())}")
    print("")

    # Collectors
    records_perfold = []
    records_summary = []
    pred_rows = []

    # 3) Trait-wise CV loops
    for trait in traits:
        print(f"[TRAIT] {trait}")

        y_full = y_df[trait].to_numpy(dtype=float)
        mask_good = ~np.isnan(y_full)
        n_total = len(y_full)
        n_used = int(mask_good.sum())
        n_drop = n_total - n_used

        if n_drop > 0:
            print(f"  [INFO] Dropping {n_drop}/{n_total} samples with NaN for trait '{trait}'.")
        if n_used < 3:
            print(f"  [WARN] Only {n_used} non-NaN samples for trait '{trait}'. Skipping.")
            continue

        X_trait = X[mask_good, :]
        y_trait = y_full[mask_good]
        sample_ids_trait = [sid for sid, keep in zip(sample_ids, mask_good) if keep]

        for scheme_name, folds_all in cv_designs.items():
            if len(folds_all) != X.shape[0]:
                raise RuntimeError(
                    f"Fold array length mismatch for {scheme_name}: "
                    f"{len(folds_all)} vs n_samples={X.shape[0]}"
                )

            folds = folds_all[mask_good]
            unique_folds = sorted(set(folds))
            print(f"  [SCHEME] {scheme_name} (unique folds={unique_folds})")

            # Storage for per-sample predictions (for this trait × scheme)
            y_pred_xgb = np.full_like(y_trait, np.nan, dtype=float)
            y_pred_rf = np.full_like(y_trait, np.nan, dtype=float)

            # --- XGBoost ---
            print("  -> XGBoost")
            r_vals_xgb, r_mean_xgb = xgb_cv_with_fixed_folds(
                X_trait, y_trait, folds, random_state=args.seed
            )
            # Collect per-sample predictions
            for fold in unique_folds:
                test_mask = (folds == fold)
                train_mask = ~test_mask
                X_train = X_trait[train_mask, :]
                X_test = X_trait[test_mask, :]
                y_train = y_trait[train_mask]

                model = XGBRegressor(
                    n_estimators=400,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.7,
                    colsample_bytree=0.3,
                    reg_lambda=1.0,
                    reg_alpha=0.0,
                    objective="reg:squarederror",
                    random_state=args.seed,
                    n_jobs=-1,
                    tree_method="hist",
                )
                model.fit(X_train, y_train)
                y_pred_fold = model.predict(X_test)
                y_pred_xgb[test_mask] = y_pred_fold

            # Record per-fold metrics for XGB
            for fold_label, r_val in zip(unique_folds, r_vals_xgb):
                records_perfold.append(
                    {
                        "trait": trait,
                        "scheme": scheme_name,
                        "model": "xgb",
                        "fold": int(fold_label),
                        "r": r_val,
                        "n_used": n_used,
                        "n_total": n_total,
                    }
                )
            records_summary.append(
                {
                    "trait": trait,
                    "scheme": scheme_name,
                    "model": "xgb",
                    "mean_r": float(np.nanmean(r_vals_xgb)),
                    "std_r": float(np.nanstd(r_vals_xgb)),
                    "n_folds": len(r_vals_xgb),
                    "n_used": n_used,
                    "n_total": n_total,
                }
            )

            # --- Random Forest ---
            print("  -> Random Forest")
            r_vals_rf, r_mean_rf = rf_cv_with_fixed_folds(
                X_trait, y_trait, folds, random_state=args.seed
            )
            # Collect per-sample predictions
            for fold in unique_folds:
                test_mask = (folds == fold)
                train_mask = ~test_mask
                X_train = X_trait[train_mask, :]
                X_test = X_trait[test_mask, :]
                y_train = y_trait[train_mask]

                model = RandomForestRegressor(
                    n_estimators=400,
                    max_depth=None,
                    max_features="sqrt",
                    min_samples_split=2,
                    min_samples_leaf=1,
                    n_jobs=-1,
                    random_state=args.seed,
                )
                model.fit(X_train, y_train)
                y_pred_fold = model.predict(X_test)
                y_pred_rf[test_mask] = y_pred_fold

            # Record per-fold metrics for RF
            for fold_label, r_val in zip(unique_folds, r_vals_rf):
                records_perfold.append(
                    {
                        "trait": trait,
                        "scheme": scheme_name,
                        "model": "rf",
                        "fold": int(fold_label),
                        "r": r_val,
                        "n_used": n_used,
                        "n_total": n_total,
                    }
                )
            records_summary.append(
                {
                    "trait": trait,
                    "scheme": scheme_name,
                    "model": "rf",
                    "mean_r": float(np.nanmean(r_vals_rf)),
                    "std_r": float(np.nanstd(r_vals_rf)),
                    "n_folds": len(r_vals_rf),
                    "n_used": n_used,
                    "n_total": n_total,
                }
            )

            # --- Per-sample predictions (for this trait × scheme) ---
            for sid, y_true, y_hat_x, y_hat_r, fold_label in zip(
                sample_ids_trait, y_trait, y_pred_xgb, y_pred_rf, folds
            ):
                pred_rows.append(
                    {
                        "sample_id": sid,
                        "trait": trait,
                        "scheme": scheme_name,
                        "fold": int(fold_label),
                        "y_true": float(y_true),
                        "y_pred_xgb": float(y_hat_x),
                        "y_pred_rf": float(y_hat_r),
                    }
                )

        print("")

    # 4) Save outputs
    perfold_df = pd.DataFrame.from_records(records_perfold)
    summary_df = pd.DataFrame.from_records(records_summary)
    preds_df = pd.DataFrame.from_records(pred_rows)

    perfold_path = os.path.join(args.outdir, "results_xgb_rf_perfold.csv")
    summary_path = os.path.join(args.outdir, "results_xgb_rf_summary.csv")
    preds_path = os.path.join(args.outdir, "predictions_xgb_rf.csv")

    perfold_df.to_csv(perfold_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    preds_df.to_csv(preds_path, index=False)

    print(f"[SAVE] Per-fold metrics -> {perfold_path}")
    print(f"[SAVE] Summary metrics  -> {summary_path}")
    print(f"[SAVE] Predictions      -> {preds_path}")
    print("")
    print("[OK] XGBoost / Random Forest models for Idea 2 completed.")


if __name__ == "__main__":
    main()
