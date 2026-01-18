#!/usr/bin/env python
r"""
03_linear_baselines.py

Run baseline linear models (ridge regression) for Mango GS - Idea 2.

This script:
  * Loads ML-ready genotype + phenotype from Idea 2 core step:
      - output\idea_2\core_ml\X_full.npy
      - output\idea_2\core_ml\y_traits.csv
      - output\idea_2\core_ml\samples.csv
  * Loads cross-validation designs from:
      - output\idea_2\cv_design\cv_random_k5.csv
      - output\idea_2\cv_design\cv_cluster_kmeans.csv
      - (optionally) output\idea_2\cv_design\cv_ancestry.csv
  * For each trait and each CV scheme, fits ridge regression models:
      - without PC correction (naive)
      - with within-fold PC correction (regress y on PCs, then ridge on residuals)
  * Writes per-fold and summary metrics to:
      - output\idea_2\results_baseline\baseline_ridge_perfold.csv
      - output\idea_2\results_baseline\baseline_ridge_summary.csv
"""

import argparse
import os
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import pandas as pd
except ImportError as e:
    raise SystemExit("pandas is required. Install with: pip install pandas") from e

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge, LinearRegression
except ImportError as e:
    raise SystemExit(
        "scikit-learn is required. Install with:\n\n  pip install scikit-learn\n"
    ) from e


# =========================
# DEFAULT PATHS / PARAMS
# =========================

DEFAULT_X_PATH = r"C:\Users\ms\Desktop\mango\output\idea_2\core_ml\X_full.npy"
DEFAULT_Y_PATH = r"C:\Users\ms\Desktop\mango\output\idea_2\core_ml\y_traits.csv"
DEFAULT_SAMPLES_PATH = r"C:\Users\ms\Desktop\mango\output\idea_2\core_ml\samples.csv"
DEFAULT_CV_DIR = r"C:\Users\ms\Desktop\mango\output\idea_2\cv_design"
DEFAULT_OUTDIR = r"C:\Users\ms\Desktop\mango\output\idea_2\results_baseline"

DEFAULT_N_PCS = 6
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

    # Check alignment
    if X.shape[0] != len(sample_ids):
        raise RuntimeError(
            f"Row count mismatch: X has {X.shape[0]} rows, samples.csv has {len(sample_ids)}."
        )

    # Reindex y_df to sample_ids order
    y_df.index = y_df.index.astype(str)
    y_df = y_df.reindex(sample_ids)

    print(f"[INFO] Loaded X: {X.shape}")
    print(f"[INFO] Loaded y_traits: {y_df.shape}")
    print(f"[INFO] Samples: {len(sample_ids)}")

    return X, y_df, sample_ids


def load_cv_design(cv_dir: str) -> Dict[str, np.ndarray]:
    """
    Load all CV design CSVs from cv_dir.

    Each CSV must have columns: sample_id, fold
    Returns a dict: { scheme_name: fold_array (length n_samples) }
    """
    cv_designs: Dict[str, np.ndarray] = {}

    if not os.path.isdir(cv_dir):
        raise FileNotFoundError(f"CV design directory not found: {cv_dir}")

    for fname in os.listdir(cv_dir):
        if not fname.endswith(".csv"):
            continue

        fpath = os.path.join(cv_dir, fname)
        df = pd.read_csv(fpath)

        if "sample_id" not in df.columns or "fold" not in df.columns:
            print(f"  [WARN] Skipping {fname}: missing 'sample_id' or 'fold' column.")
            continue

        scheme_name = fname.replace(".csv", "")
        fold_arr = df["fold"].to_numpy()
        cv_designs[scheme_name] = fold_arr
        print(f"  [INFO] Loaded CV scheme '{scheme_name}' with {len(np.unique(fold_arr))} folds.")

    return cv_designs


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Pearson correlation; returns NaN if degenerate."""
    if y_true.size < 2:
        return float("nan")
    if np.allclose(y_true, y_true[0]) or np.allclose(y_pred, y_pred[0]):
        return float("nan")
    r = np.corrcoef(y_true, y_pred)[0, 1]
    return float(r)


def ridge_cv_naive(
    X: np.ndarray,
    y: np.ndarray,
    fold_ids: np.ndarray,
    seed: int = RANDOM_STATE,
    sample_ids: Optional[np.ndarray] = None,
    trait: str = "",
    scheme: str = "",
    scenario: str = "naive",
    feature_set: str = "STRUCT",
    model_name: str = "ridge",
    oof_rows: Optional[List[Dict[str, object]]] = None,
) -> Tuple[List[float], float]:
    """
    Ridge regression CV without PC correction.
    """
    unique_folds = sorted(set(int(f) for f in fold_ids))
    r_values: List[float] = []

    for fold in unique_folds:
        test_mask = fold_ids == fold
        train_mask = ~test_mask

        X_train = X[train_mask, :]
        X_test = X[test_mask, :]
        y_train = y[train_mask]
        y_test = y[test_mask]

        scaler = StandardScaler(with_mean=True, with_std=True)
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = Ridge(alpha=1.0, random_state=seed)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        if oof_rows is not None and sample_ids is not None:
            ids_test = sample_ids[test_mask]
            for sid, yt, yp in zip(ids_test, y_test, y_pred):
                oof_rows.append(
                    {
                        "source": "idea2",
                        "feature_set": feature_set,
                        "model": model_name,
                        "scheme": scheme,
                        "scenario": scenario,
                        "trait": trait,
                        "fold": int(fold),
                        "sample_id": str(sid),
                        "y_true": float(yt),
                        "y_pred": float(yp),
                        "y_true_resid": np.nan,
                        "y_pred_resid": np.nan,
                        "y_fixed_pred": np.nan,
                    }
                )

        r = pearson_r(y_test, y_pred)
        r_values.append(r)

    r_mean = float(np.nanmean(r_values)) if r_values else float("nan")
    return r_values, r_mean


def ridge_cv_pc_corrected(
    X: np.ndarray,
    y: np.ndarray,
    fold_ids: np.ndarray,
    n_pcs: int = 6,
    seed: int = RANDOM_STATE,
    sample_ids: Optional[np.ndarray] = None,
    trait: str = "",
    scheme: str = "",
    scenario: str = "pc_corrected",
    feature_set: str = "STRUCT",
    model_name: str = "ridge",
    oof_rows: Optional[List[Dict[str, object]]] = None,
) -> Tuple[List[float], float]:
    """
    Ridge regression CV with within-fold PC correction.

    Steps per fold:
      1. Fit PCA on training X, transform train and test.
      2. Regress y_train on PCs to get residuals; apply same to y_test.
      3. Fit ridge on residuals.
      4. Predict on test residuals.
    """
    unique_folds = sorted(set(int(f) for f in fold_ids))
    r_values: List[float] = []

    for fold in unique_folds:
        test_mask = fold_ids == fold
        train_mask = ~test_mask

        X_train = X[train_mask, :]
        X_test = X[test_mask, :]
        y_train = y[train_mask]
        y_test = y[test_mask]

        n_samples_train = X_train.shape[0]
        n_snps = X_train.shape[1]
        max_pcs = min(n_pcs, n_samples_train - 1, n_snps)

        if max_pcs < 1:
            r_values.append(float("nan"))
            continue

        # Scale for PCA
        scaler_pc = StandardScaler(with_mean=True, with_std=True)
        X_train_scaled = scaler_pc.fit_transform(X_train)
        X_test_scaled = scaler_pc.transform(X_test)

        # PCA
        pca = PCA(n_components=max_pcs, random_state=seed)
        PCs_train = pca.fit_transform(X_train_scaled)
        PCs_test = pca.transform(X_test_scaled)

        # Regress y on PCs
        lr = LinearRegression()
        lr.fit(PCs_train, y_train)
        y_hat_train = lr.predict(PCs_train)
        y_hat_test = lr.predict(PCs_test)

        y_train_res = y_train - y_hat_train
        y_test_res = y_test - y_hat_test

        # Ridge on residuals
        scaler_ridge = StandardScaler(with_mean=True, with_std=True)
        X_train_ridge = scaler_ridge.fit_transform(X_train)
        X_test_ridge = scaler_ridge.transform(X_test)

        model = Ridge(alpha=1.0, random_state=seed)
        model.fit(X_train_ridge, y_train_res)
        y_pred_res = model.predict(X_test_ridge)

        y_pred_pheno = y_hat_test + y_pred_res

        if oof_rows is not None and sample_ids is not None:
            ids_test = sample_ids[test_mask]
            for sid, yt, yp, ytr, ypr, yfix in zip(
                ids_test, y_test, y_pred_pheno, y_test_res, y_pred_res, y_hat_test
            ):
                oof_rows.append(
                    {
                        "source": "idea2",
                        "feature_set": feature_set,
                        "model": model_name,
                        "scheme": scheme,
                        "scenario": scenario,
                        "trait": trait,
                        "fold": int(fold),
                        "sample_id": str(sid),
                        "y_true": float(yt),
                        "y_pred": float(yp),
                        "y_true_resid": float(ytr),
                        "y_pred_resid": float(ypr),
                        "y_fixed_pred": float(yfix),
                    }
                )

        r = pearson_r(y_test_res, y_pred_res)
        r_values.append(r)

    r_mean = float(np.nanmean(r_values)) if r_values else float("nan")
    return r_values, r_mean


# =========================
# MAIN
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline ridge regression models for Mango GS Idea 2."
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
        help=f"Output directory for baseline results (default: {DEFAULT_OUTDIR})",
    )
    parser.add_argument(
        "--n-pcs",
        type=int,
        default=DEFAULT_N_PCS,
        help=f"Number of PCs for PC correction (default: {DEFAULT_N_PCS})",
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
    print("Mango GS - Idea 2: Baseline linear models (ridge)")
    print("=" * 72)
    print(f"[INFO] X_full:       {args.X_path}")
    print(f"[INFO] y_traits:     {args.y_path}")
    print(f"[INFO] samples:      {args.samples_path}")
    print(f"[INFO] CV dir:       {args.cv_dir}")
    print(f"[INFO] Output dir:   {args.outdir}")
    print(f"[INFO] N_PCS:        {args.n_pcs}")
    print("")

    safe_mkdir(args.outdir)

    # 1) Load core matrices
    X, y_df, sample_ids = load_core_matrices(
        X_path=args.X_path,
        y_path=args.y_path,
        samples_path=args.samples_path,
    )

    # 2) Load CV designs (fold arrays are for ALL 225 samples, in the same order)
    cv_designs = load_cv_design(args.cv_dir)

    traits = list(y_df.columns)
    print(f"[INFO] Traits: {traits}")
    print(f"[INFO] CV schemes: {list(cv_designs.keys())}")
    print("")

    # Prepare collectors
    records_perfold = []
    records_summary = []
    oof_rows: List[Dict[str, object]] = []

    # 3) Loop over traits and CV schemes
    for trait in traits:
        print(f"[TRAIT] {trait}")

        # Full y for this trait (all samples)
        y_full = y_df[trait].to_numpy(dtype=float)

        # Drop samples with NaN phenotype for THIS trait
        mask_good = ~np.isnan(y_full)
        n_total = len(y_full)
        n_used = int(mask_good.sum())
        n_drop = n_total - n_used

        if n_drop > 0:
            print(f"  [INFO] Dropping {n_drop}/{n_total} samples with NaN for trait '{trait}'.")

        if n_used < 3:
            print(f"  [WARN] Only {n_used} non-NaN samples for trait '{trait}'. Skipping this trait.")
            continue

        X_trait = X[mask_good, :]
        y_trait = y_full[mask_good]
        ids_good = np.array(sample_ids)[mask_good]

        for scheme_name, fold_arr in cv_designs.items():
            # Subset fold_arr to non-NaN samples
            folds = fold_arr[mask_good]

            print(f"  [SCHEME] {scheme_name}")

            unique_folds = sorted(set(int(f) for f in folds))

            # Naive ridge
            r_vals_naive, r_mean_naive = ridge_cv_naive(
                X=X_trait,
                y=y_trait,
                fold_ids=folds,
                seed=args.seed,
                sample_ids=ids_good,
                trait=trait,
                scheme=scheme_name,
                scenario="naive",
                feature_set="STRUCT",
                model_name="ridge",
                oof_rows=oof_rows,
            )
            print(f"    Naive:       mean_r = {r_mean_naive:.4f}")

            for fold_label, r_val in zip(unique_folds, r_vals_naive):
                records_perfold.append(
                    {
                        "trait": trait,
                        "scheme": scheme_name,
                        "model_type": "naive",
                        "fold": fold_label,
                        "r": r_val,
                        "n_used": n_used,
                        "n_total": n_total,
                    }
                )
            records_summary.append(
                {
                    "trait": trait,
                    "scheme": scheme_name,
                    "model_type": "naive",
                    "mean_r": r_mean_naive,
                    "std_r": float(np.nanstd(r_vals_naive)),
                    "n_folds": len(r_vals_naive),
                    "n_used": n_used,
                    "n_total": n_total,
                }
            )

            # PC-corrected ridge
            r_vals_pc, r_mean_pc = ridge_cv_pc_corrected(
                X=X_trait,
                y=y_trait,
                fold_ids=folds,
                n_pcs=args.n_pcs,
                seed=args.seed,
                sample_ids=ids_good,
                trait=trait,
                scheme=scheme_name,
                scenario="pc_corrected",
                feature_set="STRUCT",
                model_name="ridge",
                oof_rows=oof_rows,
            )
            print(f"    PC-corrected: mean_r = {r_mean_pc:.4f}")

            for fold_label, r_val in zip(unique_folds, r_vals_pc):
                records_perfold.append(
                    {
                        "trait": trait,
                        "scheme": scheme_name,
                        "model_type": "pc_corrected",
                        "fold": fold_label,
                        "r": r_val,
                        "n_used": n_used,
                        "n_total": n_total,
                    }
                )
            records_summary.append(
                {
                    "trait": trait,
                    "scheme": scheme_name,
                    "model_type": "pc_corrected",
                    "mean_r": r_mean_pc,
                    "std_r": float(np.nanstd(r_vals_pc)),
                    "n_folds": len(r_vals_pc),
                    "n_used": n_used,
                    "n_total": n_total,
                }
            )

        print("")

    # 4) Save outputs
    perfold_df = pd.DataFrame.from_records(records_perfold)
    summary_df = pd.DataFrame.from_records(records_summary)

    perfold_path = os.path.join(args.outdir, "baseline_ridge_perfold.csv")
    summary_path = os.path.join(args.outdir, "baseline_ridge_summary.csv")

    perfold_df.to_csv(perfold_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"[SAVE] Per-fold metrics -> {perfold_path}")
    print(f"[SAVE] Summary metrics  -> {summary_path}")

    if oof_rows:
        oof_df = pd.DataFrame(oof_rows)
        oof_path = os.path.join(args.outdir, "baseline_ridge_oof_predictions.csv")
        print(f"[SAVE] OOF predictions -> {oof_path}")
        oof_df.to_csv(oof_path, index=False)

    print("")
    print("[OK] Baseline ridge models for Idea 2 completed.")


if __name__ == "__main__":
    main()
