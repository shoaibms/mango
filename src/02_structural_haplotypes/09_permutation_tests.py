#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
09_permutation_tests.py


Permutation tests (sanity check)

This script tests whether your observed cross-validated accuracy
(for SNP, INV, and SNP+INV models) could arise by chance.

For each trait x CV scheme x model x feature set:
  1) Compute the real cross-validated Pearson r.
  2) Run n_perm permutations of the phenotype vector.
  3) Build an empirical null distribution for r and compute an empirical p-value.

Outputs:
  - permutation_replicates.csv
  - permutation_summary.csv
"""

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from joblib import Parallel, delayed
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Defaults / constants
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
N_JOBS_PARALLEL = -1  # use all cores

DEFAULT_X_PATH = r"C:\Users\ms\Desktop\mango\output\idea_2\core_ml\X_full.npy"
DEFAULT_Y_PATH = r"C:\Users\ms\Desktop\mango\output\idea_2\core_ml\y_traits.csv"
DEFAULT_SAMPLES_PATH = r"C:\Users\ms\Desktop\mango\output\idea_2\core_ml\samples.csv"
DEFAULT_CV_DIR = r"C:\Users\ms\Desktop\mango\output\idea_2\cv_design"

DEFAULT_INV_XLSX_PATH = r"C:\Users\ms\Desktop\mango\data\main_data\nph20252-sup-0001-datasetss1-s3.xlsx"
DEFAULT_INV_SHEET = "Dataset S1"
DEFAULT_INV_SAMPLE_COL = "ID"

DEFAULT_OUTDIR = r"C:\Users\ms\Desktop\mango\output\idea_2\permutation_tests"
DEFAULT_N_PERM = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_core_matrices(
    X_path: str,
    y_path: str,
    samples_path: str,
) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    """Load SNP matrix (X), phenotype table (y_df), and sample IDs."""
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

    # Reindex y_df to sample_ids order
    y_df.index = y_df.index.astype(str)
    y_df = y_df.reindex(sample_ids)

    print(f"[INFO] Loaded X: {X.shape}")
    print(f"[INFO] Loaded y_traits: {y_df.shape}")
    print(f"[INFO] Samples: {len(sample_ids)}")

    return X, y_df, sample_ids


def load_inversion_matrix(
    inv_xlsx_path: str,
    sheet_name: str,
    sample_id_col: str,
    sample_ids: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Load inversion markers from Dataset S1 Excel and align to sample_ids.
    Returns X_inv (n_samples x n_inv) and a list of inversion column names.
    """
    if not os.path.isfile(inv_xlsx_path):
        raise FileNotFoundError(f"Inversion Excel not found: {inv_xlsx_path}")

    df = pd.read_excel(inv_xlsx_path, sheet_name=sheet_name)
    if sample_id_col not in df.columns:
        raise RuntimeError(f"Sample ID column '{sample_id_col}' not in sheet '{sheet_name}'.")

    df[sample_id_col] = df[sample_id_col].astype(str)
    df = df.set_index(sample_id_col)

    # Find inversion columns (start with "miinv")
    inv_cols = [c for c in df.columns if c.lower().startswith("miinv")]
    if not inv_cols:
        raise RuntimeError("No 'miinv*' columns found in Dataset S1.")

    # Align to sample_ids
    df_aligned = df.reindex(sample_ids)
    X_inv = df_aligned[inv_cols].to_numpy(dtype=float)

    # Fill NaN with column median
    for j in range(X_inv.shape[1]):
        col = X_inv[:, j]
        mask = np.isnan(col)
        if mask.any():
            med = np.nanmedian(col)
            X_inv[mask, j] = med

    print(f"[INFO] Loaded inversion matrix: {X_inv.shape} (samples x inv markers)")
    print(f"[INFO] Inversion columns: {inv_cols}")

    return X_inv, inv_cols


def load_cv_design(cv_dir: str) -> Dict[str, np.ndarray]:
    """Load all CV design CSVs from cv_dir. Returns {scheme_name: fold_array}."""
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


def pearson_r_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Pearson correlation; returns NaN if degenerate."""
    if y_true.size < 2:
        return float("nan")
    if np.allclose(y_true, y_true[0]) or np.allclose(y_pred, y_pred[0]):
        return float("nan")
    r = np.corrcoef(y_true, y_pred)[0, 1]
    return float(r)


# ---------------------------------------------------------------------------
# Model CV functions
# ---------------------------------------------------------------------------

def ridge_cv_with_fixed_folds(
    X: np.ndarray,
    y: np.ndarray,
    fold_ids: np.ndarray,
    seed: int = RANDOM_STATE,
) -> Tuple[List[float], float]:
    """Ridge regression CV with fixed folds."""
    unique_folds = sorted(set(int(f) for f in fold_ids))
    r_values: List[float] = []

    for fold in unique_folds:
        test_mask = (fold_ids == fold)
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

        r = pearson_r_safe(y_test, y_pred)
        r_values.append(r)

    r_mean = float(np.nanmean(r_values)) if r_values else float("nan")
    return r_values, r_mean


def xgb_cv_with_fixed_folds(
    X: np.ndarray,
    y: np.ndarray,
    fold_ids: np.ndarray,
    seed: int = RANDOM_STATE,
) -> Tuple[List[float], float]:
    """XGBoost CV with fixed folds."""
    unique_folds = sorted(set(int(f) for f in fold_ids))
    r_values: List[float] = []

    for fold in unique_folds:
        test_mask = (fold_ids == fold)
        train_mask = ~test_mask

        X_train = X[train_mask, :]
        X_test = X[test_mask, :]
        y_train = y[train_mask]
        y_test = y[test_mask]

        model = XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.7,
            colsample_bytree=0.3,
            reg_lambda=1.0,
            n_jobs=1,        # per-fold
            random_state=seed,
            tree_method="hist",
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r = pearson_r_safe(y_test, y_pred)
        r_values.append(r)

    r_mean = float(np.nanmean(r_values)) if r_values else float("nan")
    return r_values, r_mean


def rf_cv_with_fixed_folds(
    X: np.ndarray,
    y: np.ndarray,
    fold_ids: np.ndarray,
    seed: int = RANDOM_STATE,
) -> Tuple[List[float], float]:
    """Random Forest CV with fixed folds."""
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
            n_jobs=1,        # per-fold
            random_state=seed,
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r = pearson_r_safe(y_test, y_pred)
        r_values.append(r)

    r_mean = float(np.nanmean(r_values)) if r_values else float("nan")
    return r_values, r_mean


MODEL_FUNCS = {
    "ridge": ridge_cv_with_fixed_folds,
    "xgb": xgb_cv_with_fixed_folds,
    "rf": rf_cv_with_fixed_folds,
}


# ---------------------------------------------------------------------------
# Permutation worker
# ---------------------------------------------------------------------------

def run_single_permutation(
    rep: int,
    X_feat: np.ndarray,
    y_trait: np.ndarray,
    folds: np.ndarray,
    model_name: str,
    base_seed: int,
) -> Dict:
    """Single permutation replicate: shuffle y, recompute CV r."""
    current_seed = base_seed + rep
    rng = np.random.RandomState(current_seed)

    y_perm = rng.permutation(y_trait)
    model_func = MODEL_FUNCS[model_name]

    _, perm_r_mean = model_func(
        X=X_feat,
        y=y_perm,
        fold_ids=folds,
        seed=current_seed,
    )
    return {
        "replicate": rep,
        "mean_r_perm": float(perm_r_mean),
    }


# ---------------------------------------------------------------------------
# Arg parsing / main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Permutation tests for Mango GS Idea 2 (SNP vs INV vs SNP+INV)."
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
        "--inv-xlsx-path",
        type=str,
        default=DEFAULT_INV_XLSX_PATH,
        help=f"Path to Dataset S1 Excel with inversion markers (default: {DEFAULT_INV_XLSX_PATH})",
    )
    parser.add_argument(
        "--inv-sheet",
        type=str,
        default=DEFAULT_INV_SHEET,
        help=f"Sheet name in Dataset S1 Excel (default: {DEFAULT_INV_SHEET})",
    )
    parser.add_argument(
        "--inv-sample-col",
        type=str,
        default=DEFAULT_INV_SAMPLE_COL,
        help=f"Column in Dataset S1 with sample IDs (default: {DEFAULT_INV_SAMPLE_COL})",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=DEFAULT_OUTDIR,
        help=f"Output directory for permutation test results (default: {DEFAULT_OUTDIR})",
    )
    parser.add_argument(
        "--traits",
        type=str,
        nargs="*",
        default=["FBC", "AFW", "TSS", "TC", "FF"],
        help="Traits to analyse (default: FBC AFW TSS TC FF)",
    )
    parser.add_argument(
        "--schemes",
        type=str,
        nargs="*",
        default=["cv_random_k5", "cv_cluster_kmeans"],
        help="CV scheme names to use (must match files in cv_dir; default: cv_random_k5 cv_cluster_kmeans)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=["xgb", "rf", "ridge"],
        choices=["ridge", "xgb", "rf"],
        help="Models to test (default: xgb rf ridge)",
    )
    parser.add_argument(
        "--features",
        type=str,
        nargs="*",
        default=["snp", "inv", "snp+inv"],
        choices=["snp", "inv", "snp+inv"],
        help="Feature sets to test (default: snp inv snp+inv)",
    )
    parser.add_argument(
        "--n-perm",
        type=int,
        default=DEFAULT_N_PERM,
        help=f"Number of phenotype permutations per trait x scheme x model x feature set (default: {DEFAULT_N_PERM})",
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
    print("Mango GS - Idea 2: Permutation tests (sanity check)")
    print(f"* OPTIMISED MODE: Using Joblib with {N_JOBS_PARALLEL} parallel jobs *")
    print("=" * 72)
    print(f"[INFO] X_full:        {args.X_path}")
    print(f"[INFO] y_traits:      {args.y_path}")
    print(f"[INFO] samples:       {args.samples_path}")
    print(f"[INFO] CV dir:        {args.cv_dir}")
    print(f"[INFO] Inv Excel:     {args.inv_xlsx_path}")
    print(f"[INFO] Inv sheet:     {args.inv_sheet}")
    print(f"[INFO] Inv sample col:{args.inv_sample_col}")
    print(f"[INFO] Output dir:    {args.outdir}")
    print(f"[INFO] Traits:        {args.traits}")
    print(f"[INFO] Schemes:       {args.schemes}")
    print(f"[INFO] Models:        {args.models}")
    print(f"[INFO] Features:      {args.features}")
    print(f"[INFO] n_perm:        {args.n_perm}")
    print(f"[INFO] seed:          {args.seed}")
    print("")

    safe_mkdir(args.outdir)

    # Load SNP + phenotypes + sample IDs
    X_snp, y_df, sample_ids = load_core_matrices(
        X_path=args.X_path,
        y_path=args.y_path,
        samples_path=args.samples_path,
    )
    n_samples, _ = X_snp.shape

    # Load inversion matrix aligned to samples
    X_inv, inv_cols = load_inversion_matrix(
        inv_xlsx_path=args.inv_xlsx_path,
        sheet_name=args.inv_sheet,
        sample_id_col=args.inv_sample_col,
        sample_ids=sample_ids,
    )
    if X_inv.shape[0] != n_samples:
        raise RuntimeError(
            f"Row mismatch between X_snp (n={n_samples}) and X_inv (n={X_inv.shape[0]})."
        )

    # Load CV schemes
    cv_designs = load_cv_design(args.cv_dir)
    print(f"[INFO] CV schemes available: {list(cv_designs.keys())}")
    print("")

    for scheme_name, folds in cv_designs.items():
        if len(folds) != n_samples:
            raise RuntimeError(
                f"Fold array length mismatch for {scheme_name}: {len(folds)} vs n_samples={n_samples}"
            )

    traits_available = list(y_df.columns)
    print(f"[INFO] Traits available in y_traits.csv: {traits_available}")
    print("")

    rec_repl = []
    rec_sum = []

    # ----------------------------------------------------------------------
    # Main loops over traits / schemes / feature sets / models
    # ----------------------------------------------------------------------
    for trait in args.traits:
        if trait not in y_df.columns:
            print(f"[WARN] Trait '{trait}' not found in y_traits.csv; skipping.")
            continue

        print(f"[TRAIT] {trait}")
        y_full = y_df[trait].to_numpy(dtype=float)
        mask_good = ~np.isnan(y_full)
        n_total = len(y_full)
        n_used = int(mask_good.sum())
        n_drop = n_total - n_used

        if n_used < 10:
            print(
                f"  [WARN] Only {n_used} non-NaN samples for trait '{trait}' "
                f"(total {n_total}); skipping."
            )
            print("")
            continue

        if n_drop > 0:
            print(f"  [INFO] Dropping {n_drop}/{n_total} samples with NaN for trait '{trait}'.")

        X_snp_trait = X_snp[mask_good, :]
        X_inv_trait = X_inv[mask_good, :]
        y_trait = y_full[mask_good]

        for scheme_name in args.schemes:
            if scheme_name not in cv_designs:
                print(f"  [WARN] Scheme '{scheme_name}' not found in CV designs; skipping.")
                continue

            folds_all = cv_designs[scheme_name]
            folds = folds_all[mask_good]
            unique_folds = sorted(set(int(f) for f in folds))
            print(f"  [SCHEME] {scheme_name} (folds={unique_folds})")

            for feat in args.features:
                if feat not in {"snp", "inv", "snp+inv"}:
                    print(f"    [WARN] Feature set '{feat}' not recognised; skipping.")
                    continue

                if feat == "snp":
                    X_feat = X_snp_trait
                elif feat == "inv":
                    X_feat = X_inv_trait
                else:  # snp+inv
                    X_feat = np.concatenate([X_snp_trait, X_inv_trait], axis=1)

                p = X_feat.shape[1]
                print(f"    [FEATURE] {feat} (p={p})")

                for model_name in args.models:
                    if model_name not in MODEL_FUNCS:
                        print(f"      [WARN] Model '{model_name}' not implemented; skipping.")
                        continue

                    print(f"      [MODEL] {model_name}")
                    model_func = MODEL_FUNCS[model_name]

                    # Real (non-permuted) performance
                    real_r_vals, real_r_mean = model_func(
                        X=X_feat,
                        y=y_trait,
                        fold_ids=folds,
                        seed=args.seed,
                    )
                    real_r_std = float(np.nanstd(real_r_vals)) if real_r_vals else float("nan")
                    print(
                        f"        [REAL] mean r = {real_r_mean:.3f}, "
                        f"std r = {real_r_std:.3f} over {len(real_r_vals)} folds"
                    )

                    # Permutations
                    print(f"        [PERM] Starting {args.n_perm} permutations on {N_JOBS_PARALLEL} workers.")

                    parallel_results = Parallel(n_jobs=N_JOBS_PARALLEL)(
                        delayed(run_single_permutation)(
                            rep=rep,
                            X_feat=X_feat,
                            y_trait=y_trait,
                            folds=folds,
                            model_name=model_name,
                            base_seed=args.seed,
                        )
                        for rep in tqdm(range(1, args.n_perm + 1), desc="Permutations", unit="perm")
                    )

                    perm_means: List[float] = []
                    for res in parallel_results:
                        rec_repl.append(
                            {
                                "trait": trait,
                                "scheme": scheme_name,
                                "model": model_name,
                                "feature_set": feat,
                                "replicate": res["replicate"],
                                "mean_r_perm": res["mean_r_perm"],
                            }
                        )
                        perm_means.append(res["mean_r_perm"])

                    perm_arr = np.array(perm_means, dtype=float)
                    perm_mean = float(np.nanmean(perm_arr))
                    perm_std = float(np.nanstd(perm_arr))
                    perm_min = float(np.nanmin(perm_arr))
                    perm_max = float(np.nanmax(perm_arr))

                    # Empirical p-value: P(null >= observed)
                    n_ge = int(np.sum(perm_arr >= real_r_mean))
                    p_emp = (n_ge + 1) / (args.n_perm + 1)

                    print(
                        f"        [RESULT] Perm mean r = {perm_mean:.3f}, "
                        f"p_emp = {p_emp:.4f}"
                    )

                    rec_sum.append(
                        {
                            "trait": trait,
                            "scheme": scheme_name,
                            "model": model_name,
                            "feature_set": feat,
                            "n_perm": args.n_perm,
                            "real_mean_r": float(real_r_mean),
                            "real_std_r": real_r_std,
                            "perm_mean_r": perm_mean,
                            "perm_std_r": perm_std,
                            "perm_min_r": perm_min,
                            "perm_max_r": perm_max,
                            "n_perm_ge_real": n_ge,
                            "p_empirical": float(p_emp),
                        }
                    )

        print("")

    # ----------------------------------------------------------------------
    # Save outputs
    # ----------------------------------------------------------------------
    repl_df = pd.DataFrame.from_records(rec_repl)
    sum_df = pd.DataFrame.from_records(rec_sum)

    repl_path = os.path.join(args.outdir, "permutation_replicates.csv")
    sum_path = os.path.join(args.outdir, "permutation_summary.csv")

    repl_df.to_csv(repl_path, index=False)
    sum_df.to_csv(sum_path, index=False)

    print(f"[SAVE] Permutation replicate results -> {repl_path}")
    print(f"[SAVE] Permutation summary           -> {sum_path}")
    print("")
    print("[OK] Permutation tests for Idea 2 completed.")


if __name__ == "__main__":
    main()
