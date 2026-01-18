#!/usr/bin/env python
r"""
08_random_panel_benchmark.py

Random marker vs inversion control for Mango GS – Idea 2.

Goal
----
Test whether the curated 17 inversion markers (miinv*) carry
more predictive information than random sets of 17 SNPs.

We compare, for each trait × CV scheme × model:

    INV-only models  vs  Random 17-SNP models

using the same CV splits and model hyperparameters as in Idea 2.
Parallelised using joblib for efficient multi-core execution.
"""

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np

# =========================
# IMPORTS WITH CHECKS
# =========================

try:
    import pandas as pd
except ImportError as e:
    raise SystemExit("pandas is required. Install with: pip install pandas") from e

try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    raise SystemExit(
        "scikit-learn is required for Ridge and StandardScaler. Install with:\n\n  pip install scikit-learn\n"
    ) from e

try:
    from sklearn.ensemble import RandomForestRegressor
except ImportError as e:
    raise SystemExit(
        "scikit-learn is required for RandomForestRegressor. Install with:\n\n  pip install scikit-learn\n"
    ) from e

try:
    from xgboost import XGBRegressor
except ImportError as e:
    raise SystemExit("xgboost is required. Install with: pip install xgboost") from e

try:
    from joblib import Parallel, delayed
except ImportError as e:
    raise SystemExit("joblib is required for parallel processing. Install with: pip install joblib") from e


# =========================
# DEFAULT PATHS / PARAMS
# =========================

DEFAULT_X_PATH = r"C:\Users\ms\Desktop\mango\output\idea_2\core_ml\X_full.npy"
DEFAULT_Y_PATH = r"C:\Users\ms\Desktop\mango\output\idea_2\core_ml\y_traits.csv"
DEFAULT_SAMPLES_PATH = r"C:\Users\ms\Desktop\mango\output\idea_2\core_ml\samples.csv"
DEFAULT_CV_DIR = r"C:\Users\ms\Desktop\mango\output\idea_2\cv_design"

# Dataset S1 Excel (same as elsewhere)
DEFAULT_INV_XLSX_PATH = r"C:\Users\ms\Desktop\mango\data\main_data\nph20252-sup-0001-datasetss1-s3.xlsx"
DEFAULT_INV_SHEET = "Dataset S1"
DEFAULT_INV_SAMPLE_COL = "ID"

DEFAULT_OUTDIR = r"C:\Users\ms\Desktop\mango\output\idea_2\random_control"

RANDOM_STATE = 42
DEFAULT_N_RANDOM = 100
DEFAULT_N_MARKERS = 17

# Number of parallel jobs (-1 uses all available cores)
N_JOBS_PARALLEL = -1


# =========================
# UTILS: LOADING
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
        first = y_df.columns[0]
        y_df = y_df.set_index(first)
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


def load_inversion_matrix(
    inv_xlsx_path: str,
    sheet_name: str,
    sample_id_col: str,
    sample_ids: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Load inversion markers from Dataset S1 Excel and align them to sample_ids.
    """
    if not os.path.isfile(inv_xlsx_path):
        raise FileNotFoundError(f"Inversion Excel file not found: {inv_xlsx_path}")

    print(f"[LOAD] Inversion data -> {inv_xlsx_path} (sheet='{sheet_name}')")
    df = pd.read_excel(inv_xlsx_path, sheet_name=sheet_name)
    if sample_id_col not in df.columns:
        raise RuntimeError(
            f"Sample ID column '{sample_id_col}' not found in Dataset S1. "
            f"Columns: {list(df.columns)}"
        )

    df[sample_id_col] = df[sample_id_col].astype(str)
    df = df.set_index(sample_id_col)

    # Detect inversion columns (prefix 'miinv')
    inv_cols = [c for c in df.columns if str(c).startswith("miinv")]
    if not inv_cols:
        raise RuntimeError(
            f"No inversion columns detected (names starting with 'miinv') in sheet '{sheet_name}'."
        )

    # Check sample coverage
    missing_inv = [sid for sid in sample_ids if sid not in df.index]
    if missing_inv:
        raise RuntimeError(
            f"{len(missing_inv)} samples in samples.csv missing in Dataset S1.\n"
            f"Example missing IDs: {missing_inv[:5]}"
        )

    df_inv = df.loc[sample_ids, inv_cols].copy()
    X_inv = df_inv.to_numpy(dtype=float)

    print(
        f"[INFO] Inversion matrix shape: {X_inv.shape} (samples x inversions); "
        f"inversion columns: {inv_cols}"
    )
    return X_inv, inv_cols


def load_cv_design(cv_dir: str) -> Dict[str, np.ndarray]:
    """
    Load all CV design files from cv_dir.
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

def pearson_r_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Pearson correlation, returning NaN if degenerate."""
    if y_true.size < 2:
        return float("nan")
    if np.allclose(y_true, y_true[0]) or np.allclose(y_pred, y_pred[0]):
        return float("nan")
    r = np.corrcoef(y_true, y_pred)[0, 1]
    return float(r)


def ridge_cv_with_fixed_folds(
    X: np.ndarray,
    y: np.ndarray,
    fold_ids: np.ndarray,
    seed: int = RANDOM_STATE
) -> Tuple[List[float], float]:
    """
    Ridge regression with pre-defined folds (with scaling).
    """
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
    seed: int = RANDOM_STATE
) -> Tuple[List[float], float]:
    """
    XGBoostRegressor with pre-defined folds.

    Note: n_jobs=1 to enable outer-loop parallelisation via joblib.
    """
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
            reg_alpha=0.0,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=1,
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
    seed: int = RANDOM_STATE
) -> Tuple[List[float], float]:
    """
    RandomForestRegressor with pre-defined folds.

    Note: n_jobs=1 to enable outer-loop parallelisation via joblib.
    """
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
            n_jobs=1,
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


# =========================
# PARALLEL WORKER FUNCTION
# =========================

def run_single_random_replicate(
    rep: int,
    n_snps: int,
    n_markers: int,
    X_snp_trait: np.ndarray,
    y_trait: np.ndarray,
    folds: np.ndarray,
    model_name: str,
    base_seed: int
) -> Dict:
    """
    Execute one iteration of random marker sampling and cross-validation.

    Designed for parallel execution via joblib.
    """
    # Create a unique seed for this replicate based on the base seed + replicate number
    current_seed = base_seed + rep
    rng = np.random.RandomState(current_seed)
    
    # Select random columns
    cols = rng.choice(n_snps, size=n_markers, replace=False)
    X_rand = X_snp_trait[:, cols]
    
    # Retrieve the correct model function
    model_func = MODEL_FUNCS[model_name]
    
    # Run CV (using the single-threaded version of the model)
    rand_r_vals, rand_r_mean = model_func(
        X=X_rand,
        y=y_trait,
        fold_ids=folds,
        seed=current_seed
    )
    
    return {
        "replicate": rep,
        "mean_r_random": float(rand_r_mean)
    }


# =========================
# ARGPARSE / MAIN
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Random 17-SNP vs inversion-only control for Mango GS Idea 2."
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
        help=f"Output directory for random vs inversion control results (default: {DEFAULT_OUTDIR})",
    )
    parser.add_argument(
        "--traits",
        type=str,
        nargs="*",
        default=["FBC", "AFW", "TSS", "TC"],
        help="Traits to analyse (default: FBC AFW TSS TC)",
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
        default=["xgb", "rf"],
        choices=["ridge", "xgb", "rf"],
        help="Models to compare (default: xgb rf)",
    )
    parser.add_argument(
        "--n-random",
        type=int,
        default=DEFAULT_N_RANDOM,
        help=f"Number of random 17-SNP sets per trait × scheme × model (default: {DEFAULT_N_RANDOM})",
    )
    parser.add_argument(
        "--n-markers",
        type=int,
        default=DEFAULT_N_MARKERS,
        help=f"Number of markers per random set (default: {DEFAULT_N_MARKERS})",
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
    print("Mango GS – Idea 2: Random 17-SNP vs inversion-only control")
    print(f"Parallel execution using joblib (n_jobs={N_JOBS_PARALLEL})")
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
    print(f"[INFO] n_random:      {args.n_random}")
    print(f"[INFO] n_markers:     {args.n_markers}")
    print(f"[INFO] seed:          {args.seed}")
    print("")

    safe_mkdir(args.outdir)

    # 1) Core data
    X_snp, y_df, sample_ids = load_core_matrices(
        X_path=args.X_path,
        y_path=args.y_path,
        samples_path=args.samples_path,
    )
    n_samples, n_snps = X_snp.shape

    # 2) Inversion matrix
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

    # 3) CV designs
    cv_designs = load_cv_design(args.cv_dir)
    print(f"[INFO] CV schemes available: {list(cv_designs.keys())}")
    print("")

    # consistency check
    for scheme_name, folds in cv_designs.items():
        if len(folds) != n_samples:
            raise RuntimeError(
                f"Fold array length mismatch for {scheme_name}: "
                f"{len(folds)} vs n_samples={n_samples}"
            )

    traits_available = list(y_df.columns)
    print(f"[INFO] Traits available in y_traits.csv: {traits_available}")
    print("")

    # collectors
    rec_repl = []   # random set replicates
    rec_sum = []    # inversion vs random summary

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

            for model_name in args.models:
                if model_name not in MODEL_FUNCS:
                    print(f"    [WARN] Model '{model_name}' not implemented; skipping.")
                    continue

                print(f"    [MODEL] {model_name}")
                model_func = MODEL_FUNCS[model_name]

                # Inversion-only performance
                inv_r_vals, inv_r_mean = model_func(
                    X=X_inv_trait,
                    y=y_trait,
                    fold_ids=folds,
                )
                inv_r_std = float(np.nanstd(inv_r_vals)) if inv_r_vals else float("nan")
                print(
                    f"      [INV] mean r = {inv_r_mean:.3f}, "
                    f"std r = {inv_r_std:.3f} over {len(inv_r_vals)} folds"
                )

                # --- Random 17-SNP sets (parallel execution) ---
                print(f"      [RANDOM] Starting {args.n_random} replicates (n_jobs={N_JOBS_PARALLEL})...")

                parallel_results = Parallel(n_jobs=N_JOBS_PARALLEL, verbose=5)(
                    delayed(run_single_random_replicate)(
                        rep=rep,
                        n_snps=n_snps,
                        n_markers=args.n_markers,
                        X_snp_trait=X_snp_trait,
                        y_trait=y_trait,
                        folds=folds,
                        model_name=model_name,
                        base_seed=args.seed
                    )
                    for rep in range(1, args.n_random + 1)
                )

                # Collect results back into list
                rand_means: List[float] = []
                for res in parallel_results:
                    # Add metadata
                    rec_repl.append(
                        {
                            "trait": trait,
                            "scheme": scheme_name,
                            "model": model_name,
                            "n_markers": args.n_markers,
                            "replicate": res["replicate"],
                            "mean_r_random": res["mean_r_random"],
                        }
                    )
                    rand_means.append(res["mean_r_random"])

                # Calculate statistics
                rand_means_arr = np.array(rand_means, dtype=float)
                rand_mean = float(np.nanmean(rand_means_arr))
                rand_std = float(np.nanstd(rand_means_arr))
                rand_min = float(np.nanmin(rand_means_arr))
                rand_max = float(np.nanmax(rand_means_arr))

                # empirical p-value: fraction of random sets >= inversion mean
                n_ge = int(np.sum(rand_means_arr >= inv_r_mean))
                p_emp = (n_ge + 1) / (args.n_random + 1)  # add-one for stability

                print(
                     f"      [RESULT] Random mean r = {rand_mean:.3f} (p_emp = {p_emp:.4f})"
                )

                rec_sum.append(
                    {
                        "trait": trait,
                        "scheme": scheme_name,
                        "model": model_name,
                        "n_markers": args.n_markers,
                        "inversion_mean_r": float(inv_r_mean),
                        "inversion_std_r": float(inv_r_std),
                        "random_mean_r": rand_mean,
                        "random_std_r": rand_std,
                        "random_min_r": rand_min,
                        "random_max_r": rand_max,
                        "n_random": args.n_random,
                        "n_random_ge_inversion": n_ge,
                        "p_empirical": float(p_emp),
                    }
                )

        print("")

    # Save outputs
    repl_df = pd.DataFrame.from_records(rec_repl)
    sum_df = pd.DataFrame.from_records(rec_sum)

    repl_path = os.path.join(args.outdir, "random_vs_inversion_replicates.csv")
    sum_path = os.path.join(args.outdir, "random_vs_inversion_summary.csv")

    repl_df.to_csv(repl_path, index=False)
    sum_df.to_csv(sum_path, index=False)

    print(f"[SAVE] Random replicate results   -> {repl_path}")
    print(f"[SAVE] Inversion vs random summary -> {sum_path}")
    print("")
    print("[OK] Random vs inversion control for Idea 2 completed.")


if __name__ == "__main__":
    main()