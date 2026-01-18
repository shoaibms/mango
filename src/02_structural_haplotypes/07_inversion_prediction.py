#!/usr/bin/env python
r"""
07_inversion_prediction.py
Inversion-augmented genomic selection for Mango GS – Idea 2.

Compares:
  - SNP-only features (20k SNP panel)
  - inversion-only features (miinv*.0 etc. from Dataset S1)
  - SNP+inversion features (concatenated)

Models:
  - Ridge regression
  - XGBoostRegressor
  - RandomForestRegressor

CV schemes (pre-defined by Idea 2):
  - cv_random_k5
  - cv_cluster_kmeans

Inputs
------
Core ML data (Idea 2):
  - C:\Users\ms\Desktop\mango\output\idea_2\core_ml\X_full.npy
  - C:\Users\ms\Desktop\mango\output\idea_2\core_ml\y_traits.csv
  - C:\Users\ms\Desktop\mango\output\idea_2\core_ml\samples.csv

CV designs:
  - C:\Users\ms\Desktop\mango\output\idea_2\cv_design\cv_random_k5.csv
  - C:\Users\ms\Desktop\mango\output\idea_2\cv_design\cv_cluster_kmeans.csv

Inversion calls (Dataset S1 Excel):
  - C:\Users\ms\Desktop\mango\data\main_data\nph20252-sup-0001-datasetss1-s3.xlsx
    Sheet: "Dataset S1"
    Sample ID column: "ID"
    Inversion columns: columns whose names start with "miinv"

Outputs
-------
  - C:\Users\ms\Desktop\mango\output\idea_2\results_inversion\inversion_gs_perfold.csv
  - C:\Users\ms\Desktop\mango\output\idea_2\results_inversion\inversion_gs_summary.csv
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


# =========================
# DEFAULT PATHS / PARAMS
# =========================

DEFAULT_X_PATH = r"C:\Users\ms\Desktop\mango\output\idea_2\core_ml\X_full.npy"
DEFAULT_Y_PATH = r"C:\Users\ms\Desktop\mango\output\idea_2\core_ml\y_traits.csv"
DEFAULT_SAMPLES_PATH = r"C:\Users\ms\Desktop\mango\output\idea_2\core_ml\samples.csv"
DEFAULT_CV_DIR = r"C:\Users\ms\Desktop\mango\output\idea_2\cv_design"

# Dataset S1 Excel (same as in pilot/check scripts)
DEFAULT_INV_XLSX_PATH = r"C:\Users\ms\Desktop\mango\data\main_data\nph20252-sup-0001-datasetss1-s3.xlsx"
DEFAULT_INV_SHEET = "Dataset S1"
DEFAULT_INV_SAMPLE_COL = "ID"

DEFAULT_OUTDIR = r"C:\Users\ms\Desktop\mango\output\idea_2\results_inversion"

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

    - Reads the specified sheet.
    - Uses 'sample_id_col' as the key (e.g. 'ID').
    - Selects columns whose names start with 'miinv'.

    Returns:
        X_inv      : numpy array (n_samples x n_inv)
        inv_names  : list of inversion column names
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

    # Ensure numeric (0/1/2 etc.)
    X_inv = df_inv.to_numpy(dtype=float)
    print(
        f"[INFO] Inversion matrix shape: {X_inv.shape} (samples x inversions); "
        f"inversion columns: {inv_cols}"
    )
    return X_inv, inv_cols


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
) -> Tuple[List[float], float]:
    """
    Ridge regression with pre-defined folds (naive, no PC correction).

    Returns:
        r_values : list of Pearson r per fold
        r_mean   : mean r across folds
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

        model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        r = pearson_r_safe(y_test, y_pred)
        r_values.append(r)
        print(f"      Fold {fold}: r = {r:.3f} (ridge)")

    r_mean = float(np.nanmean(r_values)) if r_values else float("nan")
    return r_values, r_mean


def xgb_cv_with_fixed_folds(
    X: np.ndarray,
    y: np.ndarray,
    fold_ids: np.ndarray,
) -> Tuple[List[float], float]:
    """
    XGBoostRegressor with pre-defined folds.

    Returns:
        r_values : list of Pearson r per fold
        r_mean   : mean r across folds
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
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r = pearson_r_safe(y_test, y_pred)
        r_values.append(r)
        print(f"      Fold {fold}: r = {r:.3f} (xgb)")

    r_mean = float(np.nanmean(r_values)) if r_values else float("nan")
    return r_values, r_mean


def rf_cv_with_fixed_folds(
    X: np.ndarray,
    y: np.ndarray,
    fold_ids: np.ndarray,
) -> Tuple[List[float], float]:
    """
    RandomForestRegressor with pre-defined folds.

    Returns:
        r_values : list of Pearson r per fold
        r_mean   : mean r across folds
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
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r = pearson_r_safe(y_test, y_pred)
        r_values.append(r)
        print(f"      Fold {fold}: r = {r:.3f} (rf)")

    r_mean = float(np.nanmean(r_values)) if r_values else float("nan")
    return r_values, r_mean


# =========================
# MAIN
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inversion-augmented GS for Mango GS Idea 2 (SNP vs INV vs SNP+INV)."
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
        help=f"Output directory for inversion GS results (default: {DEFAULT_OUTDIR})",
    )
    parser.add_argument(
        "--traits",
        type=str,
        nargs="*",
        default=["FBC", "AFW", "TSS", "TC", "FF"],
        help="Traits to analyse (default: FBC AFW TSS TC FF)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 72)
    print("Mango GS – Idea 2: Inversion-augmented GS (SNP vs INV vs SNP+INV)")
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

    # 3) CV designs
    cv_designs = load_cv_design(args.cv_dir)
    print(f"[INFO] CV schemes: {list(cv_designs.keys())}")
    print("")

    # Collectors
    records_perfold = []
    records_summary = []

    traits_available = list(y_df.columns)
    print(f"[INFO] Traits available in y_traits.csv: {traits_available}")
    print("")

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
            print(
                f"  [INFO] Dropping {n_drop}/{n_total} samples with NaN for trait '{trait}'."
            )

        # Subset all matrices
        X_snp_trait = X_snp[mask_good, :]
        X_inv_trait = X_inv[mask_good, :]
        y_trait = y_full[mask_good]

        # Feature sets
        feature_sets = {
            "snp": X_snp_trait,
            "inv": X_inv_trait,
            "snp+inv": np.column_stack([X_snp_trait, X_inv_trait]),
        }

        for scheme_name, folds_all in cv_designs.items():
            if len(folds_all) != n_samples:
                raise RuntimeError(
                    f"Fold array length mismatch for {scheme_name}: "
                    f"{len(folds_all)} vs n_samples={n_samples}"
                )

            folds = folds_all[mask_good]
            unique_folds = sorted(set(int(f) for f in folds))
            print(f"  [SCHEME] {scheme_name} (folds={unique_folds})")

            for fset_name, X_feat in feature_sets.items():
                n_feat = X_feat.shape[1]
                print(f"    [FEATURE SET] {fset_name} (p={n_feat})")

                # Ridge
                print("    -> Ridge")
                r_vals_ridge, _ = ridge_cv_with_fixed_folds(
                    X=X_feat,
                    y=y_trait,
                    fold_ids=folds,
                )
                for fold_label, r_val in zip(unique_folds, r_vals_ridge):
                    records_perfold.append(
                        {
                            "trait": trait,
                            "scheme": scheme_name,
                            "feature_set": fset_name,
                            "model": "ridge",
                            "fold": int(fold_label),
                            "r": float(r_val),
                            "n_used": n_used,
                            "n_total": n_total,
                        }
                    )
                records_summary.append(
                    {
                        "trait": trait,
                        "scheme": scheme_name,
                        "feature_set": fset_name,
                        "model": "ridge",
                        "mean_r": float(np.nanmean(r_vals_ridge)),
                        "std_r": float(np.nanstd(r_vals_ridge)),
                        "n_folds": len(r_vals_ridge),
                        "n_used": n_used,
                        "n_total": n_total,
                    }
                )

                # XGB
                print("    -> XGBoost")
                r_vals_xgb, _ = xgb_cv_with_fixed_folds(
                    X=X_feat,
                    y=y_trait,
                    fold_ids=folds,
                )
                for fold_label, r_val in zip(unique_folds, r_vals_xgb):
                    records_perfold.append(
                        {
                            "trait": trait,
                            "scheme": scheme_name,
                            "feature_set": fset_name,
                            "model": "xgb",
                            "fold": int(fold_label),
                            "r": float(r_val),
                            "n_used": n_used,
                            "n_total": n_total,
                        }
                    )
                records_summary.append(
                    {
                        "trait": trait,
                        "scheme": scheme_name,
                        "feature_set": fset_name,
                        "model": "xgb",
                        "mean_r": float(np.nanmean(r_vals_xgb)),
                        "std_r": float(np.nanstd(r_vals_xgb)),
                        "n_folds": len(r_vals_xgb),
                        "n_used": n_used,
                        "n_total": n_total,
                    }
                )

                # RF
                print("    -> Random Forest")
                r_vals_rf, _ = rf_cv_with_fixed_folds(
                    X=X_feat,
                    y=y_trait,
                    fold_ids=folds,
                )
                for fold_label, r_val in zip(unique_folds, r_vals_rf):
                    records_perfold.append(
                        {
                            "trait": trait,
                            "scheme": scheme_name,
                            "feature_set": fset_name,
                            "model": "rf",
                            "fold": int(fold_label),
                            "r": float(r_val),
                            "n_used": n_used,
                            "n_total": n_total,
                        }
                    )
                records_summary.append(
                    {
                        "trait": trait,
                        "scheme": scheme_name,
                        "feature_set": fset_name,
                        "model": "rf",
                        "mean_r": float(np.nanmean(r_vals_rf)),
                        "std_r": float(np.nanstd(r_vals_rf)),
                        "n_folds": len(r_vals_rf),
                        "n_used": n_used,
                        "n_total": n_total,
                    }
                )

        print("")

    # Save outputs
    perfold_df = pd.DataFrame.from_records(records_perfold)
    summary_df = pd.DataFrame.from_records(records_summary)

    perfold_path = os.path.join(args.outdir, "inversion_gs_perfold.csv")
    summary_path = os.path.join(args.outdir, "inversion_gs_summary.csv")

    perfold_df.to_csv(perfold_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"[SAVE] Per-fold inversion GS metrics -> {perfold_path}")
    print(f"[SAVE] Summary inversion GS metrics  -> {summary_path}")
    print("")
    print("[OK] Inversion-augmented GS for Idea 2 completed.")


if __name__ == "__main__":
    main()
