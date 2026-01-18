#!/usr/bin/env python
r"""
05_model_comparison.py

Combine baseline ridge + tree-based (XGB/RF) results for Mango GS – Idea 2.

Inputs:
  - Baseline ridge summary:
      C:\Users\ms\Desktop\mango\output\idea_2\results_baseline\baseline_ridge_summary.csv

  - XGB/RF summary:
      C:\Users\ms\Desktop\mango\output\idea_2\results_xgb_rf\results_xgb_rf_summary.csv

Outputs:
  - Combined model comparison table:
      C:\Users\ms\Desktop\mango\output\idea_2\summary\model_comparison_metrics.csv

  Columns:
    trait
    scheme
    model          (ridge_naive, ridge_pc, xgb, rf)
    mean_r
    std_r
    n_folds
    n_used
    n_total
    rank_in_scheme (1 = best mean_r for that trait+scheme)
    is_best        (True for best model per trait+scheme)
"""

import argparse
import os

import numpy as np

try:
    import pandas as pd
except ImportError as e:
    raise SystemExit("pandas is required. Install with: pip install pandas") from e


# =========================
# DEFAULT PATHS
# =========================

DEFAULT_BASELINE_SUMMARY = r"C:\Users\ms\Desktop\mango\output\idea_2\results_baseline\baseline_ridge_summary.csv"
DEFAULT_XGBRF_SUMMARY = r"C:\Users\ms\Desktop\mango\output\idea_2\results_xgb_rf\results_xgb_rf_summary.csv"
DEFAULT_OUTDIR = r"C:\Users\ms\Desktop\mango\output\idea_2\summary"


# =========================
# UTILS
# =========================

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# =========================
# ARGS
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine baseline ridge and XGB/RF results for Mango GS Idea 2."
    )
    parser.add_argument(
        "--baseline-summary",
        type=str,
        default=DEFAULT_BASELINE_SUMMARY,
        help=f"Path to baseline_ridge_summary.csv (default: {DEFAULT_BASELINE_SUMMARY})",
    )
    parser.add_argument(
        "--xgbrf-summary",
        type=str,
        default=DEFAULT_XGBRF_SUMMARY,
        help=f"Path to results_xgb_rf_summary.csv (default: {DEFAULT_XGBRF_SUMMARY})",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=DEFAULT_OUTDIR,
        help=f"Output directory for combined summary (default: {DEFAULT_OUTDIR})",
    )
    return parser.parse_args()


# =========================
# MAIN
# =========================

def main() -> None:
    args = parse_args()

    print("=" * 72)
    print("Mango GS – Idea 2: Model comparison (ridge vs XGB vs RF)")
    print("=" * 72)
    print(f"[INFO] Baseline summary: {args.baseline_summary}")
    print(f"[INFO] XGB/RF summary:   {args.xgbrf_summary}")
    print(f"[INFO] Output dir:       {args.outdir}")
    print("")

    safe_mkdir(args.outdir)

    if not os.path.isfile(args.baseline_summary):
        raise FileNotFoundError(f"Baseline summary not found: {args.baseline_summary}")
    if not os.path.isfile(args.xgbrf_summary):
        raise FileNotFoundError(f"XGB/RF summary not found: {args.xgbrf_summary}")

    base_df = pd.read_csv(args.baseline_summary)
    xgbrf_df = pd.read_csv(args.xgbrf_summary)

    # --- Normalise baseline columns ----------------------------------------
    # We REQUIRE at least trait, scheme, mean_r
    required_base_core = {"trait", "scheme", "mean_r"}
    if not required_base_core.issubset(base_df.columns):
        raise RuntimeError(
            f"Baseline summary missing required columns: "
            f"{required_base_core - set(base_df.columns)}"
        )

    # Find a column that encodes the mode/model of the baseline:
    #   - modern scripts: 'mode'  (values: 'naive', 'pc_corrected')
    #   - older scripts:  'scenario' or 'model_type'
    #   - in some cases:  already 'model' (ridge_naive / ridge_pc)
    mode_col = None
    for cand in ["mode", "scenario", "model_type", "model"]:
        if cand in base_df.columns:
            mode_col = cand
            break

    if mode_col is None:
        raise RuntimeError(
            "Baseline summary has no mode/model column. Expected one of "
            "['mode', 'scenario', 'model_type', 'model'], "
            f"got columns={list(base_df.columns)}"
        )

    # Ensure n_used / n_total / n_folds / std_r exist
    for col in ["n_used", "n_total", "n_folds", "std_r"]:
        if col not in base_df.columns:
            base_df[col] = np.nan

    # Construct a unified 'model' column
    if mode_col == "model":
        # Already in 'model' form; just coerce to string
        base_df["model"] = base_df["model"].astype(str)
    else:
        raw_mode = base_df[mode_col].astype(str)

        # Normalise some common legacy values
        normalize_map = {
            "no_pc": "naive",
            "naive": "naive",
            "pc": "pc_corrected",
            "pc_corrected": "pc_corrected",
            "pc-corrected": "pc_corrected",
        }
        norm_mode = raw_mode.map(lambda x: normalize_map.get(x, x))

        mode_to_model = {
            "naive": "ridge_naive",
            "pc_corrected": "ridge_pc",
        }
        model = norm_mode.map(mode_to_model)

        # For any weird modes, just prefix with "ridge_"
        mask_na = model.isna()
        if mask_na.any():
            model.loc[mask_na] = "ridge_" + norm_mode.loc[mask_na].astype(str)

        base_df["model"] = model

    base_df_small = base_df[
        ["trait", "scheme", "model", "mean_r", "std_r", "n_folds", "n_used", "n_total"]
    ].copy()

    # --- Normalise XGB/RF columns ------------------------------------------
    required_xgbrf_cols = {"trait", "scheme", "model", "mean_r"}
    if not required_xgbrf_cols.issubset(xgbrf_df.columns):
        raise RuntimeError(
            f"XGB/RF summary missing required columns: "
            f"{required_xgbrf_cols - set(xgbrf_df.columns)}"
        )

    for col in ["n_used", "n_total", "n_folds", "std_r"]:
        if col not in xgbrf_df.columns:
            xgbrf_df[col] = np.nan

    xgbrf_df_small = xgbrf_df[
        ["trait", "scheme", "model", "mean_r", "std_r", "n_folds", "n_used", "n_total"]
    ].copy()

    # --- Combine -----------------------------------------------------------
    combined = pd.concat(
        [base_df_small, xgbrf_df_small],
        axis=0,
        ignore_index=True,
    )

    # Ensure consistent dtypes
    combined["trait"] = combined["trait"].astype(str)
    combined["scheme"] = combined["scheme"].astype(str)
    combined["model"] = combined["model"].astype(str)

    # Rank models within each trait × scheme by mean_r (higher is better)
    combined["rank_in_scheme"] = (
        combined.groupby(["trait", "scheme"])["mean_r"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    combined["is_best"] = combined["rank_in_scheme"] == 1

    # Save
    out_path = os.path.join(args.outdir, "model_comparison_metrics.csv")
    combined.to_csv(out_path, index=False)
    print(f"[SAVE] Combined model comparison -> {out_path}")

    # Print compact summary to console
    print("")
    print("[SUMMARY] Best model per trait × scheme:")
    summary_rows = (
        combined[combined["is_best"]]
        .sort_values(["trait", "scheme"])
        [["trait", "scheme", "model", "mean_r"]]
        .reset_index(drop=True)
    )
    for _, row in summary_rows.iterrows():
        print(
            f"  Trait={row['trait']:>4} | Scheme={row['scheme']:>15} | "
            f"Best={row['model']:>12} | mean_r={row['mean_r']:.3f}"
        )

    print("")
    print("[OK] Model comparison for Idea 2 completed.")


if __name__ == "__main__":
    main()
