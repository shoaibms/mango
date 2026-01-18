#!/usr/bin/env python
r"""
12_results_summary_initial.py

Collect key outputs from Idea 2 into two tidy summary files:

1) Model performance:
   - C:\Users\ms\Desktop\mango\output\idea_2\summary\idea2_model_performance_long.csv

2) Candidate genes (from GWAS+ML+GFF):
   - C:\Users\ms\Desktop\mango\output\idea_2\summary\idea2_candidate_genes_alltraits.csv
"""

import os
import glob
import pandas as pd

ROOT_OUT = r"C:\Users\ms\Desktop\mango\output\idea_2"

BASELINE_SUMMARY = os.path.join(ROOT_OUT, "results_baseline", "baseline_ridge_summary.csv")
XGBRF_SUMMARY = os.path.join(ROOT_OUT, "results_xgb_rf", "results_xgb_rf_summary.csv")
INV_SUMMARY = os.path.join(ROOT_OUT, "results_inversion", "inversion_gs_summary.csv")

GENES_DIR = os.path.join(ROOT_OUT, "genes")
SUMMARY_DIR = os.path.join(ROOT_OUT, "summary")

os.makedirs(SUMMARY_DIR, exist_ok=True)


def load_baseline():
    """
    Load baseline ridge summary CSV and standardise column names.
    
    Expected input columns from Script 03: trait, scheme, model_type, mean_r, std_r, n_folds, n_used, n_total
    FIX: Now handles both 'model_type' (correct) and 'mode' (legacy) column names for backwards compatibility.
    """
    if not os.path.isfile(BASELINE_SUMMARY):
        print(f"[WARN] Baseline summary not found: {BASELINE_SUMMARY}")
        return pd.DataFrame()

    df = pd.read_csv(BASELINE_SUMMARY)
    df_out = df.copy()
    df_out["model_family"] = "linear"
    df_out["feature_set"] = "snp"  # baseline uses SNP-only
    
    # FIX: Handle both 'model_type' (new correct name) and 'mode' (old name) for compatibility
    if "model_type" in df_out.columns:
        df_out.rename(columns={"model_type": "model"}, inplace=True)
    elif "mode" in df_out.columns:
        # Backwards compatibility with old Script 03 outputs
        df_out.rename(columns={"mode": "model"}, inplace=True)
    else:
        print("[WARN] Baseline summary missing 'model_type' or 'mode' column; 'model' will be NA.")
        df_out["model"] = pd.NA
    
    return df_out


def load_xgbrf():
    if not os.path.isfile(XGBRF_SUMMARY):
        print(f"[WARN] XGB/RF summary not found: {XGBRF_SUMMARY}")
        return pd.DataFrame()

    df = pd.read_csv(XGBRF_SUMMARY)
    # columns: trait, scheme, model, mean_r, std_r, n_folds, n_used, n_total
    df_out = df.copy()
    df_out["model_family"] = df_out["model"]  # xgb or rf
    df_out["feature_set"] = "snp"  # still SNP-only
    return df_out


def load_inversion():
    if not os.path.isfile(INV_SUMMARY):
        print(f"[WARN] Inversion GS summary not found: {INV_SUMMARY}")
        return pd.DataFrame()

    df = pd.read_csv(INV_SUMMARY)
    # columns: trait, scheme, feature_set (snp/inv/snp+inv), model, mean_r, std_r, n_folds, n_used, n_total
    df_out = df.copy()
    df_out["model_family"] = df_out["model"]  # ridge/xgb/rf, consistent with others
    return df_out


def summarise_model_performance():
    df_baseline = load_baseline()
    df_xgbrf = load_xgbrf()
    df_inv = load_inversion()

    frames = []

    if not df_baseline.empty:
        frames.append(df_baseline)

    if not df_xgbrf.empty:
        frames.append(df_xgbrf)

    if not df_inv.empty:
        frames.append(df_inv)

    if not frames:
        print("[WARN] No model summary files found; skipping model summary.")
        return

    df_all = pd.concat(frames, ignore_index=True)

    # Standardise columns
    needed_cols = [
        "trait",
        "scheme",
        "model_family",
        "model",
        "feature_set",
        "mean_r",
        "std_r",
        "n_folds",
        "n_used",
        "n_total",
    ]
    for col in needed_cols:
        if col not in df_all.columns:
            df_all[col] = pd.NA

    df_all = df_all[needed_cols]

    out_path = os.path.join(SUMMARY_DIR, "idea2_model_performance_long.csv")
    df_all.to_csv(out_path, index=False)
    print(f"[SAVE] Model performance summary -> {out_path}")


def summarise_candidate_genes():
    if not os.path.isdir(GENES_DIR):
        print(f"[WARN] Genes dir not found: {GENES_DIR}")
        return

    pattern = os.path.join(GENES_DIR, "candidate_snps_with_genes_*.csv")
    files = glob.glob(pattern)
    if not files:
        print(f"[WARN] No candidate_snps_with_genes_*.csv files found in {GENES_DIR}")
        return

    frames = []
    for path in files:
        trait = os.path.basename(path).replace("candidate_snps_with_genes_", "").replace(".csv", "")
        df = pd.read_csv(path)
        df["trait_file"] = trait
        frames.append(df)

    df_all = pd.concat(frames, ignore_index=True)

    # Standardise a core set of columns for inspection
    keep_cols = [
        "trait",  # if present
        "trait_file",
        "snp_id",
        "chrom",
        "pos",
        "beta",
        "p",
        "log10p",
        "xgb_importance",
        "rf_importance",
        "gwas_rank",
        "xgb_rank",
        "rf_rank",
        "gene_id",
        "gene_name",
        "distance_to_gene_bp",
        "evidence",
    ]
    cols_exist = [c for c in keep_cols if c in df_all.columns]
    df_all = df_all[cols_exist]

    out_path = os.path.join(SUMMARY_DIR, "idea2_candidate_genes_alltraits.csv")
    df_all.to_csv(out_path, index=False)
    print(f"[SAVE] Candidate genes summary -> {out_path}")


def main():
    print("=== Summarising Idea 2 results ===")
    summarise_model_performance()
    summarise_candidate_genes()
    print("[OK] Idea 2 summaries written.")


if __name__ == "__main__":
    main()
