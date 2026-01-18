# -*- coding: utf-8 -*-
"""
08_summarise_results.py

Mango GS – Idea 1: Global summary of structure, prediction, and GWAS integration.

This script consolidates all Idea 1 outputs into a compact set of summary tables
and plotting-ready CSVs.

It supports and enriches the main Result sections by:

  1. Phenotype summary (Table S1-style)
     - n, mean, sd, CV (%) for each fruit quality trait.

  2. Genetic structure summary
     - PCA + K-means clusters on genome-wide SNPs.
     - Cluster sizes and PC scores (for Figure 1A).

  3. Transferability of genomic prediction across structure (Table S2-style)
     - PC-corrected random K-fold ridge accuracy per trait.
     - Cluster-balanced and leave-cluster-out accuracy.
     - Absolute and % drops in accuracy (true cross-ancestry performance).

  4. GWAS-informed genomic prediction (Idea 1 vs naive ridge)
     - Baseline vs GWAS-weighted vs major-QTL+background models.
     - Per-trait deltas and t-test p-values (fold-level).

Outputs are written under:

  <idea1_root>/summary
    where idea1_root = parent directory of cfg.CORE_DATA_DIR
    e.g. .../output/idea_1/summary
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
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
except ImportError as e:
    raise SystemExit(
        "scikit-learn is required. Install with: pip install scikit-learn"
    ) from e

try:
    from scipy import stats
except ImportError as e:
    raise SystemExit("scipy is required. Install with: pip install scipy") from e

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit("matplotlib is required. Install with: pip install matplotlib") from e

import config_idea1 as cfg


# ======================================================================
# Helpers: output paths
# ======================================================================

def get_summary_dir() -> str:
    """
    Derive the Idea 1 summary directory from CORE_DATA_DIR and ensure it exists.

    If CORE_DATA_DIR is ".../output/idea_1/core_data",
    then summary_dir = ".../output/idea_1/summary".
    """
    cfg.ensure_output_dirs()
    idea1_root = os.path.abspath(os.path.join(cfg.CORE_DATA_DIR, os.pardir))
    summary_dir = os.path.join(idea1_root, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    return summary_dir


# ======================================================================
# 1. Phenotype summary (Table S1-style)
# ======================================================================

def summarise_phenotypes(pheno_path: str, summary_dir: str) -> pd.DataFrame:
    """
    Summarise phenotypes: n, mean, sd, CV (%) per trait.

    Returns the DataFrame and writes "pheno_trait_summary.csv".
    """
    print("[SECTION] Phenotype summary (Table S1-style)")
    if not os.path.exists(pheno_path):
        raise FileNotFoundError(
            f"pheno_core.csv not found at {pheno_path}. "
            "Run 01_build_core_matrices.py first."
        )

    pheno_df = pd.read_csv(pheno_path, index_col=0)
    pheno_df.index = pheno_df.index.astype(str)

    records: List[Dict[str, object]] = []

    for trait in cfg.TRAITS_DEFAULT:
        if trait not in pheno_df.columns:
            print(f"  [WARN] Trait {trait} not found in pheno_core; skipping.")
            continue

        y = pheno_df[trait].astype(float)
        mask = ~np.isnan(y.values)
        y_used = y.values[mask]

        n_used = int(mask.sum())
        mean = float(np.nanmean(y_used)) if n_used > 0 else np.nan
        sd = float(np.nanstd(y_used, ddof=1)) if n_used > 1 else np.nan
        if mean is None or mean == 0 or np.isnan(mean) or np.isnan(sd):
            cv = np.nan
        else:
            cv = float(100.0 * sd / abs(mean))

        records.append(
            {
                "trait": trait,
                "n_non_missing": n_used,
                "mean": mean,
                "sd": sd,
                "cv_percent": cv,
            }
        )

    pheno_summary = pd.DataFrame.from_records(records)
    out_path = os.path.join(summary_dir, "pheno_trait_summary.csv")
    print(f"  [INFO] Saving phenotype summary -> {out_path}")
    pheno_summary.to_csv(out_path, index=False)

    # Echo in console for quick sanity check
    if not pheno_summary.empty:
        print()
        print(pheno_summary.to_string(index=False))
    print()

    return pheno_summary


# ======================================================================
# 2. Genetic structure: PCA + K-means clusters (Figure 1A)
# ======================================================================

def compute_pca_and_clusters(
    geno_path: str,
    summary_dir: str,
    n_clusters: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Recompute PCA + K-means clusters on the core genotype matrix.

    This mirrors the logic in the structure-aware CV script:

      - Standardise G (samples x SNPs).
      - Use up to 10 PCs (or n_samples-1, n_snps, whichever is smaller).
      - K-means (n_clusters, random_state=cfg.RANDOM_STATE).

    Outputs:

      - pc_scores_clusters.csv
          sample_id, cluster, PC1, PC2, PC3, ..., PCn

      - cluster_sizes.csv
          cluster, n_samples
    """
    print("[SECTION] Genetic structure: PCA + K-means clusters")

    if not os.path.exists(geno_path):
        raise FileNotFoundError(
            f"geno_core.npz not found at {geno_path}. "
            "Run 01_build_core_matrices.py first."
        )

    npz = np.load(geno_path, allow_pickle=True)
    G = npz["G"]  # (n_samples, n_snps)
    sample_ids = npz["sample_ids"].astype(str)

    n_samples, n_snps = G.shape
    print(f"  [INFO] Genotype matrix: {n_samples} samples x {n_snps} SNPs")

    scaler = StandardScaler(with_mean=True, with_std=True)
    G_scaled = scaler.fit_transform(G)

    n_pcs = min(10, n_samples - 1, n_snps)
    if n_pcs <= 0:
        raise RuntimeError(
            f"Cannot compute PCs for clustering (n_pcs={n_pcs}). "
            "Check genotype dimensions."
        )

    print(f"  [INFO] Using {n_pcs} PCs for clustering")
    pca = PCA(n_components=n_pcs, random_state=cfg.RANDOM_STATE)
    PCs = pca.fit_transform(G_scaled)  # (n_samples, n_pcs)

    km = KMeans(
        n_clusters=n_clusters,
        random_state=cfg.RANDOM_STATE,
        n_init=10,
    )
    clusters = km.fit_predict(PCs)

    # Summarise cluster sizes
    unique, counts = np.unique(clusters, return_counts=True)
    cluster_records = [
        {"cluster": int(c), "n_samples": int(n)} for c, n in zip(unique, counts)
    ]
    cluster_df = pd.DataFrame.from_records(cluster_records).sort_values("cluster")

    cluster_out = os.path.join(summary_dir, "cluster_sizes.csv")
    print(f"  [INFO] Saving cluster sizes -> {cluster_out}")
    cluster_df.to_csv(cluster_out, index=False)

    # PC scores + cluster assignment (for Figure 1A)
    pc_cols = [f"PC{i+1}" for i in range(n_pcs)]
    pc_df = pd.DataFrame(PCs, columns=pc_cols)
    pc_df.insert(0, "sample_id", sample_ids)
    pc_df.insert(1, "cluster", clusters.astype(int))

    pcs_out = os.path.join(summary_dir, "pc_scores_clusters.csv")
    print(f"  [INFO] Saving PC scores + clusters -> {pcs_out}")
    pc_df.to_csv(pcs_out, index=False)

    print()
    print("  Cluster sizes:")
    if not cluster_df.empty:
        print(cluster_df.to_string(index=False))
    print()

    return pc_df, cluster_df


# ======================================================================
# 2b. PC3/PC4 substructure check (Americas vs South Asia within Cluster 1)
# ======================================================================

def check_pc34_substructure(
    pc_df: pd.DataFrame,
    summary_dir: str,
    origin_path: str = None,
    cluster_of_interest: int = 1,
    group_a: str = "Americas",
    group_b: str = "South Asia",
) -> pd.DataFrame:
    """
    Check whether PC3/PC4 separate Americas vs South Asia within a cluster.

    This addresses the reviewer question: "Is there substructure on PC3/PC4
    where Americas and South Asia cluster separately?"

    Parameters
    ----------
    pc_df : pd.DataFrame
        Output from compute_pca_and_clusters with sample_id, cluster, PC1..PCn.
    summary_dir : str
        Directory to save outputs.
    origin_path : str, optional
        Path to origin.csv with Sample_ID and Origin_Type columns.
        If None, attempts cfg.ORIGIN_PATH or skips.
    cluster_of_interest : int
        Which cluster to analyse (default 1).
    group_a, group_b : str
        Origin_Type labels to compare.

    Outputs
    -------
    - pc34_substructure_plot.png
    - pc34_substructure_summary.csv
    """
    print("[SECTION] PC3/PC4 substructure check (Americas vs South Asia)")

    # Locate origin file
    if origin_path is None:
        origin_path = getattr(cfg, "ORIGIN_PATH", None)
    if origin_path is None or not os.path.exists(origin_path):
        print(f"  [WARN] origin.csv not found at {origin_path}; skipping PC3/PC4 check.")
        return pd.DataFrame()

    origin = pd.read_csv(origin_path)

    # Standardise column names
    if "Sample_ID" in origin.columns:
        origin = origin.rename(columns={"Sample_ID": "sample_id"})
    if "sample_id" not in origin.columns:
        print("  [WARN] origin.csv missing sample_id/Sample_ID column; skipping.")
        return pd.DataFrame()

    origin["sample_id"] = origin["sample_id"].astype(str)
    pc_df = pc_df.copy()
    pc_df["sample_id"] = pc_df["sample_id"].astype(str)

    # Merge
    df = origin.merge(pc_df, on="sample_id", how="inner")
    print(f"  [INFO] Merged rows: {df.shape[0]}")

    if "Origin_Type" not in df.columns:
        print("  [WARN] Origin_Type column not found; skipping.")
        return pd.DataFrame()

    # Filter to cluster of interest and target groups
    cluster_col = "cluster"
    if cluster_col not in df.columns:
        print(f"  [WARN] '{cluster_col}' column not found; skipping.")
        return pd.DataFrame()

    sub = df[df[cluster_col] == cluster_of_interest].copy()
    sub = sub[sub["Origin_Type"].isin([group_a, group_b])].copy()

    print(f"  [INFO] Cluster {cluster_of_interest} rows with {group_a}/{group_b}: {sub.shape[0]}")
    print(sub["Origin_Type"].value_counts(dropna=False).to_string())

    if sub.shape[0] < 4:
        print("  [WARN] Too few samples for PC3/PC4 analysis.")
        return pd.DataFrame()

    # Check PC3/PC4 exist
    if "PC3" not in sub.columns or "PC4" not in sub.columns:
        print("  [WARN] PC3/PC4 not in data; skipping plot.")
        return pd.DataFrame()

    # --- PLOT PC3 vs PC4 ---
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"Americas": "#1f77b4", "South Asia": "#ff7f0e"}
    for g, d in sub.groupby("Origin_Type"):
        ax.scatter(
            d["PC3"], d["PC4"],
            label=g,
            alpha=0.7,
            s=50,
            c=colors.get(g, None),
        )
    ax.set_xlabel("PC3")
    ax.set_ylabel("PC4")
    ax.set_title(f"PC3 vs PC4 within Cluster {cluster_of_interest} ({group_a} vs {group_b})")
    ax.legend()
    plt.tight_layout()

    plot_out = os.path.join(summary_dir, "pc34_substructure_plot.png")
    fig.savefig(plot_out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [INFO] Saved plot -> {plot_out}")

    # --- COHEN'S D ---
    def cohens_d(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        nx, ny = len(x), len(y)
        if nx < 2 or ny < 2:
            return np.nan
        sx, sy = x.std(ddof=1), y.std(ddof=1)
        sp = np.sqrt(((nx - 1) * sx * sx + (ny - 1) * sy * sy) / (nx + ny - 2))
        return (x.mean() - y.mean()) / sp if sp > 0 else np.nan

    A = sub[sub["Origin_Type"] == group_a]
    B = sub[sub["Origin_Type"] == group_b]

    effect_sizes = {}
    for pcname in ["PC3", "PC4"]:
        d = cohens_d(A[pcname].values, B[pcname].values)
        effect_sizes[pcname] = d
        print(f"  [EFFECT] {pcname}: Cohen's d ({group_a} - {group_b}) = {d:.3f}")

    # --- SUMMARY TABLE ---
    summary = pd.DataFrame({
        "group": [group_a, group_b],
        "n": [len(A), len(B)],
        "PC3_mean": [A["PC3"].mean(), B["PC3"].mean()],
        "PC3_sd": [A["PC3"].std(), B["PC3"].std()],
        "PC4_mean": [A["PC4"].mean(), B["PC4"].mean()],
        "PC4_sd": [A["PC4"].std(), B["PC4"].std()],
    })
    # Add effect sizes as extra row
    effect_row = pd.DataFrame({
        "group": ["Cohen_d"],
        "n": [np.nan],
        "PC3_mean": [effect_sizes["PC3"]],
        "PC3_sd": [np.nan],
        "PC4_mean": [effect_sizes["PC4"]],
        "PC4_sd": [np.nan],
    })
    summary = pd.concat([summary, effect_row], ignore_index=True)

    summary_out = os.path.join(summary_dir, "pc34_substructure_summary.csv")
    summary.to_csv(summary_out, index=False)
    print(f"  [INFO] Saved summary -> {summary_out}")

    print()
    print("  PC3/PC4 substructure summary:")
    print(summary.to_string(index=False))
    print()

    return summary


# ======================================================================
# 3. Transferability of prediction across structure (Table S2-style)
# ======================================================================

def summarise_structure_transferability(summary_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combine baseline CV (random K-fold) and structure-aware CV results into a
    single summary.

    Requires:

      - cfg.CV_BASELINE_DIR / "cv_baseline_summary.csv"
          trait, scenario, mean_r, n_folds

      - cfg.CV_STRUCTURE_DIR / "cv_structure_summary.csv"
          trait, scheme, scenario, mean_r, n_folds
          where scheme in {cluster_balanced, leave_cluster_out}

    Outputs:

      - cv_transferability_summary.csv
          trait,
          r_random_pc,
          r_cluster_balanced_pc,
          r_leave_cluster_out_pc,
          drop_abs_cluster_balanced,
          drop_pct_cluster_balanced,
          drop_abs_leave_cluster,
          drop_pct_leave_cluster

      - cv_transferability_long.csv
          trait, cv_scheme, mean_r
          (tidy format for plotting)
    """
    print("[SECTION] Transferability across genetic structure")

    base_path = os.path.join(cfg.CV_BASELINE_DIR, "cv_baseline_summary.csv")
    struct_path = os.path.join(cfg.CV_STRUCTURE_DIR, "cv_structure_summary.csv")

    if not os.path.exists(base_path):
        raise FileNotFoundError(
            f"Baseline CV summary not found at {base_path}. "
            "Run 02_gs_kfold_baseline.py first."
        )
    if not os.path.exists(struct_path):
        raise FileNotFoundError(
            f"Structure-aware CV summary not found at {struct_path}. "
            "Run 03_gs_structure_aware_cv.py first."
        )

    base_df = pd.read_csv(base_path)
    struct_df = pd.read_csv(struct_path)

    # Focus on PC-corrected scenarios for the main transferability story
    base_pc = base_df[base_df["scenario"] == "pc_corrected"].copy()
    struct_pc = struct_df[struct_df["scenario"] == "pc_corrected"].copy()

    # Random K-fold (baseline)
    base_pc = base_pc.rename(columns={"mean_r": "r_random_pc"})
    base_pc = base_pc[["trait", "r_random_pc"]]

    # Cluster-balanced and leave-cluster-out
    cb = struct_pc[struct_pc["scheme"] == "cluster_balanced"].copy()
    cb = cb.rename(columns={"mean_r": "r_cluster_balanced_pc"})
    cb = cb[["trait", "r_cluster_balanced_pc"]]

    lco = struct_pc[struct_pc["scheme"] == "leave_cluster_out"].copy()
    lco = lco.rename(columns={"mean_r": "r_leave_cluster_out_pc"})
    lco = lco[["trait", "r_leave_cluster_out_pc"]]

    merged = base_pc.merge(cb, on="trait", how="inner").merge(lco, on="trait", how="inner")

    # Compute drops in accuracy
    drops_abs_cb = merged["r_random_pc"] - merged["r_cluster_balanced_pc"]
    drops_abs_lco = merged["r_random_pc"] - merged["r_leave_cluster_out_pc"]

    def safe_pct_drop(r_random: float, r_new: float) -> float:
        if r_random is None or np.isnan(r_random) or r_random == 0.0:
            return np.nan
        return 100.0 * (1.0 - (r_new / r_random))

    drops_pct_cb = [
        safe_pct_drop(r0, r1)
        for r0, r1 in zip(merged["r_random_pc"], merged["r_cluster_balanced_pc"])
    ]
    drops_pct_lco = [
        safe_pct_drop(r0, r1)
        for r0, r1 in zip(merged["r_random_pc"], merged["r_leave_cluster_out_pc"])
    ]

    merged["drop_abs_cluster_balanced"] = drops_abs_cb
    merged["drop_pct_cluster_balanced"] = drops_pct_cb
    merged["drop_abs_leave_cluster"] = drops_abs_lco
    merged["drop_pct_leave_cluster"] = drops_pct_lco

    # Save wide summary
    summary_out = os.path.join(summary_dir, "cv_transferability_summary.csv")
    print(f"  [INFO] Saving CV transferability summary -> {summary_out}")
    merged.to_csv(summary_out, index=False)

    # Tidy/long format for plotting
    long_records: List[Dict[str, object]] = []
    for _, row in merged.iterrows():
        trait = row["trait"]
        long_records.append({"trait": trait, "cv_scheme": "random_pc", "mean_r": row["r_random_pc"]})
        long_records.append(
            {
                "trait": trait,
                "cv_scheme": "cluster_balanced_pc",
                "mean_r": row["r_cluster_balanced_pc"],
            }
        )
        long_records.append(
            {
                "trait": trait,
                "cv_scheme": "leave_cluster_out_pc",
                "mean_r": row["r_leave_cluster_out_pc"],
            }
        )

    long_df = pd.DataFrame.from_records(long_records)
    long_out = os.path.join(summary_dir, "cv_transferability_long.csv")
    print(f"  [INFO] Saving CV transferability (long) -> {long_out}")
    long_df.to_csv(long_out, index=False)

    print()
    print("  Transferability summary (PC-corrected):")
    if not merged.empty:
        print(merged.to_string(index=False))
    print()

    return merged, long_df


# ======================================================================
# 4. GWAS-integrated GS (baseline vs weighted vs major+bg)
# ======================================================================

def summarise_gwas_integration(summary_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Summarise GWAS-integrated GS from 05_gs_weighted_and_fixed_effects.py.

    Requires:

      - cfg.GWAS_WEIGHTS_DIR / "internal_gwas_summary.csv"
          trait, n_snps, n_samples_used, n_major_qtl

      - cfg.CV_GWAS_INTEGRATION_DIR / "cv_gwas_integration_summary.csv"
          trait, model_type, scenario, mean_r, n_folds

      - cfg.CV_GWAS_INTEGRATION_DIR / "cv_gwas_integration_results.csv"
          trait, model_type, scenario, fold, r

    Outputs:

      - gwas_internal_summary_copy.csv
          (straight copy for convenience)

      - gwas_integration_summary_wide.csv
          trait, scenario,
          r_baseline, r_weighted, r_major_plus_bg,
          delta_weighted_vs_baseline, delta_major_vs_baseline

      - gwas_integration_tests.csv
          per trait & scenario:
            mean_delta_weighted,
            p_weighted_gt_baseline (two-sided t-test on fold-level deltas),
            mean_delta_major,
            p_major_vs_baseline
    """
    print("[SECTION] GWAS-integrated GS (baseline vs weighted vs major+bg)")

    internal_path = os.path.join(cfg.GWAS_WEIGHTS_DIR, "internal_gwas_summary.csv")
    cv_sum_path = os.path.join(cfg.CV_GWAS_INTEGRATION_DIR, "cv_gwas_integration_summary.csv")
    cv_res_path = os.path.join(cfg.CV_GWAS_INTEGRATION_DIR, "cv_gwas_integration_results.csv")

    # Internal GWAS summary (copy)
    if os.path.exists(internal_path):
        internal_df = pd.read_csv(internal_path)
        copy_out = os.path.join(summary_dir, "gwas_internal_summary_copy.csv")
        print(f"  [INFO] Copying internal GWAS summary -> {copy_out}")
        internal_df.to_csv(copy_out, index=False)
    else:
        internal_df = pd.DataFrame()
        print(f"  [WARN] Internal GWAS summary not found at {internal_path}")

    if not os.path.exists(cv_sum_path) or not os.path.exists(cv_res_path):
        print(f"  [WARN] GWAS-integration CV files not found in {cfg.CV_GWAS_INTEGRATION_DIR}")
        return internal_df, pd.DataFrame()

    cv_sum = pd.read_csv(cv_sum_path)
    cv_res = pd.read_csv(cv_res_path)

    # Pivot summary into wide format: baseline / weighted / major_plus_bg
    # for each trait & scenario.
    methods = ["baseline", "weighted", "major_plus_bg"]
    records: List[Dict[str, object]] = []

    for trait in sorted(cv_sum["trait"].unique()):
        for scenario in sorted(cv_sum["scenario"].unique()):
            row = {"trait": trait, "scenario": scenario}
            for m in methods:
                sub = cv_sum[
                    (cv_sum["trait"] == trait)
                    & (cv_sum["scenario"] == scenario)
                    & (cv_sum["model_type"] == m)
                ]
                if len(sub) == 0:
                    row[f"r_{m}"] = np.nan
                else:
                    row[f"r_{m}"] = float(sub["mean_r"].iloc[0])
            # Deltas
            r_base = row.get("r_baseline", np.nan)
            r_weighted = row.get("r_weighted", np.nan)
            r_major = row.get("r_major_plus_bg", np.nan)

            row["delta_weighted_vs_baseline"] = (
                np.nan if (np.isnan(r_base) or np.isnan(r_weighted)) else r_weighted - r_base
            )
            row["delta_major_vs_baseline"] = (
                np.nan if (np.isnan(r_base) or np.isnan(r_major)) else r_major - r_base
            )

            records.append(row)

    wide_df = pd.DataFrame.from_records(records)
    wide_out = os.path.join(summary_dir, "gwas_integration_summary_wide.csv")
    print(f"  [INFO] Saving GWAS-integration wide summary -> {wide_out}")
    wide_df.to_csv(wide_out, index=False)

    # Fold-level tests: deltas vs baseline
    test_records: List[Dict[str, object]] = []

    for trait in sorted(cv_res["trait"].unique()):
        for scenario in sorted(cv_res["scenario"].unique()):
            for m, label_prefix in [
                ("weighted", "weighted"),
                ("major_plus_bg", "major"),
            ]:
                base_sub = cv_res[
                    (cv_res["trait"] == trait)
                    & (cv_res["scenario"] == scenario)
                    & (cv_res["model_type"] == "baseline")
                ]
                m_sub = cv_res[
                    (cv_res["trait"] == trait)
                    & (cv_res["scenario"] == scenario)
                    & (cv_res["model_type"] == m)
                ]

                if base_sub.empty or m_sub.empty:
                    continue

                # Align folds
                merged = base_sub.merge(
                    m_sub,
                    on=["trait", "scenario", "fold"],
                    suffixes=("_baseline", f"_{label_prefix}"),
                )
                deltas = merged[f"r_{label_prefix}"].values - merged["r_baseline"].values
                deltas = deltas.astype(float)

                n = len(deltas)
                mean_delta = float(np.nanmean(deltas)) if n > 0 else np.nan
                sd_delta = float(np.nanstd(deltas, ddof=1)) if n > 1 else np.nan

                if n < 2 or np.isnan(sd_delta) or sd_delta == 0.0:
                    p_val = np.nan
                else:
                    t_stat = mean_delta / (sd_delta / np.sqrt(n))
                    # two-sided t-test
                    p_val = float(2.0 * stats.t.sf(np.abs(t_stat), df=n - 1))

                test_records.append(
                    {
                        "trait": trait,
                        "scenario": scenario,
                        "comparison": f"{label_prefix}_vs_baseline",
                        "n_folds": n,
                        "mean_delta": mean_delta,
                        "sd_delta": sd_delta,
                        "p_value": p_val,
                    }
                )

    tests_df = pd.DataFrame.from_records(test_records)
    tests_out = os.path.join(summary_dir, "gwas_integration_tests.csv")
    print(f"  [INFO] Saving GWAS-integration tests -> {tests_out}")
    tests_df.to_csv(tests_out, index=False)

    print()
    print("  GWAS-integration wide summary:")
    if not wide_df.empty:
        print(wide_df.to_string(index=False))
    print()

    print("  Fold-level tests (weighted/major vs baseline):")
    if not tests_df.empty:
        print(tests_df.to_string(index=False))
    print()

    return wide_df, tests_df


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    summary_dir = get_summary_dir()
    geno_path = os.path.join(cfg.CORE_DATA_DIR, "geno_core.npz")
    pheno_path = os.path.join(cfg.CORE_DATA_DIR, "pheno_core.csv")

    print("=" * 72)
    print("Mango GS – Idea 1: Global summary (06_idea1_summary.py)")
    print("=" * 72)
    print(f"[INFO] Summary directory: {summary_dir}")
    print(f"[INFO] Geno core:         {geno_path}")
    print(f"[INFO] Pheno core:        {pheno_path}")
    print()

    # 1. Phenotype summary
    summarise_phenotypes(pheno_path, summary_dir)

    # 2. Structure: PCA + clusters
    pc_df, cluster_df = compute_pca_and_clusters(geno_path, summary_dir, n_clusters=3)

    # 2b. PC3/PC4 substructure check (Americas vs South Asia)
    check_pc34_substructure(
    pc_df, 
    summary_dir,
    origin_path=r"C:\Users\ms\Desktop\mango\data\main_data\origin.csv"
    )

    # 3. Transferability across structure (random vs cluster-aware CV)
    summarise_structure_transferability(summary_dir)

    # 4. GWAS-integrated GS (baseline vs weighted vs major+bg)
    summarise_gwas_integration(summary_dir)

    print("[DONE] Idea 1 summary complete.")


if __name__ == "__main__":
    main()