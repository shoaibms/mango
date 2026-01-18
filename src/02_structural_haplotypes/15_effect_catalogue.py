#!/usr/bin/env python
"""
15_effect_catalogue.py

Breeder's Haplotype Effect Catalogue (Inversion markers)

Inputs
------
- output/idea_1/core_data/pheno_core.csv
- output/idea_1/core_data/meta_core.csv

Assumptions
-----------
- Rows are samples, indexed by sample_id in BOTH pheno_core and meta_core.
- Structural inversion markers live in meta_core and have column names
  starting with 'miinv' (case-insensitive), with genotypes coded as 0/1/2.
- Traits of interest: FBC, FF, AFW, TSS, TC.

Outputs
-------
- output/idea_2/breeder_tools/Breeder_Haplotype_Effects.csv
    * one row per Trait × Inversion marker
    * mean phenotype per genotype, effect size (G2–G0), additivity flag
- output/idea_2/breeder_tools/Breeder_CheatSheet_FBC.png
- output/idea_2/breeder_tools/Breeder_CheatSheet_AFW.png
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    _HAS_SEABORN = True
except ImportError:  # graceful fallback
    _HAS_SEABORN = False

# ================= CONFIG =================

# ROOT_DIR: resolve from this file location → /.../mango
ROOT_DIR: Path = Path(__file__).resolve().parents[2]

CORE_DIR: Path = ROOT_DIR / "output" / "idea_1" / "core_data"
OUT_DIR: Path = ROOT_DIR / "output" / "idea_2" / "breeder_tools"

PHENO_FILE: Path = CORE_DIR / "pheno_core.csv"
META_FILE: Path = CORE_DIR / "meta_core.csv"

# Canonical trait order (matches Idea 1 / config_idea1.TRAITS_DEFAULT)
TRAIT_ORDER: List[str] = ["FBC", "FF", "AFW", "TSS", "TC"]

# Minimum n per genotype class to be included
MIN_N_PER_CLASS: int = 5

# Effect size threshold (in SD units) to keep in the "top" catalogue
EFFECT_SD_THRESHOLD: float = 0.3

# Number of top markers per trait to show in cheat-sheet plots
N_TOP_FOR_PLOT: int = 5


# ==========================================

def safe_mkdir(path: Path) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    print("============================================================")
    print("Mango GS – Breeder's Haplotype Effect Catalogue (Idea 2)")
    print("============================================================")
    print(f"[INFO] ROOT_DIR:   {ROOT_DIR}")
    print(f"[INFO] PHENO_FILE: {PHENO_FILE}")
    print(f"[INFO] META_FILE:  {META_FILE}")
    safe_mkdir(OUT_DIR)

    # 1. Load data ------------------------------------------------------------
    if not PHENO_FILE.exists():
        print(f"[ERROR] pheno_core.csv not found at {PHENO_FILE}")
        return
    if not META_FILE.exists():
        print(f"[ERROR] meta_core.csv not found at {META_FILE}")
        return

    pheno = pd.read_csv(PHENO_FILE, index_col=0)
    meta = pd.read_csv(META_FILE, index_col=0)

    print(f"[INFO] pheno_core shape: {pheno.shape}")
    print(f"[INFO] meta_core shape:  {meta.shape}")

    # Ensure indices are aligned on sample_id
    common_ids = pheno.index.intersection(meta.index)
    if len(common_ids) == 0:
        print("[ERROR] No overlapping sample IDs between pheno_core and meta_core.")
        return

    pheno = pheno.loc[common_ids].copy()
    meta = meta.loc[common_ids].copy()

    # 2. Identify inversion markers in metadata -------------------------------
    inv_cols: List[str] = [
        c for c in meta.columns
        if str(c).lower().startswith("miinv")
    ]
    print(f"[INFO] Found {len(inv_cols)} inversion markers: {inv_cols}")

    if len(inv_cols) == 0:
        print("[WARN] No inversion columns (prefix 'miinv') found in meta_core. "
              "Nothing to catalogue.")
        return

    # Merge phenotypes and inversions
    df = pd.merge(pheno, meta[inv_cols], left_index=True, right_index=True)

    # 3. Calculate effects per trait × inversion -----------------------------
    summary_records: List[Dict[str, object]] = []

    for trait in TRAIT_ORDER:
        if trait not in df.columns:
            print(f"[WARN] Trait '{trait}' not found in pheno_core; skipping.")
            continue

        print(f"\n[TRAIT] {trait}")
        trait_series = df[trait].astype(float)
        global_mean = float(trait_series.mean())
        global_sd = float(trait_series.std(ddof=1))

        if global_sd == 0 or np.isnan(global_sd):
            print(f"  [WARN] Global SD for {trait} is zero/NaN; "
                  f"Effect_Std will be NaN.")
            global_sd = np.nan  # avoid divide by zero

        for inv in inv_cols:
            # Group by genotype class 0/1/2
            sub = df[[trait, inv]].dropna()
            # ensure numeric genotype
            try:
                sub[inv] = sub[inv].astype(int)
            except Exception:
                # if conversion fails, skip this marker
                continue

            stats = sub.groupby(inv)[trait].agg(["mean", "count", "std"])

            # Filter out rare classes
            stats = stats[stats["count"] >= MIN_N_PER_CLASS]

            # Require both homozygous classes to estimate effect
            if 0 not in stats.index or 2 not in stats.index:
                continue

            mean_g0 = float(stats.loc[0, "mean"])
            n_g0 = int(stats.loc[0, "count"])

            mean_g2 = float(stats.loc[2, "mean"])
            n_g2 = int(stats.loc[2, "count"])

            diff_raw = mean_g2 - mean_g0  # positive means G2 > G0

            if not np.isnan(global_sd) and global_sd > 0:
                diff_std = diff_raw / global_sd
            else:
                diff_std = np.nan

            # Approximate additivity via heterozygote position
            is_additive = False
            mean_g1 = np.nan
            n_g1 = 0
            if 1 in stats.index:
                mean_g1 = float(stats.loc[1, "mean"])
                n_g1 = int(stats.loc[1, "count"])
                midpoint = 0.5 * (mean_g0 + mean_g2)
                # "Additive" if G1 is close to midpoint (within 10% of total effect)
                if abs(mean_g1 - midpoint) <= 0.1 * abs(diff_raw):
                    is_additive = True

            summary_records.append(
                {
                    "Trait": trait,
                    "Marker": inv,
                    "Global_Mean": global_mean,
                    "Global_SD": global_sd,
                    "Mean_G0": mean_g0,
                    "Mean_G1": mean_g1,
                    "Mean_G2": mean_g2,
                    "N_G0": n_g0,
                    "N_G1": n_g1,
                    "N_G2": n_g2,
                    "Effect_Raw": diff_raw,  # G2 - G0
                    "Effect_Std": diff_std,  # standardised by global SD
                    "Is_Additive": is_additive,
                }
            )

    if not summary_records:
        print("[WARN] No valid Trait × Marker combinations with sufficient data.")
        return

    res_df = pd.DataFrame(summary_records)

    # Enforce canonical trait order in output
    res_df["Trait"] = pd.Categorical(
        res_df["Trait"],
        categories=TRAIT_ORDER,
        ordered=True,
    )
    res_df.sort_values(["Trait", "Effect_Std"], ascending=[True, False], inplace=True)

    # 4. Save catalogue (filtered high-value hits) ---------------------------
    top_hits = res_df[
        res_df["Effect_Std"].abs() > EFFECT_SD_THRESHOLD
    ].copy()

    if top_hits.empty:
        print("[WARN] No markers exceed Effect_Std threshold; "
              "saving full catalogue instead.")
        top_hits = res_df.copy()

    csv_path = OUT_DIR / "Breeder_Haplotype_Effects.csv"
    top_hits.to_csv(csv_path, index=False)
    print(f"\n[SUCCESS] Saved Breeder's Catalogue → {csv_path}")
    print(top_hits.head(10))

    # 5. Cheat-sheet plots for key traits ------------------------------------
    if not _HAS_SEABORN:
        print("[INFO] seaborn not installed; skipping cheat-sheet plots.")
    else:
        import seaborn as sns  # type: ignore

        for plot_trait in ["FBC", "AFW"]:
            trait_df = res_df[res_df["Trait"] == plot_trait].copy()
            if trait_df.empty:
                print(f"[INFO] No records for {plot_trait}; skipping plot.")
                continue

            trait_df["Abs_Effect"] = trait_df["Effect_Raw"].abs()
            top_for_plot = trait_df.sort_values(
                "Abs_Effect", ascending=False
            ).head(N_TOP_FOR_PLOT)

            if top_for_plot.empty:
                print(f"[INFO] No strong effects for {plot_trait}; skipping plot.")
                continue

            # Build long dataframe for boxplot: Value vs Genotype by Marker
            plot_data_frames = []
            for marker in top_for_plot["Marker"].unique():
                sub = df[[plot_trait, marker]].dropna().copy()
                try:
                    sub[marker] = sub[marker].astype(int)
                except Exception:
                    continue
                sub.columns = ["Value", "Genotype"]
                sub["Marker"] = marker
                plot_data_frames.append(sub)

            if not plot_data_frames:
                print(f"[INFO] No usable data for {plot_trait} top markers; "
                      "skipping plot.")
                continue

            plot_df = pd.concat(plot_data_frames, axis=0)

            plt.figure(figsize=(10, 6))
            sns.boxplot(
                data=plot_df,
                x="Marker",
                y="Value",
                hue="Genotype",
            )
            plt.title(f"Top {N_TOP_FOR_PLOT} haplotype effects for {plot_trait}")
            plt.ylabel(plot_trait)
            plt.xlabel("Inversion marker")
            plt.legend(title="Genotype", loc="best")
            plt.tight_layout()

            img_path = OUT_DIR / f"Breeder_CheatSheet_{plot_trait}.png"
            plt.savefig(img_path, dpi=300)
            plt.close()
            print(f"[PLOT] Saved cheat-sheet plot for {plot_trait} → {img_path}")

    print("\n[DONE] Breeder's Effect Catalogue generation complete.")


if __name__ == "__main__":
    main()
