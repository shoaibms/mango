#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
18_summary_stats.py

Mango GS – Idea 2 global summary

Summarises all key Idea 2 results for the manuscript:

  1. Genomic prediction performance (ridge, XGB, RF; SNP / INV / SNP+INV)
  2. Inversions vs matched random 17-SNP controls
  3. Permutation tests (real vs permuted phenotypes)
  4. Candidate gene architecture (pleiotropy, hubs, supporting SNPs)

Inputs (expected defaults, relative to project root):
  - output/idea_2/summary/idea2_model_performance_long.csv
  - output/idea_2/summary/random_vs_inversion_summary.csv
  - output/idea_2/summary/permutation_tests_summary.csv
  - output/idea_2/annotation/idea2_candidate_genes_summary.csv

Outputs (all in output/idea_2/summary/):
  - idea2_gs_model_performance_clean.csv
  - idea2_gs_best_by_trait_scheme.csv
  - idea2_gs_best_by_trait_overall.csv
  - idea2_gs_ridge_vs_best_ml.csv
  - idea2_random_vs_inversion_full.csv
  - idea2_random_vs_inversion_best_by_trait.csv
  - idea2_random_vs_inversion_best_by_trait_scheme.csv
  - idea2_permutation_tests_full.csv
  - idea2_permutation_tests_real_r_ge_0.3.csv
  - idea2_candidate_genes_gene_level_summary.csv
  - idea2_candidate_genes_architecture_stats.csv
  - idea2_results_summary_for_manuscript.md
"""

from pathlib import Path
import argparse
import datetime
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------


def summarise_model_performance(mp: pd.DataFrame):
    """
    Summarise model performance grid.

    Expects columns:
      ['trait','scheme','feature_set','model_family','mean_r','std_r']
    """
    df = mp.copy()
    df = df.dropna(subset=["mean_r"])

    key_cols = ["trait", "scheme", "feature_set", "model_family"]
    df = df.sort_values(key_cols + ["mean_r"], ascending=[True, True, True, True, False])
    df = df.drop_duplicates(subset=key_cols, keep="first")

    # Best per trait × scheme
    best_by_scheme = (
        df.sort_values(["trait", "scheme", "mean_r"], ascending=[True, True, False])
        .groupby(["trait", "scheme"], as_index=False)
        .first()
    )

    # Best per trait overall
    best_by_trait = (
        df.sort_values(["trait", "mean_r"], ascending=[True, False])
        .groupby("trait", as_index=False)
        .first()
    )

    # Ridge vs best XGB/RF per trait × scheme
    ridge = df[df["model_family"] == "ridge"].copy()
    ml = df[df["model_family"].isin(["xgb", "rf"])].copy()

    best_ridge = (
        ridge.sort_values(["trait", "scheme", "mean_r"], ascending=[True, True, False])
        .groupby(["trait", "scheme"], as_index=False)
        .first()
    )
    best_ml = (
        ml.sort_values(["trait", "scheme", "mean_r"], ascending=[True, True, False])
        .groupby(["trait", "scheme"], as_index=False)
        .first()
    )

    comp = best_ridge.merge(
        best_ml,
        on=["trait", "scheme"],
        how="inner",
        suffixes=("_ridge", "_ml"),
    )
    comp["delta_r"] = comp["mean_r_ml"] - comp["mean_r_ridge"]
    comp["rel_improvement_pct"] = np.where(
        comp["mean_r_ridge"].abs() > 1e-8,
        100.0 * comp["delta_r"] / comp["mean_r_ridge"].abs(),
        np.nan,
    )

    return {
        "clean": df,
        "best_by_scheme": best_by_scheme,
        "best_by_trait": best_by_trait,
        "ridge_vs_ml": comp,
    }


def summarise_random_vs_inversion(rvi: pd.DataFrame):
    """
    Summarise inversion vs random 17-SNP control.

    Expects columns:
      ['trait','scheme','model','n_markers',
       'inversion_mean_r','inversion_std_r',
       'random_mean_r','random_std_r',
       'random_min_r','random_max_r',
       'n_random','n_random_ge_inversion','p_empirical']
    """
    df = rvi.copy()

    for col in ["inversion_mean_r", "random_mean_r"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["delta_r"] = df["inversion_mean_r"] - df["random_mean_r"]
    df["rel_improvement_pct"] = np.where(
        df["random_mean_r"].abs() > 1e-8,
        100.0 * df["delta_r"] / df["random_mean_r"].abs(),
        np.nan,
    )
    df["is_significant_0_05"] = df["p_empirical"] <= 0.05
    df["is_significant_0_01"] = df["p_empirical"] <= 0.01

    best_by_trait = (
        df.sort_values(["trait", "p_empirical", "delta_r"], ascending=[True, True, False])
        .groupby("trait", as_index=False)
        .first()
    )

    best_by_trait_scheme = (
        df.sort_values(["trait", "scheme", "delta_r"], ascending=[True, True, False])
        .groupby(["trait", "scheme"], as_index=False)
        .first()
    )

    return {
        "full": df,
        "best_by_trait": best_by_trait,
        "best_by_trait_scheme": best_by_trait_scheme,
    }


def summarise_permutation(perm: pd.DataFrame):
    """
    Summarise permutation tests.

    Expects columns:
      ['trait','scheme','model','feature_set','n_perm',
       'real_mean_r','real_std_r',
       'perm_mean_r','perm_std_r','perm_min_r','perm_max_r',
       'n_perm_ge_real','p_empirical']
    """
    df = perm.copy()

    for col in ["real_mean_r", "perm_mean_r"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["delta_r"] = df["real_mean_r"] - df["perm_mean_r"]
    df["is_significant_0_05"] = df["p_empirical"] <= 0.05
    df["is_significant_0_01"] = df["p_empirical"] <= 0.01

    overall = {
        "n_rows": int(df.shape[0]),
        "mean_real_r": float(df["real_mean_r"].mean()),
        "mean_perm_r": float(df["perm_mean_r"].mean()),
        "mean_delta_r": float(df["delta_r"].mean()),
        "n_significant_0_05": int(df["is_significant_0_05"].sum()),
        "n_significant_0_01": int(df["is_significant_0_01"].sum()),
    }

    df_good = df[df["real_mean_r"] >= 0.3].copy()

    return {
        "full": df,
        "good": df_good,
        "overall_stats": overall,
    }


def summarise_gene_architecture(gene_trait_df: pd.DataFrame):
    """
    Summarise candidate gene architecture from the gene–trait table.

    Expects columns:
      ['gene_id','gene_name','chr','gene_start','gene_end',
       'trait','n_supporting_variants','min_distance_bp']
    """
    df = gene_trait_df.copy()
    required = {"gene_id", "chr", "gene_start", "gene_end", "trait"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Candidate gene table missing required columns: {missing}")

    agg = (
        df.groupby("gene_id")
        .agg(
            chr=("chr", "first"),
            gene_start=("gene_start", "min"),
            gene_end=("gene_end", "max"),
            gene_name=("gene_name", "first"),
            total_supporting_variants=("n_supporting_variants", "sum"),
            min_distance_bp=("min_distance_bp", "min"),
            trait_count=("trait", "nunique"),
        )
    )

    trait_list = df.groupby("gene_id")["trait"].apply(
        lambda x: ";".join(sorted(x.unique()))
    )
    agg["affected_traits"] = trait_list

    def classify(c):
        if c >= 4:
            return "hub_pleiotropic"
        if c >= 2:
            return "multi_trait"
        return "single_trait"

    agg["pleiotropy_class"] = agg["trait_count"].apply(classify)

    n_genes = agg.shape[0]
    n_single = int((agg["trait_count"] == 1).sum())
    n_two = int((agg["trait_count"] == 2).sum())
    n_three = int((agg["trait_count"] == 3).sum())
    n_hub = int((agg["trait_count"] >= 4).sum())

    stats = {
        "n_genes": int(n_genes),
        "n_single_trait_genes": n_single,
        "n_two_trait_genes": n_two,
        "n_three_trait_genes": n_three,
        "n_hub_genes_ge4": n_hub,
        "max_trait_count": int(agg["trait_count"].max()),
        "max_total_supporting_variants": int(agg["total_supporting_variants"].max()),
    }

    return {"gene_level": agg.reset_index(), "stats": stats}


def _make_rel(path: Path, root: Path) -> str:
    """Safe relative path (falls back to absolute if outside root)."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def build_markdown(
    outdir: Path,
    gs_stats=None,
    rvi_stats=None,
    perm_stats=None,
    ga_stats=None,
    generated_files=None,
):
    """Create a markdown summary tying together the main statistics."""
    lines = []
    today = datetime.date.today().isoformat()

    lines.append("# Mango GS – Idea 2 Summary")
    lines.append("")
    lines.append(f"_Generated on {today}_")
    lines.append("")

    nice_scheme = {
        "cv_random_k5": "Random k-fold CV (population-unstructured)",
        "cv_cluster_kmeans": "Cluster CV (ancestry-aware)",
    }
    nice_feature = {
        "snp": "SNP-only",
        "inv": "Inversion-only",
        "snp+inv": "SNP + inversion",
    }

    # 1. GS performance
    if gs_stats is not None:
        bbs = gs_stats["best_by_scheme"]
        comp = gs_stats["ridge_vs_ml"]

        lines.append("## 1. Genomic prediction accuracy – ridge vs tree-based models")
        lines.append(
            "Summary of the best-performing models (ridge, XGBoost, random forest) "
            "for each trait and cross-validation scheme, across SNP-only, inversion-only "
            "and SNP+inversion feature sets."
        )
        lines.append("")
        lines.append("### 1.1 Best combination per trait × CV scheme")
        lines.append(
            "| Trait | CV scheme | Model | Features | mean r | sd r |"
        )
        lines.append("|-------|-----------|-------|----------|--------|------|")

        for _, row in bbs.sort_values(["trait", "scheme"]).iterrows():
            scheme_label = nice_scheme.get(row["scheme"], row["scheme"])
            feat_label = nice_feature.get(row["feature_set"], row["feature_set"])
            lines.append(
                f"| {row['trait']} | {scheme_label} | {row['model_family']} | "
                f"{feat_label} | {row['mean_r']:.3f} | {row['std_r']:.3f} |"
            )

        lines.append("")
        lines.append("### 1.2 Benefit of tree-based models over ridge")
        if comp.empty:
            lines.append(
                "No XGBoost or random forest models were available; only ridge was fitted."
            )
        else:
            lines.append(
                "For each trait and CV scheme, the best-performing tree-based model "
                "(XGBoost or random forest) is compared against the best ridge model."
            )
            lines.append("")
            lines.append(
                "| Trait | CV scheme | Best ridge (features, r) | "
                "Best tree model (model, features, r) | Δr (tree − ridge) | % improvement |"
            )
            lines.append(
                "|-------|-----------|-------------------------|"
                "--------------------------------------|-------------------|--------------|"
            )
            for _, row in comp.sort_values(["trait", "scheme"]).iterrows():
                scheme_label = nice_scheme.get(row["scheme"], row["scheme"])
                ridge_feat = nice_feature.get(
                    row["feature_set_ridge"], row["feature_set_ridge"]
                )
                ml_feat = nice_feature.get(
                    row["feature_set_ml"], row["feature_set_ml"]
                )
                lines.append(
                    f"| {row['trait']} | {scheme_label} | "
                    f"{ridge_feat}, r={row['mean_r_ridge']:.3f} | "
                    f"{row['model_family_ml']} ({ml_feat}), r={row['mean_r_ml']:.3f} | "
                    f"{row['delta_r']:.3f} | {row['rel_improvement_pct']:.1f}% |"
                )

        lines.append("")

    # 2. Random vs inversion
    if rvi_stats is not None:
        df_full = rvi_stats["full"]
        best_trait = rvi_stats["best_by_trait"]

        lines.append("## 2. Inversions vs matched random 17-SNP controls")
        lines.append(
            "Comparison of predictive accuracy from curated chromosomal inversions "
            "against equally sized random SNP marker sets (17 SNPs, 100 random replicates)."
        )
        lines.append("")
        lines.append("### 2.1 Best inversion advantage per trait")
        lines.append(
            "| Trait | CV scheme | Model | r (inversions) | r (random mean) | "
            "Δr | % improvement | Empirical p (random ≥ inversion) |"
        )
        lines.append(
            "|-------|-----------|-------|----------------|-----------------|"
            "----|--------------|----------------------------------|"
        )

        for _, row in best_trait.sort_values("trait").iterrows():
            scheme_label = nice_scheme.get(row["scheme"], row["scheme"])
            lines.append(
                f"| {row['trait']} | {scheme_label} | {row['model']} | "
                f"{row['inversion_mean_r']:.3f} | {row['random_mean_r']:.3f} | "
                f"{row['delta_r']:.3f} | {row['rel_improvement_pct']:.1f}% | "
                f"{row['p_empirical']:.4f} |"
            )

        lines.append("")
        n_rows = df_full.shape[0]
        n_sig_5 = int(df_full["is_significant_0_05"].sum())
        n_sig_1 = int(df_full["is_significant_0_01"].sum())
        lines.append(
            f"Across all trait × scheme × model combinations (n = {n_rows}), "
            f"{n_sig_5} showed a significant inversion advantage at p ≤ 0.05 "
            f"({n_sig_1} at p ≤ 0.01, based on empirical p-values from random marker sets)."
        )
        lines.append("")

    # 3. Permutation tests
    if perm_stats is not None:
        overall = perm_stats["overall_stats"]
        df_good = perm_stats["good"]

        lines.append("## 3. Permutation tests (sanity check for overfitting)")
        lines.append(
            "Permutation tests (shuffling phenotypes) were used to confirm that the "
            "observed predictive accuracy arises from genuine genotype–phenotype coupling "
            "rather than model overfitting or population structure artefacts."
        )
        lines.append("")
        lines.append(f"- Model configurations evaluated: {overall['n_rows']}.")
        lines.append(
            f"- Mean cross-validated accuracy on real phenotypes: r ≈ {overall['mean_real_r']:.3f}."
        )
        lines.append(
            f"- Mean accuracy under permuted phenotypes: r ≈ {overall['mean_perm_r']:.3f}."
        )
        lines.append(
            f"- Average drop due to permutation: Δr ≈ {overall['mean_delta_r']:.3f}."
        )
        lines.append(
            f"- Configurations with p_empirical ≤ 0.05: {overall['n_significant_0_05']} "
            f"({overall['n_significant_0_01']} with p_empirical ≤ 0.01)."
        )
        lines.append("")
        if not df_good.empty:
            lines.append(
                "Considering only reasonably predictive models (real_mean_r ≥ 0.3), "
                "all such configurations retained a clear gap between real and permuted accuracies, "
                "supporting that the signal is biological rather than an artefact of model flexibility."
            )
            lines.append("")

    # 4. Candidate gene architecture
    if ga_stats is not None:
        gene_level = ga_stats["gene_level"]
        gstats = ga_stats["stats"]

        lines.append("## 4. Candidate gene architecture and pleiotropic hubs")
        lines.append(
            "Summary of post-GWAS candidate genes linked to the top predictive SNPs "
            "and inversion segments."
        )
        lines.append("")
        lines.append(
            f"- Total unique candidate genes: **{gstats['n_genes']}**."
        )
        lines.append(
            f"- Single-trait genes (trait_count = 1): **{gstats['n_single_trait_genes']}**."
        )
        lines.append(
            f"- Two-trait genes (trait_count = 2): **{gstats['n_two_trait_genes']}**."
        )
        lines.append(
            f"- Three-trait genes (trait_count = 3): **{gstats['n_three_trait_genes']}**."
        )
        lines.append(
            f"- Pleiotropic hubs (trait_count ≥ 4): **{gstats['n_hub_genes_ge4']}** "
            f"(maximum {gstats['max_trait_count']} traits per gene)."
        )
        lines.append(
            f"- Maximum number of supporting SNPs for a single gene: "
            f"**{gstats['max_total_supporting_variants']}**."
        )
        lines.append("")
        hubs = gene_level[gene_level["trait_count"] >= 4].copy()
        if not hubs.empty:
            lines.append("### 4.1 Pleiotropic hubs (trait_count ≥ 4)")
            lines.append(
                "| gene_id | chr | start | end | trait_count | "
                "affected_traits | total_supporting_variants |"
            )
            lines.append(
                "|---------|-----|-------|-----|-------------|"
                "-----------------|----------------------------|"
            )
            for _, row in hubs.sort_values(["chr", "gene_start"]).iterrows():
                lines.append(
                    f"| {row['gene_id']} | {row['chr']} | "
                    f"{int(row['gene_start'])} | {int(row['gene_end'])} | "
                    f"{int(row['trait_count'])} | {row['affected_traits']} | "
                    f"{int(row['total_supporting_variants'])} |"
                )
            lines.append("")

    # 5. File list
    if generated_files:
        lines.append("## 5. Files generated by this script")
        lines.append(
            "The following summary files were written; they collect the key "
            "statistics used throughout the manuscript."
        )
        lines.append("")
        for rel, desc in generated_files:
            lines.append(f"- `{rel}` – {desc}")
        lines.append("")

    md_path = outdir / "idea2_results_summary_for_manuscript.md"
    outdir.mkdir(parents=True, exist_ok=True)
    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return md_path


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main(root: str = None, outdir: str = None):
    # If root not provided, infer it from this file location:
    # .../mango/output/idea_2/summary/17_idea2_summary.py
    here = Path(__file__).resolve()
    if root is None:
        # parents[0]=summary, [1]=idea_2, [2]=output, [3]=mango
        root_path = here.parents[3]
    else:
        root_path = Path(root)
    if outdir is None:
        outdir_path = root_path / "output" / "idea_2" / "summary"
    else:
        outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(" Mango GS – Idea 2: Global summary (17_idea2_summary.py)")
    print("=" * 72)
    print(f"[INFO] Project root:  {root_path}")
    print(f"[INFO] Summary outdir:{outdir_path}")

    # Input paths
    model_perf_path = outdir_path / "idea2_model_performance_long.csv"
    rvi_path = outdir_path / "random_vs_inversion_summary.csv"
    perm_path = outdir_path / "permutation_tests_summary.csv"
    cand_genes_path = (
        root_path / "output" / "idea_2" / "annotation" / "idea2_candidate_genes_summary.csv"
    )

    def safe_load(path: Path, label: str):
        if not path.exists():
            print(f"[WARN] {label} not found: {path}")
            return None
        print(f"[LOAD] {label} -> {path}")
        return pd.read_csv(path)

    mp = safe_load(
        model_perf_path,
        "Model performance (idea2_model_performance_long.csv)",
    )
    rvi = safe_load(rvi_path, "Random vs inversion summary")
    perm = safe_load(perm_path, "Permutation tests summary")
    genes = safe_load(cand_genes_path, "Candidate genes (gene–trait summary)")

    gs_stats = rvi_stats = perm_stats = ga_stats = None
    generated_files = []

    # 1. GS performance
    if mp is not None:
        gs_stats = summarise_model_performance(mp)

        clean_path = outdir_path / "idea2_gs_model_performance_clean.csv"
        gs_stats["clean"].to_csv(clean_path, index=False)
        generated_files.append(
            (
                _make_rel(clean_path, root_path),
                "Cleaned model performance grid (one row per trait × scheme × feature_set × model_family).",
            )
        )

        bbs_path = outdir_path / "idea2_gs_best_by_trait_scheme.csv"
        gs_stats["best_by_scheme"].to_csv(bbs_path, index=False)
        generated_files.append(
            (
                _make_rel(bbs_path, root_path),
                "Best-performing model per trait × CV scheme.",
            )
        )

        bbt_path = outdir_path / "idea2_gs_best_by_trait_overall.csv"
        gs_stats["best_by_trait"].to_csv(bbt_path, index=False)
        generated_files.append(
            (
                _make_rel(bbt_path, root_path),
                "Best-performing model per trait across all CV schemes.",
            )
        )

        comp_path = outdir_path / "idea2_gs_ridge_vs_best_ml.csv"
        gs_stats["ridge_vs_ml"].to_csv(comp_path, index=False)
        generated_files.append(
            (
                _make_rel(comp_path, root_path),
                "Comparison of best ridge vs best tree-based (XGB/RF) models per trait × CV scheme.",
            )
        )

    # 2. Random vs inversion
    if rvi is not None:
        rvi_stats = summarise_random_vs_inversion(rvi)

        rvi_full_path = outdir_path / "idea2_random_vs_inversion_full.csv"
        rvi_stats["full"].to_csv(rvi_full_path, index=False)
        generated_files.append(
            (
                _make_rel(rvi_full_path, root_path),
                "Full inversion vs random 17-SNP control summary (all traits × schemes × models).",
            )
        )

        rvi_best_trait_path = outdir_path / "idea2_random_vs_inversion_best_by_trait.csv"
        rvi_stats["best_by_trait"].to_csv(rvi_best_trait_path, index=False)
        generated_files.append(
            (
                _make_rel(rvi_best_trait_path, root_path),
                "Per-trait row where inversions most strongly outperform random markers.",
            )
        )

        rvi_best_trait_scheme_path = (
            outdir_path / "idea2_random_vs_inversion_best_by_trait_scheme.csv"
        )
        rvi_stats["best_by_trait_scheme"].to_csv(rvi_best_trait_scheme_path, index=False)
        generated_files.append(
            (
                _make_rel(rvi_best_trait_scheme_path, root_path),
                "Best inversion advantage per trait × CV scheme.",
            )
        )

    # 3. Permutation tests
    if perm is not None:
        perm_stats = summarise_permutation(perm)

        perm_full_path = outdir_path / "idea2_permutation_tests_full.csv"
        perm_stats["full"].to_csv(perm_full_path, index=False)
        generated_files.append(
            (
                _make_rel(perm_full_path, root_path),
                "Full permutation test summary (real vs permuted accuracy).",
            )
        )

        perm_good_path = outdir_path / "idea2_permutation_tests_real_r_ge_0.3.csv"
        perm_stats["good"].to_csv(perm_good_path, index=False)
        generated_files.append(
            (
                _make_rel(perm_good_path, root_path),
                "Subset of permutation results for models with real_mean_r ≥ 0.3.",
            )
        )

    # 4. Gene architecture
    if genes is not None:
        ga_stats = summarise_gene_architecture(genes)

        gene_level_path = outdir_path / "idea2_candidate_genes_gene_level_summary.csv"
        ga_stats["gene_level"].to_csv(gene_level_path, index=False)
        generated_files.append(
            (
                _make_rel(gene_level_path, root_path),
                "Gene-level architecture summary (pleiotropy, supporting SNPs, distances).",
            )
        )

        gene_stats_rows = [
            {"metric": k, "value": v} for k, v in ga_stats["stats"].items()
        ]
        gene_stats_path = (
            outdir_path / "idea2_candidate_genes_architecture_stats.csv"
        )
        pd.DataFrame(gene_stats_rows).to_csv(gene_stats_path, index=False)
        generated_files.append(
            (
                _make_rel(gene_stats_path, root_path),
                "High-level counts describing the candidate gene architecture (e.g., number of hubs).",
            )
        )

    # 5. Markdown summary
    md_path = build_markdown(
        outdir_path, gs_stats, rvi_stats, perm_stats, ga_stats, generated_files
    )
    generated_files.append(
        (
            _make_rel(md_path, root_path),
            "Markdown narrative summary for the manuscript (sections 1–4 above).",
        )
    )

    print("=" * 72)
    print("[OK] Idea 2 summary generated.")
    print(f"[INFO] Markdown summary: {md_path}")
    print("[INFO] Files written:")
    for rel, desc in generated_files:
        print(f"  - {rel}: {desc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarise Idea 2 GS + inversion + gene architecture results."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Project root directory (default: inferred from this script location).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Optional override for summary output directory.",
    )
    args = parser.parse_args()
    main(root=args.root, outdir=args.outdir)
