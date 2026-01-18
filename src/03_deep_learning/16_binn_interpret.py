# -*- coding: utf-8 -*-
r"""
16_binn_interpret.py

Mango GS – Idea 3
=================
Interpret the trained BINN models from 12_binn_train.py and produce
gene-level importance tables suitable for the manuscript.

Inputs:
  * output\idea_3\binn_training\weights\binn_gene_weights_fold*.npz
      - kernel: (n_genes, n_traits) gene→trait weights
      - bias:   (n_traits,)
      - gene_ids: (n_genes,)
      - trait_names: (n_traits,)
  * output\idea_3\binn_maps\binn_gene_table.csv
      - gene_id, gene_name, chr, start, end, strand, product, n_snps, n_traits, traits
  * output\idea_2\annotation\idea2_candidate_genes_summary.csv
      - gene_id + per-trait candidate status (columns may vary)

Outputs (under OUT_EXPLAIN_DIR):
  - binn_gene_scores_wide.csv
      gene_id, gene_name, chr, start, end, n_snps, [one column per trait: score_mean_abs]
  - binn_gene_scores_long.csv
      gene_id, trait, score_mean_abs, score_sd_abs, rank_trait, gene_name, chr, start, end,
      n_snps, is_candidate_any, candidate_traits
  - binn_gene_pleiotropy_scores.csv
      gene_id, gene_name, chr, start, end, n_snps,
      max_score, n_traits_above_90pct, traits_above_90pct
  - binn_explain_summary.txt
      human-readable summary + top 10 genes per trait


"""

from __future__ import annotations

import glob
import os
from typing import Dict, List

import numpy as np
import pandas as pd


# =========================
# CONFIG
# =========================

# Gene table from 10_binn_build_maps.py
BINN_GENE_TABLE = r"C:\Users\ms\Desktop\mango\output\idea_3\binn_maps\binn_gene_table.csv"

# Weights from 12_binn_train.py
BINN_WEIGHTS_DIR = r"C:\Users\ms\Desktop\mango\output\idea_3\binn_training\weights"

# Candidate gene summary from Idea 2
CANDIDATE_GENES_SUMMARY = (
    r"C:\Users\ms\Desktop\mango\output\idea_2\annotation\idea2_candidate_genes_summary.csv"
)

# Output directory for BINN interpretation
OUT_ROOT = r"C:\Users\ms\Desktop\mango\output\idea_3"
OUT_EXPLAIN_DIR = os.path.join(OUT_ROOT, "binn_explain")

# Canonical trait order (used for pretty printing / ordering if needed)
TRAIT_ORDER: List[str] = ["FBC", "FF", "AFW", "TSS", "TC"]


# =========================
# UTILITIES
# =========================

def safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_gene_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"BINN gene table not found:\n  {path}\n"
            "Run 10_binn_build_maps.py first."
        )
    df = pd.read_csv(path)
    if "gene_id" not in df.columns:
        raise KeyError("binn_gene_table.csv must have a 'gene_id' column.")
    df["gene_id"] = df["gene_id"].astype(str)
    df.set_index("gene_id", inplace=True)
    return df


def load_candidate_summary(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[WARN] Candidate gene summary not found:\n  {path}\n"
              "Proceeding without candidate annotations.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    if "gene_id" not in df.columns:
        print("[WARN] Candidate summary has no 'gene_id' column; ignoring file.")
        return pd.DataFrame()

    df["gene_id"] = df["gene_id"].astype(str)
    df.set_index("gene_id", inplace=True)
    return df


def load_all_weight_files(weights_dir: str) -> Dict[str, np.ndarray]:
    """
    Load all binn_gene_weights_fold*.npz files and stack them into a 3D array.

    Returns:
      {
        "kernel_all": (n_folds, n_genes, n_traits),
        "bias_all":   (n_folds, n_traits),
        "gene_ids":   (n_genes,),
        "trait_names":(n_traits,),
        "fold_ids":   list of fold numbers
      }
    """
    pattern = os.path.join(weights_dir, "binn_gene_weights_fold*.npz")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No weight files matching pattern:\n  {pattern}\n"
            "Run 12_binn_train.py first."
        )

    print(f"[INFO] Found {len(files)} weight files:")
    for f in files:
        print(f"  - {os.path.basename(f)}")

    kernels = []
    biases = []
    gene_ids_ref = None
    trait_names_ref = None
    fold_ids = []

    for f in files:
        npz = np.load(f, allow_pickle=True)
        if "kernel" not in npz.files or "bias" not in npz.files:
            raise KeyError(f"{f} missing 'kernel' or 'bias' arrays.")
        kernel = npz["kernel"]
        bias = npz["bias"]

        gene_ids = npz["gene_ids"].astype(str) if "gene_ids" in npz.files else None
        trait_names = npz["trait_names"].astype(str) if "trait_names" in npz.files else None

        if gene_ids_ref is None:
            gene_ids_ref = gene_ids
        else:
            if gene_ids is not None and not np.array_equal(gene_ids_ref, gene_ids):
                raise ValueError(f"Gene ID ordering differs in {f} vs previous folds.")

        if trait_names_ref is None:
            trait_names_ref = trait_names
        else:
            if trait_names is not None and not np.array_equal(trait_names_ref, trait_names):
                raise ValueError(f"Trait name ordering differs in {f} vs previous folds.")

        kernels.append(kernel)
        biases.append(bias)

        # Extract fold number from file name if possible
        base = os.path.basename(f)
        # e.g., binn_gene_weights_fold3.npz -> fold3
        fold_str = base.replace("binn_gene_weights_fold", "").replace(".npz", "")
        try:
            fold_id = int(fold_str)
        except ValueError:
            fold_id = len(fold_ids) + 1  # fallback
        fold_ids.append(fold_id)

    kernel_all = np.stack(kernels, axis=0)  # (n_folds, n_genes, n_traits)
    bias_all = np.stack(biases, axis=0)     # (n_folds, n_traits)

    return {
        "kernel_all": kernel_all.astype(np.float32),
        "bias_all": bias_all.astype(np.float32),
        "gene_ids": gene_ids_ref,
        "trait_names": trait_names_ref,
        "fold_ids": fold_ids,
    }


# =========================
# MAIN LOGIC
# =========================

def main():
    print("=" * 72)
    print("Mango GS – Idea 3: BINN explanation (13_binn_explain.py)")
    print("=" * 72)
    print(f"[INFO] Gene table:       {BINN_GENE_TABLE}")
    print(f"[INFO] Weights dir:      {BINN_WEIGHTS_DIR}")
    print(f"[INFO] Candidate summary: {CANDIDATE_GENES_SUMMARY}")
    print(f"[INFO] Output dir:       {OUT_EXPLAIN_DIR}")
    print("")

    safe_makedirs(OUT_EXPLAIN_DIR)

    # 1. Load inputs
    gene_table = load_gene_table(BINN_GENE_TABLE)
    cand_df = load_candidate_summary(CANDIDATE_GENES_SUMMARY)
    weights = load_all_weight_files(BINN_WEIGHTS_DIR)

    kernel_all = weights["kernel_all"]      # (n_folds, n_genes, n_traits)
    gene_ids = weights["gene_ids"]          # (n_genes,)
    trait_names = list(weights["trait_names"])
    fold_ids = weights["fold_ids"]

    n_folds, n_genes, n_traits = kernel_all.shape
    print(f"[INFO] kernel_all shape: {kernel_all.shape} (folds x genes x traits)")
    print(f"[INFO] Traits in BINN:   {trait_names}")
    print(f"[INFO] Folds:            {fold_ids}")
    print("")

    # Ensure gene_table index aligns with gene_ids from weights
    missing_genes = [g for g in gene_ids if g not in gene_table.index]
    if missing_genes:
        print(f"[WARN] {len(missing_genes)} gene_ids from weights not found in binn_gene_table.")
        # We proceed, but those genes will have limited annotation

    # 2. Compute mean and sd of |weights| across folds
    abs_kernel = np.abs(kernel_all)                     # (n_folds, n_genes, n_traits)
    mean_abs = abs_kernel.mean(axis=0)                  # (n_genes, n_traits)
    sd_abs = abs_kernel.std(axis=0, ddof=1)             # (n_genes, n_traits)

    # 3. Build wide gene × trait matrix
    wide_rows = []
    for i, gid in enumerate(gene_ids):
        row = {
            "gene_id": gid,
        }
        # Basic annotation from gene_table if available
        if gid in gene_table.index:
            g = gene_table.loc[gid]
            row["gene_name"] = g.get("gene_name", "")
            row["chr"] = g.get("chr", "")
            row["start"] = g.get("start", np.nan)
            row["end"] = g.get("end", np.nan)
            row["n_snps"] = g.get("n_snps", np.nan)
        else:
            row["gene_name"] = ""
            row["chr"] = ""
            row["start"] = np.nan
            row["end"] = np.nan
            row["n_snps"] = np.nan

        for j, trait in enumerate(trait_names):
            row[f"score_{trait}"] = float(mean_abs[i, j])
        wide_rows.append(row)

    wide_df = pd.DataFrame(wide_rows)
    wide_df.set_index("gene_id", inplace=True)

    # 4. Build long gene × trait table with ranks
    long_rows = []
    for j, trait in enumerate(trait_names):
        scores = mean_abs[:, j]
        # Rank: higher score = more important (1 = top)
        order = np.argsort(-scores)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order) + 1)

        for i, gid in enumerate(gene_ids):
            score_mean = float(mean_abs[i, j])
            score_sd = float(sd_abs[i, j]) if not np.isnan(sd_abs[i, j]) else np.nan

            row = {
                "gene_id": gid,
                "trait": trait,
                "score_mean_abs": score_mean,
                "score_sd_abs": score_sd,
                "rank_trait": int(ranks[i]),
            }

            # Annotation from gene_table
            if gid in gene_table.index:
                g = gene_table.loc[gid]
                row["gene_name"] = g.get("gene_name", "")
                row["chr"] = g.get("chr", "")
                row["start"] = g.get("start", np.nan)
                row["end"] = g.get("end", np.nan)
                row["n_snps"] = g.get("n_snps", np.nan)
            else:
                row["gene_name"] = ""
                row["chr"] = ""
                row["start"] = np.nan
                row["end"] = np.nan
                row["n_snps"] = np.nan

            # Candidate annotations (collapsed)
            if not cand_df.empty and gid in cand_df.index:
                c = cand_df.loc[gid]
                # Try to construct a summary of traits for which this gene is candidate
                cand_traits = []
                # If candidate table has a 'traits' column, use it
                if "traits" in c.index:
                    # could be per-row; we just take first
                    val = c["traits"]
                    if isinstance(val, str):
                        cand_traits = [t.strip() for t in val.split(";") if t.strip()]
                else:
                    # Fallback: look for binary columns per trait
                    for t in trait_names:
                        colname = f"is_candidate_{t}"
                        if colname in c.index:
                            try:
                                flag = bool(c[colname])
                            except Exception:
                                flag = False
                            if flag:
                                cand_traits.append(t)
                row["candidate_traits"] = ";".join(sorted(set(cand_traits)))
                row["is_candidate_any"] = int(len(cand_traits) > 0)
            else:
                row["candidate_traits"] = ""
                row["is_candidate_any"] = 0

            long_rows.append(row)

    long_df = pd.DataFrame(long_rows)
    # For nice ordering
    if set(trait_names).issuperset(set(TRAIT_ORDER)):
        trait_cat = pd.Categorical(long_df["trait"], categories=TRAIT_ORDER, ordered=True)
        long_df["trait"] = trait_cat
        long_df.sort_values(["trait", "rank_trait"], inplace=True)
    else:
        long_df.sort_values(["trait", "rank_trait"], inplace=True)

    # 5. Pleiotropy: how many traits does each gene strongly influence?
    # We'll use a per-trait 90th percentile of scores to define "high".
    thresholds = {}
    for j, trait in enumerate(trait_names):
        scores = mean_abs[:, j]
        thresholds[trait] = float(np.nanpercentile(scores, 90.0))

    pleio_rows = []
    for i, gid in enumerate(gene_ids):
        scores_gene = mean_abs[i, :]
        trait_high = [trait for j, trait in enumerate(trait_names)
                      if scores_gene[j] >= thresholds[trait]]
        pleio_row = {
            "gene_id": gid,
            "max_score": float(np.nanmax(scores_gene)),
            "n_traits_above_90pct": int(len(trait_high)),
            "traits_above_90pct": ";".join(trait_high),
        }

        if gid in gene_table.index:
            g = gene_table.loc[gid]
            pleio_row["gene_name"] = g.get("gene_name", "")
            pleio_row["chr"] = g.get("chr", "")
            pleio_row["start"] = g.get("start", np.nan)
            pleio_row["end"] = g.get("end", np.nan)
            pleio_row["n_snps"] = g.get("n_snps", np.nan)
        else:
            pleio_row["gene_name"] = ""
            pleio_row["chr"] = ""
            pleio_row["start"] = np.nan
            pleio_row["end"] = np.nan
            pleio_row["n_snps"] = np.nan

        pleio_rows.append(pleio_row)

    pleio_df = pd.DataFrame(pleio_rows)
    pleio_df.sort_values(["n_traits_above_90pct", "max_score"], ascending=[False, False], inplace=True)

    # 6. Save outputs
    safe_makedirs(OUT_EXPLAIN_DIR)

    wide_csv = os.path.join(OUT_EXPLAIN_DIR, "binn_gene_scores_wide.csv")
    long_csv = os.path.join(OUT_EXPLAIN_DIR, "binn_gene_scores_long.csv")
    pleio_csv = os.path.join(OUT_EXPLAIN_DIR, "binn_gene_pleiotropy_scores.csv")
    summary_txt = os.path.join(OUT_EXPLAIN_DIR, "binn_explain_summary.txt")

    wide_df.to_csv(wide_csv)
    long_df.to_csv(long_csv, index=False)
    pleio_df.to_csv(pleio_csv, index=False)

    print(f"[SAVE] Gene×trait wide scores -> {wide_csv}")
    print(f"[SAVE] Long gene×trait table  -> {long_csv}")
    print(f"[SAVE] Pleiotropy scores      -> {pleio_csv}")

    # 7. Human-readable summary with top 10 genes per trait
    with open(summary_txt, "w", encoding="utf-8") as fh:
        fh.write("Mango GS – Idea 3: BINN explanation summary\n")
        fh.write("==========================================\n\n")
        fh.write(f"Gene weights dir:   {BINN_WEIGHTS_DIR}\n")
        fh.write(f"Gene table:         {BINN_GENE_TABLE}\n")
        fh.write(f"Candidate summary:  {CANDIDATE_GENES_SUMMARY}\n\n")
        fh.write(f"n_folds:  {len(fold_ids)}\n")
        fh.write(f"n_genes:  {n_genes}\n")
        fh.write(f"n_traits: {n_traits}\n")
        fh.write(f"traits:   {', '.join(trait_names)}\n\n")

        fh.write("Top 10 genes per trait (by mean |gene→trait weight|):\n")
        for trait in trait_names:
            fh.write(f"\nTrait: {trait}\n")
            fh.write("-" * (7 + len(trait)) + "\n")
            sub = long_df[long_df["trait"] == trait].sort_values("rank_trait").head(10)
            for _, row in sub.iterrows():
                gid = row["gene_id"]
                gname = row.get("gene_name", "")
                score = row["score_mean_abs"]
                rank = int(row["rank_trait"])
                fh.write(f"  #{rank:2d}  {gid:20s}  {gname:30s}  score={score:.4e}\n")

        fh.write("\nTop 20 pleiotropic genes (above 90th percentile in >=2 traits):\n")
        pleio_sub = pleio_df[pleio_df["n_traits_above_90pct"] >= 2].head(20)
        if pleio_sub.empty:
            fh.write("  None exceed the 90th percentile in >=2 traits.\n")
        else:
            for _, row in pleio_sub.iterrows():
                fh.write(
                    f"  {row['gene_id']:20s}  {row.get('gene_name',''):30s}  "
                    f"max_score={row['max_score']:.4e}  "
                    f"n_traits_above_90pct={int(row['n_traits_above_90pct'])}  "
                    f"traits={row['traits_above_90pct']}\n"
                )

    print(f"[SAVE] Summary text           -> {summary_txt}")
    print("\n[DONE] BINN explanation complete.")


if __name__ == "__main__":
    main()
