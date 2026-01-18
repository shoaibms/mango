#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
11_candidate_genes.py

Mango GS – Idea 2
=================
Build final candidate variant and candidate gene tables by combining:
  * Top SNPs per trait (from idea2_candidate_genes_alltraits.csv)
  * Core gene annotation (from gene_annotation_core.csv)

Steps
-----
1. Load candidate variant list for all traits.
2. Load gene annotation dictionary (one row per gene).
3. For each variant (chrom, pos):
     - Find overlapping gene(s); if none, find nearest gene.
     - Assign gene_id, gene_name, product, coordinates, distance.
4. Produce:
     (a) Variant-level table with enriched gene annotation.
     (b) Gene-level summary table per trait, aggregating variant support.

Outputs
-------
- idea2_candidate_variants_alltraits.csv
- idea2_candidate_genes_summary.csv

"""

import os
import sys
import argparse
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Defaults (adapt if your paths differ)
# ---------------------------------------------------------------------

# Candidate variants (top SNPs per trait) – currently stored in summary/
DEFAULT_CAND_FILE = (
    r"C:\Users\ms\Desktop\mango\output\idea_2\summary\idea2_candidate_genes_alltraits.csv"
)

# Gene annotation produced by script 11 (now based on NCBI genomic.gff)
DEFAULT_GENE_ANNOT_FILE = (
    r"C:\Users\ms\Desktop\mango\output\idea_2\annotation\gene_annotation_core.csv"
)

DEFAULT_OUTDIR = r"C:\Users\ms\Desktop\mango\output\idea_2\annotation"

DEFAULT_VARIANT_OUT = "idea2_candidate_variants_alltraits.csv"
DEFAULT_GENE_OUT = "idea2_candidate_genes_summary.csv"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def detect_trait_column(df: pd.DataFrame) -> str:
    """Identify the column that encodes trait name."""
    if "trait" in df.columns:
        return "trait"
    if "trait_file" in df.columns:
        return "trait_file"
    raise ValueError("Could not detect trait column (expected 'trait' or 'trait_file').")


def ensure_numeric_pos(df: pd.DataFrame, pos_col: str = "pos") -> pd.DataFrame:
    """Ensure genomic position column is integer."""
    if pos_col not in df.columns:
        raise ValueError(f"Candidate file is missing position column '{pos_col}'.")
    df = df.copy()
    df[pos_col] = pd.to_numeric(df[pos_col], errors="coerce")
    n_na = df[pos_col].isna().sum()
    if n_na > 0:
        print(f"[WARN] {n_na} variants have non-numeric positions; they will be dropped.")
        df = df[df[pos_col].notna()].copy()
    df[pos_col] = df[pos_col].astype(int)
    return df


# ---------------------------------------------------------------------
# Gene mapping logic
# ---------------------------------------------------------------------

def build_gene_index(gene_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Build a per-chromosome index for fast lookup.

    Expects columns: 'chr', 'start', 'end', 'gene_id', 'gene_name', 'product'.
    """
    required = {"chr", "start", "end", "gene_id"}
    missing = required - set(gene_df.columns)
    if missing:
        raise ValueError(f"Gene annotation file is missing columns: {missing}")

    gene_df = gene_df.copy()
    # Ensure numeric positions
    gene_df["start"] = pd.to_numeric(gene_df["start"], errors="coerce")
    gene_df["end"] = pd.to_numeric(gene_df["end"], errors="coerce")
    gene_df = gene_df[gene_df["start"].notna() & gene_df["end"].notna()].copy()
    gene_df["start"] = gene_df["start"].astype(int)
    gene_df["end"] = gene_df["end"].astype(int)

    # Sort per chromosome
    idx: Dict[str, pd.DataFrame] = {}
    for chrom, sub in gene_df.groupby("chr"):
        sub = sub.sort_values("start", kind="mergesort").reset_index(drop=True)
        idx[chrom] = sub
    return idx


def map_variant_to_gene(
    chrom: str,
    pos: int,
    gene_idx: Dict[str, pd.DataFrame],
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[int], Optional[int], Optional[int], bool]:
    """
    Map a single variant (chrom, pos) to nearest/overlapping gene.

    Returns
    -------
    (gene_id, gene_name, product, gene_chr, gene_start, gene_end, distance_bp, overlaps)
    """
    if chrom not in gene_idx:
        return (None, None, None, None, None, None, None, False)

    genes = gene_idx[chrom]
    starts = genes["start"].values
    ends = genes["end"].values

    # Overlap first
    overlap_mask = (starts <= pos) & (ends >= pos)
    if overlap_mask.any():
        overlap_genes = genes.loc[overlap_mask]
        mids = (overlap_genes["start"].values + overlap_genes["end"].values) / 2.0
        idx_min = int(np.argmin(np.abs(mids - pos)))
        g = overlap_genes.iloc[idx_min]
        return (
            g.get("gene_id"),
            g.get("gene_name"),
            g.get("product"),
            g.get("chr"),
            int(g.get("start")),
            int(g.get("end")),
            0,
            True,
        )

    # No overlap: nearest gene by boundary distance
    # Distance to each gene's interval
    dist_to_start = np.abs(pos - starts)
    dist_to_end = np.abs(pos - ends)
    dist = np.minimum(dist_to_start, dist_to_end)
    idx_min = int(np.argmin(dist))
    g = genes.iloc[idx_min]
    d = int(dist[idx_min])

    return (
        g.get("gene_id"),
        g.get("gene_name"),
        g.get("product"),
        g.get("chr"),
        int(g.get("start")),
        int(g.get("end")),
        d,
        False,
    )


def annotate_variants_with_genes(
    cand_df: pd.DataFrame,
    gene_df: pd.DataFrame,
    force_remap: bool = False,
) -> pd.DataFrame:
    """
    For each variant, assign gene annotation using gene_df.

    If force_remap=False:
      - Only rows with missing gene_id are remapped.
    If force_remap=True:
      - All rows are remapped, overwriting existing gene_id / gene_name.
    """
    trait_col = detect_trait_column(cand_df)
    if "chrom" not in cand_df.columns:
        raise ValueError("Candidate file is missing 'chrom' column.")
    if "snp_id" not in cand_df.columns:
        raise ValueError("Candidate file is missing 'snp_id' column.")

    df = cand_df.copy()
    df = ensure_numeric_pos(df, pos_col="pos")

    # Normalise dtype
    df["chrom"] = df["chrom"].astype(str)
    df["snp_id"] = df["snp_id"].astype(str)

    if "gene_id" not in df.columns:
        df["gene_id"] = pd.NA
    if "gene_name" not in df.columns:
        df["gene_name"] = pd.NA

    # Prepare gene index
    gene_idx = build_gene_index(gene_df)

    print(f"[INFO] Variants to annotate: {df.shape[0]:,}")
    n_missing_before = df["gene_id"].isna().sum()
    print(f"[INFO] Variants with missing gene_id before mapping: {n_missing_before:,}")

    # Decide which rows to map
    if force_remap:
        mask_to_map = np.ones(df.shape[0], dtype=bool)
    else:
        mask_to_map = df["gene_id"].isna().values

    mapped_gene_id = [None] * df.shape[0]
    mapped_gene_name = [None] * df.shape[0]
    mapped_product = [None] * df.shape[0]
    mapped_chr = [None] * df.shape[0]
    mapped_start = [None] * df.shape[0]
    mapped_end = [None] * df.shape[0]
    mapped_dist = [None] * df.shape[0]
    mapped_overlap = [False] * df.shape[0]

    # Pre-group variant indices by chrom to be efficient
    for chrom, idxs in df[mask_to_map].groupby("chrom").groups.items():
        idxs = list(idxs)
        for i in idxs:
            pos = int(df.at[i, "pos"])
            (gid, gname, prod, gchr, gstart, gend, dist, overlap) = map_variant_to_gene(
                chrom=chrom,
                pos=pos,
                gene_idx=gene_idx,
            )
            mapped_gene_id[i] = gid
            mapped_gene_name[i] = gname
            mapped_product[i] = prod
            mapped_chr[i] = gchr
            mapped_start[i] = gstart
            mapped_end[i] = gend
            mapped_dist[i] = dist
            mapped_overlap[i] = overlap

    df["mapped_gene_id"] = mapped_gene_id
    df["mapped_gene_name"] = mapped_gene_name
    df["mapped_product"] = mapped_product
    df["mapped_chr"] = mapped_chr
    df["mapped_gene_start"] = mapped_start
    df["mapped_gene_end"] = mapped_end
    df["distance_to_gene_bp"] = mapped_dist
    df["overlaps_gene"] = mapped_overlap

    # Merge into main gene_id / gene_name columns
    if force_remap:
        df["gene_id"] = df["mapped_gene_id"]
        df["gene_name"] = df["mapped_gene_name"]
    else:
        df["gene_id"] = df["gene_id"].fillna(df["mapped_gene_id"])
        df["gene_name"] = df["gene_name"].fillna(df["mapped_gene_name"])

    # Add product if not present
    if "product" not in df.columns:
        df["product"] = df["mapped_product"]
    else:
        df["product"] = df["product"].fillna(df["mapped_product"])

    n_missing_after = df["gene_id"].isna().sum()
    print(f"[INFO] Variants with missing gene_id after mapping: {n_missing_after:,}")

    # Provide a clean 'trait' column (not just 'trait_file')
    if trait_col != "trait":
        df["trait"] = df[trait_col].astype(str)
    else:
        df["trait"] = df["trait"].astype(str)

    # Feature type: if not present, assume SNPs
    if "feature_type" not in df.columns:
        df["feature_type"] = "snp"

    # Use consistent gene coordinate columns
    df["gene_chr"] = df["mapped_chr"]
    df["gene_start"] = df["mapped_gene_start"]
    df["gene_end"] = df["mapped_gene_end"]

    return df


# ---------------------------------------------------------------------
# Gene-level summary
# ---------------------------------------------------------------------

def build_gene_level_summary(annot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct a gene-level summary table from the annotated variant-level table.

    One row per (gene_id, trait), with:
      - n_supporting_variants
      - basic gene annotation (name, product, chr, start, end)
      - minimum distance to any associated variant
    """
    df = annot_df.copy()
    # Keep only rows with a mapped gene
    df = df[df["gene_id"].notna()].copy()
    if df.empty:
        print("[WARN] No variants with gene_id available; gene-level summary will be empty.")
        return pd.DataFrame(
            columns=[
                "gene_id",
                "trait",
                "gene_name",
                "product",
                "chr",
                "gene_start",
                "gene_end",
                "n_supporting_variants",
                "min_distance_bp",
                "variant_ids",
            ]
        )

    group_cols = ["gene_id", "trait"]
    records = []
    for (gene_id, trait), sub in df.groupby(group_cols):
        # Consistent gene-level annotation from first row
        gene_name = sub["gene_name"].iloc[0]
        product = sub["product"].iloc[0]
        chr_val = sub["gene_chr"].iloc[0]
        gstart = sub["gene_start"].iloc[0]
        gend = sub["gene_end"].iloc[0]
        n_var = sub.shape[0]
        min_dist = (
            int(sub["distance_to_gene_bp"].min())
            if "distance_to_gene_bp" in sub.columns and sub["distance_to_gene_bp"].notna().any()
            else None
        )
        var_ids = ";".join(sorted(set(sub["snp_id"].astype(str))))

        records.append(
            {
                "gene_id": gene_id,
                "trait": trait,
                "gene_name": gene_name,
                "product": product,
                "chr": chr_val,
                "gene_start": gstart,
                "gene_end": gend,
                "n_supporting_variants": n_var,
                "min_distance_bp": min_dist,
                "variant_ids": var_ids,
            }
        )

    gene_summary = pd.DataFrame(records)
    # Order nicely
    gene_summary = gene_summary.sort_values(
        ["trait", "chr", "gene_start", "gene_id"], kind="mergesort"
    ).reset_index(drop=True)

    return gene_summary


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build final candidate variant and gene tables for Mango GS Idea 2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cand",
        type=str,
        default=DEFAULT_CAND_FILE,
        help=f"Path to candidate variant list (idea2_candidate_genes_alltraits.csv). Default: {DEFAULT_CAND_FILE}",
    )
    parser.add_argument(
        "--gene-annot",
        type=str,
        default=DEFAULT_GENE_ANNOT_FILE,
        help=f"Path to core gene annotation CSV (gene_annotation_core.csv). Default: {DEFAULT_GENE_ANNOT_FILE}",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=DEFAULT_OUTDIR,
        help=f"Output directory for final candidate tables. Default: {DEFAULT_OUTDIR}",
    )
    parser.add_argument(
        "--force-remap",
        action="store_true",
        help="If set, remap all variants to genes (overwrite existing gene_id/gene_name).",
    )
    parser.add_argument(
        "--variant-out",
        type=str,
        default=DEFAULT_VARIANT_OUT,
        help="Output file name for variant-level table.",
    )
    parser.add_argument(
        "--gene-out",
        type=str,
        default=DEFAULT_GENE_OUT,
        help="Output file name for gene-level summary table.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    print("========================================================================")
    print("Mango GS – Idea 2: Build candidate variant and gene tables")
    print("========================================================================")
    print(f"[INFO] Candidate variants: {args.cand}")
    print(f"[INFO] Gene annotation:    {args.gene_annot}")
    print(f"[INFO] Output dir:          {args.outdir}")
    print(f"[INFO] Force remap:         {args.force_remap}")
    print("")

    if not os.path.exists(args.cand):
        print(f"[ERROR] Candidate file not found: {args.cand}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.gene_annot):
        print(f"[ERROR] Gene annotation file not found: {args.gene_annot}", file=sys.stderr)
        sys.exit(1)

    safe_makedirs(args.outdir)

    # Load inputs
    cand_df = pd.read_csv(args.cand)
    gene_df = pd.read_csv(args.gene_annot)

    print(f"[INFO] Loaded candidates: {cand_df.shape[0]:,} rows, {cand_df.shape[1]} columns")
    print(f"[INFO] Loaded gene annotations: {gene_df.shape[0]:,} genes")

    # Annotate variants
    annot_df = annotate_variants_with_genes(
        cand_df=cand_df,
        gene_df=gene_df,
        force_remap=args.force_remap,
    )

    # Save variant-level table
    variant_out_path = os.path.join(args.outdir, args.variant_out)
    annot_df.to_csv(variant_out_path, index=False)
    print(f"[SAVE] Candidate variants (annotated) -> {variant_out_path}")

    # Build gene-level summary
    gene_summary_df = build_gene_level_summary(annot_df)
    gene_out_path = os.path.join(args.outdir, args.gene_out)
    gene_summary_df.to_csv(gene_out_path, index=False)
    print(f"[SAVE] Candidate gene summary       -> {gene_out_path}")

    print("")
    print("[OK] Candidate variant and gene tables built successfully.")


if __name__ == "__main__":
    main()
