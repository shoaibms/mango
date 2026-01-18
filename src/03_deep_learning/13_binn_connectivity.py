# -*- coding: utf-8 -*-
r"""
10_binn_build_maps.py

Mango GS – Idea 3
=================
Build SNP→gene maps and trait masks for the Biologically Informed Neural Network (BINN).

This script stitches together:
  * Idea 1:   geno_core.npz (frozen 20k-SNP genotype panel)
  * Idea 2:   post-GWAS candidate variant table (idea2_candidate_variants_alltraits.csv)
              + gene_annotation_core.csv (parsed from GFF)
  * Idea 3:   BINN-specific mapping files for downstream modelling

Conceptually, it does four things:

1. Load geno_core.npz and extract:
     - G (we only need the column indices)
     - sample_ids (sanity check)
     - variant_ids (SNP IDs used across the project)

2. Load the Idea 2 candidate variant table and gene annotation:
     - idea2_candidate_variants_alltraits.csv
         columns like: snp_id, chrom, pos, trait, gene_id, gene_name, ...
     - gene_annotation_core.csv
         per-gene metadata (chr, start, end, name/product, etc.)

3. Restrict to a clean BINN SNP set:
     - Only SNP features (feature_type == "snp" if present)
     - Only rows with non-null gene_id
     - Only rows whose snp_id is found in geno_core.variant_ids
     - Collapse multiple rows per SNP (across traits) into a single SNP entry:
         * Choose a unique gene_id per SNP
         * Track all traits for which the SNP is a candidate

4. Produce three outputs under Idea 3:

   (a) binn_snp_map.npz
       - snp_ids           : (n_snps,)    variant IDs (string, matches geno_core.variant_ids)
       - snp_core_index    : (n_snps,)    column index in geno_core['G']
       - snp_chr           : (n_snps,)    chromosome / contig ID (object)
       - snp_pos           : (n_snps,)    genomic position (int; NaN becomes -1)
       - snp_gene_index    : (n_snps,)    integer index into gene_ids array
       - snp_trait_matrix  : (n_snps, n_traits) 0/1 mask (candidate for trait)
       - trait_names       : (n_traits,)  trait labels (e.g. ["AFW","FBC",...])
       - gene_ids          : (n_genes,)   gene IDs in the same order as gene table

   (b) binn_snp_table.csv
       One row per SNP used in the BINN:
         snp_id, core_index, chrom, pos, gene_id, traits (semicolon-separated), n_traits

   (c) binn_gene_table.csv
       One row per gene used in the BINN:
         gene_id, gene_name, chr, start, end, strand, product,
         n_snps, n_traits, traits (semicolon-separated)

   (d) binn_maps_summary.txt
       Human-readable summary (counts, basic sanity checks).

Dependencies:
  - numpy
  - pandas

"""

import os
from typing import Dict, List

import numpy as np

try:
    import pandas as pd
except ImportError as e:
    raise SystemExit("pandas is required. Install it with: pip install pandas") from e


# =========================
# CONFIG
# =========================

# Core data (Idea 1)
GENO_CORE_PATH = r"C:\Users\ms\Desktop\mango\output\idea_1\core_data\geno_core.npz"

# Post-GWAS candidate variants (Idea 2; output of 12_build_candidate_gene_tables_idea2.py)
CANDIDATE_VARIANTS_FILE = (
    r"C:\Users\ms\Desktop\mango\output\idea_2\annotation\idea2_candidate_variants_alltraits.csv"
)

# Gene annotation (Idea 2; output of 11_build_gene_annotation_dict_idea2.py)
GENE_ANNOT_FILE = (
    r"C:\Users\ms\Desktop\mango\output\idea_2\annotation\gene_annotation_core.csv"
)

# Output root for Idea 3
OUT_ROOT = r"C:\Users\ms\Desktop\mango\output\idea_3"
OUT_BINN_DIR = os.path.join(OUT_ROOT, "binn_maps")

# If you want to hard-enforce a trait ordering, set it here; otherwise we detect from data.
# Example:
# TRAIT_ORDER = ["FBC", "FF", "AFW", "TSS", "TC"]
TRAIT_ORDER: List[str] = []  # empty => auto-detect from candidate table


# =========================
# UTILITIES
# =========================

def safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_geno_core(path: str) -> Dict[str, np.ndarray]:
    """
    Load geno_core.npz from Idea 1 and return a dict with at least:
      - X:          genotype matrix (n_samples x n_snps)
      - sample_ids: array-like of length n_samples
      - snp_ids:    array-like of length n_snps

    This mirrors the flexible loader used in Idea 3 (01_ai_core_data.py),
    but we only expose what we actually need.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"geno_core file not found:\n  {path}")

    print(f"[INFO] Loading geno_core from:\n  {path}")
    npz = np.load(path, allow_pickle=True)
    keys = list(npz.files)
    print(f"[INFO] geno_core keys: {keys}")

    geno_key_candidates = ["X", "geno", "geno_matrix", "G"]
    sample_key_candidates = ["sample_ids", "samples", "lines"]
    snp_key_candidates = ["snp_ids", "markers", "variant_ids"]

    def _pick(candidates, label):
        for c in candidates:
            if c in npz.files:
                return c
        raise KeyError(
            f"Could not find a key for {label}. "
            f"Tried {candidates}, available keys: {keys}"
        )

    geno_key = _pick(geno_key_candidates, "genotype matrix")
    sample_key = _pick(sample_key_candidates, "sample IDs")
    snp_key = _pick(snp_key_candidates, "SNP IDs")

    X = npz[geno_key]
    sample_ids = npz[sample_key]
    snp_ids = npz[snp_key]

    if X.ndim != 2:
        raise ValueError(f"Genotype matrix {geno_key} must be 2D, got shape {X.shape}")

    n_samples, n_snps = X.shape
    print(f"[INFO] Genotype matrix shape: {X.shape} (samples x SNPs)")

    if len(sample_ids) != n_samples:
        raise ValueError(
            f"sample_ids length ({len(sample_ids)}) does not match "
            f"number of rows in X ({n_samples})"
        )
    if len(snp_ids) != n_snps:
        raise ValueError(
            f"snp_ids length ({len(snp_ids)}) does not match "
            f"number of columns in X ({n_snps})"
        )

    return {
        "X": X,
        "sample_ids": np.array(sample_ids),
        "snp_ids": np.array(snp_ids),
    }


def detect_trait_column(df: pd.DataFrame) -> str:
    """Identify the column that encodes trait name."""
    if "trait" in df.columns:
        return "trait"
    if "trait_file" in df.columns:
        return "trait_file"
    raise ValueError("Candidate file is missing a trait column ('trait' or 'trait_file').")


def normalise_pos(df: pd.DataFrame, pos_col: str = "pos") -> pd.DataFrame:
    """Ensure genomic position column is numeric integer; drop rows where this fails."""
    if pos_col not in df.columns:
        print(f"[WARN] Candidate file has no '{pos_col}' column; positions will be set to -1.")
        df = df.copy()
        df[pos_col] = -1
        return df

    df = df.copy()
    df[pos_col] = pd.to_numeric(df[pos_col], errors="coerce")
    n_na = df[pos_col].isna().sum()
    if n_na > 0:
        print(f"[WARN] {n_na} candidate rows have non-numeric positions; they will be dropped.")
        df = df[df[pos_col].notna()].copy()
    df[pos_col] = df[pos_col].astype(int)
    return df


# =========================
# MAIN LOGIC
# =========================

def main():
    print("=" * 72)
    print("Mango GS – Idea 3: BINN SNP/Gene mapping (10_binn_build_maps.py)")
    print("=" * 72)
    print(f"[INFO] Geno core:          {GENO_CORE_PATH}")
    print(f"[INFO] Candidate variants: {CANDIDATE_VARIANTS_FILE}")
    print(f"[INFO] Gene annotations:   {GENE_ANNOT_FILE}")
    print(f"[INFO] Output dir:         {OUT_BINN_DIR}")
    print("")

    # ------------------------------------------------------------------
    # 1. Load geno_core
    # ------------------------------------------------------------------
    geno = load_geno_core(GENO_CORE_PATH)
    variant_ids = np.asarray(geno["snp_ids"]).astype(str)
    n_snps_core = variant_ids.shape[0]
    print(f"[INFO] Core SNPs (geno_core): {n_snps_core:,}")
    idx_lookup: Dict[str, int] = {vid: i for i, vid in enumerate(variant_ids)}

    # ------------------------------------------------------------------
    # 2. Load candidate variants + gene annotation
    # ------------------------------------------------------------------
    if not os.path.exists(CANDIDATE_VARIANTS_FILE):
        raise FileNotFoundError(
            f"Candidate variant file not found:\n  {CANDIDATE_VARIANTS_FILE}\n"
            "Run 12_build_candidate_gene_tables_idea2.py first."
        )
    if not os.path.exists(GENE_ANNOT_FILE):
        raise FileNotFoundError(
            f"Gene annotation file not found:\n  {GENE_ANNOT_FILE}\n"
            "Run 11_build_gene_annotation_dict_idea2.py first."
        )

    cand_df = pd.read_csv(CANDIDATE_VARIANTS_FILE)
    gene_df = pd.read_csv(GENE_ANNOT_FILE)

    print(f"[INFO] Loaded candidate variants: {cand_df.shape[0]:,} rows, {cand_df.shape[1]} columns")
    print(f"[INFO] Loaded gene annotations:   {gene_df.shape[0]:,} genes")
    print("")

    # ------------------------------------------------------------------
    # 3. Clean candidate table: trait column, SNP-only, gene_id present,
    #    align to geno_core.snps
    # ------------------------------------------------------------------
    trait_col = detect_trait_column(cand_df)
    cand_df = cand_df.copy()
    cand_df["trait"] = cand_df[trait_col].astype(str)

    # Feature type filter (if present)
    if "feature_type" in cand_df.columns:
        before = cand_df.shape[0]
        mask_snp = cand_df["feature_type"].fillna("").str.lower().eq("snp")
        cand_df = cand_df[mask_snp].copy()
        print(f"[INFO] Filtered to SNP features via 'feature_type': {cand_df.shape[0]:,} / {before:,} rows")
    else:
        print("[INFO] No 'feature_type' column; assuming all rows are SNPs.")

    # Require snp_id, chrom, pos, gene_id
    if "snp_id" not in cand_df.columns:
        raise ValueError("Candidate variant file must contain a 'snp_id' column.")
    if "chrom" not in cand_df.columns:
        print("[WARN] Candidate variant file has no 'chrom' column; setting chrom='NA'.")
        cand_df["chrom"] = "NA"

    cand_df = normalise_pos(cand_df, pos_col="pos")

    # Drop rows without a gene_id – BINN needs a gene layer
    if "gene_id" not in cand_df.columns:
        raise SystemExit(
            "Candidate variant table has no 'gene_id' column.\n"
            "Run 12_build_candidate_gene_tables_idea2.py first to attach gene annotations."
        )
    before_gene = cand_df.shape[0]
    cand_df = cand_df[cand_df["gene_id"].notna()].copy()
    print(f"[INFO] Rows with non-null gene_id: {cand_df.shape[0]:,} / {before_gene:,}")

    # Align snp_id with geno_core variant_ids
    cand_df["snp_id"] = cand_df["snp_id"].astype(str)
    cand_df["core_index"] = cand_df["snp_id"].map(idx_lookup)
    n_missing_core = cand_df["core_index"].isna().sum()
    if n_missing_core > 0:
        print(
            f"[WARN] {n_missing_core:,} candidate rows have snp_id not found in geno_core; "
            "they will be dropped."
        )
        cand_df = cand_df[cand_df["core_index"].notna()].copy()
    cand_df["core_index"] = cand_df["core_index"].astype(int)

    print(f"[INFO] Candidate rows after geno_core alignment: {cand_df.shape[0]:,}")
    print("")

    if cand_df.empty:
        raise SystemExit("[ERROR] No candidate SNPs overlap with geno_core; BINN maps cannot be built.")

    # ------------------------------------------------------------------
    # 4. Collapse to unique SNPs, aggregate traits per SNP
    # ------------------------------------------------------------------
    # Determine trait order
    if TRAIT_ORDER:
        trait_names = [t for t in TRAIT_ORDER if t in set(cand_df["trait"])]
    else:
        trait_names = sorted(cand_df["trait"].unique())
    trait_index: Dict[str, int] = {t: i for i, t in enumerate(trait_names)}

    print(f"[INFO] Trait set for BINN: {trait_names}")
    print(f"[INFO] Total traits: {len(trait_names)}")
    print("")

    snp_records = []
    for snp_id, sub in cand_df.groupby("snp_id", sort=False):
        core_idx = int(sub["core_index"].iloc[0])
        chrom = str(sub["chrom"].iloc[0]) if "chrom" in sub.columns else "NA"

        pos_val = sub["pos"].iloc[0] if "pos" in sub.columns else -1
        if pd.isna(pos_val):
            pos = -1
        else:
            try:
                pos = int(pos_val)
            except Exception:
                pos = -1

        # gene_id – assume consistent per SNP (if not, we warn and pick the first)
        gene_ids_for_snp = sub["gene_id"].dropna().astype(str).unique()
        gene_id = gene_ids_for_snp[0] if len(gene_ids_for_snp) > 0 else "NA"
        if len(gene_ids_for_snp) > 1:
            print(
                f"[WARN] SNP {snp_id} maps to multiple gene_ids in candidate table: "
                f"{list(gene_ids_for_snp)}; using {gene_id}."
            )

        traits_for_snp = sorted(sub["trait"].astype(str).unique())
        snp_records.append(
            {
                "snp_id": snp_id,
                "core_index": core_idx,
                "chrom": chrom,
                "pos": pos,
                "gene_id": gene_id,
                "traits": ";".join(traits_for_snp),
                "n_traits": len(traits_for_snp),
            }
        )

    snp_df = pd.DataFrame(snp_records)
    print(f"[INFO] Unique BINN SNPs: {snp_df.shape[0]:,}")
    print("")

    # ------------------------------------------------------------------
    # 5. Build gene table (restricted to genes appearing in SNP table)
    # ------------------------------------------------------------------
    gene_df = gene_df.copy()
    gene_df["gene_id"] = gene_df["gene_id"].astype(str)

    unique_gene_ids = sorted(snp_df["gene_id"].astype(str).unique())
    # Drop placeholders if any
    unique_gene_ids = [g for g in unique_gene_ids if g not in {"", "NA", "None"}]

    gene_records = []
    for gid in unique_gene_ids:
        gsub = gene_df[gene_df["gene_id"] == gid]
        if gsub.empty:
            # Gene not found in annotation (should be rare if annotation is consistent)
            chr_val = "NA"
            start = -1
            end = -1
            strand = "."
            gene_name = ""
            product = ""
        else:
            g = gsub.iloc[0]
            chr_val = str(g.get("chr", "NA"))
            start = int(g.get("start", -1)) if not pd.isna(g.get("start", np.nan)) else -1
            end = int(g.get("end", -1)) if not pd.isna(g.get("end", np.nan)) else -1
            strand = str(g.get("strand", "."))
            # Try several name/product fields
            gene_name = str(g.get("gene_name", "")) if "gene_name" in g.index else ""
            if not gene_name or gene_name == "nan":
                gene_name = str(g.get("Name", "")) if "Name" in g.index else ""
            product = str(g.get("product", "")) if "product" in g.index else ""
            if not product or product == "nan":
                product = str(g.get("description", "")) if "description" in g.index else ""

        snp_sub = snp_df[snp_df["gene_id"] == gid]
        n_snps_for_gene = snp_sub.shape[0]
        traits_for_gene = sorted(
            {t for ts in snp_sub["traits"].astype(str).tolist() for t in ts.split(";") if t}
        )
        gene_records.append(
            {
                "gene_id": gid,
                "gene_name": gene_name,
                "chr": chr_val,
                "start": start,
                "end": end,
                "strand": strand,
                "product": product,
                "n_snps": n_snps_for_gene,
                "n_traits": len(traits_for_gene),
                "traits": ";".join(traits_for_gene),
            }
        )

    gene_table = pd.DataFrame(gene_records)
    print(f"[INFO] Unique BINN genes: {gene_table.shape[0]:,}")
    print("")

    # Map gene_id -> index
    gene_ids_arr = gene_table["gene_id"].astype(str).values
    gene_id_to_idx: Dict[str, int] = {gid: i for i, gid in enumerate(gene_ids_arr)}

    # ------------------------------------------------------------------
    # 6. Build numpy mapping arrays
    # ------------------------------------------------------------------
    n_snps = snp_df.shape[0]
    n_genes = gene_table.shape[0]
    n_traits = len(trait_names)

    snp_ids_arr = snp_df["snp_id"].astype(str).values
    snp_core_index = snp_df["core_index"].astype(int).values
    snp_chr = snp_df["chrom"].astype(str).values
    # pos may contain -1; store as int64
    snp_pos = snp_df["pos"].astype(int).values

    snp_gene_index = np.full(n_snps, -1, dtype=np.int64)
    for i, gid in enumerate(snp_df["gene_id"].astype(str).values):
        if gid in gene_id_to_idx:
            snp_gene_index[i] = gene_id_to_idx[gid]
        else:
            snp_gene_index[i] = -1

    snp_trait_matrix = np.zeros((n_snps, n_traits), dtype=np.int8)
    for i, trait_str in enumerate(snp_df["traits"].astype(str).tolist()):
        if not trait_str:
            continue
        for t in trait_str.split(";"):
            t = t.strip()
            if not t:
                continue
            j = trait_index.get(t)
            if j is not None:
                snp_trait_matrix[i, j] = 1

    # ------------------------------------------------------------------
    # 7. Save outputs
    # ------------------------------------------------------------------
    safe_makedirs(OUT_BINN_DIR)

    snp_map_npz = os.path.join(OUT_BINN_DIR, "binn_snp_map.npz")
    snp_table_csv = os.path.join(OUT_BINN_DIR, "binn_snp_table.csv")
    gene_table_csv = os.path.join(OUT_BINN_DIR, "binn_gene_table.csv")
    summary_txt = os.path.join(OUT_BINN_DIR, "binn_maps_summary.txt")

    print(f"[SAVE] SNP map NPZ   -> {snp_map_npz}")
    np.savez_compressed(
        snp_map_npz,
        snp_ids=snp_ids_arr,
        snp_core_index=snp_core_index,
        snp_chr=snp_chr,
        snp_pos=snp_pos,
        snp_gene_index=snp_gene_index,
        snp_trait_matrix=snp_trait_matrix,
        trait_names=np.array(trait_names, dtype=object),
        gene_ids=gene_ids_arr,
    )

    print(f"[SAVE] SNP table CSV -> {snp_table_csv}")
    snp_df.to_csv(snp_table_csv, index=False)

    print(f"[SAVE] Gene table CSV -> {gene_table_csv}")
    gene_table.to_csv(gene_table_csv, index=False)

    # Summary text
    with open(summary_txt, "w", encoding="utf-8") as fh:
        fh.write("Mango GS – Idea 3: BINN SNP/Gene Mapping\n")
        fh.write("========================================\n\n")
        fh.write(f"geno_core path:          {GENO_CORE_PATH}\n")
        fh.write(f"candidate variants:      {CANDIDATE_VARIANTS_FILE}\n")
        fh.write(f"gene annotation:         {GENE_ANNOT_FILE}\n")
        fh.write(f"output dir:              {OUT_BINN_DIR}\n\n")
        fh.write(f"Core SNPs (geno_core):   {n_snps_core}\n")
        fh.write(f"BINN SNPs (used):        {n_snps}\n")
        fh.write(f"BINN genes (used):       {n_genes}\n")
        fh.write(f"Traits (n={n_traits}):   {', '.join(trait_names)}\n\n")

        fh.write("Notes:\n")
        fh.write("  * Only SNPs present in both geno_core and the candidate variant table\n")
        fh.write("    AND with non-null gene_id are used.\n")
        fh.write("  * snp_trait_matrix[i, j] == 1 means SNP i is a candidate for trait j.\n")
        fh.write("  * snp_gene_index[i] is the row index into binn_gene_table.csv / gene_ids.\n")

    print(f"[SAVE] Summary         -> {summary_txt}")
    print("")
    print("[DONE] BINN SNP/Gene maps built successfully.")


if __name__ == "__main__":
    main()
