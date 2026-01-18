#!/usr/bin/env python
"""
13_gene_mapping_qc.py

Inspect and validate post-GWAS gene mapping results.

Objectives:
    1. Verify chromosome IDs match NCBI RefSeq format.
    2. Summarize per-trait counts of genes and variants.
    3. Display example rows for manuscript reporting.
"""

import os
import pandas as pd

# =========================
# CONFIG
# =========================
INPUT_DIR = r"C:\Users\ms\Desktop\mango\output\idea_2\annotation"
VARIANTS_FILE = os.path.join(INPUT_DIR, "idea2_candidate_variants_alltraits.csv")
GENES_FILE = os.path.join(INPUT_DIR, "idea2_candidate_genes_summary.csv")

def main():
    print("="*80)
    print(" INSPECTION: Post-GWAS Gene Mapping (NCBI RefSeq)")
    print("="*80)

    # -------------------------------------------------------
    # 1. Load Data
    # -------------------------------------------------------
    if not os.path.exists(VARIANTS_FILE) or not os.path.exists(GENES_FILE):
        print("[ERROR] Required input files not found.")
        return

    df_var = pd.read_csv(VARIANTS_FILE)
    df_gene = pd.read_csv(GENES_FILE)

    print(f"[LOAD] Variants Table: {df_var.shape[0]} rows")
    print(f"[LOAD] Genes Table:    {df_gene.shape[0]} rows")
    print("-" * 80)

    # -------------------------------------------------------
    # 2. Check Chromosome IDs
    # -------------------------------------------------------
    print("\n[CHECK 1] Chromosome ID Format:")
    unique_chroms = df_var['chrom'].unique()
    
    print(f"  Found in variants: {unique_chroms[:5]}")
    if any(str(c).startswith("NC_") for c in unique_chroms):
        print("  [OK] Detected NCBI RefSeq IDs (NC_...). Mapping is valid.")
    elif any(str(c).startswith("GWH") for c in unique_chroms):
        print("  [WARNING] Detected GWH IDs. This implies a mismatch if VCF was NC_.")
    else:
        print("  [NOTE] Chromosome IDs do not follow standard format.")

    # -------------------------------------------------------
    # 3. Per-Trait Counts
    # -------------------------------------------------------
    print("\n[CHECK 2] Summary Counts per Trait:")
    
    # Group variant table
    var_counts = df_var.groupby('trait')['snp_id'].count().rename("Variants")
    
    # Group gene table (count unique genes per trait)
    gene_counts = df_gene.groupby('trait')['gene_id'].nunique().rename("Unique Genes")
    
    # Combine
    summary = pd.concat([var_counts, gene_counts], axis=1).fillna(0).astype(int)
    print(summary)

    # -------------------------------------------------------
    # 4. Check Mapping Success Rate
    # -------------------------------------------------------
    print("\n[CHECK 3] Mapping Success Rate:")
    mapped = df_var['gene_id'].notna().sum()
    total = len(df_var)
    print(f"  Variants successfully linked to a gene: {mapped} / {total} ({mapped/total*100:.1f}%)")
    
    if mapped == total:
        print("  [OK] 100% of top candidates mapped to genes.")
    
    # -------------------------------------------------------
    # 5. Example Rows
    # -------------------------------------------------------
    print("\n[CHECK 4] Example Result Rows (Top 3 Variants):")
    cols_to_show = ['trait', 'chrom', 'pos', 'gene_id', 'gene_name', 'product', 'dist_to_gene']
    
    # Filter cols that actually exist
    cols = [c for c in cols_to_show if c in df_var.columns]
    print(df_var[cols].head(3).to_string(index=False))

    print("\n[CHECK 5] Example Result Rows (Top 3 Genes):")
    cols_gene = ['trait', 'gene_id', 'gene_name', 'product', 'n_variants']
    cols_g = [c for c in cols_gene if c in df_gene.columns]
    print(df_gene[cols_g].head(3).to_string(index=False))

    print("\n" + "="*80)
    print("[OK] Inspection complete.")

if __name__ == "__main__":
    main()