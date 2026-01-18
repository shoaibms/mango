#!/usr/bin/env python
"""
14_manuscript_tables.py

Generate manuscript tables from candidate gene summary data.

Outputs:
    - Table S3: Full supplementary candidate gene list with pleiotropy scores.
    - Table 2: Main manuscript selection highlighting key loci.
"""

import pandas as pd
import os

# =========================
# CONFIG
# =========================
INPUT_FILE = r"C:\Users\ms\Desktop\mango\output\idea_2\annotation\idea2_candidate_genes_summary.csv"
OUTPUT_DIR = r"C:\Users\ms\Desktop\mango\output\idea_2\manuscript_tables"

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] Input file not found: {INPUT_FILE}")
        return
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Load Data
    df = pd.read_csv(INPUT_FILE)
    print(f"[INFO] Loaded {len(df)} gene-trait associations.")

    # 2. Calculate Pleiotropy (How many traits does each gene affect?)
    # Group by gene_id and count unique traits
    pleiotropy = df.groupby('gene_id')['trait'].nunique().rename("trait_count")
    
    # Merge back
    df = df.merge(pleiotropy, on='gene_id')
    
    # Create a "Traits" string (e.g., "FBC;AFW;TSS")
    trait_lists = df.groupby('gene_id')['trait'].apply(lambda x: ";".join(sorted(set(x)))).rename("affected_traits")
    df = df.merge(trait_lists, on='gene_id')

    # 3. Generate Table S3 (Full Supplement)
    # Columns: Gene ID, Chromosome, Start, End, Affected Traits, Max Supporting SNPs, Min Distance
    
    # Aggregate per gene (taking max SNPs and min distance across traits)
    table_s3 = df.groupby('gene_id').agg({
        'chr': 'first',
        'gene_start': 'first',
        'gene_end': 'first',
        'affected_traits': 'first',
        'trait_count': 'first',
        'n_supporting_variants': 'max', # Max SNPs support across any trait
        'min_distance_bp': 'min'        # Closest variant across any trait
    }).reset_index()

    # Sort by Pleiotropy (High to Low), then Chromosome
    table_s3 = table_s3.sort_values(by=['trait_count', 'chr', 'gene_start'], ascending=[False, True, True])
    
    s3_path = os.path.join(OUTPUT_DIR, "Table_S3_All_Candidate_Genes.csv")
    table_s3.to_csv(s3_path, index=False)
    print(f"[SAVE] Table S3 saved to: {s3_path}")

    # 4. Generate Table 2 (Main Manuscript Selection)
    # Criteria:
    # A. The "Pan-Trait" Loci (Pleiotropy >= 4)
    # B. The Top Gene for each specific trait (Highest SNP support)
    
    # A. Pan-Trait
    pan_trait_genes = table_s3[table_s3['trait_count'] >= 4].copy()
    pan_trait_genes['Category'] = 'Pleiotropic Hub'

    # B. Top Specific Genes (For traits not fully covered by hubs)
    # We look at the original 'df' to find the best gene for each trait
    top_specific = []
    for trait in ['FBC', 'AFW', 'TSS', 'TC']:
        # Get genes for this trait
        trait_genes = df[df['trait'] == trait].sort_values('n_supporting_variants', ascending=False)
        
        # Pick the top one that ISN'T already a pan-trait hub
        for _, row in trait_genes.iterrows():
            if row['gene_id'] not in pan_trait_genes['gene_id'].values:
                # Convert series to frame row
                row_frame = table_s3[table_s3['gene_id'] == row['gene_id']].copy()
                row_frame['Category'] = f'Top {trait} Locus'
                top_specific.append(row_frame)
                break
    
    if top_specific:
        df_specific = pd.concat(top_specific)
        table_2 = pd.concat([pan_trait_genes, df_specific])
    else:
        table_2 = pan_trait_genes

    # Clean up Table 2 for display
    table_2 = table_2[['Category', 'gene_id', 'chr', 'affected_traits', 'n_supporting_variants', 'min_distance_bp']]
    
    t2_path = os.path.join(OUTPUT_DIR, "Table_2_Key_Loci.csv")
    table_2.to_csv(t2_path, index=False)
    print(f"[SAVE] Table 2 saved to: {t2_path}")
    print("\n[SUMMARY]")
    print(f"  - Total Unique Genes: {len(table_s3)}")
    print(f"  - Pan-Trait Hubs (>=4 traits): {len(pan_trait_genes)}")
    print(f"  - Selected for Table 2: {len(table_2)}")

if __name__ == "__main__":
    main()