"""
02_external_validation.py

External validation against Jighly et al. (2026) integrated mango panel.

Analyses:
  1. ADMIXTURE concordance (your K=3 vs their K=6)
  2. QTL-TagSNP overlap analysis

Outputs (to output/idea_2/external_validation/):
  - admixture_crosstab.csv
  - admixture_q_by_cluster.csv
  - admixture_merged_samples.csv
  - qtl_tagsnp_overlap.csv
  - chromosome_concordance.csv
  - external_validation_summary.csv
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ------------------ PATH SETUP ------------------
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
except NameError:
    ROOT = r"C:\Users\ms\Desktop\mango"

# ------------------ FILE PATHS ------------------
JIGHLY_XLSX = os.path.join(ROOT, "data", "main_data", "jighly.xlsx")
SRR_FUR_MAP = os.path.join(ROOT, "data", "main_data", "srr_to_fur_map.csv")
CLUSTER_CSV = os.path.join(ROOT, "output", "idea_2", "core_ml", "sample_metadata_ml.csv")
TABLE_S9 = os.path.join(ROOT, "output", "idea_2", "breeder_tools", "Table_S9_Inversion_Tag_SNP_Assays.csv")
OUT_DIR = os.path.join(ROOT, "output", "idea_2", "external_validation")

WINDOW_KB = 500  # ±500 kb for QTL overlap

os.makedirs(OUT_DIR, exist_ok=True)


# ------------------ UTILITY FUNCTIONS ------------------
def log(msg, level="INFO"):
    prefix = {"INFO": "[INFO]", "OK": "[OK]", "WARN": "[WARN]", 
              "ERROR": "[ERROR]", "DONE": "[DONE]"}.get(level, "[INFO]")
    print(f"{prefix} {msg}")


def find_column(df, candidates, default_idx=0):
    """Find first matching column from candidates list."""
    for c in candidates:
        if c in df.columns:
            return c
    return df.columns[default_idx] if len(df.columns) > default_idx else None


def extract_chr_number(chrom_val):
    """Extract chromosome number from various formats."""
    if pd.isna(chrom_val):
        return None
    s = str(chrom_val).strip()
    
    # NC_058137.1 -> 1
    if "NC_058" in s:
        try:
            num = int(s.replace("NC_", "").split(".")[0]) - 58136
            if 1 <= num <= 20:
                return num
        except:
            pass
    
    # GWHABLA00000001 -> 1
    if "GWHABLA" in s.upper():
        try:
            return int(s.upper().replace("GWHABLA", ""))
        except:
            pass
    
    # chr1 -> 1
    if s.lower().startswith("chr"):
        try:
            return int(s[3:])
        except:
            pass
    
    # Integer
    try:
        num = int(float(s))
        if 1 <= num <= 20:
            return num
    except:
        pass
    
    return None


# ==============================================================================
# PART 1: ADMIXTURE CONCORDANCE
# ==============================================================================
def run_admixture_concordance():
    """Compare your K=3 clusters with Jighly K=6 ADMIXTURE."""
    
    log("=" * 60)
    log("PART 1: ADMIXTURE CONCORDANCE")
    log("=" * 60)
    
    results = {"ari": None, "nmi": None, "n_matched": 0}
    
    if not os.path.exists(SRR_FUR_MAP):
        log("SRR->FUR mapping not found. Run 01_id_mapping_validation.py first.", "WARN")
        return results
    
    try:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    except ImportError:
        log("sklearn not available", "WARN")
        return results
    
    # Load data
    cluster_df = pd.read_csv(CLUSTER_CSV)
    cluster_col = find_column(cluster_df, ["cluster_kmeans", "cluster", "Cluster"])
    id_col = find_column(cluster_df, ["sample_id", "Sample", "ID", "Accession"])
    
    hr_df = pd.read_excel(JIGHLY_XLSX, sheet_name="TableS1")
    hr_aus = hr_df[hr_df["Population"] == "Australia"].copy()
    
    map_df = pd.read_csv(SRR_FUR_MAP, dtype=str)
    
    # Merge
    hr_aus["Geno"] = hr_aus["Geno"].astype(str)
    hr_merged = hr_aus.merge(map_df, on="Geno", how="inner")
    
    cluster_df[id_col] = cluster_df[id_col].astype(str)
    final = hr_merged.merge(cluster_df[[id_col, cluster_col]], 
                            left_on="sample_id", right_on=id_col, how="inner")
    
    results["n_matched"] = len(final)
    log(f"Matched samples: {len(final)}")
    
    if len(final) < 100:
        log("Too few matches", "WARN")
        return results
    
    # Metrics
    your_k3 = final[cluster_col].values
    jighly_k6 = final["ADMIXTURE_Pop"].values
    
    ari = adjusted_rand_score(your_k3, jighly_k6)
    nmi = normalized_mutual_info_score(your_k3, jighly_k6)
    
    results["ari"] = round(ari, 3)
    results["nmi"] = round(nmi, 3)
    
    log(f"ARI: {ari:.3f}, NMI: {nmi:.3f}", "OK")
    
    # Cross-tabulation
    crosstab = pd.crosstab(final[cluster_col], final["ADMIXTURE_Pop"], 
                           margins=True, margins_name="Total")
    
    # Q-proportions
    q_cols = [c for c in final.columns if c.startswith("Q")]
    if q_cols:
        q_summary = final.groupby(cluster_col)[q_cols].mean()
        q_summary.to_csv(os.path.join(OUT_DIR, "admixture_q_by_cluster.csv"))
    
    # Save
    crosstab.to_csv(os.path.join(OUT_DIR, "admixture_crosstab.csv"))
    final.to_csv(os.path.join(OUT_DIR, "admixture_merged_samples.csv"), index=False)
    
    log("ADMIXTURE concordance complete", "DONE")
    return results


# ==============================================================================
# PART 2: QTL-TAGSNP OVERLAP
# ==============================================================================
def run_qtl_tagsnp_overlap():
    """Check overlap between Jighly QTL peaks and your inversion tag SNPs."""
    
    log("\n" + "=" * 60)
    log("PART 2: QTL-TAGSNP OVERLAP")
    log("=" * 60)
    
    results = {"n_overlaps": 0, "n_inversions_with_overlap": 0, 
               "chr_with_fw_qtl": 0, "chr_with_tss_qtl": 0, "chr_total": 0}
    
    if not os.path.exists(TABLE_S9):
        log(f"Table S9 not found: {TABLE_S9}", "WARN")
        return results
    
    # Load tag SNPs
    tag_df = pd.read_csv(TABLE_S9)
    chr_col = find_column(tag_df, ["Chrom", "FASTA_Chrom", "Chromosome", "chr"])
    pos_col = find_column(tag_df, ["Pos", "Position", "pos"])
    inv_col = find_column(tag_df, ["Inversion", "inversion", "Marker"])
    
    # Load QTL
    qtl_df = pd.read_excel(JIGHLY_XLSX, sheet_name="TableS3")
    qtl_chr_col = find_column(qtl_df, ["chr", "Chr", "Chromosome"])
    qtl_pos_col = find_column(qtl_df, ["ps", "pos", "Position"])
    qtl_trait_col = find_column(qtl_df, ["Trait", "trait"])
    
    # Map chromosomes
    tag_df["chr_num"] = tag_df[chr_col].apply(extract_chr_number)
    qtl_df["chr_num"] = qtl_df[qtl_chr_col].apply(extract_chr_number)
    
    valid_tags = tag_df["chr_num"].notna().sum()
    log(f"Tag SNPs with valid chr: {valid_tags}/{len(tag_df)}")
    
    if valid_tags == 0:
        log("No valid chromosome mappings", "ERROR")
        return results
    
    # Find overlaps
    window_bp = WINDOW_KB * 1000
    overlaps = []
    
    for _, qtl in qtl_df.iterrows():
        qtl_chr = qtl["chr_num"]
        qtl_pos = qtl[qtl_pos_col]
        qtl_trait = qtl.get(qtl_trait_col, "Unknown") if qtl_trait_col else "Unknown"
        
        if pd.isna(qtl_chr) or pd.isna(qtl_pos):
            continue
        
        same_chr = tag_df[tag_df["chr_num"] == qtl_chr]
        
        for _, tag in same_chr.iterrows():
            tag_pos = tag[pos_col]
            if pd.isna(tag_pos):
                continue
            
            distance = abs(int(tag_pos) - int(qtl_pos))
            
            if distance <= window_bp:
                overlaps.append({
                    "QTL_Trait": qtl_trait,
                    "QTL_chr": int(qtl_chr),
                    "QTL_pos": int(qtl_pos),
                    "Inversion": tag.get(inv_col, "Unknown") if inv_col else "Unknown",
                    "Tag_SNP_pos": int(tag_pos),
                    "Distance_kb": round(distance / 1000, 1),
                })
    
    overlap_df = pd.DataFrame(overlaps)
    results["n_overlaps"] = len(overlap_df)
    
    if len(overlap_df) > 0:
        results["n_inversions_with_overlap"] = overlap_df["Inversion"].nunique()
        overlap_df.to_csv(os.path.join(OUT_DIR, "qtl_tagsnp_overlap.csv"), index=False)
        log(f"Overlaps found: {len(overlap_df)} (±{WINDOW_KB}kb)", "OK")
    else:
        log("No bp-level overlaps found", "WARN")
    
    # Chromosome concordance
    inv_chroms = set(tag_df["chr_num"].dropna().astype(int))
    results["chr_total"] = len(inv_chroms)
    
    chr_summary = []
    for chr_num in sorted(inv_chroms):
        inv_list = tag_df[tag_df["chr_num"] == chr_num][inv_col].unique().tolist() if inv_col else []
        chr_qtl = qtl_df[qtl_df["chr_num"] == chr_num]
        has_fw = "FW" in chr_qtl[qtl_trait_col].values if qtl_trait_col and len(chr_qtl) > 0 else False
        has_tss = "TSS" in chr_qtl[qtl_trait_col].values if qtl_trait_col and len(chr_qtl) > 0 else False
        
        chr_summary.append({
            "Chromosome": chr_num,
            "Inversions": ", ".join(map(str, inv_list)),
            "Has_FW_QTL": has_fw,
            "Has_TSS_QTL": has_tss,
        })
    
    chr_summary_df = pd.DataFrame(chr_summary)
    
    if len(chr_summary_df) > 0:
        results["chr_with_fw_qtl"] = chr_summary_df["Has_FW_QTL"].sum()
        results["chr_with_tss_qtl"] = chr_summary_df["Has_TSS_QTL"].sum()
        chr_summary_df.to_csv(os.path.join(OUT_DIR, "chromosome_concordance.csv"), index=False)
        log(f"Chr with FW QTL: {results['chr_with_fw_qtl']}/{results['chr_total']}")
        log(f"Chr with TSS QTL: {results['chr_with_tss_qtl']}/{results['chr_total']}")
    
    log("QTL-TagSNP overlap complete", "DONE")
    return results


# ==============================================================================
# PART 3: CROSS-COLLECTION TRANSFER DATA
# ==============================================================================
def export_cross_collection_data():
    """Export cross-collection transfer data from Jighly et al. Table 2."""
    
    log("\n" + "=" * 60)
    log("PART 3: CROSS-COLLECTION TRANSFER DATA")
    log("=" * 60)
    
    # Load from Jighly Excel if available, otherwise use published values
    # These are from Jighly et al. (2026) Table 2, single-trait GP model
    
    # Try to load from Excel first
    transfer_data = []
    collections = ["AUS", "USA", "CHN"]
    
    # FW transfer matrix (Reference -> Validation)
    fw_matrix = {
        ("AUS", "AUS"): 0.71, ("AUS", "USA"): 0.63, ("AUS", "CHN"): 0.04,
        ("USA", "AUS"): 0.60, ("USA", "USA"): 0.60, ("USA", "CHN"): 0.03,
        ("CHN", "AUS"): 0.00, ("CHN", "USA"): 0.00, ("CHN", "CHN"): 0.20,
    }
    
    # TSS transfer matrix
    tss_matrix = {
        ("AUS", "AUS"): 0.69, ("AUS", "USA"): 0.52, ("AUS", "CHN"): 0.07,
        ("USA", "AUS"): 0.52, ("USA", "USA"): 0.61, ("USA", "CHN"): 0.04,
        ("CHN", "AUS"): 0.05, ("CHN", "USA"): 0.03, ("CHN", "CHN"): 0.14,
    }
    
    for ref in collections:
        for val in collections:
            transfer_data.append({
                "Reference": ref,
                "Validation": val,
                "FW_accuracy": fw_matrix[(ref, val)],
                "TSS_accuracy": tss_matrix[(ref, val)],
            })
    
    transfer_df = pd.DataFrame(transfer_data)
    out_path = os.path.join(OUT_DIR, "cross_collection_transfer.csv")
    transfer_df.to_csv(out_path, index=False)
    
    log(f"Saved: {out_path}", "OK")
    log("Cross-collection data export complete", "DONE")
    
    return {"n_collections": len(collections)}


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("\n" + "=" * 60)
    print("EXTERNAL VALIDATION - Jighly et al. (2026)")
    print("=" * 60)
    print(f"Output: {OUT_DIR}\n")
    
    # Run analyses
    admix = run_admixture_concordance()
    qtl = run_qtl_tagsnp_overlap()
    transfer = export_cross_collection_data()
    
    # Summary table
    summary = [
        {"Metric": "ADMIXTURE_ARI", "Value": admix.get("ari", "NA")},
        {"Metric": "ADMIXTURE_NMI", "Value": admix.get("nmi", "NA")},
        {"Metric": "ADMIXTURE_N_matched", "Value": admix.get("n_matched", 0)},
        {"Metric": "QTL_overlaps", "Value": qtl.get("n_overlaps", 0)},
        {"Metric": "Inversions_with_overlap", "Value": qtl.get("n_inversions_with_overlap", 0)},
        {"Metric": "Chr_with_FW_QTL", "Value": f"{qtl.get('chr_with_fw_qtl', 0)}/{qtl.get('chr_total', 0)}"},
        {"Metric": "Chr_with_TSS_QTL", "Value": f"{qtl.get('chr_with_tss_qtl', 0)}/{qtl.get('chr_total', 0)}"},
        {"Metric": "Cross_collection_n", "Value": transfer.get("n_collections", 0)},
    ]
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(OUT_DIR, "external_validation_summary.csv"), index=False)
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"\nOutputs: {OUT_DIR}/")
    for f in ["admixture_crosstab.csv", "admixture_q_by_cluster.csv", 
              "qtl_tagsnp_overlap.csv", "chromosome_concordance.csv",
              "cross_collection_transfer.csv", "external_validation_summary.csv"]:
        print(f"  - {f}")


if __name__ == "__main__":
    main()