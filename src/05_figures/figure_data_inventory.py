
r"""
Figure Data Inventory Script
=============================
Generates comprehensive inventory of all data files required for manuscript figures.

Output:
  - Console log with debug information
  - Markdown report: C:\Users\ms\Desktop\mango\output\figures\figure_data_inventory.md
  - Text summary: C:\Users\ms\Desktop\mango\output\figures\figure_data_inventory.txt

"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = r"C:\Users\ms\Desktop\mango"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

# Ensure output directory exists
os.makedirs(FIGURES_DIR, exist_ok=True)

# Output files
INVENTORY_MD = os.path.join(FIGURES_DIR, "figure_data_inventory.md")
INVENTORY_TXT = os.path.join(FIGURES_DIR, "figure_data_inventory.txt")

# Debug flag
DEBUG = True

# ============================================================================
# REQUIRED DATA FILES - Organized by Figure
# ============================================================================

REQUIRED_FILES = {
    # ========================================================================
    # FIGURE 1: Population Structure and Structure Cliff
    # ========================================================================
    "Figure 1": {
        "1A_1B_pc_scores": {
            "path": "idea_1/summary/pc_scores_clusters.csv",
            "description": "PCA scores with cluster assignments for population structure",
            "expected_cols": ["PC1", "PC2", "cluster"],
            "panel": "A, B"
        },
        "1B_cluster_sizes": {
            "path": "idea_1/summary/cluster_sizes.csv",
            "description": "Size of each ancestry cluster",
            "expected_cols": ["cluster", "n_samples"],
            "panel": "B"
        },
        "1C_1E_cv_transferability": {
            "path": "idea_1/summary/cv_transferability_summary.csv",
            "description": "CV accuracy across schemes (Random, Balanced, LCO)",
            "expected_cols": ["trait", "r_random_pc", "r_cluster_balanced_pc", "r_leave_cluster_out_pc"],
            "panel": "C, E"
        },
        "1D_pheno_summary": {
            "path": "idea_1/summary/pheno_trait_summary.csv",
            "description": "Phenotype statistics (mean, SD, CV%) per trait",
            "expected_cols": ["trait", "mean", "sd", "cv_percent"],
            "panel": "D"
        },
        "1F_pheno_core": {
            "path": "idea_1/core_data/pheno_core.csv",
            "description": "Raw phenotype values for correlation matrix",
            "expected_cols": ["FBC", "FF", "AFW", "TSS", "TC"],
            "panel": "F"
        },
    },
    
    # ========================================================================
    # FIGURE 2: Structural Haplotypes as Ultra-Efficient Predictors
    # ========================================================================
    "Figure 2": {
        "2A_2E_haplotype_effects": {
            "path": "idea_2/breeder_tools/Breeder_Haplotype_Effects.csv",
            "description": "Inversion haplotype effects (standardized) per trait",
            "expected_cols": ["Trait", "Marker", "Effect_Std", "Is_Additive"],
            "panel": "A, E"
        },
        "2A_assay_design": {
            "path": "idea_2/breeder_tools/Supplementary_Table_Assay_Design.csv",
            "description": "KASP assay design with genomic positions",
            "expected_cols": ["Marker", "Trait", "Chromosome", "Position"],
            "panel": "A, 5D"
        },
        "2B_random_vs_inversion": {
            "path": "idea_2/random_control/random_vs_inversion_replicates.csv",
            "description": "Replicate-level comparison of inversion vs random panels",
            "expected_cols": ["trait", "replicate", "random_r", "inversion_r"],
            "panel": "B"
        },
        "2C_meta_core": {
            "path": "idea_1/core_data/meta_core.csv",
            "description": "Sample metadata including inversion genotypes",
            "expected_cols": ["miinv6.0", "miinv11.0", "miinv17.0"],
            "panel": "C"
        },
        "2D_model_performance": {
            "path": "idea_2/summary/idea2_gs_model_performance_clean.csv",
            "description": "Model performance grid (trait × scheme × model)",
            "expected_cols": ["trait", "cv_scheme", "feature_set", "mean_r"],
            "panel": "D"
        },
        "2F_genetic_gain": {
            "path": "idea_2/breeder_tools/Estimated_Genetic_Gain.csv",
            "description": "Expected genetic gain per cycle",
            "expected_cols": ["Trait", "Gain_Percent"],
            "panel": "F, 5C"
        },
    },
    
    # ========================================================================
    # FIGURE 3: Deep Learning Confirms Additive Mechanisms
    # ========================================================================
    "Figure 3": {
        "3A_model_performance": {
            "path": "idea_3/metrics/idea3_model_performance_summary.csv",
            "description": "Model comparison (Ridge, XGB, RF, MLP, Wide&Deep)",
            "expected_cols": ["trait", "model", "pearson_r_mean"],
            "panel": "A"
        },
        "3B_ai_gwas_FBC": {
            "path": "idea_3/interpretation/ai_vs_gwas/ai_gwas_merged_trait-FBC.csv",
            "description": "Merged AI saliency + GWAS for FBC",
            "expected_cols": ["snp_id", "saliency_FBC", "p_FBC", "beta_FBC"],
            "panel": "B"
        },
        "3C_saliency_summary": {
            "path": "idea_3/interpretation/idea3_saliency_summary.csv",
            "description": "Saliency concentration by top SNP percentile",
            "expected_cols": ["trait", "top_1pct_share", "top_5pct_share"],
            "panel": "C"
        },
        "3D_block_synergy": {
            "path": "idea_3/interpretation/editing/advanced/haplotype_block_synergy.csv",
            "description": "Virtual editing block vs sum-of-singles comparison",
            "expected_cols": ["block_effect", "sum_singles", "synergy"],
            "panel": "D"
        },
        "3E_editing_tradeoff": {
            "path": "idea_3/interpretation/idea3_editing_tradeoff_summary.csv",
            "description": "Cross-trait effects when editing FBC SNPs",
            "expected_cols": ["target_trait", "delta_FBC", "delta_AFW"],
            "panel": "E"
        },
        "3F_concordance_summary": {
            "path": "idea_3/interpretation/ai_vs_gwas/ai_gwas_concordance_summary.csv",
            "description": "Correlation between AI importance and GWAS",
            "expected_cols": ["trait", "corr_saliency_logp", "corr_saliency_beta"],
            "panel": "F"
        },
    },
    
    # ========================================================================
    # FIGURE 4: Polygenic Backbones and Pleiotropic Hub Genes
    # ========================================================================
    "Figure 4": {
        "4A_binn_scores_wide": {
            "path": "idea_3/binn_explain/binn_gene_scores_wide.csv",
            "description": "BINN gene importance scores (gene × trait matrix)",
            "expected_cols": ["gene_id", "score_FBC", "score_AFW"],
            "panel": "A"
        },
        "4B_polygenic_architecture": {
            "path": "idea_3/breeder_resources/Polygenic_Architecture_Summary.csv",
            "description": "Polygenic summary (top 1%, 5% weight share)",
            "expected_cols": ["trait", "top_1pct_weight_share", "top_5pct_weight_share"],
            "panel": "B"
        },
        "4C_polygenic_eval": {
            "path": "idea_3/breeder_resources/Mango_Polygenic_Evaluation_File.csv",
            "description": "Per-SNP weights for cumulative variance plots",
            "expected_cols": ["snp_id", "weight_FBC", "weight_AFW"],
            "panel": "C"
        },
        "4D_pleiotropy_scores": {
            "path": "idea_3/binn_explain/binn_gene_pleiotropy_scores.csv",
            "description": "Gene pleiotropy classification",
            "expected_cols": ["gene_id", "trait_count", "traits_above_90pct"],
            "panel": "D"
        },
        "4E_saliency_matrix": {
            "path": "idea_3/interpretation/saliency/saliency_matrix_block-raw.csv",
            "description": "Full saliency matrix for cross-trait correlations",
            "expected_cols": ["saliency_FBC", "saliency_AFW", "saliency_FF"],
            "panel": "E"
        },
        "4F_binn_cv_summary": {
            "path": "idea_3/binn_training/binn_cv_summary.csv",
            "description": "BINN cross-validation performance",
            "expected_cols": ["trait", "mean_r", "std_r"],
            "panel": "F"
        },
    },
    
    # ========================================================================
    # FIGURE 5: Precision Breeding Hierarchy
    # ========================================================================
    "Figure 5": {
        "5A_hierarchy_inputs": {
            "path": "idea_1/summary/cv_transferability_summary.csv",
            "description": "Transferability scores for hierarchy map (same as 1C)",
            "expected_cols": ["trait", "r_random_pc", "r_leave_cluster_out_pc"],
            "panel": "A"
        },
        "5A_structural_scores": {
            "path": "idea_3/breeder_resources/Polygenic_Architecture_Summary.csv",
            "description": "Structural dominance scores (same as 4B)",
            "expected_cols": ["trait", "top_1pct_weight_share"],
            "panel": "A"
        },
        "5C_genetic_gain": {
            "path": "idea_2/breeder_tools/Estimated_Genetic_Gain.csv",
            "description": "Genetic gain for tier comparison (same as 2F)",
            "expected_cols": ["Trait", "Gain_Percent"],
            "panel": "C"
        },
        "5D_assay_panel": {
            "path": "idea_2/breeder_tools/Supplementary_Table_Assay_Design.csv",
            "description": "Assay panel details (same as 2A)",
            "expected_cols": ["Marker", "Trait", "Chromosome", "Position"],
            "panel": "D"
        },
    },
    
    # ========================================================================
    # ADDITIONAL/ALTERNATIVE DATA SOURCES
    # ========================================================================
    "Additional": {
        "gwas_summary": {
            "path": "idea_1/gwas_weights/internal_gwas_summary.csv",
            "description": "GWAS summary statistics for all traits",
            "expected_cols": ["variant_id", "chrom", "pos", "p_pc", "beta_pc"],
            "panel": "Various"
        },
        "gwas_by_trait": {
            "path": "idea_1/gwas/gwas_summary_by_trait.csv",
            "description": "Combined GWAS by trait (wide format)",
            "expected_cols": ["snp_id", "p_FBC", "beta_FBC", "p_AFW"],
            "panel": "Various"
        },
        "binn_gene_scores_long": {
            "path": "idea_3/binn_explain/binn_gene_scores_long.csv",
            "description": "BINN scores in long format",
            "expected_cols": ["gene_id", "trait", "score"],
            "panel": "4A alt"
        },
        "model_performance_long": {
            "path": "idea_2/summary/idea2_model_performance_long.csv",
            "description": "Full model performance table",
            "expected_cols": ["trait", "cv_scheme", "model", "mean_r"],
            "panel": "2D alt"
        },
        "candidate_genes_summary": {
            "path": "idea_2/annotation/idea2_candidate_genes_summary.csv",
            "description": "Candidate gene annotations",
            "expected_cols": ["gene_id", "gene_name", "trait"],
            "panel": "4A annotation"
        },
        "binn_gene_table": {
            "path": "idea_3/binn_maps/binn_gene_table.csv",
            "description": "BINN gene mapping table",
            "expected_cols": ["gene_id", "n_snps", "chromosome"],
            "panel": "4A annotation"
        },
        "shap_top_snps": {
            "path": "idea_3/interpretation/shap/SHAP_TopSNPs_FBC.csv",
            "description": "Top SHAP SNPs for FBC",
            "expected_cols": ["snp_id", "mean_abs_shap"],
            "panel": "3B annotation"
        },
    }
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def log(msg, level="INFO"):
    """Print log message with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = {
        "INFO": "[INFO]",
        "DEBUG": "[DEBUG]",
        "SUCCESS": "[OK]",
        "WARNING": "[WARN]",
        "ERROR": "[ERROR]",
        "SECTION": "\n" + "=" * 70 + "\n"
    }.get(level, "[INFO]")
    
    if level == "DEBUG" and not DEBUG:
        return
    
    if level == "SECTION":
        print(f"{prefix}{msg}\n" + "=" * 70)
    else:
        print(f"{timestamp} {prefix} {msg}")


def get_dtype_summary(series):
    """Get human-readable dtype summary"""
    dtype = str(series.dtype)
    if 'int' in dtype:
        return 'integer'
    elif 'float' in dtype:
        return 'float'
    elif 'object' in dtype:
        # Check if it's really string
        sample = series.dropna().head(5)
        if len(sample) > 0 and isinstance(sample.iloc[0], str):
            return 'string'
        return 'mixed/object'
    elif 'bool' in dtype:
        return 'boolean'
    elif 'datetime' in dtype:
        return 'datetime'
    else:
        return dtype


def get_value_summary(series, max_unique=10):
    """Get summary of values in a column"""
    non_null = series.dropna()
    n_total = len(series)
    n_null = series.isna().sum()
    n_unique = series.nunique()
    
    summary = {
        'n_total': n_total,
        'n_null': n_null,
        'n_unique': n_unique,
        'pct_null': round(100 * n_null / n_total, 1) if n_total > 0 else 0
    }
    
    # Numeric summary
    if pd.api.types.is_numeric_dtype(series):
        if len(non_null) > 0:
            summary['min'] = non_null.min()
            summary['max'] = non_null.max()
            summary['mean'] = round(non_null.mean(), 4)
            summary['std'] = round(non_null.std(), 4)
            summary['sample_values'] = list(non_null.head(3).values)
    else:
        # Categorical/string summary
        if n_unique <= max_unique:
            summary['unique_values'] = list(series.dropna().unique())
        else:
            summary['sample_values'] = list(non_null.head(5).values)
    
    return summary


def analyze_file(filepath, expected_cols):
    """Analyze a single data file and return detailed info"""
    result = {
        'exists': False,
        'path': filepath,
        'error': None,
        'n_rows': None,
        'n_cols': None,
        'columns': [],
        'column_details': {},
        'expected_cols_found': [],
        'expected_cols_missing': [],
        'file_size_kb': None
    }
    
    if not os.path.exists(filepath):
        result['error'] = "FILE NOT FOUND"
        return result
    
    result['exists'] = True
    result['file_size_kb'] = round(os.path.getsize(filepath) / 1024, 1)
    
    try:
        # Try reading as CSV
        df = pd.read_csv(filepath)
        result['n_rows'] = len(df)
        result['n_cols'] = len(df.columns)
        result['columns'] = list(df.columns)
        
        # Check expected columns
        for col in expected_cols:
            # Flexible matching (case-insensitive, partial match)
            found = False
            for actual_col in df.columns:
                if col.lower() in actual_col.lower() or actual_col.lower() in col.lower():
                    result['expected_cols_found'].append(f"{col} → {actual_col}")
                    found = True
                    break
            if not found:
                result['expected_cols_missing'].append(col)
        
        # Detailed column analysis
        for col in df.columns:
            result['column_details'][col] = {
                'dtype': get_dtype_summary(df[col]),
                'summary': get_value_summary(df[col])
            }
        
    except Exception as e:
        result['error'] = f"READ ERROR: {str(e)}"
    
    return result


def format_column_report(col_name, col_info):
    """Format column details for report"""
    dtype = col_info['dtype']
    summary = col_info['summary']
    
    lines = [f"    - **{col_name}** ({dtype})"]
    
    if 'min' in summary:
        lines.append(f"      Range: [{summary['min']:.4g}, {summary['max']:.4g}], Mean: {summary['mean']:.4g} ± {summary['std']:.4g}")
    
    if 'unique_values' in summary:
        vals = summary['unique_values'][:10]
        lines.append(f"      Values: {vals}")
    elif 'sample_values' in summary:
        vals = summary['sample_values'][:5]
        lines.append(f"      Sample: {vals}")
    
    if summary['pct_null'] > 0:
        lines.append(f"      Missing: {summary['n_null']}/{summary['n_total']} ({summary['pct_null']}%)")
    
    return "\n".join(lines)


# ============================================================================
# MAIN INVENTORY FUNCTION
# ============================================================================

def generate_inventory():
    """Generate comprehensive data inventory"""
    
    log("Figure Data Inventory Script", "SECTION")
    log(f"Project Root: {PROJECT_ROOT}")
    log(f"Output Directory: {FIGURES_DIR}")
    
    # Check project root exists
    if not os.path.exists(PROJECT_ROOT):
        log(f"PROJECT ROOT NOT FOUND: {PROJECT_ROOT}", "ERROR")
        log("Please update PROJECT_ROOT in the script", "ERROR")
        return
    
    # Initialize report
    report_md = []
    report_txt = []
    
    # Header
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_md.append(f"# Figure Data Inventory Report")
    report_md.append(f"\n**Generated:** {timestamp}")
    report_md.append(f"\n**Project Root:** `{PROJECT_ROOT}`\n")
    
    report_txt.append("=" * 80)
    report_txt.append("FIGURE DATA INVENTORY REPORT")
    report_txt.append(f"Generated: {timestamp}")
    report_txt.append(f"Project Root: {PROJECT_ROOT}")
    report_txt.append("=" * 80 + "\n")
    
    # Summary counters
    total_files = 0
    found_files = 0
    missing_files = 0
    files_with_issues = 0
    
    # Process each figure
    for figure_name, files_dict in REQUIRED_FILES.items():
        log(f"Processing {figure_name}...", "SECTION")
        
        report_md.append(f"\n## {figure_name}\n")
        report_txt.append(f"\n{'=' * 60}")
        report_txt.append(f"{figure_name}")
        report_txt.append("=" * 60 + "\n")
        
        for file_key, file_info in files_dict.items():
            total_files += 1
            
            rel_path = file_info['path']
            full_path = os.path.join(OUTPUT_DIR, rel_path)
            expected_cols = file_info.get('expected_cols', [])
            description = file_info.get('description', '')
            panel = file_info.get('panel', 'N/A')
            
            log(f"Checking: {rel_path}", "DEBUG")
            
            # Analyze file
            analysis = analyze_file(full_path, expected_cols)
            
            # Status
            if analysis['exists']:
                found_files += 1
                if analysis['expected_cols_missing']:
                    status = "[WARN] PARTIAL"
                    files_with_issues += 1
                else:
                    status = "[OK]"
            else:
                missing_files += 1
                status = "[MISSING]"
            
            # Add to reports
            report_md.append(f"### {file_key}")
            report_md.append(f"- **Status:** {status}")
            report_md.append(f"- **Panel(s):** {panel}")
            report_md.append(f"- **Description:** {description}")
            report_md.append(f"- **Path:** `output/{rel_path}`")
            
            report_txt.append(f"\n[{file_key}]")
            report_txt.append(f"  Status: {status}")
            report_txt.append(f"  Panel: {panel}")
            report_txt.append(f"  Path: output/{rel_path}")
            
            if analysis['exists']:
                report_md.append(f"- **Size:** {analysis['file_size_kb']} KB")
                report_md.append(f"- **Dimensions:** {analysis['n_rows']} rows × {analysis['n_cols']} columns")
                report_md.append(f"- **Columns:** `{', '.join(analysis['columns'][:10])}`" + 
                               (" ..." if len(analysis['columns']) > 10 else ""))
                
                report_txt.append(f"  Size: {analysis['file_size_kb']} KB")
                report_txt.append(f"  Dimensions: {analysis['n_rows']} rows × {analysis['n_cols']} columns")
                report_txt.append(f"  Columns: {', '.join(analysis['columns'][:10])}" +
                                (" ..." if len(analysis['columns']) > 10 else ""))
                
                # Expected columns check
                if analysis['expected_cols_found']:
                    report_md.append(f"- **Expected columns found:** {', '.join(analysis['expected_cols_found'])}")
                if analysis['expected_cols_missing']:
                    report_md.append(f"- **[WARN] Expected columns MISSING:** {', '.join(analysis['expected_cols_missing'])}")
                    report_txt.append(f"  WARNING - Missing expected columns: {', '.join(analysis['expected_cols_missing'])}")
                
                # Column details
                report_md.append("\n**Column Details:**\n")
                for col_name, col_info in list(analysis['column_details'].items())[:15]:  # Limit to 15 cols
                    report_md.append(format_column_report(col_name, col_info))
                
                if len(analysis['column_details']) > 15:
                    report_md.append(f"\n    ... and {len(analysis['column_details']) - 15} more columns")
                
            else:
                report_md.append(f"- **Error:** {analysis['error']}")
                report_txt.append(f"  ERROR: {analysis['error']}")
                
                # Try to find similar files
                parent_dir = os.path.dirname(full_path)
                if os.path.exists(parent_dir):
                    similar = [f for f in os.listdir(parent_dir) if f.endswith('.csv')][:5]
                    if similar:
                        report_md.append(f"- **Similar files in directory:** `{', '.join(similar)}`")
                        report_txt.append(f"  Similar files found: {', '.join(similar)}")
                else:
                    report_md.append(f"- **Note:** Parent directory does not exist: `{parent_dir}`")
                    report_txt.append(f"  Parent directory missing: {parent_dir}")
            
            report_md.append("")  # Blank line
            
            # Console output
            log(f"{status} {file_key}: {rel_path}", 
                "SUCCESS" if analysis['exists'] and not analysis['expected_cols_missing'] else 
                "WARNING" if analysis['exists'] else "ERROR")
    
    # ========================================================================
    # SUMMARY SECTION
    # ========================================================================
    
    log("Generating Summary...", "SECTION")
    
    summary_md = [
        "\n---\n",
        "## Summary\n",
        f"| Metric | Count |",
        f"|--------|-------|",
        f"| Total files required | {total_files} |",
        f"| Files found | {found_files} |",
        f"| Files missing | {missing_files} |",
        f"| Files with column issues | {files_with_issues} |",
        f"| **Success rate** | **{round(100*found_files/total_files, 1)}%** |",
    ]
    
    summary_txt = [
        "\n" + "=" * 80,
        "SUMMARY",
        "=" * 80,
        f"Total files required:    {total_files}",
        f"Files found:             {found_files}",
        f"Files missing:           {missing_files}",
        f"Files with issues:       {files_with_issues}",
        f"Success rate:            {round(100*found_files/total_files, 1)}%",
    ]
    
    # Add critical missing files
    if missing_files > 0:
        summary_md.append("\n### [CRITICAL] Missing Files\n")
        summary_txt.append("\nCRITICAL - Missing Files:")
        
        for figure_name, files_dict in REQUIRED_FILES.items():
            for file_key, file_info in files_dict.items():
                full_path = os.path.join(OUTPUT_DIR, file_info['path'])
                if not os.path.exists(full_path):
                    summary_md.append(f"- `{file_info['path']}` ({file_key})")
                    summary_txt.append(f"  - {file_info['path']}")
    
    # Recommendations
    summary_md.append("\n### Recommendations\n")
    summary_txt.append("\nRECOMMENDATIONS:")
    
    if missing_files == 0 and files_with_issues == 0:
        summary_md.append("[OK] All required data files are present and have expected columns. Ready for figure generation!")
        summary_txt.append("  All files present and valid. Ready for figure generation!")
    else:
        if missing_files > 0:
            summary_md.append(f"1. Run missing analysis scripts to generate {missing_files} missing file(s)")
            summary_txt.append(f"  1. Run missing analysis scripts ({missing_files} files needed)")
        if files_with_issues > 0:
            summary_md.append(f"2. Check column names in {files_with_issues} file(s) with partial matches")
            summary_txt.append(f"  2. Verify column names in {files_with_issues} files")
    
    report_md.extend(summary_md)
    report_txt.extend(summary_txt)
    
    # ========================================================================
    # WRITE REPORTS
    # ========================================================================
    
    # Write Markdown report
    try:
        with open(INVENTORY_MD, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_md))
        log(f"Markdown report saved: {INVENTORY_MD}", "SUCCESS")
    except Exception as e:
        log(f"Failed to write Markdown report: {e}", "ERROR")
    
    # Write Text report
    try:
        with open(INVENTORY_TXT, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_txt))
        log(f"Text report saved: {INVENTORY_TXT}", "SUCCESS")
    except Exception as e:
        log(f"Failed to write Text report: {e}", "ERROR")
    
    # Final console summary
    log("Inventory Complete", "SECTION")
    print(f"\n  Total files:    {total_files}")
    print(f"  Found:          {found_files}")
    print(f"  Missing:        {missing_files}")
    print(f"  With issues:    {files_with_issues}")
    print(f"\n  Success rate:   {round(100*found_files/total_files, 1)}%")
    print(f"\n  Reports saved to:")
    print(f"    - {INVENTORY_MD}")
    print(f"    - {INVENTORY_TXT}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("MANGO GWAS - FIGURE DATA INVENTORY SCRIPT")
    print("=" * 80 + "\n")
    
    # Check if running on correct system
    if not os.path.exists(PROJECT_ROOT):
        print(f"[ERROR] Project root not found: {PROJECT_ROOT}")
        print("[INFO] This script is designed for Windows with project at C:\\Users\\ms\\Desktop\\mango")
        print("[INFO] If running elsewhere, update PROJECT_ROOT at the top of the script")
        sys.exit(1)
    
    generate_inventory()
    
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80 + "\n")