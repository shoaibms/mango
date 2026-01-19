"""
Configuration module for Mango GS – Idea 1.

This centralises paths and analysis-wide constants so that the
idea_1 scripts stay clean and reproducible.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

# =========================
# Base directories
# =========================

# Root of the mango project (adjust this path to your local setup)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Raw / intermediate data
DATA_ROOT = PROJECT_ROOT / "data"
MAIN_DATA_DIR = DATA_ROOT / "main_data"

# Code and output roots for Idea 1
IDEA1_CODE_ROOT = PROJECT_ROOT / "code" / "idea_1"
IDEA1_OUTPUT_ROOT = PROJECT_ROOT / "output" / "idea_1"


# =========================
# Raw data paths
# =========================

# VCFs with dense WGS SNPs (225 samples across both files)
VCF_PATHS: List[Path] = [
    MAIN_DATA_DIR / "11_QF1.vcf.gz",
    MAIN_DATA_DIR / "12_QF2.vcf.gz",
]

# Reference genome and gene annotation (from data audit scripts)
FASTA_PATH: Path = DATA_ROOT / "GWHABLA00000000.genome.fasta.gz"
GFF_PATH: Path = DATA_ROOT / "GWHABLA00000000.gff.gz"

# Phenotype Excel (New Phytologist supplementary Dataset S1–S3)
PHENO_XLSX_PATH: Path = MAIN_DATA_DIR / "nph20252-sup-0001-datasetss1-s3.xlsx"

# Sheet names discovered in dataset_s1_header_check.txt
PHENO_SHEETS: Dict[str, str] = {
    "si": "SI",
    "dataset_s1": "Dataset S1",
    "dataset_s2": "Dataset S2",
    "dataset_s3": "Dataset S3",
}

# Column in Dataset S1 that matches VCF sample IDs
PHENO_SAMPLE_ID_COL: str = "ID"

# Backwards-compatibility aliases expected by v7/v8 scripts
PHENO_SHEET: str = PHENO_SHEETS["dataset_s1"]  # default sheet for Idea 1
SAMPLE_ID_COL: str = PHENO_SAMPLE_ID_COL       # used by load_pheno_and_meta

# Optional: expected number of samples (used only for sanity checks)
EXPECTED_N_SAMPLES: int = 225


# =========================
# Trait configuration
# =========================

# Mapping from human-friendly trait keys to their column names in Dataset S1
# (based on pilot_gs_mango.py and the paper's notation).
TRAIT_CONFIG: Dict[str, Dict[str, str]] = {
    # Fruit blush colour (hue angle)
    "FBC": {
        "column": "BC",
        "sheet": PHENO_SHEETS["dataset_s1"],
        "description": "Fruit blush colour (hue angle)",
    },
    # Fruit firmness
    "FF": {
        "column": "FF",
        "sheet": PHENO_SHEETS["dataset_s1"],
        "description": "Fruit firmness",
    },
    # Average fruit weight (square-root transformed)
    "AFW": {
        "column": "Square Root[FW]",
        "sheet": PHENO_SHEETS["dataset_s1"],
        "description": "Square-root–transformed fruit weight",
    },
    # Total soluble solids (log10-transformed)
    "TSS": {
        "column": "Log10[TSS]",
        "sheet": PHENO_SHEETS["dataset_s1"],
        "description": "Log10-transformed total soluble solids",
    },
    # Trunk circumference
    "TC": {
        "column": "TC",
        "sheet": PHENO_SHEETS["dataset_s1"],
        "description": "Trunk circumference",
    },
}

# Simple mapping used by 01_build_core_matrices_v7/v8.load_pheno_and_meta
# { "FBC": "BC", "FF": "FF", ... }
TRAIT_COL_MAP: Dict[str, str] = {
    trait_key: cfg["column"] for trait_key, cfg in TRAIT_CONFIG.items()
}

# Traits that Idea 1 will model by default (order is meaningful for summaries)
TRAITS_DEFAULT: List[str] = ["FBC", "FF", "AFW", "TSS", "TC"]


# =========================
# Structure / ancestry configuration
# =========================

# Dataset S1 carries multiple "miinvXX.X" columns that encode inversion /
# ancestry information. Rather than hard-coding all names, downstream scripts
# can select any columns that start with these prefixes.
STRUCTURE_COLUMN_PREFIXES: List[str] = ["miinv"]

# Name of the column that holds the main genotype name (useful for joins)
PHENO_GENOTYPE_NAME_COL: str = "Genotype Name"


# =========================
# SNP sampling and QC
# =========================

# Total number of SNPs to sample across both VCFs for the core matrices.
# This is a *target*; the exact number may differ slightly depending on QC.
TOTAL_SNPS_TARGET: int = 20000  # 20k SNPs total (as in pilot_gs_mango.py)

# SNP-level QC thresholds
MIN_MAF: float = 0.05   # minor allele frequency threshold
MAX_MISS: float = 0.10  # max fraction missing per SNP

# Random seed for any stochastic steps (sampling SNPs, CV splits, etc.)
RANDOM_STATE: int = 42


# =========================
# Cross-validation configuration
# =========================

# K-fold CV settings for baseline models
N_SPLITS: int = 5

# Number of PCs to use when performing PC-correction (if enabled)
N_PCS: int = 6


# =========================
# GWAS configuration
# =========================

# Thresholds for defining "major QTL" SNPs in internal GWAS (Option B)
GWAS_P_THRESHOLD_MAJOR: float = 5e-8
GWAS_N_TOP_MAJOR: int = 200

# External GWAS summary config (Option A; 04_gwas_to_snp_weights.py).
# Keep empty by default – safe, and the script will just skip traits if unused.
GWAS_SUMMARY_CONFIG: Dict[str, Dict[str, object]] = {}


# =========================
# Output subdirectories
# =========================

# These are the standard subfolders under IDEA1_OUTPUT_ROOT that all scripts
# should use, so paths remain consistent across the project.

CORE_DATA_DIR: Path = IDEA1_OUTPUT_ROOT / "core_data"
CV_BASELINE_DIR: Path = IDEA1_OUTPUT_ROOT / "cv_baseline"
CV_STRUCTURE_DIR: Path = IDEA1_OUTPUT_ROOT / "cv_structure"
GWAS_WEIGHTS_DIR: Path = IDEA1_OUTPUT_ROOT / "gwas_weights"
CV_GWAS_INTEGRATION_DIR: Path = IDEA1_OUTPUT_ROOT / "cv_gwas_integration"
LOW_DENSITY_PANEL_DIR: Path = IDEA1_OUTPUT_ROOT / "low_density_panel"
SUMMARY_DIR: Path = IDEA1_OUTPUT_ROOT / "summary"


def ensure_output_dirs() -> None:
    """
    Create the standard output directories for Idea 1 if they do not exist.

    Call this near the top of each script that writes to these folders.
    """
    for path in [
        IDEA1_OUTPUT_ROOT,
        CORE_DATA_DIR,
        CV_BASELINE_DIR,
        CV_STRUCTURE_DIR,
        GWAS_WEIGHTS_DIR,
        CV_GWAS_INTEGRATION_DIR,
        LOW_DENSITY_PANEL_DIR,
        SUMMARY_DIR,
    ]:
        os.makedirs(path, exist_ok=True)


if __name__ == "__main__":
    # Convenience: running this module directly will create the output folders
    ensure_output_dirs()
    print(f"Initialised Idea 1 output folders under: {IDEA1_OUTPUT_ROOT}")
