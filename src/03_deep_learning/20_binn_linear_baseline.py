#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
20_binn_linear_baseline.py

Mango GS — Idea 3: Linear Baseline on BINN SNP Subset

PURPOSE:
    Test whether BINN's accuracy gains come from:
    (A) Feature selection (using 490 candidate gene SNPs vs 20k noisy SNPs), OR
    (B) Non-linear interactions captured by the neural network (epistasis)

TEST LOGIC:
    - Run Ridge regression using ONLY the 490 SNPs that BINN uses
    - Compare with BINN performance
    
INTERPRETATION:
    - If Ridge(490 SNPs) ≈ BINN → Gains from feature selection, NOT epistasis
    - If Ridge(490 SNPs) << BINN → Gains require non-linearity → possible epistasis

"""

from __future__ import annotations

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# =============================================================================
# CONFIG — EDIT PATHS HERE IF NEEDED
# =============================================================================
ROOT_DIR = r"C:\Users\ms\Desktop\mango"

# Input files
GENO_CORE_PATH = os.path.join(ROOT_DIR, "output", "idea_1", "core_data", "geno_core.npz")
PHENO_CORE_PATH = os.path.join(ROOT_DIR, "output", "idea_1", "core_data", "pheno_core.csv")
BINN_SNP_MAP_PATH = os.path.join(ROOT_DIR, "output", "idea_3", "binn_maps", "binn_snp_map.npz")

# Optional: CV folds from BINN training (for exact replication)
CV_FOLDS_PATH = os.path.join(ROOT_DIR, "output", "idea_3", "tensors", "cv_folds.json")

# Output directory
OUT_DIR = os.path.join(ROOT_DIR, "output", "idea_3", "binn_decomposition")

# Traits to evaluate (canonical order matching BINN)
TRAITS = ["FBC", "FF", "AFW", "TSS", "TC"]

# CV settings (used if cv_folds.json not found)
N_FOLDS = 5
RANDOM_STATE = 42

# Ridge regularisation
RIDGE_ALPHA = 1.0

# Debug mode
DEBUG = True

# =============================================================================
# UTILITIES
# =============================================================================

def safe_mkdir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def debug_print(msg: str) -> None:
    """Print debug messages if DEBUG is enabled."""
    if DEBUG:
        print(f"[DEBUG] {msg}")


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Pearson correlation, handling edge cases."""
    if len(y_true) < 3:
        return np.nan
    if np.std(y_pred) < 1e-9 or np.std(y_true) < 1e-9:
        return np.nan
    return np.corrcoef(y_true, y_pred)[0, 1]


# =============================================================================
# DATA LOADING WITH EXTENSIVE VALIDATION
# =============================================================================

def load_geno_core(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load genotype core matrix with validation.
    
    Returns:
        G: genotype matrix (n_samples x n_snps)
        sample_ids: array of sample identifiers
        variant_ids: array of SNP identifiers
    """
    print(f"\n[LOAD] Loading genotype data from:\n  {path}")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Genotype file not found: {path}")
    
    geno = np.load(path, allow_pickle=True)
    keys = list(geno.files)
    print(f"[INFO] geno_core keys: {keys}")
    
    # Try different key names for the genotype matrix
    if 'G' in keys:
        G = geno['G']
    elif 'X' in keys:
        G = geno['X']
    else:
        raise KeyError(f"Cannot find genotype matrix. Available keys: {keys}")
    
    # Get sample IDs
    if 'sample_ids' in keys:
        sample_ids = geno['sample_ids']
    elif 'samples' in keys:
        sample_ids = geno['samples']
    else:
        print("[WARN] No sample_ids found, using indices")
        sample_ids = np.arange(G.shape[0])
    
    # Get variant IDs
    if 'variant_ids' in keys:
        variant_ids = geno['variant_ids']
    elif 'snp_ids' in keys:
        variant_ids = geno['snp_ids']
    else:
        print("[WARN] No variant_ids found, using indices")
        variant_ids = np.arange(G.shape[1])
    
    print(f"[INFO] Genotype matrix shape: {G.shape} (samples x SNPs)")
    debug_print(f"Sample IDs (first 5): {sample_ids[:5]}")
    debug_print(f"Variant IDs (first 5): {variant_ids[:5]}")
    debug_print(f"G dtype: {G.dtype}, range: [{np.nanmin(G):.2f}, {np.nanmax(G):.2f}]")
    debug_print(f"Missing values: {np.isnan(G).sum()} ({100*np.isnan(G).mean():.2f}%)")
    
    return G, np.array(sample_ids), np.array(variant_ids)


def load_pheno_core(path: str) -> pd.DataFrame:
    """
    Load phenotype data with validation.
    
    Returns:
        DataFrame with sample IDs as index and traits as columns
    """
    print(f"\n[LOAD] Loading phenotype data from:\n  {path}")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Phenotype file not found: {path}")
    
    # Try to detect index column
    pheno_raw = pd.read_csv(path, nrows=5)
    print(f"[INFO] Phenotype columns: {list(pheno_raw.columns)}")
    
    # Check if first column looks like an index
    first_col = pheno_raw.columns[0]
    if first_col in ['ID', 'id', 'Sample', 'sample', 'Unnamed: 0', 'sample_id', 'accession']:
        pheno = pd.read_csv(path, index_col=0)
        print(f"[INFO] Using '{first_col}' as index column")
    else:
        pheno = pd.read_csv(path)
        print(f"[INFO] No index column detected, using row numbers")
    
    print(f"[INFO] Phenotype table shape: {pheno.shape} (samples x traits)")
    debug_print(f"Index (first 5): {list(pheno.index[:5])}")
    debug_print(f"Columns: {list(pheno.columns)}")
    
    # Check for trait columns
    available_traits = [t for t in TRAITS if t in pheno.columns]
    missing_traits = [t for t in TRAITS if t not in pheno.columns]
    
    if missing_traits:
        print(f"[WARN] Missing trait columns: {missing_traits}")
    print(f"[INFO] Available traits: {available_traits}")
    
    # Summary statistics per trait
    for trait in available_traits:
        n_valid = pheno[trait].notna().sum()
        n_total = len(pheno)
        print(f"  - {trait}: {n_valid}/{n_total} non-missing ({100*n_valid/n_total:.1f}%)")
    
    return pheno


def load_binn_snp_map(path: str) -> Dict[str, np.ndarray]:
    """
    Load BINN SNP mapping with validation.
    
    Returns:
        Dictionary with SNP indices and gene mappings
    """
    print(f"\n[LOAD] Loading BINN SNP map from:\n  {path}")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"BINN SNP map not found: {path}")
    
    binn_map = np.load(path, allow_pickle=True)
    keys = list(binn_map.files)
    print(f"[INFO] binn_snp_map keys: {keys}")
    
    # Required key
    if 'snp_core_index' not in keys:
        raise KeyError(f"'snp_core_index' not found in BINN map. Available: {keys}")
    
    snp_indices = binn_map['snp_core_index']
    print(f"[INFO] BINN uses {len(snp_indices)} SNPs")
    debug_print(f"SNP index range: [{snp_indices.min()}, {snp_indices.max()}]")
    debug_print(f"First 10 indices: {snp_indices[:10]}")
    
    # Load additional info if available
    result = {'snp_core_index': snp_indices}
    
    if 'snp_ids' in keys:
        result['snp_ids'] = binn_map['snp_ids']
        debug_print(f"BINN SNP IDs (first 5): {result['snp_ids'][:5]}")
    
    if 'gene_ids' in keys:
        result['gene_ids'] = binn_map['gene_ids']
        print(f"[INFO] BINN maps to {len(result['gene_ids'])} genes")
    
    if 'snp_gene_index' in keys:
        result['snp_gene_index'] = binn_map['snp_gene_index']
    
    return result


def load_cv_folds(path: str, n_samples: int) -> List[Dict]:
    """
    Load CV folds from JSON, or create new ones if not found.
    
    Returns:
        List of fold dictionaries with 'train_idx' and 'test_idx'
    """
    if os.path.exists(path):
        print(f"\n[LOAD] Loading CV folds from:\n  {path}")
        with open(path, 'r') as f:
            folds = json.load(f)
        print(f"[INFO] Loaded {len(folds)} folds from file")
        
        # Validate fold structure
        for i, fold in enumerate(folds):
            n_train = len(fold['train_idx'])
            n_test = len(fold['test_idx'])
            debug_print(f"Fold {i}: train={n_train}, test={n_test}")
        
        return folds
    else:
        print(f"\n[WARN] CV folds file not found: {path}")
        print(f"[INFO] Creating new {N_FOLDS}-fold CV with random_state={RANDOM_STATE}")
        
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        folds = []
        for fold_id, (train_idx, test_idx) in enumerate(kf.split(np.arange(n_samples))):
            folds.append({
                'fold_id': fold_id,
                'train_idx': train_idx.tolist(),
                'test_idx': test_idx.tolist()
            })
            debug_print(f"Fold {fold_id}: train={len(train_idx)}, test={len(test_idx)}")
        
        return folds


def load_cv_design_csv_as_folds(cv_path: str, sample_ids: List[str]) -> Tuple[List[Dict], str]:
    df = pd.read_csv(cv_path)
    if not {"sample_id", "fold"}.issubset(df.columns):
        raise ValueError(f"cv_design must have columns sample_id, fold. Found: {list(df.columns)}")
    scheme_name = os.path.splitext(os.path.basename(cv_path))[0]
    fold_map = df.set_index("sample_id")["fold"]

    missing = [sid for sid in sample_ids if sid not in fold_map.index]
    if missing:
        raise ValueError(f"{len(missing)} sample_ids not found in cv_design. Example: {missing[:5]}")

    fold_ids = np.array([int(fold_map.loc[sid]) for sid in sample_ids], dtype=int)

    folds = []
    for f in sorted(set(fold_ids.tolist())):
        test_idx = np.where(fold_ids == f)[0]
        train_idx = np.where(fold_ids != f)[0]
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        folds.append({
            "fold_id": int(f),
            "train_idx": train_idx.tolist(),
            "test_idx": test_idx.tolist()
        })
    return folds, scheme_name


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_ridge_cv(
    X: np.ndarray,
    y: np.ndarray,
    folds: List[Dict],
    alpha: float = RIDGE_ALPHA,
    trait_name: str = "unknown",
    sample_ids: Optional[List[str]] = None,
    oof_rows: Optional[List[Dict]] = None,
    source: str = "ridge490",
    feature_set: str = "BINN_490_SNPs",
    scheme: str = "",
    scenario: str = "ridge490",
) -> Dict:
    """
    Run Ridge regression with cross-validation.
    
    Returns:
        Dictionary with per-fold and summary results
    """
    fold_results = []
    
    for fold in folds:
        fold_id = fold.get('fold_id', len(fold_results))
        train_idx = np.array(fold['train_idx'])
        test_idx = np.array(fold['test_idx'])
        
        # Get data for this fold
        X_train_full, X_test_full = X[train_idx], X[test_idx]
        y_train_full, y_test_full = y[train_idx], y[test_idx]
        
        # Remove samples with missing phenotypes
        train_mask = ~np.isnan(y_train_full)
        test_mask = ~np.isnan(y_test_full)
        
        X_train = X_train_full[train_mask]
        y_train = y_train_full[train_mask]
        X_test = X_test_full[test_mask]
        y_test = y_test_full[test_mask]
        
        if len(y_train) < 10 or len(y_test) < 5:
            debug_print(f"Fold {fold_id}: Skipping (train={len(y_train)}, test={len(y_test)})")
            continue
        
        # Standardise features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Handle any remaining NaN in genotypes (impute to 0 after scaling)
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0)
        
        # Fit Ridge
        model = Ridge(alpha=alpha)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        if oof_rows is not None and sample_ids is not None and len(y_test) > 0:
            ids_test_full = np.array(sample_ids)[test_idx]
            ids_test = ids_test_full[test_mask]
            for sid, yt, yp in zip(ids_test, y_test, y_pred):
                oof_rows.append({
                    "source": source,
                    "feature_set": feature_set,
                    "model": "ridge",
                    "scheme": scheme,
                    "scenario": scenario,
                    "trait": trait_name,
                    "fold": int(fold_id),
                    "sample_id": str(sid),
                    "y_true": float(yt),
                    "y_pred": float(yp),
                    "y_true_resid": np.nan,
                    "y_pred_resid": np.nan,
                    "y_fixed_pred": np.nan,
                })

        # Compute metrics
        r = pearson_r(y_test, y_pred)
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        mae = np.mean(np.abs(y_test - y_pred))
        
        fold_results.append({
            'fold_id': fold_id,
            'n_train': len(y_train),
            'n_test': len(y_test),
            'pearson_r': r,
            'rmse': rmse,
            'mae': mae
        })
        
        debug_print(f"Fold {fold_id}: r={r:.3f}, n_train={len(y_train)}, n_test={len(y_test)}")
    
    # Compute summary statistics
    if len(fold_results) == 0:
        return {'error': 'No valid folds'}
    
    rs = [f['pearson_r'] for f in fold_results if not np.isnan(f['pearson_r'])]
    
    summary = {
        'trait': trait_name,
        'n_folds': len(rs),
        'mean_r': np.mean(rs) if rs else np.nan,
        'sd_r': np.std(rs, ddof=1) if len(rs) > 1 else np.nan,
        'min_r': np.min(rs) if rs else np.nan,
        'max_r': np.max(rs) if rs else np.nan,
        'fold_results': fold_results
    }
    
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv_design", type=str, default="", help="Optional Idea2 cv_design CSV (sample_id, fold).")
    args = parser.parse_args()

    print("=" * 72)
    print("Mango GS — Idea 3: BINN Linear Baseline (21_binn_linear_baseline.py)")
    print("=" * 72)
    print("\nPURPOSE: Test if BINN's gains come from feature selection vs epistasis")
    print("-" * 72)
    
    # Create output directory
    safe_mkdir(OUT_DIR)
    
    # -------------------------------------------------------------------------
    # 1. Load all data
    # -------------------------------------------------------------------------
    try:
        G, sample_ids_geno, variant_ids = load_geno_core(GENO_CORE_PATH)
        pheno = load_pheno_core(PHENO_CORE_PATH)
        binn_map = load_binn_snp_map(BINN_SNP_MAP_PATH)
    except Exception as e:
        print(f"\n[ERROR] Failed to load data: {e}")
        sys.exit(1)
    
    # -------------------------------------------------------------------------
    # 2. Align genotype and phenotype samples
    # -------------------------------------------------------------------------
    print("\n[ALIGN] Aligning genotype and phenotype samples...")
    
    # Convert to strings for matching
    geno_ids = [str(s) for s in sample_ids_geno]
    pheno_ids = [str(s) for s in pheno.index]
    
    debug_print(f"Geno sample IDs (first 5): {geno_ids[:5]}")
    debug_print(f"Pheno sample IDs (first 5): {pheno_ids[:5]}")
    
    # Check alignment
    if geno_ids == pheno_ids:
        print("[INFO] Sample IDs are already aligned")
        G_aligned = G
        pheno_aligned = pheno
    else:
        # Find common samples
        common_ids = [s for s in geno_ids if s in pheno_ids]
        print(f"[INFO] Found {len(common_ids)} common samples out of {len(geno_ids)} geno / {len(pheno_ids)} pheno")
        
        if len(common_ids) < 100:
            print("[WARN] Very few common samples — checking for ID format mismatch...")
            # Try numeric matching
            try:
                geno_numeric = [s.split('_')[-1] if '_' in s else s for s in geno_ids]
                pheno_numeric = [s.split('_')[-1] if '_' in s else s for s in pheno_ids]
                debug_print(f"Numeric geno (first 5): {geno_numeric[:5]}")
                debug_print(f"Numeric pheno (first 5): {pheno_numeric[:5]}")
            except:
                pass
        
        # Reindex
        geno_idx = [geno_ids.index(s) for s in common_ids]
        G_aligned = G[geno_idx]
        pheno_aligned = pheno.loc[common_ids]
        geno_ids = common_ids
        
        print(f"[INFO] Aligned data: {G_aligned.shape[0]} samples")
    
    n_samples = G_aligned.shape[0]
    
    # -------------------------------------------------------------------------
    # 3. Extract BINN SNP subset
    # -------------------------------------------------------------------------
    print("\n[SUBSET] Extracting BINN SNP subset...")
    
    binn_indices = binn_map['snp_core_index']
    
    # Validate indices
    max_idx = binn_indices.max()
    n_snps_total = G_aligned.shape[1]
    
    if max_idx >= n_snps_total:
        print(f"[ERROR] BINN index {max_idx} exceeds genotype matrix columns ({n_snps_total})")
        sys.exit(1)
    
    X_binn = G_aligned[:, binn_indices]
    print(f"[INFO] Full SNP matrix: {G_aligned.shape[1]} SNPs")
    print(f"[INFO] BINN SNP subset: {X_binn.shape[1]} SNPs")
    print(f"[INFO] Reduction: {100 * (1 - X_binn.shape[1] / G_aligned.shape[1]):.1f}%")
    
    # -------------------------------------------------------------------------
    # 4. Load or create CV folds
    # -------------------------------------------------------------------------
    scheme_name = ""
    if args.cv_design:
        folds, scheme_name = load_cv_design_csv_as_folds(args.cv_design, geno_ids)
        print(f"[INFO] Using external CV design: {args.cv_design} (scheme={scheme_name})")
    else:
        folds = load_cv_folds(CV_FOLDS_PATH, n_samples)
        scheme_name = "kfold_internal"
    
    # -------------------------------------------------------------------------
    # 5. Run Ridge regression for each trait
    # -------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("RUNNING RIDGE REGRESSION ON BINN SNP SUBSET")
    print("=" * 72)
    
    all_results = []
    comparison_data = []
    oof_rows = []
    
    for trait in TRAITS:
        if trait not in pheno_aligned.columns:
            print(f"\n[SKIP] Trait '{trait}' not found in phenotype data")
            continue
        
        print(f"\n[TRAIT] {trait}")
        print("-" * 40)
        
        y = pheno_aligned[trait].values
        n_valid = np.sum(~np.isnan(y))
        print(f"[INFO] Non-missing phenotypes: {n_valid}/{len(y)}")
        
        # Run on BINN subset
        result_binn = run_ridge_cv(
            X_binn, y, folds, trait_name=trait,
            sample_ids=geno_ids, oof_rows=oof_rows,
            source="ridge490", feature_set="BINN_490_SNPs",
            scheme=scheme_name, scenario="ridge490"
        )
        
        # Run on full SNP matrix for comparison
        print(f"\n[INFO] Also running on full {G_aligned.shape[1]} SNPs for comparison...")
        result_full = run_ridge_cv(G_aligned, y, folds, trait_name=trait)
        
        # Store results
        all_results.append({
            'trait': trait,
            'model': 'Ridge_BINN_490_SNPs',
            'n_snps': X_binn.shape[1],
            'n_folds': result_binn['n_folds'],
            'mean_r': result_binn['mean_r'],
            'sd_r': result_binn['sd_r']
        })
        
        all_results.append({
            'trait': trait,
            'model': 'Ridge_Full_20k_SNPs',
            'n_snps': G_aligned.shape[1],
            'n_folds': result_full['n_folds'],
            'mean_r': result_full['mean_r'],
            'sd_r': result_full['sd_r']
        })
        
        # Print comparison
        print(f"\n  Ridge (BINN 490 SNPs):  r = {result_binn['mean_r']:.3f} ± {result_binn['sd_r']:.3f}")
        print(f"  Ridge (Full 20k SNPs): r = {result_full['mean_r']:.3f} ± {result_full['sd_r']:.3f}")
        
        comparison_data.append({
            'trait': trait,
            'Ridge_BINN_490': result_binn['mean_r'],
            'Ridge_Full_20k': result_full['mean_r'],
            'delta_feature_selection': result_binn['mean_r'] - result_full['mean_r']
        })

    if oof_rows:
        oof_df = pd.DataFrame(oof_rows)
        oof_path = os.path.join(OUT_DIR, "ridge490_oof_predictions.csv")
        print(f"[SAVE] Ridge-490 OOF -> {oof_path}")
        oof_df.to_csv(oof_path, index=False)
    
    # -------------------------------------------------------------------------
    # 6. Save results
    # -------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SAVING RESULTS")
    print("=" * 72)
    
    # Save detailed results
    df_results = pd.DataFrame(all_results)
    results_path = os.path.join(OUT_DIR, "binn_linear_baseline_results.csv")
    df_results.to_csv(results_path, index=False)
    print(f"\n[SAVE] Detailed results -> {results_path}")
    
    # Save comparison table
    df_comparison = pd.DataFrame(comparison_data)
    comparison_path = os.path.join(OUT_DIR, "binn_vs_full_comparison.csv")
    df_comparison.to_csv(comparison_path, index=False)
    print(f"[SAVE] Comparison table -> {comparison_path}")
    
    # -------------------------------------------------------------------------
    # 7. Print interpretation guide
    # -------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("INTERPRETATION GUIDE")
    print("=" * 72)
    print("""
Compare these results with BINN CV summary (binn_cv_summary.csv):

┌─────────────────────────────────────────────────────────────────────┐
│ SCENARIO A: Ridge(490 SNPs) ≈ BINN                                  │
│   → BINN's gains come from FEATURE SELECTION                        │
│   → The 490 candidate gene SNPs capture the signal                  │
│   → No evidence for epistasis                                       │
├─────────────────────────────────────────────────────────────────────┤
│ SCENARIO B: Ridge(490 SNPs) << BINN                                 │
│   → BINN's gains require NON-LINEARITY                              │
│   → Gene-gene interactions may be present                           │
│   → Further investigation of epistasis warranted                    │
└─────────────────────────────────────────────────────────────────────┘

For manuscript: Compare the 'Ridge_BINN_490_SNPs' column with your 
existing BINN results to determine if the neural network architecture
provides additional predictive power beyond feature selection.
""")
    
    # Print summary table for easy comparison
    print("\n" + "-" * 72)
    print("SUMMARY TABLE (paste into manuscript)")
    print("-" * 72)
    print(f"\n{'Trait':<6} {'Ridge(490)':<12} {'Ridge(20k)':<12} {'Δ(selection)':<12}")
    print("-" * 42)
    for row in comparison_data:
        print(f"{row['trait']:<6} {row['Ridge_BINN_490']:<12.3f} {row['Ridge_Full_20k']:<12.3f} {row['delta_feature_selection']:+.3f}")
    
    print("\n" + "=" * 72)
    print("[DONE] Analysis complete!")
    print("=" * 72)


if __name__ == "__main__":
    main()