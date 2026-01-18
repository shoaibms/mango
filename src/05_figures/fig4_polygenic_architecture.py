#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 4: Polygenic Backbones and Pleiotropic Hub Genes
========================================================

Theme: Strong structural effects sit atop broadly polygenic backbones 
       with sparse hub genes.

Panels:
    A: Polygenic "iceberg" - share of model weight by top 1%/5% SNPs per trait
    B: Cumulative variance explained as function of top-k SNP fraction
    C: BINN gene importance heatmap for top genes across traits
    D: Pleiotropic hub structure - genes vs traits with hub degree
    E: BINN Model Performance

"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DEBUG = True

# Paths
OUTPUT_DIR = r"C:\Users\ms\Desktop\mango\output"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
FIGURE_SUBDIR = os.path.join(FIGURES_DIR, "figure_4")

# Data paths for Figure 4
DATA_PATHS = {
    # Panel A: Polygenic architecture
    'polygenic_arch': os.path.join(OUTPUT_DIR, "idea_3", "breeder_resources", "Polygenic_Architecture_Summary.csv"),
    
    # Panel B: Cumulative variance
    'polygenic_eval': os.path.join(OUTPUT_DIR, "idea_3", "breeder_resources", "Mango_Polygenic_Evaluation_File.csv"),
    
    # Panel C: BINN gene scores heatmap
    'binn_scores_wide': os.path.join(OUTPUT_DIR, "idea_3", "binn_explain", "binn_gene_scores_wide.csv"),
    
    # Panel D: Pleiotropy scores
    'pleiotropy_scores': os.path.join(OUTPUT_DIR, "idea_3", "binn_explain", "binn_gene_pleiotropy_scores.csv"),
    
    # Supporting: Saliency matrix
    'saliency_matrix': os.path.join(OUTPUT_DIR, "idea_3", "interpretation", "saliency", "saliency_matrix_block-raw.csv"),
    
    # BINN CV summary
    'binn_cv': os.path.join(OUTPUT_DIR, "idea_3", "binn_training", "binn_cv_summary.csv"),
    
    # BINN Decomposition - Ridge(20k) vs Ridge(490) vs BINN comparison
    'binn_decomposition': os.path.join(OUTPUT_DIR, "idea_3", "binn_decomposition", "binn_vs_full_comparison.csv"),
}

# Ensure output directories exist
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(FIGURE_SUBDIR, exist_ok=True)


# ============================================================================
# COLOR CONFIGURATION
# ============================================================================

class Colors:
    """
    Color palette configuration matching figure_config.py
    """
    # Core palette
    limegreen = '#32CD32'
    mediumseagreen = '#3CB371'
    springgreen = '#00FF7F'
    turquoise = '#40E0D0'
    steelblue = '#4682B4'
    royalblue = '#4169E1'
    deepskyblue = '#00BFFF'
    seagreen = '#2E8B57'
    teal = '#008080'
    darkseagreen = '#8FBC8F'
    teal_green = '#00A087'
    
    # Neutrals
    gray = '#808080'
    lightgray = '#D3D3D3'
    darkgray = '#A9A9A9'
    
    # Special
    coral_red = "#94CB64"
    
    # Trait colors (standard order: FBC, AFW, FF, TC, TSS)
    trait_order = ['FBC', 'AFW', 'FF', 'TC', 'TSS']
    
    trait_colors = {
        'FBC': mediumseagreen,   # #3CB371
        'AFW': royalblue,        # #4169E1
        'FF': turquoise,         # #40E0D0
        'TC': steelblue,         # #4682B4
        'TSS': limegreen         # #32CD32
    }
    
    # Distinct trait colors
    trait_colors_distinct = {
        'FBC': springgreen,      # #00FF7F
        'AFW': royalblue,        # #4169E1
        'FF': gray,              # #808080
        'TC': teal_green,        # #00A087
        'TSS': mediumseagreen    # #3CB371
    }
    
    # Gene class colors
    class_colors = {
        'hub': limegreen,         # #32CD32
        'multi_trait': mediumseagreen, # #3CB371
        'single_trait': gray      # #808080
    }
    
    # Heatmap colormaps
    @staticmethod
    def get_green_cmap():
        return LinearSegmentedColormap.from_list(
            'greens', 
            ['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#005a32']
        )

colors = Colors()


# ============================================================================
# LOGGING
# ============================================================================

def log(msg, level="INFO"):
    prefix = {
        "INFO": "[INFO]",
        "DEBUG": "[DEBUG]" if DEBUG else None,
        "WARN": "[WARN]",
        "ERROR": "[ERROR]",
        "OK": "[OK]"
    }
    if prefix.get(level):
        print(f"{prefix[level]} {msg}")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load all data files for Figure 4."""
    log("\n" + "=" * 70)
    log("LOADING DATA FOR FIGURE 4")
    log("=" * 70)
    
    data = {}
    
    # Panel A: Polygenic architecture
    log(f"Loading polygenic architecture: {DATA_PATHS['polygenic_arch']}")
    if os.path.exists(DATA_PATHS['polygenic_arch']):
        data['polygenic_arch'] = pd.read_csv(DATA_PATHS['polygenic_arch'])
        log(f"  Shape: {data['polygenic_arch'].shape}", "OK")
    else:
        log(f"  File not found!", "ERROR")
        data['polygenic_arch'] = None
    
    # Panel B: Polygenic evaluation (per-SNP weights)
    log(f"Loading polygenic evaluation: {DATA_PATHS['polygenic_eval']}")
    if os.path.exists(DATA_PATHS['polygenic_eval']):
        data['polygenic_eval'] = pd.read_csv(DATA_PATHS['polygenic_eval'])
        log(f"  Shape: {data['polygenic_eval'].shape}", "OK")
    else:
        log(f"  File not found!", "ERROR")
        data['polygenic_eval'] = None
    
    # Panel C: BINN gene scores
    log(f"Loading BINN scores (wide): {DATA_PATHS['binn_scores_wide']}")
    if os.path.exists(DATA_PATHS['binn_scores_wide']):
        data['binn_scores_wide'] = pd.read_csv(DATA_PATHS['binn_scores_wide'])
        log(f"  Shape: {data['binn_scores_wide'].shape}", "OK")
    else:
        log(f"  File not found!", "ERROR")
        data['binn_scores_wide'] = None
    
    # Panel D: Pleiotropy scores
    log(f"Loading pleiotropy scores: {DATA_PATHS['pleiotropy_scores']}")
    if os.path.exists(DATA_PATHS['pleiotropy_scores']):
        data['pleiotropy_scores'] = pd.read_csv(DATA_PATHS['pleiotropy_scores'])
        log(f"  Shape: {data['pleiotropy_scores'].shape}", "OK")
    else:
        log(f"  File not found!", "ERROR")
        data['pleiotropy_scores'] = None
    
    # BINN CV summary
    log(f"Loading BINN CV summary: {DATA_PATHS['binn_cv']}")
    if os.path.exists(DATA_PATHS['binn_cv']):
        data['binn_cv'] = pd.read_csv(DATA_PATHS['binn_cv'])
        log(f"  Shape: {data['binn_cv'].shape}", "OK")
    else:
        data['binn_cv'] = None
    
    # BINN Decomposition results
    log(f"Loading BINN decomposition: {DATA_PATHS['binn_decomposition']}")
    if os.path.exists(DATA_PATHS['binn_decomposition']):
        data['binn_decomposition'] = pd.read_csv(DATA_PATHS['binn_decomposition'])
        log(f"  Shape: {data['binn_decomposition'].shape}", "OK")
    else:
        log(f"  File not found - will show BINN only", "WARN")
        data['binn_decomposition'] = None
    
    log("=" * 70)
    log("DATA LOADING COMPLETE", "OK")
    log("=" * 70)
    
    return data


# ============================================================================
# PANEL A: Polygenic Architecture "Iceberg"
# ============================================================================

def create_panel_A(data, ax):
    """
    Panel A: Bar chart showing share of model weight carried by top 1% SNPs.
    Shows that most signal is distributed across many SNPs (polygenic).
    """
    log("\n[PANEL A] Polygenic Architecture")
    
    arch_df = data['polygenic_arch']
    
    if arch_df is None:
        ax.text(0.5, 0.5, 'Polygenic architecture\ndata not available', 
               ha='center', va='center', fontsize=11, transform=ax.transAxes,
               color=colors.coral_red)
        ax.text(-0.15, 1.05, 'A', transform=ax.transAxes,
               fontsize=16, fontweight='bold', va='bottom', ha='left')
        log("  NO DATA - Panel A empty", "ERROR")
        return
    
    # Get trait and weight share columns
    trait_col = 'trait' if 'trait' in arch_df.columns else arch_df.columns[0]
    weight_col = None
    for col in ['top_1pct_weight_share', 'top1_weight_share', 'weight_share']:
        if col in arch_df.columns:
            weight_col = col
            break
    
    if weight_col is None:
        log(f"  Cannot find weight share column", "ERROR")
        ax.text(0.5, 0.5, 'Cannot find\nweight column', 
               ha='center', va='center', fontsize=11, transform=ax.transAxes)
        ax.text(-0.15, 1.05, 'A', transform=ax.transAxes,
               fontsize=16, fontweight='bold', va='bottom', ha='left')
        return
    
    # Filter to known traits and order
    arch_df = arch_df[arch_df[trait_col].isin(colors.trait_order)].copy()
    arch_df['trait_order'] = arch_df[trait_col].apply(lambda x: colors.trait_order.index(x))
    arch_df = arch_df.sort_values('trait_order')
    
    traits = arch_df[trait_col].values
    weights = arch_df[weight_col].values * 100  # Convert to percentage
    
    # Bar colors by trait
    bar_colors = [colors.trait_colors.get(t, colors.gray) for t in traits]
    
    x = np.arange(len(traits))
    bars = ax.bar(x, weights, color=bar_colors, edgecolor='white', linewidth=1.5)
    
    # Add "rest" segment to show iceberg effect
    rest = 100 - weights
    ax.bar(x, rest, bottom=weights, color=colors.lightgray, edgecolor='white', 
           linewidth=1.5, alpha=0.5)
    
    # Add value labels
    for i, (bar, w) in enumerate(zip(bars, weights)):
        ax.text(bar.get_x() + bar.get_width()/2, w/2, f'{w:.1f}%',
               ha='center', va='center', fontsize=10, fontweight='bold',
               color='black')
        ax.text(bar.get_x() + bar.get_width()/2, w + (100-w)/2, 
               f'{100-w:.0f}%', ha='center', va='center', fontsize=9,
               color=colors.darkgray)
    
    # Styling
    ax.set_xticks(x)
    ax.set_xticklabels(traits, fontsize=10, fontweight='bold')
    ax.set_ylabel('Weight Share (%)', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 105)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=colors.mediumseagreen, label='Top 1% SNPs'),
        mpatches.Patch(facecolor=colors.lightgray, alpha=0.5, label='Remaining 99%'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Panel label
    ax.text(-0.15, 1.05, 'A', transform=ax.transAxes,
           fontsize=16, fontweight='bold', va='bottom', ha='left')
    
    log("  Panel A complete", "OK")


# ============================================================================
# PANEL B: Cumulative Variance Explained
# ============================================================================

def create_panel_B(data, ax):
    """
    Panel B: Line plots showing cumulative variance explained as function
    of top-k SNP fraction for each trait.
    """
    log("\n[PANEL B] Cumulative Variance Explained")
    
    eval_df = data['polygenic_eval']
    
    if eval_df is None:
        ax.text(0.5, 0.5, 'Polygenic evaluation\ndata not available', 
               ha='center', va='center', fontsize=11, transform=ax.transAxes,
               color=colors.coral_red)
        ax.text(-0.08, 1.05, 'B', transform=ax.transAxes,
               fontsize=16, fontweight='bold', va='bottom', ha='left')
        log("  NO DATA - Panel B empty", "ERROR")
        return
    
    # Find weight columns for each trait (e.g., Weight_FBC, Pct_Var_FBC)
    weight_cols = {}
    var_cols = {}
    
    for trait in colors.trait_order:
        # Look for weight columns
        for col in eval_df.columns:
            if f'Weight_{trait}' in col:
                weight_cols[trait] = col
            elif f'Pct_Var_{trait}' in col:
                var_cols[trait] = col
    
    if not weight_cols and not var_cols:
        log("  Cannot find weight or variance columns", "ERROR")
        ax.text(0.5, 0.5, 'Cannot find\nweight columns', 
               ha='center', va='center', fontsize=11, transform=ax.transAxes)
        ax.text(-0.08, 1.05, 'B', transform=ax.transAxes,
               fontsize=16, fontweight='bold', va='bottom', ha='left')
        return
    
    # Use weight columns to compute cumulative variance
    n_snps = len(eval_df)
    fractions = np.linspace(0, 1, 100)
    
    for trait in colors.trait_order:
        col = weight_cols.get(trait) or var_cols.get(trait)
        if col is None:
            continue
        
        # Get absolute weights and sort
        weights = np.abs(eval_df[col].values)
        sorted_idx = np.argsort(weights)[::-1]  # Descending
        sorted_weights = weights[sorted_idx]
        
        # Compute cumulative sum
        cumsum = np.cumsum(sorted_weights)
        total = cumsum[-1] if cumsum[-1] > 0 else 1
        cumsum_norm = cumsum / total
        
        # Interpolate to standard fractions
        snp_fractions = np.arange(1, n_snps + 1) / n_snps
        cumsum_interp = np.interp(fractions, snp_fractions, cumsum_norm)
        
        color = colors.trait_colors_distinct.get(trait, colors.gray)
        ax.plot(fractions * 100, cumsum_interp * 100, 
               label=trait, color=color, linewidth=2)
    
    # Add diagonal reference line (uniform distribution)
    ax.plot([0, 100], [0, 100], 'k--', linewidth=1, alpha=0.5, label='Uniform')
    
    # Highlight key percentiles
    ax.axvline(1, color=colors.coral_red, linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axvline(5, color=colors.coral_red, linestyle=':', linewidth=1.5, alpha=0.5)
    ax.text(1.5, 95, '1%', fontsize=8, color=colors.coral_red)
    ax.text(5.5, 95, '5%', fontsize=8, color=colors.coral_red)
    
    # Styling
    ax.set_xlabel('SNP Fraction (%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Cumulative Variance (%)', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3, linestyle='--')
    
    # Panel label
    ax.text(-0.08, 1.05, 'B', transform=ax.transAxes,
           fontsize=16, fontweight='bold', va='bottom', ha='left')
    
    log("  Panel B complete", "OK")


# ============================================================================
# PANEL C: BINN Gene Importance Heatmap
# ============================================================================

def create_panel_C(data, ax):
    """
    Panel C: Heatmap of BINN gene scores across traits.
    Shows top 50 genes by maximum score.
    """
    log("\n[PANEL C] BINN Gene Importance Heatmap")
    
    binn_df = data['binn_scores_wide']
    
    if binn_df is None:
        ax.text(0.5, 0.5, 'BINN scores\nnot available', 
               ha='center', va='center', fontsize=11, transform=ax.transAxes,
               color=colors.coral_red)
        ax.text(-0.1, 1.05, 'C', transform=ax.transAxes,
               fontsize=16, fontweight='bold', va='bottom', ha='left')
        log("  NO DATA - Panel C empty", "ERROR")
        return
    
    # Find gene ID and score columns
    gene_col = None
    for col in ['gene_id', 'gene_name', 'Gene']:
        if col in binn_df.columns:
            gene_col = col
            break
    
    if gene_col is None:
        gene_col = binn_df.columns[0]
    
    # Find score columns (score_FBC, score_AFW, etc.)
    score_cols = {}
    for trait in colors.trait_order:
        for col in binn_df.columns:
            if f'score_{trait}' in col or col == trait:
                score_cols[trait] = col
                break
    
    if len(score_cols) == 0:
        log("  Cannot find score columns", "ERROR")
        ax.text(0.5, 0.5, 'Cannot find\nscore columns', 
               ha='center', va='center', fontsize=11, transform=ax.transAxes)
        ax.text(-0.1, 1.05, 'C', transform=ax.transAxes,
               fontsize=16, fontweight='bold', va='bottom', ha='left')
        return
    
    # Prepare matrix
    binn_df = binn_df.copy()
    
    # Calculate max score across traits for ranking
    score_matrix = binn_df[[score_cols[t] for t in score_cols]].values
    binn_df['max_score'] = np.abs(score_matrix).max(axis=1)
    
    # Select top 50 genes by max score
    top_n = 50
    top_genes = binn_df.nlargest(top_n, 'max_score')
    
    log(f"  Top {top_n} genes selected", "DEBUG")
    
    # Create heatmap data
    heatmap_data = []
    gene_labels = []
    
    for _, row in top_genes.iterrows():
        gene_label = row.get('gene_name', '') or row[gene_col]
        if pd.isna(gene_label) or gene_label == '':
            gene_label = row[gene_col]
        gene_labels.append(str(gene_label)[:15])  # Truncate long names
        
        scores = [row[score_cols[t]] if t in score_cols else 0 for t in colors.trait_order]
        heatmap_data.append(scores)
    
    heatmap_array = np.array(heatmap_data)
    
    # Create heatmap
    cmap = colors.get_green_cmap()
    
    im = ax.imshow(np.abs(heatmap_array), cmap=cmap, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('|BINN Score|', fontsize=9)
    
    # Styling
    ax.set_xticks(range(len(colors.trait_order)))
    ax.set_xticklabels(colors.trait_order, fontsize=9, fontweight='bold')
    ax.set_yticks(range(len(gene_labels)))
    ax.set_yticklabels(gene_labels, fontsize=6)
    
    ax.set_xlabel('Trait', fontsize=10, fontweight='bold')
    ax.set_ylabel(f'Top {top_n} Genes', fontsize=10, fontweight='bold')
    
    
    # Panel label
    ax.text(-0.12, 1.05, 'C', transform=ax.transAxes,
           fontsize=16, fontweight='bold', va='bottom', ha='left')
    
    log("  Panel C complete", "OK")


# ============================================================================
# PANEL D: Pleiotropic Hub Structure
# ============================================================================

def create_panel_D(data, ax):
    """
    Panel D: Dot plot showing pleiotropic hub genes and their trait coverage.
    """
    log("\n[PANEL D] Pleiotropic Hub Structure")
    
    pleio_df = data['pleiotropy_scores']
    
    if pleio_df is None:
        ax.text(0.5, 0.5, 'Pleiotropy data\nnot available', 
               ha='center', va='center', fontsize=11, transform=ax.transAxes,
               color=colors.coral_red)
        ax.text(-0.15, 1.05, 'D', transform=ax.transAxes,
               fontsize=16, fontweight='bold', va='bottom', ha='left')
        log("  NO DATA - Panel D empty", "ERROR")
        return
    
    # Get trait count column
    trait_count_col = None
    for col in ['n_traits_above_90pct', 'trait_count', 'n_traits']:
        if col in pleio_df.columns:
            trait_count_col = col
            break
    
    if trait_count_col is None:
        log("  Cannot find trait count column", "ERROR")
        ax.text(0.5, 0.5, 'Cannot find\ntrait count column', 
               ha='center', va='center', fontsize=11, transform=ax.transAxes)
        ax.text(-0.15, 1.05, 'D', transform=ax.transAxes,
               fontsize=16, fontweight='bold', va='bottom', ha='left')
        return
    
    # Get distribution of genes by trait count
    trait_counts = pleio_df[trait_count_col].value_counts().sort_index()
    
    log(f"  Trait count distribution:\n{trait_counts}", "DEBUG")
    
    # Create bar plot of distribution
    x_vals = trait_counts.index.values
    y_vals = trait_counts.values
    
    # Color by hub status (≥4 traits = hub)
    hub_threshold = 4
    bar_colors = [colors.class_colors['hub'] if x >= hub_threshold else
                  colors.class_colors['multi_trait'] if x >= 2 else
                  colors.class_colors['single_trait'] for x in x_vals]
    
    bars = ax.bar(x_vals, y_vals, color=bar_colors, edgecolor='black', linewidth=1)
    
    # Add count labels
    for bar, count in zip(bars, y_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
               f'{int(count)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Highlight hub threshold
    ax.axvline(hub_threshold - 0.5, color=colors.coral_red, linestyle='--',
              linewidth=2, label=f'Hub threshold (≥{hub_threshold})')
    
    # Add hub count annotation - Moved to avoid overlap
    n_hubs = pleio_df[pleio_df[trait_count_col] >= hub_threshold].shape[0]
    ax.text(0.95, 0.85, f'Hub genes: {n_hubs}', transform=ax.transAxes,
           fontsize=10, fontweight='bold', ha='right', va='top',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                    edgecolor=colors.class_colors['hub'], alpha=0.9))
    
    # Styling
    ax.set_xlabel('Number of Traits (top 10% score)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Number of Genes', fontsize=10, fontweight='bold')
    
    ax.set_xticks(x_vals)
    ax.legend(loc='upper right', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Panel label
    ax.text(-0.15, 1.05, 'D', transform=ax.transAxes,
           fontsize=16, fontweight='bold', va='bottom', ha='left')
    
    log("  Panel D complete", "OK")


# ============================================================================
# PANEL E: BINN Model Performance / Decomposition
# ============================================================================

def create_panel_E(data, ax):
    """
    Panel E: BINN Decomposition - Ridge(20k) vs Ridge(490) vs BINN comparison.
    
    Shows that BINN's accuracy gains derive primarily from:
    1. Feature selection (Ridge 20k → Ridge 490)
    2. Gene-level architecture (Ridge 490 → BINN)
    
    Falls back to BINN-only display if decomposition data unavailable.
    """
    log("\n[PANEL E] BINN Decomposition Comparison")
    
    binn_cv = data.get('binn_cv')
    decomp = data.get('binn_decomposition')
    
    # Define trait order
    trait_order = colors.trait_order
    
    # Check if we have decomposition data
    if decomp is None:
        log("  No decomposition data - showing BINN only", "WARN")
        _create_panel_E_binn_only(binn_cv, ax, trait_order)
        return
    
    log(f"  Decomposition data columns: {list(decomp.columns)}", "DEBUG")
    
    # Gather data for each trait
    ridge_20k = []
    ridge_490 = []
    binn_vals = []
    valid_traits = []
    
    for trait in trait_order:
        # Get decomposition values
        d_row = decomp[decomp['trait'] == trait]
        if len(d_row) == 0:
            continue
            
        r_20k = d_row['Ridge_Full_20k'].values[0] if 'Ridge_Full_20k' in decomp.columns else np.nan
        r_490 = d_row['Ridge_BINN_490'].values[0] if 'Ridge_BINN_490' in decomp.columns else np.nan
        
        # Get BINN value from binn_cv
        if binn_cv is not None:
            b_row = binn_cv[binn_cv['trait'] == trait]
            if len(b_row) > 0:
                r_col = 'mean_r' if 'mean_r' in binn_cv.columns else 'r'
                r_binn = b_row[r_col].values[0]
            else:
                r_binn = np.nan
        else:
            r_binn = np.nan
        
        ridge_20k.append(r_20k)
        ridge_490.append(r_490)
        binn_vals.append(r_binn)
        valid_traits.append(trait)
    
    if len(valid_traits) == 0:
        log("  No valid trait data found", "ERROR")
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.text(-0.1, 1.05, 'E', transform=ax.transAxes,
               fontsize=16, fontweight='bold', va='bottom', ha='left')
        return
    
    # Convert to arrays
    ridge_20k = np.array(ridge_20k)
    ridge_490 = np.array(ridge_490)
    binn_vals = np.array(binn_vals)
    
    # Create grouped bar chart
    x = np.arange(len(valid_traits))
    width = 0.25
    
    # Colors
    color_20k = '#A9A9A9'     # Dark gray
    color_490 = '#4682B4'     # Steel blue  
    color_binn = '#32CD32'    # Lime green
    
    # Plot bars
    bars1 = ax.bar(x - width, ridge_20k, width, label='Ridge\n(20k)', 
                   color=color_20k, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x, ridge_490, width, label='Ridge\n(490)', 
                   color=color_490, edgecolor='white', linewidth=0.5)
    bars3 = ax.bar(x + width, binn_vals, width, label='BINN', 
                   color=color_binn, edgecolor='white', linewidth=0.5)
    
    # Add value labels on BINN bars only
    for bar, val in zip(bars3, binn_vals):
        if np.isfinite(val) and val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Styling
    ax.set_xticks(x)
    ax.set_xticklabels(valid_traits, fontsize=9, fontweight='bold', rotation=45, ha='right')
    ax.set_ylabel('Accuracy (r)', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.0)
    
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9, 
              handlelength=1, handletextpad=0.3, borderpad=0.3)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Panel label
    ax.text(-0.1, 1.05, 'E', transform=ax.transAxes,
           fontsize=16, fontweight='bold', va='bottom', ha='left')
    
    # Log summary
    for i, trait in enumerate(valid_traits):
        gain_sel = ridge_490[i] - ridge_20k[i]
        gain_arch = binn_vals[i] - ridge_490[i]
        log(f"  {trait}: 20k={ridge_20k[i]:.2f} → 490={ridge_490[i]:.2f} (+{gain_sel:.2f}) → BINN={binn_vals[i]:.2f} (+{gain_arch:.2f})", "DEBUG")
    
    log("  Panel E complete", "OK")


def _create_panel_E_binn_only(binn_cv, ax, trait_order):
    """Fallback: Show only BINN results if decomposition data unavailable."""
    if binn_cv is None:
        ax.text(0.5, 0.5, 'BINN CV data\nnot available', 
               ha='center', va='center', fontsize=10, transform=ax.transAxes,
               color='#008080')
        ax.text(-0.1, 1.05, 'E', transform=ax.transAxes,
               fontsize=16, fontweight='bold', va='bottom', ha='left')
        return
    
    # Get r column
    r_col = None
    for col in ['mean_r', 'r', 'pearson_r']:
        if col in binn_cv.columns:
            r_col = col
            break
    
    if r_col is None:
        ax.text(0.5, 0.5, 'Cannot find\naccuracy column', 
               ha='center', va='center', fontsize=10, transform=ax.transAxes)
        ax.text(-0.1, 1.05, 'E', transform=ax.transAxes,
               fontsize=16, fontweight='bold', va='bottom', ha='left')
        return
    
    # Filter and order traits
    trait_col = 'trait' if 'trait' in binn_cv.columns else binn_cv.columns[0]
    plot_data = binn_cv[binn_cv[trait_col].isin(trait_order)].copy()
    plot_data['order'] = plot_data[trait_col].apply(lambda x: trait_order.index(x) if x in trait_order else 999)
    plot_data = plot_data.sort_values('order')
    
    traits = plot_data[trait_col].values
    binn_r = plot_data[r_col].values
    
    x = np.arange(len(traits))
    
    # Trait colors
    bar_colors = [colors.trait_colors.get(t, colors.gray) for t in traits]
    
    bars = ax.bar(x, binn_r, color=bar_colors, edgecolor='white', linewidth=1.5)
    
    for bar, val in zip(bars, binn_r):
        if np.isfinite(val) and val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.text(0.5, 0.95, '473 genes', transform=ax.transAxes,
           ha='center', va='top', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#32CD32', 
                    alpha=0.3, edgecolor='#2E8B57'))
    
    ax.set_xticks(x)
    ax.set_xticklabels(traits, fontsize=10, fontweight='bold', rotation=45, ha='right')
    ax.set_ylabel('Accuracy (r)', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.0)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    ax.text(-0.1, 1.05, 'E', transform=ax.transAxes,
           fontsize=16, fontweight='bold', va='bottom', ha='left')


# ============================================================================
# MAIN FIGURE ASSEMBLY
# ============================================================================

def create_figure_4():
    """
    Assemble all panels into Figure 4.
    
    Layout:
        Row 0: A (2 cols) + B (4 cols)
        Row 1: C (3 cols) + D (2 cols) + E (1 col)
    """
    log("\n" + "=" * 70)
    log("ASSEMBLING FIGURE 4: Polygenic Backbones and Pleiotropic Hub Genes")
    log("=" * 70)
    
    # Load data
    data = load_data()
    
    # Create figure with gridspec
    fig = plt.figure(figsize=(16, 11))
    
    gs = gridspec.GridSpec(
        nrows=2, ncols=6,
        height_ratios=[1.0, 1.2],
        width_ratios=[1.0, 1.0, 0.4, 0.4, 1.6, 1.6],
        hspace=0.25,
        wspace=0.58
    )
    
    # Create axes
    ax_A = fig.add_subplot(gs[0, 0:2])
    ax_B = fig.add_subplot(gs[0, 2:6])
    ax_C = fig.add_subplot(gs[1, 0:2])
    ax_D = fig.add_subplot(gs[1, 2:4])
    ax_E = fig.add_subplot(gs[1, 4:6])
    
    # Create panels
    log("\nCreating panels...")
    
    create_panel_A(data, ax_A)
    create_panel_B(data, ax_B)
    create_panel_C(data, ax_C)
    create_panel_D(data, ax_D)
    create_panel_E(data, ax_E)
    
    # Save figure
    output_path = os.path.join(FIGURES_DIR, "figure_4.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    log(f"\n[OK] Figure saved: {output_path}", "OK")
    
    # Save PDF
    output_pdf = os.path.join(FIGURES_DIR, "figure_4.pdf")
    fig.savefig(output_pdf, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    log(f"[OK] PDF saved: {output_pdf}", "OK")
    
    # Save source data
    log("\nSaving source data...")
    
    if data['polygenic_arch'] is not None:
        arch_out = os.path.join(FIGURE_SUBDIR, "panel_A_polygenic_architecture.csv")
        data['polygenic_arch'].to_csv(arch_out, index=False)
        log(f"  [OK] Panel A source: {arch_out}", "OK")
    
    if data['binn_scores_wide'] is not None:
        binn_out = os.path.join(FIGURE_SUBDIR, "panel_C_binn_scores.csv")
        data['binn_scores_wide'].to_csv(binn_out, index=False)
        log(f"  [OK] Panel C source: {binn_out}", "OK")
    
    if data['pleiotropy_scores'] is not None:
        pleio_out = os.path.join(FIGURE_SUBDIR, "panel_D_pleiotropy.csv")
        data['pleiotropy_scores'].to_csv(pleio_out, index=False)
        log(f"  [OK] Panel D source: {pleio_out}", "OK")
    
    if data['binn_cv'] is not None:
        binn_cv_out = os.path.join(FIGURE_SUBDIR, "panel_E_binn_cv.csv")
        data['binn_cv'].to_csv(binn_cv_out, index=False)
        log(f"  [OK] Panel E source: {binn_cv_out}", "OK")
    
    if data.get('binn_decomposition') is not None:
        decomp_out = os.path.join(FIGURE_SUBDIR, "panel_E_binn_decomposition.csv")
        data['binn_decomposition'].to_csv(decomp_out, index=False)
        log(f"  [OK] Panel E decomposition source: {decomp_out}", "OK")
    
    log("\n" + "=" * 70)
    log("FIGURE 4 COMPLETE!", "OK")
    log("=" * 70)
    
    return fig


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MANGO GWAS - FIGURE 4 GENERATOR")
    print("Polygenic Backbones and Pleiotropic Hub Genes")
    print("=" * 70)
    print(f"Output directory: {FIGURES_DIR}")
    print("-" * 70)
    
    try:
        fig = create_figure_4()
        print("\n" + "=" * 70)
        print("SUCCESS!")
        print(f"  Figure: {os.path.join(FIGURES_DIR, 'figure_4.png')}")
        print(f"  Data:   {FIGURE_SUBDIR}")
        print("=" * 70)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)