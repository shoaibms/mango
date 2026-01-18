#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure S6: BINN Training Behaviour and Hub Gene Details
========================================================

Panels:
    A: BINN CV performance summary (accuracy per trait per fold)
    B: Top 10 genes per trait (horizontal bar chart for FBC)
    C: SNP density per gene histogram (distribution across genes)
    D: Pleiotropy scores - genes affecting multiple traits

"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# Import configuration
try:
    import figure_config
    config = figure_config.config
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import figure_config
    config = figure_config.config

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
OUTPUT_DIR = config.paths.OUTPUT_DIR
FIGURES_DIR = config.paths.FIGURES_DIR
FIGURE_SUBDIR = config.paths.figure_subdir('s6')

# Data paths
DATA_PATHS = {
    'binn_cv_summary': os.path.join(OUTPUT_DIR, "idea_3", "binn_training", "binn_cv_summary.csv"),
    'binn_cv_results': os.path.join(OUTPUT_DIR, "idea_3", "binn_training", "binn_cv_results.csv"),
    'gene_scores_wide': os.path.join(OUTPUT_DIR, "idea_3", "binn_explain", "binn_gene_scores_wide.csv"),
    'gene_scores_long': os.path.join(OUTPUT_DIR, "idea_3", "binn_explain", "binn_gene_scores_long.csv"),
    'gene_table': os.path.join(OUTPUT_DIR, "idea_3", "binn_maps", "binn_gene_table.csv"),
    'pleiotropy': os.path.join(OUTPUT_DIR, "idea_3", "binn_explain", "binn_gene_pleiotropy_scores.csv"),
}

# Standard trait order
TRAIT_ORDER = config.trait_order  # ['FBC', 'AFW', 'FF', 'TC', 'TSS']

# Ensure output directories exist
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(FIGURE_SUBDIR, exist_ok=True)


# ============================================================================
# DATA LOADING
# ============================================================================


def load_data():
    """Load all data for Figure S6."""
    print("Loading data for Figure S6...")
    
    data = {}
    
    for key, path in DATA_PATHS.items():
        if os.path.exists(path):
            data[key] = pd.read_csv(path)
        else:
            print(f"Warning: File not found: {path}")
            data[key] = None
    
    return data


# ============================================================================
# PANEL A: BINN CV Performance Summary
# ============================================================================

def create_panel_A(data, ax):
    """
    Panel A: BINN cross-validation performance per trait.
    """
    cv_summary = data['binn_cv_summary']
    cv_results = data['binn_cv_results']
    
    # Try to use per-fold results for boxplot, or summary for bars
    if cv_results is not None and 'trait' in cv_results.columns:
        # Find the accuracy column
        r_col = None
        for col in ['r', 'pearson_r', 'val_r', 'accuracy']:
            if col in cv_results.columns:
                r_col = col
                break
        
        if r_col is None:
            create_panel_A_from_summary(cv_summary, ax)
            return
        
        # Order traits using standard order
        ordered_traits = [t for t in TRAIT_ORDER if t in cv_results['trait'].unique()]
        
        # Create boxplot data
        box_data = []
        box_positions = []
        box_colors = []
        
        for i, trait in enumerate(ordered_traits):
            trait_data = cv_results[cv_results['trait'] == trait][r_col].dropna().values
            if len(trait_data) > 0:
                box_data.append(trait_data)
                box_positions.append(i)
                box_colors.append(config.trait_colors.get(trait, config.colors.gray))
        
        if not box_data:
            create_panel_A_from_summary(cv_summary, ax)
            return
        
        # Create boxplot
        bp = ax.boxplot(box_data, positions=box_positions, widths=0.6, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add individual points
        for i, (vals, pos) in enumerate(zip(box_data, box_positions)):
            jitter = np.random.uniform(-0.15, 0.15, len(vals))
            ax.scatter([pos + j for j in jitter], vals, c=[box_colors[i]], 
                      s=40, alpha=0.7, edgecolors='white', linewidth=0.5, zorder=3)
        
        ax.set_xticks(range(len(ordered_traits)))
        ax.set_xticklabels(ordered_traits, fontsize=10, fontweight='bold')
        
    elif cv_summary is not None:
        create_panel_A_from_summary(cv_summary, ax)
        return
    else:
        ax.text(0.5, 0.5, 'BINN CV data\nnot available', ha='center', va='center',
               fontsize=10, transform=ax.transAxes)
        config.add_panel_label(ax, 'A')
        return
    
    # Reference line at 0
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    
    # Styling
    ax.set_ylabel('Prediction Accuracy (r)', fontsize=10, fontweight='bold')
    config.style_axis(ax, spines=['bottom', 'left'], grid=True)
    ax.grid(axis='x', alpha=0.0)
    
    config.add_panel_label(ax, 'A')


def create_panel_A_from_summary(cv_summary, ax):
    """Create Panel A from summary data (bar chart)."""
    if cv_summary is None:
        ax.text(0.5, 0.5, 'No CV summary', ha='center', va='center')
        config.add_panel_label(ax, 'A')
        return
    
    # Find mean accuracy column
    r_col = None
    for col in ['mean_r', 'r_mean', 'mean_pearson_r', 'accuracy']:
        if col in cv_summary.columns:
            r_col = col
            break
    
    if r_col is None or 'trait' not in cv_summary.columns:
        ax.text(0.5, 0.5, 'Cannot parse\nCV summary', ha='center', va='center')
        config.add_panel_label(ax, 'A')
        return
    
    # Order traits using standard order
    ordered_traits = [t for t in TRAIT_ORDER if t in cv_summary['trait'].values]
    
    plot_df = cv_summary.set_index('trait').loc[ordered_traits].reset_index()
    
    x = np.arange(len(plot_df))
    values = plot_df[r_col].values
    bar_colors = [config.trait_colors.get(t, config.colors.gray) for t in plot_df['trait']]
    
    ax.bar(x, values, color=bar_colors, edgecolor='white', width=0.6)
    
    # Value labels
    for i, val in enumerate(values):
        ax.text(i, val + 0.02, f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df['trait'], fontsize=10, fontweight='bold')
    ax.set_ylabel('Mean Accuracy (r)', fontsize=10, fontweight='bold')
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    config.style_axis(ax, spines=['bottom', 'left'], grid=True)
    ax.grid(axis='x', alpha=0.0)
    
    config.add_panel_label(ax, 'A')


# ============================================================================
# PANEL B: Top Genes by BINN Score (FBC)
# ============================================================================

def create_panel_B(data, ax):
    """
    Panel B: Top 10 genes by BINN importance score for FBC.
    """
    gene_scores = data['gene_scores_wide']
    
    if gene_scores is None:
        ax.text(0.5, 0.5, 'Gene scores\nnot available', ha='center', va='center',
               fontsize=10, transform=ax.transAxes)
        config.add_panel_label(ax, 'B', x=-0.15)
        return
    
    # Find score column for FBC
    score_col = None
    for col in ['score_FBC', 'FBC', 'FBC_score']:
        if col in gene_scores.columns:
            score_col = col
            break
    
    if score_col is None:
        ax.text(0.5, 0.5, 'Cannot find\nFBC scores', ha='center', va='center')
        config.add_panel_label(ax, 'B', x=-0.15)
        return
    
    # Get top 10 genes
    top_n = 10
    top_genes = gene_scores.nlargest(top_n, score_col)
    
    # Get gene names
    name_col = 'gene_name' if 'gene_name' in top_genes.columns else 'gene_id'
    
    # Reverse for horizontal bar chart (top gene at top)
    y_pos = np.arange(top_n)
    values = top_genes[score_col].values[::-1]
    labels = top_genes[name_col].values[::-1]
    
    # Shorten long gene names
    labels = [str(l)[:25] + '...' if len(str(l)) > 25 else str(l) for l in labels]
    
    # Color gradient using config colormap
    cmap = config.colors.cmap_greens
    norm_vals = np.linspace(0.3, 0.9, top_n)[::-1]
    bar_colors = [cmap(v) for v in norm_vals]
    
    bars = ax.barh(y_pos, values, color=bar_colors, edgecolor='white', height=0.7)
    
    # Value labels
    for bar, val in zip(bars, values):
        ax.text(val + max(values)*0.02, bar.get_y() + bar.get_height()/2,
               f'{val:.3f}', va='center', fontsize=8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('BINN Gene Score', fontsize=10, fontweight='bold')
    
    config.style_axis(ax, spines=['bottom', 'left'], grid=True)
    ax.grid(axis='y', alpha=0.0)
    
    config.add_panel_label(ax, 'B', x=-0.15)


# ============================================================================
# PANEL C: SNP Density per Gene
# ============================================================================

def create_panel_C(data, ax):
    """
    Panel C: Histogram of SNP count per gene.
    """
    gene_table = data['gene_table']
    
    if gene_table is None:
        # Try gene_scores_wide as fallback
        gene_table = data['gene_scores_wide']
    
    if gene_table is None:
        ax.text(0.5, 0.5, 'Gene table\nnot available', ha='center', va='center',
               fontsize=10, transform=ax.transAxes)
        config.add_panel_label(ax, 'C')
        return
    
    # Find n_snps column
    snp_col = None
    for col in ['n_snps', 'snp_count', 'num_snps']:
        if col in gene_table.columns:
            snp_col = col
            break
    
    if snp_col is None:
        ax.text(0.5, 0.5, 'Cannot find\nSNP count column', ha='center', va='center')
        config.add_panel_label(ax, 'C')
        return
    
    snp_counts = gene_table[snp_col].dropna().values
    
    # Create histogram
    bins = np.arange(0, max(snp_counts) + 2, 1)
    ax.hist(snp_counts, bins=bins, color=config.colors.royalblue, 
           edgecolor='white', alpha=0.7)
    
    # Add statistics
    mean_snps = np.mean(snp_counts)
    median_snps = np.median(snp_counts)
    
    ax.axvline(mean_snps, color=config.colors.limegreen, linestyle='--', 
              linewidth=2, label=f'Mean: {mean_snps:.1f}')
    ax.axvline(median_snps, color=config.colors.turquoise, linestyle=':', 
              linewidth=2, label=f'Median: {median_snps:.0f}')
    
    # Styling
    ax.set_xlabel('Number of SNPs per Gene', fontsize=10, fontweight='bold')
    ax.set_ylabel('Number of Genes', fontsize=10, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, frameon=False)
    
    config.style_axis(ax, spines=['bottom', 'left'], grid=True)
    ax.grid(axis='x', alpha=0.0)
    
    config.add_panel_label(ax, 'C')


# ============================================================================
# PANEL D: Pleiotropy Scores
# ============================================================================

def create_panel_D(data, ax):
    """
    Panel D: Pleiotropy analysis - genes affecting multiple traits.
    """
    pleiotropy = data['pleiotropy']
    
    if pleiotropy is None:
        # Try to compute from gene_scores_wide
        gene_scores = data['gene_scores_wide']
        if gene_scores is not None:
            create_panel_D_from_scores(gene_scores, ax)
        else:
            ax.text(0.5, 0.5, 'Pleiotropy data\nnot available', ha='center', va='center',
                   fontsize=10, transform=ax.transAxes)
        config.add_panel_label(ax, 'D')
        return
    
    # Find relevant columns
    n_traits_col = None
    for col in ['n_traits_above_90pct', 'n_traits', 'pleiotropy_degree']:
        if col in pleiotropy.columns:
            n_traits_col = col
            break
    
    if n_traits_col is None:
        create_panel_D_from_scores(data['gene_scores_wide'], ax)
        return
    
    # Count genes by number of traits affected
    trait_counts = pleiotropy[n_traits_col].value_counts().sort_index()
    
    x = trait_counts.index.astype(int)
    y = trait_counts.values
    
    # Color by pleiotropy level using config colors
    bar_colors = []
    for n in x:
        if n >= 4:
            bar_colors.append(config.trait_colors['FBC'])  # Hub genes
        elif n >= 2:
            bar_colors.append(config.colors.royalblue)
        else:
            bar_colors.append(config.colors.lightgray)
    
    ax.bar(x, y, color=bar_colors, edgecolor='white', width=0.7)
    
    # Value labels
    for xi, yi in zip(x, y):
        ax.text(xi, yi + max(y)*0.02, str(yi), ha='center', va='bottom', fontsize=9)
    
    # Add annotation for hub genes
    n_hubs = sum(pleiotropy[n_traits_col] >= 3)
    config.add_stats_box(ax, f'Hub genes (≥3 traits): {n_hubs}', x=0.95, y=0.95, ha='right')
    
    # Styling
    ax.set_xlabel('Number of Traits Affected', fontsize=10, fontweight='bold')
    ax.set_ylabel('Number of Genes', fontsize=10, fontweight='bold')
    ax.set_xticks(range(int(x.min()), int(x.max()) + 1))
    
    config.style_axis(ax, spines=['bottom', 'left'], grid=True)
    ax.grid(axis='x', alpha=0.0)
    
    config.add_panel_label(ax, 'D')


def create_panel_D_from_scores(gene_scores, ax):
    """Create pleiotropy panel from gene scores wide format."""
    if gene_scores is None:
        ax.text(0.5, 0.5, 'No gene scores', ha='center', va='center')
        return
    
    # Find score columns
    score_cols = [c for c in gene_scores.columns if c.startswith('score_')]
    
    if not score_cols:
        ax.text(0.5, 0.5, 'No score columns', ha='center', va='center')
        return
    
    # Calculate number of traits with significant score (> 90th percentile)
    n_traits = []
    for _, row in gene_scores.iterrows():
        count = 0
        for col in score_cols:
            threshold = gene_scores[col].quantile(0.9)
            if row[col] > threshold:
                count += 1
        n_traits.append(count)
    
    # Count distribution
    from collections import Counter
    trait_counts = Counter(n_traits)
    
    x = sorted(trait_counts.keys())
    y = [trait_counts[k] for k in x]
    
    ax.bar(x, y, color=config.colors.royalblue, edgecolor='white', width=0.7)
    
    ax.set_xlabel('Number of Traits (top 10%)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Number of Genes', fontsize=10, fontweight='bold')
    
    config.style_axis(ax, spines=['bottom', 'left'], grid=True)
    ax.grid(axis='x', alpha=0.0)


# ============================================================================
# MAIN FIGURE ASSEMBLY
# ============================================================================

def create_figure_s6():
    """
    Assemble Figure S6: BINN Training and Hub Gene Details.
    
    Layout (2x2):
        A: BINN CV performance
        B: Top 10 genes for FBC
        C: SNP density per gene
        D: Pleiotropy distribution
    """
    print("Assembling Figure S6...")
    
    data = load_data()
    
    fig = plt.figure(figsize=(14, 9))
    
    gs = gridspec.GridSpec(
        nrows=2, ncols=2,
        hspace=0.25,
        wspace=0.3
    )
    
    # Create panels
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[1, 0])
    ax_D = fig.add_subplot(gs[1, 1])
    
    # Panel A: BINN CV performance
    create_panel_A(data, ax_A)
    
    # Panel B: Top genes for FBC
    create_panel_B(data, ax_B)
    
    # Panel C: SNP density per gene
    create_panel_C(data, ax_C)
    
    # Panel D: Pleiotropy distribution
    create_panel_D(data, ax_D)
    
    # Save
    output_path = os.path.join(FIGURES_DIR, "figure_s6.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure saved: {output_path}")
    
    output_pdf = os.path.join(FIGURES_DIR, "figure_s6.pdf")
    fig.savefig(output_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"PDF saved: {output_pdf}")
    
    return fig


if __name__ == "__main__":
    try:
        fig = create_figure_s6()
        print("Success!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)