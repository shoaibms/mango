#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure S1: Phenotypic Data Distribution and Genetic Structure

Panel A: Phenotype histograms (FBC, AFW, TSS, TC, FF)
Panel B: PC3 vs PC4 scatter (all samples, colored by cluster)
Panel C: Trait distribution by genetic cluster (violin plots)
Panel D: PC3 vs PC4 within Cluster 1 (Americas vs South Asia origin)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import warnings
import figure_config as config

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
OUTPUT_DIR = r"C:\Users\ms\Desktop\mango\output"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
FIGURE_SUBDIR = os.path.join(FIGURES_DIR, "figure_s1")

# Data paths
DATA_PATHS = {
    'pheno_core': os.path.join(OUTPUT_DIR, "idea_1", "core_data", "pheno_core.csv"),
    'pc_scores': os.path.join(OUTPUT_DIR, "idea_1", "summary", "pc_scores_clusters.csv"),
    'origin': r"C:\Users\ms\Desktop\mango\data\main_data\origin.csv",
}

# Ensure output directories exist
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(FIGURE_SUBDIR, exist_ok=True)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load all data files for Figure S1."""
    print("Loading data...")
    
    data = {}
    for key, path in DATA_PATHS.items():
        if os.path.exists(path):
            # Load pheno_core with index; pc_scores with sample_id index when present
            if key == 'pheno_core':
                data[key] = pd.read_csv(path, index_col=0)
            elif key == 'pc_scores':
                df = pd.read_csv(path)
                if 'sample_id' in df.columns:
                    df = df.set_index('sample_id')
                data[key] = df
            else:
                data[key] = pd.read_csv(path)
            print(f"  Loaded {key}: {data[key].shape}")
        else:
            print(f"  [ERROR] File not found: {path}")
            data[key] = None
    
    return data


# ============================================================================
# PANEL A: Phenotype Histograms
# ============================================================================

def create_panel_A(data, axes):
    """
    Panel A: Faceted histograms showing distribution of each trait.
    """
    pheno_df = data['pheno_core']
    
    if pheno_df is None:
        for ax in axes:
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center',
                   fontsize=10, transform=ax.transAxes)
        axes[0].text(-0.15, 1.05, 'A', transform=axes[0].transAxes,
                    fontsize=14, fontweight='bold')
        return
    
    # Identify trait columns
    trait_cols = [col for col in config.trait_order if col in pheno_df.columns]
    
    if not trait_cols:
        potential_traits = [col for col in pheno_df.columns if col != 'ID']
        trait_cols = potential_traits[:5]
    
    # Plot histograms
    for i, (ax, trait) in enumerate(zip(axes, config.trait_order)):
        if trait in pheno_df.columns:
            values = pheno_df[trait].dropna().values
            color = config.trait_colors.get(trait, config.config.colors.steelblue)
            
            ax.hist(values, bins=20, color=color, edgecolor='white', 
                   linewidth=0.5, alpha=0.8)
            
            # Add statistics annotation
            mean_val = np.mean(values)
            std_val = np.std(values)
            n_val = len(values)
            
            stats_text = f'n={n_val}\nμ={mean_val:.1f}\nσ={std_val:.1f}'
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                   fontsize=8, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel(trait, fontsize=10, fontweight='bold')
            ax.set_ylabel('Count' if i == 0 else '', fontsize=9)
            
        else:
            ax.text(0.5, 0.5, f'{trait}\nnot found', ha='center', va='center',
                   fontsize=10, transform=ax.transAxes)
            ax.set_xlabel(trait, fontsize=10)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=8)
    
    # Panel label on first subplot
    axes[0].text(-0.25, 1.05, 'A', transform=axes[0].transAxes,
                fontsize=14, fontweight='bold')


# ============================================================================
# PANEL B: PC3 vs PC4
# ============================================================================

def create_panel_B(data, ax):
    """
    Panel B: PC3 vs PC4 scatter plot colored by cluster.
    """
    pc_df = data['pc_scores']
    
    if pc_df is None:
        ax.text(0.5, 0.5, 'Data not available', ha='center', va='center',
               fontsize=12, transform=ax.transAxes)
        ax.text(-0.12, 1.05, 'B', transform=ax.transAxes,
               fontsize=14, fontweight='bold')
        return
    
    required = ['PC3', 'PC4', 'cluster']
    missing = [col for col in required if col not in pc_df.columns]
    
    if missing:
        ax.text(0.5, 0.5, f'Missing columns:\n{missing}', ha='center', va='center',
               fontsize=10, transform=ax.transAxes)
        ax.text(-0.12, 1.05, 'B', transform=ax.transAxes,
               fontsize=14, fontweight='bold')
        return
    
    # Plot by cluster
    clusters = sorted(pc_df['cluster'].unique())
    
    for cluster in clusters:
        mask = pc_df['cluster'] == cluster
        color = config.cluster_colors.get(cluster, config.config.colors.gray)
        
        ax.scatter(pc_df.loc[mask, 'PC3'], 
                  pc_df.loc[mask, 'PC4'],
                  c=[color], s=40, alpha=0.7, 
                  edgecolors='white', linewidth=0.3,
                  label=f'Cluster {cluster + 1} (n={mask.sum()})')
    
    ax.set_xlabel('PC3', fontsize=10, fontweight='bold')
    ax.set_ylabel('PC4', fontsize=10, fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3, linestyle='--')
    
    ax.legend(loc='best', fontsize=8, framealpha=0.9)
    
    ax.text(-0.12, 1.05, 'B', transform=ax.transAxes,
           fontsize=14, fontweight='bold')


# ============================================================================
# PANEL C: Trait Distribution by Genetic Cluster
# ============================================================================

def create_panel_C(data, ax):
    """
    Panel C: Trait distributions stratified by genetic cluster.
    Shows violin plots of each trait split by Cluster 1, 2, 3.
    """
    pheno_df = data['pheno_core']
    pc_df = data['pc_scores']
    
    if pheno_df is None or pc_df is None:
        ax.text(0.5, 0.5, 'Data not available', ha='center', va='center',
               fontsize=12, transform=ax.transAxes)
        ax.text(-0.12, 1.05, 'C', transform=ax.transAxes,
               fontsize=14, fontweight='bold')
        return
    
    if 'cluster' not in pc_df.columns:
        ax.text(0.5, 0.5, 'Cluster data\nnot available', ha='center', va='center',
               fontsize=10, transform=ax.transAxes)
        ax.text(-0.12, 1.05, 'C', transform=ax.transAxes,
               fontsize=14, fontweight='bold')
        return
    
    trait_cols = [col for col in config.trait_order if col in pheno_df.columns]
    
    if len(trait_cols) == 0:
        ax.text(0.5, 0.5, 'No trait columns found', ha='center', va='center',
               fontsize=10, transform=ax.transAxes)
        ax.text(-0.12, 1.05, 'C', transform=ax.transAxes,
               fontsize=14, fontweight='bold')
        return
    
    pheno_df = pheno_df.copy()
    pc_df = pc_df.copy()
    
    if pheno_df.index.name != pc_df.index.name:
        if 'sample_id' in pc_df.columns:
            pc_df = pc_df.set_index('sample_id')
    
    merged = pheno_df.join(pc_df[['cluster']], how='inner')
    
    if merged.empty or 'cluster' not in merged.columns:
        ax.text(0.5, 0.5, 'Cannot merge\npheno + cluster', ha='center', va='center',
               fontsize=10, transform=ax.transAxes)
        ax.text(-0.12, 1.05, 'C', transform=ax.transAxes,
               fontsize=14, fontweight='bold')
        return
    
    clusters = sorted(merged['cluster'].dropna().unique())
    n_clusters = len(clusters)
    
    positions = []
    trait_centers = []
    width = 0.25
    
    for i, trait in enumerate(trait_cols):
        center = i
        trait_centers.append(center)
        for j, cluster in enumerate(clusters):
            offset = (j - (n_clusters - 1) / 2) * width
            positions.append(center + offset)
    
    violin_data = []
    violin_positions = []
    violin_colors = []
    
    cluster_color_map = getattr(config, 'cluster_colors', {
        0: '#1f77b4',
        1: '#2ca02c',
        2: '#ff7f0e',
    })
    
    pos_idx = 0
    for i, trait in enumerate(trait_cols):
        for j, cluster in enumerate(clusters):
            mask = merged['cluster'] == cluster
            values = merged.loc[mask, trait].dropna().values
            
            if len(values) > 0:
                trait_mean = merged[trait].mean()
                trait_std = merged[trait].std()
                if trait_std > 0:
                    values_std = (values - trait_mean) / trait_std
                else:
                    values_std = values - trait_mean
                
                violin_data.append(values_std)
                violin_positions.append(positions[pos_idx])
                violin_colors.append(cluster_color_map.get(cluster, '#888888'))
            
            pos_idx += 1
    
    if violin_data:
        parts = ax.violinplot(violin_data, positions=violin_positions, 
                              widths=width * 0.9, showmeans=False, 
                              showmedians=True, showextrema=False)
        
        for idx, pc in enumerate(parts['bodies']):
            pc.set_facecolor(violin_colors[idx])
            pc.set_edgecolor('white')
            pc.set_alpha(0.7)
            pc.set_linewidth(0.5)
        
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(1.5)
    
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    
    ax.set_xticks(trait_centers)
    trait_names_short = getattr(config, 'trait_names_short', {})
    trait_labels = [trait_names_short.get(t, t) for t in trait_cols]
    ax.set_xticklabels(trait_labels, fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Standardized Value (z-score)', fontsize=10, fontweight='bold')
    
    legend_handles = []
    for cluster in clusters:
        color = cluster_color_map.get(cluster, '#888888')
        n_samples = (merged['cluster'] == cluster).sum()
        patch = mpatches.Patch(color=color, alpha=0.7, 
                               label=f'Cluster {cluster + 1} (n={n_samples})')
        legend_handles.append(patch)
    
    ax.legend(handles=legend_handles, loc='upper right', fontsize=8, 
              framealpha=0.9, title='Genetic Cluster', title_fontsize=8)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    ax.text(-0.12, 1.05, 'C', transform=ax.transAxes,
           fontsize=14, fontweight='bold')
    
    source_out = os.path.join(FIGURE_SUBDIR, "panel_C_trait_by_cluster.csv")
    merged[trait_cols + ['cluster']].to_csv(source_out)


# ============================================================================
# PANEL D: PC3/PC4 Substructure within Cluster 1 (Americas vs South Asia)
# ============================================================================

def create_panel_D(data, ax):
    """
    Panel D: PC3 vs PC4 for Cluster 1 only, colored by Origin_Type.
    Shows substructure separating Americas vs South Asia within the admixed cluster.
    """
    pc_df = data.get('pc_scores')
    origin_df = data.get('origin')
    
    if pc_df is None or origin_df is None:
        ax.text(0.5, 0.5, 'Data not available', ha='center', va='center',
               fontsize=10, transform=ax.transAxes)
        ax.text(-0.12, 1.05, 'D', transform=ax.transAxes,
               fontsize=14, fontweight='bold')
        return
    
    # Prepare for merge
    pc_df = pc_df.copy()
    origin_df = origin_df.copy()
    
    # Standardise sample ID columns
    if pc_df.index.name == 'sample_id':
        pc_df = pc_df.reset_index()
    if 'Sample_ID' in origin_df.columns:
        origin_df = origin_df.rename(columns={'Sample_ID': 'sample_id'})
    
    pc_df['sample_id'] = pc_df['sample_id'].astype(str)
    origin_df['sample_id'] = origin_df['sample_id'].astype(str)
    
    # Merge
    merged = pc_df.merge(origin_df[['sample_id', 'Origin_Type']], on='sample_id', how='inner')
    
    # Filter to Cluster 1 and Americas/South Asia
    cluster_of_interest = 1
    group_a, group_b = "Americas", "South Asia"
    
    sub = merged[merged['cluster'] == cluster_of_interest].copy()
    sub = sub[sub['Origin_Type'].isin([group_a, group_b])].copy()
    
    if sub.shape[0] < 4 or 'PC3' not in sub.columns or 'PC4' not in sub.columns:
        ax.text(0.5, 0.5, 'Insufficient data\nfor PC3/PC4', ha='center', va='center',
               fontsize=10, transform=ax.transAxes)
        ax.text(-0.12, 1.05, 'D', transform=ax.transAxes,
               fontsize=14, fontweight='bold')
        return
    
    # Colors for origin types
    origin_colors = {
        'Americas': config.config.colors.royalblue,
        'South Asia': config.config.colors.mediumseagreen,
    }
    
    # Plot by Origin_Type
    for origin_type in [group_a, group_b]:
        mask = sub['Origin_Type'] == origin_type
        n_samples = mask.sum()
        color = origin_colors.get(origin_type, '#888888')
        
        ax.scatter(sub.loc[mask, 'PC3'], 
                  sub.loc[mask, 'PC4'],
                  c=[color], s=50, alpha=0.7, 
                  edgecolors='white', linewidth=0.5,
                  label=f'{origin_type} (n={n_samples})')
    
    # Calculate and display Cohen's d
    A = sub[sub['Origin_Type'] == group_a]
    B = sub[sub['Origin_Type'] == group_b]
    
    def cohens_d(x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        nx, ny = len(x), len(y)
        if nx < 2 or ny < 2:
            return np.nan
        sx, sy = x.std(ddof=1), y.std(ddof=1)
        sp = np.sqrt(((nx - 1) * sx**2 + (ny - 1) * sy**2) / (nx + ny - 2))
        return abs((x.mean() - y.mean()) / sp) if sp > 0 else np.nan
    
    d_pc3 = cohens_d(A['PC3'].values, B['PC3'].values)
    d_pc4 = cohens_d(A['PC4'].values, B['PC4'].values)
    
    # Add effect size annotation
    stats_text = f"|d| PC3 = {d_pc3:.2f}\n|d| PC4 = {d_pc4:.2f}"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
           fontsize=9, ha='left', va='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('PC3', fontsize=10, fontweight='bold')
    ax.set_ylabel('PC4', fontsize=10, fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3, linestyle='--')
    
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9,
              title='Origin (Cluster 1)', title_fontsize=8)
    
    ax.text(-0.12, 1.05, 'D', transform=ax.transAxes,
           fontsize=14, fontweight='bold')
    
    # Save source data
    source_out = os.path.join(FIGURE_SUBDIR, "panel_D_pc34_origin.csv")
    sub[['sample_id', 'Origin_Type', 'PC3', 'PC4']].to_csv(source_out, index=False)


# ============================================================================
# MAIN FIGURE ASSEMBLY
# ============================================================================

def create_figure_s1():
    """
    Assemble all 4 panels into Figure S1.
    
    Layout:
      Row 0: Panel A - 5 trait histograms
      Row 1: Panel B (PC3/PC4 by cluster) + Panel D (PC3/PC4 Cluster 1 by origin)
      Row 2: Panel C - Trait distributions by cluster (violins)
    """
    print("Assembling Figure S1...")
    
    data = load_data()
    
    fig = plt.figure(figsize=(14, 12))
    
    # Create GridSpec: 3 rows
    gs = gridspec.GridSpec(
        nrows=3, ncols=5,
        height_ratios=[1.0, 1.2, 1.2],
        hspace=0.30,
        wspace=0.35
    )
    
    # Row 0: Panel A - 5 histograms
    axes_A = [fig.add_subplot(gs[0, i]) for i in range(5)]
    
    # Row 1: Panel B (left 2.5 cols) + Panel D (right 2.5 cols)
    ax_B = fig.add_subplot(gs[1, 0:3])
    ax_D = fig.add_subplot(gs[1, 3:5])
    
    # Row 2: Panel C (full width)
    ax_C = fig.add_subplot(gs[2, :])
    
    # Create panels
    create_panel_A(data, axes_A)
    create_panel_B(data, ax_B)
    create_panel_D(data, ax_D)
    create_panel_C(data, ax_C)
    
    # Save
    output_path = os.path.join(FIGURES_DIR, "figure_s1.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure saved: {output_path}")
    
    output_pdf = os.path.join(FIGURES_DIR, "figure_s1.pdf")
    fig.savefig(output_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    
    # Save source data
    for key, df in data.items():
        if df is not None:
            out_path = os.path.join(FIGURE_SUBDIR, f"source_{key}.csv")
            df.to_csv(out_path, index=False)
    
    return fig


if __name__ == "__main__":
    try:
        fig = create_figure_s1()
        print("Done.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)