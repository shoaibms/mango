#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure S5: Deep Learning Saliency and GWAS Concordance
=======================================================

Theme: Extended interpretability views showing how deep learning
       models identify the same important features as GWAS.

Panels:
    A: SHAP beeswarm plot for FBC (embed existing image)
    B: Top 20 SNPs by SHAP importance (horizontal bar chart)
    C: AI-GWAS concordance summary (correlation between saliency and -log10(p))
    D: Saliency concentration summary

"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from PIL import Image
import warnings

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: shap package not available. Panel A will use fallback visualization.")

# Import configuration
try:
    import figure_config
    config = figure_config.config
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import figure_config
    config = figure_config.config

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
OUTPUT_DIR = config.paths.OUTPUT_DIR
FIGURES_DIR = config.paths.FIGURES_DIR
FIGURE_SUBDIR = config.paths.figure_subdir('s5')

# Data paths
DATA_PATHS = {
    'shap_top_snps': config.data.SHAP_TOP_SNPS,
    'concordance_summary': config.data.CONCORDANCE_SUMMARY,
    'merged_fbc': config.data.AI_GWAS_FBC,
    'saliency_summary': config.data.SALIENCY_SUMMARY
}

# SHAP data paths (for generating plot directly)
SHAP_DATA_PATHS = {
    'X_background': os.path.join(config.paths.IDEA3_DIR, "tensors", "X_background.npy"),
    'feature_map': os.path.join(config.paths.IDEA3_DIR, "tensors", "feature_map.tsv"),
    'shap_values': os.path.join(config.paths.IDEA3_DIR, "interpretation", "shap", "SHAP_Values_FBC.npy"),
}

# Panel A configuration
PANEL_A_CMAP = "BuGn"  # Colormap for SHAP beeswarm plot
PANEL_A_MAX_DISPLAY = 20  # Number of top features to display

# Ensure output directories exist
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(FIGURE_SUBDIR, exist_ok=True)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load all data for Figure S5."""
    print("Loading data for Figure S5...")
    
    data = {}
    
    # Load standard CSV data
    for key, path in DATA_PATHS.items():
        if path and os.path.exists(path):
            data[key] = pd.read_csv(path)
        else:
            print(f"Warning: File not found: {path}")
            data[key] = None
    
    # Load SHAP data for Panel A
    shap_data = {}
    for key, path in SHAP_DATA_PATHS.items():
        if path and os.path.exists(path):
            if path.endswith('.npy'):
                shap_data[key] = np.load(path)
            elif path.endswith('.tsv'):
                shap_data[key] = pd.read_csv(path, sep='\t')
            else:
                shap_data[key] = pd.read_csv(path)
        else:
            print(f"Warning: SHAP data not found: {path}")
            shap_data[key] = None
    
    data['shap_data'] = shap_data
    
    return data

# ============================================================================
# PANEL A: SHAP Beeswarm Plot (Generated Directly)
# ============================================================================

def create_panel_A(data, ax):
    """
    Panel A: Generate SHAP beeswarm plot directly with configurable colormap.
    """
    shap_data = data.get('shap_data', {})
    shap_values = shap_data.get('shap_values')
    X_background = shap_data.get('X_background')
    feature_map = shap_data.get('feature_map')
    
    # Check if we have necessary data
    if shap_values is None or X_background is None:
        # Fallback: try to load pre-generated image
        fallback_path = os.path.join(config.paths.IDEA3_DIR, "interpretation", "shap", "SHAP_Summary_FBC.png")
        if os.path.exists(fallback_path):
            img = Image.open(fallback_path)
            ax.imshow(img)
            ax.axis('off')
            config.add_panel_label(ax, 'A')
            return
        else:
            ax.text(0.5, 0.5, 'SHAP data not available', ha='center', va='center',
                   fontsize=12, transform=ax.transAxes)
            ax.axis('off')
            config.add_panel_label(ax, 'A')
            return
    
    # Squeeze SHAP values if needed
    if shap_values.ndim > 2:
        shap_values = np.squeeze(shap_values)
    
    # Get feature names
    if feature_map is not None and 'snp_id' in feature_map.columns:
        feature_names = feature_map['snp_id'].values
    else:
        feature_names = [f"SNP_{i}" for i in range(shap_values.shape[1])]
    
    # Generate beeswarm plot directly
    _create_beeswarm_plot(ax, shap_values, X_background, feature_names, 
                          cmap=PANEL_A_CMAP, max_display=PANEL_A_MAX_DISPLAY)
    
    config.add_panel_label(ax, 'A')


def _create_beeswarm_plot(ax, shap_values, features, feature_names, cmap="BuGn", max_display=20):
    """
    Create a custom beeswarm-style SHAP summary plot with configurable colormap.
    
    Parameters:
        ax: matplotlib axis
        shap_values: array of shape (n_samples, n_features)
        features: array of shape (n_samples, n_features) - feature values
        feature_names: list of feature names
        cmap: colormap name (e.g., 'BuGn', 'RdBu', 'coolwarm')
        max_display: number of top features to display
    """
    # Calculate mean absolute SHAP for ranking
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    
    # Get top features by importance
    top_indices = np.argsort(-mean_abs_shap)[:max_display]
    
    # Prepare data for plotting
    n_samples = shap_values.shape[0]
    
    # Ensure features match shap_values sample count
    if features.shape[0] != n_samples:
        features = features[:n_samples]
    
    # Get colormap
    colormap = plt.get_cmap(cmap)
    
    # For each feature, create scattered points
    y_positions = []
    x_values = []
    colors = []
    
    for i, idx in enumerate(top_indices[::-1]):  # Reverse so top is at top
        shap_vals = shap_values[:, idx]
        feat_vals = features[:, idx]
        
        # Normalize feature values for coloring
        feat_min, feat_max = np.nanmin(feat_vals), np.nanmax(feat_vals)
        if feat_max > feat_min:
            feat_norm = (feat_vals - feat_min) / (feat_max - feat_min)
        else:
            feat_norm = np.full_like(feat_vals, 0.5)
        
        # Add jitter to y positions to create beeswarm effect
        y_jitter = np.random.normal(0, 0.15, n_samples)
        y_pos = i + y_jitter
        
        y_positions.extend(y_pos)
        x_values.extend(shap_vals)
        colors.extend(feat_norm)
    
    # Plot
    scatter = ax.scatter(x_values, y_positions, c=colors, cmap=colormap,
                        s=8, alpha=0.7, edgecolors='none')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=30, pad=0.02)
    cbar.set_label('Feature Value', fontsize=9)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Low', 'High'])
    
    # Set y-axis labels (shortened SNP IDs)
    y_labels = []
    for idx in top_indices[::-1]:
        snp_name = str(feature_names[idx])
        if len(snp_name) > 25:
            # Shorten long SNP names
            if ':' in snp_name:
                parts = snp_name.split(':')
                snp_name = f"{parts[0][-6:]}:{parts[1][:8]}" if len(parts) > 1 else snp_name[:25]
            else:
                snp_name = snp_name[:25]
        y_labels.append(snp_name)
    
    ax.set_yticks(range(max_display))
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.set_xlabel('SHAP Value (impact on model output)', fontsize=10, fontweight='bold')
    
    # Add vertical line at 0
    ax.axvline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Style
    config.style_axis(ax, spines=['bottom', 'left'], grid=False)
    ax.set_ylim(-0.5, max_display - 0.5)

# ============================================================================
# PANEL B: Top SNPs by SHAP Importance
# ============================================================================

def create_panel_B(data, ax):
    """
    Panel B: Horizontal bar chart of top 20 SNPs by mean |SHAP|.
    """
    shap_df = data.get('shap_top_snps')
    
    if shap_df is None:
        ax.text(0.5, 0.5, 'SHAP data not available', ha='center', va='center',
               fontsize=10, transform=ax.transAxes)
        config.add_panel_label(ax, 'B')
        return
    
    # Get top 20 SNPs
    top_n = 20
    if 'mean_abs_shap' in shap_df.columns:
        shap_col = 'mean_abs_shap'
    else:
        # Try to find the SHAP value column
        shap_col = [c for c in shap_df.columns if 'shap' in c.lower()]
        if shap_col:
            shap_col = shap_col[0]
        else:
            ax.text(0.5, 0.5, 'Cannot find SHAP column', ha='center', va='center')
            config.add_panel_label(ax, 'B')
            return
    
    top_df = shap_df.head(top_n).copy()
    
    # Shorten SNP IDs for display
    if 'snp_id' in top_df.columns:
        snp_col = 'snp_id'
    else:
        snp_col = top_df.columns[1]  # Assume second column is SNP ID
    
    # Clean SNP names (shorten long IDs)
    def shorten_snp(snp):
        snp = str(snp)
        if ':' in snp:
            parts = snp.split(':')
            chrom = parts[0].replace('NC_', '').replace('.1', '')
            pos = parts[1] if len(parts) > 1 else ''
            # Try to make chromosome number readable
            try:
                chrom_num = int(chrom.split('058')[-1]) - 140 + 1 if '058' in chrom else chrom
                return f"Chr{chrom_num}:{pos[:6]}"
            except:
                return snp[:20]
        return snp[:20]
    
    top_df['snp_short'] = top_df[snp_col].apply(shorten_snp)
    
    # Create horizontal bar chart (reversed so #1 is at top)
    y_pos = np.arange(top_n)
    values = top_df[shap_col].values[::-1]  # Reverse for top-to-bottom
    labels = top_df['snp_short'].values[::-1]
    
    # Color gradient based on rank - using BuGn as requested
    cmap = plt.cm.BuGn
    norm_vals = np.linspace(0.4, 1.0, top_n)[::-1]
    bar_colors = [cmap(v) for v in norm_vals]
    
    bars = ax.barh(y_pos, values, color=bar_colors, edgecolor='white', height=0.7)
    
    # Value labels
    for bar, val in zip(bars, values):
        ax.text(val + max(values)*0.02, bar.get_y() + bar.get_height()/2,
               f'{val:.4f}', va='center', fontsize=7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Mean |SHAP|', fontsize=10, fontweight='bold')
    
    config.style_axis(ax, spines=['bottom', 'left'], grid=True)
    ax.grid(axis='y', alpha=0.0) # Only x grid
    
    config.add_panel_label(ax, 'B')

# ============================================================================
# PANEL C: AI-GWAS Concordance Summary
# ============================================================================

def create_panel_C(data, ax):
    """
    Panel C: Bar chart showing correlation between AI saliency and GWAS
             for each trait.
    """
    conc_df = data.get('concordance_summary')
    
    if conc_df is None:
        ax.text(0.5, 0.5, 'Concordance data not available', ha='center', va='center',
               fontsize=10, transform=ax.transAxes)
        config.add_panel_label(ax, 'C')
        return
    
    # The actual column name from the data
    corr_col = None
    # Map config names or fallback
    potential_cols = [config.cols.CONCORDANCE_SUMMARY.get('spearman_logp'),
                     'spearman_sal_vs_neglog10p', 'pearson_sal_vs_neglog10p', 
                     'spearman_rho', 'pearson_r']
    
    for col in potential_cols:
        if col and col in conc_df.columns:
            corr_col = col
            break
            
    if corr_col is None:
        ax.text(0.5, 0.5, 'Cannot find correlation column', ha='center', va='center')
        config.add_panel_label(ax, 'C')
        return
    
    # Get trait column
    trait_col = 'trait' if 'trait' in conc_df.columns else conc_df.columns[0]
    
    # Order by standard trait order from config
    ordered_traits = [t for t in config.trait_order if t in conc_df[trait_col].values]
    
    plot_df = conc_df.set_index(trait_col).loc[ordered_traits].reset_index()
    
    x = np.arange(len(plot_df))
    correlations = plot_df[corr_col].values
    
    # Color bars by trait
    bar_colors = [config.trait_colors.get(t, config.colors.gray) for t in plot_df[trait_col]]
    
    bars = ax.bar(x, correlations, color=bar_colors, edgecolor='white', width=0.6)
    
    # Value labels
    for bar, val in zip(bars, correlations):
        y_pos = val + 0.02 if val >= 0 else val - 0.05
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, f'{val:.3f}',
               ha='center', va='bottom' if val >= 0 else 'top', fontsize=9, fontweight='bold')
    
    # Reference line
    ax.axhline(0, color='black', linewidth=1, alpha=0.3)
    
    # Styling
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df[trait_col], fontsize=10, fontweight='bold')
    ax.set_ylabel('Spearman ρ\n(Saliency vs -log₁₀p)', fontsize=10, fontweight='bold')
    
    # Set y limits
    if len(correlations) > 0 and np.any(np.isfinite(correlations)):
        max_abs = max(abs(np.nanmin(correlations)), abs(np.nanmax(correlations)))
        ax.set_ylim(-max(max_abs * 1.3, 0.1), max(max_abs * 1.3, 0.5))
    
    config.style_axis(ax, spines=['bottom', 'left'], grid=True)
    ax.grid(axis='x', alpha=0.0)
    
    config.add_panel_label(ax, 'C')

# ============================================================================
# PANEL D: Saliency Concentration
# ============================================================================

def create_panel_D(data, ax):
    """
    Panel D: Saliency concentration - shows what fraction of total saliency
             is captured by top 1%, 10%, 100 SNPs per trait.
    """
    sal_summary = data.get('saliency_summary')
    
    if sal_summary is None or len(sal_summary) == 0:
        # Fallback: compute from merged data if available
        merged_df = data.get('merged_fbc')
        if merged_df is not None:
            create_saliency_concentration_from_merged(merged_df, ax)
        else:
            ax.text(0.5, 0.5, 'Saliency data not available', ha='center', va='center',
                   fontsize=10, transform=ax.transAxes)
        config.add_panel_label(ax, 'D')
        return
    
    # Check for concentration columns
    share_cols = [c for c in sal_summary.columns if 'share' in c.lower()]
    
    if not share_cols:
        ax.text(0.5, 0.5, 'Cannot find concentration data', ha='center', va='center')
        config.add_panel_label(ax, 'D')
        return
    
    # Create grouped bar chart of concentration
    trait_col = 'trait'
    
    # Filter and sort traits based on config order
    available_traits = set(sal_summary[trait_col].values)
    ordered_traits = [t for t in config.trait_order if t in available_traits]
    
    if not ordered_traits:
         ordered_traits = list(available_traits)

    # Re-index dataframe to match order
    sal_summary_ordered = sal_summary.set_index(trait_col).loc[ordered_traits].reset_index()

    # Get share values - use top1, top10, top100 if available
    share_data = {}
    for col in share_cols:
        if 'top1' in col.lower() and '10' not in col.lower():
            share_data['Top 1%'] = sal_summary_ordered[col].values
        elif 'top10' in col.lower() and '100' not in col.lower():
            share_data['Top 10%'] = sal_summary_ordered[col].values
        elif 'top100' in col.lower():
            share_data['Top 100'] = sal_summary_ordered[col].values
    
    if not share_data:
        # Use whatever share columns are available
        for col in share_cols[:3]:
            share_data[col] = sal_summary_ordered[col].values
    
    # Create grouped bar chart
    x = np.arange(len(ordered_traits))
    width = 0.25
    multiplier = 0
    
    # Use config colors for the groups
    bar_colors = [config.colors.limegreen, config.colors.steelblue, config.colors.turquoise]
    
    for i, (label, values) in enumerate(share_data.items()):
        offset = width * multiplier
        bars = ax.bar(x + offset, values * 100, width, label=label, 
                     color=bar_colors[i % len(bar_colors)], edgecolor='white')
        multiplier += 1
    
    # Styling
    ax.set_xticks(x + width * (len(share_data) - 1) / 2)
    ax.set_xticklabels(ordered_traits, fontsize=10, fontweight='bold')
    ax.set_ylabel('% of Total Saliency', fontsize=10, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, frameon=False)
    
    config.style_axis(ax, spines=['bottom', 'left'], grid=True)
    ax.grid(axis='x', alpha=0.0)
    
    config.add_panel_label(ax, 'D')

def create_saliency_concentration_from_merged(merged_df, ax):
    """Compute saliency concentration from merged data."""
    sal_col = None
    for col in ['saliency_FBC_norm', 'saliency_FBC', 'saliency_FBC_raw']:
        if col in merged_df.columns:
            sal_col = col
            break
    
    if sal_col is None:
        ax.text(0.5, 0.5, 'No saliency column', ha='center', va='center')
        return
    
    # Sort by saliency
    sorted_sal = merged_df[sal_col].dropna().sort_values(ascending=False).values
    n_snps = len(sorted_sal)
    total = sorted_sal.sum()
    
    # Calculate cumulative share
    cumsum = np.cumsum(sorted_sal) / total
    x_pct = np.arange(1, n_snps + 1) / n_snps * 100
    
    ax.plot(x_pct, cumsum * 100, color=config.trait_colors['FBC'], linewidth=2)
    ax.fill_between(x_pct, 0, cumsum * 100, alpha=0.3, color=config.trait_colors['FBC'])
    
    # Reference line (uniform)
    ax.plot([0, 100], [0, 100], 'k--', linewidth=1, alpha=0.5, label='Uniform')
    
    # Annotate key points
    for pct in [1, 10]:
        idx = int(n_snps * pct / 100)
        share = cumsum[idx] * 100 if idx < len(cumsum) else 100
        ax.axvline(pct, color='gray', linestyle=':', alpha=0.5)
        ax.text(pct + 1, share, f'{share:.0f}%', fontsize=8, va='bottom')
    
    ax.set_xlabel('% of SNPs (ranked by saliency)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Cumulative % of Saliency', fontsize=10, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    
    config.style_axis(ax, spines=['bottom', 'left'], grid=True)
    
    config.add_panel_label(ax, 'D')

# ============================================================================
# MAIN FIGURE ASSEMBLY
# ============================================================================

def create_figure_s5():
    """
    Assemble Figure S5: Deep Learning Saliency and GWAS Concordance.
    """
    print("Assembling Figure S5...")
    
    data = load_data()
    
    # Reduce size slightly as requested to avoid empty space
    fig = plt.figure(figsize=(14, 11))
    
    # Adjusted gridspec with less hspace
    gs = gridspec.GridSpec(
        nrows=2, ncols=2,
        height_ratios=[1.2, 1.0],
        hspace=0.25, # Reduced from 0.3
        wspace=0.3
    )
    
    # Create panels
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[1, 0])
    ax_D = fig.add_subplot(gs[1, 1])
    
    # Panel A: SHAP beeswarm
    create_panel_A(data, ax_A)
    
    # Panel B: Top SNPs by SHAP
    create_panel_B(data, ax_B)
    
    # Panel C: AI-GWAS concordance
    create_panel_C(data, ax_C)
    
    # Panel D: Saliency vs GWAS scatter
    create_panel_D(data, ax_D)
    
    # Save
    output_path = os.path.join(FIGURES_DIR, "figure_s5.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure saved: {output_path}")
    
    output_pdf = os.path.join(FIGURES_DIR, "figure_s5.pdf")
    fig.savefig(output_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"PDF saved: {output_pdf}")
    
    return fig

if __name__ == "__main__":
    try:
        fig = create_figure_s5()
        print("Success!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
