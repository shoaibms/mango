#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 3: Deep Learning Supports Predominantly Additive Supergene Effects
==========================================================================

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# FIGURE SIZE - EASY TO CHANGE
# ============================================================================
FIGURE_WIDTH = 11
FIGURE_HEIGHT = 8.25

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = r"C:\Users\ms\Desktop\mango"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
FIGURE_SUBDIR = os.path.join(FIGURES_DIR, "figure_3")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(FIGURE_SUBDIR, exist_ok=True)

DATA_PATHS = {
    'model_perf': os.path.join(OUTPUT_DIR, "idea_3", "metrics", "idea3_model_performance_summary.csv"),
    'ai_gwas_FBC': os.path.join(OUTPUT_DIR, "idea_3", "interpretation", "ai_vs_gwas", "ai_gwas_merged_trait-FBC.csv"),
    'concordance': os.path.join(OUTPUT_DIR, "idea_3", "interpretation", "ai_vs_gwas", "ai_gwas_concordance_summary.csv"),
    'block_synergy': os.path.join(OUTPUT_DIR, "idea_3", "interpretation", "editing", "advanced", "haplotype_block_synergy.csv"),
    'editing_tradeoffs': os.path.join(OUTPUT_DIR, "idea_3", "interpretation", "idea3_editing_tradeoff_summary.csv"),
}

# ============================================================================
# COLOURS - FROM figure_config.py
# ============================================================================

TRAIT_COLOURS = {
    'FBC': '#3CB371',
    'AFW': '#4169E1',
    'FF': '#40E0D0',
    'TC': '#4682B4',
    'TSS': '#32CD32',
}

MODEL_COLOURS = {
    'Wide&Deep': '#94CB64',
    'wide_deep': '#94CB64',
    'CNN': '#2E8B57',
    'cnn': '#2E8B57',
    'Ridge': '#4682B4',
    'ridge': '#4682B4',
    'XGBoost': '#40E0D0',
    'xgb': '#40E0D0',
    'RF': '#8FBC8F',
    'rf': '#8FBC8F',
}

COLOUR_SUM_SINGLES = '#4682B4'
COLOUR_BLOCK = '#32CD32'
COLOUR_RAW = '#3CB371'
COLOUR_RESIDUALISED = '#D3D3D3'

# Background zone colours (matching Figures 1 & 2)
ZONE_GREEN = '#E8F5E9'
ZONE_GRAY = '#F5F5F5'
ZONE_GREEN_DARK = '#C8E6C9'

GRAY = '#808080'
LIGHTGRAY = '#D3D3D3'
DARKGRAY = '#A9A9A9'

TRAIT_ORDER = ['FBC', 'AFW', 'FF', 'TC', 'TSS']

# ============================================================================
# STYLING
# ============================================================================

FONTSIZE_LABEL = 11
FONTSIZE_TICK = 9
FONTSIZE_PANEL_LETTER = 16
FONTSIZE_ANNOTATION = 10
DPI = 300

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': FONTSIZE_TICK,
    'axes.labelsize': FONTSIZE_LABEL,
    'xtick.labelsize': FONTSIZE_TICK,
    'ytick.labelsize': FONTSIZE_TICK,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.2,
})

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    print("=" * 60)
    print("Loading data for Figure 3")
    print("=" * 60)
    data = {}
    for key, path in DATA_PATHS.items():
        if os.path.exists(path):
            data[key] = pd.read_csv(path)
            print(f"[OK] {key}: {data[key].shape}")
        else:
            print(f"[MISSING] {key}")
            data[key] = None
    return data

# ============================================================================
# PANEL A: Model Performance (Wide)
# ============================================================================

def create_panel_A(data, ax):
    """Panel A: Model performance with background zones."""
    
    perf_df = data['model_perf']
    if perf_df is None:
        ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', 
                transform=ax.transAxes, fontsize=11)
        ax.text(-0.08, 1.05, 'A', transform=ax.transAxes, 
                fontsize=FONTSIZE_PANEL_LETTER, fontweight='bold')
        return
    
    if 'trait_group' in perf_df.columns:
        perf_df = perf_df[perf_df['trait_group'] == 'raw'].copy()
    
    r_col = 'mean_r' if 'mean_r' in perf_df.columns else 'pearson_r_mean'
    
    model_map = {
        'wide_deep': 'Wide&Deep', 'WideDeep': 'Wide&Deep',
        'cnn': 'CNN', 'shallow_cnn': 'CNN',
        'ridge': 'Ridge', 'xgb': 'XGBoost', 'rf': 'RF',
    }
    
    perf_df['model_display'] = perf_df['model'].map(lambda x: model_map.get(x, x))
    summary = perf_df.groupby(['trait', 'model_display'])[r_col].max().reset_index()
    summary = summary[summary['trait'].isin(TRAIT_ORDER)]
    
    models_present = [m for m in ['Wide&Deep', 'CNN', 'Ridge', 'XGBoost', 'RF'] 
                      if m in summary['model_display'].unique()]
    traits_present = [t for t in TRAIT_ORDER if t in summary['trait'].unique()]
    
    # Background zones
    ax.axhspan(0.5, 1.0, color=ZONE_GREEN, alpha=0.5, zorder=0)
    ax.axhspan(0.3, 0.5, color=ZONE_GREEN, alpha=0.25, zorder=0)
    ax.axhspan(0, 0.3, color=ZONE_GRAY, alpha=0.5, zorder=0)
    
    # Zone labels on right side only
    ax.text(len(traits_present) - 0.3, 0.75, 'High', fontsize=8, fontstyle='italic', 
            color='#2E7D32', ha='right', va='center')
    ax.text(len(traits_present) - 0.3, 0.4, 'Moderate', fontsize=8, fontstyle='italic', 
            color='#558B2F', ha='right', va='center')
    ax.text(len(traits_present) - 0.3, 0.15, 'Low', fontsize=8, fontstyle='italic', 
            color=GRAY, ha='right', va='center')
    
    x = np.arange(len(traits_present))
    n_models = len(models_present)
    width = 0.75 / max(n_models, 1)
    
    for i, model in enumerate(models_present):
        model_data = summary[summary['model_display'] == model]
        values = [model_data[model_data['trait'] == t][r_col].values[0] 
                  if len(model_data[model_data['trait'] == t]) > 0 else np.nan 
                  for t in traits_present]
        
        offset = (i - n_models/2 + 0.5) * width
        colour = MODEL_COLOURS.get(model, GRAY)
        bars = ax.bar(x + offset, values, width * 0.9, label=model, 
                      color=colour, edgecolor='white', linewidth=1, zorder=3)
        
        for bar, val in zip(bars, values):
            if np.isfinite(val) and val > 0.05:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(traits_present, fontsize=FONTSIZE_LABEL, fontweight='bold')
    ax.set_ylabel('Prediction Accuracy (r)', fontsize=FONTSIZE_LABEL, fontweight='bold')
    ax.set_ylim(0, 0.95)
    ax.set_xlim(-0.5, len(traits_present) - 0.3)
    ax.legend(
        title='Model',
        loc='upper center',
        bbox_to_anchor=(0.5, 0.94),
        fontsize=8,
        ncol=2,
        framealpha=0.95
    )
    ax.text(-0.08, 1.05, 'A', transform=ax.transAxes, 
            fontsize=FONTSIZE_PANEL_LETTER, fontweight='bold')

# ============================================================================
# PANEL B: AI-GWAS Concordance
# ============================================================================

def create_panel_B(data, ax):
    """Panel B: AI saliency vs GWAS significance."""
    
    ai_gwas = data['ai_gwas_FBC']
    if ai_gwas is None:
        ax.text(0.5, 0.5, 'Data not available', ha='center', va='center',
                transform=ax.transAxes, fontsize=11)
        ax.text(-0.12, 1.05, 'B', transform=ax.transAxes,
                fontsize=FONTSIZE_PANEL_LETTER, fontweight='bold')
        return
    
    saliency_col = 'saliency_FBC_norm' if 'saliency_FBC_norm' in ai_gwas.columns else None
    neglog_col = 'neglog10p'
    
    if neglog_col not in ai_gwas.columns and 'p_FBC' in ai_gwas.columns:
        ai_gwas = ai_gwas.copy()
        ai_gwas['neglog10p'] = -np.log10(ai_gwas['p_FBC'].clip(lower=1e-300))
    
    if saliency_col is None or neglog_col not in ai_gwas.columns:
        return
    
    plot_df = ai_gwas[[saliency_col, neglog_col]].dropna()
    plot_df = plot_df[np.isfinite(plot_df[neglog_col])]
    
    x = plot_df[neglog_col].values
    y = plot_df[saliency_col].values
    
    # Background zone for high saliency
    ax.axhspan(0.5, 1.0, color=ZONE_GREEN, alpha=0.3, zorder=0)
    
    hb = ax.hexbin(x, y, gridsize=35, cmap='Greens', mincnt=1, alpha=0.85, zorder=2)
    cbar = plt.colorbar(hb, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Count', fontsize=8)
    
    from scipy import stats
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() > 10:
        rho, _ = stats.spearmanr(x[mask], y[mask])
        coef = np.polyfit(x[mask], y[mask], 1)
        x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
        y_line = np.polyval(coef, x_line)
        ax.plot(x_line, y_line, color=TRAIT_COLOURS['TC'], linewidth=2.5, linestyle='--', zorder=4)
        
        # Annotation box like Figures 1 & 2
        ax.text(0.05, 0.95, f'rho = {rho:.2f}', transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor=TRAIT_COLOURS['FBC'], linewidth=2))
    
    ax.set_xlabel(r'GWAS Significance (-log$_{10}$p)', fontsize=FONTSIZE_LABEL, fontweight='bold')
    ax.set_ylabel('AI Saliency (normalised)', fontsize=FONTSIZE_LABEL, fontweight='bold')
    ax.text(-0.12, 1.05, 'B', transform=ax.transAxes,
            fontsize=FONTSIZE_PANEL_LETTER, fontweight='bold')

# ============================================================================
# PANEL C: Block Synergy (Dumbbell Plot - Enhanced)
# ============================================================================

def create_panel_C(data, ax):
    """Panel C: Dumbbell plot showing Block effect vs Sum of Singles (additivity test)."""
    
    synergy_df = data['block_synergy']
    if synergy_df is None:
        ax.text(0.5, 0.5, 'Data not available', ha='center', va='center',
                transform=ax.transAxes, fontsize=11)
        ax.text(-0.15, 1.05, 'C', transform=ax.transAxes,
                fontsize=FONTSIZE_PANEL_LETTER, fontweight='bold')
        return
    
    required = ['trait', 'block_effect', 'sum_singles', 'synergy']
    if not all(c in synergy_df.columns for c in required):
        return
    
    plot_df = synergy_df[synergy_df['trait'].isin(TRAIT_ORDER)].copy()
    plot_df['trait'] = pd.Categorical(plot_df['trait'], categories=TRAIT_ORDER, ordered=True)
    plot_df = plot_df.sort_values('trait', ascending=False).dropna(subset=['trait'])
    
    traits = plot_df['trait'].astype(str).values
    block_effects = plot_df['block_effect'].values
    sum_singles = plot_df['sum_singles'].values
    synergy = plot_df['synergy'].values
    
    y = np.arange(len(traits))
    
    # Single colours for categories (not trait-wise)
    BLOCK_COLOUR = '#3CB371'      # mediumseagreen for Block
    SUM_SINGLES_COLOUR = '#4169E1' # royalblue for Sum Singles
    
    # Background zones
    ax.axvspan(-0.02, 0.08, color=ZONE_GRAY, alpha=0.4, zorder=0)
    ax.axvspan(0.08, 0.35, color=ZONE_GREEN, alpha=0.3, zorder=0)
    
    # Zone labels
    ax.text(0.03, len(traits) - 0.3, 'Low\nEffect', fontsize=7, fontstyle='italic',
            color=GRAY, ha='center', va='top')
    ax.text(0.20, len(traits) - 0.3, 'Moderate\nEffect', fontsize=7, fontstyle='italic',
            color='#2E7D32', ha='center', va='top')
    
    # Draw for each trait
    for i, (trait, be, ss, syn) in enumerate(zip(traits, block_effects, sum_singles, synergy)):
        
        # Horizontal line from 0 to Block (lollipop style)
        ax.plot([0, be], [y[i], y[i]], color=BLOCK_COLOUR, linewidth=2, 
                zorder=1, alpha=0.4, linestyle='-')
        
        # Connecting line between Sum Singles and Block
        ax.plot([ss, be], [y[i], y[i]], color=BLOCK_COLOUR, linewidth=3, zorder=2, alpha=0.6)
        
        # Block Effect - BIG diamond (single colour)
        ax.scatter(be, y[i], s=280, color=BLOCK_COLOUR, edgecolor='white', linewidth=2, 
                   zorder=4, marker='D', label='Block' if i == 0 else '')
        
        # Sum Singles - small dot INSIDE/ON TOP of position (single colour)
        ax.scatter(ss, y[i], s=80, color=SUM_SINGLES_COLOUR, edgecolor='white', linewidth=1.5, 
                   zorder=5, marker='o', label='Sum Singles' if i == 0 else '')
        
        # Synergy annotation with box
        max_val = max(ss, be)
        ax.text(max_val + 0.02, y[i], f'D={syn:.2f}', ha='left', va='center',
                fontsize=9, fontweight='bold', color='#2F2F2F',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                          edgecolor=BLOCK_COLOUR, alpha=0.9, linewidth=1))
    
    # Vertical reference line at 0
    ax.axvline(0, color='black', linewidth=1, zorder=1)
    
    ax.set_yticks(y)
    ax.set_yticklabels(traits, fontsize=FONTSIZE_TICK, fontweight='bold')
    ax.set_xlabel('Effect Size', fontsize=FONTSIZE_LABEL, fontweight='bold')
    ax.set_xlim(-0.02, max(max(block_effects), max(sum_singles)) * 1.5)
    
    # Legend with actual colours
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='D', color='w', markerfacecolor=BLOCK_COLOUR, 
               markeredgecolor='white', markersize=12, label='Block'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=SUM_SINGLES_COLOUR, 
               markeredgecolor='white', markersize=8, label='Sum Singles')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8, framealpha=0.95)
    
    ax.text(-0.25, 1.05, 'C', transform=ax.transAxes,
            fontsize=FONTSIZE_PANEL_LETTER, fontweight='bold')

# ============================================================================
# PANEL D: Cross-Trait Trade-offs (Heatmap)
# ============================================================================

def create_panel_D(data, ax):
    """Panel D: Trade-off heatmap."""
    
    tradeoffs_df = data['editing_tradeoffs']
    if tradeoffs_df is None:
        ax.text(0.5, 0.5, 'Data not available', ha='center', va='center',
                transform=ax.transAxes, fontsize=11)
        ax.text(-0.15, 1.05, 'D', transform=ax.transAxes,
                fontsize=FONTSIZE_PANEL_LETTER, fontweight='bold')
        return
    
    if 'target_trait' not in tradeoffs_df.columns:
        return
    
    delta_col = next((c for c in ['mean_delta', 'delta', 'mean_abs_delta'] 
                      if c in tradeoffs_df.columns), None)
    if delta_col is None:
        return
    
    pivot_df = tradeoffs_df.pivot_table(index='target_trait', columns='affected_trait',
                                         values=delta_col, aggfunc='mean')
    present_idx = [t for t in TRAIT_ORDER if t in pivot_df.index]
    present_cols = [t for t in TRAIT_ORDER if t in pivot_df.columns]
    pivot_df = pivot_df.reindex(index=present_idx, columns=present_cols)
    
    cmap = LinearSegmentedColormap.from_list('green_gray',
        [TRAIT_COLOURS['TC'], '#FFFFFF', TRAIT_COLOURS['FBC']])
    
    vmax = np.nanmax(np.abs(pivot_df.values))
    im = ax.imshow(pivot_df.values, cmap=cmap, aspect='auto', vmin=-vmax, vmax=vmax)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Mean D', fontsize=9)
    
    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            val = pivot_df.values[i, j]
            if np.isfinite(val):
                text_colour = 'white' if abs(val) > vmax * 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=10, fontweight='bold', color=text_colour)
    
    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_xticklabels(pivot_df.columns, fontsize=FONTSIZE_TICK, fontweight='bold')
    ax.set_yticklabels(pivot_df.index, fontsize=FONTSIZE_TICK, fontweight='bold')
    ax.set_xlabel('Affected Trait', fontsize=FONTSIZE_LABEL, fontweight='bold')
    ax.set_ylabel('Edited Trait', fontsize=FONTSIZE_LABEL, fontweight='bold')
    ax.text(-0.08, 1.05, 'D', transform=ax.transAxes,
            fontsize=FONTSIZE_PANEL_LETTER, fontweight='bold')

# ============================================================================
# PANEL E: PC-Residualised Collapse (Wide with zones)
# ============================================================================

def create_panel_E(data, ax):
    """Panel E: Raw vs PC-residualised with background zones like Figure 1C."""
    
    perf_df = data['model_perf']
    if perf_df is None or 'trait_group' not in perf_df.columns:
        ax.text(0.5, 0.5, 'Data not available', ha='center', va='center',
                transform=ax.transAxes, fontsize=11)
        ax.text(-0.08, 1.05, 'E', transform=ax.transAxes,
                fontsize=FONTSIZE_PANEL_LETTER, fontweight='bold')
        return
    
    r_col = 'mean_r' if 'mean_r' in perf_df.columns else 'pearson_r_mean'
    
    model_priority = ['wide_deep', 'WideDeep', 'cnn', 'CNN']
    selected_model = next((m for m in model_priority if m in perf_df['model'].values), 
                          perf_df['model'].iloc[0])
    
    raw_df = perf_df[(perf_df['trait_group'] == 'raw') & (perf_df['model'] == selected_model)]
    resid_df = perf_df[(perf_df['trait_group'].isin(['resid', 'pc_resid', 'residualised'])) & 
                        (perf_df['model'] == selected_model)]
    
    traits_with_data = []
    raw_values = []
    resid_values = []
    
    for trait in TRAIT_ORDER:
        raw_val = raw_df[raw_df['trait'] == trait][r_col].values
        resid_val = resid_df[resid_df['trait'] == trait][r_col].values
        if len(raw_val) > 0:
            traits_with_data.append(trait)
            raw_values.append(raw_val[0])
            resid_values.append(resid_val[0] if len(resid_val) > 0 else 0)
    
    if len(traits_with_data) == 0:
        return
    
    raw_values = np.array(raw_values)
    resid_values = np.array(resid_values)
    
    # Background zones like Figure 1C
    ax.axhspan(0.3, 1.0, color=ZONE_GREEN, alpha=0.4, zorder=0)
    ax.axhspan(0, 0.3, color=ZONE_GRAY, alpha=0.5, zorder=0)
    ax.axhspan(-0.1, 0, color='#FFEBEE', alpha=0.5, zorder=0)
    
    # Zone labels on right
    ax.text(len(traits_with_data) - 0.3, 0.6, 'Transferable', fontsize=9, fontstyle='italic',
            color='#2E7D32', ha='right', va='center')
    ax.text(len(traits_with_data) - 0.3, 0.15, 'Marginal', fontsize=9, fontstyle='italic',
            color=GRAY, ha='right', va='center')
    ax.text(len(traits_with_data) - 0.3, -0.05, 'Collapsed', fontsize=9, fontstyle='italic',
            color='#C62828', ha='right', va='center')
    
    x = np.arange(len(traits_with_data))
    width = 0.35
    
    bars_raw = ax.bar(x - width/2, raw_values, width, label='Raw Phenotypes',
                      color=COLOUR_RAW, edgecolor='white', linewidth=1.5, zorder=3)
    bars_resid = ax.bar(x + width/2, resid_values, width, label='PC-Residualised',
                        color=COLOUR_RESIDUALISED, edgecolor='white', linewidth=1.5, zorder=3)
    
    # Value labels and delta annotations
    for i, (rv, rsv, trait) in enumerate(zip(raw_values, resid_values, traits_with_data)):
        if np.isfinite(rv) and rv > 0.02:
            ax.text(x[i] - width/2, rv + 0.02, f'{rv:.2f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='#2E7D32')
        if np.isfinite(rsv):
            ax.text(x[i] + width/2, max(rsv, 0) + 0.02, f'{rsv:.2f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color=DARKGRAY)
            
            # Delta annotation for collapsed traits (like Figure 2C)
            if trait in ['FF', 'TC'] and rsv < 0.1 and rv > 0.1:
                delta = rsv - rv
                ax.annotate(f'D = {delta:.2f}', 
                            xy=(x[i], (rv + rsv) / 2),
                            xytext=(x[i] + 0.6, (rv + rsv) / 2 + 0.15),
                            fontsize=9, fontweight='bold', color=TRAIT_COLOURS['TC'],
                            arrowprops=dict(arrowstyle='->', color=TRAIT_COLOURS['TC'], lw=1.5),
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                      edgecolor=TRAIT_COLOURS['TC'], linewidth=1.5),
                            zorder=5)
    
    ax.axhline(0, color='black', linewidth=1, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(traits_with_data, fontsize=FONTSIZE_LABEL, fontweight='bold')
    ax.set_ylabel('Prediction Accuracy (r)', fontsize=FONTSIZE_LABEL, fontweight='bold')
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.10),
        fontsize=9,
        ncol=2,
        framealpha=0.95
    )
    ax.set_ylim(-0.1, max(raw_values) * 1.25)
    ax.set_xlim(-0.5, len(traits_with_data) - 0.3)
    ax.text(-0.08, 1.05, 'E', transform=ax.transAxes,
            fontsize=FONTSIZE_PANEL_LETTER, fontweight='bold')

# ============================================================================
# MAIN FIGURE ASSEMBLY - OPTION 3 LAYOUT
# ============================================================================

def create_figure_3():
    """Assemble 5-panel Figure 3 with asymmetric L-shaped layout."""
    
    print("\n" + "=" * 60)
    print("FIGURE 3: Deep Learning Confirms Additive Supergenes")
    print("Asymmetric L-shaped layout")
    print("=" * 60)
    
    data = load_data()
    
    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    
    # Custom GridSpec for L-shaped layout
    # 3 rows, 4 columns
    gs = gridspec.GridSpec(3, 4, 
                           height_ratios=[1.0, 0.9, 0.9],
                           width_ratios=[1, 1, 1, 1],
                           hspace=0.45, wspace=0.50,
                           left=0.08, right=0.95, top=0.95, bottom=0.06)
    
    # Panel A: Top-left, spans 3 columns
    ax_A = fig.add_subplot(gs[0, 0:3])
    
    # Panel B: Top-right, 1 column
    ax_B = fig.add_subplot(gs[0, 3])
    
    # Panel C: Left side, spans rows 1-2 (tall), 1 column
    ax_C = fig.add_subplot(gs[1:3, 0])
    
    # Panel E: Middle-right, spans 3 columns, row 1
    ax_E = fig.add_subplot(gs[1, 1:4])
    
    # Panel D: Bottom-right, spans 3 columns, row 2
    ax_D = fig.add_subplot(gs[2, 1:4])
    
    print("\nCreating panels...")
    create_panel_A(data, ax_A)
    create_panel_B(data, ax_B)
    create_panel_C(data, ax_C)
    create_panel_D(data, ax_D)
    create_panel_E(data, ax_E)
    
    output_path = os.path.join(FIGURES_DIR, "figure_3.png")
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    print(f"\n[OK] Saved: {output_path}")
    
    output_pdf = os.path.join(FIGURES_DIR, "figure_3.pdf")
    fig.savefig(output_pdf, dpi=DPI, bbox_inches='tight', facecolor='white')
    print(f"[OK] Saved: {output_pdf}")
    
    for key, df in data.items():
        if df is not None:
            df.to_csv(os.path.join(FIGURE_SUBDIR, f"source_{key}.csv"), index=False)
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    
    return fig

if __name__ == "__main__":
    create_figure_3()