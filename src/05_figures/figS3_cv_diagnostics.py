#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure S3: Cross-Validation Diagnostics and Structure-Aware Design
===================================================================

Demonstrates stability/variance across folds for different CV schemes
and the effect of population structure correction.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

# Import unified configuration
from figure_config import config, trait_colors, cluster_colors, cv_colors

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
OUTPUT_DIR = config.paths.OUTPUT_DIR
FIGURES_DIR = config.paths.FIGURES_DIR
FIGURE_SUBDIR = os.path.join(FIGURES_DIR, "figure_s3")

# Data paths
DATA_PATHS = {
    'cv_baseline': os.path.join(OUTPUT_DIR, "idea_1", "cv_baseline", "cv_baseline_results.csv"),
    'cv_structure': os.path.join(OUTPUT_DIR, "idea_1", "cv_structure", "cv_structure_results.csv"),
    'cluster_sizes': os.path.join(OUTPUT_DIR, "idea_1", "summary", "cluster_sizes.csv"),
}

# Traits and order
TRAITS = ['FBC', 'AFW', 'FF', 'TC', 'TSS']

# Ensure output directories exist
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(FIGURE_SUBDIR, exist_ok=True)


# ============================================================================
# LOGGING
# ============================================================================

def log(msg):
    """Simple logger."""
    print(f"[INFO] {msg}")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load all CV results data for Figure S3."""
    log("Loading data for Figure S3...")
    
    data = {}
    
    for key, path in DATA_PATHS.items():
        if os.path.exists(path):
            data[key] = pd.read_csv(path)
            log(f"Loaded {key}: {data[key].shape}")
        else:
            log(f"File not found: {path}")
            data[key] = None
    
    return data


# ============================================================================
# PANEL A: Per-Fold Accuracy Distribution
# ============================================================================

def create_panel_A(data, ax):
    """
    Panel A: Boxplots showing per-fold accuracy variance for each CV scheme.
    """
    
    baseline_df = data['cv_baseline']
    structure_df = data['cv_structure']
    
    if baseline_df is None and structure_df is None:
        ax.text(0.5, 0.5, 'CV data not available', ha='center', va='center',
               fontsize=12, transform=ax.transAxes)
        ax.text(-0.08, 1.05, 'A', transform=ax.transAxes,
               fontsize=14, fontweight='bold')
        return
    
    # Combine data - focus on PC-corrected results
    all_data = []
    
    # Random K-fold (from baseline)
    if baseline_df is not None:
        random_df = baseline_df[baseline_df['scenario'] == 'pc_corrected'].copy()
        random_df['scheme'] = 'Random'
        all_data.append(random_df[['trait', 'scheme', 'r']])
    
    # Cluster-balanced and LCO (from structure)
    if structure_df is not None:
        struct_pc = structure_df[structure_df['scenario'] == 'pc_corrected'].copy()
        
        cb_df = struct_pc[struct_pc['scheme'] == 'cluster_balanced'].copy()
        cb_df['scheme'] = 'Cluster-Bal'
        all_data.append(cb_df[['trait', 'scheme', 'r']])
        
        lco_df = struct_pc[struct_pc['scheme'] == 'leave_cluster_out'].copy()
        lco_df['scheme'] = 'LCO'
        all_data.append(lco_df[['trait', 'scheme', 'r']])
    
    if not all_data:
        ax.text(0.5, 0.5, 'No CV data found', ha='center', va='center')
        return
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Create grouped boxplot
    schemes = ['Random', 'Cluster-Bal', 'LCO']
    # Use colors from config
    scheme_colors_list = [cv_colors['random'], 
                          cv_colors['cluster_balanced'],
                          cv_colors['leave_cluster_out']]
    
    positions = []
    box_data = []
    box_colors = []
    
    trait_positions = {}
    pos = 0
    
    for trait in TRAITS:
        trait_start = pos
        for i, scheme in enumerate(schemes):
            mask = (combined['trait'] == trait) & (combined['scheme'] == scheme)
            values = combined.loc[mask, 'r'].values
            
            if len(values) > 0:
                box_data.append(values)
                positions.append(pos)
                box_colors.append(scheme_colors_list[i])
            pos += 1
        
        trait_positions[trait] = (trait_start + pos - 1) / 2  # center position
        pos += 0.5  # gap between traits
    
    # Create boxplots
    bp = ax.boxplot(box_data, positions=positions, widths=0.7, patch_artist=True,
                   showfliers=True, flierprops=dict(marker='o', markersize=4, alpha=0.5))
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add individual points
    for i, (vals, pos) in enumerate(zip(box_data, positions)):
        jitter = np.random.uniform(-0.15, 0.15, len(vals))
        ax.scatter([pos + j for j in jitter], vals, c=[box_colors[i]], 
                  s=25, alpha=0.6, edgecolors='white', linewidth=0.5, zorder=3)
    
    # Reference line at 0
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    
    # X-axis labels (traits)
    ax.set_xticks(list(trait_positions.values()))
    ax.set_xticklabels(list(trait_positions.keys()), fontsize=10, fontweight='bold')
    
    # Styling
    ax.set_ylabel('Prediction Accuracy (r)', fontsize=10, fontweight='bold')
    # Removed title as requested
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Legend
    legend_patches = [Patch(facecolor=c, alpha=0.7, label=s) 
                     for s, c in zip(schemes, scheme_colors_list)]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=8)
    
    ax.text(-0.08, 1.05, 'A', transform=ax.transAxes, fontsize=14, fontweight='bold')


# ============================================================================
# PANEL B: CV Scheme Composition
# ============================================================================

def create_panel_B(data, ax):
    """
    Panel B: Schematic showing how different CV schemes partition data.
    """
    
    # Define CV schemes conceptually
    schemes = [
        {
            'name': 'Random K-fold',
            'description': 'All clusters mixed\nin each fold',
            'train_clusters': [0, 1, 2],
            'test_clusters': [0, 1, 2],
            'color': cv_colors['random']
        },
        {
            'name': 'Cluster-Balanced',
            'description': 'Equal cluster ratios\nin train & test',
            'train_clusters': [0, 1, 2],
            'test_clusters': [0, 1, 2],
            'color': cv_colors['cluster_balanced']
        },
        {
            'name': 'Leave-Cluster-Out',
            'description': 'Test on entirely\nunseen cluster',
            'train_clusters': [0, 1],
            'test_clusters': [2],
            'color': cv_colors['leave_cluster_out']
        }
    ]
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Removed title as requested
    
    # Draw each scheme
    y_positions = [7, 4.5, 2]
    
    for scheme, y in zip(schemes, y_positions):
        # Scheme name box
        from matplotlib.patches import FancyBboxPatch
        box = FancyBboxPatch((0.2, y-0.6), 2.8, 1.4, boxstyle='round,pad=0.1',
                            facecolor=scheme['color'], alpha=0.3, 
                            edgecolor=scheme['color'], linewidth=2)
        ax.add_patch(box)
        ax.text(1.6, y+0.1, scheme['name'], fontsize=9, fontweight='bold',
               ha='center', va='center', color=scheme['color'])
        
        # Train section
        ax.text(4.5, y+0.3, 'Train', fontsize=8, fontweight='bold', ha='center')
        train_x = 3.5
        for i, c in enumerate(scheme['train_clusters']):
            circle = plt.Circle((train_x + i*0.7, y-0.2), 0.25, 
                               color=cluster_colors[c], alpha=0.8)
            ax.add_patch(circle)
        
        # Arrow
        ax.annotate('', xy=(6.5, y), xytext=(5.5, y),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
        
        # Test section
        ax.text(8, y+0.3, 'Test', fontsize=8, fontweight='bold', ha='center')
        test_x = 7.2
        for i, c in enumerate(scheme['test_clusters']):
            circle = plt.Circle((test_x + i*0.7, y-0.2), 0.25,
                               color=cluster_colors[c], alpha=0.8)
            ax.add_patch(circle)
        
        # Description
        ax.text(9.5, y, scheme['description'], fontsize=7, ha='left', va='center',
               style='italic', color='gray')
    
    # Cluster legend
    ax.text(0.5, 0.5, 'Clusters:', fontsize=8, fontweight='bold')
    for i, c in enumerate([0, 1, 2]):
        circle = plt.Circle((2 + i*1.5, 0.5), 0.2, color=cluster_colors[c])
        ax.add_patch(circle)
        ax.text(2.4 + i*1.5, 0.5, f'C{c}', fontsize=8, va='center')
    
    ax.text(-0.05, 1.02, 'B', transform=ax.transAxes, fontsize=14, fontweight='bold')


# ============================================================================
# PANEL C: PC Correction Effect
# ============================================================================

def create_panel_C(data, ax):
    """
    Panel C: Shows the effect of PC correction on accuracy.
    """
    
    baseline_df = data['cv_baseline']
    structure_df = data['cv_structure']
    
    # Combine baseline and structure data
    all_data = []
    
    if baseline_df is not None:
        baseline_df = baseline_df.copy()
        baseline_df['scheme'] = 'random'
        all_data.append(baseline_df)
    
    if structure_df is not None:
        all_data.append(structure_df)
    
    if not all_data:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        ax.text(-0.08, 1.05, 'C', transform=ax.transAxes, fontsize=14, fontweight='bold')
        return
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Calculate mean accuracy per trait and scenario (across all schemes)
    summary = combined.groupby(['trait', 'scenario'])['r'].mean().reset_index()
    
    # Pivot to get no_pc and pc_corrected side by side
    pivot = summary.pivot(index='trait', columns='scenario', values='r')
    
    if 'no_pc' not in pivot.columns or 'pc_corrected' not in pivot.columns:
        ax.text(0.5, 0.5, 'Missing scenario data', ha='center', va='center')
        ax.text(-0.08, 1.05, 'C', transform=ax.transAxes, fontsize=14, fontweight='bold')
        return
    
    # Order traits
    traits_ordered = [t for t in TRAITS if t in pivot.index]
    pivot = pivot.loc[traits_ordered]
    
    x = np.arange(len(traits_ordered))
    width = 0.35
    
    # Colors for PC correction comparison
    col_no_pc = config.colors.cornflowerblue # Lighter blue
    col_pc = config.colors.royalblue         # Darker blue
    
    # Bars
    bars1 = ax.bar(x - width/2, pivot['no_pc'], width, 
                   label='No PC correction', color=col_no_pc,
                   edgecolor='white', linewidth=1)
    bars2 = ax.bar(x + width/2, pivot['pc_corrected'], width,
                   label='PC-corrected', color=col_pc,
                   edgecolor='white', linewidth=1)
    
    # Value labels
    for bar, val in zip(bars1, pivot['no_pc']):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}',
               ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars2, pivot['pc_corrected']):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}',
               ha='center', va='bottom', fontsize=8)
    
    # Arrows showing drop
    for i, trait in enumerate(traits_ordered):
        no_pc_val = pivot.loc[trait, 'no_pc']
        pc_val = pivot.loc[trait, 'pc_corrected']
        drop = no_pc_val - pc_val
        
        if drop > 0.05:  # Only annotate significant drops
            # mid_y = (no_pc_val + pc_val) / 2
            ax.annotate('', xy=(i + width/2, pc_val + 0.08), 
                       xytext=(i - width/2, no_pc_val + 0.08),
                       arrowprops=dict(arrowstyle='->', color='gray', lw=1))
            ax.text(i, max(no_pc_val, pc_val) + 0.12, f'Δ={drop:.2f}',
                   ha='center', fontsize=7, color='gray')
    
    # Reference line at 0
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    
    # Styling
    ax.set_xticks(x)
    ax.set_xticklabels(traits_ordered, fontsize=10, fontweight='bold')
    ax.set_ylabel('Mean Accuracy (r)', fontsize=10, fontweight='bold')
    # Removed title as requested
    
    ax.legend(loc='upper right', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    ax.text(-0.08, 1.05, 'C', transform=ax.transAxes, fontsize=14, fontweight='bold')


# ============================================================================
# MAIN FIGURE ASSEMBLY
# ============================================================================

def create_figure_s3():
    """
    Assemble Figure S3: Cross-Validation Diagnostics.
    """
    log("ASSEMBLING FIGURE S3")
    
    data = load_data()
    
    fig = plt.figure(figsize=(14, 10))
    
    # Adjusted GridSpec to reduce row space (hspace) and increase col space (wspace)
    gs = gridspec.GridSpec(
        nrows=2, ncols=2,
        height_ratios=[1.2, 1.0],
        width_ratios=[1.2, 1.0],
        hspace=0.20,  # Reduced from 0.35 to avoid gap after title removal
        wspace=0.45   # Increased from 0.25 to separate B and C
    )
    
    # Panel A: Full width top row
    ax_A = fig.add_subplot(gs[0, :])
    
    # Panel B: Bottom left
    ax_B = fig.add_subplot(gs[1, 0])
    
    # Panel C: Bottom right
    ax_C = fig.add_subplot(gs[1, 1])
    
    # Create panels
    create_panel_A(data, ax_A)
    create_panel_B(data, ax_B)
    create_panel_C(data, ax_C)
    
    # Removed suptitle as requested
    
    # Save
    output_path = os.path.join(FIGURES_DIR, "figure_s3.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    log(f"Figure saved: {output_path}")
    
    output_pdf = os.path.join(FIGURES_DIR, "figure_s3.pdf")
    fig.savefig(output_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    log(f"PDF saved: {output_pdf}")
    
    # Save source data
    for key, df in data.items():
        if df is not None:
            out_path = os.path.join(FIGURE_SUBDIR, f"source_{key}.csv")
            df.to_csv(out_path, index=False)
            log(f"Source data saved: {out_path}")
    
    log("FIGURE S3 COMPLETE")
    
    return fig


if __name__ == "__main__":
    print("=" * 70)
    print("MANGO GWAS - FIGURE S3 GENERATOR")
    print("=" * 70)
    
    try:
        fig = create_figure_s3()
        print("\nSUCCESS!")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
