"""
Figure 1: Population Structure and the "Structure Cliff"
=========================================================
Theme: Genomic prediction accuracy depends critically on population structure 
       some traits survive cross-ancestry prediction while others collapse.

Panels:
  A (LARGE)  : PCA of 19,790 SNPs - three ancestry clusters visualized
  B (MEDIUM) : Cluster composition - horizontal bar showing sizes
  C (LARGE)  : The Structure Cliff - slopegraph showing accuracy drop across CV schemes
  D (MEDIUM) : Trait variability profile - CV% and sample sizes
  E (MEDIUM) : Genomic transferability index - portability metric
  F (SMALL)  : Trait correlation heatmap

"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Project paths
# Note: Script assumes execution from project root
PROJECT_ROOT = os.getcwd()
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
FIGURE_SUBDIR = os.path.join(FIGURES_DIR, "figure_1")

# Ensure directories exist
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(FIGURE_SUBDIR, exist_ok=True)

# Debug mode
DEBUG = False

def log(msg, level="INFO"):
    """Print log message"""
    prefix = {"INFO": "[INFO]", "OK": "[OK]", "WARN": "[WARN]", 
              "ERROR": "[ERROR]", "DEBUG": "[DEBUG]"}.get(level, "[INFO]")
    if level == "DEBUG" and not DEBUG:
        return
    print(f"{prefix} {msg}")

# ============================================================================
# COLOR CONFIGURATION (from figure_config.py)
# ============================================================================

class Colors:
    """Color palette - Blue-Teal-Green Theme"""
    
    # Primary colors (from figure_config.py)
    limegreen = '#32CD32'
    mediumseagreen = '#3CB371'
    springgreen = '#00FF7F'
    turquoise = '#40E0D0'
    mediumturquoise = '#48D1CC'
    deepskyblue = '#00BFFF'
    steelblue = '#4682B4'
    royalblue = '#4169E1'
    seagreen = '#2E8B57'
    
    # Accent
    coral_red = "#94CB64"
    teal_green = '#00A087'
    
    # Neutrals
    gray = '#808080'
    lightgray = '#D3D3D3'
    darkgray = '#A9A9A9'
    
    # Trait colors (standard order: FBC, AFW, FF, TC, TSS)
    trait_colors = {
        'FBC': mediumseagreen,   # #3CB371
        'AFW': royalblue,        # #4169E1
        'FF': turquoise,         # #40E0D0
        'TC': steelblue,         # #4682B4
        'TSS': limegreen,        # #32CD32
    }
    
    # Distinct trait colors (for slopegraph - high contrast)
    trait_colors_distinct = {
        'FBC': springgreen,      # #00FF7F
        'AFW': royalblue,        # #4169E1
        'FF': gray,              # #808080
        'TC': teal_green,        # #00A087
        'TSS': mediumseagreen,   # #3CB371
    }
    
    # Cluster colors (for PCA)
    cluster_colors = {
        0: coral_red,        # #94CB64
        1: teal_green,       # #00A087
        2: royalblue,        # #4169E1
    }
    
    # CV scheme colors
    cv_colors = {
        'random': deepskyblue,      # #00BFFF
        'balanced': mediumseagreen, # #3CB371
        'lco': royalblue,           # #4169E1
    }
    
    trait_order = ['FBC', 'AFW', 'FF', 'TC', 'TSS']
    
    trait_names = {
        'FBC': 'Fruit Blush\nColour',
        'AFW': 'Average Fruit\nWeight',
        'FF': 'Fruit\nFirmness',
        'TC': 'Trunk\nCircumference',
        'TSS': 'Total Soluble\nSolids',
    }
    
    trait_names_short = {
        'FBC': 'FBC',
        'AFW': 'AFW',
        'FF': 'FF',
        'TC': 'TC',
        'TSS': 'TSS',
    }

colors = Colors()

# ============================================================================
# DATA PATHS
# ============================================================================

class DataPaths:
    """Data file paths"""
    PC_SCORES = os.path.join(OUTPUT_DIR, "idea_1", "summary", "pc_scores_clusters.csv")
    CLUSTER_SIZES = os.path.join(OUTPUT_DIR, "idea_1", "summary", "cluster_sizes.csv")
    CV_TRANSFER = os.path.join(OUTPUT_DIR, "idea_1", "summary", "cv_transferability_summary.csv")
    PHENO_SUMMARY = os.path.join(OUTPUT_DIR, "idea_1", "summary", "pheno_trait_summary.csv")
    PHENO_CORE = os.path.join(OUTPUT_DIR, "idea_1", "core_data", "pheno_core.csv")

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load all required data files with validation"""
    data = {}
    
    log("=" * 70)
    log("LOADING DATA FOR FIGURE 1")
    log("=" * 70)
    
    # 1. PC scores with clusters
    log(f"Loading PC scores: {DataPaths.PC_SCORES}")
    if not os.path.exists(DataPaths.PC_SCORES):
        raise FileNotFoundError(f"PC scores file not found: {DataPaths.PC_SCORES}")
    
    data['pc_scores'] = pd.read_csv(DataPaths.PC_SCORES)
    log(f"  Shape: {data['pc_scores'].shape}", "OK")
    log(f"  Columns: {list(data['pc_scores'].columns)}", "DEBUG")
    
    # 2. Cluster sizes
    log(f"Loading cluster sizes: {DataPaths.CLUSTER_SIZES}")
    if not os.path.exists(DataPaths.CLUSTER_SIZES):
        raise FileNotFoundError(f"Cluster sizes file not found: {DataPaths.CLUSTER_SIZES}")
    
    data['cluster_sizes'] = pd.read_csv(DataPaths.CLUSTER_SIZES)
    log(f"  Shape: {data['cluster_sizes'].shape}", "OK")
    
    # 3. CV transferability
    log(f"Loading CV transferability: {DataPaths.CV_TRANSFER}")
    if not os.path.exists(DataPaths.CV_TRANSFER):
        raise FileNotFoundError(f"CV transferability file not found: {DataPaths.CV_TRANSFER}")
    
    data['cv_transfer'] = pd.read_csv(DataPaths.CV_TRANSFER)
    log(f"  Shape: {data['cv_transfer'].shape}", "OK")
    log(f"  Columns: {list(data['cv_transfer'].columns)}", "DEBUG")
    
    # 4. Phenotype summary
    log(f"Loading phenotype summary: {DataPaths.PHENO_SUMMARY}")
    if not os.path.exists(DataPaths.PHENO_SUMMARY):
        raise FileNotFoundError(f"Phenotype summary file not found: {DataPaths.PHENO_SUMMARY}")
    
    data['pheno_summary'] = pd.read_csv(DataPaths.PHENO_SUMMARY)
    log(f"  Shape: {data['pheno_summary'].shape}", "OK")
    
    # 5. Raw phenotypes for correlation
    log(f"Loading raw phenotypes: {DataPaths.PHENO_CORE}")
    if not os.path.exists(DataPaths.PHENO_CORE):
        raise FileNotFoundError(f"Raw phenotypes file not found: {DataPaths.PHENO_CORE}")
    
    data['pheno_core'] = pd.read_csv(DataPaths.PHENO_CORE)
    log(f"  Shape: {data['pheno_core'].shape}", "OK")
    
    log("=" * 70)
    log("DATA LOADING COMPLETE", "OK")
    log("=" * 70)
    
    return data

# ============================================================================
# PANEL A: PCA SCATTER PLOT
# ============================================================================

def create_panel_A(data, ax):
    """
    Panel A: PCA of 19,790 SNPs showing three ancestry clusters.
    Large scatter plot with cluster coloring and convex hulls.
    """
    log("\n[PANEL A] PCA Population Structure")
    
    pc_df = data['pc_scores']
    
    # Get PC columns
    pc1_col = 'PC1'
    pc2_col = 'PC2'
    cluster_col = 'cluster'
    
    # Verify columns exist
    for col in [pc1_col, pc2_col, cluster_col]:
        if col not in pc_df.columns:
            log(f"  Missing column: {col}", "ERROR")
            raise ValueError(f"Required column '{col}' not found in PC scores")
    
    # Get cluster info
    clusters = sorted(pc_df[cluster_col].unique())
    n_clusters = len(clusters)
    log(f"  Found {n_clusters} clusters: {clusters}", "DEBUG")
    
    # Plot each cluster with distinct styling
    for i, cluster in enumerate(clusters):
        mask = pc_df[cluster_col] == cluster
        cluster_data = pc_df[mask]
        n_samples = len(cluster_data)
        
        color = colors.cluster_colors.get(cluster, colors.gray)
        
        # Scatter points
        ax.scatter(
            cluster_data[pc1_col], 
            cluster_data[pc2_col],
            c=color,
            s=60,
            alpha=0.7,
            edgecolors='white',
            linewidth=0.5,
            label=f'Cluster {cluster + 1} (n={n_samples})',
            zorder=3
        )
        
        # Add convex hull / ellipse for cluster boundary
        from scipy.spatial import ConvexHull
        try:
            points = cluster_data[[pc1_col, pc2_col]].values
            if len(points) >= 3:
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                # Close the polygon
                hull_points = np.vstack([hull_points, hull_points[0]])
                ax.fill(hull_points[:, 0], hull_points[:, 1], 
                       color=color, alpha=0.1, zorder=1)
                ax.plot(hull_points[:, 0], hull_points[:, 1], 
                       color=color, linewidth=1.5, alpha=0.5, linestyle='--', zorder=2)
        except Exception as e:
            log(f"  Could not draw hull for cluster {cluster}: {e}", "DEBUG")
    
    # Styling
    ax.set_xlabel('PC1 (10.1% variance)', fontsize=11, fontweight='bold')
    ax.set_ylabel('PC2 (7.5% variance)', fontsize=11, fontweight='bold')

    # Legend
    legend = ax.legend(loc='upper right', bbox_to_anchor=(0.92, 1), fontsize=9, frameon=True, 
                      fancybox=True, framealpha=0.95)
    legend.get_frame().set_edgecolor(colors.gray)
    
    # Grid and spines
    ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add sample size annotation
    total_n = len(pc_df)
    ax.text(0.02, 0.98, f'N = {total_n}', transform=ax.transAxes,
           fontsize=10, va='top', ha='left',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=colors.lightgray))
    
    # Panel label
    ax.text(-0.08, 1.05, 'A', transform=ax.transAxes,
           fontsize=16, fontweight='bold', va='bottom', ha='left')
    
    log("  Panel A complete", "OK")

# ============================================================================
# PANEL B: CLUSTER SIZE BAR CHART
# ============================================================================

def create_panel_B(data, ax):
    """
    Panel B: Horizontal bar chart showing cluster sizes.
    """
    log("\n[PANEL B] Cluster Composition")
    
    cluster_df = data['cluster_sizes']
    
    # Get columns
    cluster_col = 'cluster'
    size_col = 'n_samples'
    
    # Sort by cluster number
    cluster_df = cluster_df.sort_values(cluster_col)
    
    clusters = cluster_df[cluster_col].values
    sizes = cluster_df[size_col].values
    
    log(f"  Clusters: {clusters}, Sizes: {sizes}", "DEBUG")
    
    # Create horizontal bars
    y_pos = np.arange(len(clusters))
    bar_colors = [colors.cluster_colors.get(c, colors.gray) for c in clusters]
    
    bars = ax.barh(y_pos, sizes, height=0.6, color=bar_colors, 
                   edgecolor='white', linewidth=1.5)
    
    # Add value labels inside bars
    for i, (bar, size) in enumerate(zip(bars, sizes)):
        # Position text inside bar
        ax.text(bar.get_width() - 5, bar.get_y() + bar.get_height()/2,
               f'{size}', ha='right', va='center', fontsize=11, 
               fontweight='bold', color='white')
    
    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'Cluster {c+1}' for c in clusters], fontsize=10)
    ax.set_xlabel('Number of Accessions', fontsize=10, fontweight='bold')
    
    # Total annotation
    total = sum(sizes)
    ax.axvline(x=total/len(clusters), color=colors.gray, linestyle=':', 
              alpha=0.7, linewidth=1.5)
    ax.text(total/len(clusters) + 2, len(clusters) - 0.3, f'Mean\n({total//len(clusters)})', 
           fontsize=8, ha='left', va='top', color=colors.gray)
    
    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, max(sizes) * 1.15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Panel label
    ax.text(-0.18, 1.05, 'B', transform=ax.transAxes,
           fontsize=16, fontweight='bold', va='bottom', ha='left')
    
    log("  Panel B complete", "OK")

# ============================================================================
# PANEL C: SLOPEGRAPH - THE STRUCTURE CLIFF
# ============================================================================

def create_panel_C(data, ax):
    """
    Panel C: Slopegraph showing the "Structure Cliff" - accuracy drop across CV schemes.
    This is the primary panel showing the key finding.
    """
    log("\n[PANEL C] The Structure Cliff (Slopegraph)")
    
    cv_df = data['cv_transfer']
    
    # Column mapping (from inventory)
    trait_col = 'trait'
    r_random_col = 'r_random_pc'
    r_balanced_col = 'r_cluster_balanced_pc'
    r_lco_col = 'r_leave_cluster_out_pc'
    
    # Verify columns
    for col in [trait_col, r_random_col, r_balanced_col, r_lco_col]:
        if col not in cv_df.columns:
            log(f"  Missing column: {col}", "ERROR")
            log(f"  Available: {list(cv_df.columns)}", "DEBUG")
            raise ValueError(f"Required column '{col}' not found")
    
    # Reorder traits to standard order
    cv_df = cv_df.set_index(trait_col).reindex(colors.trait_order).reset_index()
    
    # X positions for the three CV schemes (expanded - heatmap moved further right)
    x_positions = [0, 0.8, 1.6]
    x_labels = ['Random\nK-fold', 'Cluster-\nBalanced', 'Leave-\nCluster-Out']
    
    # Plot lines and points for each trait
    for idx, row in cv_df.iterrows():
        trait = row[trait_col]
        y_values = [row[r_random_col], row[r_balanced_col], row[r_lco_col]]
        
        color = colors.trait_colors_distinct.get(trait, colors.gray)
        
        # Determine line style based on outcome
        if y_values[2] < 0:  # Collapsed (negative LCO)
            linestyle = ':'
            alpha = 0.6
            linewidth = 2.5
        elif y_values[2] > 0.1:  # Survived
            linestyle = '-'
            alpha = 1.0
            linewidth = 3
        else:  # Marginal
            linestyle = '--'
            alpha = 0.8
            linewidth = 2.5
        
        # Draw connecting lines
        ax.plot(x_positions, y_values, color=color, linewidth=linewidth, 
               linestyle=linestyle, alpha=alpha, zorder=2)
        
        # Draw points
        ax.scatter(x_positions, y_values, color=color, s=120, zorder=3,
                  edgecolors='white', linewidth=2)
        
        # Add trait label at the end (right side)
        y_end = y_values[2]
        offset = 0.03 if y_end >= 0 else -0.03
        va = 'bottom' if y_end >= 0 else 'top'
        
        ax.annotate(colors.trait_names_short[trait], 
                   xy=(1.68, y_end),
                   fontsize=10, fontweight='bold', color=color,
                   va='center', ha='left')
        
        # Add value labels at each point
        for i, (x, y) in enumerate(zip(x_positions, y_values)):
            y_offset = 0.04 if i == 0 else (-0.04 if y < 0 else 0.04)
            ax.text(x, y + y_offset, f'{y:.2f}', ha='center', va='bottom' if y_offset > 0 else 'top',
                   fontsize=8, color=color, fontweight='bold')
    
    # Add horizontal reference line at 0
    ax.axhline(y=0, color=colors.darkgray, linestyle='-', linewidth=1, alpha=0.5, zorder=1)
    
    # Add shaded regions for interpretation
    ax.axhspan(0.15, 0.8, alpha=0.08, color=colors.limegreen, zorder=0)
    ax.axhspan(-0.15, 0.15, alpha=0.08, color=colors.gray, zorder=0)
    ax.axhspan(-0.3, -0.15, alpha=0.08, color=colors.coral_red, zorder=0)
    
    # Zone labels (positioned after trait labels, before heatmap)
    ax.text(1.78, 0.4, 'Transferable', fontsize=9, ha='left', va='center', 
           color=colors.seagreen, style='italic', alpha=0.8)
    ax.text(1.78, 0.0, 'Marginal', fontsize=9, ha='left', va='center', 
           color=colors.gray, style='italic', alpha=0.8)
    ax.text(1.78, -0.12, 'Collapsed', fontsize=9, ha='left', va='center', 
           color=colors.coral_red, style='italic', alpha=0.8)
    
    # Styling
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=11, fontweight='bold')
    ax.set_ylabel('Prediction Accuracy (Pearson r)', fontsize=11, fontweight='bold')
    
    ax.set_xlim(-0.3, 2.8)
    ax.set_ylim(-0.2, 0.7)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add annotation arrow showing the "cliff" (position adjusted for expanded layout)
    ax.annotate('', xy=(1.45, 0.05), xytext=(1.45, 0.45),
               arrowprops=dict(arrowstyle='->', color=colors.coral_red, 
                              lw=2, mutation_scale=15))
    ax.text(1.30, 0.25, 'Structure\nCliff', fontsize=9, ha='right', va='center',
           color=colors.coral_red, fontweight='bold', rotation=90)
    
    # Panel label
    ax.text(-0.06, 1.05, 'C', transform=ax.transAxes,
           fontsize=16, fontweight='bold', va='bottom', ha='left')
    
    # ----------------------------------------------------------------
    # INSET: Cross-collection transfer heatmap (Jighly et al. 2026)
    # External validation of structure cliff at global scale
    # Positioned in top-right corner (content shifted left to make room)
    # ----------------------------------------------------------------
    from matplotlib.colors import LinearSegmentedColormap
    
    # Load transfer data from validation output
    transfer_csv = os.path.join(OUTPUT_DIR, "idea_2", "external_validation", "cross_collection_transfer.csv")
    
    if os.path.exists(transfer_csv):
        transfer_df = pd.read_csv(transfer_csv)
        
        # Pivot to matrix form
        collections = ["AUS", "USA", "CHN"]
        transfer_fw = transfer_df.pivot(index="Reference", columns="Validation", values="FW_accuracy")
        transfer_fw = transfer_fw.reindex(index=collections, columns=collections).values
        
        # Create inset axes (TOP-RIGHT corner - size increased 20%)
        # Position: [left, bottom, width, height] in axes fraction coordinates
        ax_inset = ax.inset_axes([0.73, 0.36, 0.32, 0.54])
        
        # Custom colormap: gray -> white -> teal
        cmap_transfer = LinearSegmentedColormap.from_list(
            'transfer', 
            [(0.0, colors.gray), (0.4, '#FFFFFF'), (1.0, colors.teal_green)]
        )
        
        # Plot heatmap
        im = ax_inset.imshow(transfer_fw, cmap=cmap_transfer, vmin=0, vmax=0.8, aspect='equal')
        
        # Annotations (font size increased for larger heatmap)
        for i in range(3):
            for j in range(3):
                val = transfer_fw[i, j]
                color = 'white' if val > 0.5 or val < 0.15 else 'black'
                ax_inset.text(j, i, f'{val:.2f}', ha='center', va='center',
                             fontsize=8, fontweight='bold', color=color)
        
        # Labels (font size increased for larger heatmap)
        ax_inset.set_xticks([0, 1, 2])
        ax_inset.set_yticks([0, 1, 2])
        ax_inset.set_xticklabels(collections, fontsize=8)
        ax_inset.set_yticklabels(collections, fontsize=8)
        ax_inset.set_xlabel('Validation', fontsize=8, labelpad=2)
        ax_inset.set_ylabel('Reference', fontsize=8, labelpad=2)
        
        # Title (font size increased for larger heatmap)
        ax_inset.set_title('Global transfer (FW)\nJighly et al. 2026', 
                           fontsize=8, fontweight='bold', color=colors.teal_green, pad=3)
        
        # Grid
        ax_inset.set_xticks([0.5, 1.5], minor=True)
        ax_inset.set_yticks([0.5, 1.5], minor=True)
        ax_inset.grid(which='minor', color='white', linewidth=1)
        
        # Border
        for spine in ax_inset.spines.values():
            spine.set_edgecolor(colors.teal_green)
            spine.set_linewidth(1.5)
        
        log("  Cross-collection heatmap inset added (top-right)", "OK")
    else:
        log(f"  Cross-collection data not found: {transfer_csv}", "WARN")
    
    log("  Panel C complete", "OK")

# ============================================================================
# PANEL D: PHENOTYPE VARIABILITY
# ============================================================================

def create_panel_D(data, ax):
    """
    Panel D: Bar chart showing phenotypic variability (CV%) per trait.
    """
    log("\n[PANEL D] Trait Variability Profile")
    
    pheno_df = data['pheno_summary']
    
    # Column mapping
    trait_col = 'trait'
    n_col = 'n_non_missing'
    cv_col = 'cv_percent'
    
    # Reorder to standard trait order
    pheno_df = pheno_df.set_index(trait_col).reindex(colors.trait_order).reset_index()
    
    traits = pheno_df[trait_col].values
    cv_values = pheno_df[cv_col].values
    n_values = pheno_df[n_col].values
    
    log(f"  CV% values: {cv_values}", "DEBUG")
    
    # Create bars
    x_pos = np.arange(len(traits))
    bar_colors = [colors.trait_colors.get(t, colors.gray) for t in traits]
    
    bars = ax.bar(x_pos, cv_values, width=0.7, color=bar_colors,
                 edgecolor='white', linewidth=1.5)
    
    # Add n values on top of bars
    for i, (bar, n) in enumerate(zip(bars, n_values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'n={int(n)}', ha='center', va='bottom', fontsize=8, color=colors.darkgray)
    
    # Styling
    ax.set_xticks(x_pos)
    ax.set_xticklabels([colors.trait_names_short[t] for t in traits], fontsize=9, rotation=0)
    ax.set_ylabel('Coefficient of Variation (%)', fontsize=10, fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(cv_values) * 1.2)
    
    # Panel label
    ax.text(-0.15, 1.05, 'D', transform=ax.transAxes,
           fontsize=16, fontweight='bold', va='bottom', ha='left')
    
    log("  Panel D complete", "OK")

# ============================================================================
# PANEL E: GENOMIC TRANSFERABILITY INDEX
# ============================================================================

def create_panel_E(data, ax):
    """
    Panel E: Horizontal bar chart showing genomic transferability index (r_LCO / r_random).
    """
    log("\n[PANEL E] Genomic Transferability Index")
    
    cv_df = data['cv_transfer']
    
    # Column mapping
    trait_col = 'trait'
    r_random_col = 'r_random_pc'
    r_lco_col = 'r_leave_cluster_out_pc'
    
    # Calculate transferability index
    cv_df = cv_df.copy()
    cv_df['transferability'] = cv_df.apply(
        lambda row: max(0, row[r_lco_col]) / row[r_random_col] if row[r_random_col] > 0 else 0, 
        axis=1
    )
    
    # Reorder to standard trait order
    cv_df = cv_df.set_index(trait_col).reindex(colors.trait_order).reset_index()
    
    traits = cv_df[trait_col].values
    transfer_values = cv_df['transferability'].values
    
    log(f"  Transferability: {transfer_values}", "DEBUG")
    
    # Use standard trait order (reversed so FBC is at top of plot)
    traits_sorted = traits[::-1]
    values_sorted = transfer_values[::-1]
    
    # Create horizontal bars
    y_pos = np.arange(len(traits_sorted))
    bar_colors = [colors.trait_colors.get(t, colors.gray) for t in traits_sorted]
    
    bars = ax.barh(y_pos, values_sorted, height=0.6, color=bar_colors,
                   edgecolor='white', linewidth=1.5)
    
    # Add value labels
    for bar, val in zip(bars, values_sorted):
        x_pos = bar.get_width() + 0.02
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
               f'{val:.2f}', ha='left', va='center', fontsize=9, fontweight='bold')
    
    # Reference line at 0.5 (50% retention)
    ax.axvline(x=0.5, color=colors.gray, linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(0.52, len(traits) - 0.3, '50%\nretention', fontsize=8, ha='left', va='top',
           color=colors.gray, style='italic')
    
    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels([colors.trait_names_short[t] for t in traits_sorted], fontsize=10)
    ax.set_xlabel('Transferability Index\n(r_LCO / r_Random)', fontsize=10, fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, 1.0)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Panel label
    ax.text(-0.18, 1.05, 'E', transform=ax.transAxes,
           fontsize=16, fontweight='bold', va='bottom', ha='left')
    
    log("  Panel E complete", "OK")

# ============================================================================
# PANEL F: TRAIT CORRELATION HEATMAP
# ============================================================================

def create_panel_F(data, ax):
    """
    Panel F: Heatmap showing phenotypic correlations among traits.
    Uses green colormap as requested.
    """
    log("\n[PANEL F] Trait Correlation Matrix")
    
    pheno_df = data['pheno_core']
    
    # Get trait columns in standard order
    trait_cols = [t for t in colors.trait_order if t in pheno_df.columns]
    log(f"  Trait columns: {trait_cols}", "DEBUG")
    
    # Calculate correlation matrix
    corr_matrix = pheno_df[trait_cols].corr()
    
    log(f"  Correlation matrix:\n{corr_matrix.round(2)}", "DEBUG")
    
    # Create mask for upper triangle (optional - show full matrix for small size)
    # mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    # Custom green diverging colormap (blue for negative, green for positive)
    from matplotlib.colors import LinearSegmentedColormap
    cmap_corr = LinearSegmentedColormap.from_list(
        'corr_green',
        ['#4169E1', 'white', '#2E8B57']  # royalblue -> white -> seagreen
    )
    
    # Create heatmap
    sns.heatmap(corr_matrix, ax=ax, cmap=cmap_corr, center=0,
               vmin=-1, vmax=1,
               annot=True, fmt='.2f', annot_kws={'fontsize': 8, 'fontweight': 'bold'},
               square=True, linewidths=1, linecolor='white',
               cbar_kws={'shrink': 0.8, 'label': 'Pearson r'})
    
    # Styling
    ax.set_xticklabels([colors.trait_names_short[t] for t in trait_cols], 
                       fontsize=9, rotation=45, ha='right')
    ax.set_yticklabels([colors.trait_names_short[t] for t in trait_cols], 
                       fontsize=9, rotation=0)
    
    # Panel label
    ax.text(-0.25, 1.05, 'F', transform=ax.transAxes,
           fontsize=16, fontweight='bold', va='bottom', ha='left')
    
    log("  Panel F complete", "OK")

# ============================================================================
# MAIN FIGURE ASSEMBLY
# ============================================================================

def create_figure_1():
    """
    Assemble all panels into Figure 1 with variable panel sizes.
    
    Layout (Variable Panel Sizes):
        Row 0: A (4 cols) + B (2 cols)          - height 1.3
        Row 1: C (full width - 6 cols)          - height 1.5
        Row 2: D (2 cols) + E (2 cols) + F (2 cols) - height 1.0
    """
    log("\n" + "=" * 70)
    log("ASSEMBLING FIGURE 1: Population Structure & Structure Cliff")
    log("=" * 70)
    
    # Load data
    data = load_data()
    
    # Create figure with custom gridspec
    fig = plt.figure(figsize=(14, 14))
    
    # Define grid: 3 rows, 6 columns with variable heights
    gs = gridspec.GridSpec(
        nrows=3, ncols=6,
        height_ratios=[1.3, 1.5, 1.0],
        width_ratios=[1, 1, 1, 1, 0.9, 0.9],
        hspace=0.35,
        wspace=0.35
    )
    
    # Create axes for each panel
    ax_A = fig.add_subplot(gs[0, 0:4])  # Row 0, cols 0-3 (large)
    ax_B = fig.add_subplot(gs[0, 4:6])  # Row 0, cols 4-5 (medium)
    ax_C = fig.add_subplot(gs[1, 0:6])  # Row 1, full width (hero panel)
    ax_D = fig.add_subplot(gs[2, 0:2])  # Row 2, cols 0-1 (left)
    ax_E = fig.add_subplot(gs[2, 2:4])  # Row 2, cols 2-3 (middle)
    ax_F = fig.add_subplot(gs[2, 4:6])  # Row 2, cols 4-5
    
    # Create each panel
    log("\nCreating panels...")
    
    create_panel_A(data, ax_A)
    create_panel_B(data, ax_B)
    create_panel_C(data, ax_C)
    create_panel_D(data, ax_D)
    create_panel_E(data, ax_E)
    create_panel_F(data, ax_F)
    
    # Save figure
    output_path = os.path.join(FIGURES_DIR, "figure_1.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    log(f"\n[OK] Figure saved: {output_path}", "OK")
    
    # Also save as PDF for publication
    output_pdf = os.path.join(FIGURES_DIR, "figure_1.pdf")
    fig.savefig(output_pdf, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    log(f"[OK] PDF saved: {output_pdf}", "OK")
    
    # Save source data
    log("\nSaving source data...")
    
    # Save CV transferability data used in slopegraph
    cv_out = os.path.join(FIGURE_SUBDIR, "panel_C_cv_transferability.csv")
    data['cv_transfer'].to_csv(cv_out, index=False)
    log(f"  [OK] Panel C source: {cv_out}", "OK")
    
    # Save phenotype summary
    pheno_out = os.path.join(FIGURE_SUBDIR, "panel_D_pheno_summary.csv")
    data['pheno_summary'].to_csv(pheno_out, index=False)
    log(f"  [OK] Panel D source: {pheno_out}", "OK")
    
    # Save correlation matrix
    trait_cols = [t for t in colors.trait_order if t in data['pheno_core'].columns]
    corr_matrix = data['pheno_core'][trait_cols].corr()
    corr_out = os.path.join(FIGURE_SUBDIR, "panel_F_correlation_matrix.csv")
    corr_matrix.to_csv(corr_out)
    log(f"  [OK] Panel F source: {corr_out}", "OK")
    
    log("\n" + "=" * 70)
    log("FIGURE 1 COMPLETE!", "OK")
    log("=" * 70)
    
    return fig

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MANGO GWAS - FIGURE 1 GENERATOR")
    print("Population Structure and the 'Structure Cliff'")
    print("=" * 70)
    print(f"\nOutput directory: {FIGURES_DIR}")
    print("-" * 70)
    
    try:
        fig = create_figure_1()
        
        print("\n" + "=" * 70)
        print("SUCCESS!")
        print(f"  Figure: {os.path.join(FIGURES_DIR, 'figure_1.png')}")
        print(f"  Data:   {FIGURE_SUBDIR}")
        print("=" * 70 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] File not found: {e}")
        print("\nPlease verify the data files exist at the expected paths.")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()