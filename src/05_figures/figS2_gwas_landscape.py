"""
Figure 2: Structural Haplotypes as Ultra-Efficient Predictors
==============================================================
Theme: A 17-marker inversion panel matches or exceeds dense SNP-based prediction 
       for colour, and individual inversions show quasi-Mendelian additive effects.

Panels:
  A (WIDE)   : Inversion chromosome ideogram - genomic positions of 17 inversions
  B (LARGE)  : Inversion vs Random panel comparison - ridge/violin plot
  C (LARGE)  : Additive super-gene effects - boxplots across genotypes (0/1/2)
  D (MEDIUM) : Model comparison across CV schemes
  E (MEDIUM) : Effect size catalogue - top inversions per trait
  F (SMALL)  : Expected genetic gain per cycle


"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.collections import PatchCollection
import seaborn as sns
import warnings
import figure_config

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Detect project root relative to this script (assuming code/figures/figure_2.py)
# Adjust this path if running from a different location
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(script_dir, "..", ".."))
except NameError:
    # Fallback for interactive environments
    PROJECT_ROOT = os.getcwd()

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
FIGURE_SUBDIR = os.path.join(FIGURES_DIR, "figure_2")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(FIGURE_SUBDIR, exist_ok=True)

DEBUG = False

def log(msg, level="INFO"):
    prefix = {"INFO": "[INFO]", "OK": "[OK]", "WARN": "[WARN]", 
              "ERROR": "[ERROR]", "DEBUG": "[DEBUG]"}.get(level, "[INFO]")
    if level == "DEBUG" and not DEBUG:
        return
    print(f"{prefix} {msg}")

# ============================================================================
# COLOR CONFIGURATION
# ============================================================================

# Load colors from unified configuration
colors = figure_config.config.colors

# Add figure-specific colors not in global config
colors.chrom_colors_light = '#E8F5E9'
colors.chrom_colors_dark = '#C8E6C9'

# Genotype colors (for boxplots)
colors.genotype_colors = {
    0: '#E8F5E9',  # Light green
    1: '#81C784',  # Medium green
    2: '#2E7D32',  # Dark green
}

# Model comparison colors (map local keys to global colors)
colors.model_colors.update({
    'Dense SNP': colors.royalblue,      # royalblue
    'Inversion': colors.limegreen,       # limegreen
    'Random-17': colors.gray,       # gray
})

# ============================================================================
# DATA PATHS
# ============================================================================

class DataPaths:
    HAPLOTYPE_EFFECTS = os.path.join(OUTPUT_DIR, "idea_2", "breeder_tools", "Breeder_Haplotype_Effects.csv")
    ASSAY_DESIGN = os.path.join(OUTPUT_DIR, "idea_2", "breeder_tools", "Supplementary_Table_Assay_Design.csv")
    RANDOM_VS_INV = os.path.join(OUTPUT_DIR, "idea_2", "random_control", "random_vs_inversion_replicates.csv")
    META_CORE = os.path.join(OUTPUT_DIR, "idea_1", "core_data", "meta_core.csv")
    PHENO_CORE = os.path.join(OUTPUT_DIR, "idea_1", "core_data", "pheno_core.csv")
    MODEL_PERF = os.path.join(OUTPUT_DIR, "idea_2", "summary", "idea2_gs_model_performance_clean.csv")
    GENETIC_GAIN = os.path.join(OUTPUT_DIR, "idea_2", "breeder_tools", "Estimated_Genetic_Gain.csv")

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load all required data files"""
    data = {}
    
    log("=" * 70)
    log("LOADING DATA FOR FIGURE 2")
    log("=" * 70)
    
    # 1. Haplotype effects
    log(f"Loading haplotype effects: {DataPaths.HAPLOTYPE_EFFECTS}")
    if os.path.exists(DataPaths.HAPLOTYPE_EFFECTS):
        data['haplotype_effects'] = pd.read_csv(DataPaths.HAPLOTYPE_EFFECTS)
        log(f"  Shape: {data['haplotype_effects'].shape}", "OK")
        log(f"  Columns: {list(data['haplotype_effects'].columns)}", "DEBUG")
    else:
        raise FileNotFoundError(f"Haplotype effects not found: {DataPaths.HAPLOTYPE_EFFECTS}")
    
    # 2. Assay design (for chromosome positions)
    log(f"Loading assay design: {DataPaths.ASSAY_DESIGN}")
    if os.path.exists(DataPaths.ASSAY_DESIGN):
        data['assay_design'] = pd.read_csv(DataPaths.ASSAY_DESIGN)
        log(f"  Shape: {data['assay_design'].shape}", "OK")
    else:
        log("  Assay design not found - will use marker names for positions", "WARN")
        data['assay_design'] = None
    
    # 3. Random vs Inversion comparison
    log(f"Loading random vs inversion: {DataPaths.RANDOM_VS_INV}")
    if os.path.exists(DataPaths.RANDOM_VS_INV):
        data['random_vs_inv'] = pd.read_csv(DataPaths.RANDOM_VS_INV)
        log(f"  Shape: {data['random_vs_inv'].shape}", "OK")
        log(f"  Columns: {list(data['random_vs_inv'].columns)}", "DEBUG")
    else:
        raise FileNotFoundError(f"Random vs inversion not found: {DataPaths.RANDOM_VS_INV}")
    
    # 4. Meta core (inversion genotypes)
    log(f"Loading meta core: {DataPaths.META_CORE}")
    if os.path.exists(DataPaths.META_CORE):
        data['meta_core'] = pd.read_csv(DataPaths.META_CORE)
        log(f"  Shape: {data['meta_core'].shape}", "OK")
    else:
        raise FileNotFoundError(f"Meta core not found: {DataPaths.META_CORE}")
    
    # 5. Pheno core (phenotypes)
    log(f"Loading pheno core: {DataPaths.PHENO_CORE}")
    if os.path.exists(DataPaths.PHENO_CORE):
        data['pheno_core'] = pd.read_csv(DataPaths.PHENO_CORE)
        log(f"  Shape: {data['pheno_core'].shape}", "OK")
    else:
        raise FileNotFoundError(f"Pheno core not found: {DataPaths.PHENO_CORE}")
    
    # 6. Model performance
    log(f"Loading model performance: {DataPaths.MODEL_PERF}")
    if os.path.exists(DataPaths.MODEL_PERF):
        data['model_perf'] = pd.read_csv(DataPaths.MODEL_PERF)
        log(f"  Shape: {data['model_perf'].shape}", "OK")
    else:
        raise FileNotFoundError(f"Model performance not found: {DataPaths.MODEL_PERF}")
    
    # 7. Genetic gain
    log(f"Loading genetic gain: {DataPaths.GENETIC_GAIN}")
    if os.path.exists(DataPaths.GENETIC_GAIN):
        data['genetic_gain'] = pd.read_csv(DataPaths.GENETIC_GAIN)
        log(f"  Shape: {data['genetic_gain'].shape}", "OK")
    else:
        raise FileNotFoundError(f"Genetic gain not found: {DataPaths.GENETIC_GAIN}")
    
    log("=" * 70)
    log("DATA LOADING COMPLETE", "OK")
    log("=" * 70)
    
    return data

# ============================================================================
# PANEL A: CHROMOSOME IDEOGRAM WITH INVERSIONS
# ============================================================================

def create_panel_A(data, ax):
    """
    Panel A: Horizontal chromosome ideogram showing locations of 17 structural inversions.
    Highlights key inversions (miinv6.0, miinv11.0, miinv17.0).
    REDESIGNED: Horizontal layout with chromosomes as columns for better space usage.
    """
    log("\n[PANEL A] Inversion Chromosome Map")
    
    hap_df = data['haplotype_effects']
    
    # Get unique inversions
    inv_col = 'Marker'
    inversions = hap_df[inv_col].unique()
    log(f"  Found {len(inversions)} inversions: {inversions}", "DEBUG")
    
    # Parse chromosome numbers from inversion names
    inv_data = []
    for inv in inversions:
        try:
            parts = inv.replace('miinv', '').split('.')
            chrom = int(parts[0])
            sub_idx = int(parts[1]) if len(parts) > 1 else 0
            inv_data.append({
                'marker': inv,
                'chromosome': chrom,
                'sub_index': sub_idx,
            })
        except:
            log(f"  Could not parse: {inv}", "WARN")
    
    inv_df = pd.DataFrame(inv_data)
    inv_df = inv_df.sort_values(['chromosome', 'sub_index'])
    
    # Key inversions to highlight
    key_inversions = {
        'miinv6.0': {'trait': 'FBC', 'effect': '+1.12 SD'},
        'miinv11.0': {'trait': 'TC', 'effect': '-1.73 SD'},
        'miinv17.0': {'trait': 'AFW', 'effect': '-1.10 SD'},
    }
    
    # HORIZONTAL LAYOUT: Chromosomes as vertical bars arranged horizontally
    chrom_width = 0.6
    chrom_height = 3.8
    chrom_spacing = 1.1
    
    # Only show chromosomes that have inversions, plus a few for context
    chroms_with_inv = set(inv_df['chromosome'].unique())
    
    # Draw chromosomes horizontally
    x_positions = {}
    x_current = 0.5
    
    for i in range(1, 21):
        x_positions[i] = x_current
        
        # Chromosome background
        color = colors.chrom_colors_light if i % 2 == 0 else colors.chrom_colors_dark
        alpha = 0.9 if i in chroms_with_inv else 0.3
        
        # Draw chromosome as rounded rectangle (vertical)
        chrom_rect = FancyBboxPatch(
            (x_current - chrom_width/2, 0), chrom_width, chrom_height,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            facecolor=color, edgecolor=colors.darkgray if i in chroms_with_inv else colors.lightgray, 
            linewidth=0.8 if i in chroms_with_inv else 0.4,
            alpha=alpha
        )
        ax.add_patch(chrom_rect)
        
        # Chromosome label below
        ax.text(x_current, -0.25, f'{i}', ha='center', va='top', 
               fontsize=7, fontweight='bold' if i in chroms_with_inv else 'normal',
               color='black' if i in chroms_with_inv else colors.gray)
        
        x_current += chrom_spacing
    
    # Place inversions on chromosomes
    # Track y-positions for multiple inversions on same chromosome
    chrom_inv_count = {}
    
    for _, row in inv_df.iterrows():
        chrom = row['chromosome']
        marker = row['marker']
        sub_idx = row['sub_index']
        
        if chrom not in chrom_inv_count:
            chrom_inv_count[chrom] = 0
        
        x_pos = x_positions[chrom]
        # Stagger y positions for multiple inversions on same chromosome
        y_offset = 0.5 + chrom_inv_count[chrom] * 0.7
        chrom_inv_count[chrom] += 1
        
        is_key = marker in key_inversions
        
        if is_key:
            trait = key_inversions[marker]['trait']
            color = colors.trait_colors.get(trait, colors.limegreen)
            
            # Draw diamond marker for key inversions
            diamond = plt.Polygon([
                (x_pos, y_offset + 0.15),
                (x_pos + 0.12, y_offset),
                (x_pos, y_offset - 0.15),
                (x_pos - 0.12, y_offset),
            ], facecolor=color, edgecolor='black', linewidth=1.5, zorder=10)
            ax.add_patch(diamond)
            
            # Add label to the right with offset
            effect_text = key_inversions[marker]['effect']
            ax.annotate(
                f'{marker}\n{effect_text}',
                xy=(x_pos + 0.2, y_offset),
                fontsize=7, fontweight='bold', ha='left', va='center',
                color=color,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', 
                         edgecolor=color, alpha=0.95, linewidth=1.5)
            )
        else:
            # Regular inversion - small circle
            marker_dot = plt.Circle((x_pos, y_offset), 0.08, 
                                   facecolor=colors.seagreen, edgecolor='white',
                                   linewidth=0.8, alpha=0.85, zorder=5)
            ax.add_patch(marker_dot)
    
    # Styling - TIGHTER X-LIMITS to reduce whitespace
    ax.set_xlim(0, x_current - 0.2)
    ax.set_ylim(-0.7, chrom_height + 0.4)
    ax.axis('off')
    
    # X-axis label
    ax.text(x_current/2, -0.5, 'Chromosome', ha='center', va='top', 
           fontsize=9, fontweight='bold')
    
    # Legend - compact horizontal (NS = non-significant inversions)
    legend_elements = [
        mpatches.Patch(facecolor=colors.trait_colors['FBC'], edgecolor='black', 
                      linewidth=1.5, label='FBC'),
        mpatches.Patch(facecolor=colors.trait_colors['TC'], edgecolor='black', 
                      linewidth=1.5, label='TC'),
        mpatches.Patch(facecolor=colors.trait_colors['AFW'], edgecolor='black', 
                      linewidth=1.5, label='AFW'),
        mpatches.Patch(facecolor=colors.seagreen, edgecolor='white',
                      label='NS'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, 
             frameon=True, fancybox=True, ncol=4)
    
    # Panel label
    ax.text(-0.02, 1.02, 'A', transform=ax.transAxes,
           fontsize=16, fontweight='bold', va='bottom', ha='left')
    
    log("  Panel A complete", "OK")

# ============================================================================
# PANEL B: INVERSION VS RANDOM PANEL COMPARISON
# ============================================================================

def create_panel_B(data, ax):
    """
    Panel B: Violin/box plot comparing inversion-only models vs random 17-SNP panels.
    
    DATA STRUCTURE (from inventory):
    - random_vs_inversion_replicates.csv: trait, scheme, model, n_markers, replicate, mean_r_random
      (Contains 100 replicates of random 17-SNP panels per trait/scheme/model)
    - random_vs_inversion_summary.csv: Contains inversion_mean_r for comparison
    
    NO FALLBACK - uses actual data only.
    """
    log("\n[PANEL B] Inversion vs Random Panel Efficiency")
    
    rvi_df = data['random_vs_inv']  # replicates file
    
    log(f"  Columns: {list(rvi_df.columns)}", "DEBUG")
    log(f"  Unique traits: {rvi_df['trait'].unique()}", "DEBUG")
    log(f"  Unique schemes: {rvi_df['scheme'].unique()}", "DEBUG")
    log(f"  Unique models: {rvi_df['model'].unique()}", "DEBUG")
    
    # Try to load the summary file which has inversion accuracies
    summary_path = os.path.join(OUTPUT_DIR, "idea_2", "random_control", "random_vs_inversion_summary.csv")
    
    if os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path)
        log(f"  Loaded summary file: {summary_df.shape}", "DEBUG")
        log(f"  Summary columns: {list(summary_df.columns)}", "DEBUG")
        has_summary = True
    else:
        log(f"  Summary file not found: {summary_path}", "WARN")
        has_summary = False
    
    # Focus on key traits and best scheme (cv_random_k5 for fair comparison)
    target_traits = ['FBC', 'AFW']
    target_scheme = 'cv_random_k5'
    
    # Check if scheme exists, otherwise use first available
    available_schemes = rvi_df['scheme'].unique()
    if target_scheme not in available_schemes:
        target_scheme = available_schemes[0]
        log(f"  Using scheme: {target_scheme}", "DEBUG")
    
    # Prepare data for plotting
    plot_data = []
    
    for trait in target_traits:
        # Get random panel replicates for this trait and scheme
        mask = (rvi_df['trait'] == trait) & (rvi_df['scheme'] == target_scheme)
        trait_data = rvi_df[mask]
        
        if len(trait_data) == 0:
            log(f"  No data for {trait} under {target_scheme}", "WARN")
            continue
        
        # Random panel results (100 replicates)
        random_vals = trait_data['mean_r_random'].dropna().values
        log(f"  {trait}: {len(random_vals)} random replicates, mean={np.mean(random_vals):.3f}", "DEBUG")
        
        for v in random_vals:
            plot_data.append({'Trait': trait, 'Panel': 'Random-17', 'Accuracy': v})
        
        # Get inversion accuracy from summary file
        if has_summary:
            inv_mask = (summary_df['trait'] == trait) & (summary_df['scheme'] == target_scheme)
            inv_data = summary_df[inv_mask]
            
            if len(inv_data) > 0 and 'inversion_mean_r' in inv_data.columns:
                # Get best model's inversion accuracy
                inv_acc = inv_data['inversion_mean_r'].max()
                log(f"  {trait}: inversion accuracy = {inv_acc:.3f}", "DEBUG")
                
                # Add as single point (or replicate to show as distribution)
                # Since inversion is deterministic, we add it as a single value
                # but we need multiple points for violin plot - use the CV fold results
                # For visualization, we can repeat the mean or add slight jitter
                plot_data.append({'Trait': trait, 'Panel': 'Inversion', 'Accuracy': inv_acc})
            else:
                log(f"  No inversion_mean_r found for {trait}", "WARN")
    
    if len(plot_data) == 0:
        ax.text(0.5, 0.5, 'No comparison data available\nCheck data files', 
               ha='center', va='center', fontsize=11, transform=ax.transAxes,
               color=colors.coral_red)
        ax.text(-0.08, 1.05, 'B', transform=ax.transAxes,
               fontsize=16, fontweight='bold', va='bottom', ha='left')
        log("  NO DATA AVAILABLE - Panel B empty", "ERROR")
        return
    
    plot_df = pd.DataFrame(plot_data)
    log(f"  Plot data: {len(plot_df)} points", "DEBUG")
    
    # Color palette
    palette = {
        'Random-17': colors.gray,
        'Inversion': colors.limegreen,
    }
    
    # Create visualization
    # Use boxplot for random (many points) and scatter for inversion (single/few points)
    
    traits_in_data = plot_df['Trait'].unique()
    x_positions = {trait: i for i, trait in enumerate(traits_in_data)}
    
    # Plot random distributions as violin
    random_df = plot_df[plot_df['Panel'] == 'Random-17']
    inv_df = plot_df[plot_df['Panel'] == 'Inversion']
    
    # Violin for random panels
    if len(random_df) > 0:
        parts = ax.violinplot(
            [random_df[random_df['Trait'] == t]['Accuracy'].values for t in traits_in_data],
            positions=list(range(len(traits_in_data))),
            showmeans=True, showmedians=False, widths=0.7
        )
        
        # Style the violin
        for pc in parts['bodies']:
            pc.set_facecolor(colors.gray)
            pc.set_alpha(0.6)
            pc.set_edgecolor('black')
        
        # Handle different matplotlib versions for violin parts
        if 'cmeans' in parts:
            parts['cmeans'].set_color('black')
            parts['cmeans'].set_linewidth(2)
        if 'cbars' in parts:
            parts['cbars'].set_visible(False)
        if 'cmins' in parts:
            parts['cmins'].set_visible(False)
        if 'cmaxs' in parts:
            parts['cmaxs'].set_visible(False)
    
    # Add individual random points
    for trait in traits_in_data:
        x_base = x_positions[trait]
        trait_random = random_df[random_df['Trait'] == trait]['Accuracy'].values
        
        # Jittered scatter for random points
        jitter = np.random.normal(0, 0.08, len(trait_random))
        ax.scatter(x_base - 0.15 + jitter, trait_random, 
                  c=colors.gray, alpha=0.3, s=15, zorder=2)
        
        # Mean line and annotation for random
        mean_random = np.mean(trait_random)
        ax.hlines(mean_random, x_base - 0.35, x_base + 0.05, 
                 colors='black', linestyles='--', linewidth=2, zorder=3)
        ax.text(x_base - 0.4, mean_random, f'{mean_random:.2f}', 
               ha='right', va='center', fontsize=9, fontweight='bold', color=colors.darkgray)
    
    # Plot inversion values as large diamonds
    for trait in traits_in_data:
        x_base = x_positions[trait]
        trait_inv = inv_df[inv_df['Trait'] == trait]['Accuracy'].values
        
        if len(trait_inv) > 0:
            inv_val = trait_inv[0]  # Should be single value
            ax.scatter(x_base, inv_val, 
                      c=colors.limegreen, s=200, marker='D', 
                      edgecolors='black', linewidth=2, zorder=5,
                      label='Inversion' if trait == traits_in_data[0] else '')
            
            # Annotation
            ax.text(x_base + 0.2, inv_val, f'{inv_val:.2f}', 
                   ha='left', va='center', fontsize=10, fontweight='bold', 
                   color='black')
    
    # Styling
    ax.set_xticks(range(len(traits_in_data)))
    ax.set_xticklabels(traits_in_data, fontsize=11, fontweight='bold')
    ax.set_ylabel('Prediction Accuracy (r)', fontsize=11, fontweight='bold')
    
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        mpatches.Patch(facecolor=colors.gray, edgecolor='black', alpha=0.6, label='Random-17 (n=100)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=colors.limegreen, 
               markersize=12, markeredgecolor='black', markeredgewidth=2, label='Inversion Panel'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.95)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_xlim(-0.6, len(traits_in_data) - 0.4)
    
    # Panel label
    ax.text(-0.12, 1.05, 'B', transform=ax.transAxes,
           fontsize=16, fontweight='bold', va='bottom', ha='left')
    
    log("  Panel B complete", "OK")

# ============================================================================
# PANEL C: ADDITIVE SUPER-GENE EFFECTS
# ============================================================================

def create_panel_C(data, ax):
    """
    Panel C: Boxplots showing phenotype values across genotype classes (0/1/2)
    for key inversions: miinv6.0→FBC, miinv11.0→TC, miinv17.0→AFW
    """
    log("\n[PANEL C] Additive Super-Gene Effects")
    
    meta_df = data['meta_core']
    pheno_df = data['pheno_core']
    hap_df = data['haplotype_effects']
    
    # Merge phenotype and genotype data
    id_col_meta = 'ID' if 'ID' in meta_df.columns else meta_df.columns[0]
    id_col_pheno = 'ID' if 'ID' in pheno_df.columns else pheno_df.columns[0]
    
    merged = pd.merge(pheno_df, meta_df, left_on=id_col_pheno, right_on=id_col_meta, how='inner')
    log(f"  Merged data shape: {merged.shape}", "DEBUG")
    
    # Key inversion-trait pairs - USE TRAIT CODES for labels
    inv_trait_pairs = [
        ('miinv6.0', 'FBC', 'FBC'),
        ('miinv11.0', 'TC', 'TC'),
        ('miinv17.0', 'AFW', 'AFW'),
    ]
    
    available_inversions = [col for col in merged.columns if col.startswith('miinv')]
    log(f"  Available inversions: {available_inversions}", "DEBUG")
    
    plot_data_all = []
    valid_pairs = []
    
    for inv, trait, trait_short in inv_trait_pairs:
        if inv in merged.columns and trait in merged.columns:
            valid_pairs.append((inv, trait, trait_short))
            temp = merged[[inv, trait]].dropna()
            temp['Genotype'] = temp[inv].astype(int)
            temp['Phenotype'] = temp[trait]
            temp['Pair'] = f'{trait_short}\n({inv})'
            temp['Trait'] = trait
            plot_data_all.append(temp[['Genotype', 'Phenotype', 'Pair', 'Trait']])
        else:
            log(f"  Missing: {inv} or {trait}", "WARN")
    
    if len(plot_data_all) == 0:
        ax.text(0.5, 0.5, 'Inversion genotype data\nnot available', 
               ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.text(-0.08, 1.05, 'C', transform=ax.transAxes,
               fontsize=16, fontweight='bold', va='bottom', ha='left')
        return
    
    plot_df = pd.concat(plot_data_all, ignore_index=True)
    
    pairs = plot_df['Pair'].unique()
    
    # Custom color palette for genotypes (light to dark green)
    genotype_palette = {0: '#C8E6C9', 1: '#66BB6A', 2: '#2E7D32'}
    
    # Boxplot
    sns.boxplot(data=plot_df, x='Pair', y='Phenotype', hue='Genotype',
               palette=genotype_palette, ax=ax, linewidth=1.5,
               flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.5})
    
    # Add strip plot
    sns.stripplot(data=plot_df, x='Pair', y='Phenotype', hue='Genotype',
                 palette=genotype_palette, ax=ax, dodge=True, alpha=0.4,
                 size=3, legend=False)
    
    # Calculate effect sizes and add annotations FOR ALL PAIRS
    for i, pair in enumerate(pairs):
        pair_data = plot_df[plot_df['Pair'] == pair]
        trait = pair_data['Trait'].iloc[0]
        
        # Calculate means per genotype
        means = pair_data.groupby('Genotype')['Phenotype'].mean()
        
        # Get effect from G0 to G2
        if 0 in means.index and 2 in means.index:
            effect = means[2] - means[0]
            sd = pair_data['Phenotype'].std()
            effect_std = effect / sd if sd > 0 else 0
            
            # Add effect annotation above - USE TRAIT COLOR
            y_max = pair_data['Phenotype'].max()
            y_range = pair_data['Phenotype'].max() - pair_data['Phenotype'].min()
            sign = '+' if effect_std > 0 else ''
            
            # Get trait color for annotation
            trait_color = colors.trait_colors.get(trait, colors.seagreen)
            
            ax.text(i, y_max + (y_range * 0.12), f'Δ = {sign}{effect_std:.2f} SD',
                   ha='center', va='bottom', fontsize=10, fontweight='bold',
                   color=trait_color,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                            edgecolor=trait_color, alpha=0.9, linewidth=1.5))
            
            log(f"  {pair}: effect = {effect_std:.2f} SD", "DEBUG")
    
    # Styling
    ax.set_xlabel('')
    ax.set_ylabel('Phenotype Value', fontsize=11, fontweight='bold')
    
    # Update legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:3], ['G0 (Ref)', 'G1 (Het)', 'G2 (Alt)'], 
             title='Genotype', loc='upper right', fontsize=9)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Panel label
    ax.text(-0.1, 1.05, 'C', transform=ax.transAxes,
           fontsize=16, fontweight='bold', va='bottom', ha='left')
    
    log("  Panel C complete", "OK")

# ============================================================================
# PANEL D: MODEL COMPARISON ACROSS CV SCHEMES
# ============================================================================

def create_panel_D(data, ax):
    """
    Panel D: Grouped bar chart comparing Dense SNP, Inversion-only, and Random-17
    panel accuracy under different CV schemes.
    """
    log("\n[PANEL D] Model Comparison Across CV Schemes")
    
    perf_df = data['model_perf']
    
    log(f"  Columns: {list(perf_df.columns)}", "DEBUG")
    log(f"  Unique traits: {perf_df['trait'].unique()}", "DEBUG")
    log(f"  Unique schemes: {perf_df['scheme'].unique()}", "DEBUG")
    log(f"  Unique feature sets: {perf_df['feature_set'].unique() if 'feature_set' in perf_df.columns else 'N/A'}", "DEBUG")
    
    # Focus on FBC - best trait for inversion comparison
    target_trait = 'FBC'
    trait_data = perf_df[perf_df['trait'] == target_trait].copy()
    
    if len(trait_data) == 0:
        ax.text(0.5, 0.5, 'No model performance\ndata for FBC', 
               ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.text(-0.15, 1.05, 'D', transform=ax.transAxes,
               fontsize=16, fontweight='bold', va='bottom', ha='left')
        return
    
    # Get CV schemes
    schemes = trait_data['scheme'].unique()
    
    # Prepare summary by scheme and model type
    summary_data = []
    
    for scheme in schemes:
        scheme_data = trait_data[trait_data['scheme'] == scheme]
        
        if 'feature_set' in scheme_data.columns:
            for feat in scheme_data['feature_set'].unique():
                feat_data = scheme_data[scheme_data['feature_set'] == feat]
                if len(feat_data) > 0:
                    best_r = feat_data['mean_r'].max()
                    summary_data.append({
                        'Scheme': scheme.replace('cv_', '').replace('_', '\n'),
                        'Feature': feat,
                        'Accuracy': best_r
                    })
        else:
            best_r = scheme_data['mean_r'].max()
            summary_data.append({
                'Scheme': scheme.replace('cv_', '').replace('_', '\n'),
                'Feature': 'All',
                'Accuracy': best_r
            })
    
    if len(summary_data) == 0:
        ax.text(0.5, 0.5, 'Could not extract\nmodel comparison data',
               ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.text(-0.15, 1.05, 'D', transform=ax.transAxes,
               fontsize=16, fontweight='bold', va='bottom', ha='left')
        return
    
    summary_df = pd.DataFrame(summary_data)
    log(f"  Summary data:\n{summary_df}", "DEBUG")
    
    # Create grouped bar plot
    schemes_unique = summary_df['Scheme'].unique()
    features_unique = summary_df['Feature'].unique()
    
    x = np.arange(len(schemes_unique))
    n_features = len(features_unique)
    width = 0.7 / n_features  # Adjust width based on number of features
    
    feature_colors = {
        'snp': colors.royalblue,
        'inv': colors.limegreen,
        'snp+inv': colors.seagreen,
    }
    
    # Cleaner feature labels
    feature_labels = {
        'snp': 'SNP',
        'inv': 'Inversion',
        'snp+inv': 'Combined',
    }
    
    for i, feat in enumerate(features_unique):
        feat_data = summary_df[summary_df['Feature'] == feat]
        values = [feat_data[feat_data['Scheme'] == s]['Accuracy'].values[0] 
                 if len(feat_data[feat_data['Scheme'] == s]) > 0 else 0 
                 for s in schemes_unique]
        
        color = feature_colors.get(feat, colors.gray)
        label = feature_labels.get(feat, feat)
        offset = (i - n_features/2 + 0.5) * width
        
        bars = ax.bar(x + offset, values, width * 0.9, label=label, color=color,
                     edgecolor='white', linewidth=1.5)
        
        # Add value labels on top
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=8,
                       fontweight='bold')
    
    # Styling
    ax.set_xticks(x)
    ax.set_xticklabels(schemes_unique, fontsize=9)
    ax.set_ylabel('Accuracy (r)', fontsize=10, fontweight='bold')
    
    # Legend BELOW the plot or at bottom
    ax.legend(title='Features', loc='upper left', fontsize=8, framealpha=0.95)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 0.85)
    
    # Panel label
    ax.text(-0.18, 1.05, 'D', transform=ax.transAxes,
           fontsize=16, fontweight='bold', va='bottom', ha='left')
    
    log("  Panel D complete", "OK")

# ============================================================================
# PANEL E: EFFECT SIZE CATALOGUE
# ============================================================================

def create_panel_E(data, ax):
    """
    Panel E: Horizontal bar chart showing standardized effect sizes for top inversions.
    """
    log("\n[PANEL E] Effect Size Catalogue")
    
    hap_df = data['haplotype_effects']
    
    log(f"  Columns: {list(hap_df.columns)}", "DEBUG")
    
    # Column mapping
    trait_col = 'Trait'
    marker_col = 'Marker'
    effect_col = 'Effect_Std'
    
    # Check if Effect_Std exists, otherwise calculate from raw
    if effect_col not in hap_df.columns:
        if 'Effect_Raw' in hap_df.columns and 'Global_SD' in hap_df.columns:
            hap_df[effect_col] = hap_df['Effect_Raw'] / hap_df['Global_SD']
        else:
            log("  Cannot find or calculate Effect_Std", "ERROR")
            ax.text(0.5, 0.5, 'Effect size data\nnot available', 
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            return
    
    # Get top effects per trait
    hap_df = hap_df.copy()
    hap_df['abs_effect'] = hap_df[effect_col].abs()
    
    # Select top 2 inversions per trait
    top_effects = []
    for trait in colors.trait_order:
        trait_data = hap_df[hap_df[trait_col] == trait].nlargest(2, 'abs_effect')
        top_effects.append(trait_data)
    
    top_df = pd.concat(top_effects)
    top_df = top_df.sort_values('abs_effect', ascending=True)
    
    log(f"  Top effects:\n{top_df[[trait_col, marker_col, effect_col]]}", "DEBUG")
    
    # Create SHORT labels (marker only, trait shown by color)
    top_df['label'] = top_df[marker_col]
    
    # Create horizontal bars
    y_pos = np.arange(len(top_df))
    effects = top_df[effect_col].values
    traits = top_df[trait_col].values
    bar_colors = [colors.trait_colors.get(t, colors.gray) for t in traits]
    
    # Diverging bars from 0
    bars = ax.barh(y_pos, effects, height=0.65, color=bar_colors,
                   edgecolor='white', linewidth=1.2)
    
    # Add value labels at end of bars
    for i, (bar, effect, trait) in enumerate(zip(bars, effects, traits)):
        # Position label outside bar
        x_pos = effect + (0.08 if effect > 0 else -0.08)
        ha = 'left' if effect > 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
               f'{effect:+.2f}', ha=ha, va='center', fontsize=8, fontweight='bold',
               color=bar_colors[i])
    
    # Reference line at 0
    ax.axvline(x=0, color='black', linewidth=1.2)
    
    # Styling - use SHORT labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_df['label'], fontsize=8)
    ax.set_xlabel('Effect (SD)', fontsize=10, fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Symmetric x-axis
    max_abs = max(abs(effects.min()), abs(effects.max()))
    ax.set_xlim(-max_abs * 1.35, max_abs * 1.35)
    
    # Add trait color legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=colors.trait_colors[t], lw=6, label=t) 
                      for t in colors.trait_order if t in traits]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=7, 
             framealpha=0.95, ncol=2)
    
    # Panel label
    ax.text(-0.18, 1.05, 'E', transform=ax.transAxes,
           fontsize=16, fontweight='bold', va='bottom', ha='left')
    
    log("  Panel E complete", "OK")

# ============================================================================
# PANEL F: EXPECTED GENETIC GAIN
# ============================================================================

def create_panel_F(data, ax):
    """
    Panel F: Bar chart showing expected genetic gain per selection cycle.
    """
    log("\n[PANEL F] Expected Genetic Gain")
    
    gain_df = data['genetic_gain']
    
    log(f"  Columns: {list(gain_df.columns)}", "DEBUG")
    log(f"  Data:\n{gain_df}", "DEBUG")
    
    # Column mapping
    trait_col = 'Trait'
    gain_col = 'Gain_Percent'
    
    # Filter for within-population scenario if available
    if 'Scenario' in gain_df.columns:
        # Prefer within-population gains
        within_mask = gain_df['Scenario'].str.contains('Within|within|Random', case=False, na=False)
        if within_mask.any():
            gain_df = gain_df[within_mask]
    
    # Order by trait
    gain_df = gain_df.set_index(trait_col).reindex(colors.trait_order).reset_index().dropna()
    
    traits = gain_df[trait_col].values
    gains = gain_df[gain_col].values
    
    # Create bars
    x_pos = np.arange(len(traits))
    bar_colors = [colors.trait_colors.get(t, colors.gray) for t in traits]
    
    bars = ax.bar(x_pos, gains, width=0.7, color=bar_colors,
                 edgecolor='white', linewidth=1.5)
    
    # Add value labels
    for bar, gain in zip(bars, gains):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
               f'{gain:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Styling - USE TRAIT CODES not long names
    ax.set_xticks(x_pos)
    ax.set_xticklabels(traits, fontsize=9, fontweight='bold')
    ax.set_ylabel('Gain per Cycle (%)', fontsize=10, fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(gains) * 1.25 if len(gains) > 0 else 25)
    
    # Panel label
    ax.text(-0.2, 1.05, 'F', transform=ax.transAxes,
           fontsize=16, fontweight='bold', va='bottom', ha='left')
    
    log("  Panel F complete", "OK")

# ============================================================================
# MAIN FIGURE ASSEMBLY
# ============================================================================

def create_figure_2():
    """
    Assemble all panels into Figure 2 with variable panel sizes.
    
    Layout:
        Row 0: A (full width) - Chromosome ideogram (shorter height)
        Row 1: B (3 cols) + C (3 cols) - Main evidence
        Row 2: D (2 cols) + E (2.5 cols) + F (1.5 cols) - Supporting
    """
    log("\n" + "=" * 70)
    log("ASSEMBLING FIGURE 2")
    log("=" * 70)
    
    # Load data
    data = load_data()
    
    # Create figure with custom gridspec
    fig = plt.figure(figsize=(13, 11))
    
    gs = gridspec.GridSpec(
        nrows=3, ncols=6,
        height_ratios=[0.7, 1.3, 1.0],  # Reduced height for Panel A
        width_ratios=[1, 1, 1, 1, 1, 1],
        hspace=0.35,  # Increased from 0.25 to add space between rows 2 and 3
        wspace=0.6   # Increased from 0.4 to add gap between D, E, F
    )
    
    # Create axes
    ax_A = fig.add_subplot(gs[0, 0:6])  # Row 0, full width
    ax_B = fig.add_subplot(gs[1, 0:3])  # Row 1, left half
    ax_C = fig.add_subplot(gs[1, 3:6])  # Row 1, right half
    ax_D = fig.add_subplot(gs[2, 0:2])  # Row 2, cols 0-1
    ax_E = fig.add_subplot(gs[2, 2:4])  # Row 2, cols 2-3 (wider for labels)
    ax_F = fig.add_subplot(gs[2, 4:6])  # Row 2, cols 4-5
    
    # Create panels
    log("\nCreating panels...")
    
    create_panel_A(data, ax_A)
    create_panel_B(data, ax_B)
    create_panel_C(data, ax_C)
    create_panel_D(data, ax_D)
    create_panel_E(data, ax_E)
    create_panel_F(data, ax_F)
    
    # Save figure
    output_path = os.path.join(FIGURES_DIR, "figure_2.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    log(f"\n[OK] Figure saved: {output_path}", "OK")
    
    # Save PDF
    output_pdf = os.path.join(FIGURES_DIR, "figure_2.pdf")
    fig.savefig(output_pdf, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    log(f"[OK] PDF saved: {output_pdf}", "OK")
    
    # Save source data
    log("\nSaving source data...")
    
    hap_out = os.path.join(FIGURE_SUBDIR, "panel_A_E_haplotype_effects.csv")
    data['haplotype_effects'].to_csv(hap_out, index=False)
    log(f"  [OK] Panel A/E source: {hap_out}", "OK")
    
    gain_out = os.path.join(FIGURE_SUBDIR, "panel_F_genetic_gain.csv")
    data['genetic_gain'].to_csv(gain_out, index=False)
    log(f"  [OK] Panel F source: {gain_out}", "OK")
    
    log("\n" + "=" * 70)
    log("FIGURE 2 COMPLETE!", "OK")
    log("=" * 70)
    
    return fig

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MANGO GWAS - FIGURE 2 GENERATOR")
    print("Structural Haplotypes as Ultra-Efficient Predictors")
    print("=" * 70)
    print(f"\nOutput directory: {FIGURES_DIR}")
    print("-" * 70)
    
    try:
        fig = create_figure_2()
        
        print("\n" + "=" * 70)
        print("SUCCESS!")
        print(f"  Figure: {os.path.join(FIGURES_DIR, 'figure_2.png')}")
        print(f"  Data:   {FIGURE_SUBDIR}")
        print("=" * 70 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] File not found: {e}")
        print("\nPlease verify the data files exist at the expected paths.")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()