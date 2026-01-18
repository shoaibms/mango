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
    LCO_PER_CLUSTER = os.path.join(OUTPUT_DIR, "idea_1", "cv_structure", "table_s14_lco_per_cluster.csv")

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
    
    # 8. LCO per-cluster results
    log(f"Loading LCO per-cluster: {DataPaths.LCO_PER_CLUSTER}")
    if os.path.exists(DataPaths.LCO_PER_CLUSTER):
        data['lco_per_cluster'] = pd.read_csv(DataPaths.LCO_PER_CLUSTER)
        log(f"  Shape: {data['lco_per_cluster'].shape}", "OK")
    else:
        log("  LCO per-cluster not found - Panel D will use fallback", "WARN")
        data['lco_per_cluster'] = None
    
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
            showmeans=False, showmedians=True, widths=0.7
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
    random_point_color = "#75d86d"
    for trait in traits_in_data:
        x_base = x_positions[trait]
        trait_random = random_df[random_df['Trait'] == trait]['Accuracy'].values
        
        # Jittered scatter for random points
        jitter = np.random.normal(0, 0.08, len(trait_random))
        ax.scatter(x_base - 0.15 + jitter, trait_random, 
                  c=random_point_color, alpha=0.3, s=15, zorder=2)
        
        # Median line and annotation for random
        median_random = np.median(trait_random)
        ax.hlines(median_random, x_base - 0.35, x_base + 0.05, 
                 colors='black', linestyles='-', linewidth=2, zorder=3)
        ax.text(x_base - 0.4, median_random, f'{median_random:.2f}', 
               ha='right', va='center', fontsize=9, fontweight='bold', color='black')
    
    # Plot inversion values as large diamonds
    inv_point_color = '#4ec27a'
    for trait in traits_in_data:
        x_base = x_positions[trait]
        trait_inv = inv_df[inv_df['Trait'] == trait]['Accuracy'].values
        
        if len(trait_inv) > 0:
            inv_val = trait_inv[0]  # Should be single value
            ax.scatter(x_base, inv_val, 
                      c=inv_point_color, s=200, marker='D', 
                      edgecolors='black', linewidth=2, zorder=5,
                      label='Inversion' if trait == traits_in_data[0] else '')
            
            # Annotation
            ax.text(x_base + 0.25, inv_val, f'{inv_val:.2f}', 
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
    Panel C: Reaction norm plot showing phenotype values across genotype dosage (0/1/2)
    for key inversions: miinv6.0→FBC, miinv11.0→TC, miinv17.0→AFW.
    Shows additive expectation (dashed) vs actual heterozygote position to visualize dominance.
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import t
    
    log("\n[PANEL C] Reaction Norm - Additive vs Dominance")
    
    meta_df = data['meta_core']
    pheno_df = data['pheno_core']
    
    # Merge phenotype and genotype data
    id_col_meta = 'ID' if 'ID' in meta_df.columns else meta_df.columns[0]
    id_col_pheno = 'ID' if 'ID' in pheno_df.columns else pheno_df.columns[0]
    
    merged = pd.merge(pheno_df, meta_df, left_on=id_col_pheno, right_on=id_col_meta, how='inner')
    log(f"  Merged data shape: {merged.shape}", "DEBUG")
    
    # Key inversion-trait pairs
    inv_trait_pairs = [
        ('miinv6.0', 'FBC', 'FBC'),
        ('miinv11.0', 'TC', 'TC'),
        ('miinv17.0', 'AFW', 'AFW'),
    ]
    
    available_inversions = [col for col in merged.columns if col.startswith('miinv')]
    log(f"  Available inversions: {available_inversions}", "DEBUG")
    
    valid_pairs = []
    pair_data_dict = {}
    
    for inv, trait, trait_short in inv_trait_pairs:
        if inv in merged.columns and trait in merged.columns:
            valid_pairs.append((inv, trait, trait_short))
            temp = merged[[inv, trait]].dropna()
            temp['Genotype'] = temp[inv].astype(int)
            temp['Phenotype'] = temp[trait]
            pair_data_dict[(inv, trait)] = temp
        else:
            log(f"  Missing: {inv} or {trait}", "WARN")
    
    if len(valid_pairs) == 0:
        ax.text(0.5, 0.5, 'Inversion genotype data\nnot available', 
               ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.text(-0.08, 1.05, 'C', transform=ax.transAxes,
               fontsize=16, fontweight='bold', va='bottom', ha='left')
        return
    
    pair_width = 2.5  # Width allocated to each trait group
    genotype_positions = np.array([0, 1, 2])  # Dosage scale
    genotype_colors = {0: '#C8E6C9', 1: '#66BB6A', 2: '#2E7D32'}
    
    all_y_values = []
    rng = np.random.default_rng(42)
    plotted_pairs = []
    
    for _, (inv, trait, trait_short) in enumerate(valid_pairs):
        pair_data = pair_data_dict[(inv, trait)]
        
        # Calculate statistics per genotype
        stats_per_geno = {}
        for geno in [0, 1, 2]:
            geno_data = pair_data[pair_data['Genotype'] == geno]['Phenotype']
            if len(geno_data) > 0:
                mean = geno_data.mean()
                n = len(geno_data)
                sem = geno_data.sem()
                ci95 = t.ppf(0.975, df=n - 1) * sem if n > 1 else 0.0
                stats_per_geno[geno] = {'mean': mean, 'sem': sem, 'ci95': ci95, 'n': n, 'values': geno_data.values}
                all_y_values.extend(geno_data.values)
        
        # Skip if missing genotypes
        if not all(g in stats_per_geno for g in [0, 1, 2]):
            log(f"  Missing genotype classes for {trait}", "WARN")
            continue
        
        pair_idx = len(plotted_pairs)
        plotted_pairs.append((inv, trait, trait_short))
        x_offset = pair_idx * pair_width
        
        trait_color = colors.trait_colors.get(trait, colors.seagreen)
        
        # Individual points with jitter
        for geno in [0, 1, 2]:
            values = stats_per_geno[geno]['values']
            jitter = rng.normal(0, 0.08, len(values))
            x_positions = x_offset + geno + jitter
            ax.scatter(x_positions, values, 
                      c=genotype_colors[geno], 
                      alpha=0.4, s=25, 
                      edgecolors='white', linewidth=0.3,
                      zorder=2)
        
        # Additive expectation line (dashed) between G0 and G2
        mean_g0 = stats_per_geno[0]['mean']
        mean_g2 = stats_per_geno[2]['mean']
        expected_g1 = (mean_g0 + mean_g2) / 2
        ax.plot([x_offset + 0, x_offset + 2], 
               [mean_g0, mean_g2],
               linestyle='--', color='gray', linewidth=2, alpha=0.7,
               zorder=3, label='Additive expectation' if pair_idx == 0 else '')
        ax.scatter([x_offset + 1], [expected_g1], 
                  marker='o', s=60, facecolors='none', 
                  edgecolors='gray', linewidth=2, zorder=4)
        
        # Actual means with error bars
        actual_means = [stats_per_geno[g]['mean'] for g in [0, 1, 2]]
        actual_ci = [stats_per_geno[g]['ci95'] for g in [0, 1, 2]]
        x_geno = [x_offset + g for g in [0, 1, 2]]
        ax.plot(x_geno, actual_means, 
               linestyle='-', color=trait_color, linewidth=2.5,
               zorder=5, marker='o', markersize=10,
               markerfacecolor=trait_color, markeredgecolor='white',
               markeredgewidth=2, label='Actual means' if pair_idx == 0 else '')
        ax.errorbar(x_geno, actual_means, yerr=actual_ci,
                   fmt='none', ecolor=trait_color, elinewidth=2,
                   capsize=4, capthick=2, zorder=6)
        
        # Effect size and dominance
        effect = mean_g2 - mean_g0
        sd = pair_data['Phenotype'].std()
        effect_std = effect / sd if sd > 0 else 0
        actual_g1 = stats_per_geno[1]['mean']
        half_effect = abs(effect) / 2
        dominance_d = (actual_g1 - expected_g1) / half_effect if half_effect > 0 else 0
        
        y_min_pair = pair_data['Phenotype'].min()
        y_max_pair = pair_data['Phenotype'].max()
        y_range_pair = y_max_pair - y_min_pair if y_max_pair != y_min_pair else 1
        
        sign = '+' if effect_std > 0 else ''
        ax.text(x_offset + 1, y_max_pair + (y_range_pair * 0.08), 
               f'Δ = {sign}{effect_std:.2f} SD',
               ha='center', va='bottom', fontsize=10, fontweight='bold',
               color=trait_color,
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                        edgecolor=trait_color, alpha=0.9, linewidth=1.5))
        
        log(f"  {trait} ({inv}): effect = {effect_std:.2f} SD, dominance d = {dominance_d:.2f}", "OK")
    
    if len(plotted_pairs) == 0:
        ax.text(0.5, 0.5, 'Inversion genotype data\ninsufficient for plotting', 
               ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.text(-0.08, 1.05, 'C', transform=ax.transAxes,
               fontsize=16, fontweight='bold', va='bottom', ha='left')
        return
    
    # Axis labels and ticks
    x_centers = [idx * pair_width + 1 for idx in range(len(plotted_pairs))]
    x_labels = [f'{trait_short}\n({inv})' for inv, trait, trait_short in plotted_pairs]
    ax.set_xticks(x_centers)
    ax.set_xticklabels(x_labels, fontsize=10, fontweight='bold')
    ax.set_ylabel('Phenotype Value', fontsize=11, fontweight='bold')
    ax.set_xlabel('Genotype Dosage', fontsize=10, color='gray')
    ax.set_xlim(-0.5, len(plotted_pairs) * pair_width - 0.5)
    
    # Genotype tick labels below each group
    if all_y_values:
        y_min_global = min(all_y_values)
        y_max_global = max(all_y_values)
        y_range_global = max(y_max_global - y_min_global, 1)
    else:
        y_min_global, y_range_global = 0, 1
    
    for pair_idx in range(len(plotted_pairs)):
        x_offset = pair_idx * pair_width
        for geno in [0, 1, 2]:
            ax.text(x_offset + geno, y_min_global - y_range_global * 0.08,
                   str(geno), ha='center', va='top', fontsize=8, color='gray')
    
    # Legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    
    legend_elements = [
        Line2D([0], [0], linestyle='-', color='gray', linewidth=2,
               marker='o', markersize=6, markerfacecolor='none',
               label='Additive expectation'),
        Line2D([0], [0], linestyle='-', color='black', linewidth=2.5,
               marker='o', markersize=8, markerfacecolor='black',
               markeredgecolor='white', markeredgewidth=2,
               label='Actual means (trait color)'),
        Patch(facecolor=genotype_colors[0], edgecolor='white', label='G0 (Ref)'),
        Patch(facecolor=genotype_colors[1], edgecolor='white', label='G1 (Het)'),
        Patch(facecolor=genotype_colors[2], edgecolor='white', label='G2 (Alt)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, 
             framealpha=0.95, ncol=1)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Panel label
    ax.text(-0.1, 1.05, 'C', transform=ax.transAxes,
           fontsize=16, fontweight='bold', va='bottom', ha='left')
    
    log("  Panel C complete", "OK")

# ============================================================================
# PANEL D: PER-CLUSTER LCO ACCURACY (Revised for Reviewer Response)
# ============================================================================

def create_panel_D(data, ax):
    """
    Panel D: Per-cluster Leave-Cluster-Out accuracy showing meta-mean with CI
    and individual cluster results. Demonstrates portability is not driven by
    one "easy" cluster.
    """
    log("\n[PANEL D] Per-Cluster LCO Accuracy")
    
    lco_df = data.get('lco_per_cluster')
    
    if lco_df is None or len(lco_df) == 0:
        # Fallback to old panel
        log("  No LCO per-cluster data, using placeholder", "WARN")
        ax.text(0.5, 0.5, 'Per-cluster LCO data\nnot available', 
               ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.text(-0.18, 1.05, 'D', transform=ax.transAxes,
               fontsize=16, fontweight='bold', va='bottom', ha='left')
        return
    
    log(f"  Columns: {list(lco_df.columns)}", "DEBUG")
    
    # Filter for PC-corrected scenario and leave_cluster_out scheme
    # Adjust column names based on your actual CSV structure
    if 'scenario' in lco_df.columns:
        lco_df = lco_df[lco_df['scenario'] == 'pc_corrected'].copy()
    
    # DEBUG: Inspect CSV structure
    print("\n[DEBUG] table_s14_lco_per_cluster columns:")
    print(lco_df.columns.tolist())
    print(lco_df.head(10))
    
    # Traits to show (portable vs non-portable)
    show_traits = ['AFW', 'FBC', 'TSS', 'FF']  # Order: portable first
    
    # Cluster names mapping
    CLUSTER_NAMES = {0: 'Oceania', 1: 'Americas-SA', 2: 'SE Asia', 
                     'meta-mean': 'Meta-mean'}
    CLUSTER_SHORT = {0: 'OC', 1: 'AM', 2: 'SE', 'meta-mean': 'Mean'}
    
    # Prepare data
    plot_records = []
    for trait in show_traits:
        trait_data = lco_df[lco_df['trait'] == trait]
        if len(trait_data) == 0:
            continue
        
        for _, row in trait_data.iterrows():
            cluster = row.get('held_out_cluster', row.get('cluster', None))
            r_val = row.get('r', row.get('mean_r', np.nan))
            ci_low = row.get('ci_low', np.nan)
            ci_high = row.get('ci_high', np.nan)
            
            plot_records.append({
                'trait': trait,
                'cluster': cluster,
                'cluster_name': CLUSTER_NAMES.get(cluster, str(cluster)),
                'cluster_short': CLUSTER_SHORT.get(cluster, str(cluster)),
                'r': r_val,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'is_meta': cluster == 'meta-mean'
            })
    
    if len(plot_records) == 0:
        ax.text(0.5, 0.5, 'No valid LCO data\nfor selected traits', 
               ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.text(-0.18, 1.05, 'D', transform=ax.transAxes,
               fontsize=16, fontweight='bold', va='bottom', ha='left')
        return
    
    plot_df = pd.DataFrame(plot_records)
    
    # Plot setup
    traits_present = [t for t in show_traits if t in plot_df['trait'].values]
    n_traits = len(traits_present)
    
    # Cluster markers (small) and meta-mean (large diamond)
    cluster_markers = {0: 'o', 1: 's', 2: '^'}  # circle, square, triangle
    cluster_colors = {0: '#66BB6A', 1: '#42A5F5', 2: '#FFA726'}  # green, blue, orange
    
    x_positions = np.arange(n_traits)
    
    # Plot each cluster as small point
    for cluster in [0, 1, 2]:
        cluster_data = plot_df[(plot_df['cluster'] == cluster)]
        if len(cluster_data) == 0:
            continue
        
        y_vals = []
        for trait in traits_present:
            trait_cluster = cluster_data[cluster_data['trait'] == trait]
            if len(trait_cluster) > 0:
                y_vals.append(trait_cluster['r'].values[0])
            else:
                y_vals.append(np.nan)
        
        # Offset each cluster slightly
        offset = (cluster - 1) * 0.12
        ax.scatter(x_positions + offset, y_vals,
                  marker=cluster_markers[cluster], s=60,
                  c=cluster_colors[cluster], edgecolors='white',
                  linewidth=1, zorder=3, alpha=0.8,
                  label=CLUSTER_NAMES[cluster])
    
    # Plot meta-mean as large diamond with CI whiskers
    meta_data = plot_df[plot_df['is_meta']]
    
    # DEBUG: Print actual CI values being plotted
    print("\n[DEBUG] Panel D - CI values:")
    for trait in traits_present:
        trait_meta = meta_data[meta_data['trait'] == trait]
        if len(trait_meta) > 0:
            r = trait_meta['r'].values[0]
            ci_lo = trait_meta['ci_low'].values[0]
            ci_hi = trait_meta['ci_high'].values[0]
            print(f"  {trait}: r={r:.3f}, CI=[{ci_lo:.3f}, {ci_hi:.3f}], "
                  f"err_low={r-ci_lo:.3f}, err_high={ci_hi-r:.3f}")
    if len(meta_data) > 0:
        meta_y = []
        meta_ci_low = []
        meta_ci_high = []
        
        for trait in traits_present:
            trait_meta = meta_data[meta_data['trait'] == trait]
            if len(trait_meta) > 0:
                meta_y.append(trait_meta['r'].values[0])
                meta_ci_low.append(trait_meta['ci_low'].values[0])
                meta_ci_high.append(trait_meta['ci_high'].values[0])
            else:
                meta_y.append(np.nan)
                meta_ci_low.append(np.nan)
                meta_ci_high.append(np.nan)
        
        meta_y = np.array(meta_y)
        meta_ci_low = np.array(meta_ci_low)
        meta_ci_high = np.array(meta_ci_high)
        
        # CI whiskers
        yerr_low = meta_y - meta_ci_low
        yerr_high = meta_ci_high - meta_y
        yerr = np.array([yerr_low, yerr_high])
        
        ax.errorbar(x_positions, meta_y, yerr=yerr,
                   fmt='none', ecolor='black', elinewidth=1.5,
                   capsize=4, capthick=1.5, zorder=4)
        
        # Large diamond for meta-mean
        ax.scatter(x_positions, meta_y,
                  marker='D', s=150, c='black', edgecolors='white',
                  linewidth=2, zorder=5, label='Meta-mean')
    
    # Zero reference line
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    
    # Portable/non-portable shading (consistent with figure palette)
    portable_traits = ['AFW', 'FBC']
    nonportable_traits = ['TSS', 'FF']
    
    for i, trait in enumerate(traits_present):
        if trait in portable_traits:
            ax.axvspan(i - 0.4, i + 0.4, alpha=0.15, color=colors.limegreen, zorder=0)
        elif trait in nonportable_traits:
            ax.axvspan(i - 0.4, i + 0.4, alpha=0.15, color=colors.gray, zorder=0)
    
    # Styling
    ax.set_xticks(x_positions)
    ax.set_xticklabels(traits_present, fontsize=10, fontweight='bold')
    ax.set_ylabel('LCO Accuracy (r)', fontsize=10, fontweight='bold')
    ax.set_xlabel('Trait', fontsize=9)
    
    # Dynamic y-limits
    all_r = plot_df['r'].dropna().values
    if len(all_r) > 0:
        y_min = min(-0.4, all_r.min() - 0.1)
        y_max = max(0.5, all_r.max() + 0.1)
        ax.set_ylim(y_min, y_max)
    
    ax.set_xlim(-0.5, n_traits - 0.5)
    
    # Legend (compact)
    ax.legend(loc='upper right', fontsize=7, framealpha=0.95, ncol=2)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Annotation for portable vs non-portable
    ax.text(0.02, 0.98, 'Portable', transform=ax.transAxes, fontsize=8,
           va='top', ha='left', color=colors.seagreen, fontweight='bold', alpha=0.8)
    ax.text(0.98, 0.02, 'Non-portable', transform=ax.transAxes, fontsize=8,
           va='bottom', ha='right', color=colors.darkgray, fontweight='bold', alpha=0.8)
    
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