"""
Unified Figure Configuration for Mango GWAS Manuscript
=======================================================
Merged configuration combining:
- Color palette (blue-teal-green favorites)
- Column name mappings (based on data inventory)
- Path configuration
- Panel layouts for all 5 figures

TRAIT ORDER (standard): FBC, AFW, FF, TC, TSS

Project: Structural haplotypes function as additive super-genes
"""

import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

class PathConfig:
    """Central path configuration"""
    
    # Project root
    ROOT = r"C:\Users\ms\Desktop\mango"
    
    # Data directories
    DATA_DIR = os.path.join(ROOT, "data")
    OUTPUT_DIR = os.path.join(ROOT, "output")
    
    # Analysis output directories
    IDEA1_DIR = os.path.join(OUTPUT_DIR, "idea_1")
    IDEA2_DIR = os.path.join(OUTPUT_DIR, "idea_2")
    IDEA3_DIR = os.path.join(OUTPUT_DIR, "idea_3")
    
    # Figure output
    FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
    CODE_FIGURES_DIR = os.path.join(ROOT, "code", "figures")
    
    # Subdirectories for each figure's supporting data
    @classmethod
    def figure_subdir(cls, fig_num):
        """Get/create subdirectory for figure-specific outputs"""
        path = os.path.join(cls.FIGURES_DIR, f"figure_{fig_num}")
        os.makedirs(path, exist_ok=True)
        return path
    
    @classmethod
    def ensure_dirs(cls):
        """Create all required directories"""
        os.makedirs(cls.FIGURES_DIR, exist_ok=True)
        os.makedirs(cls.CODE_FIGURES_DIR, exist_ok=True)
        for i in range(1, 6):
            cls.figure_subdir(i)


# ============================================================================
# DATA FILE PATHS (verified from inventory)
# ============================================================================

class DataPaths:
    """All data file paths verified from inventory"""
    
    _root = PathConfig.OUTPUT_DIR
    
    # ========================================================================
    # FIGURE 1: Population Structure and Structure Cliff
    # ========================================================================
    
    # Panel A, B: PCA scores with clusters
    PC_SCORES = os.path.join(_root, "idea_1", "summary", "pc_scores_clusters.csv")
    
    # Panel B: Cluster sizes
    CLUSTER_SIZES = os.path.join(_root, "idea_1", "summary", "cluster_sizes.csv")
    
    # Panel C, E: CV transferability
    CV_TRANSFERABILITY = os.path.join(_root, "idea_1", "summary", "cv_transferability_summary.csv")
    
    # Panel D: Phenotype summary
    PHENO_SUMMARY = os.path.join(_root, "idea_1", "summary", "pheno_trait_summary.csv")
    
    # Panel F: Raw phenotypes for correlation
    PHENO_CORE = os.path.join(_root, "idea_1", "core_data", "pheno_core.csv")
    
    # ========================================================================
    # FIGURE 2: Structural Haplotypes
    # ========================================================================
    
    # Panel A, E: Haplotype effects
    HAPLOTYPE_EFFECTS = os.path.join(_root, "idea_2", "breeder_tools", "Breeder_Haplotype_Effects.csv")
    
    # Panel A, 5D: Assay design
    ASSAY_DESIGN = os.path.join(_root, "idea_2", "breeder_tools", "Supplementary_Table_Assay_Design.csv")
    
    # Panel B: Random vs inversion comparison
    RANDOM_VS_INVERSION = os.path.join(_root, "idea_2", "random_control", "random_vs_inversion_replicates.csv")
    
    # Panel C: Inversion genotypes
    META_CORE = os.path.join(_root, "idea_1", "core_data", "meta_core.csv")
    
    # Panel D: Model performance
    MODEL_PERF_IDEA2 = os.path.join(_root, "idea_2", "summary", "idea2_gs_model_performance_clean.csv")
    
    # Panel F, 5C: Genetic gain
    GENETIC_GAIN = os.path.join(_root, "idea_2", "breeder_tools", "Estimated_Genetic_Gain.csv")
    
    # ========================================================================
    # FIGURE 3: Deep Learning
    # ========================================================================
    
    # Panel A: Model performance comparison
    MODEL_PERF_IDEA3 = os.path.join(_root, "idea_3", "metrics", "idea3_model_performance_summary.csv")
    
    # Panel B: AI vs GWAS merged (FBC)
    AI_GWAS_FBC = os.path.join(_root, "idea_3", "interpretation", "ai_vs_gwas", "ai_gwas_merged_trait-FBC.csv")
    
    # Panel C: Saliency summary
    SALIENCY_SUMMARY = os.path.join(_root, "idea_3", "interpretation", "idea3_saliency_summary.csv")
    
    # Panel D: Block synergy
    BLOCK_SYNERGY = os.path.join(_root, "idea_3", "interpretation", "editing", "advanced", "haplotype_block_synergy.csv")
    
    # Panel E: Editing tradeoff
    EDITING_TRADEOFF = os.path.join(_root, "idea_3", "interpretation", "idea3_editing_tradeoff_summary.csv")
    
    # Panel F: Concordance summary
    CONCORDANCE_SUMMARY = os.path.join(_root, "idea_3", "interpretation", "ai_vs_gwas", "ai_gwas_concordance_summary.csv")
    
    # ========================================================================
    # FIGURE 4: Polygenic & Hub Genes
    # ========================================================================
    
    # Panel A: BINN scores wide
    BINN_SCORES_WIDE = os.path.join(_root, "idea_3", "binn_explain", "binn_gene_scores_wide.csv")
    
    # Panel B: Polygenic architecture
    POLYGENIC_ARCH = os.path.join(_root, "idea_3", "breeder_resources", "Polygenic_Architecture_Summary.csv")
    
    # Panel C: Polygenic evaluation (per-SNP weights)
    POLYGENIC_EVAL = os.path.join(_root, "idea_3", "breeder_resources", "Mango_Polygenic_Evaluation_File.csv")
    
    # Panel D: Pleiotropy scores
    PLEIOTROPY_SCORES = os.path.join(_root, "idea_3", "binn_explain", "binn_gene_pleiotropy_scores.csv")
    
    # Panel E: Saliency matrix
    SALIENCY_MATRIX = os.path.join(_root, "idea_3", "interpretation", "saliency", "saliency_matrix_block-raw.csv")
    
    # Panel F: BINN CV summary
    BINN_CV_SUMMARY = os.path.join(_root, "idea_3", "binn_training", "binn_cv_summary.csv")
    
    # ========================================================================
    # ADDITIONAL DATA SOURCES
    # ========================================================================
    
    # GWAS by trait (wide format)
    GWAS_BY_TRAIT = os.path.join(_root, "idea_1", "gwas", "gwas_summary_by_trait.csv")
    
    # BINN gene scores long format
    BINN_SCORES_LONG = os.path.join(_root, "idea_3", "binn_explain", "binn_gene_scores_long.csv")
    
    # Model performance long format
    MODEL_PERF_LONG = os.path.join(_root, "idea_2", "summary", "idea2_model_performance_long.csv")
    
    # Candidate genes
    CANDIDATE_GENES = os.path.join(_root, "idea_2", "annotation", "idea2_candidate_genes_summary.csv")
    
    # BINN gene table
    BINN_GENE_TABLE = os.path.join(_root, "idea_3", "binn_maps", "binn_gene_table.csv")
    
    # SHAP top SNPs
    SHAP_TOP_SNPS = os.path.join(_root, "idea_3", "interpretation", "shap", "SHAP_TopSNPs_FBC.csv")


# ============================================================================
# COLUMN NAME MAPPINGS (based on data inventory)
# ============================================================================

class ColumnMaps:
    """
    Column name mappings for each data file.
    Maps expected/standard names to actual column names in files.
    """
    
    # ========================================================================
    # FIGURE 1
    # ========================================================================
    
    PC_SCORES = {
        'sample_id': 'sample_id',
        'cluster': 'cluster',
        'PC1': 'PC1',
        'PC2': 'PC2',
        'PC3': 'PC3',
        'PC4': 'PC4',
    }
    
    CLUSTER_SIZES = {
        'cluster': 'cluster',
        'n_samples': 'n_samples',
    }
    
    CV_TRANSFERABILITY = {
        'trait': 'trait',
        'r_random': 'r_random_pc',
        'r_balanced': 'r_cluster_balanced_pc',
        'r_lco': 'r_leave_cluster_out_pc',
        'drop_pct_lco': 'drop_pct_leave_cluster',
    }
    
    PHENO_SUMMARY = {
        'trait': 'trait',
        'n': 'n_non_missing',
        'mean': 'mean',
        'sd': 'sd',
        'cv_percent': 'cv_percent',
    }
    
    PHENO_CORE = {
        'id': 'ID',
        'FBC': 'FBC',
        'FF': 'FF',
        'AFW': 'AFW',
        'TSS': 'TSS',
        'TC': 'TC',
    }
    
    # ========================================================================
    # FIGURE 2
    # ========================================================================
    
    HAPLOTYPE_EFFECTS = {
        'trait': 'Trait',
        'marker': 'Marker',
        'effect_std': 'Effect_Std',
        'effect_raw': 'Effect_Raw',
        'is_additive': 'Is_Additive',
        'mean_g0': 'Mean_G0',
        'mean_g1': 'Mean_G1',
        'mean_g2': 'Mean_G2',
        'n_g0': 'N_G0',
        'n_g1': 'N_G1',
        'n_g2': 'N_G2',
        'global_mean': 'Global_Mean',
        'global_sd': 'Global_SD',
    }
    
    ASSAY_DESIGN = {
        'trait': 'Trait',
        'marker_id': 'Marker_ID',
        'chromosome': 'GWAS_Chrom',  # Note: GWAS_Chrom not Chromosome
        'fasta_chrom': 'FASTA_Chrom',
        'position': 'Position',
        'ref_allele': 'Ref_Allele',
        'sequence': 'Sequence_Context',
    }
    
    RANDOM_VS_INVERSION = {
        'trait': 'trait',
        'scheme': 'scheme',
        'model': 'model',
        'n_markers': 'n_markers',
        'replicate': 'replicate',
        'mean_r': 'mean_r_random',  # Note: mean_r_random not random_r
    }
    
    META_CORE = {
        'id': 'ID',
        'genotype_name': 'Genotype Name',
        # Inversion markers
        'miinv1.0': 'miinv1.0',
        'miinv1.1': 'miinv1.1',
        'miinv3.0': 'miinv3.0',
        'miinv4.0': 'miinv4.0',
        'miinv4.1': 'miinv4.1',
        'miinv5.0': 'miinv5.0',
        'miinv6.0': 'miinv6.0',
        'miinv7.0': 'miinv7.0',
        'miinv8.0': 'miinv8.0',
        'miinv9.0': 'miinv9.0',
        'miinv11.0': 'miinv11.0',
        'miinv13.0': 'miinv13.0',
        'miinv14.0': 'miinv14.0',
        'miinv17.0': 'miinv17.0',
        'miinv17.1': 'miinv17.1',
        'miinv20.0': 'miinv20.0',
    }
    
    MODEL_PERF_IDEA2 = {
        'trait': 'trait',
        'scheme': 'scheme',
        'model_family': 'model_family',
        'model': 'model',
        'feature_set': 'feature_set',
        'mean_r': 'mean_r',
        'std_r': 'std_r',
        'n_folds': 'n_folds',
    }
    
    GENETIC_GAIN = {
        'trait': 'Trait',
        'scenario': 'Scenario',
        'accuracy': 'Accuracy_r',
        'gain_percent': 'Gain_Percent',
    }
    
    # ========================================================================
    # FIGURE 3
    # ========================================================================
    
    MODEL_PERF_IDEA3 = {
        'trait_group': 'trait_group',
        'trait': 'trait',
        'model': 'model',
        'mean_r': 'mean_r',  # Note: mean_r not pearson_r_mean
        'sd_r': 'sd_r',
        'min_r': 'min_r',
        'max_r': 'max_r',
        'n_folds': 'n_folds',
    }
    
    AI_GWAS_FBC = {
        'feature_index': 'feature_index',
        'snp_id': 'snp_id',
        'saliency': 'saliency_FBC_norm',
        'p_value': 'p_FBC',
        'beta': 'beta_FBC',
        'neglog10p': 'neglog10p',
    }
    
    SALIENCY_SUMMARY = {
        'trait': 'trait',
        'total_saliency': 'total_saliency',
        'top1_saliency': 'top1_saliency',
        'top1_share': 'top1_share',  # Note: top1_share not top_1pct_share
        'top10_share': 'top10_share',
        'top100_share': 'top100_share',
        'n_snps': 'n_snps',
    }
    
    BLOCK_SYNERGY = {
        'trait': 'trait',
        'block_effect': 'block_effect',
        'sum_singles': 'sum_singles',
        'synergy': 'synergy',
        'type': 'type',
    }
    
    EDITING_TRADEOFF = {
        'target_trait': 'target_trait',
        'affected_trait': 'affected_trait',
        'mean_delta': 'mean_delta',
        'mean_abs_delta': 'mean_abs_delta',
        'max_abs_delta': 'max_abs_delta',
        'n_snps': 'n_snps',
    }
    
    CONCORDANCE_SUMMARY = {
        'trait': 'trait',
        'n_snps': 'n_overlap_snps',
        'corr_saliency_logp': 'pearson_sal_vs_neglog10p',  # Note: different name
        'corr_saliency_beta': 'pearson_sal_vs_absbeta',    # Note: different name
        'spearman_logp': 'spearman_sal_vs_neglog10p',
        'spearman_beta': 'spearman_sal_vs_absbeta',
    }
    
    # ========================================================================
    # FIGURE 4
    # ========================================================================
    
    BINN_SCORES_WIDE = {
        'gene_id': 'gene_id',
        'gene_name': 'gene_name',
        'chr': 'chr',
        'start': 'start',
        'end': 'end',
        'n_snps': 'n_snps',
        'score_FBC': 'score_FBC',
        'score_FF': 'score_FF',
        'score_AFW': 'score_AFW',
        'score_TSS': 'score_TSS',
        'score_TC': 'score_TC',
    }
    
    POLYGENIC_ARCH = {
        'trait': 'trait',
        'top_1pct_share': 'top_1pct_weight_share',
        'architecture': 'architecture',
        'n_snps': 'n_snps',
    }
    
    POLYGENIC_EVAL = {
        'snp_id': 'SNP_ID',
        'maf': 'MAF',
        'weight_FBC': 'Weight_FBC',
        'weight_FF': 'Weight_FF',
        'weight_AFW': 'Weight_AFW',
        'weight_TSS': 'Weight_TSS',
        'weight_TC': 'Weight_TC',
        'pct_var_FBC': 'Pct_Var_FBC',
        'pct_var_FF': 'Pct_Var_FF',
        'pct_var_AFW': 'Pct_Var_AFW',
        'pct_var_TSS': 'Pct_Var_TSS',
        'pct_var_TC': 'Pct_Var_TC',
    }
    
    PLEIOTROPY_SCORES = {
        'gene_id': 'gene_id',
        'max_score': 'max_score',
        'n_traits_top': 'n_traits_above_90pct',
        'traits_top': 'traits_above_90pct',
        'gene_name': 'gene_name',
        'chr': 'chr',
        'start': 'start',
        'end': 'end',
        'n_snps': 'n_snps',
    }
    
    SALIENCY_MATRIX = {
        'feature_index': 'feature_index',
        'snp_id': 'snp_id',
        'saliency_FBC': 'saliency_FBC_norm',
        'saliency_FF': 'saliency_FF_norm',
        'saliency_AFW': 'saliency_AFW_norm',
        'saliency_TSS': 'saliency_TSS_norm',
        'saliency_TC': 'saliency_TC_norm',
    }
    
    BINN_CV_SUMMARY = {
        'trait': 'trait',
        'n_folds': 'n_folds',
        'mean_r': 'mean_r',
        'sd_r': 'sd_r',  # Note: sd_r not std_r
        'mean_rmse': 'mean_rmse',
    }


# ============================================================================
# COLOR CONFIGURATION
# ============================================================================

class ColorConfig:
    """
    Color palette configuration.
    Favorites: blue-teal-green tones.
    """
    
    def __init__(self):
        # ====================================================================
        # CORE NAMED COLORS (blue-teal-green favorites)
        # ====================================================================
        
        # Primary favorites
        self.limegreen = '#32CD32'
        self.mediumseagreen = '#3CB371'
        self.springgreen = '#00FF7F'
        self.turquoise = '#40E0D0'
        self.mediumturquoise = '#48D1CC'
        self.deepskyblue = '#00BFFF'
        self.steelblue = '#4682B4'
        self.darkseagreen = '#8FBC8F'
        self.seagreen = '#2E8B57'
        self.royalblue = '#4169E1'
        
        # Secondary favorites
        self.cornflowerblue = '#6495ED'
        self.teal = '#008080'
        self.darkcyan = '#008B8B'
        self.cadetblue = '#5F9EA0'
        self.darkturquoise = '#00CED1'
        self.paleturquoise = '#AFEEEE'
        self.aquamarine = '#7FFFD4'
        self.mediumaquamarine = '#66CDAA'
        
        # Greens
        self.forestgreen = "#2EAC2E"
        self.palegreen = '#98FB98'
        self.yellowgreen = '#9ACD32'
        self.mintcream = '#F5FFFA'
        
        # Neutrals
        self.gray = '#808080'
        self.lightgray = '#D3D3D3'
        self.darkgray = '#A9A9A9'
        self.gainsboro = '#DCDCDC'
        self.slategray = '#708090'
        
        # Accent colors (for contrast when needed)
        self.coral_red = "#94CB64"
        self.teal_green = '#00A087'
        self.brown = "#6791AA"
        self.mediumpurple = "#3A95EA"
        
        # ====================================================================
        # TRAIT COLORS - Standard Order: FBC, AFW, FF, TC, TSS
        # ====================================================================
        
        # Default: harmonious blue-teal-green
        self.trait_colors = {
            'FBC': self.mediumseagreen,   # #3CB371
            'AFW': self.royalblue,        # #4169E1
            'FF': self.turquoise,         # #40E0D0
            'TC': self.steelblue,         # #4682B4
            'TSS': self.limegreen,        # #32CD32
        }
        
        # Distinct: high-contrast for slopegraphs/comparisons
        self.trait_colors_distinct = {
            'FBC': self.springgreen,      # #00DB60
            'AFW': self.royalblue,        # #4169E1
            'FF': self.gray,              # #808080
            'TC': self.teal_green,        # #00A087
            'TSS': self.mediumseagreen,   # #209253
        }
        
        # Trait order (use everywhere for consistency)
        self.trait_order = ['FBC', 'AFW', 'FF', 'TC', 'TSS']
        
        # Trait full names
        self.trait_names = {
            'FBC': 'Fruit Blush Colour',
            'AFW': 'Average Fruit Weight',
            'FF': 'Fruit Firmness',
            'TC': 'Trunk Circumference',
            'TSS': 'Total Soluble Solids',
        }
        
        # ====================================================================
        # CLUSTER COLORS (for PCA plots)
        # ====================================================================
        
        self.cluster_colors = {
            0: self.coral_red,        # Cluster 0
            1: self.teal_green,       # Cluster 1
            2: self.royalblue,        # Cluster 2
        }
        
        self.cluster_colors_list = [self.coral_red, self.teal_green, self.royalblue]
        
        # ====================================================================
        # CV SCHEME COLORS
        # ====================================================================
        
        self.cv_colors = {
            'random': self.deepskyblue,
            'random_kfold': self.deepskyblue,
            'cv_random_k5': self.deepskyblue,
            'balanced': self.mediumseagreen,
            'cluster_balanced': self.mediumseagreen,
            'cluster_aware': self.mediumseagreen,
            'cv_cluster_kmeans': self.mediumseagreen,
            'lco': self.royalblue,
            'leave_cluster_out': self.royalblue,
        }
        
        # ====================================================================
        # MODEL COLORS
        # ====================================================================
        
        self.model_colors = {
            # Feature sets
            'SNP_all': self.royalblue,
            'snp': self.royalblue,
            'Inversion': self.limegreen,
            'inv': self.limegreen,
            'Combined': self.mediumseagreen,
            'snp+inv': self.mediumseagreen,
            # Algorithms
            'Ridge': self.steelblue,
            'ridge': self.steelblue,
            'GBLUP': self.steelblue,
            'XGBoost': self.turquoise,
            'xgb': self.turquoise,
            'Random_Forest': self.darkseagreen,
            'rf': self.darkseagreen,
            'MLP': self.mediumpurple,
            'mlp': self.mediumpurple,
            'Wide&Deep': self.coral_red,
            'wide_deep': self.coral_red,
            'BINN': self.seagreen,
            'binn': self.seagreen,
        }
        
        # ====================================================================
        # CLASSIFICATION COLORS
        # ====================================================================
        
        self.class_colors = {
            'hub': self.limegreen,
            'multi_trait': self.mediumseagreen,
            'single_trait': self.gray,
            'platinum': self.limegreen,
            'modulator': self.mediumseagreen,
            'driver': self.royalblue,
        }
        
        # ====================================================================
        # SIGNIFICANCE COLORS
        # ====================================================================
        
        self.sig_colors = {
            'significant': self.limegreen,
            'borderline': self.mediumturquoise,
            'nonsignificant': self.gray,
        }
        
        # ====================================================================
        # TIER COLORS (Precision Breeding Hierarchy)
        # ====================================================================
        
        self.tier_colors = {
            1: self.limegreen,        # Tier 1: Global markers
            2: self.royalblue,        # Tier 2: Global GS
            3: self.gray,             # Tier 3: Local GS
            'Tier 1': self.limegreen,
            'Tier 2': self.royalblue,
            'Tier 3': self.gray,
        }
        
        # ====================================================================
        # COLORMAPS
        # ====================================================================
        
        # Sequential green (for heatmaps - PRIMARY CHOICE)
        self.cmap_green = LinearSegmentedColormap.from_list(
            'green_sequential',
            [self.mintcream, self.palegreen, self.mediumseagreen, self.seagreen, self.forestgreen]
        )
        
        # Alternative sequential options
        self.cmap_sequential = LinearSegmentedColormap.from_list(
            'sequential',
            [self.mintcream, self.turquoise, self.deepskyblue, self.cornflowerblue]
        )
        
        # Blue-Green sequential (BuGn style)
        self.cmap_bugreen = LinearSegmentedColormap.from_list(
            'blue_green',
            ['#f7fcfd', '#e5f5f9', '#ccece6', '#99d8c9', '#66c2a4', '#41ae76', '#238b45', '#005824']
        )
        
        # Yellow-Green sequential (YlGn style)
        self.cmap_ylgreen = LinearSegmentedColormap.from_list(
            'yellow_green',
            ['#ffffe5', '#f7fcb9', '#d9f0a3', '#addd8e', '#78c679', '#41ab5d', '#238443', '#005a32']
        )
        
        # Greens (matplotlib style - light to dark)
        self.cmap_greens = LinearSegmentedColormap.from_list(
            'greens',
            ['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#005a32']
        )
        
        # Diverging (for correlation matrices: negative=blue, zero=white, positive=green)
        self.cmap_diverging = LinearSegmentedColormap.from_list(
            'diverging_bg',
            [self.royalblue, self.cornflowerblue, 'white', self.mediumseagreen, self.seagreen]
        )
        
        # Diverging red-green (use sparingly - colorblind issues)
        self.cmap_diverging_rg = LinearSegmentedColormap.from_list(
            'diverging_rg',
            [self.coral_red, 'white', self.limegreen]
        )
        
        # ====================================================================
        # HEATMAP COLORMAPS (PRIMARY CHOICES)
        # ====================================================================
        
        # DEFAULT heatmap colormap - use this for most heatmaps
        self.cmap_heatmap = self.cmap_greens
        
        # BINN gene scores heatmap
        self.cmap_binn = self.cmap_ylgreen
        
        # Correlation heatmap (diverging: blue-white-green)
        self.cmap_correlation = self.cmap_diverging
        
        # Saliency/importance heatmap
        self.cmap_importance = self.cmap_bugreen


# ============================================================================
# FIGURE STYLE CONFIGURATION
# ============================================================================

class StyleConfig:
    """Figure styling parameters"""
    
    # DPI
    DPI = 300
    DPI_DISPLAY = 100
    
    # Figure sizes (inches) - for full figures
    FIGSIZE_FULL = (12, 10)      # Full page figure
    FIGSIZE_WIDE = (14, 8)       # Wide figure
    FIGSIZE_TALL = (10, 12)      # Tall figure
    FIGSIZE_SQUARE = (10, 10)    # Square figure
    
    # Panel sizes (inches)
    PANEL_LARGE = (5, 4)
    PANEL_MEDIUM = (3.5, 3)
    PANEL_SMALL = (2.5, 2)
    
    # Font sizes
    FONTSIZE_TITLE = 12
    FONTSIZE_SUBTITLE = 11
    FONTSIZE_LABEL = 10
    FONTSIZE_TICK = 8
    FONTSIZE_LEGEND = 8
    FONTSIZE_ANNOTATION = 7
    FONTSIZE_PANEL_LETTER = 14
    
    # Line widths
    LINEWIDTH_AXIS = 1.0
    LINEWIDTH_PLOT = 1.5
    LINEWIDTH_THIN = 0.8
    LINEWIDTH_GRID = 0.5
    
    # Marker sizes
    MARKERSIZE_LARGE = 50
    MARKERSIZE_MEDIUM = 30
    MARKERSIZE_SMALL = 20
    
    # Alpha values
    ALPHA_FILL = 0.3
    ALPHA_SCATTER = 0.6
    ALPHA_CI = 0.2
    ALPHA_GRID = 0.3
    
    @classmethod
    def configure_matplotlib(cls):
        """Apply global matplotlib settings"""
        plt.rcParams.update({
            'figure.dpi': cls.DPI_DISPLAY,
            'savefig.dpi': cls.DPI,
            'savefig.bbox': 'tight',
            'savefig.facecolor': 'white',
            'savefig.edgecolor': 'none',
            'font.size': cls.FONTSIZE_LABEL,
            'axes.labelsize': cls.FONTSIZE_LABEL,
            'axes.titlesize': cls.FONTSIZE_TITLE,
            'xtick.labelsize': cls.FONTSIZE_TICK,
            'ytick.labelsize': cls.FONTSIZE_TICK,
            'legend.fontsize': cls.FONTSIZE_LEGEND,
            'axes.linewidth': cls.LINEWIDTH_AXIS,
            'grid.linewidth': cls.LINEWIDTH_GRID,
            'lines.linewidth': cls.LINEWIDTH_PLOT,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'axes.spines.top': False,
            'axes.spines.right': False,
        })


# ============================================================================
# PANEL LAYOUTS FOR EACH FIGURE
# ============================================================================

class PanelLayouts:
    """
    GridSpec layouts for each figure.
    Returns dict with panel positions and sizes.
    """
    
    @staticmethod
    def figure_1():
        """
        Figure 1: Population Structure and Structure Cliff
        Layout: 3 rows, variable columns
        
        Row 1: A (large PCA) | B (cluster bar)
        Row 2: C (slopegraph - spans full width)
        Row 3: D (pheno stats) | E (transferability) | F (correlation)
        """
        return {
            'figsize': (14, 12),
            'gridspec': {'nrows': 3, 'ncols': 6, 'height_ratios': [1, 1.2, 1],
                        'hspace': 0.35, 'wspace': 0.4},
            'panels': {
                'A': {'pos': (0, slice(0, 4)), 'label_pos': (-0.08, 1.02)},  # PCA
                'B': {'pos': (0, slice(4, 6)), 'label_pos': (-0.15, 1.02)},  # Cluster sizes
                'C': {'pos': (1, slice(0, 6)), 'label_pos': (-0.04, 1.02)},  # Slopegraph
                'D': {'pos': (2, slice(0, 2)), 'label_pos': (-0.15, 1.02)},  # Pheno stats
                'E': {'pos': (2, slice(2, 4)), 'label_pos': (-0.15, 1.02)},  # Transferability
                'F': {'pos': (2, slice(4, 6)), 'label_pos': (-0.15, 1.02)},  # Correlation
            }
        }
    
    @staticmethod
    def figure_2():
        """
        Figure 2: Structural Haplotypes
        Layout: 3 rows
        
        Row 1: A (chromosome ideogram - full width)
        Row 2: B (inversion vs random) | C (boxplots)
        Row 3: D (model comparison) | E (effect sizes) | F (genetic gain)
        """
        return {
            'figsize': (14, 13),
            'gridspec': {'nrows': 3, 'ncols': 6, 'height_ratios': [0.8, 1.2, 1],
                        'hspace': 0.35, 'wspace': 0.4},
            'panels': {
                'A': {'pos': (0, slice(0, 6)), 'label_pos': (-0.04, 1.02)},  # Ideogram
                'B': {'pos': (1, slice(0, 3)), 'label_pos': (-0.08, 1.02)},  # Inv vs Random
                'C': {'pos': (1, slice(3, 6)), 'label_pos': (-0.08, 1.02)},  # Boxplots
                'D': {'pos': (2, slice(0, 2)), 'label_pos': (-0.15, 1.02)},  # Model comparison
                'E': {'pos': (2, slice(2, 4)), 'label_pos': (-0.15, 1.02)},  # Effect sizes
                'F': {'pos': (2, slice(4, 6)), 'label_pos': (-0.15, 1.02)},  # Genetic gain
            }
        }
    
    @staticmethod
    def figure_3():
        """
        Figure 3: Deep Learning
        Layout: 3 rows
        
        Row 1: A (model comparison) | B (Manhattan overlay)
        Row 2: C (saliency concentration) | D (block synergy)
        Row 3: E (editing tradeoffs) | F (concordance)
        """
        return {
            'figsize': (14, 12),
            'gridspec': {'nrows': 3, 'ncols': 6, 'height_ratios': [1, 1, 1],
                        'hspace': 0.35, 'wspace': 0.4},
            'panels': {
                'A': {'pos': (0, slice(0, 3)), 'label_pos': (-0.08, 1.02)},  # Model comparison
                'B': {'pos': (0, slice(3, 6)), 'label_pos': (-0.08, 1.02)},  # Manhattan
                'C': {'pos': (1, slice(0, 3)), 'label_pos': (-0.08, 1.02)},  # Saliency
                'D': {'pos': (1, slice(3, 6)), 'label_pos': (-0.08, 1.02)},  # Synergy
                'E': {'pos': (2, slice(0, 3)), 'label_pos': (-0.08, 1.02)},  # Tradeoffs
                'F': {'pos': (2, slice(3, 6)), 'label_pos': (-0.08, 1.02)},  # Concordance
            }
        }
    
    @staticmethod
    def figure_4():
        """
        Figure 4: Polygenic & Hub Genes
        Layout: 3 rows
        
        Row 1: A (BINN heatmap - large, spans most of top)
        Row 2: B (polygenic arch) | C (cumulative variance)
        Row 3: D (pleiotropy) | E (saliency correlation) | F (BINN performance)
        """
        return {
            'figsize': (14, 14),
            'gridspec': {'nrows': 3, 'ncols': 6, 'height_ratios': [1.5, 1, 1],
                        'hspace': 0.35, 'wspace': 0.4},
            'panels': {
                'A': {'pos': (0, slice(0, 6)), 'label_pos': (-0.04, 1.02)},  # BINN heatmap
                'B': {'pos': (1, slice(0, 3)), 'label_pos': (-0.08, 1.02)},  # Polygenic arch
                'C': {'pos': (1, slice(3, 6)), 'label_pos': (-0.08, 1.02)},  # Cumulative var
                'D': {'pos': (2, slice(0, 2)), 'label_pos': (-0.15, 1.02)},  # Pleiotropy
                'E': {'pos': (2, slice(2, 4)), 'label_pos': (-0.15, 1.02)},  # Saliency corr
                'F': {'pos': (2, slice(4, 6)), 'label_pos': (-0.15, 1.02)},  # BINN perf
            }
        }
    
    @staticmethod
    def figure_5():
        """
        Figure 5: Precision Breeding Hierarchy
        Layout: 2 rows
        
        Row 1: A (hierarchy 2D plot - large) | B (tier table)
        Row 2: C (tier gain) | D (assay panel) | E (flowchart)
        """
        return {
            'figsize': (14, 10),
            'gridspec': {'nrows': 2, 'ncols': 6, 'height_ratios': [1.3, 1],
                        'hspace': 0.35, 'wspace': 0.4},
            'panels': {
                'A': {'pos': (0, slice(0, 4)), 'label_pos': (-0.06, 1.02)},  # 2D hierarchy
                'B': {'pos': (0, slice(4, 6)), 'label_pos': (-0.15, 1.02)},  # Tier table
                'C': {'pos': (1, slice(0, 2)), 'label_pos': (-0.15, 1.02)},  # Tier gain
                'D': {'pos': (1, slice(2, 4)), 'label_pos': (-0.15, 1.02)},  # Assay panel
                'E': {'pos': (1, slice(4, 6)), 'label_pos': (-0.15, 1.02)},  # Flowchart
            }
        }


# ============================================================================
# UNIFIED FIGURE CONFIG CLASS
# ============================================================================

class FigureConfig:
    """
    Unified figure configuration combining all settings.
    Main interface for figure generation scripts.
    """
    
    def __init__(self):
        # Initialize sub-configs
        self.paths = PathConfig
        self.data = DataPaths
        self.cols = ColumnMaps
        self.colors = ColorConfig()
        self.style = StyleConfig
        self.layouts = PanelLayouts
        
        # Convenience aliases
        self.trait_colors = self.colors.trait_colors
        self.trait_colors_distinct = self.colors.trait_colors_distinct
        self.trait_order = self.colors.trait_order
        self.trait_names = self.colors.trait_names
        self.cluster_colors = self.colors.cluster_colors
        self.model_colors = self.colors.model_colors
        self.cv_colors = self.colors.cv_colors
        self.tier_colors = self.colors.tier_colors
        
        # Apply matplotlib settings
        self.style.configure_matplotlib()
        
        # Ensure directories exist
        self.paths.ensure_dirs()
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def add_panel_label(self, ax, label, x=-0.12, y=1.08, **kwargs):
        """Add panel letter (A, B, C, etc.)"""
        defaults = {
            'transform': ax.transAxes,
            'fontsize': self.style.FONTSIZE_PANEL_LETTER,
            'fontweight': 'bold',
            'va': 'bottom',
            'ha': 'left',
        }
        defaults.update(kwargs)
        ax.text(x, y, label, **defaults)
    
    def add_stats_box(self, ax, text, x=0.03, y=0.97, **kwargs):
        """Add statistics annotation box"""
        defaults = {
            'transform': ax.transAxes,
            'fontsize': self.style.FONTSIZE_ANNOTATION,
            'va': 'top',
            'ha': 'left',
            'bbox': dict(facecolor='white', alpha=0.9, 
                        edgecolor=self.colors.gray, lw=0.8),
        }
        defaults.update(kwargs)
        ax.text(x, y, text, **defaults)
    
    def style_axis(self, ax, spines=['left', 'bottom'], grid=True):
        """Apply standard axis styling"""
        for spine in ['top', 'right', 'left', 'bottom']:
            if spine in spines:
                ax.spines[spine].set_visible(True)
                ax.spines[spine].set_linewidth(self.style.LINEWIDTH_AXIS)
            else:
                ax.spines[spine].set_visible(False)
        
        if grid:
            ax.grid(True, alpha=self.style.ALPHA_GRID, 
                   linewidth=self.style.LINEWIDTH_GRID, linestyle='--')
            ax.set_axisbelow(True)
    
    def save_figure(self, fig, filename, fig_num=None, **kwargs):
        """
        Save figure to output directory.
        
        Args:
            fig: matplotlib figure
            filename: filename (e.g., 'figure_1.png')
            fig_num: optional figure number for subdirectory
        """
        if fig_num:
            outdir = self.paths.figure_subdir(fig_num)
        else:
            outdir = self.paths.FIGURES_DIR
        
        filepath = os.path.join(outdir, filename)
        
        defaults = {
            'dpi': self.style.DPI,
            'bbox_inches': 'tight',
            'facecolor': 'white',
            'edgecolor': 'none',
        }
        defaults.update(kwargs)
        
        fig.savefig(filepath, **defaults)
        print(f"[OK] Figure saved: {filepath}")
        return filepath
    
    def get_layout(self, fig_num):
        """Get layout configuration for a figure"""
        layouts = {
            1: self.layouts.figure_1,
            2: self.layouts.figure_2,
            3: self.layouts.figure_3,
            4: self.layouts.figure_4,
            5: self.layouts.figure_5,
        }
        return layouts.get(fig_num, lambda: None)()
    
    def create_figure(self, fig_num):
        """
        Create figure with GridSpec layout.
        
        Returns:
            fig, axes_dict
        """
        import matplotlib.gridspec as gridspec
        
        layout = self.get_layout(fig_num)
        if not layout:
            raise ValueError(f"No layout defined for Figure {fig_num}")
        
        fig = plt.figure(figsize=layout['figsize'])
        gs = gridspec.GridSpec(**layout['gridspec'])
        
        axes = {}
        for panel_name, panel_info in layout['panels'].items():
            row, col = panel_info['pos']
            axes[panel_name] = fig.add_subplot(gs[row, col])
        
        return fig, axes, layout
    
    def get_trait_palette(self, n=5, distinct=False):
        """Get list of trait colors in standard order"""
        colors = self.trait_colors_distinct if distinct else self.trait_colors
        return [colors[t] for t in self.trait_order[:n]]
    
    def format_pvalue(self, p, threshold=0.001):
        """Format p-value for display"""
        if p < threshold:
            return f"p < {threshold}"
        return f"p = {p:.3f}"
    
    def format_r(self, r, p=None):
        """Format correlation coefficient"""
        text = f"r = {r:.3f}"
        if p is not None:
            text += f" ({self.format_pvalue(p)})"
        return text


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

# Create global config instance
config = FigureConfig()

# Convenience exports
trait_colors = config.trait_colors
trait_colors_distinct = config.trait_colors_distinct
trait_order = config.trait_order
trait_names = config.trait_names
cluster_colors = config.cluster_colors
model_colors = config.model_colors
cv_colors = config.cv_colors
tier_colors = config.tier_colors

# Path exports
paths = config.paths
data_paths = config.data

# Column mapping exports
col_maps = config.cols


# ============================================================================
# TEST / VALIDATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("MANGO GWAS - UNIFIED FIGURE CONFIGURATION")
    print("=" * 80)
    
    print("\n[PATHS]")
    print(f"  Project root: {PathConfig.ROOT}")
    print(f"  Figures dir:  {PathConfig.FIGURES_DIR}")
    
    print("\n[TRAITS]")
    print(f"  Order: {trait_order}")
    for t in trait_order:
        print(f"    {t}: {trait_names[t]} -> {trait_colors[t]}")
    
    print("\n[DATA FILES - Checking existence]")
    import os
    files_to_check = [
        ('PC_SCORES', DataPaths.PC_SCORES),
        ('CV_TRANSFERABILITY', DataPaths.CV_TRANSFERABILITY),
        ('HAPLOTYPE_EFFECTS', DataPaths.HAPLOTYPE_EFFECTS),
        ('MODEL_PERF_IDEA3', DataPaths.MODEL_PERF_IDEA3),
        ('BINN_SCORES_WIDE', DataPaths.BINN_SCORES_WIDE),
        ('GENETIC_GAIN', DataPaths.GENETIC_GAIN),
    ]
    for name, path in files_to_check:
        exists = "[OK]" if os.path.exists(path) else "[MISSING]"
        print(f"  {exists} {name}")
    
    print("\n[LAYOUTS]")
    for i in range(1, 6):
        layout = config.get_layout(i)
        panels = list(layout['panels'].keys())
        print(f"  Figure {i}: {len(panels)} panels ({', '.join(panels)})")
    
    print("\n[STYLE]")
    print(f"  DPI: {StyleConfig.DPI}")
    print(f"  Font sizes: Title={StyleConfig.FONTSIZE_TITLE}, "
          f"Label={StyleConfig.FONTSIZE_LABEL}, Tick={StyleConfig.FONTSIZE_TICK}")
    
    print("\n" + "=" * 80)
    print("Configuration loaded successfully!")
    print("=" * 80)
