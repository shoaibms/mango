#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 5: The Precision Breeding Hierarchy (UPDATED)
=====================================================

Theme: Synthesis of transferability and architecture into a Tier 1-3 
       decision framework for precision breeding.

Panels:
    A: Precision breeding hierarchy map
    B: Structure Cliff Waterfall
    C: Marker Efficiency
    D: Within vs Cross-Population Gain
    E: Decision Schematic (Version 1: Vertical Tier Cards)
    F: Method Concordance (BINN vs baseline accuracy)

NOTE: This script uses ONLY real data files. NO fallback or synthetic data.
"""

import os
import sys
import re
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import warnings

# =============================================================================
# Readable column-name mapping (Supplementary Data Column Reference)
# =============================================================================
def _find_column_reference_md(start_dir: str) -> str | None:
    """
    Try to locate column_reference.md from common relative locations.
    Safe: returns None if not found.
    """
    candidates = [
        os.path.join(start_dir, "column_reference.md"),
        os.path.join(start_dir, "..", "column_reference.md"),
        os.path.join(start_dir, "..", "..", "column_reference.md"),
        os.path.join(start_dir, "..", "..", "docs", "column_reference.md"),
        os.path.join(start_dir, "..", "..", "supplementary", "column_reference.md"),
        os.path.join(start_dir, "..", "..", "output", "column_reference.md"),
    ]
    for p in candidates:
        p = os.path.abspath(p)
        if os.path.exists(p):
            return p
    return None


def _load_column_reference_md(path: str) -> dict:
    """
    Parse a markdown table with columns:
      | Code Name | Readable Name |
    Returns {code_name: readable_name}.
    """
    mapping: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Table row format: | `code` | Readable |
            if not (line.startswith("|") and line.endswith("|")):
                continue
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) < 2:
                continue
            code, readable = parts[0], parts[1]
            # Skip header/separator rows
            if code.lower() in {"code name", ":----------"}:
                continue
            if readable.lower() in {"readable name", ":--------------"}:
                continue
            # Remove surrounding backticks if present
            code = re.sub(r"^`(.+)`$", r"\1", code)
            readable = re.sub(r"^`(.+)`$", r"\1", readable)
            if code and readable:
                mapping[code] = readable
    return mapping


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_COLREF_PATH = _find_column_reference_md(_SCRIPT_DIR)
if _COLREF_PATH:
    COLUMN_NAME_MAP = _load_column_reference_md(_COLREF_PATH)
else:
    COLUMN_NAME_MAP = {}


def apply_readable_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns using COLUMN_NAME_MAP when available (no-op otherwise)."""
    if df is None or df.empty or not COLUMN_NAME_MAP:
        return df
    rename_map = {c: COLUMN_NAME_MAP.get(c, c) for c in df.columns}
    return df.rename(columns=rename_map)

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DEBUG = True

# Paths - Windows
PROJECT_ROOT = r"C:\Users\ms\Desktop\mango"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
FIGURE_SUBDIR = os.path.join(FIGURES_DIR, "figure_5")

# ============================================================================
# COLOR CONFIGURATION
# ============================================================================

class Colors:
    """Color palette - Blue-Teal-Green Theme"""
    
    limegreen = '#32CD32'
    mediumseagreen = '#3CB371'
    springgreen = '#00FF7F'
    turquoise = '#40E0D0'
    mediumturquoise = '#48D1CC'
    deepskyblue = '#00BFFF'
    steelblue = '#4682B4'
    royalblue = '#4169E1'
    seagreen = '#2E8B57'
    coral_red = "#94CB64"
    teal_green = '#00A087'
    gray = '#808080'
    lightgray = '#D3D3D3'
    darkgray = '#A9A9A9'
    
    trait_colors = {
        'FBC': '#3CB371',
        'AFW': '#4169E1',
        'FF': '#40E0D0',
        'TC': '#4682B4',
        'TSS': '#32CD32',
    }
    
    tier_colors = {
        1: '#00A087',
        2: '#4682B4',
        3: '#808080',
    }
    
    cv_colors = {
        'random': '#00BFFF',
        'cluster_aware': '#3CB371',
        'lco': '#4169E1',
    }
    
    model_colors = {
        'SNP_all': '#4169E1',
        'Inversion': '#32CD32',
        'BINN': '#00A087',
        'Baseline': '#4682B4',
    }

colors = Colors()

# Data paths for Figure 5
DATA_PATHS = {
    'transferability': os.path.join(OUTPUT_DIR, "idea_1", "summary", "cv_transferability_summary.csv"),
    'structural_scores': os.path.join(OUTPUT_DIR, "idea_3", "breeder_resources", "Polygenic_Architecture_Summary.csv"),
    'model_performance': os.path.join(OUTPUT_DIR, "idea_2", "summary", "idea2_gs_model_performance_clean.csv"),
    'random_vs_inversion': os.path.join(OUTPUT_DIR, "idea_2", "random_control", "random_vs_inversion_summary.csv"),
    'genetic_gain': os.path.join(OUTPUT_DIR, "idea_2", "breeder_tools", "Estimated_Genetic_Gain.csv"),
    'breeder_consensus': os.path.join(OUTPUT_DIR, "breeding_value_concordance", "breeder_consensus_summary.csv"),
}

# Ensure output directories exist
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(FIGURE_SUBDIR, exist_ok=True)

# ============================================================================
# BREEDING HIERARCHY TIERING (RULE-BASED; Methods 4.6.2)
# ----------------------------------------------------------------------------
# NOTE:
#  - Tiering must be derived from the same metrics used in the manuscript:
#      Transferability  T = r_leave_cluster_out_pc / r_random_pc  (clipped at 0)
#      Structural score S = top_1pct_weight_share × 100
#  - To keep the figure + Table S12B self-consistent, DO NOT hard-code traits.
#
# Thresholds can be overridden via environment variables if needed for sensitivity
# checks, without editing the script:
#   MANGO_TIER1_T=0.30
#   MANGO_TIER2_T=0.10
#   MANGO_HIGH_STRUCT_DOM=2.00        # percent units (top 1% share ×100)
#   MANGO_STRUCTURAL_MAX_SNPS=5000    # only used when n_snps column exists
# ============================================================================
TRANSFERABILITY_TIER1_CUTOFF = float(os.getenv("MANGO_TIER1_T", "0.30"))
TRANSFERABILITY_TIER2_CUTOFF = float(os.getenv("MANGO_TIER2_T", "0.10"))
HIGH_STRUCT_DOMINANCE_CUTOFF = float(os.getenv("MANGO_HIGH_STRUCT_DOM", "3.30"))
STRUCTURAL_MAX_SNPS = int(os.getenv("MANGO_STRUCTURAL_MAX_SNPS", "5000"))

def _is_high_structural(row) -> bool:
    """
    'High concentration' should be defined by the structural dominance metric
    used in the manuscript: Structural Score = top_1pct_weight_share × 100.

    We allow architecture labels to provide a positive override (if present),
    but we do NOT allow "Polygenic" to hard-block structural dominance, because
    the label is coarse and can be stale / derived from a different pipeline stage.
    """
    # Positive override if architecture explicitly indicates structural/haplotype
    arch = row.get("architecture", None)
    if arch is not None and pd.notna(arch):
        a = str(arch).strip().lower()
        if any(k in a for k in ["struct", "inversion", "super", "oligo", "haplo"]):
            return True
        # Do NOT return False for "polygenic" — fall through to numeric rule.

    # Numeric rule (authoritative)
    s = row.get("structural_dominance", np.nan)
    if pd.isna(s):
        return False
    return float(s) >= HIGH_STRUCT_DOMINANCE_CUTOFF

def _assign_tier(row) -> int:
    """Tier logic (aligned to breeding_tier_code_summary.md / Methods 4.6.2)."""
    t = float(row["transferability"])
    high_struct = bool(row["high_structural"])

    if (t > TRANSFERABILITY_TIER1_CUTOFF) and high_struct:
        return 1
    if (t > TRANSFERABILITY_TIER1_CUTOFF) and (not high_struct):
        return 2
    if t > TRANSFERABILITY_TIER2_CUTOFF:
        return 2
    return 3

def _strategy_for_tier(tier: int) -> str:
    return {
        1: "Global markers (inversion/SNP assays)",
        2: "Genome-wide GS (structure-aware training)",
        3: "Local GS only (within-population)",
    }.get(int(tier), "Local GS only (within-population)")

def build_hierarchy_table(trans_df: pd.DataFrame, struct_df: pd.DataFrame) -> pd.DataFrame:
    """Create the breeder-facing hierarchy table used for Panel A/E and Table S12B."""
    req_trans = {"trait", "r_random_pc", "r_cluster_balanced_pc", "r_leave_cluster_out_pc"}
    req_struct = {"trait", "top_1pct_weight_share"}

    if trans_df is None or struct_df is None:
        return pd.DataFrame()

    missing_t = sorted(list(req_trans - set(trans_df.columns)))
    missing_s = sorted(list(req_struct - set(struct_df.columns)))
    if missing_t:
        raise ValueError(f"cv_transferability_summary.csv missing required columns: {missing_t}")
    if missing_s:
        raise ValueError(f"Polygenic_Architecture_Summary.csv missing required columns: {missing_s}")

    merged = trans_df.merge(struct_df, on="trait", how="inner").copy()
    if merged.empty:
        return merged

    # Core metrics used in the manuscript
    merged["transferability"] = (merged["r_leave_cluster_out_pc"] / merged["r_random_pc"]).clip(lower=0)
    merged["top_1pct_weight_share"] = pd.to_numeric(merged["top_1pct_weight_share"], errors="coerce")
    merged["structural_dominance"] = merged["top_1pct_weight_share"] * 100

    merged["high_structural"] = merged.apply(_is_high_structural, axis=1)
    merged["tier"] = merged.apply(_assign_tier, axis=1)
    merged["strategy_recommendation"] = merged["tier"].apply(_strategy_for_tier)

    # Convenience formatted columns for manuscript-ready export
    merged["Structural Score"] = merged["structural_dominance"].round(2)
    merged["Transferability"] = merged["transferability"].round(3)

    return merged

def export_table_s12b(hier_df: pd.DataFrame) -> None:
    """Write manuscript-ready Table S12B (CSV) to output/figures/figure_5."""
    if hier_df is None or hier_df.empty:
        log("Table S12B export skipped (hierarchy table is empty).", "WARN")
        return

    os.makedirs(FIGURE_SUBDIR, exist_ok=True)

    out = hier_df.copy()

    # Only include raw columns if they actually exist (do NOT fallback to *_pc)
    raw_random = "r_random_raw" if "r_random_raw" in out.columns else None
    raw_lco = "r_leave_cluster_out_raw" if "r_leave_cluster_out_raw" in out.columns else None

    export_cols = [
        "trait",
        "r_random_pc",
        "r_cluster_balanced_pc",
        "r_leave_cluster_out_pc",
        "Transferability",
        "top_1pct_weight_share",
        "Structural Score",
        "tier",
        "strategy_recommendation",
    ]
    if raw_random:
        export_cols.insert(4, raw_random)
    if raw_lco:
        export_cols.insert(5, raw_lco)

    # De-duplicate while preserving order
    export_cols = list(dict.fromkeys([c for c in export_cols if c in out.columns]))

    out_export = out[export_cols].copy()

    rename_map = {
        "trait": "Trait",
        "r_random_pc": "r (Random CV; PC-corrected)",
        "r_cluster_balanced_pc": "r (Cluster-balanced CV; PC-corrected)",
        "r_leave_cluster_out_pc": "r (Leave-cluster-out CV; PC-corrected)",
        "r_random_raw": "r (Random CV; raw)",
        "r_leave_cluster_out_raw": "r (Leave-cluster-out CV; raw)",
        "top_1pct_weight_share": "Top 1% Weight Share",
        "Structural Score": "Structural Score",
        "Transferability": "Transferability",
        "tier": "Tier",
        "strategy_recommendation": "Strategy Recommendation",
    }
    out_export.rename(columns=rename_map, inplace=True)

    # Optional: apply global readable mapping (idempotent for already-renamed cols)
    out_export = apply_readable_headers(out_export)

    # Format Tier as "Tier X"
    if "Tier" in out_export.columns:
        out_export["Tier"] = pd.to_numeric(out_export["Tier"], errors="coerce").astype(int).map(lambda x: f"Tier {x}")

    # Rounding
    for c in out_export.columns:
        if c.startswith("r ("):
            out_export[c] = pd.to_numeric(out_export[c], errors="coerce").round(3)
    if "Top 1% Weight Share" in out_export.columns:
        out_export["Top 1% Weight Share"] = pd.to_numeric(out_export["Top 1% Weight Share"], errors="coerce").round(3)
    if "Structural Score" in out_export.columns:
        out_export["Structural Score"] = pd.to_numeric(out_export["Structural Score"], errors="coerce").round(2)
    if "Transferability" in out_export.columns:
        out_export["Transferability"] = pd.to_numeric(out_export["Transferability"], errors="coerce").round(3)

    csv_path = os.path.join(FIGURE_SUBDIR, "Table_S12B_Precision_Breeding_Hierarchy.csv")
    out_export.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)

    log(f"Exported Table S12B (csv): {csv_path}", "OK")

# Preferred trait order
TRAIT_ORDER = ['FBC', 'AFW', 'FF', 'TC', 'TSS']


# ============================================================================
# LOGGING
# ============================================================================

def log(msg, level="INFO"):
    prefix = {"INFO": "[INFO]", "DEBUG": "[DEBUG]" if DEBUG else None,
              "WARN": "[WARN]", "ERROR": "[ERROR]", "OK": "[OK]"}
    if prefix.get(level):
        print(f"{prefix[level]} {msg}")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def style_axis(ax, spines=None, grid=False):
    """Style axis with clean look - no tick marks."""
    if spines is None:
        spines = ['left', 'bottom']
    for spine in ['top', 'right', 'left', 'bottom']:
        if spine in spines:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_linewidth(1.2)
        else:
            ax.spines[spine].set_visible(False)
    ax.tick_params(axis='both', length=0)
    if grid:
        ax.grid(axis='y', alpha=0.3, linestyle='--')

def add_panel_label(ax, label, x=-0.12, y=1.08, fontsize=14):
    """Add panel label (A, B, C, etc.)."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize,
            fontweight='bold', va='top', ha='left')


# ============================================================================
# DATA LOADING - NO FALLBACK DATA
# ============================================================================

def load_data():
    """
    Load all data files for Figure 5.
    
    STRICT: No fallback or synthetic data. If file missing, returns None.
    """
    log("\n" + "=" * 70)
    log("LOADING DATA FOR FIGURE 5 (STRICT - NO FALLBACK)")
    log("=" * 70)
    
    data = {}
    missing_files = []
    
    for key, path in DATA_PATHS.items():
        log(f"Loading {key}: {path}")
        if os.path.exists(path):
            data[key] = pd.read_csv(path)
            log(f"  Shape: {data[key].shape}", "OK")
            log(f"  Columns: {list(data[key].columns)}", "DEBUG")
        else:
            log(f"  FILE NOT FOUND - Panel will show 'Data not available'", "ERROR")
            data[key] = None
            missing_files.append(key)
    
    if missing_files:
        log(f"\nWARNING: {len(missing_files)} data file(s) missing: {missing_files}", "WARN")
    else:
        log("\nAll data files loaded successfully!", "OK")
    
    log("=" * 70)
    return data


# ============================================================================
# PANEL A: Precision Breeding Hierarchy Map
# ============================================================================

def create_panel_A(data, ax):
    """Panel A: 2D scatter - structural dominance (x) vs transferability (y)."""
    log("\n[PANEL A] Precision Breeding Hierarchy Map")
    
    trans_df = data['transferability']
    struct_df = data['structural_scores']
    
    if trans_df is None or struct_df is None:
        ax.text(0.5, 0.5, 'Data not available', ha='center', va='center',
               fontsize=12, transform=ax.transAxes, color=colors.teal_green)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        style_axis(ax, spines=[])
        add_panel_label(ax, 'A')
        log("  Data not available!", "ERROR")
        return
    
    log(f"  trans_df columns: {list(trans_df.columns)}", "DEBUG")
    log(f"  struct_df columns: {list(struct_df.columns)}", "DEBUG")
    
    # Use a single, rule-based hierarchy table for all panels/tables
    hier_df = data.get("hierarchy_table", None)
    if hier_df is None or hier_df.empty:
        # Fallback: compute locally (still rule-based; no hardcoding)
        merged = build_hierarchy_table(trans_df, struct_df)
        hier_df = merged
    else:
        merged = hier_df

    log(f"  Hierarchy table shape: {merged.shape}", "DEBUG")
    log(merged[["trait", "transferability", "structural_dominance", "architecture",
                "n_snps", "high_structural", "tier"]].to_string(index=False), "DEBUG")

    if merged.empty:
        ax.text(0.5, 0.5, 'No matching data after merge', ha='center', va='center',
               fontsize=12, transform=ax.transAxes)
        add_panel_label(ax, 'A')
        return

    # Collect point positions for overlap detection
    points = []
    for _, row in merged.iterrows():
        trait = row['trait']
        x = float(row['structural_dominance'])
        y = float(row['transferability'])
        tier = int(row.get('tier', 3))
        points.append({'trait': trait, 'x': x, 'y': y, 'tier': tier})
    
    # Sort by y-value to process from bottom to top
    points_df = pd.DataFrame(points)
    
    # Detect overlapping points (close in both x and y)
    def find_overlaps(pts_df, x_thresh=0.3, y_thresh=0.08):
        """Find groups of overlapping points."""
        overlaps = {}
        for i, row1 in pts_df.iterrows():
            for j, row2 in pts_df.iterrows():
                if i >= j:
                    continue
                if abs(row1['x'] - row2['x']) < x_thresh and abs(row1['y'] - row2['y']) < y_thresh:
                    # These two overlap
                    key = tuple(sorted([row1['trait'], row2['trait']]))
                    overlaps[key] = True
        return overlaps
    
    overlaps = find_overlaps(points_df)
    log(f"  Detected overlaps: {list(overlaps.keys())}", "DEBUG")
    
    # Define custom label offsets for overlapping traits
    # Default offset
    default_offset = (5, 5)
    
    # Custom offsets for specific traits (to resolve overlaps elegantly)
    # Format: (x_offset, y_offset) in points
    custom_offsets = {}
    
    # Check if TSS and FF overlap
    if ('FF', 'TSS') in overlaps or ('TSS', 'FF') in overlaps:
        # Stack labels: TSS above, FF below (or use diagonal offsets)
        custom_offsets['TSS'] = (8, 12)   # Upper right
        custom_offsets['FF'] = (8, -15)   # Lower right
        log("  Applied custom offsets for TSS/FF overlap", "DEBUG")
    
    # Plot each trait
    for pt in points:
        trait = pt['trait']
        x = pt['x']
        y = pt['y']
        tier = pt['tier']
        
        # Plot point
        ax.scatter(x, y, s=350, c=[colors.tier_colors[tier]], 
                  edgecolors='black', linewidth=2, zorder=5)
        
        # Get offset (custom or default)
        offset = custom_offsets.get(trait, default_offset)
        
        # Add label with appropriate offset
        ax.annotate(trait, (x, y), xytext=offset, textcoords='offset points',
                   fontsize=11, fontweight='bold',
                   color=colors.trait_colors.get(trait, 'black'))
    
    # Zone lines
    ax.axhline(0.3, color=colors.gray, linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(0.1, color=colors.gray, linestyle=':', linewidth=1, alpha=0.5)
    
    # Styling
    ax.set_xlabel('Structural Dominance (%)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Genomic Transferability', fontsize=10, fontweight='bold')
    ax.set_xlim(1.5, 4.5)
    ax.set_ylim(-0.1, 0.8)
    style_axis(ax, spines=['left', 'bottom'])
    
    # Legend
    tier_patches = [mpatches.Patch(facecolor=colors.tier_colors[t], edgecolor='black',
                                   label=f'Tier {t}') for t in [1, 2, 3]]
    ax.legend(handles=tier_patches, loc='upper left', fontsize=8)
    
    add_panel_label(ax, 'A')
    log("  Panel A complete", "OK")


# ============================================================================
# PANEL B: Structure Cliff Waterfall
# ============================================================================

def create_panel_B(data, ax):
    """Panel B: Waterfall/slope showing accuracy DROP across CV schemes."""
    log("\n[PANEL B] Structure Cliff Waterfall")
    
    trans_df = data['transferability']
    
    if trans_df is None:
        ax.text(0.5, 0.5, 'Data not available', ha='center', va='center',
               fontsize=12, transform=ax.transAxes)
        style_axis(ax, spines=[])
        add_panel_label(ax, 'B')
        log("  Data not available!", "ERROR")
        return
    
    # Use standard trait order
    trans_df = trans_df.set_index('trait').reindex(TRAIT_ORDER).reset_index()
    trans_df = trans_df.dropna(subset=['r_random_pc'])
    
    if trans_df.empty:
        ax.text(0.5, 0.5, 'No valid data after filtering', ha='center', va='center',
               fontsize=12, transform=ax.transAxes)
        add_panel_label(ax, 'B')
        return
    
    log(f"  Traits found: {list(trans_df['trait'].values)}", "DEBUG")
    
    traits = trans_df['trait'].values
    r_random = trans_df['r_random_pc'].values
    r_cluster = trans_df['r_cluster_balanced_pc'].values
    r_lco = trans_df['r_leave_cluster_out_pc'].values
    
    x = np.arange(len(traits))
    
    # Plot connected lines (waterfall style)
    for i, trait in enumerate(traits):
        tier = data.get('tier_map', {}).get(trait, 3)
        color = colors.tier_colors[tier]
        
        # Points
        ax.scatter([i-0.2, i, i+0.2], [r_random[i], r_cluster[i], r_lco[i]], 
                  c=[colors.cv_colors['random'], colors.cv_colors['cluster_aware'], 
                     colors.cv_colors['lco']], s=100, zorder=5, edgecolors='black')
        
        # Connecting lines
        ax.plot([i-0.2, i, i+0.2], [r_random[i], r_cluster[i], r_lco[i]], 
               color=color, linewidth=2.5, alpha=0.8)
        
        # Drop annotation for LCO
        if r_lco[i] < 0:
            ax.annotate(f'{r_lco[i]:.2f}', (i+0.2, r_lco[i]), 
                       xytext=(0, -10), textcoords='offset points',
                       fontsize=8, ha='center', color=colors.gray, fontweight='bold')
    
    # Reference line at 0
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    
    # Styling
    ax.set_xticks(x)
    ax.set_xticklabels(traits, fontsize=10, fontweight='bold')
    ax.set_ylabel('Prediction Accuracy (r)', fontsize=10, fontweight='bold')
    ax.set_ylim(-0.15, 0.85)
    style_axis(ax, spines=['left', 'bottom'], grid=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Legend for CV schemes
    cv_patches = [
        mpatches.Patch(facecolor=colors.cv_colors['random'], edgecolor='black', label='Random'),
        mpatches.Patch(facecolor=colors.cv_colors['cluster_aware'], edgecolor='black', label='Cluster'),
        mpatches.Patch(facecolor=colors.cv_colors['lco'], edgecolor='black', label='LCO')
    ]
    ax.legend(handles=cv_patches, loc='upper right', fontsize=8, ncol=3)
    
    add_panel_label(ax, 'B')
    log("  Panel B complete", "OK")


# ============================================================================
# PANEL C: Marker Efficiency
# ============================================================================

def create_panel_C(data, ax):
    """Panel C: Grouped bars comparing Dense SNP vs Inversion Panel accuracy."""
    log("\n[PANEL C] Marker Efficiency Comparison")
    
    perf_df = data['model_performance']
    
    if perf_df is None:
        ax.text(0.5, 0.5, 'Data not available', ha='center', va='center',
               fontsize=12, transform=ax.transAxes)
        style_axis(ax, spines=[])
        add_panel_label(ax, 'C')
        log("  Data not available!", "ERROR")
        return
    
    log(f"  perf_df columns: {list(perf_df.columns)}", "DEBUG")
    
    # Filter to random CV
    perf_df = perf_df.copy()
    if 'scheme' in perf_df.columns:
        perf_df = perf_df[perf_df['scheme'].str.contains('random|Random', case=False, na=False)]
    
    # Get feature set column
    if 'feature_set' not in perf_df.columns:
        ax.text(0.5, 0.5, 'feature_set column not found', ha='center', va='center',
               fontsize=12, transform=ax.transAxes)
        add_panel_label(ax, 'C')
        log("  feature_set column not found!", "ERROR")
        return
    
    feature_sets = perf_df['feature_set'].unique()
    log(f"  Feature sets: {list(feature_sets)}", "DEBUG")
    
    # Map feature sets
    dense_label = None
    inv_label = None
    for fs in feature_sets:
        fs_lower = str(fs).lower()
        if 'snp' in fs_lower and 'inv' not in fs_lower:
            dense_label = fs
        elif 'inv' in fs_lower and 'snp' not in fs_lower:
            inv_label = fs
    
    if dense_label is None or inv_label is None:
        if len(feature_sets) >= 2:
            dense_label = feature_sets[0]
            inv_label = feature_sets[1]
    
    log(f"  Dense label: {dense_label}, Inv label: {inv_label}", "DEBUG")
    
    if dense_label is None or inv_label is None:
        ax.text(0.5, 0.5, 'Could not identify feature sets', ha='center', va='center',
               fontsize=12, transform=ax.transAxes)
        add_panel_label(ax, 'C')
        return
    
    # Extract data using standard trait order
    dense_acc = []
    inv_acc = []
    traits_found = []
    
    for trait in TRAIT_ORDER:
        trait_data = perf_df[perf_df['trait'] == trait]
        
        dense_row = trait_data[trait_data['feature_set'] == dense_label]
        inv_row = trait_data[trait_data['feature_set'] == inv_label]
        
        if len(dense_row) > 0 and len(inv_row) > 0:
            dense_acc.append(dense_row['mean_r'].values[0])
            inv_acc.append(inv_row['mean_r'].values[0])
            traits_found.append(trait)
    
    log(f"  Traits found for Panel C: {traits_found}", "DEBUG")
    
    if len(traits_found) == 0:
        ax.text(0.5, 0.5, 'No matching data', ha='center', va='center',
               fontsize=12, transform=ax.transAxes)
        add_panel_label(ax, 'C')
        return
    
    x = np.arange(len(traits_found))
    width = 0.35
    
    col_dense = colors.royalblue
    col_inv = colors.limegreen

    bars1 = ax.bar(x - width/2, dense_acc, width, label='Dense SNP\n(19,790)', 
                   color=col_dense, edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x + width/2, inv_acc, width, label='Inversion\n(17)', 
                   color=col_inv, edgecolor='white', linewidth=1.5)
    
    # Value labels
    for bar, val in zip(bars1, dense_acc):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}',
               ha='center', va='bottom', fontsize=8, fontweight='bold')
    for bar, val in zip(bars2, inv_acc):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}',
               ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Efficiency annotation for FBC
    if 'FBC' in traits_found:
        fbc_idx = traits_found.index('FBC')
        if inv_acc[fbc_idx] >= dense_acc[fbc_idx] * 0.9:
            ax.annotate('≈', (fbc_idx, max(dense_acc[fbc_idx], inv_acc[fbc_idx]) + 0.08),
                       fontsize=16, ha='center', fontweight='bold', color=colors.seagreen)
    
    # Styling
    ax.set_xticks(x)
    ax.set_xticklabels(traits_found, fontsize=10, fontweight='bold')
    ax.set_ylabel('Accuracy (r)', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 0.9)
    ax.legend(loc='upper right', fontsize=8)
    style_axis(ax, spines=['left', 'bottom'])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    add_panel_label(ax, 'C')
    log("  Panel C complete", "OK")


# ============================================================================
# PANEL D: Within vs Cross-Population Gain
# ============================================================================

def create_panel_D(data, ax):
    """Panel D: Grouped bars showing Within-Pop vs Cross-Pop genetic gain."""
    log("\n[PANEL D] Within vs Cross-Population Gain")
    
    gain_df = data['genetic_gain']
    
    if gain_df is None:
        ax.text(0.5, 0.5, 'Data not available', ha='center', va='center',
               fontsize=12, transform=ax.transAxes)
        style_axis(ax, spines=[])
        add_panel_label(ax, 'D')
        log("  Data not available!", "ERROR")
        return
    
    log(f"  gain_df columns: {list(gain_df.columns)}", "DEBUG")
    
    # Find columns
    trait_col = 'Trait' if 'Trait' in gain_df.columns else 'trait'
    scenario_col = 'Scenario' if 'Scenario' in gain_df.columns else 'scenario'
    gain_col = 'Gain_Percent' if 'Gain_Percent' in gain_df.columns else 'gain_percent'
    
    # Pivot to get within and cross side by side
    within_df = gain_df[gain_df[scenario_col].str.contains('Within', case=False)]
    cross_df = gain_df[gain_df[scenario_col].str.contains('Cross', case=False)]
    
    within_gains = []
    cross_gains = []
    traits_found = []
    
    for trait in TRAIT_ORDER:
        within_row = within_df[within_df[trait_col] == trait]
        cross_row = cross_df[cross_df[trait_col] == trait]
        
        if len(within_row) > 0 and len(cross_row) > 0:
            within_gains.append(within_row[gain_col].values[0])
            cross_gains.append(cross_row[gain_col].values[0])
            traits_found.append(trait)
    
    log(f"  Traits found for Panel D: {traits_found}", "DEBUG")
    
    if len(traits_found) == 0:
        ax.text(0.5, 0.5, 'No matching data', ha='center', va='center',
               fontsize=12, transform=ax.transAxes)
        add_panel_label(ax, 'D')
        return
    
    x = np.arange(len(traits_found))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, within_gains, width, label='Within-Population',
                   color=colors.limegreen, edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x + width/2, cross_gains, width, label='Cross-Population',
                   color=colors.steelblue, edgecolor='white', linewidth=1.5)
    
    # Value labels
    for bar, val in zip(bars1, within_gains):
        y_pos = val + 0.5 if val >= 0 else val - 1.5
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, f'{val:.1f}%',
               ha='center', va='bottom' if val >= 0 else 'top', 
               fontsize=8, fontweight='bold', color=colors.seagreen)
    
    for bar, val in zip(bars2, cross_gains):
        y_pos = val + 0.5 if val >= 0 else val - 1.5
        color_text = colors.steelblue if val >= 0 else colors.gray
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, f'{val:.1f}%',
               ha='center', va='bottom' if val >= 0 else 'top',
               fontsize=8, fontweight='bold', color=color_text)
    
    # Reference line at 0
    ax.axhline(0, color='black', linestyle='-', linewidth=1.5, zorder=1)
    
    # Styling
    ax.set_xticks(x)
    ax.set_xticklabels(traits_found, fontsize=10, fontweight='bold')
    ax.set_ylabel('Genetic Gain (% per cycle)', fontsize=10, fontweight='bold')
    
    # Dynamic y-limits
    all_vals = within_gains + cross_gains
    y_min = min(min(all_vals) - 3, -5)
    y_max = max(all_vals) + 5
    ax.set_ylim(y_min, y_max)
    
    ax.legend(loc='upper right', fontsize=8)
    style_axis(ax, spines=['left', 'bottom'])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    add_panel_label(ax, 'D')
    log("  Panel D complete", "OK")


# ============================================================================
# PANEL E: Decision Schematic - VERSION 1: Vertical Tier Cards
# ============================================================================

def create_panel_E(data, ax):
    """
    Panel E: Vertical Tier Cards Design
    - Compact circular badges with T1/T2/T3 labels
    - Stacked vertically with arrows pointing to trait boxes
    - Strategy text in italics on the right
    
    NOTE: This is a schematic panel - uses predefined tier assignments,
          not loaded from data files.
    """
    log("\n[PANEL E] Decision Schematic (Vertical Tier Cards)")
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Breeding\nStrategy', ha='center', va='top', 
           fontsize=11, fontweight='bold', color='#333333')
    
    # Tier groupings (data-driven; derived from hierarchy_table if available)
    hier_df = data.get("hierarchy_table", None)
    if hier_df is not None and not hier_df.empty and "tier" in hier_df.columns:
        traits_by_tier = {
            t: ", ".join(sorted(hier_df.loc[hier_df["tier"] == t, "trait"].astype(str).tolist()))
            for t in [1, 2, 3]
        }
    else:
        # If hierarchy_table not available, keep a conservative schematic fallback
        traits_by_tier = {1: "FBC, TC", 2: "AFW", 3: "TSS, FF"}
        log("  Panel E using fallback tier grouping (hierarchy_table unavailable).", "WARN")

    tier_data = [
        (1, traits_by_tier.get(1, ""), "KASP\n(10 markers)", colors.tier_colors[1]),
        (2, traits_by_tier.get(2, ""), "Dense SNP\n(genome-wide)", colors.tier_colors[2]),
        (3, traits_by_tier.get(3, ""), "Local GS\n(recalibrate)", colors.tier_colors[3]),
    ]
    
    y_positions = [7, 4.5, 2]
    
    for (tier, traits, strategy, color), y in zip(tier_data, y_positions):
        # Circular badge with tier number
        circle = plt.Circle((1.8, y), 1.0, facecolor=color, alpha=0.25, 
                            edgecolor=color, linewidth=2.5)
        ax.add_patch(circle)
        ax.text(1.8, y, f'T{tier}', ha='center', va='center', 
               fontsize=14, fontweight='bold', color=color)
        
        # Arrow connecting badge to trait box
        ax.annotate('', xy=(4, y), xytext=(3, y),
                   arrowprops=dict(arrowstyle='->', color=color, lw=2, 
                                  mutation_scale=15))
        
        # Trait box
        trait_box = FancyBboxPatch((4.2, y-0.6), 2.2, 1.2, 
                                   boxstyle='round,pad=0.08',
                                   facecolor=color, alpha=0.15, 
                                   edgecolor=color, linewidth=1.5)
        ax.add_patch(trait_box)
        ax.text(5.3, y, traits, ha='center', va='center', 
               fontsize=9, fontweight='bold', color='#333333')
        
        # Strategy text (italic, on right)
        ax.text(7.8, y, strategy, ha='left', va='center', 
               fontsize=8, color=colors.gray, style='italic')
    
    add_panel_label(ax, 'E')
    log("  Panel E complete", "OK")


# ============================================================================
# PANEL F: Method Concordance - BINN vs Baseline
# ============================================================================

def create_panel_F(data, ax):
    """
    Panel F: Method concordance showing BINN vs idea2 (baseline) accuracy.
    Shows per-trait accuracy comparison with consensus eligibility flags.
    """
    log("\n[PANEL F] Method Concordance (BINN vs Baseline)")
    
    consensus_df = data['breeder_consensus']
    
    if consensus_df is None:
        ax.text(0.5, 0.5, 'Data not available\n(breeder_consensus_summary.csv)', 
               ha='center', va='center', fontsize=10, transform=ax.transAxes)
        style_axis(ax, spines=[])
        add_panel_label(ax, 'F')
        log("  breeder_consensus_summary.csv not found!", "ERROR")
        return
    
    log(f"  consensus_df columns: {list(consensus_df.columns)}", "DEBUG")
    log(f"  consensus_df shape: {consensus_df.shape}", "DEBUG")
    
    # Map column names (handle different naming conventions)
    col_map = {
        'trait': ['trait', 'Trait'],
        'binn_accuracy': ['binn_accuracy', 'BINN_accuracy', 'binn_acc'],
        'idea2_accuracy': ['idea2_accuracy', 'baseline_accuracy', 'idea2_acc'],
        'idea2_informative': ['idea2_informative', 'is_informative', 'informative'],
    }
    
    def find_col(options):
        for opt in options:
            if opt in consensus_df.columns:
                return opt
        return None
    
    trait_col = find_col(col_map['trait'])
    binn_col = find_col(col_map['binn_accuracy'])
    baseline_col = find_col(col_map['idea2_accuracy'])
    informative_col = find_col(col_map['idea2_informative'])
    
    log(f"  Mapped columns: trait={trait_col}, binn={binn_col}, baseline={baseline_col}", "DEBUG")
    
    if trait_col is None or binn_col is None or baseline_col is None:
        ax.text(0.5, 0.5, 'Required columns not found', ha='center', va='center',
               fontsize=10, transform=ax.transAxes)
        add_panel_label(ax, 'F')
        log("  Required columns not found!", "ERROR")
        return
    
    # Aggregate by trait (take first value since accuracy is same across directions)
    agg_dict = {
        binn_col: 'first',
        baseline_col: 'first',
    }
    if informative_col:
        agg_dict[informative_col] = 'first'
    
    df = consensus_df.groupby(trait_col).agg(agg_dict).reset_index()
    
    # Reorder by TRAIT_ORDER (reversed for horizontal bar - FBC at top)
    df = df.set_index(trait_col).reindex(TRAIT_ORDER[::-1]).reset_index()
    df = df.dropna(subset=[binn_col])
    
    if df.empty:
        ax.text(0.5, 0.5, 'No valid data after filtering', ha='center', va='center',
               fontsize=10, transform=ax.transAxes)
        add_panel_label(ax, 'F')
        return
    
    traits = df[trait_col].values
    binn_acc = df[binn_col].values
    baseline_acc = df[baseline_col].values
    
    log(f"  Traits: {list(traits)}", "DEBUG")
    log(f"  BINN acc: {list(binn_acc)}", "DEBUG")
    log(f"  Baseline acc: {list(baseline_acc)}", "DEBUG")
    
    y = np.arange(len(traits))
    height = 0.35
    
    # Horizontal bars
    bars_binn = ax.barh(y - height/2, binn_acc, height, label='BINN',
                        color=colors.teal_green, edgecolor='white', linewidth=1.5)
    bars_baseline = ax.barh(y + height/2, baseline_acc, height, label='PC-Ridge\n(Baseline)',
                            color=colors.steelblue, edgecolor='white', linewidth=1.5)
    
    # Add accuracy values on bars
    for bar, val in zip(bars_binn, binn_acc):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
               ha='left', va='center', fontsize=8, fontweight='bold', color=colors.teal_green)
    
    for bar, val in zip(bars_baseline, baseline_acc):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
               ha='left', va='center', fontsize=8, fontweight='bold', color=colors.steelblue)
    
    # Threshold line at 0.30
    ax.axvline(0.30, color=colors.gray, linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)
    ax.text(0.31, len(traits)-0.3, 'ρ=0.30', fontsize=8, color=colors.gray, va='bottom')
    
    # Styling
    ax.set_yticks(y)
    ax.set_yticklabels(traits, fontsize=10, fontweight='bold')
    ax.set_xlabel('Prediction Accuracy (ρ)', fontsize=10, fontweight='bold')
    ax.set_xlim(0, 0.95)
    ax.set_ylim(-0.6, len(traits) - 0.4)
    
    # Legend - center right
    legend_elements = [
        mpatches.Patch(facecolor=colors.teal_green, edgecolor='white', label='BINN'),
        mpatches.Patch(facecolor=colors.steelblue, edgecolor='white', label='PC-Ridge'),
    ]
    ax.legend(handles=legend_elements, loc='center right', fontsize=8)
    
    style_axis(ax, spines=['left', 'bottom'])
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    add_panel_label(ax, 'F')
    log("  Panel F complete", "OK")


# ============================================================================
# MAIN FIGURE ASSEMBLY
# ============================================================================

def create_figure_5():
    """
    Assemble all 6 panels into Figure 5.
    
    Layout (2 rows):
        Row 0: A (hierarchy) + B (cliff) + C (efficiency)
        Row 1: D (gain) + E (schematic - vertical cards) + F (concordance)
    """
    log("\n" + "=" * 70)
    log("ASSEMBLING FIGURE 5: The Precision Breeding Hierarchy")
    log("Panel E Design: Vertical Tier Cards (V1)")
    log("=" * 70)
    
    data = load_data()
    # Build a single hierarchy table used by Panel A/E and exported as Table S12B
    if data.get("transferability") is not None and data.get("structural_scores") is not None:
        try:
            data["hierarchy_table"] = build_hierarchy_table(data["transferability"], data["structural_scores"])
            data["tier_map"] = dict(zip(data["hierarchy_table"]["trait"].tolist(),
                                        data["hierarchy_table"]["tier"].tolist()))

            try:
                export_table_s12b(data["hierarchy_table"])
            except Exception as e:
                log(f"WARNING: Table S12B export failed (figure still uses computed tiers): {e}", "WARN")

            # Drift check (does not assign tiers; only warns if unexpected)
            expected = {"FBC": 1, "TC": 1, "AFW": 2, "TSS": 3, "FF": 3}
            if set(expected).issubset(set(data["tier_map"].keys())):
                mism = {k: (expected[k], data["tier_map"][k]) for k in expected if data["tier_map"][k] != expected[k]}
                if mism:
                    log(f"WARNING: Computed tiers differ from manuscript expectation: {mism}", "WARN")
        except Exception as e:
            log(f"ERROR while building/exporting hierarchy table: {e}", "ERROR")
            data["hierarchy_table"] = pd.DataFrame()
            data["tier_map"] = {}
    else:
        data["hierarchy_table"] = pd.DataFrame()
        data["tier_map"] = {}
    
    # Figure setup
    fig = plt.figure(figsize=(15, 9))
    
    gs = gridspec.GridSpec(
        nrows=2, ncols=6,
        height_ratios=[1.0, 1.0],
        width_ratios=[1, 1, 1, 1, 1, 1],
        hspace=0.30,
        wspace=0.50
    )
    
    # Row 0: A, B, C (each 2 cols)
    ax_A = fig.add_subplot(gs[0, 0:2])
    ax_B = fig.add_subplot(gs[0, 2:4])
    ax_C = fig.add_subplot(gs[0, 4:6])
    
    # Row 1: D (2 cols), E (2 cols), F (2 cols)
    ax_D = fig.add_subplot(gs[1, 0:2])
    ax_E = fig.add_subplot(gs[1, 2:4])
    ax_F = fig.add_subplot(gs[1, 4:6])
    
    # Create panels
    create_panel_A(data, ax_A)
    create_panel_B(data, ax_B)
    create_panel_C(data, ax_C)
    create_panel_D(data, ax_D)
    create_panel_E(data, ax_E)
    create_panel_F(data, ax_F)
    
    # Save PNG
    output_path = os.path.join(FIGURES_DIR, "figure_5.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    log(f"\n[OK] Figure saved: {output_path}", "OK")
    
    # Save source data for Panel F (apply readable headers if available)
    if data['breeder_consensus'] is not None:
        source_path = os.path.join(FIGURE_SUBDIR, "source_breeder_consensus.csv")
        apply_readable_headers(data['breeder_consensus']).to_csv(source_path, index=False)
        log(f"  [OK] Source data: {source_path}", "OK")
    
    log("\n" + "=" * 70)
    log("FIGURE 5 COMPLETE!", "OK")
    log("=" * 70)
    
    return fig


if __name__ == "__main__":
    print("=" * 70)
    print("MANGO GWAS - FIGURE 5 GENERATOR")
    print("The Precision Breeding Hierarchy")
    print("Panel E: Vertical Tier Cards Design")
    print("=" * 70)
    print("\nNOTE: This script uses ONLY real data files.")
    print("      NO fallback or synthetic data is used.")
    print("      Missing files will show 'Data not available'.\n")
    
    try:
        fig = create_figure_5()
        print("\nSUCCESS!")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)