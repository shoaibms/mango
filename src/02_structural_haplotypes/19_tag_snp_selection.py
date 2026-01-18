#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
19_tag_snp_selection.py

Minimal, marker-development-grade validation that each miinv inversion (0/1/2 dosage)
can be tagged by 1–2 SNPs with strong LD (r²) overall and stable across clusters.

Outputs:
  - output/idea_2/breeder_tools/tag_snps/inversion_tag_snps_selected.csv
  - output/idea_2/breeder_tools/tag_snps/inversion_tag_snps_candidates_topK.csv
  - output/idea_2/breeder_tools/tag_snps/inversion_tag_snps_manuscript_table.csv

Updates (v4):
  - OPTIMIZED: Memory-efficient chunk-wise indexing (no full matrix materialization)
  - FIXED: tag2 must be on SAME chromosome as tag1 (cis-LD, not structure correlation)
  - FIXED: Cluster labels explicitly aligned by sample_id (not row order assumption)
  - Physical distance filter: tag2 must be ≥1kb from tag1 for true redundancy
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import openpyxl  # noqa: F401
except ImportError as e:
    raise SystemExit(
        "openpyxl is required to read the Dataset S1 Excel file.\n"
        "Install it with:\n\n  pip install openpyxl\n"
    ) from e


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_ROOT = r"C:\Users\ms\Desktop\mango"
DEFAULT_TOP_K = 200
DEFAULT_MIN_CLUSTER_N = 20
DEFAULT_R2_GLOBAL_THRESH = 0.80
DEFAULT_R2_CLUSTER_THRESH = 0.70
DEFAULT_MIN_TAG_DISTANCE = 1000  # Minimum bp between tag1 and tag2 for redundancy
DEFAULT_CHUNK_SIZE = 50_000      # SNPs per chunk for memory efficiency


# =============================================================================
# Variant ID Parsing
# =============================================================================

_VARIANT_PATTERNS = [
    re.compile(r"^(?P<chr>[^:]+):(?P<pos>\d+):(?P<ref>[ACGT]):(?P<alt>[ACGT])$", re.I),
    re.compile(r"^(?P<chr>[^:]+):(?P<pos>\d+)[\-_](?P<ref>[ACGT])[\-_](?P<alt>[ACGT])$", re.I),
    re.compile(r"^(?P<chr>[^_]+)_(?P<pos>\d+)_(?P<ref>[ACGT])_(?P<alt>[ACGT])$", re.I),
    re.compile(r"^(?P<chr>[^:]+):(?P<pos>\d+)$", re.I),
    re.compile(r"^(?P<chr>[^_]+)_(?P<pos>\d+)$", re.I),
]


def parse_variant_id(variant_id: str) -> Tuple[Optional[str], Optional[int], Optional[str], Optional[str]]:
    """Parse variant ID to extract chrom, pos, ref, alt."""
    vid = str(variant_id).strip()
    for pat in _VARIANT_PATTERNS:
        m = pat.match(vid)
        if m:
            chrom = m.groupdict().get("chr")
            pos_s = m.groupdict().get("pos")
            pos = int(pos_s) if pos_s is not None else None
            ref = m.groupdict().get("ref")
            alt = m.groupdict().get("alt")
            return chrom, pos, (ref.upper() if ref else None), (alt.upper() if alt else None)
    return None, None, None, None


# =============================================================================
# Statistical Utilities
# =============================================================================

def maf_from_dosage(x: np.ndarray) -> float:
    """Compute minor allele frequency from dosage array."""
    p = float(np.mean(x) / 2.0)
    return min(p, 1.0 - p)


def safe_r2_from_dot(dot_xy: np.ndarray, n: int, std_x: np.ndarray, std_y: float) -> np.ndarray:
    """Compute r² from dot product, handling edge cases."""
    denom = (n * std_x * std_y)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(denom > 0, dot_xy / denom, 0.0)
        r2 = np.nan_to_num(r * r, nan=0.0, posinf=0.0, neginf=0.0)
    return r2


def dosage_concordance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute concordance between true and predicted dosages (0/1/2)."""
    y_hat = np.rint(np.clip(y_pred, 0, 2)).astype(int)
    y_true_i = np.rint(np.clip(y_true, 0, 2)).astype(int)
    return float(np.mean(y_hat == y_true_i))


# =============================================================================
# Tag Candidate Data Structure
# =============================================================================

@dataclass
class TagCandidate:
    """Container for tag SNP candidate information."""
    inversion: str
    variant_id: str
    chrom: Optional[str]
    pos: Optional[int]
    ref: Optional[str]
    alt: Optional[str]
    r2_overall: float
    maf: float
    r2_min_cluster: float
    r2_mean_cluster: float
    r2_by_cluster: Dict[int, float] = field(default_factory=dict)


# =============================================================================
# Core LD Computation (MEMORY-EFFICIENT VERSION)
# =============================================================================

def compute_r2_scan_memeff(
    X: np.memmap,
    y: np.ndarray,
    row_idx: np.ndarray,
    chunk_size: int = DEFAULT_CHUNK_SIZE
) -> np.ndarray:
    """
    Genome-wide LD scan: compute r²(SNP dosage, inversion dosage) for all SNPs.
    
    MEMORY-EFFICIENT: Reads from memmap chunk-wise using row_idx, never
    materializing the full aligned matrix. Peak RAM ≈ one chunk.
    
    Args:
        X: Memory-mapped genotype matrix (n_samples × n_snps)
        y: Inversion dosage vector (already aligned to row_idx order)
        row_idx: Row indices to use from X (sample alignment)
        chunk_size: Number of SNPs to process per chunk
    
    Returns:
        r² array of length n_snps
    """
    n = len(row_idx)
    p = X.shape[1]
    
    y = y.astype(np.float64)
    y_centered = y - float(np.mean(y))
    std_y = float(np.std(y_centered, ddof=0))
    
    if std_y == 0.0:
        return np.zeros(p, dtype=np.float64)

    r2_all = np.zeros(p, dtype=np.float64)
    
    for j0 in range(0, p, chunk_size):
        j1 = min(p, j0 + chunk_size)
        
        # Index into memmap: only load the rows we need, for this chunk of columns
        # This is the key memory optimization: X[row_idx, j0:j1] reads only what's needed
        Xc = np.asarray(X[row_idx, j0:j1], dtype=np.float64)
        
        Xc_centered = Xc - Xc.mean(axis=0)
        std_x = np.std(Xc, axis=0, ddof=0)
        dot_xy = Xc_centered.T @ y_centered
        r2_all[j0:j1] = safe_r2_from_dot(dot_xy, n=n, std_x=std_x, std_y=std_y)
    
    return r2_all


def get_snp_column_memeff(X: np.memmap, row_idx: np.ndarray, col_idx: int) -> np.ndarray:
    """
    Extract a single SNP column from memmap using row indices.
    Memory-efficient: only loads the required rows.
    """
    return np.asarray(X[row_idx, col_idx], dtype=np.float64)


def per_cluster_r2(
    X_col: np.ndarray, 
    y: np.ndarray, 
    clusters: np.ndarray, 
    min_n: int = 20
) -> Tuple[float, float, Dict[int, float]]:
    """
    Compute r² within each ancestry cluster.
    Returns (min_r2, mean_r2, dict of per-cluster r²).
    """
    r2s: Dict[int, float] = {}
    valid = []
    
    for c in sorted(set(int(v) for v in clusters if not np.isnan(v))):
        mask = clusters == c
        if int(mask.sum()) < min_n:
            continue
        
        x = X_col[mask].astype(np.float64)
        yy = y[mask].astype(np.float64)
        
        x_centered = x - float(np.mean(x))
        yy_centered = yy - float(np.mean(yy))
        
        sx = float(np.std(x, ddof=0))
        sy = float(np.std(yy, ddof=0))
        
        if sx == 0.0 or sy == 0.0:
            r2 = 0.0
        else:
            r = float((x_centered @ yy_centered) / (len(x) * sx * sy))
            r2 = r * r
        
        r2s[c] = r2
        valid.append(r2)
    
    if not valid:
        return 0.0, 0.0, r2s
    
    return float(min(valid)), float(np.mean(valid)), r2s


def regression_checks(y: np.ndarray, X_cols: np.ndarray) -> Tuple[float, float]:
    """
    OLS regression check: inv ~ tag1 (+ tag2).
    Returns (R², dosage_concordance).
    """
    y = y.astype(np.float64)
    y_mean = float(np.mean(y))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    
    if ss_tot == 0.0:
        return 0.0, 1.0
    
    A = np.column_stack([np.ones(len(y)), X_cols.astype(np.float64)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    ss_res = float(np.sum((y - yhat) ** 2))
    r2 = 1.0 - (ss_res / ss_tot)
    acc = dosage_concordance(y, yhat)
    
    return float(r2), float(acc)


# =============================================================================
# Tag SNP Selection (same-chromosome only for tag2)
# =============================================================================

def select_tag_snps(
    candidates: List[TagCandidate],
    min_tag_distance: int = DEFAULT_MIN_TAG_DISTANCE
) -> Tuple[Optional[TagCandidate], Optional[TagCandidate]]:
    """
    Select tag1 (best overall) and tag2 (best redundant, same chrom, ≥min_tag_distance from tag1).
    
    IMPORTANT: tag2 must be on the SAME chromosome as tag1 to ensure cis-LD tagging.
    Trans-chromosome correlations reflect population structure, not physical linkage
    to the inversion, and are NOT suitable for marker development.
    """
    if not candidates:
        return None, None
    
    # Sort by min_cluster r² (stability), then by overall r² (power)
    sorted_cands = sorted(
        candidates, 
        key=lambda c: (c.r2_min_cluster, c.r2_overall), 
        reverse=True
    )
    
    tag1 = sorted_cands[0]
    tag2 = None
    
    # Find tag2: must be on SAME chromosome and ≥min_tag_distance from tag1
    if tag1.pos is not None and tag1.chrom is not None:
        for cand in sorted_cands[1:]:
            # CRITICAL: Only consider same-chromosome candidates for cis-LD tagging
            if cand.chrom != tag1.chrom:
                continue  # Skip trans-chrom (would be structure correlation, not LD)
            
            if cand.pos is not None:
                dist = abs(cand.pos - tag1.pos)
                if dist >= min_tag_distance:
                    tag2 = cand
                    break
    
    return tag1, tag2


def classify_tag_status(
    r2_overall: float,
    r2_min_cluster: float,
    r2_global_thresh: float = DEFAULT_R2_GLOBAL_THRESH,
    r2_cluster_thresh: float = DEFAULT_R2_CLUSTER_THRESH
) -> str:
    """
    Classify tag SNP deployment status.
    """
    if r2_overall >= r2_global_thresh and r2_min_cluster >= r2_cluster_thresh:
        return "GLOBAL"
    elif r2_overall >= r2_global_thresh and r2_min_cluster < r2_cluster_thresh:
        return "RESTRICTED"
    elif r2_overall >= 0.5:
        return "MODERATE"
    else:
        return "WEAK"


# =============================================================================
# I/O and Main Pipeline
# =============================================================================

def load_inversion_dosages(xlsx_path: str, sheet_name: str = "Dataset S1") -> pd.DataFrame:
    """Load miinv* inversion dosages from the supplementary Excel file."""
    print(f"[INFO] Loading inversion dosages from: {xlsx_path}")
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    
    # Identify inversion columns (miinv*)
    inv_cols = [c for c in df.columns if str(c).lower().startswith("miinv")]
    print(f"[INFO] Found {len(inv_cols)} inversion columns: {inv_cols[:5]}...")
    
    # Identify sample ID column
    id_col = None
    for c in ["Accession", "Sample", "ID", "sample_id"]:
        if c in df.columns:
            id_col = c
            break
    
    if id_col is None:
        id_col = df.columns[0]
    
    print(f"[INFO] Using '{id_col}' as sample identifier")
    
    return df[[id_col] + inv_cols].rename(columns={id_col: "sample_id"})


def align_clusters_by_sample_id(
    metadata_path: Path,
    sample_ids: np.ndarray
) -> np.ndarray:
    """
    Align cluster labels to sample_ids by explicit join (not row-order assumption).
    """
    metadata_df = pd.read_csv(metadata_path)
    
    # Find sample ID column in metadata
    id_col = None
    for c in ["sample_id", "Sample", "ID", "Accession"]:
        if c in metadata_df.columns:
            id_col = c
            break
    if id_col is None:
        id_col = metadata_df.columns[0]
    
    # Reindex to match sample_ids order
    metadata_df[id_col] = metadata_df[id_col].astype(str)
    clusters_series = (
        metadata_df.set_index(id_col)["cluster_kmeans"]
        .reindex(sample_ids)
    )
    
    # Sanity check: fail if any NaN after alignment
    n_missing = clusters_series.isna().sum()
    if n_missing > 0:
        print(f"[ERROR] {n_missing} samples in samples.csv have no cluster assignment in metadata!")
        print(f"        First few missing: {list(sample_ids[clusters_series.isna()][:5])}")
        sys.exit(1)
    
    return clusters_series.values.astype(int)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Select and validate 1–2 tag SNPs per miinv inversion (Idea 2)."
    )
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help="Project root directory")
    ap.add_argument("--top_k", type=int, default=DEFAULT_TOP_K,
                    help="Number of top candidates to retain per inversion")
    ap.add_argument("--min_cluster_n", type=int, default=DEFAULT_MIN_CLUSTER_N,
                    help="Minimum samples per cluster for r² calculation")
    ap.add_argument("--r2_global_thresh", type=float, default=DEFAULT_R2_GLOBAL_THRESH,
                    help="r² threshold for global deployability")
    ap.add_argument("--r2_cluster_thresh", type=float, default=DEFAULT_R2_CLUSTER_THRESH,
                    help="Minimum cluster r² for stability")
    ap.add_argument("--min_tag_distance", type=int, default=DEFAULT_MIN_TAG_DISTANCE,
                    help="Minimum bp between tag1 and tag2 for redundancy")
    ap.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE,
                    help="SNPs per chunk for memory-efficient processing")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    
    # ---------------------------------------------------------------------
    # Paths
    # ---------------------------------------------------------------------
    x_path = root / "output" / "idea_2" / "core_ml" / "X_full.npy"
    samples_path = root / "output" / "idea_2" / "core_ml" / "samples.csv"
    snp_ids_path = root / "output" / "idea_2" / "core_ml" / "snp_ids.csv"
    metadata_path = root / "output" / "idea_2" / "core_ml" / "sample_metadata_ml.csv"
    inv_xlsx_path = root / "data" / "main_data" / "nph20252-sup-0001-datasetss1-s3.xlsx"
    
    out_dir = root / "output" / "idea_2" / "breeder_tools" / "tag_snps"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_selected = out_dir / "inversion_tag_snps_selected.csv"
    out_candidates = out_dir / "inversion_tag_snps_candidates_topK.csv"
    out_manuscript = out_dir / "inversion_tag_snps_manuscript_table.csv"
    
    print("=" * 70)
    print("TAG SNP SELECTION PER INVERSION (v4 - memory-efficient)")
    print("=" * 70)
    print(f"[CONFIG] Top K candidates: {args.top_k}")
    print(f"[CONFIG] r² global threshold: {args.r2_global_thresh}")
    print(f"[CONFIG] r² cluster threshold: {args.r2_cluster_thresh}")
    print(f"[CONFIG] Min tag distance: {args.min_tag_distance} bp")
    print(f"[CONFIG] Chunk size: {args.chunk_size} SNPs (memory optimization)")
    print(f"[CONFIG] tag2 selection: SAME CHROMOSOME ONLY (cis-LD requirement)")
    print()
    
    # ---------------------------------------------------------------------
    # Load data (memmap for genotypes)
    # ---------------------------------------------------------------------
    print("[LOAD] SNP dosage matrix (memory-mapped)...")
    X = np.load(str(x_path), mmap_mode="r")
    n_samples, n_snps = X.shape
    print(f"       Shape: {n_samples} samples × {n_snps} SNPs")
    print(f"       Mode: memmap (peak RAM ≈ {args.chunk_size * n_samples * 8 / 1e6:.1f} MB per chunk)")
    
    samples_df = pd.read_csv(samples_path)
    sample_ids = samples_df.iloc[:, 0].astype(str).values
    
    snp_df = pd.read_csv(snp_ids_path)
    snp_ids = snp_df.iloc[:, 0].astype(str).values
    
    # Align clusters by sample_id
    print("[LOAD] Cluster assignments (aligned by sample_id)...")
    clusters = align_clusters_by_sample_id(metadata_path, sample_ids)
    print(f"       Clusters: {sorted(set(clusters))}")
    
    # Load inversions
    if not inv_xlsx_path.exists():
        raise FileNotFoundError(f"Inversion file not found: {inv_xlsx_path}")
    
    inv_df = load_inversion_dosages(str(inv_xlsx_path))
    inv_df["sample_id"] = inv_df["sample_id"].astype(str)
    
    # Build row index mapping (DO NOT materialize full matrix)
    sample_to_idx = {s: i for i, s in enumerate(sample_ids)}
    inv_row_idx = []
    inv_keep_mask = []
    
    for sid in inv_df["sample_id"]:
        if sid in sample_to_idx:
            inv_row_idx.append(sample_to_idx[sid])
            inv_keep_mask.append(True)
        else:
            inv_keep_mask.append(False)
    
    inv_df = inv_df[inv_keep_mask].reset_index(drop=True)
    inv_row_idx = np.array(inv_row_idx, dtype=np.int64)
    clusters_aligned = clusters[inv_row_idx]
    
    print(f"[INFO] Aligned {len(inv_row_idx)} samples between SNPs and inversions")
    print(f"[INFO] Memory mode: chunk-wise indexing (no full matrix copy)")
    
    inv_cols = [c for c in inv_df.columns if c.lower().startswith("miinv")]
    
    # ---------------------------------------------------------------------
    # Process each inversion
    # ---------------------------------------------------------------------
    all_candidates: List[Dict] = []
    selected_tags: List[Dict] = []
    
    for inv_name in inv_cols:
        print(f"\n[INVERSION] {inv_name}")
        
        y_inv = inv_df[inv_name].values.astype(np.float64)
        
        # Skip if no variance
        if np.std(y_inv) < 1e-6:
            print(f"  [SKIP] No variance in inversion dosage")
            continue
        
        # Genome-wide LD scan (MEMORY-EFFICIENT)
        print(f"  [SCAN] Computing r² for {n_snps:,} SNPs (chunk-wise)...")
        r2_all = compute_r2_scan_memeff(X, y_inv, inv_row_idx, chunk_size=args.chunk_size)
        
        # Top-K shortlist
        top_k_idx = np.argsort(r2_all)[-args.top_k:][::-1]
        print(f"  [TOP-K] Best r²: {r2_all[top_k_idx[0]]:.4f}, K={args.top_k} cutoff: {r2_all[top_k_idx[-1]]:.4f}")
        
        # Evaluate candidates
        candidates: List[TagCandidate] = []
        
        for idx in top_k_idx:
            vid = snp_ids[idx]
            chrom, pos, ref, alt = parse_variant_id(vid)
            
            # Get SNP column (memory-efficient)
            x_snp = get_snp_column_memeff(X, inv_row_idx, idx)
            maf = maf_from_dosage(x_snp)
            r2_overall = r2_all[idx]
            
            # Per-cluster r²
            r2_min, r2_mean, r2_by_cluster = per_cluster_r2(
                x_snp, y_inv, clusters_aligned, min_n=args.min_cluster_n
            )
            
            cand = TagCandidate(
                inversion=inv_name,
                variant_id=vid,
                chrom=chrom,
                pos=pos,
                ref=ref,
                alt=alt,
                r2_overall=r2_overall,
                maf=maf,
                r2_min_cluster=r2_min,
                r2_mean_cluster=r2_mean,
                r2_by_cluster=r2_by_cluster
            )
            candidates.append(cand)
            
            # Store for candidates output
            all_candidates.append({
                "inversion": inv_name,
                "variant_id": vid,
                "chrom": chrom,
                "pos": pos,
                "ref": ref,
                "alt": alt,
                "r2_overall": r2_overall,
                "maf": maf,
                "r2_min_cluster": r2_min,
                "r2_mean_cluster": r2_mean,
                **{f"r2_cluster_{k}": v for k, v in r2_by_cluster.items()}
            })
        
        # Select tag1 and tag2 (same-chromosome only for cis-LD)
        tag1, tag2 = select_tag_snps(candidates, min_tag_distance=args.min_tag_distance)
        
        if tag1 is None:
            print(f"  [WARN] No valid tag SNP found")
            continue
        
        # Regression checks
        x_tag1 = get_snp_column_memeff(X, inv_row_idx, np.where(snp_ids == tag1.variant_id)[0][0])
        reg_r2_1, concordance_1 = regression_checks(y_inv, x_tag1.reshape(-1, 1))
        
        reg_r2_2, concordance_2 = None, None
        tag_distance = None
        
        if tag2 is not None:
            x_tag2 = get_snp_column_memeff(X, inv_row_idx, np.where(snp_ids == tag2.variant_id)[0][0])
            x_both = np.column_stack([x_tag1, x_tag2])
            reg_r2_2, concordance_2 = regression_checks(y_inv, x_both)
            
            if tag1.pos and tag2.pos:
                tag_distance = abs(tag2.pos - tag1.pos)
        
        # Classify deployment status
        status = classify_tag_status(
            tag1.r2_overall, tag1.r2_min_cluster,
            args.r2_global_thresh, args.r2_cluster_thresh
        )
        
        print(f"  [TAG1] {tag1.variant_id}: r²={tag1.r2_overall:.3f}, min_cluster={tag1.r2_min_cluster:.3f}, concordance={concordance_1:.3f} → {status}")
        
        if tag2:
            print(f"  [TAG2] {tag2.variant_id}: r²={tag2.r2_overall:.3f}, distance={tag_distance} bp (same-chrom cis-LD)")
        else:
            print(f"  [TAG2] None found (no same-chrom candidate ≥{args.min_tag_distance} bp away)")
        
        # Store selected tags
        selected_tags.append({
            "inversion": inv_name,
            "tag1_variant_id": tag1.variant_id,
            "tag1_chrom": tag1.chrom,
            "tag1_pos": tag1.pos,
            "tag1_ref": tag1.ref,
            "tag1_alt": tag1.alt,
            "tag1_r2_overall": tag1.r2_overall,
            "tag1_r2_min_cluster": tag1.r2_min_cluster,
            "tag1_r2_mean_cluster": tag1.r2_mean_cluster,
            "tag1_maf": tag1.maf,
            "tag1_reg_r2": reg_r2_1,
            "tag1_concordance": concordance_1,
            "tag2_variant_id": tag2.variant_id if tag2 else None,
            "tag2_chrom": tag2.chrom if tag2 else None,
            "tag2_pos": tag2.pos if tag2 else None,
            "tag2_ref": tag2.ref if tag2 else None,
            "tag2_alt": tag2.alt if tag2 else None,
            "tag2_r2_overall": tag2.r2_overall if tag2 else None,
            "tag2_r2_min_cluster": tag2.r2_min_cluster if tag2 else None,
            "tag2_reg_r2": reg_r2_2,
            "tag2_concordance": concordance_2,
            "tag_distance_bp": tag_distance,
            "deployment_status": status,
        })
    
    # ---------------------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[SAVE] Writing output files...")
    
    # Full candidates table
    cand_df = pd.DataFrame(all_candidates)
    cand_df.to_csv(out_candidates, index=False)
    print(f"  → {out_candidates}")
    
    # Selected tags table
    sel_df = pd.DataFrame(selected_tags)
    sel_df.to_csv(out_selected, index=False)
    print(f"  → {out_selected}")
    
    # Manuscript-ready table
    manuscript_rows = []
    for row in selected_tags:
        manuscript_rows.append({
            "Inversion": row["inversion"],
            "Tag_SNP": row["tag1_variant_id"],
            "Chromosome": row["tag1_chrom"],
            "Position": row["tag1_pos"],
            "Ref": row["tag1_ref"],
            "Alt": row["tag1_alt"],
            "r2_Overall": round(row["tag1_r2_overall"], 3),
            "r2_Min_Cluster": round(row["tag1_r2_min_cluster"], 3),
            "Concordance": round(row["tag1_concordance"], 3),
            "Redundant_Tag": row["tag2_variant_id"],
            "Tag_Distance_bp": row["tag_distance_bp"],
            "Status": row["deployment_status"],
        })
    
    manu_df = pd.DataFrame(manuscript_rows)
    manu_df.to_csv(out_manuscript, index=False)
    print(f"  → {out_manuscript}")
    
    # ---------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[SUMMARY]")
    n_global = sum(1 for r in selected_tags if r["deployment_status"] == "GLOBAL")
    n_restricted = sum(1 for r in selected_tags if r["deployment_status"] == "RESTRICTED")
    n_moderate = sum(1 for r in selected_tags if r["deployment_status"] == "MODERATE")
    n_weak = sum(1 for r in selected_tags if r["deployment_status"] == "WEAK")
    n_with_tag2 = sum(1 for r in selected_tags if r["tag2_variant_id"] is not None)
    
    print(f"  Total inversions processed: {len(selected_tags)}")
    print(f"  GLOBAL (r²≥{args.r2_global_thresh}, min_cluster≥{args.r2_cluster_thresh}): {n_global}")
    print(f"  RESTRICTED (high overall, low cluster stability): {n_restricted}")
    print(f"  MODERATE (r²≥0.5): {n_moderate}")
    print(f"  WEAK (r²<0.5): {n_weak}")
    print(f"  With redundant tag2 (same-chrom, ≥{args.min_tag_distance}bp): {n_with_tag2}")
    print()
    print("[DONE]")


if __name__ == "__main__":
    main()