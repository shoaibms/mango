# -*- coding: utf-8 -*-
"""
21_breeding_values.py

Compare out-of-fold (OOF) "breeding values" (i.e., CV-predicted GEBVs) across methods.

Scientific comparison: All 4 methods (idea1, idea2, ridge490, BINN)
Breeder consensus: BINN + idea2 (PC-corrected ridge) as independent baseline

Design decisions:
  - ridge490 excluded from baseline check (BINN-derived, not independent)
  - idea2 (PC-corrected) used as the baseline check (cleaner messaging than argmax)
  - For traits where idea2 is uninformative (ρ < threshold), BINN-only + validation flag

Inputs (CSV with standard schema):
  - Idea1 ridge SNP:      output/idea_1/cv_baseline/cv_baseline_oof_predictions.csv
  - Idea2 ridge STRUCT:   output/idea_2/results_baseline/baseline_ridge_oof_predictions.csv
  - BINN:                 output/idea_3/binn_training/binn_oof_predictions.csv
  - ridge490:             output/idea_3/binn_decomposition/ridge490_oof_predictions.csv

Outputs:
  Scientific (all 4 methods):
    - merged_oof.csv
    - method_vs_pheno.csv
    - pairwise_method_spearman.csv
    - topk_overlap.csv
    - topk_overlap_by_direction.csv
    - rank_shift_analysis.csv
  
  Breeder-facing (BINN + idea2 baseline):
    - breeder_recommendations.csv
    - breeder_consensus_summary.csv
"""

from __future__ import annotations

import argparse
import itertools
import os
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# TRAIT DIRECTION CONFIGURATION
# =============================================================================
TRAIT_DIRECTIONS: Dict[str, str] = {
    "TSS": "higher",    # Total soluble solids (sugar) - higher is sweeter
    "FF":  "higher",    # Fruit firmness - higher is better for shelf-life
    "AFW": "both",      # Average fruit weight - market prefers specific size grades
    "FBC": "both",      # Fruit blush colour (hue angle) - lower = redder, depends on market
    "TC":  "both",      # Trunk circumference - lower if dwarf preferred, higher if vigour wanted
}

TRAIT_DIRECTION_NOTES: Dict[str, str] = {
    "TSS": "Higher = sweeter fruit",
    "FF":  "Higher = firmer, better shelf-life",
    "AFW": "Target-dependent: market prefers specific size grades",
    "FBC": "Lower = redder blush (if red preferred); Higher = more yellow",
    "TC":  "Lower = compact/dwarf habit; Higher = vigorous establishment",
}

# =============================================================================
# DEFAULT THRESHOLD
# =============================================================================
DEFAULT_BASELINE_THRESHOLD = 0.30


def _get_repo_root() -> str:
    """Determine the repository root directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.basename(os.path.dirname(script_dir))
    grandparent = os.path.basename(script_dir)
    
    if parent == "code" or grandparent.startswith("idea_"):
        return os.path.abspath(os.path.join(script_dir, "..", ".."))
    
    if os.path.isdir(os.path.join(os.getcwd(), "output")):
        return os.getcwd()
    
    for levels in [1, 2, 3]:
        candidate = os.path.abspath(os.path.join(script_dir, *[".."] * levels))
        if os.path.isdir(os.path.join(candidate, "output")):
            return candidate
    
    return os.getcwd()


REPO_ROOT = _get_repo_root()


def _resolve_path(p: str) -> str:
    """Resolve path relative to REPO_ROOT."""
    p = p.replace("\\", "/")
    if os.path.isabs(p):
        return os.path.normpath(p)
    return os.path.normpath(os.path.join(REPO_ROOT, p))


# Default paths
DEFAULT_IDEA1_OOF    = "output/idea_1/cv_baseline/cv_baseline_oof_predictions.csv"
DEFAULT_IDEA2_OOF    = "output/idea_2/results_baseline/baseline_ridge_oof_predictions.csv"
DEFAULT_BINN_OOF     = "output/idea_3/binn_training/binn_oof_predictions.csv"
DEFAULT_RIDGE490_OOF = "output/idea_3/binn_decomposition/ridge490_oof_predictions.csv"
DEFAULT_OUTDIR       = "output/breeding_value_concordance"

REQ_COLS = [
    "source", "feature_set", "model", "scheme", "scenario", "trait", "fold", "sample_id",
    "y_true", "y_pred", "y_true_resid", "y_pred_resid", "y_fixed_pred"
]


def _mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _read_oof(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing OOF file:\n  {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"OOF file is missing required columns {missing}:\n  {path}")
    df["sample_id"] = df["sample_id"].astype(str)
    df["trait"] = df["trait"].astype(str)
    df["scenario"] = df["scenario"].astype(str)
    df["scheme"] = df["scheme"].astype(str)
    df["model"] = df["model"].astype(str)
    df["source"] = df["source"].astype(str)
    df["feature_set"] = df["feature_set"].astype(str)
    return df


def _mask_finite(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.isfinite(a) & np.isfinite(b)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    m = _mask_finite(x, y)
    if m.sum() < 3:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    m = _mask_finite(x, y)
    if m.sum() < 3:
        return float("nan")
    sx = pd.Series(x[m])
    sy = pd.Series(y[m])
    return float(sx.corr(sy, method="spearman"))


def _get_selection_count(df: pd.DataFrame, k_frac: float) -> int:
    """Get the actual number of selections that would be made (matches selection logic)."""
    n_valid = df.dropna(subset=["y_pred"]).shape[0]
    if n_valid == 0:
        return 0
    return max(1, int(np.floor(k_frac * n_valid)))


def _topk_ids(df: pd.DataFrame, k_frac: float) -> List[str]:
    """Select top k% by highest y_pred (higher is better)."""
    df2 = df.dropna(subset=["y_pred"]).copy()
    n = df2.shape[0]
    if n == 0:
        return []
    k = max(1, int(np.floor(k_frac * n)))
    df2 = df2.sort_values("y_pred", ascending=False)
    return df2["sample_id"].head(k).astype(str).tolist()


def _bottomk_ids(df: pd.DataFrame, k_frac: float) -> List[str]:
    """Select bottom k% by lowest y_pred (lower is better)."""
    df2 = df.dropna(subset=["y_pred"]).copy()
    n = df2.shape[0]
    if n == 0:
        return []
    k = max(1, int(np.floor(k_frac * n)))
    df2 = df2.sort_values("y_pred", ascending=True)
    return df2["sample_id"].head(k).astype(str).tolist()


def _selectk_ids(df: pd.DataFrame, k_frac: float, direction: str) -> List[str]:
    """Select k% based on direction ('top' or 'bottom')."""
    if direction == "top":
        return _topk_ids(df, k_frac)
    elif direction == "bottom":
        return _bottomk_ids(df, k_frac)
    else:
        raise ValueError(f"Unknown direction: {direction}")


def _filter_block(
    df: pd.DataFrame,
    source: str,
    scenario: Optional[str] = None,
    scheme: Optional[str] = None,
) -> pd.DataFrame:
    sub = df[df["source"] == source].copy()
    if scenario is not None:
        sub = sub[sub["scenario"] == scenario].copy()
    if scheme is not None:
        sub = sub[sub["scheme"] == scheme].copy()
    return sub


def _get_trait_directions_to_evaluate(trait: str) -> List[str]:
    """Get list of directions to evaluate for a trait."""
    direction = TRAIT_DIRECTIONS.get(trait, "both")
    if direction == "higher":
        return ["top"]
    elif direction == "lower":
        return ["bottom"]
    else:
        return ["top", "bottom"]


def _compute_accuracies(blocks: List[Tuple[str, pd.DataFrame]], traits: set) -> Dict[str, Dict[str, float]]:
    """Compute Spearman accuracy for each method-trait combination."""
    accuracies = {}
    for name, b in blocks:
        accuracies[name] = {}
        for trait in traits:
            bt = b[b["trait"] == trait]
            x = bt["y_true"].to_numpy(float)
            y = bt["y_pred"].to_numpy(float)
            accuracies[name][trait] = _spearman(x, y)
    return accuracies


def _check_idea2_informative(
    trait: str, 
    accuracies: Dict[str, Dict[str, float]],
    threshold: float
) -> Tuple[float, bool]:
    """
    Check if idea2 (PC-corrected ridge) is informative for this trait.
    
    Simplified baseline rule: always use idea2 as the baseline check.
    This is cleaner messaging than argmax(idea1, idea2) and costs essentially nothing.
    
    Returns:
        (idea2_accuracy, is_informative)
    """
    idea2_acc = accuracies.get("idea2", {}).get(trait, 0.0)
    
    # Handle NaN
    if np.isnan(idea2_acc):
        idea2_acc = 0.0
    
    is_informative = idea2_acc >= threshold
    
    return idea2_acc, is_informative


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare OOF breeding values across methods"
    )
    ap.add_argument("--idea1_oof", default=DEFAULT_IDEA1_OOF)
    ap.add_argument("--idea2_oof", default=DEFAULT_IDEA2_OOF)
    ap.add_argument("--binn_oof", default=DEFAULT_BINN_OOF)
    ap.add_argument("--ridge490_oof", default=DEFAULT_RIDGE490_OOF)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--idea1_scenario", default="no_pc")
    ap.add_argument("--idea2_scenario", default="pc_corrected")
    ap.add_argument("--idea2_scheme", default="cv_random_k5")
    ap.add_argument("--topk_fracs", default="0.10,0.20")
    ap.add_argument("--baseline_threshold", type=float, default=DEFAULT_BASELINE_THRESHOLD,
                    help="Spearman threshold below which idea2 is considered uninformative (default: 0.30)")
    args = ap.parse_args()

    baseline_threshold = args.baseline_threshold

    print(f"[INFO] Detected REPO_ROOT: {REPO_ROOT}")

    outdir = _resolve_path(args.outdir)
    _mkdir(outdir)

    # Load all OOF files
    idea1_path = _resolve_path(args.idea1_oof)
    idea2_path = _resolve_path(args.idea2_oof)
    binn_path = _resolve_path(args.binn_oof)
    ridge490_path = _resolve_path(args.ridge490_oof)

    print(f"[INFO] Loading Idea1 OOF: {idea1_path}")
    print(f"[INFO] Loading Idea2 OOF: {idea2_path}")
    print(f"[INFO] Loading BINN OOF: {binn_path}")
    print(f"[INFO] Loading Ridge490 OOF: {ridge490_path}")

    df1 = _read_oof(idea1_path)
    df2 = _read_oof(idea2_path)
    df3 = _read_oof(binn_path)
    df4 = _read_oof(ridge490_path)
    df4["source"] = "ridge490"

    merged = pd.concat([df1, df2, df3, df4], ignore_index=True)
    merged_path = os.path.join(outdir, "merged_oof.csv")
    merged.to_csv(merged_path, index=False)

    # Filter blocks
    b1 = _filter_block(merged, "idea1", scenario=args.idea1_scenario)
    b2 = _filter_block(merged, "idea2", scenario=args.idea2_scenario, scheme=args.idea2_scheme)
    b3 = _filter_block(merged, "binn")
    b4 = _filter_block(merged, "ridge490")

    blocks = [("idea1", b1), ("idea2", b2), ("binn", b3), ("ridge490", b4)]
    method_names = [m for m, _ in blocks]

    print("\n[INFO] Block sizes after filtering:")
    for name, b in blocks:
        print(f"  {name}: {len(b)} rows, {b['trait'].nunique()} traits")

    # Common traits
    traits_common = set.intersection(*[set(b["trait"].unique()) for _, b in blocks])
    print(f"\n[INFO] Common traits: {sorted(traits_common)}")

    # Compute accuracies for all methods
    accuracies = _compute_accuracies(blocks, traits_common)

    # Print trait direction and baseline check configuration
    print("\n[INFO] Breeder consensus configuration")
    print(f"       Baseline: idea2 (PC-corrected ridge)")
    print(f"       Informativeness threshold: ρ ≥ {baseline_threshold}")
    print(f"       Rule: If idea2 informative → consensus = BINN ∩ idea2")
    print(f"             If idea2 uninformative → BINN-only + validation flag")
    print("-" * 80)
    for trait in sorted(traits_common):
        idea2_acc, is_informative = _check_idea2_informative(trait, accuracies, baseline_threshold)
        binn_acc = accuracies["binn"][trait]
        idea1_acc = accuracies["idea1"][trait]
        ridge490_acc = accuracies["ridge490"][trait]
        direction = TRAIT_DIRECTIONS.get(trait, "both")
        status = "[PASS] informative" if is_informative else "[FAIL] UNINFORMATIVE → BINN-only + validation"
        print(f"  {trait}: direction={direction}")
        print(f"       BINN ρ={binn_acc:.3f} | idea1 ρ={idea1_acc:.3f} | idea2 ρ={idea2_acc:.3f} | ridge490 ρ={ridge490_acc:.3f}")
        print(f"       Baseline check: idea2 [{status}]")

    topk_fracs = [float(x.strip()) for x in args.topk_fracs.split(",") if x.strip()]

    # =========================================================================
    # SCIENTIFIC OUTPUTS (All 4 methods)
    # =========================================================================
    
    # 1) Method vs phenotype
    rows_mv = []
    for name, b in blocks:
        for trait in sorted(b["trait"].unique()):
            bt = b[b["trait"] == trait]
            x = bt["y_true"].to_numpy(float)
            y = bt["y_pred"].to_numpy(float)
            rows_mv.append({
                "method": name,
                "trait": trait,
                "n": int(np.isfinite(x).sum()),
                "pearson_ytrue_ypred": _pearson(x, y),
                "spearman_ytrue_ypred": _spearman(x, y),
            })
    mv = pd.DataFrame(rows_mv)
    mv_path = os.path.join(outdir, "method_vs_pheno.csv")
    mv.to_csv(mv_path, index=False)

    # 2) Pairwise Spearman
    rows_pair = []
    for trait in sorted(traits_common):
        per = {}
        for name, b in blocks:
            bt = b[b["trait"] == trait][["sample_id", "y_pred"]].dropna()
            per[name] = bt.set_index("sample_id")["y_pred"]

        common = set.intersection(*[set(per[m].index) for m in method_names])
        common_list = list(common)
        if len(common_list) < 5:
            continue

        for (mA, mB) in itertools.combinations(method_names, 2):
            a = per[mA].loc[common_list].to_numpy(float)
            b_arr = per[mB].loc[common_list].to_numpy(float)
            rows_pair.append({
                "trait": trait,
                "method_A": mA,
                "method_B": mB,
                "n_common": len(common_list),
                "spearman_pred_pred": _spearman(a, b_arr),
                "pearson_pred_pred": _pearson(a, b_arr),
            })
    pair = pd.DataFrame(rows_pair)
    pair_path = os.path.join(outdir, "pairwise_method_spearman.csv")
    pair.to_csv(pair_path, index=False)

    # 3) Top-k overlap (all 4 methods)
    rows_top = []
    for trait in sorted(traits_common):
        tops = {}
        for name, b in blocks:
            bt = b[b["trait"] == trait][["sample_id", "y_pred"]]
            tops[name] = {k: _topk_ids(bt, k) for k in topk_fracs}

        for k in topk_fracs:
            for (mA, mB) in itertools.combinations(method_names, 2):
                A = set(tops[mA][k])
                B = set(tops[mB][k])
                if not A or not B:
                    continue
                overlap = len(A & B)
                denom = min(len(A), len(B))
                jacc = overlap / max(1, len(A | B))
                rows_top.append({
                    "trait": trait,
                    "k_frac": k,
                    "method_A": mA,
                    "method_B": mB,
                    "k_A": len(A),
                    "k_B": len(B),
                    "overlap": overlap,
                    "overlap_frac_minK": overlap / max(1, denom),
                    "jaccard": jacc,
                })
    top = pd.DataFrame(rows_top)
    top_path = os.path.join(outdir, "topk_overlap.csv")
    top.to_csv(top_path, index=False)

    # 3b) Direction-aware overlap (all 4 methods)
    rows_top_dir = []
    for trait in sorted(traits_common):
        directions_to_eval = _get_trait_directions_to_evaluate(trait)
        for direction in directions_to_eval:
            selections = {}
            for name, b in blocks:
                bt = b[b["trait"] == trait][["sample_id", "y_pred"]]
                selections[name] = {k: _selectk_ids(bt, k, direction) for k in topk_fracs}

            for k in topk_fracs:
                for (mA, mB) in itertools.combinations(method_names, 2):
                    A = set(selections[mA][k])
                    B = set(selections[mB][k])
                    if not A or not B:
                        continue
                    overlap = len(A & B)
                    denom = min(len(A), len(B))
                    jacc = overlap / max(1, len(A | B))
                    rows_top_dir.append({
                        "trait": trait,
                        "direction": direction,
                        "selection_goal": "higher_is_better" if direction == "top" else "lower_is_better",
                        "k_frac": k,
                        "method_A": mA,
                        "method_B": mB,
                        "k_A": len(A),
                        "k_B": len(B),
                        "overlap": overlap,
                        "overlap_frac_minK": overlap / max(1, denom),
                        "jaccard": jacc,
                    })
    top_dir = pd.DataFrame(rows_top_dir)
    top_dir_path = os.path.join(outdir, "topk_overlap_by_direction.csv")
    top_dir.to_csv(top_dir_path, index=False)

    # 4) Rank shift analysis
    rows_rankshift = []
    for trait in sorted(traits_common):
        rank_dict = {}
        for name, b in blocks:
            bt = b[b["trait"] == trait][["sample_id", "y_pred"]].dropna()
            bt = bt.sort_values("y_pred", ascending=False).reset_index(drop=True)
            bt["rank"] = bt.index + 1
            rank_dict[name] = bt.set_index("sample_id")["rank"]

        common_ids = set.intersection(*[set(rank_dict[m].index) for m in method_names])
        if len(common_ids) < 5:
            continue

        n_samples = len(common_ids)
        for sid in common_ids:
            ranks = {m: int(rank_dict[m].loc[sid]) for m in method_names}
            pct_ranks = {m: 100.0 * ranks[m] / n_samples for m in method_names}
            rank_spread = max(ranks.values()) - min(ranks.values())
            pct_spread = max(pct_ranks.values()) - min(pct_ranks.values())
            best_method = min(ranks, key=ranks.get)
            worst_method = max(ranks, key=ranks.get)
            
            row_data = {
                "trait": trait,
                "sample_id": sid,
                "rank_spread": rank_spread,
                "pct_spread": round(pct_spread, 2),
                "best_method": best_method,
                "worst_method": worst_method,
                "n_samples": n_samples,
            }
            for m in method_names:
                row_data[f"rank_{m}"] = ranks[m]
                row_data[f"pct_rank_{m}"] = round(pct_ranks[m], 2)
            rows_rankshift.append(row_data)

    rankshift = pd.DataFrame(rows_rankshift)
    rankshift = rankshift.sort_values(["trait", "rank_spread"], ascending=[True, False])
    rankshift_path = os.path.join(outdir, "rank_shift_analysis.csv")
    rankshift.to_csv(rankshift_path, index=False)

    # =========================================================================
    # BREEDER OUTPUTS (BINN + idea2 baseline)
    # =========================================================================
    
    blocks_dict = dict(blocks)
    binn_block = blocks_dict["binn"]
    idea2_block = blocks_dict["idea2"]

    rows_breeder = []
    consensus_summary = []

    for trait in sorted(traits_common):
        # Check if idea2 is informative for this trait
        idea2_acc, is_informative = _check_idea2_informative(trait, accuracies, baseline_threshold)
        binn_acc = accuracies["binn"][trait]

        directions_to_eval = _get_trait_directions_to_evaluate(trait)
        
        for direction in directions_to_eval:
            k = 0.10  # Primary selection threshold
            
            # Get BINN selections
            binn_trait = binn_block[binn_block["trait"] == trait][["sample_id", "y_pred"]]
            binn_selections = set(_selectk_ids(binn_trait, k, direction))
            n_expected = _get_selection_count(binn_trait, k)
            
            # Get idea2 selections
            idea2_trait = idea2_block[idea2_block["trait"] == trait][["sample_id", "y_pred"]]
            idea2_selections = set(_selectk_ids(idea2_trait, k, direction))
            
            # Determine consensus based on informativeness
            if is_informative:
                consensus_ids = binn_selections & idea2_selections
                consensus_type = "binn_and_idea2"
                binn_only_ids = binn_selections - consensus_ids
                # Consensus rate is meaningful
                consensus_rate = round(len(consensus_ids) / max(1, n_expected), 3)
            else:
                consensus_ids = binn_selections
                consensus_type = "binn_only_idea2_uninformative"
                binn_only_ids = set()
                # Consensus rate is NA when BINN-only (would be misleading to show 1.0)
                consensus_rate = "N/A"
            
            # Summary row
            consensus_summary.append({
                "trait": trait,
                "direction": direction,
                "selection_goal": "higher_is_better" if direction == "top" else "lower_is_better",
                "binn_accuracy": round(binn_acc, 3),
                "idea2_accuracy": round(idea2_acc, 3),
                "idea2_informative": is_informative,
                "consensus_type": consensus_type,
                "n_binn_selections": len(binn_selections),
                "n_idea2_selections": len(idea2_selections) if is_informative else "N/A",
                "n_consensus": len(consensus_ids),
                "n_expected_10pct": n_expected,
                "consensus_rate": consensus_rate,
                "validation_recommended": not is_informative,
            })
            
            # Individual recommendations - CONSENSUS
            for sid in sorted(consensus_ids):
                binn_bt = binn_trait.dropna().copy()
                binn_bt = binn_bt.sort_values("y_pred", ascending=(direction == "bottom")).reset_index(drop=True)
                binn_bt["rank"] = binn_bt.index + 1
                binn_row = binn_bt[binn_bt["sample_id"] == sid]
                
                idea2_bt = idea2_trait.dropna().copy()
                idea2_bt = idea2_bt.sort_values("y_pred", ascending=(direction == "bottom")).reset_index(drop=True)
                idea2_bt["rank"] = idea2_bt.index + 1
                idea2_row = idea2_bt[idea2_bt["sample_id"] == sid]
                
                binn_rank = int(binn_row["rank"].values[0]) if not binn_row.empty else np.nan
                binn_pred = binn_row["y_pred"].values[0] if not binn_row.empty else np.nan
                idea2_rank = int(idea2_row["rank"].values[0]) if not idea2_row.empty else np.nan
                idea2_pred = idea2_row["y_pred"].values[0] if not idea2_row.empty else np.nan
                
                if is_informative and not np.isnan(binn_rank) and not np.isnan(idea2_rank):
                    rank_spread = abs(binn_rank - idea2_rank)
                    mean_rank = (binn_rank + idea2_rank) / 2
                else:
                    rank_spread = "N/A"
                    mean_rank = binn_rank
                
                # Confidence
                if not is_informative:
                    confidence = "BINN_ONLY_VALIDATE"
                elif isinstance(rank_spread, int) and rank_spread <= 5:
                    confidence = "HIGH"
                elif isinstance(rank_spread, int) and rank_spread <= 10:
                    confidence = "MEDIUM"
                elif isinstance(rank_spread, int):
                    confidence = "LOW"
                else:
                    confidence = "BINN_ONLY_VALIDATE"
                
                rows_breeder.append({
                    "trait": trait,
                    "direction": direction,
                    "selection_goal": "higher_is_better" if direction == "top" else "lower_is_better",
                    "trait_note": TRAIT_DIRECTION_NOTES.get(trait, ""),
                    "sample_id": sid,
                    "consensus_type": consensus_type,
                    "binn_rank": binn_rank,
                    "binn_pred": round(binn_pred, 4) if not np.isnan(binn_pred) else np.nan,
                    "idea2_rank": idea2_rank if is_informative else "N/A",
                    "idea2_pred": round(idea2_pred, 4) if is_informative and not np.isnan(idea2_pred) else "N/A",
                    "mean_rank": round(mean_rank, 1) if not np.isnan(mean_rank) else np.nan,
                    "rank_spread": rank_spread,
                    "confidence": confidence,
                    "validation_recommended": not is_informative,
                    "binn_accuracy": round(binn_acc, 3),
                    "idea2_accuracy": round(idea2_acc, 3),
                })
            
            # BINN-only (when idea2 is informative but didn't agree)
            if is_informative:
                for sid in sorted(binn_only_ids):
                    binn_bt = binn_trait.dropna().copy()
                    binn_bt = binn_bt.sort_values("y_pred", ascending=(direction == "bottom")).reset_index(drop=True)
                    binn_bt["rank"] = binn_bt.index + 1
                    binn_row = binn_bt[binn_bt["sample_id"] == sid]
                    
                    idea2_bt = idea2_trait.dropna().copy()
                    idea2_bt = idea2_bt.sort_values("y_pred", ascending=(direction == "bottom")).reset_index(drop=True)
                    idea2_bt["rank"] = idea2_bt.index + 1
                    idea2_row = idea2_bt[idea2_bt["sample_id"] == sid]
                    
                    binn_rank = int(binn_row["rank"].values[0]) if not binn_row.empty else np.nan
                    binn_pred = binn_row["y_pred"].values[0] if not binn_row.empty else np.nan
                    idea2_rank = int(idea2_row["rank"].values[0]) if not idea2_row.empty else np.nan
                    idea2_pred = idea2_row["y_pred"].values[0] if not idea2_row.empty else np.nan
                    
                    rank_spread = abs(binn_rank - idea2_rank) if not np.isnan(idea2_rank) else "N/A"
                    
                    rows_breeder.append({
                        "trait": trait,
                        "direction": direction,
                        "selection_goal": "higher_is_better" if direction == "top" else "lower_is_better",
                        "trait_note": TRAIT_DIRECTION_NOTES.get(trait, ""),
                        "sample_id": sid,
                        "consensus_type": "binn_only_no_idea2_agreement",
                        "binn_rank": binn_rank,
                        "binn_pred": round(binn_pred, 4) if not np.isnan(binn_pred) else np.nan,
                        "idea2_rank": idea2_rank,
                        "idea2_pred": round(idea2_pred, 4) if not np.isnan(idea2_pred) else np.nan,
                        "mean_rank": binn_rank,
                        "rank_spread": rank_spread,
                        "confidence": "BINN_PREFERRED",
                        "validation_recommended": True,
                        "binn_accuracy": round(binn_acc, 3),
                        "idea2_accuracy": round(idea2_acc, 3),
                    })

    breeder_recs = pd.DataFrame(rows_breeder)
    breeder_recs = breeder_recs.sort_values(
        ["trait", "direction", "consensus_type", "mean_rank"],
        ascending=[True, True, True, True]
    )
    breeder_path = os.path.join(outdir, "breeder_recommendations.csv")
    breeder_recs.to_csv(breeder_path, index=False)

    cons_summary_df = pd.DataFrame(consensus_summary)
    cons_summary_path = os.path.join(outdir, "breeder_consensus_summary.csv")
    cons_summary_df.to_csv(cons_summary_path, index=False)

    # =========================================================================
    # PRINT SUMMARY
    # =========================================================================
    print("\n[DONE]")
    print(f"[SAVE] {merged_path}")
    print(f"[SAVE] {mv_path}")
    print(f"[SAVE] {pair_path}")
    print(f"[SAVE] {top_path}")
    print(f"[SAVE] {top_dir_path}")
    print(f"[SAVE] {rankshift_path}")
    print(f"[SAVE] {breeder_path}")
    print(f"[SAVE] {cons_summary_path}")

    print("\n" + "=" * 80)
    print("SCIENTIFIC SUMMARY: 4-METHOD COMPARISON")
    print("=" * 80)

    print("\n[TABLE 1] Prediction Accuracy (Spearman ρ: y_true vs y_pred)")
    print("-" * 60)
    pivot_spearman = mv.pivot(index="trait", columns="method", values="spearman_ytrue_ypred")
    pivot_spearman = pivot_spearman[["idea1", "idea2", "binn", "ridge490"]]
    print(pivot_spearman.round(3).to_string())

    if not pair.empty:
        print("\n[TABLE 2] Pairwise GEBV Rank Concordance (Spearman ρ)")
        print("-" * 60)
        pair_summary = pair.pivot_table(
            index="trait",
            columns=["method_A", "method_B"],
            values="spearman_pred_pred"
        ).round(3)
        print(pair_summary.to_string())

    print("\n" + "=" * 80)
    print("BREEDER SUMMARY: BINN + idea2 (PC-corrected ridge)")
    print("=" * 80)
    print(f"\nBaseline: idea2 (PC-corrected ridge)")
    print(f"Informativeness threshold: ρ ≥ {baseline_threshold}")
    print("\nRule:")
    print("  If idea2 ρ ≥ 0.30 → consensus = BINN ∩ idea2")
    print("  If idea2 ρ < 0.30 → BINN-only + VALIDATION REQUIRED")

    print("\n[TABLE 3] Consensus Configuration by Trait")
    print("-" * 80)
    print(cons_summary_df.to_string(index=False))

    print("\n[TABLE 4] Breeder Recommendations Summary")
    print("-" * 80)
    for trait in sorted(breeder_recs["trait"].unique()):
        trait_recs = breeder_recs[breeder_recs["trait"] == trait]
        print(f"\n{'='*60}")
        print(f"TRAIT: {trait}")
        print(f"Note: {TRAIT_DIRECTION_NOTES.get(trait, 'N/A')}")
        
        idea2_acc, is_informative = _check_idea2_informative(trait, accuracies, baseline_threshold)
        binn_acc = accuracies["binn"][trait]
        status = "[PASS] Informative" if is_informative else "[FAIL] UNINFORMATIVE"
        print(f"BINN ρ = {binn_acc:.3f} | idea2 ρ = {idea2_acc:.3f} [{status}]")
        print(f"{'='*60}")
        
        for direction in trait_recs["direction"].unique():
            dir_recs = trait_recs[trait_recs["direction"] == direction]
            goal = "HIGHER is better" if direction == "top" else "LOWER is better"
            print(f"\n  Selection direction: {goal}")
            
            # Consensus selections
            consensus = dir_recs[dir_recs["consensus_type"].isin(["binn_and_idea2", "binn_only_idea2_uninformative"])]
            n_cons = len(consensus)
            
            if is_informative:
                print(f"  CONSENSUS (BINN ∩ idea2): {n_cons}")
            else:
                print(f"  BINN-ONLY selections: {n_cons}")
                print(f"  [WARN] VALIDATION REQUIRED (idea2 uninformative for this trait)")
            
            if n_cons > 0:
                for _, row in consensus.head(5).iterrows():
                    if is_informative:
                        spread_str = f"spread={row['rank_spread']}"
                    else:
                        spread_str = "N/A"
                    print(f"    - {row['sample_id']}: rank={row['mean_rank']}, {spread_str}, conf={row['confidence']}")
            
            # BINN-preferred
            if is_informative:
                binn_pref = dir_recs[dir_recs["consensus_type"] == "binn_only_no_idea2_agreement"]
                if len(binn_pref) > 0:
                    print(f"  BINN-preferred (needs validation): {len(binn_pref)}")

    print("\n" + "=" * 80)
    print("KEY INTERPRETATION")
    print("=" * 80)
    
    informative_traits = []
    uninformative_traits = []
    for trait in sorted(traits_common):
        _, is_inf = _check_idea2_informative(trait, accuracies, baseline_threshold)
        if is_inf:
            informative_traits.append(trait)
        else:
            uninformative_traits.append(trait)
    
    print(f"\nTraits with informative baseline (idea2 ρ ≥ {baseline_threshold}): {informative_traits}")
    print(f"  → Consensus = BINN ∩ idea2")
    print(f"  → Safe to advance consensus selections")
    
    if uninformative_traits:
        print(f"\nTraits with UNINFORMATIVE baseline (idea2 ρ < {baseline_threshold}): {uninformative_traits}")
        print(f"  → Traditional ridge is noise for these traits")
        print(f"  → BINN-only selections provided")
        print(f"  → VALIDATION REQUIRED: re-phenotyping or confirmatory trials before advancement")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()