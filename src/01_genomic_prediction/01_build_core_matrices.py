# -*- coding: utf-8 -*-
r"""
	02_paralog_filter

Build "frozen" core genotype + phenotype + metadata matrices for Mango GS (Idea 1).

SINGLE-PASS QC-BASED PROPORTIONAL SAMPLING:

Key insight: Reservoir sampling already tracks n_pass_qc (total QC-passing SNPs).
We can achieve exact proportional sampling without a separate counting pass.

Strategy:
1. Single pass per VCF: each worker grabs up to TOTAL_TARGET SNPs
2. Workers return n_pass_qc (total QC-passing count seen)
3. After both complete: calculate proportional targets from n_pass_qc
4. Post-hoc trim arrays to match proportional targets

"""

from __future__ import annotations

import gzip
import io
import os
import pickle
import random
import subprocess
import shutil
import concurrent.futures
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

try:
    import pandas as pd
except ImportError as e:
    raise SystemExit("pandas is required. Install with: pip install pandas") from e

import config_idea1 as cfg


# =========================
# Pre-built genotype lookup table
# =========================

def _build_gt_lookup() -> Dict[str, float]:
    """Build lookup table mapping genotype strings to dosages."""
    lookup = {}
    nan = float('nan')
    
    for sep in ('/', '|'):
        for a1 in ('0', '1'):
            for a2 in ('0', '1'):
                gt = f"{a1}{sep}{a2}"
                lookup[gt] = float(int(a1) + int(a2))
    
    missing_patterns = ['.', './.', '.|.', './0', '0/.', './1', '1/.',
                        '.|0', '0|.', '.|1', '1|.']
    for pat in missing_patterns:
        lookup[pat] = nan
    
    for a1 in ('0', '1', '2', '3'):
        for a2 in ('2', '3'):
            for sep in ('/', '|'):
                lookup[f"{a1}{sep}{a2}"] = nan
                lookup[f"{a2}{sep}{a1}"] = nan
    
    return lookup

GT_LOOKUP = _build_gt_lookup()


# =========================
# Parallel gzip helpers
# =========================

def _check_pigz_available() -> bool:
    return shutil.which('pigz') is not None


def _open_vcf_fast(vcf_path: str, use_pigz: bool = True) -> io.TextIOWrapper:
    if use_pigz and vcf_path.endswith('.gz') and _check_pigz_available():
        proc = subprocess.Popen(
            ['pigz', '-dc', vcf_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=16 * 1024 * 1024
        )
        return io.TextIOWrapper(proc.stdout, encoding='utf-8', errors='replace')
    else:
        return gzip.open(vcf_path, 'rt', encoding='utf-8', errors='replace')


# =========================
# VCF helpers
# =========================

def parse_vcf_samples(vcf_path: str) -> List[str]:
    """Return list of sample IDs from VCF header."""
    with gzip.open(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#CHROM"):
                parts = line.rstrip("\n").split("\t")
                return parts[9:]
    raise RuntimeError(f"#CHROM header line not found in {vcf_path}")


# =========================
# SNP Sampler (single-pass, returns QC count)
# =========================

def sample_snps_from_vcf(
    vcf_path: str,
    n_target: int,
    sample_ids_reference: List[str],
    min_maf: float,
    max_miss: float,
    random_state: int = 42,
) -> Tuple[np.ndarray, List[str], int, int]:
    """
    Single-pass SNP sampling with reservoir sampling.
    
    Returns:
        G_block: genotype matrix (n_samples, n_kept)
        var_ids: variant IDs
        n_scanned: total variants scanned
        n_pass_qc: total QC-passing SNPs (for proportional calculation)
    """
    use_pigz = _check_pigz_available()
    basename = os.path.basename(vcf_path)
    print(f"[SAMPLE] {basename}: reservoir_size={n_target:,}")

    rng = random.Random(random_state)
    vcf_samples = parse_vcf_samples(vcf_path)
    sample_to_col = {s: i for i, s in enumerate(vcf_samples)}
    col_indices = [sample_to_col[sid] for sid in sample_ids_reference]
    
    n_samples = len(sample_ids_reference)
    col_offsets = [9 + idx for idx in col_indices]

    reservoir_geno = [None] * n_target
    reservoir_ids = [''] * n_target
    reservoir_count = 0

    geno = np.full(n_samples, np.nan, dtype=np.float64)
    gt_lookup_get = GT_LOOKUP.get
    nan = float('nan')

    n_scanned = 0
    n_pass_qc = 0
    truncated = False

    with _open_vcf_fast(vcf_path, use_pigz=use_pigz) as fh:
        try:
            for line in fh:
                if line.startswith("#CHROM"):
                    break

            for line in fh:
                if not line or line[0] == '#':
                    continue

                n_scanned += 1
                cols = line.split('\t')
                if len(cols) < 10:
                    continue

                # Biallelic SNP check
                ref = cols[3]
                alt = cols[4]
                if ',' in alt or len(ref) != 1 or len(alt) != 1:
                    continue

                fmt = cols[8]
                gt_index = 0 if fmt.startswith('GT:') or fmt == 'GT' else -1
                if gt_index < 0:
                    try:
                        gt_index = fmt.split(':').index('GT')
                    except ValueError:
                        continue

                geno.fill(nan)
                n_cols = len(cols)

                for i, col_offset in enumerate(col_offsets):
                    if col_offset >= n_cols:
                        continue
                    sample_field = cols[col_offset]
                    
                    if gt_index == 0:
                        colon_pos = sample_field.find(':')
                        gt = sample_field[:colon_pos] if colon_pos != -1 else sample_field
                    else:
                        fields = sample_field.split(':')
                        gt = fields[gt_index] if gt_index < len(fields) else '.'

                    dosage = gt_lookup_get(gt)
                    if dosage is not None:
                        geno[i] = dosage

                # QC: missingness
                called_mask = ~np.isnan(geno)
                n_called = np.sum(called_mask)
                if n_called == 0:
                    continue
                miss_frac = 1.0 - n_called / n_samples
                if miss_frac > max_miss:
                    continue

                # QC: MAF
                mean_dosage = np.mean(geno[called_mask])
                p = mean_dosage / 2.0
                maf = p if p <= 0.5 else 1.0 - p
                if maf < min_maf:
                    continue

                # Impute missing
                nan_mask = np.isnan(geno)
                if np.any(nan_mask):
                    geno[nan_mask] = mean_dosage

                n_pass_qc += 1

                # Variant ID
                vid = cols[2]
                var_id = f"{cols[0]}:{cols[1]}" if vid == '.' or not vid else vid

                # Reservoir sampling
                if reservoir_count < n_target:
                    reservoir_geno[reservoir_count] = geno.astype(np.float32)
                    reservoir_ids[reservoir_count] = var_id
                    reservoir_count += 1
                else:
                    j = rng.randint(0, n_pass_qc - 1)
                    if j < n_target:
                        reservoir_geno[j] = geno.astype(np.float32)
                        reservoir_ids[j] = var_id

                if n_pass_qc % 100000 == 0:
                    print(f"  [{basename}] {n_scanned:,} scanned, {n_pass_qc:,} QC-passed")

        except (EOFError, OSError) as e:
            truncated = True
            print(f"[WARN] {basename}: stream error ({e}) after {n_scanned:,} scanned, {n_pass_qc:,} QC-passed.")
            print(f"[WARN] Using {reservoir_count:,} variants collected so far.")

    if reservoir_count == 0:
        raise RuntimeError(f"No SNPs passed QC in {vcf_path}")

    reservoir_geno = reservoir_geno[:reservoir_count]
    reservoir_ids = reservoir_ids[:reservoir_count]
    G_block = np.column_stack(reservoir_geno)

    print(f"[SAMPLE] {basename}: done. scanned={n_scanned:,}, QC-passed={n_pass_qc:,}, kept={G_block.shape[1]:,}")
    return G_block, reservoir_ids, n_scanned, n_pass_qc


def process_vcf_wrapper(args):
    return sample_snps_from_vcf(*args)


def targets_from_qc_counts(qc_counts: List[int], total_target: int) -> List[int]:
    """Allocate targets proportional to QC-passing counts."""
    qc_arr = np.array(qc_counts, dtype=float)
    total_qc = qc_arr.sum()
    if total_qc <= 0:
        raise ValueError("Total QC-passing SNP count is zero")

    raw = qc_arr / total_qc * total_target
    base = np.floor(raw).astype(int)
    diff = total_target - base.sum()

    if diff != 0:
        frac = raw - base
        order = np.argsort(-frac) if diff > 0 else np.argsort(frac)
        for idx in order[:abs(diff)]:
            base[idx] += 1 if diff > 0 else -1

    return base.tolist()


# =========================
# Phenotype helpers
# =========================

def load_pheno_and_meta(
    xlsx_path: str,
    sheet_name: str,
    sample_id_col: str,
    trait_col_map: Dict[str, str],
    sample_ids_reference: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load phenotype and metadata."""
    print(f"[INFO] Loading phenotypes from {xlsx_path}")
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    df[sample_id_col] = df[sample_id_col].astype(str)
    df = df.set_index(sample_id_col)

    overlap = sorted(set(sample_ids_reference) & set(df.index))
    print(f"[INFO] Sample overlap: {len(overlap)} / {len(sample_ids_reference)}")

    trait_cols = []
    for short, source in trait_col_map.items():
        if source not in df.columns:
            raise KeyError(f"Trait column '{source}' not found")
        trait_cols.append(source)

    pheno_core = df[trait_cols].copy()
    pheno_core.columns = list(trait_col_map.keys())
    pheno_core = pheno_core.reindex(sample_ids_reference)

    meta_core = df.drop(columns=trait_cols, errors="ignore").reindex(sample_ids_reference)
    return pheno_core, meta_core


# =========================
# Checkpointing
# =========================

def get_checkpoint_path(out_dir: str, vcf_path: str) -> str:
    """Get checkpoint file path for a VCF."""
    basename = os.path.basename(vcf_path).replace('.vcf.gz', '').replace('.vcf', '')
    return os.path.join(out_dir, f".checkpoint_{basename}.pkl")


def save_checkpoint(out_dir: str, vcf_path: str, G_block: np.ndarray, 
                    var_ids: List[str], n_scan: int, n_pass: int) -> None:
    """Save checkpoint for a completed VCF."""
    ckpt_path = get_checkpoint_path(out_dir, vcf_path)
    data = {
        'G_block': G_block,
        'var_ids': var_ids,
        'n_scan': n_scan,
        'n_pass': n_pass,
        'vcf_path': vcf_path,
    }
    with open(ckpt_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"[CHECKPOINT] Saved: {ckpt_path}")


def load_checkpoint(out_dir: str, vcf_path: str):
    """Load checkpoint if exists, return None otherwise."""
    ckpt_path = get_checkpoint_path(out_dir, vcf_path)
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path, 'rb') as f:
                data = pickle.load(f)
            print(f"[CHECKPOINT] Loaded: {ckpt_path}")
            return data['G_block'], data['var_ids'], data['n_scan'], data['n_pass']
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint {ckpt_path}: {e}")
            return None
    return None


def clear_checkpoints(out_dir: str) -> None:
    """Remove all checkpoint files after successful completion."""
    for f in os.listdir(out_dir):
        if f.startswith('.checkpoint_') and f.endswith('.pkl'):
            os.remove(os.path.join(out_dir, f))
            print(f"[CHECKPOINT] Removed: {f}")


# =========================
# Main
# =========================

def main() -> None:
    cfg.ensure_output_dirs()
    out_dir = cfg.CORE_DATA_DIR
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("Mango GS — Idea 1: Build Core Matrices (v8)")
    print("Single-Pass QC-Based Proportional Sampling")
    print("=" * 70)
    print(f"[INFO] Output: {out_dir}")
    print(f"[INFO] pigz: {'available' if _check_pigz_available() else 'not found'}")
    
    vcf_paths = [str(p) for p in cfg.VCF_PATHS]
    print(f"[INFO] VCF files: {len(vcf_paths)}")
    for v in vcf_paths:
        print(f"  - {v}")
    print()

    # Get sample IDs
    sample_ids = parse_vcf_samples(vcf_paths[0])
    n_samples = len(sample_ids)
    print(f"[INFO] Samples: {n_samples}")

    # Verify other VCFs have same samples
    for vcf in vcf_paths[1:]:
        if set(parse_vcf_samples(vcf)) != set(sample_ids):
            raise RuntimeError(f"Sample mismatch in {vcf}")

    total_target = int(cfg.TOTAL_SNPS_TARGET)
    if total_target <= 0:
        raise ValueError("TOTAL_SNPS_TARGET must be positive")

    # =========================================================================
    # SINGLE PASS: Sample up to total_target from each VCF (with checkpointing)
    # =========================================================================
    print()
    print("=" * 70)
    print("Single-pass sampling (with checkpointing)")
    print("=" * 70)

    n_files = len(vcf_paths)
    raw_blocks: List[np.ndarray] = [None] * n_files
    raw_ids_list: List[List[str]] = [None] * n_files
    qc_counts: List[int] = [0] * n_files
    qc_scanned: List[int] = [0] * n_files

    # Check for existing checkpoints
    vcfs_to_process = []
    for i, vcf in enumerate(vcf_paths):
        ckpt = load_checkpoint(out_dir, vcf)
        if ckpt is not None:
            G_block, var_ids, n_scan, n_pass = ckpt
            raw_blocks[i] = G_block
            raw_ids_list[i] = var_ids
            qc_scanned[i] = n_scan
            qc_counts[i] = n_pass
            print(f"  → Using checkpoint: {os.path.basename(vcf)} ({n_pass:,} QC-passed)")
        else:
            vcfs_to_process.append((i, vcf))

    # Process remaining VCFs
    if vcfs_to_process:
        print(f"\n[INFO] Processing {len(vcfs_to_process)} VCF(s)...")
        
        tasks = []
        task_indices = []
        for i, vcf in vcfs_to_process:
            tasks.append((
                vcf,
                total_target,
                sample_ids,
                cfg.MIN_MAF,
                cfg.MAX_MISS,
                cfg.RANDOM_STATE + i + 1,
            ))
            task_indices.append(i)

        num_workers = min(len(tasks), os.cpu_count() or 1)
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {
                executor.submit(process_vcf_wrapper, t): idx
                for t, idx in zip(tasks, task_indices)
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                i = future_to_idx[future]
                vcf = vcf_paths[i]
                try:
                    G_block, var_ids, n_scan, n_pass = future.result()
                    raw_blocks[i] = G_block
                    raw_ids_list[i] = var_ids
                    qc_counts[i] = n_pass
                    qc_scanned[i] = n_scan
                    # Save checkpoint immediately
                    save_checkpoint(out_dir, vcf, G_block, var_ids, n_scan, n_pass)
                except Exception as e:
                    print(f"[ERROR] Failed processing {vcf}: {e}")
                    raise
    else:
        print("[INFO] All VCFs loaded from checkpoints!")

    # =========================================================================
    # POST-HOC: Calculate proportions and trim
    # =========================================================================
    print()
    print("=" * 70)
    print("Post-hoc proportional trimming")
    print("=" * 70)

    total_qc = sum(qc_counts)
    if total_qc <= 0:
        raise RuntimeError("No SNPs passed QC across all VCFs")

    # Safety clamp: cannot sample more than what exists
    if total_target > total_qc:
        print(f"[WARN] Target ({total_target:,}) > QC-passing ({total_qc:,}). Using {total_qc:,}.")
        total_target = total_qc

    # Calculate proportional targets
    per_file_targets = targets_from_qc_counts(qc_counts, total_target)

    print()
    print("[INFO] QC-based proportional targets:")
    for vcf, qc, target in zip(vcf_paths, qc_counts, per_file_targets):
        pct = 100.0 * qc / total_qc
        print(f"  - {os.path.basename(vcf)}: {qc:,} QC-passing ({pct:.1f}%) → {target:,} target")
    print()

    # Save sampling report
    report_path = os.path.join(out_dir, "vcf_sampling_report.txt")
    with open(report_path, 'w') as f:
        f.write("VCF Proportional Sampling Report (Single-Pass v8)\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write("=" * 60 + "\n\n")
        f.write("Method: SINGLE-PASS QC-BASED PROPORTIONAL SAMPLING\n")
        f.write("(One scan per VCF + post-hoc proportional trimming)\n\n")
        f.write(f"QC Filters: MAF >= {cfg.MIN_MAF}, Missingness <= {cfg.MAX_MISS}\n")
        f.write(f"Total target: {total_target:,}\n\n")
        f.write("Per-file breakdown:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'File':<25} {'Scanned':>12} {'QC-Pass':>12} {'Target':>10}\n")
        f.write("-" * 60 + "\n")
        for vcf, scan, qc, target in zip(vcf_paths, qc_scanned, qc_counts, per_file_targets):
            f.write(f"{os.path.basename(vcf):<25} {scan:>12,} {qc:>12,} {target:>10,}\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'TOTAL':<25} {sum(qc_scanned):>12,} {total_qc:>12,} {total_target:>10,}\n")
    print(f"[SAVED] {report_path}")

    # Trim each block to proportional target
    trimmed_blocks: List[np.ndarray] = []
    trimmed_ids: List[List[str]] = []
    per_file_stats = []
    rng = np.random.default_rng(cfg.RANDOM_STATE + 999)

    for i, vcf in enumerate(vcf_paths):
        basename = os.path.basename(vcf)
        G_raw = raw_blocks[i]
        ids_raw = raw_ids_list[i]
        target = per_file_targets[i]

        n_raw = G_raw.shape[1]
        if n_raw <= target:
            # Already have <= target, keep all
            G_trim = G_raw
            ids_trim = ids_raw
            kept = n_raw
        else:
            # Random subsample to match target
            idx_sel = rng.choice(n_raw, size=target, replace=False)
            idx_sel = np.sort(idx_sel)
            G_trim = G_raw[:, idx_sel]
            ids_trim = [ids_raw[j] for j in idx_sel]
            kept = target

        trimmed_blocks.append(G_trim)
        trimmed_ids.append(ids_trim)
        per_file_stats.append((basename, kept, qc_scanned[i], qc_counts[i]))
        print(f"  {basename}: {n_raw:,} → {kept:,} SNPs (trimmed to target)")

    G_core = np.concatenate(trimmed_blocks, axis=1)
    all_var_ids = [v for sublist in trimmed_ids for v in sublist]
    n_snps_core = G_core.shape[1]

    print()
    print(f"[INFO] Final matrix: {G_core.shape[0]} samples × {n_snps_core} SNPs")

    # Load phenotypes
    pheno_core, meta_core = load_pheno_and_meta(
        str(cfg.PHENO_XLSX_PATH), str(cfg.PHENO_SHEET),
        str(cfg.SAMPLE_ID_COL), cfg.TRAIT_COL_MAP, sample_ids
    )

    # Save outputs
    geno_path = os.path.join(out_dir, "geno_core.npz")
    pheno_path = os.path.join(out_dir, "pheno_core.csv")
    meta_path = os.path.join(out_dir, "meta_core.csv")
    summary_path = os.path.join(out_dir, "core_data_summary.txt")

    print(f"[SAVE] {geno_path}")
    np.savez_compressed(geno_path, G=G_core.astype(np.float32),
                        sample_ids=np.array(sample_ids, dtype=object),
                        variant_ids=np.array(all_var_ids, dtype=object))

    print(f"[SAVE] {pheno_path}")
    pheno_core.to_csv(pheno_path)

    print(f"[SAVE] {meta_path}")
    meta_core.to_csv(meta_path)

    print(f"[SAVE] {summary_path}")
    with open(summary_path, "w") as sf:
        sf.write("Core Data Summary (v8 - Single-Pass Proportional Sampling)\n")
        sf.write("=" * 60 + "\n\n")
        sf.write(f"Samples: {n_samples}\n")
        sf.write(f"SNPs: {n_snps_core}\n\n")
        sf.write("Per-VCF:\n")
        for basename, kept, scanned, qc in per_file_stats:
            sf.write(f"  - {basename}: scanned={scanned:,}, QC-pass={qc:,}, kept={kept:,}\n")
        sf.write(f"\nSampling report: {report_path}\n")
        sf.write("\nTraits:\n")
        for col in pheno_core.columns:
            n_val = int(pheno_core[col].notna().sum())
            sf.write(f"  - {col}: {n_val} / {n_samples}\n")

    print()
    print("=" * 70)
    print("[DONE] Core matrices built with single-pass proportional sampling.")
    print("=" * 70)

    # Clear checkpoints after successful completion
    clear_checkpoints(out_dir)



if __name__ == "__main__":
    main()
