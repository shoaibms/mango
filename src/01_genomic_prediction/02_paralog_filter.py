# -*- coding: utf-8 -*-
"""
01b_het_qc.py

Post-processing QC Step:
Filter SNPs with excessive heterozygosity (>0.65) from geno_core.npz.
This removes likely paralogs (duplicated genome regions mapped to one spot)
which cause false positives in GWAS and noise in GS.

Run this AFTER 01_build_core_matrices.py finishes.

This version:
- Reports original and final SNP counts.
- Keeps a backup of the original geno_core.npz as geno_core_old.npz.
"""

import os
import shutil
import numpy as np
import config_idea1 as cfg  # Use config to get paths automatically

# CONFIG
IN_FILE = os.path.join(cfg.CORE_DATA_DIR, "geno_core.npz")
TEMP_FILE = os.path.join(cfg.CORE_DATA_DIR, "geno_core_clean_temp.npz")
BACKUP_FILE = os.path.join(cfg.CORE_DATA_DIR, "geno_core_old.npz")

# Threshold: If >65% of samples are Heterozygous, it's likely a paralog.
MAX_HETEROZYGOSITY = 0.65


def main():
    if not os.path.exists(IN_FILE):
        raise FileNotFoundError(f"Input file not found: {IN_FILE}\nDid Step 01 finish?")

    print(f"[INFO] Loading core matrix: {IN_FILE}")
    data = np.load(IN_FILE, allow_pickle=True)
    G = data["G"]           # (n_samples, n_snps)
    sids = data["sample_ids"]
    vids = data["variant_ids"]

    n_samples, n_snps = G.shape
    print(f"[INFO] Original Shape: {n_samples} samples x {n_snps} SNPs")

    # Calculate Heterozygosity Rate
    # Genotypes are 0, 1, 2. Imputed values are floats.
    # We define "Heterozygous" as dosages roughly between 0.75 and 1.25.
    # This avoids counting "weakly imputed" values (e.g. 0.1 or 1.9) as Het.
    print("[INFO] Calculating heterozygosity rates...")
    is_het = (G > 0.75) & (G < 1.25)
    het_counts = np.sum(is_het, axis=0)
    het_rates = het_counts / float(n_samples)

    # Filter SNPs exceeding heterozygosity threshold
    keep_mask = het_rates <= MAX_HETEROZYGOSITY
    n_dropped = int(np.sum(~keep_mask))
    n_keep = int(np.sum(keep_mask))

    print(f"[INFO] SNPs before filter: {n_snps}")
    print(f"[INFO] SNPs after filter:  {n_keep}")
    print(f"[INFO] SNPs dropped:       {n_dropped}")

    if n_dropped > 0:
        print(
            f"[QC ALERT] Detected {n_dropped} SNPs with Het Rate > "
            f"{MAX_HETEROZYGOSITY:.2f}. Likely paralogs. Removing them..."
        )

        G_clean = G[:, keep_mask]
        vids_clean = vids[keep_mask]

        # Safety Check
        if G_clean.shape[1] < 10000:
            print("[WARN] Dropped a large number of SNPs. Verify this is intended.")

        # Backup original file once
        if not os.path.exists(BACKUP_FILE):
            print(f"[INFO] Creating backup of original geno_core.npz -> {BACKUP_FILE}")
            shutil.copy2(IN_FILE, BACKUP_FILE)
        else:
            print(f"[INFO] Backup already exists: {BACKUP_FILE} (not overwritten)")

        print(f"[INFO] Saving cleaned data to temp file: {TEMP_FILE}")
        np.savez_compressed(
            TEMP_FILE,
            G=G_clean.astype(np.float32),
            sample_ids=sids,
            variant_ids=vids_clean,
        )

        # Replace original file
        print(f"[INFO] Overwriting original geno_core.npz with cleaned version...")
        shutil.move(TEMP_FILE, IN_FILE)
        print(f"[SUCCESS] geno_core.npz sanitized. Final Shape: {G_clean.shape}")

    else:
        print("[QC PASS] No excess-heterozygosity SNPs detected. File remains unchanged.")


if __name__ == "__main__":
    main()
