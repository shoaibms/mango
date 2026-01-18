#!/usr/bin/env python
r"""
01_prepare_datasets.py

Prepare ML-ready genotype + phenotype + structure metadata for Mango GS (Idea 2).

Inputs (by default):
  - Geno core (from Idea 1):
      C:\Users\ms\Desktop\mango\output\idea_1\core_data\geno_core.npz
  - Pheno core (from Idea 1):
      C:\Users\ms\Desktop\mango\output\idea_1\core_data\pheno_core.csv

Outputs (by default) to:
  C:\Users\ms\Desktop\mango\output\idea_2\core_ml\

  - X_full.npy                : genotype matrix [n_samples x n_snps], float32
  - samples.csv               : sample_id in the exact row order of X_full
  - snp_ids.csv               : SNP IDs / marker IDs (if available)
  - y_traits.csv              : numeric trait matrix, index=sample_id
  - pcs.csv                   : PCs per sample (PC1..PCk), index=sample_id
  - sample_metadata_ml.csv    : PCs + k-means cluster labels per sample

This script is intentionally robust to slight differences in geno_core.npz:
it will try to auto-detect the genotype matrix, sample IDs, and SNP IDs,
and give informative errors if assumptions fail.
"""

import argparse
import os
from typing import Tuple, Optional, List

import numpy as np

try:
    import pandas as pd
except ImportError as e:
    raise SystemExit(
        "pandas is required. Install it with:\n\n  pip install pandas\n"
    ) from e

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
except ImportError as e:
    raise SystemExit(
        "scikit-learn is required. Install it with:\n\n  pip install scikit-learn\n"
    ) from e


# =========================
# DEFAULT PATHS / PARAMS
# =========================

DEFAULT_GENO_CORE = r"C:\Users\ms\Desktop\mango\output\idea_1\core_data\geno_core.npz"
DEFAULT_PHENO_CORE = r"C:\Users\ms\Desktop\mango\output\idea_1\core_data\pheno_core.csv"
DEFAULT_OUTDIR = r"C:\Users\ms\Desktop\mango\output\idea_2\core_ml"

DEFAULT_N_PCS = 10          # number of genomic PCs to compute
DEFAULT_N_CLUSTERS = 3      # for k-means clustering on PCs
RANDOM_STATE = 42


# =========================
# UTILS
# =========================

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _to_str_array(arr: np.ndarray) -> np.ndarray:
    """Convert a numpy array to a 1D array of Python strings."""
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array for IDs, got shape {arr.shape}")
    # handle bytes or other dtypes
    return arr.astype(str)


# =========================
# LOADING CORE OBJECTS
# =========================

def load_geno_core(geno_core_path: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load genotype core from an npz file and detect:
      - genotype matrix (2D array, samples x SNPs)
      - sample IDs (1D array, length = n_samples) if available
      - SNP IDs (1D array, length = n_snps) if available

    Returns:
      X         : np.ndarray, shape (n_samples, n_snps)
      sample_ids : np.ndarray of str, shape (n_samples,)
      snp_ids    : np.ndarray of str, shape (n_snps,) or None
    """
    if not os.path.isfile(geno_core_path):
        raise FileNotFoundError(f"geno_core file not found: {geno_core_path}")

    print(f"[LOAD] geno_core npz -> {geno_core_path}")
    npz = np.load(geno_core_path, allow_pickle=True)
    keys = list(npz.keys())
    print(f"[INFO] geno_core keys: {keys}")

    # 1) Detect genotype matrix
    geno_key_candidates = ["G", "X", "geno", "genotypes"]
    geno_key = None
    for k in geno_key_candidates:
        if k in npz:
            arr = npz[k]
            if isinstance(arr, np.ndarray) and arr.ndim == 2:
                geno_key = k
                break

    if geno_key is None:
        # Fallback: first 2D array we see
        for k in keys:
            arr = npz[k]
            if isinstance(arr, np.ndarray) and arr.ndim == 2:
                geno_key = k
                break

    if geno_key is None:
        raise RuntimeError(
            "Could not find a 2D array in geno_core.npz to use as genotype matrix. "
            "Please inspect the file manually (np.load) and update this script."
        )

    G = np.asarray(npz[geno_key])
    if G.ndim != 2:
        raise RuntimeError(f"Detected genotype array '{geno_key}' is not 2D: shape={G.shape}")

    n_samples, n_snps = G.shape
    print(f"[INFO] Genotype matrix '{geno_key}' shape: {G.shape} (samples x SNPs)")

    # 2) Detect sample IDs
    sample_ids = None
    sample_key_candidates = ["sample_ids", "samples", "ids", "line_ids"]
    for k in sample_key_candidates:
        if k in npz:
            arr = npz[k]
            arr = np.asarray(arr)
            if arr.ndim == 1 and arr.shape[0] == n_samples:
                sample_ids = _to_str_array(arr)
                print(f"[INFO] Using '{k}' as sample_ids.")
                break

    if sample_ids is None:
        print(
            "[WARN] No explicit sample ID array found in geno_core.npz. "
            f"Creating generic IDs: sample_0 .. sample_{n_samples-1}"
        )
        sample_ids = np.array([f"sample_{i}" for i in range(n_samples)], dtype=str)

    # 3) Detect SNP IDs (optional)
    snp_ids = None
    snp_key_candidates = ["variant_ids", "snp_ids", "marker_ids", "variants", "snpid", "snp_id"]
    for k in snp_key_candidates:
        if k in npz:
            arr = npz[k]
            arr = np.asarray(arr)
            if arr.ndim == 1 and arr.shape[0] == n_snps:
                snp_ids = _to_str_array(arr)
                print(f"[INFO] Using '{k}' as snp_ids.")
                break

    if snp_ids is None:
        print("[WARN] No SNP ID array detected in geno_core.npz. snp_ids will be None.")

    return G, sample_ids, snp_ids


def load_pheno_core(pheno_core_path: str) -> pd.DataFrame:
    """
    Load phenotype core CSV with sample IDs as index.

    Heuristics:
      - If 'sample_id' column exists -> use as index.
      - Else if 'ID' column exists -> use as index.
      - Else if first col is 'Unnamed: 0' -> read with index_col=0.
      - Otherwise assume first column is sample ID and set as index.

    Returns:
      pheno_df : DataFrame with index=sample_id and numeric trait columns.
    """
    if not os.path.isfile(pheno_core_path):
        raise FileNotFoundError(f"pheno_core file not found: {pheno_core_path}")

    print(f"[LOAD] pheno_core csv -> {pheno_core_path}")
    df = pd.read_csv(pheno_core_path)

    # Heuristic to identify sample ID column
    cols_lower = {c.lower(): c for c in df.columns}
    if "sample_id" in cols_lower:
        id_col = cols_lower["sample_id"]
        df = df.set_index(id_col)
        print(f"[INFO] Using column '{id_col}' as sample_id index.")
    elif "id" in cols_lower:
        id_col = cols_lower["id"]
        df = df.set_index(id_col)
        print(f"[INFO] Using column '{id_col}' as sample_id index.")
    elif df.columns[0].lower().startswith("unnamed"):
        # typical "index saved as first unnamed column" pattern
        df = pd.read_csv(pheno_core_path, index_col=0)
        print("[INFO] Using first column as index (Unnamed: 0 pattern).")
    else:
        # assume first column is sample ID
        id_col = df.columns[0]
        df = df.set_index(id_col)
        print(f"[WARN] Assuming first column '{id_col}' is sample_id index.")

    # Ensure index is string
    df.index = df.index.astype(str)

    # Keep only numeric trait columns by default
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise RuntimeError(
            "No numeric columns found in pheno_core.csv after indexing by sample_id.\n"
            "Check that pheno_core.csv contains trait columns."
        )

    pheno_df = df[numeric_cols].copy()
    print(f"[INFO] Phenotype table shape: {pheno_df.shape} (samples x traits)")
    print(f"[INFO] Trait columns: {list(pheno_df.columns)}")
    return pheno_df


def align_geno_pheno(
    G: np.ndarray,
    geno_sample_ids: np.ndarray,
    pheno_df: pd.DataFrame,
) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    """
    Align genotype matrix and phenotype table on common sample IDs.

    Returns:
      G_aligned      : genotype matrix with rows in aligned sample order
      pheno_aligned  : phenotype DataFrame with same order
      common_samples : list of sample IDs in aligned order
    """
    geno_ids = [str(s) for s in geno_sample_ids]
    pheno_ids = list(pheno_df.index)

    common = [sid for sid in geno_ids if sid in pheno_df.index]

    if len(common) == 0:
        raise RuntimeError(
            "No overlap between genotype sample IDs and phenotype sample IDs.\n"
            f"  n_geno_samples = {len(geno_ids)}\n"
            f"  n_pheno_samples = {len(pheno_ids)}\n"
            "Check sample ID formats in geno_core.npz and pheno_core.csv."
        )

    print(
        f"[INFO] Overlapping samples (geno vs pheno): {len(common)} / {len(geno_ids)}"
    )

    # map sample ID -> row index in G
    idx_map = {sid: i for i, sid in enumerate(geno_ids)}
    geno_indices = [idx_map[sid] for sid in common]

    G_aligned = G[geno_indices, :]
    pheno_aligned = pheno_df.loc[common].copy()

    return G_aligned, pheno_aligned, common


# =========================
# PCA + CLUSTERING
# =========================

def compute_pcs(
    G: np.ndarray,
    n_components: int = DEFAULT_N_PCS,
) -> Tuple[np.ndarray, PCA]:
    """
    Compute genomic PCs from genotype matrix.

    Steps:
      - Standardize SNPs (mean=0, std=1) across samples.
      - Run PCA on standardized matrix.
      - Return scores (PCs per sample) and fitted PCA object.
    """
    n_samples, n_snps = G.shape
    n_components_eff = min(n_components, n_samples, n_snps)
    if n_components_eff < 1:
        raise ValueError(
            f"Cannot compute PCs: n_components_eff={n_components_eff}, "
            f"n_samples={n_samples}, n_snps={n_snps}"
        )

    print(
        f"[INFO] Computing genomic PCs: requested={n_components}, effective={n_components_eff}"
    )

    scaler = StandardScaler(with_mean=True, with_std=True)
    G_scaled = scaler.fit_transform(G)

    pca = PCA(n_components=n_components_eff, random_state=RANDOM_STATE)
    pcs = pca.fit_transform(G_scaled)

    explained = pca.explained_variance_ratio_
    print("[INFO] PC explained variance ratio (first few):")
    for i, ev in enumerate(explained[:5], start=1):
        print(f"  PC{i}: {ev:.4f}")

    return pcs, pca


def cluster_on_pcs(
    pcs: np.ndarray,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """
    Run k-means clustering on PC space.

    Returns:
      labels : np.ndarray of shape (n_samples,), cluster labels 0..k-1
    """
    if n_clusters < 2:
        raise ValueError("n_clusters must be >= 2 for k-means clustering.")

    print(f"[INFO] Running KMeans clustering on PCs (k={n_clusters})")
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(pcs)
    return labels


# =========================
# MAIN
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare ML-ready genotype+phenotype datasets for Mango GS Idea 2."
    )
    parser.add_argument(
        "--geno-core",
        type=str,
        default=DEFAULT_GENO_CORE,
        help=f"Path to geno_core.npz (default: {DEFAULT_GENO_CORE})",
    )
    parser.add_argument(
        "--pheno-core",
        type=str,
        default=DEFAULT_PHENO_CORE,
        help=f"Path to pheno_core.csv (default: {DEFAULT_PHENO_CORE})",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=DEFAULT_OUTDIR,
        help=f"Output directory for ML-ready core data (default: {DEFAULT_OUTDIR})",
    )
    parser.add_argument(
        "--n-pcs",
        type=int,
        default=DEFAULT_N_PCS,
        help=f"Number of genomic PCs to compute (default: {DEFAULT_N_PCS})",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=DEFAULT_N_CLUSTERS,
        help=f"Number of KMeans clusters on PCs (default: {DEFAULT_N_CLUSTERS})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 72)
    print("Mango GS - Idea 2: Prepare ML-ready datasets")
    print("=" * 72)
    print(f"[INFO] Geno core:   {args.geno_core}")
    print(f"[INFO] Pheno core:  {args.pheno_core}")
    print(f"[INFO] Output dir:  {args.outdir}")
    print(f"[INFO] N_PCS:       {args.n_pcs}")
    print(f"[INFO] N_CLUSTERS:  {args.n_clusters}")
    print("")

    safe_mkdir(args.outdir)

    # 1) Load cores
    G, geno_sample_ids, snp_ids = load_geno_core(args.geno_core)
    pheno_df = load_pheno_core(args.pheno_core)

    # 2) Align samples
    G_aligned, pheno_aligned, common_ids = align_geno_pheno(G, geno_sample_ids, pheno_df)
    n_samples, n_snps = G_aligned.shape
    print(
        f"[INFO] After alignment: G shape = {G_aligned.shape} (samples x SNPs), "
        f"pheno shape = {pheno_aligned.shape}"
    )

    # 3) Compute PCs for structure
    pcs, pca = compute_pcs(G_aligned, n_components=args.n_pcs)

    # 4) Cluster in PC space
    cluster_labels = cluster_on_pcs(pcs, n_clusters=args.n_clusters)

    # 5) Save outputs
    # 5.1 Genotype matrix
    X_path = os.path.join(args.outdir, "X_full.npy")
    np.save(X_path, G_aligned.astype(np.float32))
    print(f"[SAVE] Genotype matrix -> {X_path}")

    # 5.2 Sample IDs in order
    samples_path = os.path.join(args.outdir, "samples.csv")
    pd.DataFrame({"sample_id": common_ids}).to_csv(samples_path, index=False)
    print(f"[SAVE] Sample IDs -> {samples_path}")

    # 5.3 SNP IDs (if available)
    if snp_ids is not None:
        snp_path = os.path.join(args.outdir, "snp_ids.csv")
        pd.DataFrame({"snp_id": snp_ids}).to_csv(snp_path, index=False)
        print(f"[SAVE] SNP IDs -> {snp_path}")
    else:
        print("[INFO] No snp_ids saved (none detected in geno_core).")

    # 5.4 Phenotypes
    y_path = os.path.join(args.outdir, "y_traits.csv")
    pheno_aligned.to_csv(y_path, index_label="sample_id")
    print(f"[SAVE] Phenotypes -> {y_path}")

    # 5.5 PCs per sample
    pc_cols = [f"PC{i+1}" for i in range(pcs.shape[1])]
    pcs_df = pd.DataFrame(pcs, index=common_ids, columns=pc_cols)
    pcs_path = os.path.join(args.outdir, "pcs.csv")
    pcs_df.to_csv(pcs_path, index_label="sample_id")
    print(f"[SAVE] PCs -> {pcs_path}")

    # 5.6 Sample metadata (PCs + cluster labels)
    meta_df = pcs_df.copy()
    meta_df["cluster_kmeans"] = cluster_labels
    meta_path = os.path.join(args.outdir, "sample_metadata_ml.csv")
    meta_df.to_csv(meta_path, index_label="sample_id")
    print(f"[SAVE] Sample metadata (PCs + cluster) -> {meta_path}")

    print("")
    print("[OK] Idea 2 core ML datasets prepared successfully.")


if __name__ == "__main__":
    main()
