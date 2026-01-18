import os
import numpy as np

try:
    import pandas as pd
except ImportError as e:
    raise SystemExit("pandas is required. Install it with: pip install pandas") from e

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
except ImportError as e:
    raise SystemExit(
        "scikit-learn is required. Install it with: pip install scikit-learn"
    ) from e


# =========================
# CONFIG
# =========================

# Core data from Idea 1
GENO_CORE_PATH = r"C:\Users\ms\Desktop\mango\output\idea_1\core_data\geno_core.npz"
PHENO_CORE_PATH = r"C:\Users\ms\Desktop\mango\output\idea_1\core_data\pheno_core.csv"

# Raw phenotype / metadata Excel (New Phytologist supplementary)
DATASET_S1_XLSX = r"C:\Users\ms\Desktop\mango\data\main_data\nph20252-sup-0001-datasetss1-s3.xlsx"
DATASET_S1_SHEET = "Dataset S1"
DATASET_S1_ID_COL = "ID"

# Output root for Idea 3
OUT_ROOT = r"C:\Users\ms\Desktop\mango\output\idea_3"
OUT_CORE_DIR = os.path.join(OUT_ROOT, "core_data")

# Canonical trait columns (what we conceptually care about)
# NOTE: these are not required to exist in pheno_core; trait columns are detected dynamically.
TRAIT_COLS_CANONICAL = [
    "BC",               # fruit blush colour
    "FF",               # fruit firmness
    "Square Root[FW]",  # square-root transformed fruit weight
    "Log10[TSS]",       # log10-transformed total soluble solids
    "TC",               # trunk circumference
]

# PCA / residual settings
N_PCS = 20  # number of global PCs to compute for structure
RANDOM_STATE = 42  # only used for reproducibility in some sklearn ops


# =========================
# UTILITIES
# =========================

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_geno_core(path: str):
    """
    Load geno_core.npz from Idea 1 and return a dict with at least:
      - X:          genotype matrix (n_samples x n_snps)
      - sample_ids: array-like of length n_samples
      - snp_ids:    array-like of length n_snps (if available)
      - chrom:      array-like of length n_snps (if available)
      - pos:        array-like of length n_snps (if available)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"geno_core file not found:\n  {path}")

    print(f"[INFO] Loading geno_core from:\n  {path}")
    npz = np.load(path, allow_pickle=True)
    keys = list(npz.files)
    print(f"[INFO] geno_core keys: {keys}")

    # Recognise alternative key names: 'G' and 'variant_ids'
    geno_key_candidates = ["X", "geno", "geno_matrix", "G"]
    sample_key_candidates = ["sample_ids", "samples", "lines"]
    snp_key_candidates = ["snp_ids", "markers", "variant_ids"]
    chrom_key_candidates = ["chrom", "chr"]
    pos_key_candidates = ["pos", "position", "bp"]

    def _pick(candidates, label):
        for c in candidates:
            if c in npz.files:
                return c
        raise KeyError(
            f"Could not find a key for {label}. "
            f"Tried {candidates}, available keys: {keys}"
        )

    geno_key = _pick(geno_key_candidates, "genotype matrix")
    sample_key = _pick(sample_key_candidates, "sample IDs")
    snp_key = None
    try:
        snp_key = _pick(snp_key_candidates, "SNP IDs")
    except KeyError:
        print("[WARN] No SNP ID key found; proceeding without explicit SNP IDs.")

    chrom_key = None
    pos_key = None
    try:
        chrom_key = _pick(chrom_key_candidates, "chromosome")
        pos_key = _pick(pos_key_candidates, "position")
    except KeyError:
        print("[WARN] No chrom/pos keys found; SNP order cannot be re-sorted by genome.")

    X = npz[geno_key]
    sample_ids = npz[sample_key]
    snp_ids = npz[snp_key] if snp_key is not None else None
    chrom = npz[chrom_key] if chrom_key is not None else None
    pos = npz[pos_key] if pos_key is not None else None

    if X.ndim != 2:
        raise ValueError(f"Genotype matrix {geno_key} must be 2D, got shape {X.shape}")

    n_samples, n_snps = X.shape
    print(f"[INFO] Genotype matrix shape: {X.shape} (samples x SNPs)")

    if len(sample_ids) != n_samples:
        raise ValueError(
            f"sample_ids length ({len(sample_ids)}) does not match "
            f"number of rows in X ({n_samples})"
        )
    if snp_ids is not None and len(snp_ids) != n_snps:
        raise ValueError(
            f"snp_ids length ({len(snp_ids)}) does not match "
            f"number of columns in X ({n_snps})"
        )

    # Keep any extra arrays as metadata, excluding the main matrix and IDs
    meta = {
        k: npz[k]
        for k in npz.files
        if k not in {geno_key, sample_key, snp_key, chrom_key, pos_key}
    }
    return {
        "X": X,
        "sample_ids": np.array(sample_ids),
        "snp_ids": np.array(snp_ids) if snp_ids is not None else None,
        "chrom": np.array(chrom) if chrom is not None else None,
        "pos": np.array(pos) if pos is not None else None,
        "meta": meta,
        "keys": {
            "geno": geno_key,
            "sample_ids": sample_key,
            "snp_ids": snp_key,
            "chrom": chrom_key,
            "pos": pos_key,
        },
    }


def sort_snps_by_genome(geno_dict: dict):
    """
    If chrom and pos are available, sort SNPs by (chrom, pos).
    Returns a new dict with X and all SNP-level arrays re-ordered.
    """
    X = geno_dict["X"]
    chrom = geno_dict["chrom"]
    pos = geno_dict["pos"]
    snp_ids = geno_dict["snp_ids"]
    meta = geno_dict["meta"]

    n_samples, n_snps = X.shape

    if chrom is None or pos is None:
        print("[INFO] No chrom/pos in geno_core; keeping SNP order as-is.")
        order = np.arange(n_snps)
    else:
        print("[INFO] Sorting SNPs by chromosome and position.")
        chrom_arr = np.asarray(chrom)
        pos_arr = np.asarray(pos)
        order = np.lexsort((pos_arr, chrom_arr))
        print(f"[INFO] SNP order changed for {np.sum(order != np.arange(n_snps))} / {n_snps} SNPs")

    X_sorted = X[:, order]

    new_geno = {
        "X": X_sorted,
        "sample_ids": geno_dict["sample_ids"],
    }

    if snp_ids is not None:
        new_geno["snp_ids"] = snp_ids[order]
    if chrom is not None:
        new_geno["chrom"] = np.asarray(chrom)[order]
    if pos is not None:
        new_geno["pos"] = np.asarray(pos)[order]

    # Reorder any SNP-level metadata arrays in meta (length == n_snps)
    for key, arr in meta.items():
        arr = np.asarray(arr)
        if arr.shape == (n_snps,):
            new_geno[key] = arr[order]
        else:
            new_geno[key] = arr  # sample-level or other shapes: keep as-is

    return new_geno


def save_ai_geno_core(geno_arrays: dict, out_path: str):
    """
    Save ai_geno_core.npz with all arrays in geno_arrays.
    """
    safe_mkdir(os.path.dirname(out_path))
    np.savez(out_path, **geno_arrays)
    print(f"[OK] Saved AI-ready geno_core to:\n  {out_path}")


def load_pheno_core(path: str):
    """
    Load pheno_core.csv from Idea 1 and make sure the index is the sample ID.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"pheno_core file not found:\n  {path}")

    print(f"[INFO] Loading pheno_core from:\n  {path}")
    df = pd.read_csv(path)

    # Try to standardise the ID column
    if "ID" in df.columns:
        df = df.set_index("ID")
    elif "sample_id" in df.columns:
        df = df.set_index("sample_id")
    else:
        # Assume the first column is the ID if not explicitly labelled
        df = df.set_index(df.columns[0])
        df.index.name = "ID"

    print(f"[INFO] pheno_core shape: {df.shape} (rows x cols)")
    return df


def augment_pheno_with_dataset_s1(pheno_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join extra metadata from Dataset S1:
      - 'Genotype Name'
      - all columns whose names start with 'miinv' (inversion dosage calls)
    """
    if not os.path.exists(DATASET_S1_XLSX):
        print(
            f"[WARN] Dataset S1 Excel not found at:\n  {DATASET_S1_XLSX}\n"
            "       Proceeding without extra metadata."
        )
        return pheno_df

    print(f"[INFO] Loading Dataset S1 from:\n  {DATASET_S1_XLSX}")
    ds1 = pd.read_excel(DATASET_S1_XLSX, sheet_name=DATASET_S1_SHEET)
    if DATASET_S1_ID_COL not in ds1.columns:
        raise KeyError(
            f"Expected ID column '{DATASET_S1_ID_COL}' not found in Dataset S1. "
            f"Columns: {list(ds1.columns)}"
        )

    ds1 = ds1.set_index(DATASET_S1_ID_COL)

    # Subset to samples present in pheno_df
    ds1 = ds1.loc[ds1.index.intersection(pheno_df.index)]

    # Keep a small, interpretable set of extra columns
    extra_cols = []
    if "Genotype Name" in ds1.columns:
        extra_cols.append("Genotype Name")

    # Any inversion columns starting with 'miinv' (case-insensitive)
    miinv_cols = [c for c in ds1.columns if c.lower().startswith("miinv")]
    extra_cols.extend(miinv_cols)

    if not extra_cols:
        print("[WARN] No 'Genotype Name' or 'miinv*' columns found in Dataset S1.")
        return pheno_df

    ds1_subset = ds1[extra_cols]
    print(
        "[INFO] Adding extra metadata columns from Dataset S1:\n  "
        + ", ".join(extra_cols)
    )

    # Align to pheno_df index (fill NaN where Dataset S1 lacks rows)
    ds1_subset = ds1_subset.reindex(pheno_df.index)

    pheno_ext = pheno_df.join(ds1_subset)
    print(f"[INFO] Extended pheno_core shape (with metadata): {pheno_ext.shape}")
    return pheno_ext


def compute_global_pcs(X: np.ndarray, sample_ids, n_pcs: int) -> pd.DataFrame:
    """
    Compute global PCs on the genotype matrix X (samples x SNPs).
    Returns a DataFrame of shape (n_samples x n_pcs) with index=sample_ids.
    """
    print(f"[INFO] Computing global PCs (n_pcs={n_pcs})")

    X_float = X.astype(float, copy=True)
    # Simple mean imputation if any missing values remain
    if np.isnan(X_float).any():
        print("[WARN] Missing genotypes detected; applying mean imputation per SNP.")
        col_means = np.nanmean(X_float, axis=0)
        inds = np.where(np.isnan(X_float))
        X_float[inds] = np.take(col_means, inds[1])

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X_float)

    max_pcs = min(n_pcs, X_scaled.shape[0] - 1, X_scaled.shape[1])
    if max_pcs < n_pcs:
        print(
            f"[WARN] Requested {n_pcs} PCs but only {max_pcs} are possible; "
            "adjusting n_pcs."
        )
        n_pcs = max_pcs

    pca = PCA(n_components=n_pcs, random_state=RANDOM_STATE)
    pcs = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_
    cum_explained = explained.cumsum()
    print(
        "[INFO] Variance explained by PCs (first 10 if available): "
        + ", ".join(f"{v:.3f}" for v in explained[:10])
    )
    print(
        f"[INFO] Cumulative variance explained by first {n_pcs} PCs: "
        f"{cum_explained[-1]:.3f}"
    )

    pc_cols = [f"PC{i+1}" for i in range(n_pcs)]
    pc_df = pd.DataFrame(pcs, index=pd.Index(sample_ids, name="ID"), columns=pc_cols)
    return pc_df


def detect_trait_cols(pheno_df: pd.DataFrame, desired_trait_cols):
    """
    Detect which columns are phenotypes, using project structure:

    - pheno_core.csv (Idea 1) has only trait columns.
    - augment_pheno_with_dataset_s1 adds 'Genotype Name' + 'miinv*' metadata.
    - So in pheno_ext:
        trait cols = all columns except 'Genotype Name' and 'miinv*'.

    We:
      1) Filter out obvious metadata.
      2) Try to intersect with desired_trait_cols.
      3) Fall back to "all non-metadata" if no exact match.
    """
    all_cols = list(pheno_df.columns)

    non_meta = [
        c for c in all_cols
        if c != "Genotype Name" and not c.lower().startswith("miinv")
    ]

    print(f"[INFO] Non-metadata columns in pheno_core (candidate traits): {non_meta}")

    # 1) Try exact match with desired canonical names
    exact = [t for t in desired_trait_cols if t in non_meta]
    if len(exact) == len(desired_trait_cols):
        print(
            "[INFO] All canonical traits found in pheno_core, using columns: "
            + ", ".join(exact)
        )
        return exact

    if len(exact) > 0:
        print(
            "[WARN] Only a subset of canonical traits found, using: "
            + ", ".join(exact)
        )
        return exact

    # 2) Fall back: use all non-metadata as trait columns
    print(
        "[WARN] None of the canonical trait names were found.\n"
        "       Falling back to all non-metadata columns as traits:"
    )
    for c in non_meta:
        print(f"       - {c}")
    return non_meta


def regress_out_pcs(pheno_df: pd.DataFrame, pc_df: pd.DataFrame, trait_cols):
    """
    For each trait in trait_cols, regress trait on PCs and compute residuals.
    Returns a new DataFrame with extra columns '<trait>_PC_resid'.
    """
    missing_traits = [t for t in trait_cols if t not in pheno_df.columns]
    if missing_traits:
        raise KeyError(
            f"The following trait columns are missing from pheno_core: {missing_traits}"
        )

    # Align PC scores to pheno index
    pc_aligned = pc_df.reindex(pheno_df.index)
    if pc_aligned.isnull().values.any():
        raise ValueError(
            "NA values detected after aligning PCs to phenotype index. "
            "Check that sample IDs match between geno_core and pheno_core."
        )

    X_pc = pc_aligned.values
    model = LinearRegression()

    pheno_with_resid = pheno_df.copy()
    for trait in trait_cols:
        y = pheno_df[trait].values.astype(float)
        mask = ~np.isnan(y)
        if mask.sum() < 3:
            print(f"[WARN] Not enough non-missing values for trait '{trait}' to fit PC regression.")
            pheno_with_resid[f"{trait}_PC_resid"] = np.nan
            continue

        model.fit(X_pc[mask, :], y[mask])
        y_hat = model.predict(X_pc[mask, :])
        resid = np.empty_like(y)
        resid[:] = np.nan
        resid[mask] = y[mask] - y_hat

        pheno_with_resid[f"{trait}_PC_resid"] = resid
        print(
            f"[INFO] Trait '{trait}': regressed on {X_pc.shape[1]} PCs; "
            f"var(y)={np.nanvar(y):.3f}, var(resid)={np.nanvar(resid):.3f}"
        )

    return pheno_with_resid


def main():
    print("=" * 72)
    print("Mango GS – Idea 3: AI-ready core data builder (01_ai_core_data.py)")
    print("=" * 72)

    # 1) Ensure output directory
    safe_mkdir(OUT_CORE_DIR)
    print(f"[INFO] Output directory:\n  {OUT_CORE_DIR}")

    # 2) Load geno_core from Idea 1
    geno = load_geno_core(GENO_CORE_PATH)

    # 3) Sort SNPs by chromosome/position if possible
    geno_sorted = sort_snps_by_genome(geno)

    # 4) Save AI-ready geno_core
    ai_geno_core_path = os.path.join(OUT_CORE_DIR, "ai_geno_core.npz")
    save_ai_geno_core(geno_sorted, ai_geno_core_path)

    # 5) Load pheno_core and align to geno sample IDs
    pheno = load_pheno_core(PHENO_CORE_PATH)

    sample_ids = geno_sorted["sample_ids"]
    # Check overlap
    missing_in_pheno = [sid for sid in sample_ids if sid not in pheno.index]
    if missing_in_pheno:
        raise RuntimeError(
            "Some genotype sample IDs not found in pheno_core.\n"
            f"Missing (first 10): {missing_in_pheno[:10]}"
        )

    # Reorder phenotypes to match genotype order
    pheno = pheno.loc[sample_ids].copy()
    print("[INFO] Reordered pheno_core rows to match geno_core sample order.")

    # 6) Add extra metadata from Dataset S1 (Genotype Name, miinv* columns)
    pheno_ext = augment_pheno_with_dataset_s1(pheno)

    # 7) Compute global PCs from AI-ready genotype matrix
    X = geno_sorted["X"]
    pc_df = compute_global_pcs(X, sample_ids, N_PCS)

    # 8) Detect trait columns from pheno_ext using project structure
    trait_cols_detected = detect_trait_cols(pheno_ext, TRAIT_COLS_CANONICAL)

    # 9) Regress out PCs from each detected trait to get PC-residual phenotypes
    pheno_with_resid = regress_out_pcs(pheno_ext, pc_df, trait_cols_detected)

    # 10) Save outputs
    ai_pheno_core_path = os.path.join(OUT_CORE_DIR, "ai_pheno_core.csv")
    pheno_with_resid.to_csv(ai_pheno_core_path, index=True)
    print(f"[OK] Saved AI-ready phenotype table (with residuals) to:\n  {ai_pheno_core_path}")

    ai_pc_scores_path = os.path.join(OUT_CORE_DIR, "ai_pc_scores.csv")
    pc_df.to_csv(ai_pc_scores_path, index=True)
    print(f"[OK] Saved global PC scores to:\n  {ai_pc_scores_path}")

    # Also export a compact residuals-only table for convenience
    resid_cols = [c for c in pheno_with_resid.columns if c.endswith("_PC_resid")]
    ai_resid_path = os.path.join(OUT_CORE_DIR, "ai_residuals.csv")
    pheno_with_resid[resid_cols].to_csv(ai_resid_path, index=True)
    print(f"[OK] Saved PC-residual phenotypes to:\n  {ai_resid_path}")

    print("[DONE] 01_ai_core_data.py complete.")


if __name__ == "__main__":
    main()
