import os
import json
import numpy as np
import concurrent.futures

try:
    import pandas as pd
except ImportError as e:
    raise SystemExit("pandas is required. Install with: pip install pandas") from e

try:
    from sklearn.model_selection import KFold
except ImportError as e:
    raise SystemExit(
        "scikit-learn is required. Install with: pip install scikit-learn"
    ) from e


# =========================
# CONFIG
# =========================

# Inputs from 01_ai_core_data.py
AI_GENO_CORE_PATH = r"C:\Users\ms\Desktop\mango\output\idea_3\core_data\ai_geno_core.npz"
AI_PHENO_CORE_PATH = r"C:\Users\ms\Desktop\mango\output\idea_3\core_data\ai_pheno_core.csv"

# Output root for tensors
OUT_ROOT = r"C:\Users\ms\Desktop\mango\output\idea_3"
OUT_TENSOR_DIR = os.path.join(OUT_ROOT, "tensors")

# CV settings
N_SPLITS = 5
RANDOM_STATE = 42
SHUFFLE = True

# HARDWARE SETTINGS
# Use roughly all cores, leaving 1 or 2 for OS background tasks
N_JOBS = max(1, os.cpu_count() - 2) 


# =========================
# UTILITIES
# =========================

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_ai_geno_core(path: str):
    """
    Load ai_geno_core.npz.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"ai_geno_core file not found:\n  {path}")

    print(f"[INFO] Loading ai_geno_core from:\n  {path}")
    npz = np.load(path, allow_pickle=True)
    keys = list(npz.files)
    print(f"[INFO] ai_geno_core keys: {keys}")

    geno_key_candidates = ["X", "G", "geno", "geno_matrix"]
    sample_key_candidates = ["sample_ids", "samples", "lines"]
    snp_key_candidates = ["snp_ids", "variant_ids", "markers"]

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

    try:
        snp_key = _pick(snp_key_candidates, "SNP IDs")
    except KeyError:
        print("[WARN] No SNP ID key found; proceeding without explicit SNP IDs.")
        snp_key = None

    X = npz[geno_key]
    sample_ids = npz[sample_key]
    snp_ids = npz[snp_key] if snp_key is not None else None

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

    extra_snp_meta = {}
    for k in keys:
        if k in {geno_key, sample_key, snp_key}:
            continue
        arr = npz[k]
        if arr.shape == (n_snps,):
            extra_snp_meta[k] = arr

    return (
        X,
        np.array(sample_ids),
        np.array(snp_ids) if snp_ids is not None else None,
        extra_snp_meta,
    )


def load_ai_pheno_core(path: str):
    """
    Load ai_pheno_core.csv and return a DataFrame indexed by sample ID.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"ai_pheno_core file not found:\n  {path}")

    print(f"[INFO] Loading ai_pheno_core from:\n  {path}")
    df = pd.read_csv(path)

    if "ID" in df.columns:
        df = df.set_index("ID")
    else:
        df = df.set_index(df.columns[0])
        df.index.name = "ID"

    print(f"[INFO] ai_pheno_core shape: {df.shape} (rows x cols)")
    return df


def align_samples(X, sample_ids, pheno_df):
    """
    Ensure that pheno_df has exactly the same samples as X in the same order.
    """
    missing = [sid for sid in sample_ids if sid not in pheno_df.index]
    if missing:
        raise RuntimeError(
            "Some genotype sample IDs not found in ai_pheno_core.\n"
            f"Missing (first 10): {missing[:10]}"
        )

    pheno_aligned = pheno_df.loc[sample_ids].copy()
    print("[INFO] Aligned ai_pheno_core rows to match ai_geno_core sample order.")
    return pheno_aligned


def detect_traits_and_residuals(pheno_df: pd.DataFrame):
    """
    Detect raw and residual trait columns.
    """
    all_cols = list(pheno_df.columns)

    meta_cols = []
    if "Genotype Name" in all_cols:
        meta_cols.append("Genotype Name")
    meta_cols.extend([c for c in all_cols if c.lower().startswith("miinv")])

    resid_cols = [c for c in all_cols if c.endswith("_PC_resid")]
    raw_candidate_cols = [
        c for c in all_cols if c not in meta_cols and not c.endswith("_PC_resid")
    ]

    print(f"[INFO] Detected metadata columns: {meta_cols}")
    print(f"[INFO] Residual phenotype columns: {resid_cols}")
    print(f"[INFO] Raw candidate phenotype columns: {raw_candidate_cols}")

    raw_traits = []
    for c in raw_candidate_cols:
        if pd.api.types.is_numeric_dtype(pheno_df[c]):
            raw_traits.append(c)
        else:
            print(f"[WARN] Column '{c}' is non-numeric; skipping as raw trait.")

    resid_traits = []
    for c in resid_cols:
        if pd.api.types.is_numeric_dtype(pheno_df[c]):
            resid_traits.append(c)
        else:
            print(f"[WARN] Column '{c}' is residual but non-numeric; skipping.")

    print(f"[INFO] Final raw trait columns: {raw_traits}")
    print(f"[INFO] Final residual trait columns: {resid_traits}")

    if not raw_traits:
        raise RuntimeError("No numeric raw traits detected in ai_pheno_core.")

    return raw_traits, resid_traits


def _impute_chunk(chunk):
    """Helper function to run imputation on a smaller chunk of columns."""
    # Check if chunk has NaNs to avoid unnecessary work
    if not np.isnan(chunk).any():
        return chunk.astype(np.float32)
    
    col_means = np.nanmean(chunk, axis=0)
    inds = np.where(np.isnan(chunk))
    chunk[inds] = np.take(col_means, inds[1])
    return chunk.astype(np.float32)


def impute_genotypes(X: np.ndarray):
    """
    Impute missing genotypes using column means.
    Uses parallel processing for efficiency on large matrices.
    """
    X = X.astype(float)
    
    # Quick check to see if any imputation is needed at all
    if not np.isnan(X).any():
        print("[INFO] No missing genotypes detected. Conversion only.")
        return X.astype(np.float32)

    print(f"[INFO] Missing genotypes detected. Imputing in parallel using {N_JOBS} cores...")
    
    n_samples, n_snps = X.shape
    # Split columns into chunks based on number of cores
    # np.array_split creates views/copies depending on memory layout, 
    # usually safer to split indices and slice.
    chunk_size = n_snps // N_JOBS
    if chunk_size < 1: chunk_size = 1
    
    # Create slices for columns
    slices = []
    for i in range(0, n_snps, chunk_size):
        end = min(i + chunk_size, n_snps)
        slices.append((i, end))

    # Wrapper to process a slice of the original matrix
    # (We extract the sub-matrix, process, and return it)
    def process_slice(s):
        start, end = s
        sub_matrix = X[:, start:end]
        return _impute_chunk(sub_matrix)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=N_JOBS) as executor:
        # Submit all tasks
        futures = {executor.submit(process_slice, s): s for s in slices}
        
        # Collect results (must be ordered)
        # We'll store them in a list ordered by slice index
        sorted_futures = sorted(futures.keys(), key=lambda f: futures[f][0])
        
        for future in sorted_futures:
            results.append(future.result())

    # Concatenate all chunks back together
    X_imputed = np.concatenate(results, axis=1)
    
    print("[INFO] Parallel imputation complete.")
    return X_imputed


def build_cv_folds(n_samples: int, n_splits: int, shuffle: bool, random_state: int):
    """
    Create KFold indices over sample indices [0..n_samples-1].
    """
    print(
        f"[INFO] Building {n_splits}-fold CV with shuffle={shuffle}, random_state={random_state}"
    )
    kf = KFold(
        n_splits=n_splits, shuffle=shuffle, random_state=random_state
    )
    folds = []
    for fold_id, (train_idx, test_idx) in enumerate(kf.split(np.arange(n_samples))):
        folds.append(
            {
                "fold_id": int(fold_id),
                "train_idx": train_idx.tolist(),
                "test_idx": test_idx.tolist(),
            }
        )
        print(
            f"[INFO] Fold {fold_id}: n_train={len(train_idx)}, n_test={len(test_idx)}"
        )
    return folds


def save_feature_map(snp_ids, extra_snp_meta, out_path: str):
    """
    Save SNP feature map with optional metadata columns.
    """
    n_snps = len(snp_ids) if snp_ids is not None else None

    safe_mkdir(os.path.dirname(out_path))

    if n_snps is not None:
        # Build a DataFrame for fast writing
        data = {
            "feature_index": np.arange(n_snps),
            "snp_id": snp_ids
        }
        
        # Add extra meta
        for k, arr in extra_snp_meta.items():
            arr = np.asarray(arr)
            if arr.shape == (n_snps,):
                data[k] = arr
            else:
                print(f"[WARN] Extra SNP meta '{k}' skipped (shape mismatch).")

        df_map = pd.DataFrame(data)
        print(f"[INFO] Writing feature map with {n_snps} rows...")
        df_map.to_csv(out_path, sep="\t", index=False)
        
    else:
        print("[WARN] No snp_ids available; writing minimal header.")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("feature_index\tsnp_id\n")

    print(f"[OK] Saved feature_map to:\n  {out_path}")


def save_summary(summary_path: str, X, raw_traits, resid_traits, folds, sample_ids):
    """
    Save a human-readable summary of tensors and CV to a text file.
    """
    safe_mkdir(os.path.dirname(summary_path))
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Mango GS – Idea 3: Tensor Summary (02_cnn_tensor_builder.py)\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Genotype matrix (X_background): shape = {X.shape}\n")
        f.write(f"Number of raw traits: {len(raw_traits)}\n")
        f.write(f"Raw traits: {', '.join(raw_traits)}\n")
        f.write(f"Number of residual traits: {len(resid_traits)}\n")
        if resid_traits:
            f.write(f"Residual traits: {', '.join(resid_traits)}\n")
        f.write("\n")
        f.write(f"Number of samples: {X.shape[0]}\n")
        f.write("Sample IDs (first 10):\n")
        for sid in sample_ids[:10]:
            f.write(f"  - {sid}\n")
        f.write("\n")
        f.write(f"CV folds (N_SPLITS={len(folds)}):\n")
        for fold in folds:
            f.write(
                f"  Fold {fold['fold_id']}: "
                f"n_train={len(fold['train_idx'])}, n_test={len(fold['test_idx'])}\n"
            )
    print(f"[OK] Saved tensor summary to:\n  {summary_path}")


# =========================
# MAIN
# =========================

def main():
    print("=" * 72)
    print(f"Mango GS – Idea 3: Tensor builder (Optimized for {N_JOBS} Cores)")
    print("=" * 72)

    # 1) Ensure output directory
    safe_mkdir(OUT_TENSOR_DIR)
    print(f"[INFO] Output directory:\n  {OUT_TENSOR_DIR}")

    # 2) Load AI-ready geno_core
    X_raw, sample_ids, snp_ids, extra_snp_meta = load_ai_geno_core(AI_GENO_CORE_PATH)

    # 3) Load AI-ready pheno_core and align samples
    pheno_df = load_ai_pheno_core(AI_PHENO_CORE_PATH)
    pheno_aligned = align_samples(X_raw, sample_ids, pheno_df)

    # 4) Detect raw and residual traits
    raw_traits, resid_traits = detect_traits_and_residuals(pheno_aligned)

    # 5) Impute genotypes (Parallelized)
    X_background = impute_genotypes(X_raw)

    n_samples, n_snps = X_background.shape

    # 6) Build y_raw and y_resid matrices
    y_raw = pheno_aligned[raw_traits].to_numpy(dtype=float)
    if resid_traits:
        y_resid = pheno_aligned[resid_traits].to_numpy(dtype=float)
    else:
        y_resid = None

    print(f"[INFO] y_raw shape: {y_raw.shape}")
    if y_resid is not None:
        print(f"[INFO] y_resid shape: {y_resid.shape}")

    # 7) Build K-fold CV indices on sample indices
    folds = build_cv_folds(n_samples, N_SPLITS, SHUFFLE, RANDOM_STATE)

    # 8) Save tensors
    X_path = os.path.join(OUT_TENSOR_DIR, "X_background.npy")
    np.save(X_path, X_background)
    print(f"[OK] Saved X_background to:\n  {X_path}")

    y_raw_path = os.path.join(OUT_TENSOR_DIR, "y_raw.npy")
    np.save(y_raw_path, y_raw)
    print(f"[OK] Saved y_raw to:\n  {y_raw_path}")

    y_raw_traits_path = os.path.join(OUT_TENSOR_DIR, "y_raw_traits.json")
    with open(y_raw_traits_path, "w", encoding="utf-8") as f:
        json.dump(raw_traits, f, indent=2)
    print(f"[OK] Saved raw trait names to:\n  {y_raw_traits_path}")

    if y_resid is not None:
        y_resid_path = os.path.join(OUT_TENSOR_DIR, "y_resid.npy")
        np.save(y_resid_path, y_resid)
        print(f"[OK] Saved y_resid to:\n  {y_resid_path}")

        y_resid_traits_path = os.path.join(OUT_TENSOR_DIR, "y_resid_traits.json")
        with open(y_resid_traits_path, "w", encoding="utf-8") as f:
            json.dump(resid_traits, f, indent=2)
        print(f"[OK] Saved residual trait names to:\n  {y_resid_traits_path}")

    # 9) Save CV folds
    cv_path = os.path.join(OUT_TENSOR_DIR, "cv_folds.json")
    with open(cv_path, "w", encoding="utf-8") as f:
        json.dump(folds, f, indent=2)
    print(f"[OK] Saved CV folds to:\n  {cv_path}")

    # 10) Save sample IDs
    sample_ids_path = os.path.join(OUT_TENSOR_DIR, "sample_ids.txt")
    with open(sample_ids_path, "w", encoding="utf-8") as f:
        for sid in sample_ids:
            f.write(str(sid) + "\n")
    print(f"[OK] Saved sample IDs to:\n  {sample_ids_path}")

    # 11) Save feature map (Optimized)
    feature_map_path = os.path.join(OUT_TENSOR_DIR, "feature_map.tsv")
    if snp_ids is not None:
        save_feature_map(snp_ids, extra_snp_meta, feature_map_path)
    else:
        print("[WARN] No snp_ids in ai_geno_core; feature_map will be minimal.")
        with open(feature_map_path, "w", encoding="utf-8") as f:
            f.write("feature_index\tsnp_id\n")
        print(f"[OK] Saved minimal feature_map to:\n  {feature_map_path}")

    # 12) Save a human-readable summary
    summary_path = os.path.join(OUT_TENSOR_DIR, "tensors_summary.txt")
    save_summary(summary_path, X_background, raw_traits, resid_traits, folds, sample_ids)

    print("[DONE] 02_cnn_tensor_builder.py complete.")


if __name__ == "__main__":
    main()