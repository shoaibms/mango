import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

# =========================
# CONFIG
# =========================

BASE_DIR = r"C:\Users\ms\Desktop\mango\output\idea_3"

TENSOR_DIR = os.path.join(BASE_DIR, "tensors")
MODEL_DIR = os.path.join(BASE_DIR, "models", "wide_deep")
INTERP_DIR = os.path.join(BASE_DIR, "interpretation")
SAL_DIR = os.path.join(INTERP_DIR, "saliency")
EDIT_DIR = os.path.join(INTERP_DIR, "editing")

# Files
X_PATH = os.path.join(TENSOR_DIR, "X_background.npy")
Y_RAW_PATH = os.path.join(TENSOR_DIR, "y_raw.npy")
Y_RAW_TRAITS_PATH = os.path.join(TENSOR_DIR, "y_raw_traits.json")
SAMPLE_IDS_PATH = os.path.join(TENSOR_DIR, "sample_ids.txt")
FEATURE_MAP_PATH = os.path.join(TENSOR_DIR, "feature_map.tsv")

# Saliency matrix from 06_ai_saliency_multitrait.py
SAL_MATRIX_PATH = os.path.join(SAL_DIR, "saliency_matrix_block-raw.csv")

# Model to use
BLOCK_NAME = "raw"
FOLD_ID = 0
MODEL_PATH = os.path.join(MODEL_DIR, f"wide_deep_block-{BLOCK_NAME}_fold-{FOLD_ID}.keras")

# Target trait (for detailed individual-level output)
TARGET_TRAIT = "FBC"

# How many top SNPs by saliency to edit
TOP_K_SNPS = 50

# Batch size for predictions
BATCH_SIZE = 32
SEED = 2024


# =========================
# UTILITIES
# =========================

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_global_seed(seed: int = 2024):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass


def load_basic_data():
    """Load X, y_raw, trait names, sample_ids (if available)."""
    if not os.path.exists(X_PATH):
        raise FileNotFoundError(f"X_background.npy not found at:\n  {X_PATH}")
    X = np.load(X_PATH).astype(np.float32)
    n_samples, n_snps = X.shape
    print(f"[INFO] X shape: {X.shape} (samples x SNPs)")

    if not os.path.exists(Y_RAW_PATH):
        raise FileNotFoundError(f"y_raw.npy not found at:\n  {Y_RAW_PATH}")
    y_raw = np.load(Y_RAW_PATH)

    with open(Y_RAW_TRAITS_PATH, "r", encoding="utf-8") as f:
        raw_traits = json.load(f)
    print(f"[INFO] y_raw shape: {y_raw.shape} | traits: {raw_traits}")

    if y_raw.shape[0] != n_samples:
        raise RuntimeError(
            f"y_raw n_samples ({y_raw.shape[0]}) != X n_samples ({n_samples})."
        )

    sample_ids = None
    if os.path.exists(SAMPLE_IDS_PATH):
        with open(SAMPLE_IDS_PATH, "r", encoding="utf-8") as f:
            sample_ids = [line.strip() for line in f if line.strip()]
        if len(sample_ids) != n_samples:
            print(
                f"[WARN] sample_ids length ({len(sample_ids)}) "
                f"!= n_samples ({n_samples}); ignoring sample IDs."
            )
            sample_ids = None

    return X, y_raw, raw_traits, sample_ids, n_snps


def load_saliency_and_feature_map(n_snps: int):
    """Load saliency matrix and feature_map (if present)."""
    if not os.path.exists(SAL_MATRIX_PATH):
        raise FileNotFoundError(f"Saliency matrix not found at:\n  {SAL_MATRIX_PATH}")
    sal = pd.read_csv(SAL_MATRIX_PATH)
    if len(sal) != n_snps:
        print(
            f"[WARN] Saliency rows ({len(sal)}) != n_snps ({n_snps}); "
            f"continuing but assume ordering matches X."
        )

    if not os.path.exists(FEATURE_MAP_PATH):
        print("[WARN] feature_map.tsv not found; SNP IDs will be indices only.")
        fmap = None
    else:
        fmap = pd.read_csv(FEATURE_MAP_PATH, sep="\t")
        if len(fmap) != n_snps:
            print(
                f"[WARN] feature_map rows ({len(fmap)}) != n_snps ({n_snps}); "
                f"will still merge by index."
            )

    return sal, fmap


def pick_top_snps_for_trait(sal_matrix: pd.DataFrame, trait: str, top_k: int):
    """
    Given saliency_matrix_block-raw.csv and a trait name, pick top_k SNP indices
    by saliency_{trait}_norm. Falls back to *_raw if *_norm not found.
    """
    col_norm = f"saliency_{trait}_norm"
    col_raw = f"saliency_{trait}_raw"

    if col_norm in sal_matrix.columns:
        col = col_norm
    elif col_raw in sal_matrix.columns:
        col = col_raw
    else:
        raise ValueError(
            f"Saliency column for trait '{trait}' not found. "
            f"Expected one of: {col_norm}, {col_raw}"
        )

    print(f"[INFO] Selecting top {top_k} SNPs for trait '{trait}' using column '{col}'.")
    sal_sub = sal_matrix[["feature_index", col]].copy()
    sal_sub = sal_sub.sort_values(col, ascending=False)
    top = sal_sub.head(top_k)
    snp_indices = top["feature_index"].to_numpy(dtype=int)
    return top, snp_indices, col


# =========================
# MAIN
# =========================

def main():
    print("=" * 72)
    print("Mango GS – Idea 3: Virtual Editing Scenarios (Trade-offs)")
    print(" (08_virtual_editing_scenarios.py)")
    print("=" * 72)

    safe_mkdir(INTERP_DIR)
    safe_mkdir(EDIT_DIR)
    set_global_seed(SEED)

    # 1. Load core data
    X, y_raw, raw_traits, sample_ids, n_snps = load_basic_data()

    # Find target trait index
    if TARGET_TRAIT not in raw_traits:
        raise ValueError(
            f"TARGET_TRAIT '{TARGET_TRAIT}' not found in raw traits: {raw_traits}"
        )
    trait_idx = raw_traits.index(TARGET_TRAIT)
    print(f"[INFO] Target trait '{TARGET_TRAIT}' index: {trait_idx}")

    # 2. Load saliency and feature map
    sal_matrix, feature_map = load_saliency_and_feature_map(n_snps)

    # 3. Select top-K SNPs by saliency for the target trait
    top_sal, top_indices, sal_col_used = pick_top_snps_for_trait(
        sal_matrix, TARGET_TRAIT, TOP_K_SNPS
    )
    print(f"[INFO] Top SNP indices (first 10): {top_indices[:10]}")

    # 4. Load the Wide&Deep model for the chosen block / fold
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at:\n  {MODEL_PATH}")
    print(f"[INFO] Loading model:\n  {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    # 5. Baseline predictions for ALL traits
    print("[INFO] Computing baseline predictions for all traits...")
    y_base_full = model.predict(X, batch_size=BATCH_SIZE, verbose=0)  # (n_samples, n_traits)
    n_samples, n_traits = y_base_full.shape

    if n_traits != len(raw_traits):
        print(
            f"[WARN] Model outputs {n_traits} traits, but raw_traits has {len(raw_traits)}; "
            f"assuming first {n_traits} correspond."
        )

    # 6. For each top SNP, flip genotype and compute Δ predictions for ALL traits
    print("[INFO] Running virtual editing for top SNPs (trade-offs)...")

    all_snp_summaries = []
    all_indiv_chunks = []

    for rank in range(len(top_sal)):
        row = top_sal.iloc[rank]
        snp_idx = int(row["feature_index"])

        # Flip genotype at this SNP: 0 <-> 2, 1 stays 1
        X_edit = X.copy()
        g = X_edit[:, snp_idx]
        X_edit[:, snp_idx] = 2.0 - g

        # Predict all traits under edited genotypes
        y_edit_full = model.predict(X_edit, batch_size=BATCH_SIZE, verbose=0)
        if y_edit_full.shape != y_base_full.shape:
            raise RuntimeError(
                f"Shape mismatch for edited predictions at SNP {snp_idx}: "
                f"{y_edit_full.shape} vs base {y_base_full.shape}"
            )

        delta_full = y_edit_full - y_base_full  # (n_samples, n_traits)

        # SNP ID (if available)
        if feature_map is not None and "snp_id" in feature_map.columns:
            snp_id = feature_map.loc[snp_idx, "snp_id"]
        else:
            snp_id = f"idx_{snp_idx}"

        sal_value = float(row[sal_col_used])
        rank_by_sal = rank + 1

        # --- A. SNP-level summary: trade-offs across all traits ---
        summary_rec = {
            "target_trait": TARGET_TRAIT,
            "model_block": BLOCK_NAME,
            "fold_id": FOLD_ID,
            "rank_by_saliency": rank_by_sal,
            "snp_index": snp_idx,
            "snp_id": snp_id,
            "saliency_used_col": sal_col_used,
            "saliency_value": sal_value,
            "n_samples": n_samples,
        }

        for t_i, t_name in enumerate(raw_traits[:n_traits]):
            d = delta_full[:, t_i]
            summary_rec[f"delta_mean_{t_name}"] = float(np.mean(d))
            summary_rec[f"abs_delta_mean_{t_name}"] = float(np.mean(np.abs(d)))

        all_snp_summaries.append(summary_rec)

        # --- B. Individual-level deltas for the TARGET_TRAIT (vectorised) ---
        df_chunk = pd.DataFrame(
            {
                "target_trait": TARGET_TRAIT,
                "model_block": BLOCK_NAME,
                "fold_id": FOLD_ID,
                "rank_by_saliency": rank_by_sal,
                "snp_index": snp_idx,
                "snp_id": snp_id,
                "sample_index": np.arange(n_samples, dtype=int),
                "baseline_pred": y_base_full[:, trait_idx],
                "edited_pred": y_edit_full[:, trait_idx],
                "delta": delta_full[:, trait_idx],
            }
        )
        if sample_ids is not None:
            df_chunk["sample_id"] = sample_ids

        all_indiv_chunks.append(df_chunk)

        if rank_by_sal <= 3:
            d_target = float(np.mean(delta_full[:, trait_idx]))
            print(
                f"  [INFO] Rank {rank_by_sal}: snp_idx={snp_idx}, snp_id={snp_id}, "
                f"mean Δ{TARGET_TRAIT} = {d_target:.4f}"
            )

    # 7. Save outputs
    safe_mkdir(EDIT_DIR)

    df_snps = pd.DataFrame(all_snp_summaries)
    out_snps = os.path.join(
        EDIT_DIR,
        f"virtual_editing_tradeoffs_{TARGET_TRAIT}_block-{BLOCK_NAME}_fold-{FOLD_ID}.csv",
    )
    df_snps.to_csv(out_snps, index=False)
    print(f"[OK] Saved SNP trade-off summary to:\n  {out_snps}")

    if all_indiv_chunks:
        df_indiv = pd.concat(all_indiv_chunks, ignore_index=True)
        out_indiv = os.path.join(
            EDIT_DIR,
            f"virtual_editing_individuals_{TARGET_TRAIT}_block-{BLOCK_NAME}_fold-{FOLD_ID}.csv",
        )
        df_indiv.to_csv(out_indiv, index=False)
        print(f"[OK] Saved individual-level deltas to:\n  {out_indiv}")
    else:
        print("[WARN] No individual-level chunks generated.")

    print("[DONE] 08_virtual_editing_scenarios.py complete.")


if __name__ == "__main__":
    main()
