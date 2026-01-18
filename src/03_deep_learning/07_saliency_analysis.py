import os
import json
import numpy as np
import tensorflow as tf
import pandas as pd


# =========================
# CONFIG
# =========================

BASE_DIR = r"C:\Users\ms\Desktop\mango\output\idea_3"
TENSOR_DIR = os.path.join(BASE_DIR, "tensors")
MODEL_DIR = os.path.join(BASE_DIR, "models", "wide_deep")
OUT_DIR = os.path.join(BASE_DIR, "interpretation", "saliency")

# We analyse the 'raw' trait block (FBC, AFW, etc.)
BLOCK_NAME = "raw"
# Representative fold for saliency analysis
FOLD_ID = 0

# Files
X_PATH = os.path.join(TENSOR_DIR, "X_background.npy")
FEATURE_MAP_PATH = os.path.join(TENSOR_DIR, "feature_map.tsv")
TRAIT_NAMES_PATH = os.path.join(TENSOR_DIR, f"y_{BLOCK_NAME}_traits.json")
MODEL_PATH = os.path.join(MODEL_DIR, f"wide_deep_block-{BLOCK_NAME}_fold-{FOLD_ID}.keras")


# =========================
# UTILITIES
# =========================

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_feature_map() -> pd.DataFrame | None:
    """
    Load feature_map.tsv if present.

    Expected columns (from 02_cnn_tensor_builder.py):
      - feature_index
      - snp_id
      - optional extra SNP-level metadata

    Returns:
      DataFrame or None if file not found.
    """
    if not os.path.exists(FEATURE_MAP_PATH):
        print("[WARN] No feature_map.tsv found. Using only feature_index.")
        return None

    df = pd.read_csv(FEATURE_MAP_PATH, sep="\t")
    return df


# =========================
# MAIN
# =========================

def main():
    print("=" * 72)
    print("Mango GS – Idea 3, Phase 2: AI Pleiotropy Saliency Map")
    print(" (06_ai_saliency_multitrait.py)")
    print("=" * 72)

    safe_mkdir(OUT_DIR)

    # 1. Load data & trait schema
    if not os.path.exists(X_PATH):
        raise FileNotFoundError(f"X_background.npy not found at:\n  {X_PATH}")

    print(f"[INFO] Loading X from:\n  {X_PATH}")
    X = np.load(X_PATH).astype(np.float32)
    n_samples, n_snps = X.shape
    print(f"[INFO] X shape: {X.shape} (samples x SNPs)")

    if not os.path.exists(TRAIT_NAMES_PATH):
        raise FileNotFoundError(f"Trait names JSON not found at:\n  {TRAIT_NAMES_PATH}")
    with open(TRAIT_NAMES_PATH, "r", encoding="utf-8") as f:
        trait_names = json.load(f)

    n_traits = len(trait_names)
    print(f"[INFO] Traits in block '{BLOCK_NAME}' ({n_traits}): {trait_names}")

    # 2. Load model
    print(f"[INFO] Loading model:\n  {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found:\n  {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    # 3. Prepare tensors and compute gradients per trait
    X_tensor = tf.convert_to_tensor(X)

    # Dict: trait -> (n_snps,) saliency scores
    saliency_results: dict[str, np.ndarray] = {}

    print("[INFO] Computing saliency gradients (multi-trait)...")
    # We use a persistent tape to reuse forward pass
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(X_tensor)
        preds = model(X_tensor)  # shape: (n_samples, n_traits)

        # Sanity check on output dims
        if preds.shape[1] != n_traits:
            raise RuntimeError(
                f"Model output dimension ({preds.shape[1]}) does not match "
                f"number of traits in {TRAIT_NAMES_PATH} ({n_traits})."
            )

        for i, trait in enumerate(trait_names):
            print(f"  - Trait {i} / {n_traits}: {trait}")
            # Explicitly reduce sum so gradient is clear: d(sum(pred[:, i]))/dX
            target = tf.reduce_sum(preds[:, i])

            grads = tape.gradient(target, X_tensor)
            if grads is None:
                raise RuntimeError(
                    f"Got None gradients for trait '{trait}'. "
                    f"Check that the model is differentiable w.r.t. inputs."
                )

            grads_np = grads.numpy()  # shape: (n_samples, n_snps)
            # Saliency = mean absolute gradient per SNP across all samples
            saliency = np.mean(np.abs(grads_np), axis=0)  # (n_snps,)
            if saliency.shape[0] != n_snps:
                raise RuntimeError(
                    f"Saliency length {saliency.shape[0]} "
                    f"does not match n_snps={n_snps}."
                )
            saliency_results[trait] = saliency

    # Release tape
    del tape

    # 4. Build output DataFrame
    print("[INFO] Aggregating saliency results into DataFrame...")
    df_out = pd.DataFrame({"feature_index": np.arange(n_snps, dtype=int)})

    # Attach SNP IDs / metadata if available
    df_map = load_feature_map()
    if df_map is not None:
        # Drop feature_index if present to avoid duplication but keep row order
        df_map_clean = df_map.drop(columns=["feature_index"], errors="ignore")
        # Concatenate by row index (assumes same order as in 02_cnn_tensor_builder.py)
        if len(df_map_clean) != n_snps:
            print(
                f"[WARN] feature_map row count ({len(df_map_clean)}) != n_snps ({n_snps}); "
                f"concatenating anyway based on index."
            )
        df_out = pd.concat([df_out, df_map_clean], axis=1)

    # Add saliency columns for each trait (raw & min–max normalised)
    for trait in trait_names:
        scores = saliency_results[trait]
        # Normalise per trait for nicer joint plots
        s_min = scores.min()
        s_max = scores.max()
        denom = (s_max - s_min) + 1e-9
        scores_norm = (scores - s_min) / denom

        df_out[f"saliency_{trait}_raw"] = scores
        df_out[f"saliency_{trait}_norm"] = scores_norm

    # 5. Save matrix
    out_file = os.path.join(OUT_DIR, f"saliency_matrix_block-{BLOCK_NAME}.csv")
    df_out.to_csv(out_file, index=False)
    print(f"[OK] Saved multi-trait saliency matrix to:\n  {out_file}")


if __name__ == "__main__":
    main()
