import os
import json
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.model_selection import KFold

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = r"C:\Users\ms\Desktop\mango\output\idea_3"

TENSOR_DIR = os.path.join(BASE_DIR, "tensors")
INTERP_DIR = os.path.join(BASE_DIR, "interpretation")
SAL_DIR = os.path.join(INTERP_DIR, "saliency")
OUT_DIR = os.path.join(INTERP_DIR, "final_editing_comparison")

X_PATH = os.path.join(TENSOR_DIR, "X_background.npy")
Y_RAW_PATH = os.path.join(TENSOR_DIR, "y_raw.npy")
Y_TRAITS_PATH = os.path.join(TENSOR_DIR, "y_raw_traits.json")

SALIENCY_FILE = os.path.join(SAL_DIR, "saliency_matrix_block-raw.csv")

TARGET_TRAIT = "FBC"
TOP_N_SNPS = 5
N_SPLITS = 5
SEED = 2025

# Threshold for declaring "additive" (relative to block effect)
REL_SYNERGY_THRESH = 0.05  # 5% of block effect


# ============================================================
# UTILITIES
# ============================================================

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_global_seed(seed: int = 2025):
    np.random.seed(seed)


def load_data():
    if not os.path.exists(X_PATH):
        raise FileNotFoundError(f"X_background.npy not found at:\n  {X_PATH}")
    if not os.path.exists(Y_RAW_PATH):
        raise FileNotFoundError(f"y_raw.npy not found at:\n  {Y_RAW_PATH}")
    if not os.path.exists(Y_TRAITS_PATH):
        raise FileNotFoundError(f"y_raw_traits.json not found at:\n  {Y_TRAITS_PATH}")

    X = np.load(X_PATH).astype(np.float32)
    y_raw = np.load(Y_RAW_PATH)

    with open(Y_TRAITS_PATH, "r", encoding="utf-8") as f:
        traits = json.load(f)

    if TARGET_TRAIT not in traits:
        raise KeyError(f"TARGET_TRAIT '{TARGET_TRAIT}' not found in traits: {traits}")
    trait_idx = traits.index(TARGET_TRAIT)

    y = y_raw[:, trait_idx].astype(float)
    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]

    print(f"[INFO] Loaded X: {X.shape}, y({TARGET_TRAIT}): {y.shape}, "
          f"after NaN filtering.")
    return X, y, traits


def load_top_snp_indices():
    if not os.path.exists(SALIENCY_FILE):
        raise FileNotFoundError(f"Saliency matrix not found at:\n  {SALIENCY_FILE}")

    df_sal = pd.read_csv(SALIENCY_FILE)
    print(f"[INFO] Loaded saliency matrix: {df_sal.shape[0]} SNPs x {df_sal.shape[1]} cols")

    # Pick saliency column: prefer normalized, fallback to raw
    norm_col = f"saliency_{TARGET_TRAIT}_norm"
    raw_col = f"saliency_{TARGET_TRAIT}"
    if norm_col in df_sal.columns:
        col_sal = norm_col
    elif raw_col in df_sal.columns:
        col_sal = raw_col
    else:
        raise KeyError(
            f"No saliency column found for trait '{TARGET_TRAIT}'. "
            f"Tried: {norm_col}, {raw_col}"
        )

    # Index column (feature index)
    idx_col = None
    for cand in ["feature_index", "snp_index", "index"]:
        if cand in df_sal.columns:
            idx_col = cand
            break
    if idx_col is None:
        raise KeyError(
            "Could not find an index column in saliency matrix. "
            "Expected one of: feature_index, snp_index, index."
        )

    df_top = df_sal.sort_values(col_sal, ascending=False).head(TOP_N_SNPS).reset_index(drop=True)
    snp_indices = df_top[idx_col].astype(int).to_numpy()

    print(f"[INFO] Using saliency column '{col_sal}' for trait '{TARGET_TRAIT}'.")
    print(f"[INFO] Top {TOP_N_SNPS} SNP indices: {snp_indices.tolist()}")
    return snp_indices


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 72)
    print("Mango GS – Idea 3: XGBoost Interaction Verification")
    print("Purpose: Test whether 'Additive Super-Gene' is model-robust")
    print("=" * 72)

    safe_mkdir(OUT_DIR)
    set_global_seed(SEED)

    X, y, traits = load_data()
    snp_indices = load_top_snp_indices()

    n_samples, n_snps = X.shape
    if np.any(snp_indices < 0) or np.any(snp_indices >= n_snps):
        raise IndexError(
            f"Some SNP indices are out of bounds for X with n_snps={n_snps}: "
            f"{snp_indices}"
        )

    print(f"[INFO] Verifying on {n_samples} samples with {n_snps} SNPs.")

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    fold_records = []
    synergy_scores = []
    block_gains = []
    sum_singles_list = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\n[INFO] Fold {fold} ...")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = XGBRegressor(
            max_depth=3,
            n_estimators=100,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=SEED,
        )
        model.fit(X_train, y_train)

        # Base prediction (mean over test set)
        base_pred = model.predict(X_test)
        base_mean = float(np.mean(base_pred))

        # Singles: flip each SNP as in DL virtual editing
        single_gains = 0.0
        per_snp_gains = []
        for idx in snp_indices:
            X_mod = X_test.copy()
            X_mod[:, idx] = 2.0 - X_mod[:, idx]  # FLIP, not force 2.0
            pred_mod = model.predict(X_mod)
            gain = float(np.mean(pred_mod) - base_mean)
            single_gains += gain
            per_snp_gains.append(gain)

        # Block: flip all SNPs in the block
        X_block = X_test.copy()
        for idx in snp_indices:
            X_block[:, idx] = 2.0 - X_block[:, idx]
        pred_block = model.predict(X_block)
        block_gain = float(np.mean(pred_block) - base_mean)

        synergy = block_gain - single_gains

        print(
            f"  Fold {fold}: "
            f"BlockGain={block_gain:.4f} | SumSingles={single_gains:.4f} | "
            f"Synergy={synergy:.4f}"
        )

        synergy_scores.append(synergy)
        block_gains.append(block_gain)
        sum_singles_list.append(single_gains)

        fold_records.append(
            {
                "fold": fold,
                "base_mean": base_mean,
                "block_gain": block_gain,
                "sum_singles": single_gains,
                "synergy": synergy,
            }
        )

    # Aggregate across folds
    avg_block = float(np.mean(block_gains)) if block_gains else float("nan")
    avg_sum_singles = float(np.mean(sum_singles_list)) if sum_singles_list else float("nan")
    avg_synergy = avg_block - avg_sum_singles
    avg_synergy_direct = float(np.mean(synergy_scores)) if synergy_scores else float("nan")

    # Relative synergy (% of block effect)
    denom = max(abs(avg_block), 1e-6)
    rel_synergy = avg_synergy / denom

    print("\n" + "-" * 72)
    print(f"[SUMMARY] XGBoost virtual editing on trait '{TARGET_TRAIT}'")
    print(f"  Mean block gain:        {avg_block:.4f}")
    print(f"  Mean sum of singles:    {avg_sum_singles:.4f}")
    print(f"  Mean synergy (recon):   {avg_synergy:.4f}")
    print(f"  Mean synergy (perfold): {avg_synergy_direct:.4f}")
    print(f"  Relative synergy:       {rel_synergy:.4%}")

    if abs(rel_synergy) < REL_SYNERGY_THRESH:
        print("\nVERDICT: CONFIRMED ADDITIVE under XGBoost as well.")
        print("  Trees found no meaningful interaction within the 5-SNP block.")
        print("  It is safe to interpret this locus as an 'additive super-gene'")
        print("  (a recombination-suppressed block of additive alleles).")
    else:
        print("\nVERDICT: INTERACTION SIGNAL DETECTED.")
        print("  XGBoost sees non-additive behaviour in the block,")
        print("  so the 'purely additive' interpretation should be softened.")

    # Save fold-level results
    safe_mkdir(OUT_DIR)
    df_folds = pd.DataFrame(fold_records)
    out_path = os.path.join(OUT_DIR, f"xgboost_virtual_editing_{TARGET_TRAIT}.csv")
    df_folds.to_csv(out_path, index=False)
    print(f"\n[OK] Saved fold-level XGBoost verification results to:\n  {out_path}")
    print("[DONE] 08d_xgboost_verification.py complete.")


if __name__ == "__main__":
    main()
