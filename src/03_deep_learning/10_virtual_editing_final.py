import os
import json
import itertools
import numpy as np
import pandas as pd
try:
    import tensorflow as tf
except ImportError as e:
    raise SystemExit("tensorflow is required. Install with: pip install tensorflow") from e
try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
except ImportError as e:
    raise SystemExit("scikit-learn is required. Install with: pip install scikit-learn") from e

# ================= CONFIG =================
OUT_ROOT = r"C:\Users\ms\Desktop\mango\output\idea_3"
TENSOR_DIR = os.path.join(OUT_ROOT, "tensors")
MODEL_DIR = os.path.join(OUT_ROOT, "models", "wide_deep")
INTERP_DIR = os.path.join(OUT_ROOT, "interpretation")
SAL_DIR = os.path.join(INTERP_DIR, "saliency")
EDIT_DIR = os.path.join(INTERP_DIR, "editing", "advanced")

X_PATH = os.path.join(TENSOR_DIR, "X_background.npy")
Y_RAW_PATH = os.path.join(TENSOR_DIR, "y_raw.npy")
Y_RAW_TRAITS_PATH = os.path.join(TENSOR_DIR, "y_raw_traits.json")
CV_FOLDS_PATH = os.path.join(TENSOR_DIR, "cv_folds.json")
SAMPLE_IDS_PATH = os.path.join(TENSOR_DIR, "sample_ids.txt")
FEATURE_MAP_PATH = os.path.join(TENSOR_DIR, "feature_map.tsv")
SAL_MATRIX_PATH = os.path.join(SAL_DIR, "saliency_matrix_block-raw.csv")

TARGET_TRAIT = "FBC"
TOP_N_SINGLE = 5        
USE_PAIRS = True        
BLOCK_USES_TOP_N = True 

N_CLUSTERS = 2
N_PCS_FOR_CLUSTER = 10
BATCH_SIZE = 32
SEED = 2025

# ================= UTILITIES =================
def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def set_global_seed(seed: int = 2025):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def load_tensors():
    if not os.path.exists(X_PATH): raise FileNotFoundError(f"X not found: {X_PATH}")
    X = np.load(X_PATH).astype(np.float32)
    
    if not os.path.exists(Y_RAW_TRAITS_PATH): raise FileNotFoundError(f"Traits not found")
    with open(Y_RAW_TRAITS_PATH, "r") as f: raw_traits = json.load(f)
    
    with open(CV_FOLDS_PATH, "r") as f: folds = json.load(f)
    fold_ids = sorted({int(f["fold_id"]) for f in folds})
    
    y_raw = np.load(Y_RAW_PATH)
    
    # Optional metadata
    sample_ids = None
    if os.path.exists(SAMPLE_IDS_PATH):
        with open(SAMPLE_IDS_PATH, "r") as f: sample_ids = [l.strip() for l in f if l.strip()]
            
    feature_map = None
    if os.path.exists(FEATURE_MAP_PATH):
        feature_map = pd.read_csv(FEATURE_MAP_PATH, sep="\t")
        
    return X, y_raw, raw_traits, fold_ids, sample_ids, feature_map

def cluster_samples(X, n_clusters=N_CLUSTERS, n_pcs=N_PCS_FOR_CLUSTER):
    print(f"[INFO] Clustering samples...")
    pca = PCA(n_components=min(n_pcs, X.shape[1]), random_state=SEED)
    X_pca = pca.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    return km.fit_predict(X_pca)

def load_saliency_for_trait(target_trait):
    if not os.path.exists(SAL_MATRIX_PATH): raise FileNotFoundError(f"Saliency not found: {SAL_MATRIX_PATH}")
    df_sal = pd.read_csv(SAL_MATRIX_PATH)
    
    idx_col = "feature_index" if "feature_index" in df_sal.columns else "index"
    col = f"saliency_{target_trait}_norm"
    if col not in df_sal.columns: col = f"saliency_{target_trait}_raw"
    
    df_sorted = df_sal.sort_values(col, ascending=False).reset_index(drop=True)
    return df_sorted, idx_col, col

def build_edit_combos(df_sal_sorted, idx_col, sal_col):
    top = df_sal_sorted.head(TOP_N_SINGLE).copy()
    indices = top[idx_col].values.astype(int)
    # Handle missing snp_id column
    if "snp_id" in top.columns:
        ids = top["snp_id"].astype(str).values
    else:
        ids = [f"idx_{i}" for i in indices]
        
    vals = top[sal_col].values
    
    combos = []
    
    # 1. Singles
    for idx, sid, val in zip(indices, ids, vals):
        combos.append({
            "combo_id": f"single_{sid}",
            "edit_type": "single",
            "snp_indices": [idx],
            "snp_ids": [sid],
            "saliency_sum": val
        })
        
    # 2. Pairs
    if USE_PAIRS:
        for i, j in itertools.combinations(range(len(indices)), 2):
            combos.append({
                "combo_id": f"pair_{ids[i]}__{ids[j]}",
                "edit_type": "pair",
                "snp_indices": [indices[i], indices[j]],
                "snp_ids": [ids[i], ids[j]],
                "saliency_sum": vals[i] + vals[j]
            })
            
    # 3. Block
    if BLOCK_USES_TOP_N:
        combos.append({
            "combo_id": "block_topK",
            "edit_type": "block",
            "snp_indices": indices,
            "snp_ids": ids,
            "saliency_sum": vals.sum()
        })
        
    print(f"[INFO] Built {len(combos)} combinations.")
    return combos

# ================= MAIN =================
def main():
    print("=" * 72)
    print("Mango GS – Idea 3: Advanced Virtual Editing")
    print("=" * 72)
    safe_mkdir(EDIT_DIR)
    set_global_seed(SEED)

    # 1. Load Data
    X, y_raw, raw_traits, fold_ids, sample_ids, feature_map = load_tensors()
    n_samples, n_snps = X.shape
    
    if TARGET_TRAIT not in raw_traits: raise ValueError(f"Target trait {TARGET_TRAIT} missing")
    target_idx = raw_traits.index(TARGET_TRAIT)

    # 2. Cluster
    cluster_labels = cluster_samples(X)

    # 3. Build Combos
    df_sal, idx_col, sal_col = load_saliency_for_trait(TARGET_TRAIT)
    combos = build_edit_combos(df_sal, idx_col, sal_col)

    # Storage
    results = []

    # 4. Loop Folds
    for fold_id in fold_ids:
        model_path = os.path.join(MODEL_DIR, f"wide_deep_block-raw_fold-{fold_id}.keras")
        if not os.path.exists(model_path): continue
        
        print(f"  - Processing Fold {fold_id}...")
        model = tf.keras.models.load_model(model_path)
        
        # Base Prediction
        y_base = model.predict(X, batch_size=BATCH_SIZE, verbose=0)

        for combo in combos:
            indices = combo["snp_indices"]
            
            # Simulate fixing allele to Homozygous Alt (2.0) for introgression/gene editing
            X_edit = X.copy()
            for idx in indices:
                X_edit[:, idx] = 2.0 
            
            y_edit = model.predict(X_edit, batch_size=BATCH_SIZE, verbose=0)
            delta = y_edit - y_base

            # Record metrics per trait
            for t_i, t_name in enumerate(raw_traits):
                d_trait = delta[:, t_i]
                
                rec = {
                    "fold_id": fold_id,
                    "combo_id": combo["combo_id"],
                    "edit_type": combo["edit_type"],
                    "trait": t_name,
                    "global_delta": np.mean(d_trait),
                }
                
                # Cluster effects
                for cl in range(N_CLUSTERS):
                    mask = cluster_labels == cl
                    if np.sum(mask) > 0:
                        rec[f"delta_cluster_{cl}"] = np.mean(d_trait[mask])
                    else:
                        rec[f"delta_cluster_{cl}"] = np.nan
                
                results.append(rec)

    # 5. Aggregate & Save
    df_res = pd.DataFrame(results)
    summary = df_res.groupby(["combo_id", "edit_type", "trait"]).agg(
        mean_delta=("global_delta", "mean"),
        std_delta=("global_delta", "std"),
        mean_cl0=("delta_cluster_0", "mean"),
        mean_cl1=("delta_cluster_1", "mean")
    ).reset_index()
    
    out_sum = os.path.join(EDIT_DIR, f"virtual_editing_summary_{TARGET_TRAIT}.csv")
    summary.to_csv(out_sum, index=False)
    print(f"[OK] Saved Summary: {out_sum}")

    # 6. Haplotype Synergy Calculation
    # Calculate: Block_Effect vs Sum(Single_Effects)
    print("[INFO] Calculating Block Synergy...")
    
    synergy_data = []
    for trait in raw_traits:
        trait_df = summary[summary["trait"] == trait].set_index("combo_id")
        
        if "block_topK" in trait_df.index:
            block_eff = trait_df.loc["block_topK", "mean_delta"]
            
            # Sum singles
            singles_sum = 0
            for i in range(TOP_N_SINGLE):
                # Reconstruct single ID from combos list logic
                # We grab the combo_id from the 'combos' list we built earlier
                sid = combos[i]["combo_id"] # single_snpID
                if sid in trait_df.index:
                    singles_sum += trait_df.loc[sid, "mean_delta"]
            
            synergy = block_eff - singles_sum
            
            # Determine interpretation
            # If Block > Sum + 5%, it's Synergistic.
            # If Block < Sum - 5%, it's Antagonistic (Interference).
            limit = 0.05 * abs(block_eff) if block_eff != 0 else 0.001
            if synergy > limit: interp = "Synergistic"
            elif synergy < -limit: interp = "Antagonistic"
            else: interp = "Additive"

            synergy_data.append({
                "trait": trait,
                "block_effect": block_eff,
                "sum_singles": singles_sum,
                "synergy": synergy,
                "type": interp
            })

    out_syn = os.path.join(EDIT_DIR, "haplotype_block_synergy.csv")
    pd.DataFrame(synergy_data).to_csv(out_syn, index=False)
    print(f"[OK] Saved Synergy Report: {out_syn}")
    print(pd.DataFrame(synergy_data))

    print("[DONE] Complete.")

if __name__ == "__main__":
    main()