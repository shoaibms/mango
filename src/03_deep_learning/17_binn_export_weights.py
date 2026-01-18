import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

# ================= CONFIG =================
BASE_DIR = r"C:\Users\ms\Desktop\mango\output\idea_3"
MODEL_DIR = os.path.join(BASE_DIR, "models", "wide_deep")
TENSOR_DIR = os.path.join(BASE_DIR, "tensors")
OUT_DIR = os.path.join(BASE_DIR, "breeder_resources")

FEATURE_MAP_PATH = os.path.join(TENSOR_DIR, "feature_map.tsv")
Y_RAW_TRAITS_PATH = os.path.join(TENSOR_DIR, "y_raw_traits.json")
X_PATH = os.path.join(TENSOR_DIR, "X_background.npy")
MODEL_PATH = os.path.join(MODEL_DIR, "wide_deep_block-raw_fold-0.keras")
# ==========================================

def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)

def main():
    safe_mkdir(OUT_DIR)
    print("Generating Polygenic Weights & Architecture Summary...")

    # 1. Load Inputs
    if not os.path.exists(X_PATH):
        print(f"[WARN] X background file not found: {X_PATH}")
        return
    X = np.load(X_PATH).astype(np.float32)
    
    with open(Y_RAW_TRAITS_PATH, "r") as f:
        traits = json.load(f)

    fmap = pd.read_csv(FEATURE_MAP_PATH, sep="\t")
    snp_ids = fmap['snp_id'].values

    # 2. MAF Calculation
    freq_alt = np.mean(X, axis=0) / 2.0
    maf = np.minimum(freq_alt, 1.0 - freq_alt)
    var_factor = 2 * freq_alt * (1.0 - freq_alt)

    # 3. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"[WARN] Model file not found: {MODEL_PATH}")
        return
    model = tf.keras.models.load_model(MODEL_PATH)
    layer = model.get_layer("wide_linear")
    weights, biases = layer.get_weights()

    # 4. Build Master Table & Summary
    df = pd.DataFrame({'SNP_ID': snp_ids, 'MAF': maf})
    
    arch_summary = []

    for i, trait in enumerate(traits):
        w = weights[:, i]
        gvar = var_factor * (w ** 2)
        total_gvar = np.sum(gvar)

        if total_gvar <= 0:
            pct_gvar = np.zeros_like(gvar)
        else:
            pct_gvar = (gvar / total_gvar) * 100.0
        
        df[f'Weight_{trait}'] = w
        df[f'Pct_Var_{trait}'] = pct_gvar
        
        # Architecture Analysis
        # Sort absolute weights to find top 1% contribution
        w_abs = np.abs(w)
        total_sum = np.sum(w_abs)

        if total_sum <= 0:
            ratio = 0.0
        else:
            w_sorted = np.sort(w_abs)
            top_1_count = int(max(1, len(w) * 0.01))
            top_1_sum = np.sum(w_sorted[-top_1_count:])
            ratio = top_1_sum / total_sum
        
        arch_type = "Oligogenic" if ratio > 0.20 else "Polygenic"
        
        arch_summary.append({
            "trait": trait,
            "top_1pct_weight_share": ratio,
            "architecture": arch_type,
            "n_snps": len(w)
        })
        print(f"  Trait {trait}: Top 1% carries {ratio:.1%} of weight -> {arch_type}")

    # 5. Save Files
    # Main Weights
    df.to_csv(os.path.join(OUT_DIR, "Mango_Polygenic_Evaluation_File.csv"), index=False)
    
    # Architecture Summary
    pd.DataFrame(arch_summary).to_csv(
        os.path.join(OUT_DIR, "Polygenic_Architecture_Summary.csv"), 
        index=False
    )
    print("[SUCCESS] Saved Weights and Architecture Summary.")

if __name__ == "__main__":
    main()