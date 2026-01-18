import os
import json
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
import matplotlib.pyplot as plt

# ================= CONFIG =================
BASE_DIR = r"C:\Users\ms\Desktop\mango\output\idea_3"
MODEL_DIR = os.path.join(BASE_DIR, "models", "wide_deep")
TENSOR_DIR = os.path.join(BASE_DIR, "tensors")
OUT_DIR = os.path.join(BASE_DIR, "interpretation", "shap")

X_PATH = os.path.join(TENSOR_DIR, "X_background.npy")
Y_TRAITS_PATH = os.path.join(TENSOR_DIR, "y_raw_traits.json")
FEATURE_MAP_PATH = os.path.join(TENSOR_DIR, "feature_map.tsv")
MODEL_PATH = os.path.join(MODEL_DIR, "wide_deep_block-raw_fold-0.keras")

TARGET_TRAIT = "FBC" 
SAMPLE_SIZE = 100 
# ==========================================

def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)

def main():
    safe_mkdir(OUT_DIR)
    print(f"Running SHAP Analysis for {TARGET_TRAIT}...")

    # 1. Load Data
    X = np.load(X_PATH).astype(np.float32)
    background = X[:SAMPLE_SIZE] 
    
    fmap = pd.read_csv(FEATURE_MAP_PATH, sep="\t")
    snp_ids = fmap['snp_id'].values
    
    with open(Y_TRAITS_PATH, "r") as f:
        traits = json.load(f)

    # 2. Load Base Model
    print(f"[INFO] Loading base model from:\n  {MODEL_PATH}")
    base_model = tf.keras.models.load_model(MODEL_PATH)

    if TARGET_TRAIT not in traits:
        raise KeyError(f"TARGET_TRAIT '{TARGET_TRAIT}' not found in traits: {traits}")
    trait_idx = traits.index(TARGET_TRAIT)
    print(f"[INFO] TARGET_TRAIT '{TARGET_TRAIT}' has index {trait_idx}.")

    # 3. Build Functional Wrapper Model
    n_snps = X.shape[1]
    inputs = tf.keras.Input(shape=(n_snps,), name="shap_input")
    preds = base_model(inputs)
    
    def select_output(x):
        trait_col = x[:, trait_idx]
        return tf.expand_dims(trait_col, axis=-1)

    outputs = tf.keras.layers.Lambda(select_output, name="select_trait")(preds)
    target_model = tf.keras.Model(inputs=inputs, outputs=outputs, name="shap_wrapper")

    # 4. Compute SHAP
    print("[INFO] Computing SHAP values (this may take a few minutes)...")
    explainer = shap.GradientExplainer(target_model, background)
    shap_values = explainer.shap_values(background)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # Squeeze to 2D array (Samples x SNPs)
    shap_values = np.squeeze(shap_values)
    print(f"[INFO] SHAP values shape: {shap_values.shape}")
    
    # Save SHAP values for independent figure generation
    shap_values_path = os.path.join(OUT_DIR, f"SHAP_Values_{TARGET_TRAIT}.npy")
    np.save(shap_values_path, shap_values)
    print(f"[SUCCESS] Saved SHAP values: {shap_values_path}")

    # 5. Summary Plot
    print("[INFO] Generating Summary Plot...")
    plt.figure()
    shap.summary_plot(shap_values, background, feature_names=snp_ids, show=False, max_display=20)
    plot_path = os.path.join(OUT_DIR, f"SHAP_Summary_{TARGET_TRAIT}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Saved SHAP Plot: {plot_path}")
    
    # 6. Dependence Plot for Top SNP
    # Calculate mean absolute SHAP value for each feature
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0) # Shape: (n_snps,)

    # 6a. Save top SHAP SNPs as a table
    top_n = 50
    # Get indices of top SNPs
    order = np.argsort(-mean_abs_shap)[:top_n]
    
    top_df = pd.DataFrame({
        "rank": np.arange(1, top_n + 1),
        "snp_id": snp_ids[order],
        "mean_abs_shap": mean_abs_shap[order]
    })
    
    top_path = os.path.join(OUT_DIR, f"SHAP_TopSNPs_{TARGET_TRAIT}.csv")
    top_df.to_csv(top_path, index=False)
    print(f"[SUCCESS] Saved top-{top_n} SHAP SNPs: {top_path}")

    # 6b. Dependence plot
    top_idx = order[0] # Index of the #1 SNP
    top_snp_id = snp_ids[top_idx]
    print(f"[INFO] Top SHAP SNP: {top_snp_id}")

    plt.figure()
    shap.dependence_plot(
        top_idx, shap_values, background,
        feature_names=snp_ids, show=False, interaction_index=None
    )
    dep_path = os.path.join(OUT_DIR, f"SHAP_Dependence_{top_snp_id}.png")
    plt.savefig(dep_path, dpi=300, bbox_inches="tight")
    print(f"[SUCCESS] Saved Dependence Plot: {dep_path}")

if __name__ == "__main__":
    main()