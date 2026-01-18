import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================= CONFIG =================
ROOT_DIR = r"C:\Users\ms\Desktop\mango"
SUMMARY_DIR = os.path.join(ROOT_DIR, "output", "idea_1", "summary")
PHENO_FILE = os.path.join(ROOT_DIR, "output", "idea_1", "core_data", "pheno_core.csv")
OUT_DIR = os.path.join(ROOT_DIR, "output", "idea_2", "breeder_tools")

# Input: The Model Performance Summary from Idea 1
MODEL_PERF_FILE = os.path.join(SUMMARY_DIR, "cv_transferability_summary.csv")

SELECTION_INTENSITY = 1.755  # Top 10%
# ==========================================

def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)

def main():
    safe_mkdir(OUT_DIR)
    print("Calculating Realized Genetic Gain...")

    # 1. Load Data
    if not os.path.exists(PHENO_FILE) or not os.path.exists(MODEL_PERF_FILE):
        print("[WARN] Inputs missing. Wait for Idea 1 Summary (Step 06) to finish.")
        return

    pheno = pd.read_csv(PHENO_FILE, index_col=0)
    models = pd.read_csv(MODEL_PERF_FILE)

    results = []

    print(f"{'Trait':<10} {'r (Acc)':<10} {'Gain (%)':<10}")
    print("-" * 35)

    for trait in models['trait'].unique():
        if trait not in pheno.columns: continue

        sigma_p = pheno[trait].std()
        mean_p = pheno[trait].mean()

        # Get Accuracy (r)
        row = models[models['trait'] == trait].iloc[0]
        
        # We calculate for both Random CV (Optimistic) and Leave-Cluster-Out (Robust)
        # Handle potential missing columns gracefully
        r_opt = row.get('r_random_pc', 0)
        r_rob = row.get('r_leave_cluster_out_pc', 0)

        # Calculate Gains
        gain_opt_units = SELECTION_INTENSITY * r_opt * sigma_p
        gain_opt_pct = (gain_opt_units / mean_p) * 100 if mean_p != 0 else 0
        
        gain_rob_units = SELECTION_INTENSITY * r_rob * sigma_p
        gain_rob_pct = (gain_rob_units / mean_p) * 100 if mean_p != 0 else 0

        print(f"{trait:<10} {r_opt:<10.3f} {gain_opt_pct:<10.1f}%")

        results.append({
            "Trait": trait,
            "Scenario": "Within-Pop",
            "Accuracy_r": r_opt,
            "Gain_Percent": gain_opt_pct
        })
        results.append({
            "Trait": trait,
            "Scenario": "Cross-Pop",
            "Accuracy_r": r_rob,
            "Gain_Percent": gain_rob_pct
        })

    # 2. Save Table
    df_gain = pd.DataFrame(results)
    csv_path = os.path.join(OUT_DIR, "Estimated_Genetic_Gain.csv")
    df_gain.to_csv(csv_path, index=False)
    print("-" * 35)
    print(f"[SUCCESS] Gain Table saved: {csv_path}")

    # 3. Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_gain, x='Accuracy_r', y='Gain_Percent', 
                    hue='Trait', style='Scenario', s=150, palette='viridis')
    
    plt.axhline(y=5.0, color='red', linestyle=':', alpha=0.5, label='5% Threshold')
    plt.title("Projected Genetic Gain (Top 10% Selection)")
    plt.xlabel("Prediction Accuracy ($r$)")
    plt.ylabel("Genetic Gain (% increase over mean)")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plot_path = os.path.join(OUT_DIR, "Genetic_Gain_Projection.png")
    plt.savefig(plot_path, dpi=300)
    print(f"[SUCCESS] Plot saved: {plot_path}")

if __name__ == "__main__":
    main()