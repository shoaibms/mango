import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================= CONFIG =================
ROOT_DIR = r"C:\Users\ms\Desktop\mango"
SUMMARY_DIR = os.path.join(ROOT_DIR, "output", "idea_1", "summary")
OUT_DIR = os.path.join(ROOT_DIR, "output", "figures", "figure_6_concept")

# Input files
TRANS_FILE = os.path.join(SUMMARY_DIR, "cv_transferability_summary.csv")
POLY_FILE = os.path.join(
    ROOT_DIR,
    "output",
    "idea_3",
    "breeder_resources",
    "Polygenic_Architecture_Summary.csv",
)
# ==========================================

# Thresholds and settings
MIN_RANDOM_FOR_RATIO = 0.05      # avoid unstable ratios when random r ~ 0
TRANSFERABILITY_HIGH = 0.50      # above this: meaningful cross-pop prediction
STRUCTURAL_HIGH = 3.00           # % of total weight in top 1% SNPs

TABLE_FILE = os.path.join(OUT_DIR, "Hierarchy_Concept_Map.csv")
FIG_FILE = os.path.join(OUT_DIR, "Hierarchy_Concept_Map.png")


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def classify_tier(transferability: float, structural_score: float) -> str:
    """
    Tier 1: high transferability & high structural dominance
    Tier 2: high transferability & lower structural dominance
    Tier 3: low transferability (regardless of structure)
    """
    if pd.isna(transferability) or pd.isna(structural_score):
        return "Unclassified"

    if transferability >= TRANSFERABILITY_HIGH and structural_score >= STRUCTURAL_HIGH:
        return "Tier 1"
    elif transferability >= TRANSFERABILITY_HIGH and structural_score < STRUCTURAL_HIGH:
        return "Tier 2"
    else:
        return "Tier 3"


def strategy_label(tier: str) -> str:
    if tier == "Tier 1":
        return "Global markers (inversion/SNP assays)"
    if tier == "Tier 2":
        return "Global GS (dense SNP panels)"
    if tier == "Tier 3":
        return "Local GS only (within-population)"
    return "Unspecified"


def load_and_merge() -> pd.DataFrame:
    if not os.path.exists(TRANS_FILE):
        raise FileNotFoundError(f"Transferability file not found: {TRANS_FILE}")
    if not os.path.exists(POLY_FILE):
        raise FileNotFoundError(f"Polygenic architecture file not found: {POLY_FILE}")

    df_trans = pd.read_csv(TRANS_FILE)
    df_poly = pd.read_csv(POLY_FILE)

    required_trans_cols = {
        "trait",
        "r_random_pc",
        "r_cluster_balanced_pc",
        "r_leave_cluster_out_pc",
    }
    required_poly_cols = {"trait", "top_1pct_weight_share"}

    missing_t = required_trans_cols - set(df_trans.columns)
    missing_p = required_poly_cols - set(df_poly.columns)
    if missing_t:
        raise ValueError(
            f"cv_transferability_summary.csv is missing columns: {sorted(missing_t)}"
        )
    if missing_p:
        raise ValueError(
            f"Polygenic_Architecture_Summary.csv is missing columns: {sorted(missing_p)}"
        )

    # Merge
    df = pd.merge(df_trans, df_poly, on="trait", how="inner")

    # Ensure numeric
    for col in [
        "r_random_pc",
        "r_cluster_balanced_pc",
        "r_leave_cluster_out_pc",
        "top_1pct_weight_share",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clip r for stable ratio
    df["r_random_pc_clipped"] = df["r_random_pc"].clip(lower=MIN_RANDOM_FOR_RATIO)
    df["r_leave_cluster_out_pc_clipped"] = df["r_leave_cluster_out_pc"].clip(lower=0.0)

    # Transferability ratio
    df["Transferability"] = (
        df["r_leave_cluster_out_pc_clipped"] / df["r_random_pc_clipped"]
    )

    # Structural dominance (% of weight in top 1% SNPs)
    df["Structural_Score"] = df["top_1pct_weight_share"] * 100.0

    # Tier + strategy
    df["Tier"] = df.apply(
        lambda row: classify_tier(row["Transferability"], row["Structural_Score"]),
        axis=1,
    )
    df["Strategy_Recommendation"] = df["Tier"].map(strategy_label)

    return df


def save_table(df: pd.DataFrame) -> None:
    cols = [
        "trait",
        "r_random_pc",
        "r_cluster_balanced_pc",
        "r_leave_cluster_out_pc",
        "r_random_pc_clipped",
        "r_leave_cluster_out_pc_clipped",
        "Transferability",
        "top_1pct_weight_share",
        "Structural_Score",
        "Tier",
        "Strategy_Recommendation",
    ]
    cols = [c for c in cols if c in df.columns]
    df_out = df[cols].copy()
    df_out.to_csv(TABLE_FILE, index=False)
    print(f"[OK] Saved hierarchy concept table to:\n  {TABLE_FILE}")


def make_plot(df: pd.DataFrame) -> None:
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 6))

    ax = sns.scatterplot(
        data=df,
        x="Structural_Score",
        y="Transferability",
        hue="trait",
        s=150,
        edgecolor="black",
        linewidth=1.5,
    )

    # Annotate traits
    for _, row in df.iterrows():
        ax.text(
            row["Structural_Score"] + 0.05,
            row["Transferability"],
            str(row["trait"]),
            fontsize=11,
            fontweight="bold",
            va="center",
        )

    # Threshold lines
    ax.axhline(
        y=TRANSFERABILITY_HIGH,
        color="grey",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
    )
    ax.axvline(
        x=STRUCTURAL_HIGH,
        color="grey",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
    )

    # Zone labels
    ax.text(
        STRUCTURAL_HIGH + 0.1,
        TRANSFERABILITY_HIGH + 0.35,
        "TIER 1:\nGlobal markers\n(Inversion/SNP assays)",
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
    )
    ax.text(
        STRUCTURAL_HIGH - 0.1,
        TRANSFERABILITY_HIGH + 0.35,
        "TIER 2:\nGlobal GS\n(Dense SNP panels)",
        ha="right",
        va="top",
        fontsize=10,
        fontweight="bold",
    )
    ax.text(
        STRUCTURAL_HIGH - 0.1,
        TRANSFERABILITY_HIGH - 0.35,
        "TIER 3:\nLocal GS only\n(Within-pop models)",
        ha="right",
        va="top",
        fontsize=10,
        fontweight="bold",
    )

    ax.set_title("The Precision Breeding Hierarchy", fontsize=14, fontweight="bold")
    ax.set_xlabel(
        "Structural dominance (% of total SNP weight in top 1% features)",
        fontsize=12,
    )
    ax.set_ylabel(
        "Transferability (Leave-cluster-out / Random CV accuracy)",
        fontsize=12,
    )

    # Pad limits slightly to avoid label clipping
    x_min = max(0.0, df["Structural_Score"].min() - 0.5)
    x_max = df["Structural_Score"].max() + 0.5
    y_min = -0.05
    y_max = 1.05
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(FIG_FILE, dpi=300)
    plt.close()
    print(f"[OK] Saved hierarchy concept figure to:\n  {FIG_FILE}")


def main():
    safe_mkdir(OUT_DIR)
    print("Generating 'Precision Breeding Hierarchy' table and figure...")

    df = load_and_merge()
    print(df[["trait", "Transferability", "Structural_Score", "Tier"]])

    save_table(df)
    make_plot(df)

    print("[DONE] Hierarchy computation complete.")


if __name__ == "__main__":
    main()
