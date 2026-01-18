import os
import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================

METRICS_DIR = r"C:\Users\ms\Desktop\mango\output\idea_3\metrics"

CNN_METRICS_PATH = os.path.join(METRICS_DIR, "cnn_cv_metrics.csv")
WD_METRICS_PATH = os.path.join(METRICS_DIR, "wide_deep_cv_metrics.csv")

OUT_SUMMARY_PATH = os.path.join(METRICS_DIR, "model_performance_summary.csv")
OUT_BEST_PATH = os.path.join(METRICS_DIR, "model_performance_best_by_trait.csv")


# =========================
# UTILITIES
# =========================

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_metrics():
    """
    Load CNN and Wide&Deep CV metrics, annotate with a common schema, and return
    a single long-format DataFrame.
    """
    dfs = []

    # CNN metrics
    if os.path.exists(CNN_METRICS_PATH):
        print(f"[INFO] Loading CNN metrics from:\n  {CNN_METRICS_PATH}")
        cnn = pd.read_csv(CNN_METRICS_PATH)

        required_cnn_cols = {
            "trait_type",
            "trait",
            "fold_id",
            "n_train",
            "n_test",
            "pearson_r",
            "rmse",
            "mae",
        }
        missing = required_cnn_cols - set(cnn.columns)
        if missing:
            raise ValueError(
                f"cnn_cv_metrics.csv is missing required columns: {missing}"
            )

        cnn["model"] = "cnn_shallow"
        cnn["model_family"] = "deep_learning"  # Added model_family
        # trait_type is 'raw' or 'resid'
        cnn["trait_group"] = cnn["trait_type"]

        dfs.append(
            cnn[
                [
                    "model_family",  # Added
                    "model",
                    "trait_group",
                    "trait",
                    "fold_id",
                    "n_train",
                    "n_test",
                    "pearson_r",
                    "rmse",
                    "mae",
                ]
            ].copy()
        )
    else:
        print(f"[WARN] CNN metrics file not found at:\n  {CNN_METRICS_PATH}")

    # Wide&Deep metrics
    if os.path.exists(WD_METRICS_PATH):
        print(f"[INFO] Loading Wide&Deep metrics from:\n  {WD_METRICS_PATH}")
        wd = pd.read_csv(WD_METRICS_PATH)

        required_wd_cols = {
            "model",
            "block",
            "trait",
            "fold_id",
            "n_train",
            "n_test",
            "pearson_r",
            "rmse",
            "mae",
        }
        missing = required_wd_cols - set(wd.columns)
        if missing:
            raise ValueError(
                f"wide_deep_cv_metrics.csv is missing required columns: {missing}"
            )

        # block is 'raw' / 'resid'
        wd["trait_group"] = wd["block"]
        # normalise model name just in case
        wd["model"] = "wide_deep"
        wd["model_family"] = "deep_learning"  # Added model_family

        dfs.append(
            wd[
                [
                    "model_family",  # Added
                    "model",
                    "trait_group",
                    "trait",
                    "fold_id",
                    "n_train",
                    "n_test",
                    "pearson_r",
                    "rmse",
                    "mae",
                ]
            ].copy()
        )
    else:
        print(f"[WARN] Wide&Deep metrics file not found at:\n  {WD_METRICS_PATH}")

    if not dfs:
        raise RuntimeError(
            "No metrics files could be loaded. "
            "Expected at least one of cnn_cv_metrics.csv or wide_deep_cv_metrics.csv."
        )

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Combined metrics shape: {df_all.shape}")
    return df_all


def summarise_by_trait(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    For each (model_family, model, trait_group, trait), compute summary statistics over folds.
    """
    # Added model_family to grouping
    group_cols = ["model_family", "model", "trait_group", "trait"]

    def agg_fn(x):
        return pd.Series(
            {
                "pearson_r_mean": x["pearson_r"].mean(),
                "pearson_r_median": x["pearson_r"].median(),  # Added median r
                "pearson_r_std": x["pearson_r"].std(ddof=1),
                "pearson_r_min": x["pearson_r"].min(),
                "pearson_r_max": x["pearson_r"].max(),
                "rmse_mean": x["rmse"].mean(),
                "rmse_std": x["rmse"].std(ddof=1),
                "mae_mean": x["mae"].mean(),
                "mae_std": x["mae"].std(ddof=1),
                "n_folds": x["fold_id"].nunique(),
                # For info: average n_train/n_test across folds
                "n_train_mean": x["n_train"].mean(),
                "n_test_mean": x["n_test"].mean(),
            }
        )

    summary = df_all.groupby(group_cols, as_index=False).apply(agg_fn)
    # In newer pandas versions, groupby + apply may create an index column named None:
    if None in summary.columns:
        summary = summary.drop(columns=[None])

    # sort for readability
    summary = summary.sort_values(["trait_group", "trait", "pearson_r_mean"], ascending=[True, True, False]).reset_index(drop=True)
    return summary


def best_model_by_trait(summary: pd.DataFrame) -> pd.DataFrame:
    """
    For each (trait_group, trait), pick the model with highest mean Pearson r.
    """
    # We want one row per trait_group/trait with the top model by pearson_r_mean
    summary_sorted = summary.sort_values(
        ["trait_group", "trait", "pearson_r_mean"], ascending=[True, True, False]
    )
    best = summary_sorted.groupby(["trait_group", "trait"], as_index=False).head(1)
    best = best.reset_index(drop=True)
    return best


# =========================
# MAIN
# =========================

def main():
    print("=" * 72)
    print("Mango GS – Idea 3: Model performance summary")
    print(" (05_model_performance_summary.py)")
    print("=" * 72)

    safe_mkdir(METRICS_DIR)

    df_all = load_metrics()

    # 1) Summary stats
    summary = summarise_by_trait(df_all)
    print("[INFO] Per-model, per-trait summary (first 10 rows):")
    print(summary.head(10).to_string(index=False))

    summary.to_csv(OUT_SUMMARY_PATH, index=False)
    print(f"[OK] Saved model performance summary to:\n  {OUT_SUMMARY_PATH}")

    # 2) Best model per trait
    best = best_model_by_trait(summary)
    print("\n[INFO] Best model per trait (by mean Pearson r, first 10 rows):")
    print(best.head(10).to_string(index=False))

    best.to_csv(OUT_BEST_PATH, index=False)
    print(f"[OK] Saved best model per trait to:\n  {OUT_BEST_PATH}")

    print("[DONE] 05_model_performance_summary.py complete.")


if __name__ == "__main__":
    main()