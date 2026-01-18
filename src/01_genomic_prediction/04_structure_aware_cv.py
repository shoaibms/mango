# -*- coding: utf-8 -*-
"""
04_structure_aware_cv

Structure-aware genomic prediction for Mango GS (Idea 1).

This script:
  1. Derives ancestry/structure clusters from genotype PCs using K-means.
  2. Performs:
       - Cluster-balanced K-fold CV (StratifiedKFold on clusters)
       - Leave-cluster-out CV (train on all clusters except one)
  3. For each trait and scheme, evaluates:
       - Ridge without PC correction
       - Ridge with PC-corrected phenotype (residualised on PCs of X)

Inputs
------
Core matrices from 01_build_core_matrices.py:
  - geno_core.npz  (G, sample_ids, variant_ids)
  - pheno_core.csv
  - meta_core.csv  (not strictly required here but kept for completeness)

Outputs (in cfg.CV_STRUCTURE_DIR)
---------------------------------
  - cv_structure_results.csv
      trait,scheme,scenario,fold,test_group,r

  - cv_structure_summary.csv
      trait,scheme,scenario,mean_r,n_folds
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
except ImportError as e:
    raise SystemExit("pandas is required. Install with: pip install pandas") from e

try:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge, LinearRegression
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
except ImportError as e:
    raise SystemExit(
        "scikit-learn is required. Install with: pip install scikit-learn"
    ) from e

import config_idea1 as cfg


def z_from_r(r: float) -> float:
    r = np.clip(r, -0.9999, 0.9999)
    return 0.5 * np.log((1 + r) / (1 - r))


def r_from_z(z: float) -> float:
    return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)


def fisher_z_ci(r: float, n: int) -> Tuple[float, float]:
    if n <= 3 or np.isnan(r):
        return np.nan, np.nan
    z = z_from_r(r)
    se_z = 1.0 / np.sqrt(n - 3)
    return r_from_z(z - 1.96 * se_z), r_from_z(z + 1.96 * se_z)


def meta_mean_r(rs: List[float], ns: List[int]) -> Tuple[float, Tuple[float, float]]:
    valid = [(r, n) for r, n in zip(rs, ns) if not np.isnan(r) and n > 3]
    if not valid:
        return np.nan, (np.nan, np.nan)
    rs_v, ns_v = zip(*valid)
    zs = np.array([z_from_r(r) for r in rs_v])
    w = np.array([n - 3 for n in ns_v], dtype=float)
    z_bar = np.sum(w * zs) / np.sum(w)
    se_z = np.sqrt(1.0 / np.sum(w))
    return r_from_z(z_bar), (
        r_from_z(z_bar - 1.96 * se_z),
        r_from_z(z_bar + 1.96 * se_z),
    )


# =========================
# Core helpers
# =========================

def compute_structure_clusters(
    G: np.ndarray,
    n_clusters: int = 3,
    random_state: int = 42,
    sample_ids: Optional[np.ndarray] = None,
    core_dir: Optional[str] = None,
) -> np.ndarray:
    """
    Derive ancestry/structure clusters from genotype matrix G using PCA + K-means.

    Parameters
    ----------
    G : array-like, shape (n_samples, n_snps)
        Genotype matrix (dosage 0–2).
    n_clusters : int
        Number of clusters for K-means.
    random_state : int
        Seed for K-means.

    Returns
    -------
    clusters : np.ndarray, shape (n_samples,)
        Integer cluster labels (0..n_clusters-1).
    """
    n_samples, n_snps = G.shape
    print(f"[INFO] Computing structure clusters from genotype matrix {G.shape}")

    scaler = StandardScaler(with_mean=True, with_std=True)
    G_scaled = scaler.fit_transform(G)

    n_pcs = min(10, n_samples - 1, n_snps)
    if n_pcs <= 0:
        raise RuntimeError(
            f"Cannot compute PCs for clustering (n_pcs={n_pcs}). "
            "Check genotype dimensions."
        )

    print(f"[INFO] Using {n_pcs} PCs for clustering")
    pca = PCA(n_components=n_pcs, random_state=random_state)
    PCs = pca.fit_transform(G_scaled)

    km = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )
    clusters = km.fit_predict(PCs)

    # Report cluster sizes
    unique, counts = np.unique(clusters, return_counts=True)
    print("[INFO] Cluster sizes (from K-means on PCs):")
    for u, c in zip(unique, counts):
        print(f"  - cluster {u}: n = {c}")

    # ============================================================
    # SAVE CLUSTER ASSIGNMENTS (for Idea 2 Option D)
    # Output: output/idea_1/core_data/sample_metadata_ml.csv
    # ============================================================
    try:
        if sample_ids is None or core_dir is None:
            raise ValueError("sample_ids or core_dir not provided")

        # sample_ids must match the row order of the genotype matrix used for clustering
        sample_ids_array = np.asarray(sample_ids, dtype=str)
        if sample_ids_array.shape[0] != clusters.shape[0]:
            raise ValueError(
                "sample_ids length does not match number of cluster assignments"
            )

        cluster_out = pd.DataFrame(
            {"cluster": clusters.astype(int)}, index=sample_ids_array
        )
        cluster_out.index.name = "sample_id"

        os.makedirs(core_dir, exist_ok=True)
        cluster_path = os.path.join(core_dir, "sample_metadata_ml.csv")
        cluster_out.to_csv(cluster_path)
        print(f"[INFO] Saved cluster assignments -> {cluster_path}")
    except Exception as e:
        print(f"[WARN] Failed to save cluster assignments: {e}")

    return clusters


def ridge_cv_with_splits(
    X: np.ndarray,
    y: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    use_pc_correction: bool,
    n_pcs: int,
    random_state: int,
) -> List[float]:
    """
    Ridge regression CV using externally provided train/test splits.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_snps)
    y : array-like, shape (n_samples,)
    splits : list of (train_idx, test_idx)
    use_pc_correction : bool
        If True, residualise y on PCs of X within each training fold.
    n_pcs : int
        Number of PCs to use when residualising (capped internally).
    random_state : int

    Returns
    -------
    r_values : list of float
        Pearson r for each split (NaN if undefined).
    """
    r_values: List[float] = []

    if use_pc_correction:
        print(f"[INFO] Ridge CV with PC correction (n_pcs={n_pcs})")
    else:
        print("[INFO] Ridge CV without PC correction")

    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train = X[train_idx, :]
        X_test = X[test_idx, :]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # Skip degenerate folds
        if np.nanstd(y_train) == 0.0:
            print(f"  [WARN] Fold {fold_idx}: y_train has zero variance; skipping.")
            r_values.append(np.nan)
            continue

        if not use_pc_correction:
            # Standard ridge on raw y
            scaler_X = StandardScaler(with_mean=True, with_std=True)
            X_train_scaled = scaler_X.fit_transform(X_train)
            X_test_scaled = scaler_X.transform(X_test)

            model = Ridge(alpha=1.0, random_state=random_state)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            if np.nanstd(y_test) == 0.0 or np.nanstd(y_pred) == 0.0:
                r = np.nan
            else:
                r = np.corrcoef(y_test, y_pred)[0, 1]

            print(f"  Fold {fold_idx}: r = {r:.3f} (no PC correction)")

        else:
            # 1) PCs of X (training only)
            scaler_pc = StandardScaler(with_mean=True, with_std=True)
            X_train_pc = scaler_pc.fit_transform(X_train)
            X_test_pc = scaler_pc.transform(X_test)

            max_pcs = min(n_pcs, X_train_pc.shape[0] - 1, X_train_pc.shape[1])
            if max_pcs <= 0:
                print(
                    f"  [WARN] Fold {fold_idx}: cannot compute PCs (max_pcs={max_pcs}); skipping."
                )
                r_values.append(np.nan)
                continue

            pca = PCA(n_components=max_pcs, random_state=random_state)
            PCs_train = pca.fit_transform(X_train_pc)
            PCs_test = pca.transform(X_test_pc)

            # 2) Regress y on PCs in training, get residuals
            lr = LinearRegression()
            lr.fit(PCs_train, y_train)
            y_hat_train = lr.predict(PCs_train)
            y_hat_test = lr.predict(PCs_test)

            y_train_res = y_train - y_hat_train
            y_test_res = y_test - y_hat_test

            # 3) Ridge on residuals (with fresh scaling of X)
            scaler_X2 = StandardScaler(with_mean=True, with_std=True)
            X_train_scaled2 = scaler_X2.fit_transform(X_train)
            X_test_scaled2 = scaler_X2.transform(X_test)

            model = Ridge(alpha=1.0, random_state=random_state)
            model.fit(X_train_scaled2, y_train_res)
            y_pred_res = model.predict(X_test_scaled2)

            if np.nanstd(y_test_res) == 0.0 or np.nanstd(y_pred_res) == 0.0:
                r = np.nan
            else:
                r = np.corrcoef(y_test_res, y_pred_res)[0, 1]

            print(f"  Fold {fold_idx}: r = {r:.3f} (PC-corrected)")

        r_values.append(float(r))

    return r_values


# =========================
# Main
# =========================

def main() -> None:
    cfg.ensure_output_dirs()
    out_dir = cfg.CV_STRUCTURE_DIR
    os.makedirs(out_dir, exist_ok=True)

    geno_path = os.path.join(cfg.CORE_DATA_DIR, "geno_core.npz")
    pheno_path = os.path.join(cfg.CORE_DATA_DIR, "pheno_core.csv")
    meta_path = os.path.join(cfg.CORE_DATA_DIR, "meta_core.csv")

    print("=" * 72)
    print("Mango GS – Idea 1: Structure-aware CV")
    print("=" * 72)
    print(f"[INFO] Geno core:   {geno_path}")
    print(f"[INFO] Pheno core:  {pheno_path}")
    print(f"[INFO] Meta core:   {meta_path}")
    print()

    if not os.path.exists(geno_path):
        raise FileNotFoundError(
            f"geno_core.npz not found at {geno_path}. "
            "Run 01_build_core_matrices.py first."
        )
    if not os.path.exists(pheno_path):
        raise FileNotFoundError(
            f"pheno_core.csv not found at {pheno_path}. "
            "Run 01_build_core_matrices.py first."
        )
    if not os.path.exists(meta_path):
        print(
            f"[WARN] meta_core.csv not found at {meta_path}. "
            "Proceeding without explicit metadata."
        )

    # 1) Load core data
    npz = np.load(geno_path, allow_pickle=True)
    G = npz["G"]
    sample_ids = npz["sample_ids"].astype(str)

    pheno_df = pd.read_csv(pheno_path, index_col=0)
    pheno_df.index = pheno_df.index.astype(str)

    if os.path.exists(meta_path):
        meta_df = pd.read_csv(meta_path, index_col=0)
        meta_df.index = meta_df.index.astype(str)
    else:
        meta_df = pd.DataFrame(index=pheno_df.index)

    print(f"[INFO] Genotype matrix shape: {G.shape} (samples x SNPs)")
    print(f"[INFO] Phenotype table shape: {pheno_df.shape}")
    print(f"[INFO] Metadata table shape:  {meta_df.shape}")

    # 2) Align by sample ID
    common_samples = [sid for sid in sample_ids if sid in pheno_df.index]
    if len(common_samples) == 0:
        raise RuntimeError(
            "No overlapping samples between geno_core and pheno_core."
        )

    print(
        f"[INFO] Overlapping samples (geno <-> pheno): "
        f"{len(common_samples)} / {len(sample_ids)}"
    )

    sample_idx_map = {sid: i for i, sid in enumerate(sample_ids)}
    geno_idx = [sample_idx_map[sid] for sid in common_samples]

    G_aligned = G[geno_idx, :]
    pheno_aligned = pheno_df.loc[common_samples].copy()
    meta_aligned = meta_df.loc[common_samples].copy()

    # 3) Derive structure clusters from genotype PCs
    N_CLUSTERS = 3  # can be adjusted later if needed
    clusters_all = compute_structure_clusters(
        G=G_aligned,
        n_clusters=N_CLUSTERS,
        random_state=cfg.RANDOM_STATE,
        sample_ids=common_samples,
        core_dir=cfg.CORE_DATA_DIR,
    )
    assert clusters_all.shape[0] == G_aligned.shape[0]

    # 4) Traits to analyse
    traits_available = [t for t in cfg.TRAITS_DEFAULT if t in pheno_aligned.columns]
    if not traits_available:
        raise RuntimeError(
            "None of the traits in TRAITS_DEFAULT are columns of pheno_core.csv"
        )

    print("[INFO] Traits to analyse:", ", ".join(traits_available))
    print()

    lco_per_cluster_results_all: List[Dict[str, object]] = []
    CLUSTER_NAMES = {0: "Oceania", 1: "Americas-SA", 2: "SE_Asia"}
    records: List[Dict[str, object]] = []

    for trait in traits_available:
        print("-" * 72)
        print(f"[TRAIT] {trait}")

        y_full = pheno_aligned[trait].values.astype(float)
        mask = ~np.isnan(y_full)

        X_trait = G_aligned[mask, :]
        y_trait = y_full[mask]
        clusters_trait = clusters_all[mask]

        print(
            f"[INFO] Trait {trait}: {np.sum(mask)} samples with non-missing phenotype"
        )

        # Report cluster sizes in this trait subset
        uniq, counts = np.unique(clusters_trait, return_counts=True)
        print("[INFO] Cluster sizes in trait subset:")
        for u, c in zip(uniq, counts):
            print(f"  - cluster {u}: n = {c}")

        # -------------------------
        # Scheme 1: cluster-balanced K-fold
        # -------------------------
        print("[SCHEME] Cluster-balanced K-fold CV")
        try:
            skf = StratifiedKFold(
                n_splits=cfg.N_SPLITS,
                shuffle=True,
                random_state=cfg.RANDOM_STATE,
            )
            splits_balanced = list(skf.split(X_trait, clusters_trait))
        except ValueError as exc:
            print(
                "[WARN] StratifiedKFold failed "
                f"(likely due to tiny clusters): {exc}"
            )
            print(
                "[WARN] Falling back to non-stratified KFold implemented in "
                "02_gs_kfold_baseline.py would be an option; for now, this scheme "
                "is skipped for this trait."
            )
            splits_balanced = []

        if splits_balanced:
            for scenario_name, use_pc in [
                ("no_pc", False),
                ("pc_corrected", True),
            ]:
                print(f"[SCENARIO] {scenario_name} (cluster-balanced)")
                r_vals = ridge_cv_with_splits(
                    X=X_trait,
                    y=y_trait,
                    splits=splits_balanced,
                    use_pc_correction=use_pc,
                    n_pcs=cfg.N_PCS,
                    random_state=cfg.RANDOM_STATE,
                )
                for fold_idx, r in enumerate(r_vals, start=1):
                    records.append(
                        {
                            "trait": trait,
                            "scheme": "cluster_balanced",
                            "scenario": scenario_name,
                            "fold": fold_idx,
                            "test_group": "mixed",
                            "r": r,
                        }
                    )

        print()

        # -------------------------
        # Scheme 2: leave-cluster-out
        # -------------------------
        print("[SCHEME] Leave-cluster-out CV")
        valid_y = ~pd.isna(y_trait)
        for scenario_name, use_pc in [
            ("no_pc", False),
            ("pc_corrected", True),
        ]:
            print(f"[SCENARIO] {scenario_name} (leave-cluster-out)")
            lco_rows_trait: List[Dict[str, object]] = []

            for held_out_cluster in range(N_CLUSTERS):
                train_mask = (clusters_trait != held_out_cluster) & valid_y
                test_mask = (clusters_trait == held_out_cluster) & valid_y

                n_test = int(test_mask.sum())
                n_train = int(train_mask.sum())

                if n_test < 5 or n_train < 10:
                    print(
                        f"  [WARN] Cluster {held_out_cluster}: insufficient data "
                        f"(train={n_train}, test={n_test}); skipping model fit."
                    )
                    r = np.nan
                    ci_low, ci_high = np.nan, np.nan
                else:
                    train_idx = np.where(train_mask)[0]
                    test_idx = np.where(test_mask)[0]

                    r_list = ridge_cv_with_splits(
                        X=X_trait,
                        y=y_trait,
                        splits=[(train_idx, test_idx)],
                        use_pc_correction=use_pc,
                        n_pcs=cfg.N_PCS,
                        random_state=cfg.RANDOM_STATE,
                    )
                    r = float(r_list[0]) if r_list else np.nan
                    ci_low, ci_high = fisher_z_ci(r, n_test)

                lco_rows_trait.append(
                    {
                        "trait": trait,
                        "held_out_cluster": held_out_cluster,
                        "cluster_name": CLUSTER_NAMES.get(
                            held_out_cluster, f"cluster_{held_out_cluster}"
                        ),
                        "n_test": n_test,
                        "r": r,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "scenario": scenario_name,
                    }
                )
                records.append(
                    {
                        "trait": trait,
                        "scheme": "leave_cluster_out",
                        "scenario": scenario_name,
                        "fold": held_out_cluster + 1,
                        "test_group": f"cluster_{held_out_cluster}",
                        "r": r,
                    }
                )

            rs = [row["r"] for row in lco_rows_trait]
            ns = [row["n_test"] for row in lco_rows_trait]
            mean_r, (mean_ci_low, mean_ci_high) = meta_mean_r(rs, ns)

            lco_rows_trait.append(
                {
                    "trait": trait,
                    "held_out_cluster": "meta-mean",
                    "cluster_name": "Meta-analytic mean",
                    "n_test": int(sum(ns)),
                    "r": mean_r,
                    "ci_low": mean_ci_low,
                    "ci_high": mean_ci_high,
                    "scenario": scenario_name,
                }
            )

            lco_per_cluster_results_all.extend(lco_rows_trait)
            print(
                f"  [INFO] Meta-analytic mean r={mean_r:.3f} "
                f"({mean_ci_low:.3f}, {mean_ci_high:.3f})"
            )

        print()

    # Export leave-cluster-out per-cluster results
    lco_df = pd.DataFrame(lco_per_cluster_results_all)
    table_s14_path = os.path.join(out_dir, "table_s14_lco_per_cluster.csv")
    print(f"[INFO] Saving LCO per-cluster results -> {table_s14_path}")
    lco_df.to_csv(table_s14_path, index=False)

    # 5) Save results
    results_df = pd.DataFrame.from_records(records)
    results_path = os.path.join(out_dir, "cv_structure_results.csv")
    print(f"[INFO] Saving detailed results -> {results_path}")
    results_df.to_csv(results_path, index=False)

    # Summary table
    summary_df = (
        results_df.groupby(["trait", "scheme", "scenario"], dropna=False)["r"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_r", "count": "n_folds"})
    )
    summary_path = os.path.join(out_dir, "cv_structure_summary.csv")
    print(f"[INFO] Saving summary -> {summary_path}")
    summary_df.to_csv(summary_path, index=False)

    print()
    print("[DONE] Structure-aware CV complete.")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
