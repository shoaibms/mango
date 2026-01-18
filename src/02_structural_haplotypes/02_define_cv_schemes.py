#!/usr/bin/env python
r"""
02_define_cv_schemes.py

Define cross-validation schemes for Mango GS – Idea 2.

Inputs (by default):

  From Idea 2 core ML step:
    C:\Users\ms\Desktop\mango\output\idea_2\core_ml\samples.csv
    C:\Users\ms\Desktop\mango\output\idea_2\core_ml\sample_metadata_ml.csv

  From Idea 1 (optional, for ancestry-based CV if available):
    C:\Users\ms\Desktop\mango\output\idea_1\core_data\pheno_core.csv

Outputs (by default) to:
  C:\Users\ms\Desktop\mango\output\idea_2\cv_design\

  - cv_random_k5.csv
      columns: sample_id, fold   (fold = 1..5, stratified by cluster_kmeans)

  - cv_cluster_k3.csv
      columns: sample_id, fold   (fold = cluster ID 1..K; leave-one-cluster-out)

  - cv_ancestry.csv  (ONLY if ancestry-like column is detected)
      columns: sample_id, fold   (fold = 1..A, A = #ancestry groups)
"""

import argparse
import os
from typing import Optional, List, Tuple

import numpy as np

try:
    import pandas as pd
except ImportError as e:
    raise SystemExit(
        "pandas is required. Install it with:\n\n  pip install pandas\n"
    ) from e

try:
    from sklearn.model_selection import StratifiedKFold, KFold
except ImportError as e:
    raise SystemExit(
        "scikit-learn is required. Install it with:\n\n  pip install scikit-learn\n"
    ) from e


# =========================
# DEFAULT PATHS / PARAMS
# =========================

DEFAULT_SAMPLES = r"C:\Users\ms\Desktop\mango\output\idea_2\core_ml\samples.csv"
DEFAULT_META = r"C:\Users\ms\Desktop\mango\output\idea_2\core_ml\sample_metadata_ml.csv"
DEFAULT_PHENO_CORE = r"C:\Users\ms\Desktop\mango\output\idea_1\core_data\pheno_core.csv"
DEFAULT_OUTDIR = r"C:\Users\ms\Desktop\mango\output\idea_2\cv_design"

DEFAULT_N_RANDOM_FOLDS = 5
RANDOM_STATE = 42


# =========================
# UTILS
# =========================

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_samples(samples_path: str) -> List[str]:
    """
    Load samples.csv written by 01_prepare_idea2_datasets.py

    Expected format:
      sample_id
      line1
      line2
      ...
    """
    if not os.path.isfile(samples_path):
        raise FileNotFoundError(f"samples.csv not found: {samples_path}")

    df = pd.read_csv(samples_path)
    if "sample_id" not in df.columns:
        raise RuntimeError(
            f"Expected a 'sample_id' column in {samples_path}, got columns={list(df.columns)}"
        )

    sample_ids = df["sample_id"].astype(str).tolist()
    return sample_ids


def load_meta(meta_path: str) -> pd.DataFrame:
    """
    Load sample_metadata_ml.csv written by 01_prepare_idea2_datasets.py

    That file was written with:
        meta_df.to_csv(meta_path, index_label="sample_id")

    So we expect either:
      - a 'sample_id' column we can set as index, or
      - the first column already being the index.
    """
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"sample_metadata_ml.csv not found: {meta_path}")

    df = pd.read_csv(meta_path)

    if "sample_id" in df.columns:
        df = df.set_index("sample_id")
    else:
        # assume first column is sample_id index
        first = df.columns[0]
        df = df.set_index(first)
        print(f"[WARN] No explicit 'sample_id' column in metadata; using '{first}' as index.")

    df.index = df.index.astype(str)
    return df


def detect_ancestry_column(pheno_core_path: str) -> Tuple[Optional[pd.Series], Optional[str]]:
    """
    Try to detect an ancestry/population column in pheno_core.csv.

    Heuristics:
      - load pheno_core as generic DataFrame.
      - set index to sample_id or ID if present (like in 01).
      - scan non-numeric columns for a reasonable #levels (2–10).
      - preference for columns whose name contains typical ancestry/pop strings.

    Returns:
      (series, colname) if found, else (None, None).
    """
    if not os.path.isfile(pheno_core_path):
        print(f"[INFO] pheno_core not found ({pheno_core_path}); skipping ancestry-based CV.")
        return None, None

    print(f"[INFO] Attempting to detect ancestry column from: {pheno_core_path}")
    df = pd.read_csv(pheno_core_path)

    # set index to sample_id or ID if present
    cols_lower = {c.lower(): c for c in df.columns}
    if "sample_id" in cols_lower:
        df = df.set_index(cols_lower["sample_id"])
    elif "id" in cols_lower:
        df = df.set_index(cols_lower["id"])
    elif df.columns[0].lower().startswith("unnamed"):
        df = pd.read_csv(pheno_core_path, index_col=0)
    else:
        df = df.set_index(df.columns[0])

    df.index = df.index.astype(str)

    # non-numeric columns
    non_num = df.select_dtypes(exclude=[np.number])
    if non_num.shape[1] == 0:
        print("[INFO] No non-numeric columns found in pheno_core; cannot infer ancestry.")
        return None, None

    # candidate columns with small-ish number of unique values
    candidates = []
    for col in non_num.columns:
        nunique = non_num[col].nunique(dropna=True)
        if 1 < nunique <= 10:
            candidates.append((col, nunique))

    if not candidates:
        print("[INFO] Non-numeric columns exist but none look like a small number of groups; skipping ancestry.")
        return None, None

    # Rank candidates: prefer names containing ancestry/pop keywords
    priority_keywords = ["ancestry", "pop", "population", "subpop", "group", "panel"]
    scored = []
    for col, nunique in candidates:
        name_lower = col.lower()
        score = 0
        for kw in priority_keywords:
            if kw in name_lower:
                score += 1
        scored.append((score, col, nunique))

    # pick best by score, then by fewest levels
    scored.sort(key=lambda x: (-x[0], x[2]))
    best_score, best_col, best_nuniq = scored[0]

    if best_score == 0:
        # no keyword match, be conservative
        print("[INFO] Found possible group columns, but none match ancestry-like keywords clearly.")
        print("       Skipping ancestry-based CV for now.")
        return None, None

    series = non_num[best_col].copy()
    print(f"[INFO] Using column '{best_col}' as ancestry/group (levels={best_nuniq}).")
    return series, best_col


# =========================
# CV SCHEME BUILDERS
# =========================

def make_random_cv(
    sample_ids: List[str],
    cluster_labels: Optional[np.ndarray],
    n_folds: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Build random K-fold CV (optionally stratified by cluster labels).

    Returns DataFrame:
      sample_id, fold (1..n_folds)
    """
    n = len(sample_ids)
    idx = np.arange(n)

    if cluster_labels is not None:
        print("[INFO] Using StratifiedKFold for random CV (stratified by cluster_kmeans).")
        splitter = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=random_state,
        )
        y = cluster_labels
    else:
        print("[INFO] Using plain KFold for random CV (no stratification).")
        splitter = KFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=random_state,
        )
        y = None

    fold_ids = np.zeros(n, dtype=int)
    for fold_idx, (_, test_index) in enumerate(
        splitter.split(idx, y), start=1
    ):
        fold_ids[test_index] = fold_idx

    df = pd.DataFrame(
        {"sample_id": sample_ids, "fold": fold_ids.astype(int)}
    )
    return df


def make_cluster_cv(
    sample_ids: List[str],
    cluster_labels: np.ndarray,
) -> pd.DataFrame:
    """
    Build cluster-based CV: fold = cluster (1..K).

    This is effectively leave-one-k-means-cluster-out.
    """
    if cluster_labels.ndim != 1:
        raise ValueError("cluster_labels must be a 1D array")

    # ensure deterministic mapping: sort unique labels, map to 1..K
    unique_labels = np.unique(cluster_labels)
    label_to_fold = {lab: i + 1 for i, lab in enumerate(unique_labels)}

    fold_ids = np.array([label_to_fold[lab] for lab in cluster_labels], dtype=int)
    df = pd.DataFrame(
        {"sample_id": sample_ids, "fold": fold_ids}
    )
    print(f"[INFO] Cluster CV: found {len(unique_labels)} clusters -> folds 1..{len(unique_labels)}")
    return df


def make_ancestry_cv(
    sample_ids: List[str],
    ancestry_series: pd.Series,
) -> Optional[pd.DataFrame]:
    """
    Build ancestry-based CV: fold = ancestry group (1..A).

    This is similar to cluster CV but uses user-defined ancestry/pop groups.

    Returns:
      DataFrame (sample_id, fold) or None if no overlap.
    """
    # align ancestry_series to sample_ids
    ancestry_series = ancestry_series.copy()
    ancestry_series.index = ancestry_series.index.astype(str)

    missing = [sid for sid in sample_ids if sid not in ancestry_series.index]
    if len(missing) == len(sample_ids):
        print("[WARN] No overlap between samples.csv and pheno_core ancestry index; skipping ancestry CV.")
        return None
    if missing:
        print(f"[WARN] {len(missing)} samples from samples.csv missing in pheno_core; they will be dropped from ancestry CV.")

    present_ids = [sid for sid in sample_ids if sid in ancestry_series.index]
    groups = ancestry_series.loc[present_ids].astype(str)

    unique_groups = sorted(groups.unique())
    if len(unique_groups) < 2:
        print("[INFO] Ancestry/group column found but has <2 unique levels; skipping ancestry CV.")
        return None

    group_to_fold = {g: i + 1 for i, g in enumerate(unique_groups)}
    fold_ids = [group_to_fold[g] for g in groups]

    df = pd.DataFrame({"sample_id": present_ids, "fold": fold_ids})
    print(f"[INFO] Ancestry CV: groups={unique_groups} -> folds 1..{len(unique_groups)}")
    return df


# =========================
# MAIN
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Define CV schemes for Mango GS Idea 2."
    )
    parser.add_argument(
        "--samples",
        type=str,
        default=DEFAULT_SAMPLES,
        help=f"Path to samples.csv (default: {DEFAULT_SAMPLES})",
    )
    parser.add_argument(
        "--meta",
        type=str,
        default=DEFAULT_META,
        help=f"Path to sample_metadata_ml.csv (default: {DEFAULT_META})",
    )
    parser.add_argument(
        "--pheno-core",
        type=str,
        default=DEFAULT_PHENO_CORE,
        help=f"Path to pheno_core.csv for optional ancestry-based CV (default: {DEFAULT_PHENO_CORE})",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=DEFAULT_OUTDIR,
        help=f"Output directory for CV design files (default: {DEFAULT_OUTDIR})",
    )
    parser.add_argument(
        "--n-random-folds",
        type=int,
        default=DEFAULT_N_RANDOM_FOLDS,
        help=f"Number of folds for random CV (default: {DEFAULT_N_RANDOM_FOLDS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_STATE,
        help=f"Random seed for CV shuffling (default: {RANDOM_STATE})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 72)
    print("Mango GS – Idea 2: Define CV schemes")
    print("=" * 72)
    print(f"[INFO] samples.csv:   {args.samples}")
    print(f"[INFO] metadata:      {args.meta}")
    print(f"[INFO] pheno_core:    {args.pheno_core}")
    print(f"[INFO] outdir:        {args.outdir}")
    print(f"[INFO] random folds:  {args.n_random_folds}")
    print(f"[INFO] seed:          {args.seed}")
    print("")

    safe_mkdir(args.outdir)

    # 1) Load samples and metadata
    sample_ids = load_samples(args.samples)
    meta_df = load_meta(args.meta)

    # Align metadata to samples
    missing_meta = [sid for sid in sample_ids if sid not in meta_df.index]
    if missing_meta:
        raise RuntimeError(
            f"{len(missing_meta)} samples from samples.csv missing in sample_metadata_ml.csv.\n"
            f"Example missing IDs: {missing_meta[:5]}"
        )
    meta_df = meta_df.loc[sample_ids]

    # 2) Random CV (stratified by cluster if available)
    if "cluster_kmeans" in meta_df.columns:
        cluster_labels = meta_df["cluster_kmeans"].to_numpy()
    else:
        print("[WARN] 'cluster_kmeans' column not found in metadata; random CV will not be stratified.")
        cluster_labels = None

    random_cv_df = make_random_cv(
        sample_ids=sample_ids,
        cluster_labels=cluster_labels,
        n_folds=args.n_random_folds,
        random_state=args.seed,
    )
    random_cv_path = os.path.join(args.outdir, f"cv_random_k{args.n_random_folds}.csv")
    random_cv_df.to_csv(random_cv_path, index=False)
    print(f"[SAVE] Random CV design -> {random_cv_path}")

    # 3) Cluster CV (leave-one-k-means-cluster-out)
    if cluster_labels is not None:
        cluster_cv_df = make_cluster_cv(sample_ids, cluster_labels)
        cluster_cv_path = os.path.join(args.outdir, "cv_cluster_kmeans.csv")
        cluster_cv_df.to_csv(cluster_cv_path, index=False)
        print(f"[SAVE] Cluster CV design -> {cluster_cv_path}")
    else:
        print("[INFO] Skipping cluster-based CV because 'cluster_kmeans' is not available.")

    # 4) Ancestry-based CV (optional)
    ancestry_series, ancestry_col = detect_ancestry_column(args.pheno_core)
    if ancestry_series is not None:
        ancestry_cv_df = make_ancestry_cv(sample_ids, ancestry_series)
        if ancestry_cv_df is not None:
            ancestry_cv_path = os.path.join(args.outdir, "cv_ancestry.csv")
            ancestry_cv_df.to_csv(ancestry_cv_path, index=False)
            print(f"[SAVE] Ancestry CV design -> {ancestry_cv_path}")

    print("")
    print("[OK] CV schemes for Idea 2 created successfully.")


if __name__ == "__main__":
    main()
