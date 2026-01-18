# -*- coding: utf-8 -*-
r"""
15_binn_train.py

Mango GS – Idea 3
=================
Train the Biologically Informed Neural Network (BINN) using:
  * geno_core.npz + pheno_core.csv (Idea 1)
  * binn_snp_map.npz (from 10_binn_build_maps.py)

Architecture (redefined here, no imports from 11_binn_model.py):
  SNPs → [masked] SNP→Gene layer → Traits (multi-output linear head)

Key features:
  * Multi-trait training (FBC, FF, AFW, TSS, TC) in a single model
  * Canonical trait order enforced for outputs
  * 5-fold random CV
  * Missing phenotypes handled via sample weights
  * Saves per-fold gene→trait kernels for later interpretation

Outputs (under OUT_BINN_DIR):
  - binn_cv_results.csv      (one row per fold × trait)
  - binn_cv_summary.csv      (aggregated per trait)
  - models/binn_fold<k>.keras
  - weights/binn_gene_weights_fold<k>.npz
  - logs/binn_fold<k>_meta.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

try:
    import tensorflow as tf
except ImportError as e:
    raise SystemExit(
        "tensorflow is required. Install it with: pip install tensorflow"
    ) from e


# =========================
# CONFIG
# =========================

# Project root (assumed layout: ROOT_DIR/idea_3/this_file.py and ROOT_DIR/output/...)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Core data from Idea 1
GENO_CORE_PATH = r"C:\Users\ms\Desktop\mango\output\idea_1\core_data\geno_core.npz"
PHENO_CORE_PATH = r"C:\Users\ms\Desktop\mango\output\idea_1\core_data\pheno_core.csv"

# BINN map (from 10_binn_build_maps.py)
BINN_SNP_MAP_PATH = r"C:\Users\ms\Desktop\mango\output\idea_3\binn_maps\binn_snp_map.npz"

# Output root for Idea 3 BINN training
OUT_ROOT = r"C:\Users\ms\Desktop\mango\output\idea_3"
OUT_BINN_DIR = os.path.join(OUT_ROOT, "binn_training")

# Optional precomputed folds (for cross-method comparability)
CV_FOLDS_PATH = os.path.join(ROOT_DIR, "output", "idea_3", "tensors", "cv_folds.json")

# Canonical trait order for BINN outputs
TRAIT_ORDER: List[str] = ["FBC", "FF", "AFW", "TSS", "TC"]

# CV + training hyperparameters
N_FOLDS = 5
RANDOM_STATE = 123
BATCH_SIZE = 32
MAX_EPOCHS = 2000
EARLY_STOPPING_PATIENCE = 100
LEARNING_RATE = 1e-3

# Regularisation / layer hyperparams
DEFAULT_L2 = 1e-4
DEFAULT_GENE_ACTIVATION = "relu"
DEFAULT_GENE_DROPOUT = 0.0


# =========================
# UTILITIES
# =========================

def load_cv_folds(cv_folds_path: str, expected_n: int):

    import json, os

    if not os.path.exists(cv_folds_path):

        return None

    with open(cv_folds_path, "r", encoding="utf-8") as f:

        folds = json.load(f)

    # sanity

    all_test = sorted([i for fold in folds for i in fold["test_idx"]])

    if len(all_test) != expected_n or len(set(all_test)) != expected_n:

        raise ValueError("cv_folds.json test indices do not form a partition of samples.")

    return folds


def load_cv_design_fold_ids(cv_path: str, sample_ids: np.ndarray) -> Tuple[np.ndarray, str]:
    df = pd.read_csv(cv_path)
    if not {"sample_id", "fold"}.issubset(df.columns):
        raise ValueError(f"cv_design must have columns sample_id, fold. Found: {list(df.columns)}")
    scheme_name = os.path.splitext(os.path.basename(cv_path))[0]
    fold_map = df.set_index("sample_id")["fold"]

    missing = [sid for sid in sample_ids.tolist() if sid not in fold_map.index]
    if missing:
        raise ValueError(f"{len(missing)} sample_ids not found in cv_design. Example: {missing[:5]}")

    fold_ids = np.array([int(fold_map.loc[str(sid)]) for sid in sample_ids.astype(str)], dtype=int)
    return fold_ids, scheme_name


def make_splits_from_fold_ids(fold_ids: np.ndarray) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    uniq = sorted(set(int(x) for x in np.unique(fold_ids)))
    splits = []
    for f in uniq:
        val_idx = np.where(fold_ids == f)[0]
        train_idx = np.where(fold_ids != f)[0]
        if len(val_idx) == 0 or len(train_idx) == 0:
            continue
        splits.append((f, train_idx, val_idx))
    return splits


def safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_geno_core(path: str) -> Dict[str, np.ndarray]:
    """
    Load geno_core.npz and return:
      - X          : genotype matrix (n_samples x n_snps)
      - sample_ids : sample identifiers (length n_samples)
      - snp_ids    : variant IDs (length n_snps)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"geno_core file not found:\n  {path}")

    print(f"[INFO] Loading geno_core from:\n  {path}")
    npz = np.load(path, allow_pickle=True)
    keys = list(npz.files)
    print(f"[INFO] geno_core keys: {keys}")

    geno_key_candidates = ["G", "X", "geno", "geno_matrix"]
    sample_key_candidates = ["sample_ids", "samples", "lines"]
    snp_key_candidates = ["variant_ids", "snp_ids", "markers"]

    def _pick(candidates, label):
        for c in candidates:
            if c in npz.files:
                return c
        raise KeyError(
            f"Could not find a key for {label}. "
            f"Tried {candidates}, available keys: {keys}"
        )

    geno_key = _pick(geno_key_candidates, "genotype matrix")
    sample_key = _pick(sample_key_candidates, "sample IDs")
    snp_key = _pick(snp_key_candidates, "SNP IDs")

    X = npz[geno_key]
    sample_ids = npz[sample_key]
    snp_ids = npz[snp_key]

    if X.ndim != 2:
        raise ValueError(f"Genotype matrix {geno_key} must be 2D, got shape {X.shape}")

    n_samples, n_snps = X.shape
    print(f"[INFO] Genotype matrix shape: {X.shape} (samples x SNPs)")

    if len(sample_ids) != n_samples:
        raise ValueError(
            f"sample_ids length ({len(sample_ids)}) does not match "
            f"number of rows in X ({n_samples})"
        )
    if len(snp_ids) != n_snps:
        raise ValueError(
            f"snp_ids length ({len(snp_ids)}) does not match "
            f"number of columns in X ({n_snps})"
        )

    return {
        "X": X.astype(np.float32),
        "sample_ids": np.array(sample_ids).astype(str),
        "snp_ids": np.array(snp_ids).astype(str),
    }


def load_binn_snp_map(path: str = BINN_SNP_MAP_PATH) -> Dict[str, np.ndarray]:
    """
    Load binn_snp_map.npz from 10_binn_build_maps.py.

    Expected keys:
      - snp_ids
      - snp_core_index
      - snp_chr
      - snp_pos
      - snp_gene_index
      - snp_trait_matrix
      - trait_names
      - gene_ids
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"BINN SNP map not found:\n  {path}\n"
            "Run 10_binn_build_maps.py first."
        )

    print(f"[INFO] Loading BINN SNP map from:\n  {path}")
    npz = np.load(path, allow_pickle=True)
    keys = list(npz.files)
    print(f"[INFO] binn_snp_map keys: {keys}")

    required = [
        "snp_ids",
        "snp_core_index",
        "snp_chr",
        "snp_pos",
        "snp_gene_index",
        "snp_trait_matrix",
        "trait_names",
        "gene_ids",
    ]
    missing = [k for k in required if k not in keys]
    if missing:
        raise KeyError(
            f"binn_snp_map.npz missing keys: {missing}\n"
            f"Available keys: {keys}"
        )

    data = {k: npz[k] for k in required}
    data["snp_ids"] = np.asarray(data["snp_ids"]).astype(str)
    data["gene_ids"] = np.asarray(data["gene_ids"]).astype(str)
    data["snp_core_index"] = np.asarray(data["snp_core_index"]).astype(int)
    data["snp_chr"] = np.asarray(data["snp_chr"]).astype(str)
    data["snp_pos"] = np.asarray(data["snp_pos"]).astype(int)
    data["snp_gene_index"] = np.asarray(data["snp_gene_index"]).astype(int)
    data["trait_names"] = np.asarray(data["trait_names"]).astype(str)
    data["snp_trait_matrix"] = np.asarray(data["snp_trait_matrix"]).astype(np.int8)
    return data


def build_snp_gene_mask(
    snp_gene_index: np.ndarray,
    n_genes: int,
    normalise: bool = True,
) -> np.ndarray:
    """
    Build a (n_snps, n_genes) mask matrix where:
      mask[i, j] = weight_factor if SNP i maps to gene j, else 0

    If normalise=True, weight_factor is 1 / (# SNPs mapped to that gene),
    so each gene receives the *average* SNP signal.
    """
    snp_gene_index = np.asarray(snp_gene_index, dtype=int)
    n_snps = snp_gene_index.shape[0]
    mask = np.zeros((n_snps, n_genes), dtype=np.float32)

    valid = snp_gene_index >= 0
    if not np.any(valid):
        raise ValueError("No valid gene indices in snp_gene_index.")

    counts = np.bincount(snp_gene_index[valid], minlength=n_genes)
    if normalise:
        factors = np.zeros_like(counts, dtype=np.float32)
        nonzero = counts > 0
        factors[nonzero] = 1.0 / counts[nonzero].astype(np.float32)
    else:
        factors = np.ones_like(counts, dtype=np.float32)

    for snp_idx, gene_idx in enumerate(snp_gene_index):
        if gene_idx < 0 or gene_idx >= n_genes:
            continue
        mask[snp_idx, gene_idx] = factors[gene_idx]

    return mask


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Pearson correlation, guarding against NaNs / zero variance."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 3:
        return np.nan
    yt = y_true[mask]
    yp = y_pred[mask]
    if np.allclose(yt, yt[0]) or np.allclose(yp, yp[0]):
        return np.nan
    r = np.corrcoef(yt, yp)[0, 1]
    return float(r)


def standardise_X(
    X_train: np.ndarray, X_val: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score standardisation of X using training statistics.
    Returns: X_train_std, X_val_std, mean, std
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-8] = 1.0

    X_train_std = (X_train - mean) / std
    X_val_std = (X_val - mean) / std
    return X_train_std.astype(np.float32), X_val_std.astype(np.float32), mean, std


def standardise_Y_with_missing(
    Y_train: np.ndarray,
    Y_val: np.ndarray,
    W_train: np.ndarray,
    W_val: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardise Y per trait using only non-missing training values
    (where W_train == 1). Missing entries are filled with 0.0 after
    standardisation so they don't affect the loss.

    Returns:
      Y_train_std, Y_val_std, mean (vector), std (vector)
    """
    Y_train = np.asarray(Y_train, dtype=float)
    Y_val = np.asarray(Y_val, dtype=float)
    W_train = np.asarray(W_train, dtype=float)
    W_val = np.asarray(W_val, dtype=float)

    n_traits = Y_train.shape[1]
    mean = np.zeros(n_traits, dtype=float)
    std = np.ones(n_traits, dtype=float)

    Y_train_std = np.zeros_like(Y_train, dtype=float)
    Y_val_std = np.zeros_like(Y_val, dtype=float)

    for j in range(n_traits):
        mask_train = W_train[:, j] > 0
        if mask_train.sum() == 0:
            mean[j] = 0.0
            std[j] = 1.0
            continue

        yj_train = Y_train[mask_train, j]
        m = float(np.mean(yj_train))
        s = float(np.std(yj_train))
        if s < 1e-8:
            s = 1.0

        mean[j] = m
        std[j] = s

        Y_train_std[:, j] = (Y_train[:, j] - m) / s
        Y_val_std[:, j] = (Y_val[:, j] - m) / s

    Y_train_std[W_train == 0] = 0.0
    Y_val_std[W_val == 0] = 0.0

    return (
        Y_train_std.astype(np.float32),
        Y_val_std.astype(np.float32),
        mean,
        std,
    )


# =========================
# LAYERS + MODEL
# =========================

class SnpToGeneLayer(tf.keras.layers.Layer):
    """
    Masked SNP→Gene layer.

    Given input X of shape (batch, n_snps), computes gene activations:
        gene = activation( X @ (W ⊙ mask) + b )

    where:
      - W is a learnable (n_snps, n_genes) matrix,
      - mask is a fixed (n_snps, n_genes) matrix encoding allowed edges,
      - b is gene bias.
    """

    def __init__(
        self,
        snp_gene_mask: np.ndarray,
        activation: str = DEFAULT_GENE_ACTIVATION,
        l2_reg: float = DEFAULT_L2,
        dropout_rate: float = DEFAULT_GENE_DROPOUT,
        name: str = "snp_to_gene",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        mask = np.asarray(snp_gene_mask, dtype=np.float32)
        if mask.ndim != 2:
            raise ValueError(
                f"snp_gene_mask must be 2D (n_snps, n_genes), got shape {mask.shape}"
            )
        self.n_snps, self.n_genes = mask.shape
        self.snp_gene_mask = tf.constant(mask, dtype=tf.float32, name="snp_gene_mask")

        self.activation_name = activation
        self.l2_reg = float(l2_reg)
        self.dropout_rate = float(dropout_rate)

        self._activation = tf.keras.activations.get(activation)
        self._kernel_regularizer = (
            tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None
        )
        self._dropout = (
            tf.keras.layers.Dropout(self.dropout_rate)
            if self.dropout_rate > 0
            else None
        )

    def build(self, input_shape):
        if input_shape[-1] != self.n_snps:
            raise ValueError(
                f"Input last dimension {input_shape[-1]} does not match "
                f"mask n_snps {self.n_snps}."
            )

        self.W = self.add_weight(
            name="kernel",
            shape=(self.n_snps, self.n_genes),
            initializer="glorot_uniform",
            regularizer=self._kernel_regularizer,
            trainable=True,
        )
        self.b = self.add_weight(
            name="bias",
            shape=(self.n_genes,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        masked_W = self.W * self.snp_gene_mask  # (n_snps, n_genes)
        gene_pre = tf.linalg.matmul(inputs, masked_W) + self.b
        gene_act = self._activation(gene_pre)
        if self._dropout is not None:
            gene_act = self._dropout(gene_act, training=training)
        return gene_act

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "activation": self.activation_name,
                "l2_reg": self.l2_reg,
                "dropout_rate": self.dropout_rate,
                # mask is not serialised; this layer is only used in-training
            }
        )
        return cfg


def build_binn_model(
    binn_map: Dict[str, np.ndarray],
    trait_order: List[str] | None = None,
    gene_activation: str = DEFAULT_GENE_ACTIVATION,
    gene_dropout: float = DEFAULT_GENE_DROPOUT,
    l2_reg: float = DEFAULT_L2,
) -> Tuple[tf.keras.Model, Dict[str, object]]:
    """
    Build a BINN model from the SNP→gene map.

    Parameters
    ----------
    binn_map : dict
        Output of load_binn_snp_map().
    trait_order : list of str or None
        Desired output trait order. Only traits present in binn_map['trait_names']
        are used, in this order.
    gene_activation : str
        Activation for the gene layer.
    gene_dropout : float
        Dropout rate on gene activations.
    l2_reg : float
        L2 regularisation strength.

    Returns
    -------
    model : tf.keras.Model
    meta : dict with n_snps, n_genes, trait_names_out, etc.
    """
    snp_ids = binn_map["snp_ids"]
    gene_ids = binn_map["gene_ids"]
    snp_gene_index = binn_map["snp_gene_index"]
    trait_names_map = list(binn_map["trait_names"])

    n_snps = snp_ids.shape[0]
    n_genes = gene_ids.shape[0]
    n_traits_map = len(trait_names_map)

    print(f"[INFO] BINN map: n_snps={n_snps}, n_genes={n_genes}, n_traits_in_map={n_traits_map}")

    if trait_order is None:
        trait_order = TRAIT_ORDER

    trait_names_out: List[str] = [t for t in trait_order if t in trait_names_map]
    if not trait_names_out:
        raise ValueError(
            f"None of the desired traits {trait_order} appear in binn_snp_map trait_names={trait_names_map}"
        )

    trait_index_map: Dict[str, int] = {t: i for i, t in enumerate(trait_names_map)}
    trait_index_map_to_out: List[int] = [trait_index_map[t] for t in trait_names_out]

    print(f"[INFO] Trait names in map: {trait_names_map}")
    print(f"[INFO] Trait output order: {trait_names_out}")
    print(f"[INFO] Trait indices (map→out): {trait_index_map_to_out}")

    snp_gene_mask = build_snp_gene_mask(
        snp_gene_index=snp_gene_index,
        n_genes=n_genes,
        normalise=True,
    )

    inputs = tf.keras.Input(shape=(n_snps,), name="snp_input")

    gene_layer = SnpToGeneLayer(
        snp_gene_mask=snp_gene_mask,
        activation=gene_activation,
        l2_reg=l2_reg,
        dropout_rate=gene_dropout,
        name="snp_to_gene",
    )(inputs)

    # Gene → trait logits (keep this for extracting gene weights later)
    trait_logits = tf.keras.layers.Dense(
        units=len(trait_names_out),
        activation=None,
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None,
        name="trait_output",
    )(gene_layer)

    # Reshape to (batch, n_traits, 1) so we can use 2D sample_weight (batch, n_traits)
    output_layer = tf.keras.layers.Reshape(
        (len(trait_names_out), 1),
        name="trait_output_reshaped",
    )(trait_logits)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=output_layer,
        name="MangoBINN_SNP_Gene_Traits",
    )

    meta: Dict[str, object] = {
        "n_snps": n_snps,
        "n_genes": n_genes,
        "n_traits_map": n_traits_map,
        "n_traits_out": len(trait_names_out),
        "trait_names_in_map": trait_names_map,
        "trait_names_out": trait_names_out,
        "trait_index_map_to_out": trait_index_map_to_out,
        "snp_ids": snp_ids,
        "gene_ids": gene_ids,
    }
    return model, meta


# =========================
# MAIN TRAINING LOGIC
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv_design", type=str, default="", help="Optional Idea2 cv_design CSV (sample_id, fold).")
    args = parser.parse_args()

    set_global_seed(RANDOM_STATE)

    print("=" * 72)
    print("Mango GS – Idea 3: BINN training (12_binn_train.py)")
    print("=" * 72)
    print(f"[INFO] geno_core:     {GENO_CORE_PATH}")
    print(f"[INFO] pheno_core:    {PHENO_CORE_PATH}")
    print(f"[INFO] BINN SNP map:  {BINN_SNP_MAP_PATH}")
    print(f"[INFO] Output dir:    {OUT_BINN_DIR}")
    print("")

    safe_makedirs(OUT_BINN_DIR)
    models_dir = os.path.join(OUT_BINN_DIR, "models")
    logs_dir = os.path.join(OUT_BINN_DIR, "logs")
    weights_dir = os.path.join(OUT_BINN_DIR, "weights")
    safe_makedirs(models_dir)
    safe_makedirs(logs_dir)
    safe_makedirs(weights_dir)

    # 1. Load geno_core & BINN map
    geno = load_geno_core(GENO_CORE_PATH)
    X_core = geno["X"]
    sample_ids_core = geno["sample_ids"]

    binn_map = load_binn_snp_map(BINN_SNP_MAP_PATH)
    snp_core_index = binn_map["snp_core_index"]
    gene_ids = binn_map["gene_ids"]

    model_template, meta_template = build_binn_model(
        binn_map=binn_map,
        trait_order=TRAIT_ORDER,
        gene_activation=DEFAULT_GENE_ACTIVATION,
        gene_dropout=DEFAULT_GENE_DROPOUT,
        l2_reg=DEFAULT_L2,
    )

    trait_names_out: List[str] = meta_template["trait_names_out"]
    n_traits_out = len(trait_names_out)

    print(f"[INFO] BINN outputs (traits): {trait_names_out}")
    print(f"[INFO] X_core shape:          {X_core.shape} (samples x SNPs_core)")
    print(f"[INFO] SNPs used by BINN:     {snp_core_index.shape[0]}")
    print("")

    X_binn = X_core[:, snp_core_index]
    n_samples, n_snps_binn = X_binn.shape
    print(f"[INFO] X_binn shape:          {X_binn.shape} (samples x SNPs_for_BINN)")

    # 2. Load pheno_core and align to sample_ids_core
    if not os.path.exists(PHENO_CORE_PATH):
        raise FileNotFoundError(f"pheno_core file not found:\n  {PHENO_CORE_PATH}")

    pheno_df = pd.read_csv(PHENO_CORE_PATH, index_col=0)
    pheno_df.index = pheno_df.index.astype(str)
    pheno_df = pheno_df.reindex(sample_ids_core)

    print(f"[INFO] pheno_core columns: {list(pheno_df.columns)}")

    missing_traits = [t for t in trait_names_out if t not in pheno_df.columns]
    if missing_traits:
        raise KeyError(
            f"The following BINN traits are missing from pheno_core.csv: {missing_traits}"
        )

    Y_full = pheno_df[trait_names_out].to_numpy(dtype=float)  # (n_samples, n_traits_out)

    # 3. Sample weights for missing phenotypes
    W_full = (~np.isnan(Y_full)).astype(np.float32)
    Y_full_no_nan = Y_full.copy()
    Y_full_no_nan[np.isnan(Y_full_no_nan)] = 0.0

    print("[INFO] Non-missing phenotypes per trait:")
    for j, t in enumerate(trait_names_out):
        n_nonmissing = int(W_full[:, j].sum())
        print(f"  - {t}: {n_nonmissing} / {n_samples}")
    print("")

    # 4. K-fold CV
    results_records: List[Dict[str, object]] = []

    if args.cv_design:
        fold_ids, scheme_name = load_cv_design_fold_ids(args.cv_design, sample_ids_core)
        splits = make_splits_from_fold_ids(fold_ids)
        n_folds_run = len(splits)
        print(f"[INFO] Using external CV design: {args.cv_design}")
        print(f"[INFO] CV scheme_name: {scheme_name} (n_folds={n_folds_run})")
        folds = [
            {"fold_id": int(f), "train_idx": tr.tolist(), "test_idx": va.tolist()}
            for (f, tr, va) in splits
        ]
    else:
        folds = load_cv_folds(CV_FOLDS_PATH, expected_n=n_samples)
        if folds is None:
            # fallback to existing behaviour (do NOT delete it)
            kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
            folds = [
                {"fold_id": int(fid), "train_idx": tr.tolist(), "test_idx": te.tolist()}
                for fid, (tr, te) in enumerate(kf.split(np.arange(n_samples)), start=1)
            ]

    n_folds_run = len(folds)

    for fold in folds:
        fold_id = fold["fold_id"]
        train_idx = np.array(fold["train_idx"], dtype=int)
        test_idx = np.array(fold["test_idx"], dtype=int)
        fold_idx = fold_id
        val_idx = test_idx
        print("-" * 72)
        print(f"[FOLD {fold_idx}/{n_folds_run}] train={len(train_idx)}, val={len(val_idx)}")

        X_train = X_binn[train_idx, :]
        X_val = X_binn[val_idx, :]

        Y_train = Y_full_no_nan[train_idx, :]
        Y_val = Y_full_no_nan[val_idx, :]

        W_train = W_full[train_idx, :]
        W_val = W_full[val_idx, :]

        # Standardise X
        X_train_std, X_val_std, x_mean, x_std = standardise_X(X_train, X_val)

        # Standardise Y
        Y_train_std, Y_val_std, y_mean, y_std = standardise_Y_with_missing(
            Y_train, Y_val, W_train, W_val
        )

        # Expand Y to (batch, n_traits, 1) to match model output
        Y_train_std_exp = Y_train_std[..., np.newaxis]  # (n_train, 5, 1)
        Y_val_std_exp = Y_val_std[..., np.newaxis]      # (n_val, 5, 1)

        # Build a fresh model for this fold
        model_fold, meta_fold = build_binn_model(
            binn_map=binn_map,
            trait_order=TRAIT_ORDER,
            gene_activation=DEFAULT_GENE_ACTIVATION,
            gene_dropout=DEFAULT_GENE_DROPOUT,
            l2_reg=DEFAULT_L2,
        )

        opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        model_fold.compile(
            optimizer=opt,
            loss="mse",
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1,
            )
        ]

        history = model_fold.fit(
            X_train_std,
            Y_train_std_exp,
            sample_weight=W_train[..., np.newaxis],            # (batch, 5, 1)
            validation_data=(X_val_std, Y_val_std_exp, W_val[..., np.newaxis]),
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1,
            callbacks=callbacks,
        )

        best_val_loss = float(np.min(history.history["val_loss"]))
        n_epochs_run = len(history.history["loss"])
        print(f"[FOLD {fold_idx}] best val_loss={best_val_loss:.4f} after {n_epochs_run} epochs")

        # Predict on validation set (back to original scale)
        Y_val_pred_std = model_fold.predict(X_val_std, verbose=0)  # (n_val, 5, 1)

        # Squeeze back to (n_val, 5) for metrics / un-standardisation
        if Y_val_pred_std.ndim == 3 and Y_val_pred_std.shape[-1] == 1:
            Y_val_pred_std_2d = Y_val_pred_std[..., 0]
        else:
            Y_val_pred_std_2d = Y_val_pred_std

        Y_val_pred = Y_val_pred_std_2d * y_std[np.newaxis, :] + y_mean[np.newaxis, :]

        # Per-trait metrics
        for j, trait in enumerate(trait_names_out):
            mask_val_trait = W_val[:, j] > 0
            n_val_trait = int(mask_val_trait.sum())
            if n_val_trait == 0:
                r = np.nan
                mse = np.nan
                rmse = np.nan
            else:
                y_true = Y_full[val_idx, j][mask_val_trait]
                y_pred = Y_val_pred[:, j][mask_val_trait]
                r = pearson_r(y_true, y_pred)
                mse = float(np.mean((y_true - y_pred) ** 2))
                rmse = float(np.sqrt(mse))

            results_records.append(
                {
                    "fold": fold_idx,
                    "trait": trait,
                    "n_train": int(len(train_idx)),
                    "n_val": int(len(val_idx)),
                    "n_val_trait_non_missing": n_val_trait,
                    "r": r,
                    "mse": mse,
                    "rmse": rmse,
                    "best_val_loss": best_val_loss,
                    "n_epochs_run": n_epochs_run,
                }
            )

        # Save per-fold model
        model_path = os.path.join(models_dir, f"binn_fold{fold_idx}.keras")
        print(f"[FOLD {fold_idx}] Saving model -> {model_path}")
        model_fold.save(model_path)

        # Save gene→trait kernel for interpretation
        trait_layer = model_fold.get_layer("trait_output")
        kernel, bias = trait_layer.get_weights()  # kernel: (n_genes, n_traits_out)
        weights_path = os.path.join(weights_dir, f"binn_gene_weights_fold{fold_idx}.npz")
        print(f"[FOLD {fold_idx}] Saving gene weights -> {weights_path}")
        np.savez_compressed(
            weights_path,
            kernel=kernel.astype(np.float32),
            bias=bias.astype(np.float32),
            gene_ids=gene_ids,
            trait_names=np.array(trait_names_out, dtype=object),
        )

        # Save fold meta
        log_meta = {
            "fold": fold_idx,
            "train_indices": train_idx.tolist(),
            "val_indices": val_idx.tolist(),
            "x_mean_shape": list(x_mean.shape),
            "x_std_shape": list(x_std.shape),
            "y_mean": y_mean.tolist(),
            "y_std": y_std.tolist(),
            "trait_names_out": trait_names_out,
            "best_val_loss": best_val_loss,
            "n_epochs_run": n_epochs_run,
        }
        log_path = os.path.join(logs_dir, f"binn_fold{fold_idx}_meta.json")
        with open(log_path, "w", encoding="utf-8") as fh:
            json.dump(log_meta, fh, indent=2)

    # 5. Save CV results and summary
    results_df = pd.DataFrame(results_records)
    results_csv = os.path.join(OUT_BINN_DIR, "binn_cv_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\n[SAVE] Fold-level results -> {results_csv}")

    summary_records: List[Dict[str, object]] = []
    for trait in trait_names_out:
        sub = results_df[results_df["trait"] == trait]
        if sub.empty:
            continue
        summary_records.append(
            {
                "trait": trait,
                "n_folds": int(sub["fold"].nunique()),
                "mean_r": float(sub["r"].mean()),
                "sd_r": float(sub["r"].std(ddof=1)),
                "mean_rmse": float(sub["rmse"].mean()),
                "sd_rmse": float(sub["rmse"].std(ddof=1)),
            }
        )

    summary_df = pd.DataFrame(summary_records)
    summary_csv = os.path.join(OUT_BINN_DIR, "binn_cv_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"[SAVE] Summary per trait  -> {summary_csv}")

    print("\n[DONE] BINN training complete.")


if __name__ == "__main__":
    main()
