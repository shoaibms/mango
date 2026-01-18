# -*- coding: utf-8 -*-
r"""
14_binn_architecture.py

Mango GS – Idea 3
=================
Define a Biologically Informed Neural Network (BINN) that maps:
    SNPs  → Genes → Traits

This version uses:
  * SNP→gene mapping from 10_binn_build_maps.py (binn_snp_map.npz)
  * A masked linear layer so each SNP only connects to its annotated gene
  * A multi-trait output head with a canonical trait order.

The idea:
  - Input:  X (n_snps)  – genotype dosages for the SNP subset used by BINN
  - Gene layer: one node per gene; each gene aggregates only its mapped SNPs
  - Output layer: one node per trait (FBC, FF, AFW, TSS, TC) with linear outputs

This script does NOT train the model. Training and evaluation will be handled
by 12_binn_train.py, which will:
  - Load geno_core and binn_snp_map
  - Construct X_binn = geno_core[:, snp_core_index] in the SAME order as snp_ids
  - Align y with the same trait ordering used here.

"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np

try:
    import tensorflow as tf
except ImportError as e:
    raise SystemExit(
        "tensorflow is required. Install it with: pip install tensorflow"
    ) from e


# =========================
# CONFIG
# =========================

# Default path to the SNP→gene map produced by 10_binn_build_maps.py
BINN_SNP_MAP_PATH = (
    r"C:\Users\ms\Desktop\mango\output\idea_3\binn_maps\binn_snp_map.npz"
)

# Canonical trait order for BINN outputs
TRAIT_ORDER: List[str] = ["FBC", "FF", "AFW", "TSS", "TC"]

# Some default hyperparameters (can be overridden from 12_binn_train.py)
DEFAULT_L2 = 1e-4
DEFAULT_GENE_ACTIVATION = "relu"
DEFAULT_GENE_DROPOUT = 0.0


# =========================
# UTILITIES
# =========================

def load_binn_snp_map(
    path: str = BINN_SNP_MAP_PATH,
) -> Dict[str, np.ndarray]:
    """
    Load binn_snp_map.npz produced by 10_binn_build_maps.py.

    Expected keys:
      - snp_ids         : (n_snps,)   SNP IDs (string)
      - snp_core_index  : (n_snps,)   int indices into geno_core columns
      - snp_chr         : (n_snps,)
      - snp_pos         : (n_snps,)
      - snp_gene_index  : (n_snps,)   int indices into gene_ids (or -1)
      - snp_trait_matrix: (n_snps, n_traits) 0/1 mask (candidate per trait)
      - trait_names     : (n_traits,) trait labels
      - gene_ids        : (n_genes,)  gene IDs

    Returns a dict with these arrays as numpy arrays.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"BINN SNP map not found:\n  {path}\n"
            "Run 10_binn_build_maps.py first."
        )

    print(f"[INFO] Loading BINN SNP map from:\n  {path}")
    npz = np.load(path, allow_pickle=True)
    keys = list(npz.files)
    print(f"[INFO] Contents of {os.path.basename(path)}: {keys}")

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
            f"binn_snp_map.npz is missing required keys: {missing}\n"
            f"Available keys: {keys}"
        )

    data = {k: npz[k] for k in required}
    # Ensure 1D / 2D shapes where expected
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
      mask[i, j] = 1 if SNP i maps to gene j, else 0

    If normalise=True, each non-zero is scaled by 1 / n_snps_for_gene,
    i.e. the gene receives the *average* SNP signal rather than the sum.

    Parameters
    ----------
    snp_gene_index : array of shape (n_snps,)
        For SNP i, gene index j (0..n_genes-1) or -1 if unmapped.
    n_genes : int
        Number of gene nodes.
    normalise : bool
        Whether to divide each gene's incoming weights by count of SNPs.

    Returns
    -------
    mask : np.ndarray of shape (n_snps, n_genes)
        Float32 mask matrix.
    """
    snp_gene_index = np.asarray(snp_gene_index, dtype=int)
    n_snps = snp_gene_index.shape[0]

    mask = np.zeros((n_snps, n_genes), dtype=np.float32)

    # Count how many SNPs map to each gene
    valid = snp_gene_index >= 0
    if not np.any(valid):
        raise ValueError("No valid gene indices found in snp_gene_index.")

    counts = np.bincount(snp_gene_index[valid], minlength=n_genes)
    if normalise:
        # Avoid divide-by-zero; genes with 0 SNPs get factor 0
        factors = np.zeros_like(counts, dtype=np.float32)
        nonzero = counts > 0
        factors[nonzero] = 1.0 / counts[nonzero].astype(np.float32)
    else:
        factors = np.ones_like(counts, dtype=np.float32)

    # Fill mask
    for snp_idx, gene_idx in enumerate(snp_gene_index):
        if gene_idx < 0 or gene_idx >= n_genes:
            continue
        mask[snp_idx, gene_idx] = factors[gene_idx]

    return mask


# =========================
# LAYERS
# =========================

class SnpToGeneLayer(tf.keras.layers.Layer):
    """
    Masked SNP→Gene layer.

    Given input X of shape (batch_size, n_snps), this layer computes gene
    activations of shape (batch_size, n_genes) as:

        gene = activation( X @ (W ⊙ mask) + b )

    where:
      - W is a learnable (n_snps, n_genes) weight matrix,
      - mask is a fixed 0/weight_factor matrix telling which SNPs connect
        to which gene (built from snp_gene_index and optional normalisation),
      - b is a learnable bias (n_genes,).

    This enforces that each SNP only contributes to its annotated gene(s),
    preserving the biological structure.
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

        # Store mask as constant tensor (not trainable)
        self.snp_gene_mask = tf.constant(mask, dtype=tf.float32, name="snp_gene_mask")

        self.activation_name = activation
        self.l2_reg = float(l2_reg)
        self.dropout_rate = float(dropout_rate)

        self._activation = tf.keras.activations.get(activation)
        self._kernel_regularizer = tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None
        self._dropout = (
            tf.keras.layers.Dropout(self.dropout_rate) if self.dropout_rate > 0 else None
        )

    def build(self, input_shape):
        # input_shape should be (batch_size, n_snps)
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
        # Mask the kernel
        masked_W = self.W * self.snp_gene_mask  # shape: (n_snps, n_genes)
        # Dense: X @ masked_W  -> (batch, n_genes)
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
                # Note: snp_gene_mask is not serialised here; we expect to rebuild
                # the layer from numpy arrays in build_binn_model().
            }
        )
        return cfg


# =========================
# MODEL BUILDER
# =========================

def build_binn_model(
    snp_map_path: str = BINN_SNP_MAP_PATH,
    trait_order: List[str] | None = None,
    gene_activation: str = DEFAULT_GENE_ACTIVATION,
    gene_dropout: float = DEFAULT_GENE_DROPOUT,
    l2_reg: float = DEFAULT_L2,
) -> Tuple[tf.keras.Model, Dict[str, object]]:
    """
    Build a BINN model from the SNP→gene map.

    Parameters
    ----------
    snp_map_path : str
        Path to binn_snp_map.npz (from 10_binn_build_maps.py).
    trait_order : list of str or None
        Desired output trait order. If None, uses TRAIT_ORDER.
        Only traits present in the map's 'trait_names' are used, in that order.
    gene_activation : str
        Activation for the gene layer (e.g. 'relu').
    gene_dropout : float
        Dropout rate on gene activations (0.0 disables).
    l2_reg : float
        L2 regularisation strength applied to SNP→gene and gene→trait weights.

    Returns
    -------
    model : tf.keras.Model
        Keras model mapping X_binn (batch, n_snps) → Y_pred (batch, n_traits_used).
    meta : dict
        Metadata including:
            - n_snps, n_genes, n_traits_map, n_traits_out
            - trait_names_in_map
            - trait_names_out (order of outputs)
            - trait_index_map_to_out (indices into map's trait_names)
            - snp_ids, gene_ids
    """
    data = load_binn_snp_map(snp_map_path)

    snp_ids = data["snp_ids"]
    gene_ids = data["gene_ids"]
    snp_gene_index = data["snp_gene_index"]
    trait_names_map = list(data["trait_names"])

    n_snps = snp_ids.shape[0]
    n_genes = gene_ids.shape[0]
    n_traits_map = len(trait_names_map)

    print(f"[INFO] BINN map: n_snps={n_snps}, n_genes={n_genes}, n_traits_in_map={n_traits_map}")

    # Trait order handling
    if trait_order is None:
        trait_order = TRAIT_ORDER

    # Only keep traits that are present in the map
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

    # Build SNP→gene mask
    snp_gene_mask = build_snp_gene_mask(
        snp_gene_index=snp_gene_index,
        n_genes=n_genes,
        normalise=True,
    )

    # -------------------------
    # Keras model definition
    # -------------------------
    inputs = tf.keras.Input(shape=(n_snps,), name="snp_input")

    gene_layer = SnpToGeneLayer(
        snp_gene_mask=snp_gene_mask,
        activation=gene_activation,
        l2_reg=l2_reg,
        dropout_rate=gene_dropout,
        name="snp_to_gene",
    )(inputs)

    # Gene → trait logits (kept as a separate layer for exporting weights)
    trait_logits = tf.keras.layers.Dense(
        units=len(trait_names_out),
        activation=None,
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None,
        name="trait_output",
    )(gene_layer)

    # Reshape to (batch, n_traits, 1) for per-trait sample weighting
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
# MAIN (simple diagnostic)
# =========================

def main():
    print("=" * 72)
    print("Mango GS – Idea 3: BINN model definition (11_binn_model.py)")
    print("=" * 72)
    print(f"[INFO] BINN SNP map path: {BINN_SNP_MAP_PATH}")
    print(f"[INFO] Canonical trait order: {TRAIT_ORDER}")
    print("")

    model, meta = build_binn_model(
        snp_map_path=BINN_SNP_MAP_PATH,
        trait_order=TRAIT_ORDER,
        gene_activation=DEFAULT_GENE_ACTIVATION,
        gene_dropout=DEFAULT_GENE_DROPOUT,
        l2_reg=DEFAULT_L2,
    )

    print("")
    print("[INFO] Model summary:")
    model.summary(line_length=120)

    # Save a small JSON with meta information (optional, for 12_binn_train.py)
    out_dir = os.path.dirname(BINN_SNP_MAP_PATH)
    meta_path = os.path.join(out_dir, "binn_model_meta.json")
    print(f"\n[SAVE] BINN model meta JSON -> {meta_path}")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "n_snps": meta["n_snps"],
                "n_genes": meta["n_genes"],
                "n_traits_map": meta["n_traits_map"],
                "n_traits_out": meta["n_traits_out"],
                "trait_names_in_map": meta["trait_names_in_map"],
                "trait_names_out": meta["trait_names_out"],
                "trait_index_map_to_out": meta["trait_index_map_to_out"],
            },
            fh,
            indent=2,
        )

    print("\n[DONE] BINN model constructed successfully.")


if __name__ == "__main__":
    main()
