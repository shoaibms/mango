import os
import json
import math
import numpy as np

try:
    import pandas as pd
except ImportError as e:
    raise SystemExit("pandas is required. Install with: pip install pandas") from e

try:
    import tensorflow as tf
except ImportError as e:
    raise SystemExit(
        "tensorflow is required for Wide&Deep training. Install with: pip install tensorflow"
    ) from e


# =========================
# CONFIG
# =========================

TENSOR_DIR = r"C:\Users\ms\Desktop\mango\output\idea_3\tensors"

X_PATH = os.path.join(TENSOR_DIR, "X_background.npy")
Y_RAW_PATH = os.path.join(TENSOR_DIR, "y_raw.npy")
Y_RAW_TRAITS_PATH = os.path.join(TENSOR_DIR, "y_raw_traits.json")
Y_RESID_PATH = os.path.join(TENSOR_DIR, "y_resid.npy")
Y_RESID_TRAITS_PATH = os.path.join(TENSOR_DIR, "y_resid_traits.json")
CV_FOLDS_PATH = os.path.join(TENSOR_DIR, "cv_folds.json")
SAMPLE_IDS_PATH = os.path.join(TENSOR_DIR, "sample_ids.txt")

OUT_ROOT = r"C:\Users\ms\Desktop\mango\output\idea_3"
OUT_MODELS_DIR = os.path.join(OUT_ROOT, "models", "wide_deep")
OUT_METRICS_DIR = os.path.join(OUT_ROOT, "metrics")

# Optimised for N=225, 20k SNPs, strong regularisation
SEED = 2024
BATCH_SIZE = 16
MAX_EPOCHS = 150
PATIENCE = 15
VAL_FRACTION = 0.20

USE_RAW_TRAITS = True
USE_RESIDUAL_TRAITS = True


# =========================
# UTILITIES
# =========================

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_global_seed(seed: int = 2024):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass
    try:
        if hasattr(tf.config, "experimental") and hasattr(
            tf.config.experimental, "enable_op_determinism"
        ):
            tf.config.experimental.enable_op_determinism()
    except Exception as e:
        print(f"[WARN] Could not fully enforce deterministic ops: {e}")


def load_tensors():
    print(f"[INFO] Loading tensors from:\n  {TENSOR_DIR}")

    if not os.path.exists(X_PATH):
        raise FileNotFoundError(f"X_background.npy not found at:\n  {X_PATH}")
    X = np.load(X_PATH).astype(np.float32)
    print(f"[INFO] X_background shape: {X.shape} (samples x SNPs)")

    if not os.path.exists(Y_RAW_PATH):
        raise FileNotFoundError(f"y_raw.npy not found at:\n  {Y_RAW_PATH}")
    y_raw = np.load(Y_RAW_PATH)
    with open(Y_RAW_TRAITS_PATH, "r", encoding="utf-8") as f:
        raw_traits = json.load(f)
    print(f"[INFO] y_raw shape: {y_raw.shape} | raw traits: {raw_traits}")

    # Residuals (optional)
    if os.path.exists(Y_RESID_PATH) and os.path.exists(Y_RESID_TRAITS_PATH):
        y_resid = np.load(Y_RESID_PATH)
        with open(Y_RESID_TRAITS_PATH, "r", encoding="utf-8") as f:
            resid_traits = json.load(f)
        print(f"[INFO] y_resid shape: {y_resid.shape} | resid traits: {resid_traits}")
    else:
        y_resid = None
        resid_traits = []

    if not os.path.exists(CV_FOLDS_PATH):
        raise FileNotFoundError(f"cv_folds.json not found at:\n  {CV_FOLDS_PATH}")
    with open(CV_FOLDS_PATH, "r", encoding="utf-8") as f:
        folds = json.load(f)
    print(f"[INFO] Loaded {len(folds)} CV folds from cv_folds.json")

    sample_ids = []
    if os.path.exists(SAMPLE_IDS_PATH):
        with open(SAMPLE_IDS_PATH, "r", encoding="utf-8") as f:
            sample_ids = [line.strip() for line in f if line.strip()]
        if len(sample_ids) != X.shape[0]:
            print(
                f"[WARN] sample_ids length ({len(sample_ids)}) "
                f"!= n_samples ({X.shape[0]})."
            )
        else:
            print(
                f"[INFO] Loaded {len(sample_ids)} sample IDs "
                f"(first 3: {sample_ids[:3]})"
            )

    return X, y_raw, raw_traits, y_resid, resid_traits, folds, sample_ids


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Compute Pearson r, RMSE, MAE. NaN-safe.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() < 3:
        return np.nan, np.nan, np.nan

    y_true_m = y_true[mask]
    y_pred_m = y_pred[mask]

    if np.std(y_true_m) == 0 or np.std(y_pred_m) == 0:
        r = np.nan
    else:
        r = float(np.corrcoef(y_true_m, y_pred_m)[0, 1])

    mse = float(np.mean((y_true_m - y_pred_m) ** 2))
    rmse = float(math.sqrt(mse))
    mae = float(np.mean(np.abs(y_true_m - y_pred_m)))
    return r, rmse, mae


def build_wide_deep_model(input_dim: int, n_outputs: int) -> tf.keras.Model:
    """
    Multi-output Wide & Deep:
      - Wide: linear term on all SNPs (like dense ridge)
      - Deep: 2-layer MLP with strong L2 and dropout
    Designed for N=225 but expressive enough to go beyond pure linear.
    """
    inputs = tf.keras.Input(shape=(input_dim,), name="geno_flat")

    # Wide part: direct linear layer
    wide = tf.keras.layers.Dense(
        n_outputs,
        activation="linear",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="wide_linear",
    )(inputs)

    # Deep part: compressed representation
    x = tf.keras.layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-3),
        name="deep_dense1",
    )(inputs)
    x = tf.keras.layers.Dropout(0.4, name="deep_dropout1")(x)

    x = tf.keras.layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-3),
        name="deep_dense2",
    )(x)
    x = tf.keras.layers.Dropout(0.4, name="deep_dropout2")(x)

    deep_out = tf.keras.layers.Dense(
        n_outputs,
        activation="linear",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="deep_out",
    )(x)

    outputs = tf.keras.layers.Add(name="wide_plus_deep")([wide, deep_out])

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="wide_deep_multitask")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",  # average MSE across outputs
        metrics=[
            tf.keras.metrics.MeanSquaredError(name="mse"),
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
        ],
    )
    return model


# =========================
# MAIN
# =========================

def main():
    print("=" * 72)
    print("Mango GS – Idea 3: Multi-task Wide & Deep trainer (Optimised Data Handling)")
    print(" (04_train_wide_deep_multitask.py)")
    print("=" * 72)

    safe_mkdir(OUT_MODELS_DIR)
    safe_mkdir(OUT_METRICS_DIR)
    set_global_seed(SEED)

    X, y_raw, raw_traits, y_resid, resid_traits, folds, sample_ids = load_tensors()
    n_samples, n_snps = X.shape

    metrics_records = []

    def run_for_trait_block(block_name: str, y_mat: np.ndarray, trait_names: list):
        if y_mat is None or not trait_names:
            print(f"[INFO] No traits for block '{block_name}'; skipping.")
            return

        n_outputs = y_mat.shape[1]
        print(
            f"[INFO] Training Wide&Deep for block '{block_name}' "
            f"with {n_outputs} outputs: {trait_names}"
        )

        for fold in folds:
            fold_id = fold["fold_id"]
            train_idx = np.array(fold["train_idx"], dtype=int)
            test_idx = np.array(fold["test_idx"], dtype=int)

            y_train_full = y_mat[train_idx, :]
            y_test_full = y_mat[test_idx, :]

            # =============================================================
            # Optimised target handling:
            #  - Drop only rows that are all-NaN across all traits in train.
            #  - Mean-impute remaining NaNs per trait (column-wise) in train.
            #  - Keep raw NaNs in test; metrics are computed on non-NaN entries.
            # =============================================================
            row_all_nan = np.isnan(y_train_full).all(axis=1)
            if row_all_nan.any():
                keep_mask = ~row_all_nan
                y_train_used = y_train_full[keep_mask]
                X_train_used = X[train_idx[keep_mask], :]
                train_idx_eff = train_idx[keep_mask]
                print(
                    f"[INFO] Block {block_name}, fold {fold_id}: "
                    f"dropped {row_all_nan.sum()} all-NaN training rows."
                )
            else:
                y_train_used = y_train_full
                X_train_used = X[train_idx, :]
                train_idx_eff = train_idx

            # Column means for remaining train rows (ignore NaNs)
            col_means = np.nanmean(y_train_used, axis=0)
            if np.isnan(col_means).any():
                nan_cols = np.where(np.isnan(col_means))[0]
                print(
                    f"[WARN] Block {block_name}, fold {fold_id}: "
                    f"traits {nan_cols.tolist()} have all-NaN training values; "
                    f"setting their training means to 0.0."
                )
                col_means[nan_cols] = 0.0

            Y_train_imputed = y_train_used.copy()
            inds = np.where(np.isnan(Y_train_imputed))
            Y_train_imputed[inds] = np.take(col_means, inds[1])

            X_train = X_train_used
            Y_train = Y_train_imputed

            X_test = X[test_idx, :]
            Y_test = y_test_full

            n_train_eff = X_train.shape[0]
            n_test_eff = X_test.shape[0]

            if n_train_eff < 30 or n_test_eff < 5:
                print(
                    f"[WARN] Block {block_name}, fold {fold_id}: "
                    f"too few usable samples (n_train={n_train_eff}, n_test={n_test_eff}); skipping."
                )
                continue

            print(
                f"[INFO] Block {block_name}, fold {fold_id}: "
                f"n_train={n_train_eff} (imputed targets), n_test={n_test_eff}"
            )

            # Internal validation split (from train)
            rng = np.random.default_rng(SEED + fold_id * 1000 + n_outputs)
            perm = rng.permutation(n_train_eff)
            X_train = X_train[perm]
            Y_train = Y_train[perm]

            n_val = max(1, int(VAL_FRACTION * n_train_eff))
            n_val = min(n_val, n_train_eff - 3)  # keep a few for training
            if n_val < 1:
                n_val = 1

            X_val = X_train[:n_val]
            Y_val = Y_train[:n_val]
            X_train_eff = X_train[n_val:]
            Y_train_eff = Y_train[n_val:]

            print(
                f"[INFO] Fold {fold_id}: "
                f"train_eff={len(X_train_eff)}, val={len(X_val)}, test={len(X_test)}"
            )

            model = build_wide_deep_model(input_dim=n_snps, n_outputs=n_outputs)
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=PATIENCE,
                    restore_best_weights=True,
                    verbose=0,
                )
            ]

            history = model.fit(
                X_train_eff,
                Y_train_eff,
                validation_data=(X_val, Y_val),
                epochs=MAX_EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=0,
                callbacks=callbacks,
            )

            val_losses = history.history.get("val_loss", [])
            if val_losses:
                best_epoch = int(np.argmin(val_losses))
                best_val_loss = float(val_losses[best_epoch])
            else:
                best_epoch = len(history.history.get("loss", [])) - 1
                best_val_loss = float("nan")

            # Predict on test
            Y_pred = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)

            # Metrics per trait (computed on non-NaN test values)
            for j, trait_name in enumerate(trait_names):
                r, rmse, mae = compute_metrics(Y_test[:, j], Y_pred[:, j])
                print(
                    f"[INFO] Block {block_name}, trait {trait_name}, fold {fold_id} – "
                    f"r={r:.3f} | RMSE={rmse:.3f} | MAE={mae:.3f} | best_epoch={best_epoch}"
                )
                metrics_records.append(
                    {
                        "model": "wide_deep",
                        "block": block_name,
                        "trait": trait_name,
                        "fold_id": fold_id,
                        "n_train": int(n_train_eff),
                        "n_test": int(n_test_eff),
                        "best_epoch": best_epoch,
                        "best_val_loss": best_val_loss,
                        "pearson_r": r,
                        "rmse": rmse,
                        "mae": mae,
                    }
                )

            # Save model once per block/fold
            model_filename = f"wide_deep_block-{block_name}_fold-{fold_id}.keras"
            model_path = os.path.join(OUT_MODELS_DIR, model_filename)
            model.save(model_path)
            print(f"[OK] Saved Wide&Deep model to:\n  {model_path}")

    # Run for raw traits
    if USE_RAW_TRAITS:
        run_for_trait_block("raw", y_raw, raw_traits)

    # Run for residual traits (if available)
    if USE_RESIDUAL_TRAITS and y_resid is not None and len(resid_traits) > 0:
        run_for_trait_block("resid", y_resid, resid_traits)

    # Save metrics
    if metrics_records:
        df_metrics = pd.DataFrame(metrics_records)
        metrics_path = os.path.join(OUT_METRICS_DIR, "wide_deep_cv_metrics.csv")
        df_metrics.to_csv(metrics_path, index=False)
        print(f"[OK] Saved Wide&Deep CV metrics to:\n  {metrics_path}")
    else:
        print("[WARN] No Wide&Deep metrics recorded; check logs for issues.")

    print("[DONE] 04_train_wide_deep_multitask.py complete.")


if __name__ == "__main__":
    main()
