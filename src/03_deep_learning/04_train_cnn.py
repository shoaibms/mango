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
        "tensorflow is required for CNN training. Install with: pip install tensorflow"
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
OUT_MODELS_DIR = os.path.join(OUT_ROOT, "models", "cnn")
OUT_METRICS_DIR = os.path.join(OUT_ROOT, "metrics")

# =========================
# HYPERPARAMETERS
# =========================
SEED = 123
BATCH_SIZE = 16
MAX_EPOCHS = 100
PATIENCE = 10
VAL_FRACTION = 0.20

USE_RAW_TRAITS = True
USE_RESIDUAL_TRAITS = True


# =========================
# UTILITIES
# =========================

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_global_seed(seed: int = 123):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Force deterministic operations where possible
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
    X = np.load(X_PATH)
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


def expand_for_cnn(X_2d: np.ndarray) -> np.ndarray:
    """(n_samples, n_snps) -> (n_samples, n_snps, 1) for Conv1D."""
    return np.expand_dims(X_2d.astype(np.float32), axis=-1)


def build_cnn_model(input_length: int) -> tf.keras.Model:
    """
    Build a shallow, regularized 1D-CNN for genomic prediction.
    
    Architecture designed for high-dimensional, low sample size settings.
    """
    inputs = tf.keras.Input(shape=(input_length, 1), name="genotype_sequence")

    x = tf.keras.layers.Conv1D(
        filters=8,
        kernel_size=16,
        strides=2,
        padding="same",
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.01),
        name="conv1",
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=4, name="pool1")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout1")(x)

    x = tf.keras.layers.Conv1D(
        filters=16,
        kernel_size=8,
        strides=1,
        padding="same",
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.01),
        name="conv2",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="bn2")(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=4, name="pool2")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout2")(x)

    x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="linear",
        kernel_regularizer=tf.keras.regularizers.l2(0.05),
        name="output",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="shallow_cnn_regressor")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanSquaredError(name="mse"),
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
        ],
    )
    return model


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


# =========================
# MAIN TRAINING LOOP
# =========================

def main():
    print("=" * 72)
    print("Mango GS – Idea 3: Shallow 1D-CNN single-trait CV trainer")
    print(" (03_train_cnn_single_trait.py)")
    print("=" * 72)

    safe_mkdir(OUT_MODELS_DIR)
    safe_mkdir(OUT_METRICS_DIR)
    set_global_seed(SEED)

    # 1) Load tensors
    X_2d, y_raw, raw_traits, y_resid, resid_traits, folds, sample_ids = load_tensors()
    n_samples, n_snps = X_2d.shape
    X = expand_for_cnn(X_2d)
    print(f"[INFO] X for CNN shape: {X.shape} (samples x SNPs x channels)")

    metrics_records = []

    def run_for_trait_set(trait_type: str, y_mat: np.ndarray, trait_names: list):
        if y_mat is None or not trait_names:
            print(f"[INFO] No traits for trait_type={trait_type}; skipping.")
            return

        for j, trait_name in enumerate(trait_names):
            print("-" * 72)
            print(f"[INFO] Training Shallow CNN for {trait_type} trait: {trait_name}")
            y_vec = y_mat[:, j].astype(float)

            for fold in folds:
                fold_id = fold["fold_id"]
                train_idx = np.array(fold["train_idx"], dtype=int)
                test_idx = np.array(fold["test_idx"], dtype=int)

                # Handle NaNs
                mask_train = ~np.isnan(y_vec[train_idx])
                mask_test = ~np.isnan(y_vec[test_idx])
                train_idx_eff = train_idx[mask_train]
                test_idx_eff = test_idx[mask_test]

                n_train_eff = len(train_idx_eff)
                n_test_eff = len(test_idx_eff)

                if n_train_eff < 30 or n_test_eff < 5:
                    print(
                        f"[WARN] Trait {trait_name}, fold {fold_id}: "
                        f"too few usable samples (n_train={n_train_eff}, n_test={n_test_eff}); skipping."
                    )
                    continue

                print(
                    f"[INFO] Trait {trait_name}, fold {fold_id}: "
                    f"n_train={n_train_eff}, n_test={n_test_eff}"
                )

                X_train = X[train_idx_eff]
                y_train = y_vec[train_idx_eff]
                X_test = X[test_idx_eff]
                y_test = y_vec[test_idx_eff]

                # Internal validation split (from train)
                # Use specific seed for reproducible splits
                rng = np.random.default_rng(SEED + fold_id * 100 + j)
                perm = rng.permutation(n_train_eff)
                X_train = X_train[perm]
                y_train = y_train[perm]

                n_val = max(1, int(VAL_FRACTION * n_train_eff))
                n_val = min(n_val, n_train_eff - 2)  # leave at least 2 for train

                X_val = X_train[:n_val]
                y_val = y_train[:n_val]
                X_train_eff = X_train[n_val:]
                y_train_eff = y_train[n_val:]

                print(
                    f"[INFO] Fold {fold_id}: "
                    f"train_eff={len(X_train_eff)}, val={len(X_val)}, test={len(X_test)}"
                )

                model = build_cnn_model(input_length=n_snps)
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
                    y_train_eff,
                    validation_data=(X_val, y_val),
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

                # Test predictions
                y_pred = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).ravel()
                r, rmse, mae = compute_metrics(y_test, y_pred)

                print(
                    f"[INFO] Fold {fold_id} – "
                    f"r={r:.3f} | RMSE={rmse:.3f} | MAE={mae:.3f} | best_epoch={best_epoch}"
                )

                # Save model
                model_filename = (
                    f"cnn_shallow_type-{trait_type}_trait-{trait_name}_fold-{fold_id}.keras"
                )
                model_path = os.path.join(OUT_MODELS_DIR, model_filename)
                model.save(model_path)
                print(f"[OK] Saved model to:\n  {model_path}")

                # Record metrics
                metrics_records.append(
                    {
                        "cnn_arch": "shallow_l2_dropout",
                        "trait_type": trait_type,
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

    # Raw traits
    if USE_RAW_TRAITS:
        run_for_trait_set("raw", y_raw, raw_traits)

    # Residual traits
    if USE_RESIDUAL_TRAITS and y_resid is not None and len(resid_traits) > 0:
        run_for_trait_set("resid", y_resid, resid_traits)

    # Save metrics
    if metrics_records:
        df_metrics = pd.DataFrame(metrics_records)
        metrics_path = os.path.join(OUT_METRICS_DIR, "cnn_cv_metrics.csv")
        df_metrics.to_csv(metrics_path, index=False)
        print(f"[OK] Saved CNN CV metrics to:\n  {metrics_path}")
    else:
        print("[WARN] No metrics recorded; check training loop / NaNs.")

    print("[DONE] 03_train_cnn_single_trait.py complete.")


if __name__ == "__main__":
    main()