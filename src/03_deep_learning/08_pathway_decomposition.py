import os
import json
import math
import numpy as np
import pandas as pd

try:
    import tensorflow as tf
except ImportError as e:
    raise SystemExit(
        "tensorflow is required. Install with: pip install tensorflow"
    ) from e


# =========================
# CONFIG
# =========================

BASE_DIR = r"C:\Users\ms\Desktop\mango\output\idea_3"
TENSOR_DIR = os.path.join(BASE_DIR, "tensors")
MODEL_DIR = os.path.join(BASE_DIR, "models", "wide_deep")
METRICS_DIR = os.path.join(BASE_DIR, "metrics")

X_PATH = os.path.join(TENSOR_DIR, "X_background.npy")
Y_RAW_PATH = os.path.join(TENSOR_DIR, "y_raw.npy")
Y_RAW_TRAITS_PATH = os.path.join(TENSOR_DIR, "y_raw_traits.json")
Y_RESID_PATH = os.path.join(TENSOR_DIR, "y_resid.npy")
Y_RESID_TRAITS_PATH = os.path.join(TENSOR_DIR, "y_resid_traits.json")
CV_FOLDS_PATH = os.path.join(TENSOR_DIR, "cv_folds.json")

OUT_PATH = os.path.join(METRICS_DIR, "wide_deep_decomposition_cv_metrics.csv")

SEED = 2024
BATCH_SIZE = 32


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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Compute Pearson r, RMSE, MAE. NaN-safe (ignores NaNs in y_true or y_pred).
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


def load_tensors():
    """
    Load X, y_raw, y_resid, trait names, and CV folds.
    """
    if not os.path.exists(X_PATH):
        raise FileNotFoundError(f"X_background.npy not found at:\n  {X_PATH}")
    X = np.load(X_PATH).astype(np.float32)
    print(f"[INFO] X shape: {X.shape} (samples x SNPs)")

    if not os.path.exists(Y_RAW_PATH):
        raise FileNotFoundError(f"y_raw.npy not found at:\n  {Y_RAW_PATH}")
    y_raw = np.load(Y_RAW_PATH)
    with open(Y_RAW_TRAITS_PATH, "r", encoding="utf-8") as f:
        raw_traits = json.load(f)
    print(f"[INFO] y_raw shape: {y_raw.shape} | traits: {raw_traits}")

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
    print(f"[INFO] Loaded {len(folds)} CV folds.")

    return X, y_raw, raw_traits, y_resid, resid_traits, folds


def build_branch_models(base_model: tf.keras.Model):
    """
    Given a trained wide_deep model, return:
      - full_model: base_model itself
      - wide_model: outputs wide_linear only
      - deep_model: outputs deep_out only
    """
    try:
        wide_out = base_model.get_layer("wide_linear").output
        deep_out = base_model.get_layer("deep_out").output
    except ValueError as e:
        raise RuntimeError(
            "Expected layers 'wide_linear' and 'deep_out' not found in model. "
            "Check that this is a Wide&Deep model from 04_train_wide_deep_multitask.py."
        ) from e

    wide_model = tf.keras.Model(inputs=base_model.input, outputs=wide_out)
    deep_model = tf.keras.Model(inputs=base_model.input, outputs=deep_out)
    return base_model, wide_model, deep_model


# =========================
# MAIN
# =========================

def main():
    print("=" * 72)
    print("Mango GS – Idea 3: Wide vs Deep decomposition")
    print(" (07_wide_deep_decomposition.py)")
    print("=" * 72)

    safe_mkdir(METRICS_DIR)
    set_global_seed(SEED)

    X, y_raw, raw_traits, y_resid, resid_traits, folds = load_tensors()
    n_samples, n_snps = X.shape

    records = []

    def process_block(block_name: str, Y: np.ndarray, trait_names: list[str]):
        if Y is None or not trait_names:
            print(f"[INFO] No traits for block '{block_name}'; skipping.")
            return

        n_outputs = Y.shape[1]
        print(
            f"[INFO] Decomposing Wide&Deep for block '{block_name}' "
            f"({n_outputs} outputs): {trait_names}"
        )

        for fold in folds:
            fold_id = fold["fold_id"]
            train_idx = np.array(fold["train_idx"], dtype=int)
            test_idx = np.array(fold["test_idx"], dtype=int)

            model_path = os.path.join(
                MODEL_DIR, f"wide_deep_block-{block_name}_fold-{fold_id}.keras"
            )
            if not os.path.exists(model_path):
                print(
                    f"[WARN] Model for block='{block_name}', fold={fold_id} not found:\n"
                    f"       {model_path}\n       Skipping this fold."
                )
                continue

            print(
                f"[INFO] Block '{block_name}', fold {fold_id}: "
                f"loading model and computing predictions..."
            )
            base_model = tf.keras.models.load_model(model_path)
            full_model, wide_model, deep_model = build_branch_models(base_model)

            # Use all samples in the nominal test fold
            X_test = X[test_idx, :]
            Y_test = Y[test_idx, :]
            n_test = X_test.shape[0]

            # Predict
            Y_full_pred = full_model.predict(
                X_test, batch_size=BATCH_SIZE, verbose=0
            )
            Y_wide_pred = wide_model.predict(
                X_test, batch_size=BATCH_SIZE, verbose=0
            )
            Y_deep_pred = deep_model.predict(
                X_test, batch_size=BATCH_SIZE, verbose=0
            )

            if (
                Y_full_pred.shape[1] != n_outputs
                or Y_wide_pred.shape[1] != n_outputs
                or Y_deep_pred.shape[1] != n_outputs
            ):
                raise RuntimeError(
                    f"Output dimension mismatch for block '{block_name}', fold {fold_id}."
                )

            # Per-trait metrics
            for j, trait in enumerate(trait_names):
                r_full, rmse_full, mae_full = compute_metrics(Y_test[:, j], Y_full_pred[:, j])
                r_wide, rmse_wide, mae_wide = compute_metrics(Y_test[:, j], Y_wide_pred[:, j])
                r_deep, rmse_deep, mae_deep = compute_metrics(Y_test[:, j], Y_deep_pred[:, j])

                print(
                    f"[INFO] Block={block_name}, fold={fold_id}, trait={trait}: "
                    f"full r={r_full:.3f}, wide r={r_wide:.3f}, deep r={r_deep:.3f}"
                )

                records.append(
                    {
                        "block": block_name,      # 'raw' or 'resid'
                        "trait": trait,
                        "fold_id": fold_id,
                        "n_test": int(n_test),
                        "r_full": r_full,
                        "rmse_full": rmse_full,
                        "mae_full": mae_full,
                        "r_wide": r_wide,
                        "rmse_wide": rmse_wide,
                        "mae_wide": mae_wide,
                        "r_deep": r_deep,
                        "rmse_deep": rmse_deep,
                        "mae_deep": mae_deep,
                    }
                )

    # Process raw traits
    process_block("raw", y_raw, raw_traits)

    # Process residual traits if available
    if y_resid is not None and len(resid_traits) > 0:
        process_block("resid", y_resid, resid_traits)

    # Save results
    if records:
        df = pd.DataFrame(records)
        df.to_csv(OUT_PATH, index=False)
        print(f"[OK] Saved Wide vs Deep decomposition metrics to:\n  {OUT_PATH}")
    else:
        print("[WARN] No decomposition records generated. Check logs for missing models.")

    print("[DONE] 07_wide_deep_decomposition.py complete.")


if __name__ == "__main__":
    main()
