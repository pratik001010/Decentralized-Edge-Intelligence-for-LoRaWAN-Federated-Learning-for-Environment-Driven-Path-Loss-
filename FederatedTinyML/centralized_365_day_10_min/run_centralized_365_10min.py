"""
run_centralized_365_10min.py
============================
Centralized Neural Network Baseline Simulation on the 365-Day 10-Minute Staggered Dataset (206,957 rows).

Master's Thesis:
    "Decentralized Edge Intelligence for LoRaWAN: Federated Learning for
     Environment-Driven Path Loss and Link Quality Modeling"

Author  : Pratik Khadka
Uni     : University of Siegen
Date    : 2025/2026

UPDATED VERSION:
  Aligned with FedAvg simulation pipeline for 100% experimental consistency & zero data leakage:
  1. Uses canonical 10-minute dataset (365_days_staggered_10min_sampled.csv, ~206,957 rows)
  2. Matched hyperparameters (batch_size=512, lr=0.01)
  3. No data leakage: StandardScaler fitted strictly on training partition (X_train)
  4. Proper 3-way split: Train (140,729), Val (24,835), Test (41,393)
  5. Per-device evaluation on held-out test set with pooled RMSE verification
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
from tensorflow import keras

# Line unbuffering for live execution visibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Set deterministic random seeds for 100% reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Output directory & Dataset path
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__)) if __file__ else r"c:\Users\prati\Desktop\edge AI\FederatedTinyML\centralized_365_day_10_min"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

DATASET_PATHS = [
    r"365_days_staggered_10min_sampled.csv",
    r"..\365_days_staggered_10min_sampled.csv",
    r"c:\Users\prati\Desktop\edge AI\FederatedTinyML\365_days_staggered_10min_sampled.csv",
]

FEATURE_COLS = [
    "log_distance",  # 10 * log10(d / d0)
    "W_brick",       # c_walls (brick/concrete wall count)
    "W_wood",        # w_walls (wooden partition count)
    "co2",           # CO2 concentration (ppm)
    "humidity",      # Relative humidity (%)
    "pm25",          # PM2.5 (ug/m3)
    "pressure",      # Pressure in hPa (raw * 3.125)
    "temperature",   # Temperature (deg C)
    "snr",           # SNR (dB)
]

TARGET_COL = "exp_pl"
DEVICE_COL = "device_id"
EPOCHS = 50
BATCH_SIZE = 512
LEARNING_RATE = 0.01


def load_and_preprocess_data():
    dataset_path = None
    for p in DATASET_PATHS:
        if os.path.exists(p):
            dataset_path = p
            break

    if dataset_path is None:
        raise FileNotFoundError(f"Dataset not found in paths: {DATASET_PATHS}")

    print(f"Loading 365-day dataset from: {dataset_path}...")
    df = pd.read_csv(dataset_path, low_memory=False)
    raw_rows = len(df)
    print(f"Raw Dataset Shape: {df.shape}")

    if "dev_id" in df.columns and "device_id" not in df.columns:
        df["device_id"] = df["dev_id"]

    # Remove known corrupted/anomalous readings
    pa = ((df["co2"] == 21547.0) & (df["humidity"] == 156.65) & (df["temperature"] == 174.90) & (df["pressure"] == 3.21) & (df["pm25"] == 33.93))
    pb = ((df["co2"] == 16724.0) & (df["humidity"] == 210.53) & (df["temperature"] == 110.76) & (df["pressure"] == 317.45) & (df["pm25"] == 125.57))
    pc = ((df["co2"] == 0.0) & (df["humidity"] == 0.0) & (df["temperature"] == 0.0) & (df["pressure"] == 508.90) & (df["pm25"] == 0.0))
    bad = pa | pb | pc
    df = df.loc[~bad].copy()
    anom_count = int(bad.sum())

    # Convert pressure to true hPa
    if "pressure" in df.columns:
        df["pressure"] = pd.to_numeric(df["pressure"], errors="coerce")
        if df["pressure"].mean() < 500:
            df["pressure"] = df["pressure"] * 3.125

    # Feature engineering
    if "distance" in df.columns and "log_distance" not in df.columns:
        df["log_distance"] = 10.0 * np.log10(pd.to_numeric(df["distance"], errors="coerce").clip(lower=1.0))
    if "c_walls" in df.columns and "W_brick" not in df.columns:
        df["W_brick"] = pd.to_numeric(df["c_walls"], errors="coerce")
    if "w_walls" in df.columns and "W_wood" not in df.columns:
        df["W_wood"] = pd.to_numeric(df["w_walls"], errors="coerce")

    # Clean numeric columns & target boundaries
    for col in FEATURE_COLS + [TARGET_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL, DEVICE_COL]).copy()
    df = df[df[TARGET_COL].between(50, 200)].copy()

    print(f"Cleaned Usable Rows: {len(df):,} across devices {sorted(df[DEVICE_COL].unique())}")
    return df, raw_rows, anom_count


def split_and_scale_data(df):
    X_all = df[FEATURE_COLS].values.astype(np.float32)
    y_all = df[TARGET_COL].values.astype(np.float32)
    dev_all = df[DEVICE_COL].values

    # Step 1: Split 80% train+val, 20% test
    X_train_val_raw, X_test_raw, y_train_val, y_test, dev_train_val, dev_test = train_test_split(
        X_all, y_all, dev_all, test_size=0.20, random_state=SEED, stratify=dev_all
    )

    # Step 2: Split train_val into 85% train, 15% val
    X_train_raw, X_val_raw, y_train, y_val, dev_train, dev_val = train_test_split(
        X_train_val_raw, y_train_val, dev_train_val, test_size=0.15, random_state=SEED, stratify=dev_train_val
    )

    print(f"Train Set: {len(X_train_raw):,} | Val Set: {len(X_val_raw):,} | Held-Out Test Set: {len(X_test_raw):,}")

    # Scaler fitted STRICTLY on X_train_raw (zero data leakage)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val   = scaler.transform(X_val_raw)
    X_test  = scaler.transform(X_test_raw)

    return X_train, y_train, dev_train, X_val, y_val, dev_val, X_test, y_test, dev_test, scaler


def build_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(len(FEATURE_COLS),), name="input"),
        keras.layers.Dense(8, activation="relu", name="hidden",
                           kernel_initializer=keras.initializers.GlorotUniform(seed=SEED)),
        keras.layers.Dense(1, activation="linear", name="output",
                           kernel_initializer=keras.initializers.GlorotUniform(seed=SEED))
    ], name="centralized_mlp_365")

    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss="mse")
    return model


def main():
    t0 = time.time()
    print("==========================================================================")
    print("STARTING CENTRALIZED BASELINE SIMULATION (365-DAY 10-MIN DATASET)")
    print("==========================================================================")

    # 1. Load and preprocess data
    df, raw_rows, anom_count = load_and_preprocess_data()

    # 2. Split and scale data (zero data leakage)
    X_train, y_train, dev_train, X_val, y_val, dev_val, X_test, y_test, dev_test, scaler = split_and_scale_data(df)

    var_tr = np.var(y_train)
    var_val = np.var(y_val)

    # 3. Build Keras MLP Dense(9 -> 8 -> 1)
    model = build_model()
    model.summary()

    # Custom Epoch Callback tracking metrics
    class RawMetricLogger(keras.callbacks.Callback):
        def __init__(self, model, X_tr, y_tr, X_val, y_val, var_tr, var_val):
            super().__init__()
            self.model_ref = model
            self.X_tr = X_tr
            self.y_tr = y_tr
            self.X_val = X_val
            self.y_val = y_val
            self.var_tr = var_tr
            self.var_val = var_val
            self.epoch_metrics = []

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            val_loss = logs.get("val_loss")

            tr_eval = self.model_ref.evaluate(self.X_tr, self.y_tr, batch_size=2048, verbose=0)
            tr_loss = tr_eval[0] if isinstance(tr_eval, list) else tr_eval

            tr_rmse = np.sqrt(tr_loss)
            val_rmse = np.sqrt(val_loss)

            tr_r2 = 1.0 - (tr_loss / self.var_tr)
            val_r2 = 1.0 - (val_loss / self.var_val)

            self.epoch_metrics.append({
                "epoch": epoch + 1,
                "train_loss": tr_loss,
                "val_loss": val_loss,
                "train_r2": tr_r2,
                "val_r2": val_r2,
                "train_rmse": tr_rmse,
                "val_rmse": val_rmse,
            })
            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == EPOCHS - 1:
                print(f"  Epoch {epoch+1:2d}/{EPOCHS} — Train R2: {tr_r2:.4f} | Val R2: {val_r2:.4f} | Train RMSE: {tr_rmse:.2f} dB | Val RMSE: {val_rmse:.2f} dB")

    logger_cb = RawMetricLogger(model, X_train, y_train, X_val, y_val, var_tr, var_val)

    print(f"\n--- Training Centralized MLP over {EPOCHS} Epochs (lr={LEARNING_RATE}, batch_size={BATCH_SIZE}) ---")
    train_start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[logger_cb]
    )
    train_time = time.time() - train_start
    print(f"Training completed cleanly in {train_time:.2f} seconds!")

    # 4. Global Test Evaluation
    y_pred_test = model.predict(X_test, verbose=0).flatten()
    final_r2 = r2_score(y_test, y_pred_test)
    final_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print("\n==========================================================================")
    print("FINAL HELD-OUT TEST RESULTS (CENTRALIZED MODEL - 365-DAY 10-MIN)")
    print(f"  Test R^2 Score : {final_r2:.4f}")
    print(f"  Test RMSE (dB) : {final_rmse:.4f} dB")
    print("==========================================================================")

    # 5. Export Epoch Metrics
    epochs_df = pd.DataFrame(logger_cb.epoch_metrics)
    epochs_df.to_csv(os.path.join(OUTPUT_DIR, "epoch_training_metrics.csv"), index=False)

    # 6. Per-Device Evaluation on Held-Out Test Set
    per_device_records = []
    total_squared_error = 0.0
    total_test_n = 0

    for dev in sorted(np.unique(dev_test)):
        mask_dev = dev_test == dev
        dev_X_test = X_test[mask_dev]
        dev_y_test = y_test[mask_dev]

        dev_pred = model.predict(dev_X_test, verbose=0).flatten()
        r2_dev = r2_score(dev_y_test, dev_pred)
        rmse_dev = np.sqrt(mean_squared_error(dev_y_test, dev_pred))

        total_squared_error += np.sum((dev_y_test - dev_pred) ** 2)
        total_test_n += len(dev_y_test)

        per_device_records.append({
            "Device": dev,
            "Test_Samples": len(dev_y_test),
            "R2": round(r2_dev, 4),
            "RMSE_dB": round(rmse_dev, 4)
        })

    pooled_rmse = np.sqrt(total_squared_error / total_test_n)
    print(f"Pooled Per-Device RMSE: {pooled_rmse:.4f} dB vs Global Test RMSE: {final_rmse:.4f} dB")

    per_device_df = pd.DataFrame(per_device_records)
    print("\nPer-Device Evaluation Breakdown (Held-Out Test Set):")
    print(per_device_df.to_string(index=False))
    per_device_df.to_csv(os.path.join(OUTPUT_DIR, "per_device_metrics.csv"), index=False)

    # 7. Summary JSON & TXT
    results_summary = {
        "dataset": "365_days_staggered_10min_sampled.csv",
        "raw_rows": raw_rows,
        "anomalies_removed": anom_count,
        "clean_rows": len(df),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "training_time_seconds": round(train_time, 2),
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "final_test_r2": round(final_r2, 4),
        "final_test_rmse_db": round(final_rmse, 4),
        "pooled_test_rmse_db": round(pooled_rmse, 4),
        "per_device": per_device_records
    }

    with open(os.path.join(OUTPUT_DIR, "centralized_365_results.json"), "w") as f:
        json.dump(results_summary, f, indent=4)

    with open(os.path.join(OUTPUT_DIR, "centralized_365_summary.txt"), "w") as f:
        f.write("=== CENTRALIZED BASELINE SIMULATION SUMMARY (365-DAY 10-MIN DATASET) ===\n")
        f.write(f"Total Rows: {len(df):,} (Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,})\n")
        f.write(f"Training Time: {train_time:.2f} s over {EPOCHS} Epochs\n")
        f.write(f"Final Test R2 Score: {final_r2:.4f}\n")
        f.write(f"Final Test RMSE: {final_rmse:.4f} dB\n")
        f.write(f"Pooled Per-Device Test RMSE: {pooled_rmse:.4f} dB\n\n")
        f.write("Per-Device Performance Breakdown:\n")
        f.write(per_device_df.to_string(index=False) + "\n")

    # 8. High-Resolution Visualizations
    print("\n--- Generating High-Resolution Figures ---")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    epochs_x = epochs_df["epoch"]

    # Subplot 1: R^2 Score
    ax1 = axes[0]
    ax1.plot(epochs_x, epochs_df["train_r2"], label="Training $R^2$", color="#1f77b4", linewidth=2.5)
    ax1.plot(epochs_x, epochs_df["val_r2"], label="Validation $R^2$", color="#ff7f0e", linewidth=2.5, linestyle="--")
    ax1.set_xlabel("Epoch", fontsize=13)
    ax1.set_ylabel("$R^2$ Score", fontsize=13)
    ax1.set_ylim(-1.5, 1.05)
    ax1.legend(loc="lower right", fontsize=11, frameon=True, facecolor="white", framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Subplot 2: RMSE (dB)
    ax2 = axes[1]
    ax2.plot(epochs_x, epochs_df["train_rmse"], label="Training RMSE", color="#1f77b4", linewidth=2.5)
    ax2.plot(epochs_x, epochs_df["val_rmse"], label="Validation RMSE", color="#ff7f0e", linewidth=2.5, linestyle="--")
    ax2.set_xlabel("Epoch", fontsize=13)
    ax2.set_ylabel("RMSE (dB)", fontsize=13)
    ax2.legend(loc="upper right", fontsize=11, frameon=True, facecolor="white", framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "centralized_training_curves.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    total_elapsed = time.time() - t0
    print("\n==========================================================================")
    print(f"CENTRALIZED SIMULATION COMPLETED SUCCESSFULLY IN {total_elapsed:.2f} SECONDS!")
    print(f"All Outputs & Figures Saved to: {OUTPUT_DIR}")
    print("==========================================================================")


if __name__ == "__main__":
    main()
