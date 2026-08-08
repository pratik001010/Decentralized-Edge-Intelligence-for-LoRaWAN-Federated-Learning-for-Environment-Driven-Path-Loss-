"""
train_model.py
==============
Canonical Baseline Model Training & TFLite Conversion Script for Arduino Deployment.

Master's Thesis:
    "Decentralized Edge Intelligence for LoRaWAN: Federated Learning for
     Environment-Driven Path Loss and Link Quality Modeling"

Author  : Pratik Khadka
Uni     : University of Siegen
Date    : 2025/2026

This script:
1. Loads the canonical 10-minute dataset (365_days_staggered_10min_sampled.csv, ~206,957 rows)
2. Trains the canonical Dense 9-8-1 MLP path-loss regressor (89 parameters)
3. Fits StandardScaler strictly on training split (zero data leakage)
4. Exports TFLite FlatBuffer model (model.tflite) and generates C header (model.h)
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
from tensorflow import keras

# Unbuffer output for real-time progress visibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__)) if __file__ else "."
DATASET_FILE = "365_days_staggered_10min_sampled.csv"

FEATURE_COLS = [
    "log_distance",  # 10 * log10(d / d0)
    "W_brick",       # c_walls (brick/concrete walls)
    "W_wood",        # w_walls (wooden partitions)
    "co2",           # CO2 concentration (ppm)
    "humidity",      # Relative humidity (%)
    "pm25",          # PM2.5 (ug/m3)
    "pressure",      # Pressure in hPa (raw * 3.125)
    "temperature",   # Temperature (deg C)
    "snr",           # Gateway SNR (dB)
]

TARGET_COL = "exp_pl"
DEVICE_COL = "device_id"

EPOCHS = 50
BATCH_SIZE = 512
LEARNING_RATE = 0.01


def load_and_preprocess_data():
    dataset_path = None
    possible_paths = [
        os.path.join(OUTPUT_DIR, DATASET_FILE),
        os.path.join(OUTPUT_DIR, "..", DATASET_FILE),
        r"c:\Users\prati\Desktop\edge AI\FederatedTinyML\365_days_staggered_10min_sampled.csv",
    ]
    for p in possible_paths:
        if os.path.exists(p):
            dataset_path = p
            break

    if dataset_path is None:
        raise FileNotFoundError(f"Canonical dataset '{DATASET_FILE}' not found in search paths.")

    print(f"Loading canonical dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path, low_memory=False)

    if "dev_id" in df.columns and "device_id" not in df.columns:
        df["device_id"] = df["dev_id"]

    # Filter known anomalies
    pa = ((df["co2"] == 21547.0) & (df["humidity"] == 156.65) & (df["temperature"] == 174.90) & (df["pressure"] == 3.21) & (df["pm25"] == 33.93))
    pb = ((df["co2"] == 16724.0) & (df["humidity"] == 210.53) & (df["temperature"] == 110.76) & (df["pressure"] == 317.45) & (df["pm25"] == 125.57))
    pc = ((df["co2"] == 0.0) & (df["humidity"] == 0.0) & (df["temperature"] == 0.0) & (df["pressure"] == 508.90) & (df["pm25"] == 0.0))
    bad = pa | pb | pc
    df = df.loc[~bad].copy()

    # Pressure scaling to hPa
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
    return df


def build_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(len(FEATURE_COLS),), name="input"),
        keras.layers.Dense(8, activation="relu", name="hidden",
                           kernel_initializer=keras.initializers.GlorotUniform(seed=SEED)),
        keras.layers.Dense(1, activation="linear", name="output",
                           kernel_initializer=keras.initializers.GlorotUniform(seed=SEED))
    ], name="dense_9_8_1_mlp")
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss="mse")
    return model


def main():
    print("==========================================================================")
    print("CANONICAL MODEL TRAINING & TFLITE EXPORT (365-DAY 10-MIN DATASET)")
    print("==========================================================================")

    df = load_and_preprocess_data()

    X_all = df[FEATURE_COLS].values.astype(np.float32)
    y_all = df[TARGET_COL].values.astype(np.float32)
    dev_all = df[DEVICE_COL].values

    # Step 1: Split 80% train_val, 20% test
    X_tv_raw, X_test_raw, y_tv, y_test, dev_tv, dev_test = train_test_split(
        X_all, y_all, dev_all, test_size=0.20, random_state=SEED, stratify=dev_all
    )

    # Step 2: Split train_val into 85% train, 15% val
    X_tr_raw, X_val_raw, y_tr, y_val, dev_tr, dev_val = train_test_split(
        X_tv_raw, y_tv, dev_tv, test_size=0.15, random_state=SEED, stratify=dev_tv
    )

    print(f"Train: {len(X_tr_raw):,} | Val: {len(X_val_raw):,} | Test: {len(X_test_raw):,}")

    # Scaler fitted STRICTLY on X_tr_raw
    scaler = StandardScaler()
    X_tr  = scaler.fit_transform(X_tr_raw)
    X_val = scaler.transform(X_val_raw)
    X_te  = scaler.transform(X_test_raw)

    model = build_model()
    model.summary()

    print(f"\n--- Training Dense 9-8-1 MLP for {EPOCHS} Epochs ---")
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    y_pred = model.predict(X_te, verbose=0).flatten()
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n==========================================================================")
    print(f"MODEL EVALUATION — Test R^2: {r2:.4f} | Test RMSE: {rmse:.4f} dB")
    print("==========================================================================")

    # Convert to TFLite FlatBuffer
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    tflite_path = os.path.join(OUTPUT_DIR, "model.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"Exported TFLite Model: {tflite_path} ({len(tflite_model)} bytes)")

    # Print feature normalization constants for C++ firmware sync
    print("\nScaler Constants for C++ Firmware (featureMeans and featureStds):")
    print(f"float featureMeans[{len(FEATURE_COLS)}] = {{{', '.join([f'{m:.6f}' for m in scaler.mean_])}}};")
    print(f"float featureStds[{len(FEATURE_COLS)}]  = {{{', '.join([f'{s:.6f}' for s in scaler.scale_])}}};")


if __name__ == "__main__":
    main()
