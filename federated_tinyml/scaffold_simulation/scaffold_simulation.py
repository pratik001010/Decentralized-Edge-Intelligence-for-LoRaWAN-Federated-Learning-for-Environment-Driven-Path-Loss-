"""
scaffold_simulation.py
======================
SCAFFOLD Federated Learning Simulation — Comparison Against FedAvg & FedProx

Master's Thesis:
    "Decentralized Edge Intelligence for LoRaWAN: Federated Learning for
     Environment-Driven Path Loss and Link Quality Modeling"

Author  : Pratik Khadka
Uni     : University of Siegen
Date    : 2025

Algorithm Reference:
    Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for
    Federated Learning" (ICML 2020). https://arxiv.org/abs/1910.06378

What SCAFFOLD adds to FedAvg:
    SCAFFOLD maintains control variates (c, c_i) to correct gradient drift.
    During local training, each device modifies its gradient using:
        g_corrected = g - c_i + c
    Where:
        c    = global control variate (server-side, shared with all clients)
        c_i  = local control variate (device-specific, stored persistently)
    After local training, the client updates its control variate:
        c_i_new = c_i - c + (1 / K*lr) * (W_start - W_end)
        delta_c_i = c_i_new - c_i
    The server aggregates both weights AND control variates:
        W_global   = FedAvg(W_i)
        c_global   = c_global + (1/N) * sum(delta_c_i)

STRICT RULES — NO hallucinated results, numbers, or citations.
All results from this script are REAL, computed from the actual dataset.
"""

import os
import sys
import json
import warnings
import time

# Force unbuffered output so progress shows live
if not sys.stdout.line_buffering:
    sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless — safe for all environments
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — kept IDENTICAL to FedAvg & FedProx for fair comparison
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    # Paths — CSV path relative to FederatedTinyML root (run script from there)
    "csv_path"      : "thesis new/3.cleaned_dataset_per_device.csv",
    "figures_dir"   : "scaffold_simulation/figures",
    "results_json"  : "scaffold_simulation/scaffold_results.json",

    # Feature / target columns (from supervisor paper Eq. 12)
    "feature_cols"  : [
        "log_distance",  # 10 * log10(d / d0)
        "W_brick",       # concrete/brick wall count
        "W_wood",        # wood wall count
        "co2",           # CO₂ sensor (ppm)
        "humidity",      # relative humidity (%)
        "pm25",          # PM2.5 particulate (μg/m³)
        "pressure",      # barometric pressure (hPa)
        "temperature",   # ambient temperature (°C)
        "snr",           # signal-to-noise ratio (dB) — link state
    ],
    "target_col"    : "exp_pl",
    "device_col"    : "device_id",

    # Model architecture Dense(9 → 8 → 1) — 89 parameters
    "hidden_units"  : 8,
    "output_units"  : 1,
    "activation"    : "relu",

    # Training — centralized (identical to FedAvg & FedProx)
    "epochs_central": 50,
    "batch_size"    : 2048,
    "lr"            : 0.001,

    # FL simulation — identical settings for fair comparison
    "fl_rounds"       : 20,
    "fl_local_epochs" : [1, 3, 5],
    "test_split"      : 0.20,
    "random_seed"     : 42,

    # Reference values from prior simulations
    "supervisor_r2"    : 0.8219,
    "supervisor_rmse"  : 8.04,
    "fedavg_best_r2"   : 0.6682,
    "fedavg_best_rmse" : 10.8679,
    "fedprox_best_r2"  : 0.6781,
    "fedprox_best_rmse": 10.7040,

    # Matplotlib style
    "style"         : "seaborn-v0_8-whitegrid",
    "dpi"           : 150,
    "fig_ext"       : "png",
}

DEVICE_LABELS = ["ED0", "ED1", "ED2", "ED3", "ED4", "ED5"]
COLORS        = ["#2D6A9F", "#E07B39", "#3AA66C", "#C0392B", "#8E44AD", "#17A589"]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def savefig(name):
    os.makedirs(CFG["figures_dir"], exist_ok=True)
    path = os.path.join(CFG["figures_dir"], f"{name}.{CFG['fig_ext']}")
    plt.savefig(path, dpi=CFG["dpi"], bbox_inches="tight")
    plt.close("all")
    print(f"  -> Figure saved: {path}")
    return path


def metrics(y_true, y_pred, label=""):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    print(f"  [{label}]  R²={r2:.4f}  RMSE={rmse:.4f} dB  MAE={mae:.4f} dB")
    return {"r2": r2, "rmse": rmse, "mae": mae}


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & PREPROCESSING  (identical to FedAvg & FedProx)
# ─────────────────────────────────────────────────────────────────────────────

def load_and_preprocess():
    print("\n" + "="*65)
    print("STEP 1 — Data Loading & Preprocessing")
    print("="*65)

    csv = CFG["csv_path"]
    if not os.path.exists(csv):
        raise FileNotFoundError(f"Dataset not found: {csv}")

    print(f"  Loading {csv} …")
    t0 = time.time()
    df = pd.read_csv(csv, low_memory=False)
    print(f"  Raw rows: {len(df):,}  Columns: {len(df.columns)}  ({time.time()-t0:.1f}s)")

    raw_count = len(df)

    # Anomaly removal (same 3 deterministic corrupted patterns)
    pa = ((df["co2"]==21547.0) & (df["humidity"]==156.65) &
          (df["temperature"]==174.90) & (df["pressure"]==3.21) &
          (df["pm25"]==33.93))
    pb = ((df["co2"]==16724.0) & (df["humidity"]==210.53) &
          (df["temperature"]==110.76) & (df["pressure"]==317.45) &
          (df["pm25"]==125.57))
    pc = ((df["co2"]==0.0) & (df["humidity"]==0.0) &
          (df["temperature"]==0.0) & (df["pressure"]==508.90) &
          (df["pm25"]==0.0))
    bad = pa | pb | pc
    n_bad = int(bad.sum())
    df = df.loc[~bad].copy()
    print(f"  Removed {n_bad} deterministic anomalous rows")

    # Pressure correction: stored value → hPa
    df["pressure"] = pd.to_numeric(df["pressure"], errors="coerce") * 3.125

    # Feature construction
    df["log_distance"] = 10 * np.log10(pd.to_numeric(df["distance"], errors="coerce").clip(lower=1.0))
    df["W_brick"] = pd.to_numeric(df["c_walls"], errors="coerce")
    df["W_wood"]  = pd.to_numeric(df["w_walls"], errors="coerce")

    before = len(df)
    req_notnull = ["snr", "f_count", "distance", "c_walls", "w_walls"]
    existing = [c for c in req_notnull if c in df.columns]
    if existing:
        df = df.dropna(subset=existing)
    dropped_null = before - len(df)
    print(f"  Dropped {dropped_null} rows with null values in core columns")

    feat_cols  = CFG["feature_cols"]
    target_col = CFG["target_col"]
    device_col = CFG["device_col"]

    required = feat_cols + [target_col, device_col]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for col in feat_cols + [target_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=feat_cols + [target_col])
    df = df[df[target_col].between(50, 200)]

    print(f"  Final usable rows: {len(df):,}")
    print(f"  Devices present: {sorted(df[device_col].unique())}")
    print(f"  exp_pl  ->  min={df[target_col].min():.1f}  "
          f"max={df[target_col].max():.1f}  "
          f"mean={df[target_col].mean():.1f}  "
          f"std={df[target_col].std():.1f} dB")

    summary = {
        "raw_rows"         : raw_count,
        "anomalies_removed": n_bad,
        "null_dropped"     : dropped_null,
        "final_rows"       : len(df),
    }
    return df, summary


# ─────────────────────────────────────────────────────────────────────────────
# 2. MODEL FACTORY  (identical 9→8→1 architecture — 89 parameters)
# ─────────────────────────────────────────────────────────────────────────────

def build_model(lr=None):
    """Dense(9 → 8 → 1) regression model — 89 trainable parameters."""
    lr = lr or CFG["lr"]
    model = keras.Sequential([
        keras.layers.Input(shape=(9,), name="input"),
        keras.layers.Dense(CFG["hidden_units"], activation=CFG["activation"],
                           name="hidden",
                           kernel_initializer=keras.initializers.GlorotUniform(seed=42)),
        keras.layers.Dense(CFG["output_units"], activation="linear",
                           name="output"),
    ], name="path_loss_regressor")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss="mse",
                  metrics=["mae"])
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 3. CENTRALIZED BASELINE  (upper bound reference — identical to FedAvg)
# ─────────────────────────────────────────────────────────────────────────────

def run_centralized(X_train, X_test, y_train, y_test):
    print("\n" + "="*65)
    print("STEP 2 — Centralized Baseline (identical to FedAvg & FedProx)")
    print("="*65)

    model = build_model()
    model.summary()

    t0 = time.time()
    hist = model.fit(
        X_train, y_train,
        epochs=CFG["epochs_central"],
        batch_size=CFG["batch_size"],
        validation_split=0.15,
        verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True)]
    )
    elapsed = time.time() - t0
    print(f"  Training time: {elapsed:.1f}s")

    y_pred = model.predict(X_test, verbose=0).flatten()
    m = metrics(y_test, y_pred, "Centralized NN")

    results = {
        "r2"                 : m["r2"],
        "rmse"               : m["rmse"],
        "mae"                : m["mae"],
        "epochs"             : len(hist.history["loss"]),
        "train_loss_history" : hist.history["loss"],
        "val_loss_history"   : hist.history["val_loss"],
    }
    return model, results, y_pred


# ─────────────────────────────────────────────────────────────────────────────
# 4. FedAvg AGGREGATION (server-side weight averaging — same as before)
# ─────────────────────────────────────────────────────────────────────────────

def fedavg(client_weights, client_n_samples):
    """Weighted average of model weights (McMahan et al., 2017)."""
    total = sum(client_n_samples)
    aggregated = []
    for layer_idx in range(len(client_weights[0])):
        layer = np.zeros_like(client_weights[0][layer_idx])
        for c_idx, w in enumerate(client_weights):
            layer += (client_n_samples[c_idx] / total) * w[layer_idx]
        aggregated.append(layer)
    return aggregated


# ─────────────────────────────────────────────────────────────────────────────
# 5. SCAFFOLD LOCAL TRAINING  (the key algorithmic difference)
# ─────────────────────────────────────────────────────────────────────────────

def train_local_scaffold(model, X_client, y_client,
                         global_weights, global_c, local_c,
                         local_epochs, batch_size, lr):
    """
    SCAFFOLD local training with control variate gradient correction.

    Core gradient update rule per step:
        g_corrected = g - c_i + c
        w = w - lr * g_corrected

    Control variate update after local training (Option II from paper):
        c_i_new = c_i - c + (1 / K*lr) * (W_start - W_end)
        delta_c_i = c_i_new - c_i

    Parameters
    ----------
    model         : Keras model with global weights already set
    X_client      : local feature matrix  (n_samples, 9)
    y_client      : local target vector   (n_samples,)
    global_weights: snapshot of global weights (W_start — frozen)
    global_c      : global control variate c (list of numpy arrays)
    local_c       : local control variate c_i for this client (list of numpy arrays)
    local_epochs  : number of local training epochs (K)
    batch_size    : mini-batch size
    lr            : learning rate (eta_l)

    Returns
    -------
    updated_weights : new local model weights (W_end)
    new_local_c     : updated local control variate (c_i_new)
    delta_c         : change in local control variate (delta_c_i)
    """
    # Convert control variates to TensorFlow constants (frozen during training)
    global_c_tf = [tf.constant(c.astype(np.float32)) for c in global_c]
    local_c_tf  = [tf.constant(c.astype(np.float32)) for c in local_c]

    # Store starting weights W_start for control variate update later
    w_start = [w.copy() for w in global_weights]

    optimizer = keras.optimizers.SGD(learning_rate=lr)

    # Build batched dataset
    dataset = tf.data.Dataset.from_tensor_slices(
        (X_client.astype(np.float32), y_client.astype(np.float32))
    ).shuffle(buffer_size=min(10000, len(X_client)),
              seed=CFG["random_seed"]).batch(batch_size)

    # Count total local steps K (for control variate update formula)
    steps_per_epoch = max(1, len(X_client) // batch_size)
    K = local_epochs * steps_per_epoch

    for epoch in range(local_epochs):
        for X_batch, y_batch in dataset:
            with tf.GradientTape() as tape:
                y_pred = model(X_batch, training=True)
                y_pred = tf.squeeze(y_pred, axis=1)
                mse_loss = tf.reduce_mean(tf.square(y_pred - y_batch))

            # Raw gradients w.r.t. local weights
            raw_grads = tape.gradient(mse_loss, model.trainable_variables)

            # SCAFFOLD gradient correction:
            # g_corrected = g - c_i + c
            corrected_grads = [
                g - c_i + c_g
                for g, c_i, c_g in zip(raw_grads, local_c_tf, global_c_tf)
            ]

            # Apply corrected gradients using SGD
            optimizer.apply_gradients(
                zip(corrected_grads, model.trainable_variables))

    # Capture W_end after all local steps
    w_end = model.get_weights()

    # ── SCAFFOLD Option II: Update local control variate
    # c_i_new = c_i - c + (1 / K*lr) * (W_start - W_end)
    # delta_c_i = c_i_new - c_i
    new_local_c = []
    delta_c     = []
    for c_i, c_g, ws, we in zip(local_c, global_c, w_start, w_end):
        correction = (1.0 / (K * lr)) * (ws - we)
        c_i_new = c_i - c_g + correction
        new_local_c.append(c_i_new)
        delta_c.append(c_i_new - c_i)

    return w_end, new_local_c, delta_c


# ─────────────────────────────────────────────────────────────────────────────
# 6. SCAFFOLD SIMULATION  (replaces run_fl_simulation from FedAvg/FedProx)
# ─────────────────────────────────────────────────────────────────────────────

def run_scaffold_simulation(client_data, X_test, y_test, local_epochs=3):
    """
    Full SCAFFOLD simulation over CFG['fl_rounds'] rounds.
    Structure identical to FedAvg/FedProx but uses control variates.
    """
    print(f"\n  -> SCAFFOLD run: local_epochs={local_epochs}, "
          f"rounds={CFG['fl_rounds']}, clients={len(client_data)}")

    set_seed(CFG["random_seed"])
    global_model = build_model()

    # ── Initialize global weights and global control variate c = 0
    global_weights = global_model.get_weights()
    global_c = [np.zeros_like(w) for w in global_weights]

    # ── Initialize local control variates c_i = 0 for each client
    local_c_dict = {
        dev: [np.zeros_like(w) for w in global_weights]
        for dev in client_data.keys()
    }

    round_r2   = []
    round_rmse = []

    for rnd in range(1, CFG["fl_rounds"] + 1):
        global_weights = global_model.get_weights()  # W to broadcast
        client_weights  = []
        client_ns       = []
        delta_c_list    = []

        for dev, (Xc, yc) in client_data.items():
            # Build fresh local model with global weights
            local = build_model()
            local.set_weights(global_weights)

            # SCAFFOLD local training
            updated_w, new_local_c, delta_c = train_local_scaffold(
                model=local,
                X_client=Xc,
                y_client=yc,
                global_weights=global_weights,
                global_c=global_c,
                local_c=local_c_dict[dev],
                local_epochs=local_epochs,
                batch_size=CFG["batch_size"],
                lr=CFG["lr"],
            )

            # Store results
            client_weights.append(updated_w)
            client_ns.append(len(Xc))
            delta_c_list.append(delta_c)

            # Persist updated local control variate for next round
            local_c_dict[dev] = new_local_c

        # ── Server aggregation: FedAvg for weights
        new_weights = fedavg(client_weights, client_ns)
        global_model.set_weights(new_weights)

        # ── Server aggregation: Update global control variate
        # c = c + (1/N) * sum(delta_c_i)
        N = len(client_data)
        for layer_idx in range(len(global_c)):
            delta_sum = sum(dc[layer_idx] for dc in delta_c_list)
            global_c[layer_idx] = global_c[layer_idx] + (1.0 / N) * delta_sum

        # ── Evaluate on held-out test set
        y_pred = global_model.predict(X_test, verbose=0).flatten()
        r2   = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        round_r2.append(r2)
        round_rmse.append(rmse)

        if rnd % 5 == 0 or rnd == 1:
            print(f"    Round {rnd:2d}/{CFG['fl_rounds']}  "
                  f"R²={r2:.4f}  RMSE={rmse:.2f} dB")

    return global_model, round_r2, round_rmse


# ─────────────────────────────────────────────────────────────────────────────
# 7. RUN ALL SCAFFOLD EXPERIMENTS (E = 1, 3, 5 — same as FedAvg & FedProx)
# ─────────────────────────────────────────────────────────────────────────────

def run_all_scaffold_experiments(client_data, X_test, y_test):
    print("\n" + "="*65)
    print("STEP 3 — SCAFFOLD Simulation")
    print(f"         Rounds={CFG['fl_rounds']}  |  Clients={len(client_data)}")
    print("="*65)

    all_results = {}
    for le in CFG["fl_local_epochs"]:
        model, r2_hist, rmse_hist = run_scaffold_simulation(
            client_data, X_test, y_test, local_epochs=le)

        y_pred_fl = model.predict(X_test, verbose=0).flatten()
        final_m = metrics(y_test, y_pred_fl,
                          f"SCAFFOLD (E={le}, R={CFG['fl_rounds']})")
        all_results[le] = {
            "r2_history"   : r2_hist,
            "rmse_history" : rmse_hist,
            "final_r2"     : final_m["r2"],
            "final_rmse"   : final_m["rmse"],
            "final_mae"    : final_m["mae"],
            "y_pred"       : y_pred_fl,
            "model"        : model,
        }
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# 8. FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence(scaffold_results, central_r2):
    """Figure A — R² and RMSE convergence per round."""
    print("  Plotting Figure A — SCAFFOLD R² vs Communication Round …")
    plt.style.use(CFG["style"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    styles = ["-", "--", ":"]

    ax = axes[0]
    for idx, (le, res) in enumerate(scaffold_results.items()):
        rounds = range(1, len(res["r2_history"]) + 1)
        ax.plot(rounds, res["r2_history"],
                label=f"SCAFFOLD  E={le} local epochs",
                linewidth=2, linestyle=styles[idx], color=COLORS[idx])
    ax.axhline(central_r2, color="black", linewidth=1.8,
               linestyle="-.", label=f"Centralized NN  (R²={central_r2:.4f})")
    ax.axhline(CFG["supervisor_r2"], color="gray", linewidth=1.2,
               linestyle="--", label=f"Reference MLR  (R²={CFG['supervisor_r2']})")
    ax.set_xlabel("Communication Round", fontsize=11)
    ax.set_ylabel("$R^2$", fontsize=11)
    ax.set_title("SCAFFOLD — $R^2$ vs Round", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    ax2 = axes[1]
    for idx, (le, res) in enumerate(scaffold_results.items()):
        rounds = range(1, len(res["rmse_history"]) + 1)
        ax2.plot(rounds, res["rmse_history"],
                 label=f"SCAFFOLD  E={le} local epochs",
                 linewidth=2, linestyle=styles[idx], color=COLORS[idx])
    ax2.axhline(CFG["supervisor_rmse"], color="gray", linewidth=1.2,
                linestyle="--",
                label=f"Reference MLR  (RMSE={CFG['supervisor_rmse']} dB)")
    ax2.set_xlabel("Communication Round", fontsize=11)
    ax2.set_ylabel("RMSE (dB)", fontsize=11)
    ax2.set_title("SCAFFOLD — RMSE vs Round", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)

    plt.suptitle("Figure A — SCAFFOLD Convergence\n"
                 "6 Virtual Clients | Non-IID Split by Device Location",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    return savefig("figA_scaffold_convergence")


def plot_comparison(central_results, scaffold_results):
    """Figure B — SCAFFOLD vs FedProx vs FedAvg vs Centralized bar chart."""
    print("  Plotting Figure B — SCAFFOLD vs FedProx vs FedAvg vs Centralized …")
    plt.style.use(CFG["style"])

    best_le      = max(scaffold_results, key=lambda k: scaffold_results[k]["final_r2"])
    best_scaffold = scaffold_results[best_le]

    labels     = ["Centralized\nNN", "FedAvg\n(best E=5)",
                  "FedProx\n(best E=5)", f"SCAFFOLD\n(best E={best_le})"]
    r2_vals    = [central_results["r2"],
                  CFG["fedavg_best_r2"],
                  CFG["fedprox_best_r2"],
                  best_scaffold["final_r2"]]
    rmse_vals  = [central_results["rmse"],
                  CFG["fedavg_best_rmse"],
                  CFG["fedprox_best_rmse"],
                  best_scaffold["final_rmse"]]
    bar_colors = ["#2D6A9F", "#E07B39", "#8E44AD", "#3AA66C"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    bars = ax.bar(labels, r2_vals, color=bar_colors, edgecolor="white",
                  linewidth=0.8, width=0.45)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("$R^2$", fontsize=12)
    ax.set_title("$R^2$ Comparison", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold")

    ax2 = axes[1]
    bars2 = ax2.bar(labels, rmse_vals, color=bar_colors, edgecolor="white",
                    linewidth=0.8, width=0.45)
    ax2.set_ylabel("RMSE (dB)", fontsize=12)
    ax2.set_title("RMSE Comparison", fontsize=12, fontweight="bold")
    for bar, val in zip(bars2, rmse_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=10,
                 fontweight="bold")

    plt.suptitle("Figure B — SCAFFOLD vs FedProx vs FedAvg vs Centralized\n"
                 "Non-IID Indoor LoRaWAN Dataset  |  6 Edge Devices",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    return savefig("figB_scaffold_comparison")


def plot_per_client_r2(client_data, best_model, X_test, y_test):
    """Figure C — Per-client R² of the best SCAFFOLD global model."""
    print("  Plotting Figure C — Per-client R² (SCAFFOLD) …")
    plt.style.use(CFG["style"])

    client_r2    = []
    client_names = []
    client_sizes = []

    for dev, (Xc, yc) in client_data.items():
        y_c_pred = best_model.predict(Xc, verbose=0).flatten()
        r2 = r2_score(yc, y_c_pred)
        client_r2.append(r2)
        client_names.append(dev)
        client_sizes.append(len(Xc))

    global_r2 = r2_score(y_test,
                          best_model.predict(X_test, verbose=0).flatten())

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_r2 = [max(-1.0, r) for r in client_r2]
    bars = ax.barh(client_names, plot_r2,
                   color=[COLORS[i] for i in range(len(client_names))],
                   edgecolor="white", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=1.0, linestyle="-")
    ax.axvline(global_r2, color="black", linewidth=2, linestyle="--",
               label=f"Global R²={global_r2:.4f}")
    ax.axvline(CFG["supervisor_r2"], color="gray", linewidth=1.5,
               linestyle=":", label=f"Reference MLR R²={CFG['supervisor_r2']}")
    ax.set_xlabel("$R^2$ (SCAFFOLD Global Model evaluated on local data)", fontsize=10)
    ax.set_title("Figure C — SCAFFOLD Per-Client R²\n"
                 "(Non-IID Data: Each Device = Different Room Location)",
                 fontsize=11, fontweight="bold")
    for bar, val, plot_val, n in zip(bars, client_r2, plot_r2, client_sizes):
        if val >= 0:
            ax.text(plot_val + 0.01, bar.get_y() + bar.get_height()/2,
                    f"R²={val:.3f}  (n={n:,})",
                    va="center", ha="left", fontsize=9, color="black")
        else:
            ax.text(plot_val - 0.01, bar.get_y() + bar.get_height()/2,
                    f"R²={val:.3f}  (n={n:,})",
                    va="center", ha="right", fontsize=9, color="darkred")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(-1.5, 1.05)
    plt.tight_layout()
    return savefig("figC_scaffold_per_client_r2")


# ─────────────────────────────────────────────────────────────────────────────
# 9. SAVE RESULTS JSON
# ─────────────────────────────────────────────────────────────────────────────

def save_results(data_summary, central_results, scaffold_results, client_data):
    print("\n  Saving SCAFFOLD results JSON …")
    best_le = max(scaffold_results, key=lambda k: scaffold_results[k]["final_r2"])
    client_info = {d: len(v[0]) for d, v in client_data.items()}

    out = {
        "simulation_meta": {
            "algorithm"          : "SCAFFOLD",
            "timestamp"          : time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tensorflow"         : tf.__version__,
            "fl_rounds"          : CFG["fl_rounds"],
            "local_epochs_tested": CFG["fl_local_epochs"],
            "architecture"       : "Dense(9→8→1)",
            "n_params"           : 89,
        },
        "data_summary"        : data_summary,
        "client_sample_counts": client_info,
        "centralized_baseline": {
            "r2"  : central_results["r2"],
            "rmse": central_results["rmse"],
            "mae" : central_results["mae"],
        },
        "scaffold_results": {
            str(le): {
                "final_r2"  : res["final_r2"],
                "final_rmse": res["final_rmse"],
                "final_mae" : res["final_mae"],
                "r2_relative_to_centralized":
                    res["final_r2"] / central_results["r2"],
            }
            for le, res in scaffold_results.items()
        },
        "best_scaffold_config": {
            "local_epochs": best_le,
            "final_r2"    : scaffold_results[best_le]["final_r2"],
            "final_rmse"  : scaffold_results[best_le]["final_rmse"],
            "r2_drop_vs_centralized_pct":
                (central_results["r2"] - scaffold_results[best_le]["final_r2"])
                / central_results["r2"] * 100,
        },
        "three_way_comparison": {
            "fedavg_best_r2"     : CFG["fedavg_best_r2"],
            "fedavg_best_rmse"   : CFG["fedavg_best_rmse"],
            "fedprox_best_r2"    : CFG["fedprox_best_r2"],
            "fedprox_best_rmse"  : CFG["fedprox_best_rmse"],
            "scaffold_best_r2"   : scaffold_results[best_le]["final_r2"],
            "scaffold_best_rmse" : scaffold_results[best_le]["final_rmse"],
            "scaffold_vs_fedavg_r2_improvement"  :
                scaffold_results[best_le]["final_r2"] - CFG["fedavg_best_r2"],
            "scaffold_vs_fedavg_rmse_improvement":
                CFG["fedavg_best_rmse"] - scaffold_results[best_le]["final_rmse"],
            "scaffold_vs_fedprox_r2_improvement" :
                scaffold_results[best_le]["final_r2"] - CFG["fedprox_best_r2"],
            "scaffold_vs_fedprox_rmse_improvement":
                CFG["fedprox_best_rmse"] - scaffold_results[best_le]["final_rmse"],
        },
    }

    json_path = CFG["results_json"]
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  -> Saved: {json_path}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 10. PRINT THESIS TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_thesis_tables(results):
    print("\n" + "="*65)
    print("SCAFFOLD RESULTS  —  copy directly into LaTeX")
    print("="*65)

    print("\n=== Table: SCAFFOLD vs FedProx vs FedAvg vs Centralized ===")
    hdr = f"{'Model':<38} {'R²':>8} {'RMSE (dB)':>12}"
    print(hdr)
    print("-" * len(hdr))
    cm = results["centralized_baseline"]
    print(f"{'Centralized NN (upper bound)':<38} {cm['r2']:>8.4f} {cm['rmse']:>12.2f}")
    print(f"{'FedAvg best (E=5, baseline)':<38} "
          f"{CFG['fedavg_best_r2']:>8.4f} {CFG['fedavg_best_rmse']:>12.2f}")
    print(f"{'FedProx best (mu=0.01)':<38} "
          f"{CFG['fedprox_best_r2']:>8.4f} {CFG['fedprox_best_rmse']:>12.2f}")
    best = results["best_scaffold_config"]
    print(f"{'SCAFFOLD best':<38} "
          f"{best['final_r2']:>8.4f} {best['final_rmse']:>12.2f}")

    print("\n=== Table: SCAFFOLD — Per Local Epoch Setting ===")
    print(f"{'Local Epochs':<15} {'Final R²':>10} {'RMSE (dB)':>12} {'R² / Central':>14}")
    for le, v in results["scaffold_results"].items():
        print(f"E={le:<13} {v['final_r2']:>10.4f} {v['final_rmse']:>12.2f} "
              f"{v['r2_relative_to_centralized']:>14.4f}")

    cmp = results["three_way_comparison"]
    print("\n=== SCAFFOLD Improvement over FedAvg ===")
    print(f"  R² improvement:   {cmp['scaffold_vs_fedavg_r2_improvement']:+.4f}")
    print(f"  RMSE improvement: {cmp['scaffold_vs_fedavg_rmse_improvement']:+.2f} dB")
    print("\n=== SCAFFOLD Improvement over FedProx ===")
    print(f"  R² improvement:   {cmp['scaffold_vs_fedprox_r2_improvement']:+.4f}")
    print(f"  RMSE improvement: {cmp['scaffold_vs_fedprox_rmse_improvement']:+.2f} dB")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    set_seed(CFG["random_seed"])
    os.makedirs(CFG["figures_dir"], exist_ok=True)

    print("\n" + "="*65)
    print("SCAFFOLD SIMULATION — Master's Thesis, Pratik Khadka")
    print("University of Siegen")
    print("Dataset: Indoor LoRaWAN, Hölderlinstraße Campus")
    print("Algorithm: SCAFFOLD (Karimireddy et al., ICML 2020)")
    print("="*65)

    # ── 1. Load and preprocess data
    df, data_summary = load_and_preprocess()

    # ── 2. Feature preparation and scaling
    print("\n  Preparing features and scaling …")
    feat_cols  = CFG["feature_cols"]
    target_col = CFG["target_col"]
    device_col = CFG["device_col"]

    X_all   = df[feat_cols].values.astype(np.float32)
    y_all   = df[target_col].values.astype(np.float32)
    dev_all = df[device_col].values

    scaler = StandardScaler()
    X_all_norm = scaler.fit_transform(X_all)
    print(f"  Feature means: {dict(zip(feat_cols, scaler.mean_.round(4)))}")
    print(f"  Feature stds:  {dict(zip(feat_cols, scaler.scale_.round(4)))}")

    # ── 3. Train/test split — same seed as FedAvg & FedProx
    X_train_all, X_test, y_train_all, y_test, dev_train, _ = train_test_split(
        X_all_norm, y_all, dev_all,
        test_size=CFG["test_split"],
        random_state=CFG["random_seed"],
    )
    print(f"  Train: {len(X_train_all):,}  Test: {len(X_test):,}")

    # ── 4. Centralized baseline
    central_model, central_results, y_pred_central = run_centralized(
        X_train_all, X_test, y_train_all, y_test)

    # ── 5. Build Non-IID client data partitions (by device_id)
    print("\n  Building non-IID client data partitions …")
    client_data = {}
    for dev in DEVICE_LABELS:
        mask = dev_train == dev
        if mask.sum() > 0:
            client_data[dev] = (X_train_all[mask], y_train_all[mask])
            print(f"    {dev}: {mask.sum():,} samples")
        else:
            print(f"    {dev}: NOT FOUND in training set — skipping")
    data_summary["client_sample_counts"] = {d: len(v[0]) for d, v in client_data.items()}

    # ── 6. SCAFFOLD Simulation
    scaffold_results = run_all_scaffold_experiments(client_data, X_test, y_test)

    # ── 7. Figures
    figA = plot_convergence(scaffold_results, central_results["r2"])
    figB = plot_comparison(central_results, scaffold_results)

    best_le    = max(scaffold_results, key=lambda k: scaffold_results[k]["final_r2"])
    best_model = scaffold_results[best_le]["model"]
    figC = plot_per_client_r2(client_data, best_model, X_test, y_test)

    # ── 8. Save results JSON
    results = save_results(data_summary, central_results, scaffold_results, client_data)

    # ── 9. Print thesis tables
    print_thesis_tables(results)

    elapsed = time.time() - t_start
    print(f"\n{'='*65}")
    print(f"SCAFFOLD SIMULATION COMPLETE in {elapsed/60:.1f} minutes")
    print(f"Figures saved to: {CFG['figures_dir']}/")
    print(f"Results JSON:     {CFG['results_json']}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
