"""
fedprox_simulation.py
=====================
FedProx Federated Learning Simulation — Comparison Against FedAvg Baseline

Master's Thesis:
    "Decentralized Edge Intelligence for LoRaWAN: Federated Learning for
     Environment-Driven Path Loss and Link Quality Modeling"

Author  : Pratik Khadka
Uni     : University of Siegen
Date    : 2025

Algorithm Reference:
    Li et al., "Federated Optimization in Heterogeneous Networks"
    (FedProx, ICLR 2020). https://arxiv.org/abs/1812.06127

What FedProx adds to FedAvg:
    A proximal regularization term is added to the local loss function:
        L_fedprox = MSE(y_pred, y_true) + (mu / 2) * ||w - w_global||²
    This "rubber band" penalty prevents local models from drifting too far
    from the global model during local training, which is the root cause
    of RMSE degradation on Non-IID data (Client Drift).

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
# CONFIGURATION — kept IDENTICAL to FedAvg simulation for fair comparison
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    # Paths — CSV path relative to FederatedTinyML root (run script from there)
    "csv_path"      : "thesis new/3.cleaned_dataset_per_device.csv",
    "figures_dir"   : "fedprox_simulation/figures",
    "results_json"  : "fedprox_simulation/fedprox_results.json",

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

    # Training — centralized (identical to FedAvg)
    "epochs_central": 50,
    "batch_size"    : 2048,
    "lr"            : 0.001,

    # FL simulation — identical to FedAvg for fair comparison
    "fl_rounds"       : 20,
    "fl_local_epochs" : [1, 3, 5],
    "test_split"      : 0.20,
    "random_seed"     : 42,

    # ── FedProx-specific hyperparameter ──────────────────────────────────────
    # mu controls the strength of the proximal penalty.
    # mu = 0.0 → degenerates to standard FedAvg (no regularization)
    # mu = 0.01 → light regularization (recommended start for Non-IID)
    # mu = 0.1  → stronger constraint — may slow learning but reduces drift
    # We test multiple values to find the best one for this dataset.
    "fedprox_mu"    : 0.01,   # primary mu value used for result reporting

    # Reference values from literature (for comparison plots)
    "supervisor_r2"   : 0.8219,
    "supervisor_rmse" : 8.04,
    "fedavg_best_r2"  : 0.6682,   # from our FedAvg simulation (E=5)
    "fedavg_best_rmse": 10.8679,  # from our FedAvg simulation (E=5)

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
# 1. DATA LOADING & PREPROCESSING  (identical to FedAvg)
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

    # Anomaly removal (same 3 deterministic corrupted patterns as FedAvg)
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

    # Pressure correction: stored value → hPa (factor 3.125, confirmed)
    df["pressure"] = pd.to_numeric(df["pressure"], errors="coerce") * 3.125

    # Feature construction (identical to FedAvg)
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
    df = df[df[target_col].between(50, 200)]   # physically plausible PL range

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
# 2. MODEL FACTORY  (identical architecture to FedAvg — 89 parameters)
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
# 3. CENTRALIZED BASELINE  (identical to FedAvg — upper bound reference)
# ─────────────────────────────────────────────────────────────────────────────

def run_centralized(X_train, X_test, y_train, y_test):
    print("\n" + "="*65)
    print("STEP 2 — Centralized Baseline (identical to FedAvg)")
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
# 4. FedAvg AGGREGATION  (same weighted average as baseline)
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
# 5. FEDPROX LOCAL TRAINING (the key difference from FedAvg)
# ─────────────────────────────────────────────────────────────────────────────

def train_local_fedprox(model, X_client, y_client, global_weights, mu,
                        local_epochs, batch_size, lr):
    """
    FedProx local training with proximal regularization.

    Loss = MSE(y_pred, y_true) + (mu / 2) * sum(||w_k - w_global||²)

    Parameters
    ----------
    model         : freshly initialized Keras model with global weights set
    X_client      : local feature matrix  (n_samples, 9)
    y_client      : local target vector   (n_samples,)
    global_weights: frozen copy of global model weights (anchor)
    mu            : proximal regularization strength (FedProx hyperparameter)
    local_epochs  : number of local training epochs
    batch_size    : mini-batch size
    lr            : learning rate

    Returns
    -------
    Updated model weights after local FedProx training
    """
    optimizer = keras.optimizers.Adam(learning_rate=lr)

    # Convert global weights to tensors once (frozen anchor — do NOT update)
    global_w_tensors = [tf.constant(w, dtype=tf.float32)
                        for w in global_weights]

    # Build a tf.data.Dataset for efficient batching
    dataset = tf.data.Dataset.from_tensor_slices(
        (X_client.astype(np.float32), y_client.astype(np.float32))
    ).shuffle(buffer_size=min(10000, len(X_client)),
              seed=CFG["random_seed"]).batch(batch_size)

    for epoch in range(local_epochs):
        for X_batch, y_batch in dataset:
            with tf.GradientTape() as tape:
                # ── Forward pass
                y_pred = model(X_batch, training=True)
                y_pred = tf.squeeze(y_pred, axis=1)

                # ── Standard MSE loss
                mse_loss = tf.reduce_mean(tf.square(y_pred - y_batch))

                # ── Proximal term: (mu / 2) * ||w - w_global||²
                # Sum over ALL trainable weight tensors in the model
                prox_term = tf.constant(0.0, dtype=tf.float32)
                for w_local, w_global in zip(model.trainable_variables,
                                             global_w_tensors):
                    diff = w_local - w_global
                    prox_term = prox_term + tf.reduce_sum(tf.square(diff))
                prox_term = (mu / 2.0) * prox_term

                # ── Total FedProx loss
                total_loss = mse_loss + prox_term

            # ── Compute gradients w.r.t. local weights only
            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return model.get_weights()


# ─────────────────────────────────────────────────────────────────────────────
# 6. FEDPROX SIMULATION  (replaces run_fl_simulation from FedAvg)
# ─────────────────────────────────────────────────────────────────────────────

def run_fedprox_simulation(client_data, X_test, y_test,
                           local_epochs=3, mu=None):
    """
    Full FedProx simulation over CFG['fl_rounds'] rounds.
    Structure identical to FedAvg but local training uses proximal penalty.
    """
    if mu is None:
        mu = CFG["fedprox_mu"]

    print(f"\n  -> FedProx run: mu={mu}, local_epochs={local_epochs}, "
          f"rounds={CFG['fl_rounds']}, clients={len(client_data)}")

    set_seed(CFG["random_seed"])
    global_model = build_model()

    round_r2   = []
    round_rmse = []

    for rnd in range(1, CFG["fl_rounds"] + 1):
        global_weights = global_model.get_weights()  # broadcast to clients
        client_weights = []
        client_ns      = []

        for dev, (Xc, yc) in client_data.items():
            # Build a fresh local model and load global weights
            local = build_model()
            local.set_weights(global_weights)

            # FedProx local training (key difference from FedAvg)
            updated_weights = train_local_fedprox(
                model=local,
                X_client=Xc,
                y_client=yc,
                global_weights=global_weights,  # frozen anchor
                mu=mu,
                local_epochs=local_epochs,
                batch_size=CFG["batch_size"],
                lr=CFG["lr"],
            )
            client_weights.append(updated_weights)
            client_ns.append(len(Xc))

        # Server-side FedAvg aggregation (same as standard FedAvg)
        new_weights = fedavg(client_weights, client_ns)
        global_model.set_weights(new_weights)

        # Evaluate global model on held-out test set
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
# 7. RUN ALL FEDPROX EXPERIMENTS  (E = 1, 3, 5 — same as FedAvg)
# ─────────────────────────────────────────────────────────────────────────────

def run_all_fedprox_experiments(client_data, X_test, y_test):
    print("\n" + "="*65)
    print("STEP 3 — FedProx Simulation")
    print(f"         mu={CFG['fedprox_mu']}  |  Rounds={CFG['fl_rounds']}")
    print("="*65)

    all_results = {}
    for le in CFG["fl_local_epochs"]:
        model, r2_hist, rmse_hist = run_fedprox_simulation(
            client_data, X_test, y_test, local_epochs=le,
            mu=CFG["fedprox_mu"])

        y_pred_fl = model.predict(X_test, verbose=0).flatten()
        final_m = metrics(y_test, y_pred_fl,
                          f"FedProx (E={le}, mu={CFG['fedprox_mu']},"
                          f" R={CFG['fl_rounds']})")
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

def plot_convergence(fedprox_results, central_r2):
    """Figure A — R² and RMSE convergence per round for all E settings."""
    print("  Plotting Figure A — FedProx R² vs Communication Round …")
    plt.style.use(CFG["style"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    styles = ["-", "--", ":"]

    # R² subplot
    ax = axes[0]
    for idx, (le, res) in enumerate(fedprox_results.items()):
        rounds = range(1, len(res["r2_history"]) + 1)
        ax.plot(rounds, res["r2_history"],
                label=f"FedProx  E={le} local epochs",
                linewidth=2, linestyle=styles[idx], color=COLORS[idx])
    ax.axhline(central_r2, color="black", linewidth=1.8,
               linestyle="-.", label=f"Centralized NN  (R²={central_r2:.4f})")
    ax.axhline(CFG["supervisor_r2"], color="gray", linewidth=1.2,
               linestyle="--", label=f"Reference MLR  (R²={CFG['supervisor_r2']})")
    ax.set_xlabel("Communication Round", fontsize=11)
    ax.set_ylabel("$R^2$", fontsize=11)
    ax.set_title("FedProx — $R^2$ vs Round", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    # RMSE subplot
    ax2 = axes[1]
    for idx, (le, res) in enumerate(fedprox_results.items()):
        rounds = range(1, len(res["rmse_history"]) + 1)
        ax2.plot(rounds, res["rmse_history"],
                 label=f"FedProx  E={le} local epochs",
                 linewidth=2, linestyle=styles[idx], color=COLORS[idx])
    ax2.axhline(CFG["supervisor_rmse"], color="gray", linewidth=1.2,
                linestyle="--",
                label=f"Reference MLR  (RMSE={CFG['supervisor_rmse']} dB)")
    ax2.set_xlabel("Communication Round", fontsize=11)
    ax2.set_ylabel("RMSE (dB)", fontsize=11)
    ax2.set_title("FedProx — RMSE vs Round", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)

    plt.suptitle(f"Figure A — FedProx Convergence  (µ={CFG['fedprox_mu']})\n"
                 "6 Virtual Clients | Non-IID Split by Device Location",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    return savefig("figA_fedprox_convergence")


def plot_comparison(central_results, fedprox_results):
    """Figure B — FedProx vs FedAvg vs Centralized bar chart comparison."""
    print("  Plotting Figure B — FedProx vs FedAvg vs Centralized …")
    plt.style.use(CFG["style"])

    best_le     = max(fedprox_results, key=lambda k: fedprox_results[k]["final_r2"])
    best_fedprox = fedprox_results[best_le]

    labels     = ["Centralized\nNN", "FedAvg\n(best E=5)", f"FedProx\n(best E={best_le})"]
    r2_vals    = [central_results["r2"],
                  CFG["fedavg_best_r2"],
                  best_fedprox["final_r2"]]
    rmse_vals  = [central_results["rmse"],
                  CFG["fedavg_best_rmse"],
                  best_fedprox["final_rmse"]]
    bar_colors = ["#2D6A9F", "#E07B39", "#3AA66C"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # R² bars
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

    # RMSE bars
    ax2 = axes[1]
    bars2 = ax2.bar(labels, rmse_vals, color=bar_colors, edgecolor="white",
                    linewidth=0.8, width=0.45)
    ax2.set_ylabel("RMSE (dB)", fontsize=12)
    ax2.set_title("RMSE Comparison", fontsize=12, fontweight="bold")
    for bar, val in zip(bars2, rmse_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=10,
                 fontweight="bold")

    plt.suptitle(f"Figure B — FedProx (µ={CFG['fedprox_mu']}) vs FedAvg vs Centralized\n"
                 f"Non-IID Indoor LoRaWAN Dataset  |  6 Edge Devices",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    return savefig("figB_fedprox_vs_fedavg_comparison")


def plot_per_client_r2(client_data, best_model, X_test, y_test):
    """Figure C — Per-client R² of the best FedProx global model."""
    print("  Plotting Figure C — Per-client R² (FedProx) …")
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
    plot_r2 = [max(-1.0, r) for r in client_r2]  # clip extreme negatives
    bars = ax.barh(client_names, plot_r2,
                   color=[COLORS[i] for i in range(len(client_names))],
                   edgecolor="white", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=1.0, linestyle="-")
    ax.axvline(global_r2, color="black", linewidth=2, linestyle="--",
               label=f"Global R²={global_r2:.4f}")
    ax.axvline(CFG["supervisor_r2"], color="gray", linewidth=1.5,
               linestyle=":", label=f"Reference MLR R²={CFG['supervisor_r2']}")
    ax.set_xlabel("$R^2$ (FedProx Global Model evaluated on local data)", fontsize=10)
    ax.set_title("Figure C — FedProx Per-Client R²\n"
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
    return savefig("figC_fedprox_per_client_r2")


# ─────────────────────────────────────────────────────────────────────────────
# 9. SAVE RESULTS JSON
# ─────────────────────────────────────────────────────────────────────────────

def save_results(data_summary, central_results, fedprox_results, client_data):
    print("\n  Saving FedProx results JSON …")
    best_le = max(fedprox_results, key=lambda k: fedprox_results[k]["final_r2"])
    client_info = {d: len(v[0]) for d, v in client_data.items()}

    out = {
        "simulation_meta": {
            "algorithm"      : "FedProx",
            "mu"             : CFG["fedprox_mu"],
            "timestamp"      : time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tensorflow"     : tf.__version__,
            "fl_rounds"      : CFG["fl_rounds"],
            "local_epochs_tested": CFG["fl_local_epochs"],
            "architecture"   : "Dense(9→8→1)",
            "n_params"       : 89,
        },
        "data_summary"        : data_summary,
        "client_sample_counts": client_info,
        "centralized_baseline": {
            "r2"  : central_results["r2"],
            "rmse": central_results["rmse"],
            "mae" : central_results["mae"],
        },
        "fedprox_results": {
            str(le): {
                "final_r2"  : res["final_r2"],
                "final_rmse": res["final_rmse"],
                "final_mae" : res["final_mae"],
                "r2_relative_to_centralized":
                    res["final_r2"] / central_results["r2"],
            }
            for le, res in fedprox_results.items()
        },
        "best_fedprox_config": {
            "local_epochs": best_le,
            "final_r2"    : fedprox_results[best_le]["final_r2"],
            "final_rmse"  : fedprox_results[best_le]["final_rmse"],
            "r2_drop_vs_centralized_pct":
                (central_results["r2"] - fedprox_results[best_le]["final_r2"])
                / central_results["r2"] * 100,
        },
        "comparison_vs_fedavg": {
            "fedavg_best_r2"    : CFG["fedavg_best_r2"],
            "fedavg_best_rmse"  : CFG["fedavg_best_rmse"],
            "fedprox_best_r2"   : fedprox_results[best_le]["final_r2"],
            "fedprox_best_rmse" : fedprox_results[best_le]["final_rmse"],
            "r2_improvement"    : fedprox_results[best_le]["final_r2"] - CFG["fedavg_best_r2"],
            "rmse_improvement"  : CFG["fedavg_best_rmse"] - fedprox_results[best_le]["final_rmse"],
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
    print("FEDPROX RESULTS  —  copy directly into LaTeX")
    print("="*65)

    print("\n=== Table: FedProx vs FedAvg vs Centralized ===")
    hdr = f"{'Model':<35} {'R²':>8} {'RMSE (dB)':>12}"
    print(hdr)
    print("-" * len(hdr))
    cm = results["centralized_baseline"]
    print(f"{'Centralized NN (upper bound)':<35} {cm['r2']:>8.4f} {cm['rmse']:>12.2f}")
    print(f"{'FedAvg best (E=5, baseline)':<35} "
          f"{CFG['fedavg_best_r2']:>8.4f} {CFG['fedavg_best_rmse']:>12.2f}")
    best = results["best_fedprox_config"]
    print(f"{'FedProx best (mu=' + str(CFG['fedprox_mu']) + ')':<35} "
          f"{best['final_r2']:>8.4f} {best['final_rmse']:>12.2f}")

    print("\n=== Table: FedProx — Per Local Epoch Setting ===")
    print(f"{'Local Epochs':<15} {'Final R²':>10} {'RMSE (dB)':>12} {'R² / Central':>14}")
    for le, v in results["fedprox_results"].items():
        print(f"E={le:<13} {v['final_r2']:>10.4f} {v['final_rmse']:>12.2f} "
              f"{v['r2_relative_to_centralized']:>14.4f}")

    cmp = results["comparison_vs_fedavg"]
    print("\n=== FedProx Improvement over FedAvg ===")
    print(f"  R² improvement:   {cmp['r2_improvement']:+.4f}")
    print(f"  RMSE improvement: {cmp['rmse_improvement']:+.2f} dB")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    set_seed(CFG["random_seed"])
    os.makedirs(CFG["figures_dir"], exist_ok=True)

    print("\n" + "="*65)
    print("FEDPROX SIMULATION — Master's Thesis, Pratik Khadka")
    print("University of Siegen")
    print(f"Dataset: Indoor LoRaWAN, Hölderlinstraße Campus")
    print(f"Algorithm: FedProx  (µ = {CFG['fedprox_mu']})")
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

    # ── 3. Train/test split — same seed as FedAvg for identical splits
    X_train_all, X_test, y_train_all, y_test, dev_train, _ = train_test_split(
        X_all_norm, y_all, dev_all,
        test_size=CFG["test_split"],
        random_state=CFG["random_seed"],
    )
    print(f"  Train: {len(X_train_all):,}  Test: {len(X_test):,}")

    # ── 4. Centralized baseline (upper bound — identical to FedAvg run)
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

    # ── 6. FedProx Simulation
    fedprox_results = run_all_fedprox_experiments(client_data, X_test, y_test)

    # ── 7. Figures
    figA = plot_convergence(fedprox_results, central_results["r2"])
    figB = plot_comparison(central_results, fedprox_results)

    best_le    = max(fedprox_results, key=lambda k: fedprox_results[k]["final_r2"])
    best_model = fedprox_results[best_le]["model"]
    figC = plot_per_client_r2(client_data, best_model, X_test, y_test)

    # ── 8. Save results JSON
    results = save_results(data_summary, central_results, fedprox_results, client_data)

    # ── 9. Print thesis tables
    print_thesis_tables(results)

    elapsed = time.time() - t_start
    print(f"\n{'='*65}")
    print(f"FEDPROX SIMULATION COMPLETE in {elapsed/60:.1f} minutes")
    print(f"Figures saved to: {CFG['figures_dir']}/")
    print(f"Results JSON:     {CFG['results_json']}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
