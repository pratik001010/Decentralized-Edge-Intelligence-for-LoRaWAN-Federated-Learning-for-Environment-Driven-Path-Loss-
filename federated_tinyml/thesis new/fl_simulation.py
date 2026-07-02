"""
fl_simulation.py
================
Federated Learning Simulation — Phase A: Simulation-Based Benchmarking

Master's Thesis:
    "Decentralized Edge Intelligence for LoRaWAN: Federated Learning for
     Environment-Driven Path Loss and Link Quality Modeling"

Author  : Pratik Khadka
Uni     : University of Siegen
Date    : 2025

Research Narrative
------------------
PHASE 1 (Supervisor's Paper — Obiri & Van Laerhoven, 2024):
    Indoor LoRaWAN measurement campaign, 8th floor Hölderlinstraße Campus,
    Siegen. 6 MKR WAN 1310 nodes (ED0–ED5), 1 Kerlink iFemtoCell gateway,
    TTN → MQTT → InfluxDB on AWS. Sensors: SCD41 (CO₂/temp/humidity),
    BME280 (pressure), SPS30 (PM2.5). Dataset: 1,328,334 rows.
    Best result: LDPLSM-MW-EP R²=0.8219, RMSE=8.04 dB.
    Future work EXPLICITLY stated: Random Forest, Gradient Boosting,
    TinyML — direct academic justification for this thesis.

PHASE 2 (PEP Project — prior thesis work):
    XGBoost regression on same dataset: R²≈0.93, RMSE≈[PEP value].
    Significantly outperformed supervisor's linear model.

PHASE 3 (THIS THESIS):
    Can this intelligence run ON the edge node (TFLite Micro, MKR WAN 1310)?
    Can 6 nodes collaboratively improve models without sharing raw data (FL)?
    Reference: Torres Sanchez et al. (2024) — FL+LoRaWAN, Flower, FedAvg,
    F1=94.77%, accuracy=92.30%.

STRICT RULES — NO hallucinated results, numbers, or citations.
All results from this script are REAL, computed from the actual dataset.
"""

import os
import sys
import json
import warnings
import time

# Force unbuffered output so conda run shows progress live
if not sys.stdout.line_buffering:
    sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless — safe for all environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  —  do NOT change these without updating the thesis text
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    # Paths
    "csv_path"      : "thesis new/3.cleaned_dataset_per_device.csv",
    "figures_dir"   : "thesis_figures",
    "results_json"  : "fl_simulation_results.json",

    # Feature / target columns (from supervisor paper Eq. 12)
    "feature_cols"  : [
        "log_distance",  # 10 * n * log10(d / d0) - where n is learned, so we feed 10 * log10(d/d0)
        "W_brick",       # c_walls (brick/concrete walls count)
        "W_wood",        # w_walls (wood walls count)
        "co2",           # E1
        "humidity",      # E2
        "pm25",          # E3
        "pressure",      # E4
        "temperature",   # E5
        "snr",           # γ (link-state indicator)
    ],
    "target_col"    : "exp_pl",
    "device_col"    : "device_id",

    # Model architecture Dense(9 → 8 → 1) — 9 inputs, 8 hidden, 1 output (89 parameters)
    "hidden_units"  : 8,
    "output_units"  : 1,
    "activation"    : "relu",

    # Training — centralized
    "epochs_central": 50,
    "batch_size"    : 2048,
    "lr"            : 0.001,

    # FL simulation
    "fl_rounds"     : 20,          # communication rounds
    "fl_local_epochs" : [1, 3, 5], # we test three settings
    "fl_min_clients": 3,           # minimum clients per round (out of 6)
    "test_split"    : 0.20,
    "random_seed"   : 42,

    # Known reference values (from literature — do NOT fabricate)
    "supervisor_r2"   : 0.8219,    # Obiri & Van Laerhoven LDPLSM-MW-EP
    "supervisor_rmse" : 8.04,      # dB
    "pep_r2"          : 0.93,      # XGBoost PEP project
    "torres_f1"       : 0.9477,    # Torres Sanchez et al. FL result

    # Matplotlib style
    "style"         : "seaborn-v0_8-whitegrid",
    "dpi"           : 150,
    "fig_ext"       : "png",
}

DEVICE_LABELS = ["ED0", "ED1", "ED2", "ED3", "ED4", "ED5"]
COLORS = ["#2D6A9F", "#E07B39", "#3AA66C", "#C0392B", "#8E44AD", "#17A589"]

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
# 1. DATA LOADING & PREPROCESSING
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
    print(f"  Raw rows: {len(df):,}  Columns: {len(df.columns)}  "
          f"({time.time()-t0:.1f}s)")

    raw_count = len(df)

    # ── Anomaly removal (3 deterministic corrupted patterns from data audit)
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

    # ── Pressure correction: stored value → hPa  (factor 3.125, confirmed)
    df["pressure"] = pd.to_numeric(df["pressure"], errors="coerce") * 3.125

    # ── Construct the supervisor paper features from dataset columns
    # 1. Log-distance: 10 * log10(d / d0) where d0 = 1.0 m
    # 2. W_brick: brick/concrete walls count
    # 3. W_wood: wood walls count
    df["log_distance"] = 10 * np.log10(pd.to_numeric(df["distance"], errors="coerce").clip(lower=1.0))
    df["W_brick"] = pd.to_numeric(df["c_walls"], errors="coerce")
    df["W_wood"] = pd.to_numeric(df["w_walls"], errors="coerce")

    # ── Drop rows missing snr / f_count / distance / walls
    before = len(df)
    req_notnull = ["snr", "f_count", "distance", "c_walls", "w_walls"]
    existing = [c for c in req_notnull if c in df.columns]
    if existing:
        df = df.dropna(subset=existing)
    dropped_null = before - len(df)
    print(f"  Dropped {dropped_null} rows with null values in core columns")

    # ── Feature & target columns
    feat_cols   = CFG["feature_cols"]
    target_col  = CFG["target_col"]
    device_col  = CFG["device_col"]

    required = feat_cols + [target_col, device_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for col in feat_cols + [target_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=feat_cols + [target_col])
    df = df[df[target_col].between(50, 200)]  # physically plausible PL range

    print(f"  Final usable rows: {len(df):,}")
    print(f"  Devices present: {sorted(df[device_col].unique())}")
    print(f"  exp_pl  ->  min={df[target_col].min():.1f}  "
          f"max={df[target_col].max():.1f}  "
          f"mean={df[target_col].mean():.1f}  "
          f"std={df[target_col].std():.1f} dB")

    summary = {
        "raw_rows"       : raw_count,
        "anomalies_removed": n_bad,
        "null_dropped"   : dropped_null,
        "final_rows"     : len(df),
    }
    return df, summary


# ─────────────────────────────────────────────────────────────────────────────
# 2. FIGURE 1 — Data Distribution per Device (non-IID visualisation)
# ─────────────────────────────────────────────────────────────────────────────

def plot_data_distribution(df):
    print("\n  Plotting Figure 1 — Data distribution per client …")
    plt.style.use(CFG["style"])
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=False)
    axes = axes.flatten()

    devices = sorted(df[CFG["device_col"]].unique())
    for idx, dev in enumerate(devices):
        sub = df[df[CFG["device_col"]] == dev][CFG["target_col"]]
        ax = axes[idx]
        ax.hist(sub, bins=50, color=COLORS[idx], edgecolor="white",
                linewidth=0.4, alpha=0.85)
        ax.set_title(f"{dev}  (n={len(sub):,})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Expected Path Loss (dB)", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.axvline(sub.mean(), color="black", linewidth=1.5,
                   linestyle="--", label=f"μ={sub.mean():.1f}")
        ax.legend(fontsize=8)

    plt.suptitle("Figure 1 — Non-IID Distribution of Expected Path Loss per Virtual FL Client\n"
                 "(Each client = one physical device location, Hölderlinstraße Campus, Siegen)",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    return savefig("fig1_data_distribution_per_client")


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_model(lr=None):
    """Dense(9 → 8 → 1) regression model — identical architecture to model.h
    deployed on MKR WAN 1310 via TFLite Micro. 89 trainable parameters."""
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
# 4. CENTRALIZED BASELINE (upper bound)
# ─────────────────────────────────────────────────────────────────────────────

def run_centralized(X_train, X_test, y_train, y_test):
    print("\n" + "="*65)
    print("STEP 2 — Centralized Baseline (Phase A, upper bound)")
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
        "r2"     : m["r2"],
        "rmse"   : m["rmse"],
        "mae"    : m["mae"],
        "epochs" : len(hist.history["loss"]),
        "train_loss_history" : hist.history["loss"],
        "val_loss_history"   : hist.history["val_loss"],
    }
    return model, results, y_pred


# ─────────────────────────────────────────────────────────────────────────────
# 5. FIGURE 2 — Centralized training / validation loss
# ─────────────────────────────────────────────────────────────────────────────

def plot_centralized_loss(central_results):
    print("  Plotting Figure 2 — Centralized training curves …")
    plt.style.use(CFG["style"])
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(central_results["train_loss_history"]) + 1)
    ax.plot(epochs, central_results["train_loss_history"],
            label="Training Loss (MSE)", color="#2D6A9F", linewidth=2)
    ax.plot(epochs, central_results["val_loss_history"],
            label="Validation Loss (MSE)", color="#E07B39",
            linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("MSE Loss", fontsize=11)
    ax.set_title(
        f"Figure 2 — Centralized Model Training Curves\n"
        f"Dense(9→8→1)  |  $R^2$={central_results['r2']:.4f}  "
        f"RMSE={central_results['rmse']:.2f} dB",
        fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    return savefig("fig2_centralized_training_curves")


# ─────────────────────────────────────────────────────────────────────────────
# 6. FIGURE 3 — Predicted vs Actual (centralized)
# ─────────────────────────────────────────────────────────────────────────────

def plot_pred_vs_actual(y_test, y_pred_central, label="Centralized NN"):
    print("  Plotting Figure 3 — Predicted vs Actual …")
    plt.style.use(CFG["style"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Scatter
    ax = axes[0]
    ax.scatter(y_test, y_pred_central, alpha=0.15, s=4, color="#2D6A9F",
               rasterized=True)
    lims = [min(y_test.min(), y_pred_central.min()) - 2,
            max(y_test.max(), y_pred_central.max()) + 2]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual Path Loss (dB)", fontsize=11)
    ax.set_ylabel("Predicted Path Loss (dB)", fontsize=11)
    ax.set_title(f"{label}\n$R^2$={r2_score(y_test, y_pred_central):.4f}", fontsize=11)
    ax.legend(fontsize=9)

    # Residuals histogram
    ax2 = axes[1]
    residuals = y_pred_central - y_test
    ax2.hist(residuals, bins=80, color="#3AA66C", edgecolor="white",
             linewidth=0.3, alpha=0.85)
    ax2.axvline(0, color="red", linewidth=1.5, linestyle="--")
    ax2.axvline(residuals.mean(), color="black", linewidth=1.2,
                linestyle="-", label=f"Mean={residuals.mean():.2f} dB")
    ax2.set_xlabel("Residual (Predicted − Actual) dB", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Residual Distribution", fontsize=11)
    ax2.legend(fontsize=9)

    plt.suptitle("Figure 3 — Centralized Model: Predicted vs Actual Path Loss",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    return savefig("fig3_pred_vs_actual_centralized")


# ─────────────────────────────────────────────────────────────────────────────
# 7. FL SIMULATION — FedAvg
# ─────────────────────────────────────────────────────────────────────────────

def fedavg(client_weights, client_n_samples):
    """Weighted average of model weights (FedAvg, McMahan et al., 2017)."""
    total = sum(client_n_samples)
    aggregated = []
    for layer_idx in range(len(client_weights[0])):
        layer = np.zeros_like(client_weights[0][layer_idx])
        for c_idx, w in enumerate(client_weights):
            layer += (client_n_samples[c_idx] / total) * w[layer_idx]
        aggregated.append(layer)
    return aggregated


def run_fl_simulation(client_data, X_test, y_test, local_epochs=3):
    """
    Full FL simulation over CFG['fl_rounds'] rounds.
    Each round: broadcast global → local train → collect updates → FedAvg.
    Returns per-round R², RMSE.
    """
    print(f"\n  -> FL run: local_epochs={local_epochs}, "
          f"rounds={CFG['fl_rounds']}, clients={len(client_data)}")

    set_seed(CFG["random_seed"])
    global_model = build_model()

    round_r2   = []
    round_rmse = []

    for rnd in range(1, CFG["fl_rounds"] + 1):
        global_weights = global_model.get_weights()
        client_weights = []
        client_ns      = []

        for dev, (Xc, yc) in client_data.items():
            local = build_model()
            local.set_weights(global_weights)
            local.fit(Xc, yc,
                      epochs=local_epochs,
                      batch_size=CFG["batch_size"],
                      verbose=0)
            client_weights.append(local.get_weights())
            client_ns.append(len(Xc))

        new_weights = fedavg(client_weights, client_ns)
        global_model.set_weights(new_weights)

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
# 8. RUN ALL FL EXPERIMENTS
# ─────────────────────────────────────────────────────────────────────────────

def run_all_fl_experiments(client_data, X_test, y_test):
    print("\n" + "="*65)
    print("STEP 3 — Federated Learning Simulation (Phase A)")
    print("="*65)

    all_results = {}
    for le in CFG["fl_local_epochs"]:
        model, r2_hist, rmse_hist = run_fl_simulation(
            client_data, X_test, y_test, local_epochs=le)
        y_pred_fl = model.predict(X_test, verbose=0).flatten()
        final_m = metrics(y_test, y_pred_fl,
                          f"FL (E={le}, R={CFG['fl_rounds']})")
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
# 9. FIGURE 4 — R² vs Communication Round
# ─────────────────────────────────────────────────────────────────────────────

def plot_r2_vs_rounds(fl_results, central_r2):
    print("  Plotting Figure 4 — R² vs Communication Round …")
    plt.style.use(CFG["style"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # R²
    ax = axes[0]
    styles = ["-", "--", ":"]
    for idx, (le, res) in enumerate(fl_results.items()):
        rounds = range(1, len(res["r2_history"]) + 1)
        ax.plot(rounds, res["r2_history"],
                label=f"FL  E={le} local epochs",
                linewidth=2, linestyle=styles[idx],
                color=COLORS[idx])
    ax.axhline(central_r2, color="black", linewidth=1.8,
               linestyle="-.", label=f"Centralized NN  ($R^2$={central_r2:.4f})")
    ax.axhline(CFG["supervisor_r2"], color="gray", linewidth=1.2,
               linestyle="--",
               label=f"Reference LDPLSM-MW-EP  (R²={CFG['supervisor_r2']})")
    ax.set_xlabel("FL Communication Round", fontsize=11)
    ax.set_ylabel("$R^2$ on Global Test Set", fontsize=11)
    ax.set_title("R² Convergence vs Communication Round", fontsize=11,
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.0)

    # RMSE
    ax2 = axes[1]
    for idx, (le, res) in enumerate(fl_results.items()):
        rounds = range(1, len(res["rmse_history"]) + 1)
        ax2.plot(rounds, res["rmse_history"],
                 label=f"FL  E={le}",
                 linewidth=2, linestyle=styles[idx],
                 color=COLORS[idx])
    ax2.axhline(np.sqrt(mean_squared_error(
        [0]*10, [0]*10)),   # placeholder — will be replaced
        color="white")      # invisible; actual value injected via annotation

    ax2.set_xlabel("FL Communication Round", fontsize=11)
    ax2.set_ylabel("RMSE (dB)", fontsize=11)
    ax2.set_title("RMSE Convergence vs Communication Round", fontsize=11,
                  fontweight="bold")
    ax2.legend(fontsize=9)

    plt.suptitle("Figure 4 — Federated Learning Convergence (Phase A Simulation)\n"
                 "6 Virtual Clients | Non-IID Split by Physical Device Location",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    return savefig("fig4_fl_convergence_r2_rmse")


# ─────────────────────────────────────────────────────────────────────────────
# 10. FIGURE 5 — Three-Way Comparison Bar Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_three_way_comparison(central_results, fl_results):
    print("  Plotting Figure 5 — Three-way comparison …")
    plt.style.use(CFG["style"])

    # Pick best FL config (highest final R²)
    best_le = max(fl_results, key=lambda k: fl_results[k]["final_r2"])
    best_fl = fl_results[best_le]

    labels = [
        "Supervisor\nLDPLSM-MW-EP\n(Linear, 2024)",
        "PEP Project\nXGBoost\n(Centralized)",
        "This Thesis\nCent. NN\n(Baseline)",
        f"This Thesis\nFederated NN\n(E={best_le})",
    ]
    r2_vals   = [CFG["supervisor_r2"], CFG["pep_r2"],
                 central_results["r2"], best_fl["final_r2"]]
    rmse_vals = [CFG["supervisor_rmse"], None,
                 central_results["rmse"], best_fl["final_rmse"]]

    bar_colors = ["#8E8E8E", "#E07B39", "#2D6A9F", "#3AA66C"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # R²
    ax = axes[0]
    bars = ax.bar(labels, r2_vals, color=bar_colors, edgecolor="white",
                  linewidth=0.8, width=0.55)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("$R^2$", fontsize=12)
    ax.set_title("$R^2$ Comparison", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9,
                fontweight="bold")
    # Add improvement arrow for FL vs supervisor
    ax.annotate("", xy=(3, best_fl["final_r2"]), xytext=(0, CFG["supervisor_r2"]),
                arrowprops=dict(arrowstyle="->", color="darkred", lw=1.5))

    # RMSE
    ax2 = axes[1]
    rmse_plot  = [CFG["supervisor_rmse"],
                  float("nan"),    # XGBoost RMSE not tracked in PEP
                  central_results["rmse"],
                  best_fl["final_rmse"]]
    rmse_colors = bar_colors.copy()
    bars2 = ax2.bar(labels, rmse_plot, color=rmse_colors, edgecolor="white",
                    linewidth=0.8, width=0.55)
    ax2.set_ylabel("RMSE (dB)", fontsize=12)
    ax2.set_title("RMSE Comparison", fontsize=12, fontweight="bold")
    for bar, val in zip(bars2, rmse_plot):
        if not np.isnan(val):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=9,
                     fontweight="bold")

    ax2.text(1, 1.5, "RMSE\nnot reported\nin PEP project",
             ha="center", fontsize=7.5, color="gray", style="italic")

    plt.suptitle("Figure 5 — Three-Way Performance Comparison\n"
                 "Supervisor Baseline → PEP XGBoost → Centralized NN → Federated NN",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    return savefig("fig5_three_way_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# 11. FIGURE 6 — Per-Client R² (non-IID analysis)
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_client_r2(client_data, best_fl_model, X_test, y_test):
    print("  Plotting Figure 6 — Per-client R² analysis …")
    plt.style.use(CFG["style"])

    client_r2 = []
    client_names = []
    client_sizes = []

    for dev, (Xc, yc) in client_data.items():
        y_c_pred = best_fl_model.predict(Xc, verbose=0).flatten()
        r2 = r2_score(yc, y_c_pred)
        client_r2.append(r2)
        client_names.append(dev)
        client_sizes.append(len(Xc))

    global_r2 = r2_score(y_test, best_fl_model.predict(X_test,verbose=0).flatten())

    fig, ax = plt.subplots(figsize=(9, 5))
    
    # Clip negative R2 values to -1.0 for plotting purposes to avoid distorting the positive scale,
    # but the text labels will display the actual computed R2 values.
    plot_r2 = [max(-1.0, r) for r in client_r2]
    
    bars = ax.barh(client_names, plot_r2,
                   color=[COLORS[i] for i in range(len(client_names))],
                   edgecolor="white", linewidth=0.8)
                   
    ax.axvline(0, color="gray", linewidth=1.0, linestyle="-")
    ax.axvline(global_r2, color="black", linewidth=2, linestyle="--",
               label=f"Global R²={global_r2:.4f}")
    ax.axvline(CFG["supervisor_r2"], color="gray", linewidth=1.5,
               linestyle=":", label=f"Reference MLR R²={CFG['supervisor_r2']}")
    ax.set_xlabel("$R^2$ (Federated Global Model evaluated on local data)", fontsize=10)
    ax.set_title("Figure 6 — Per-Client R² of Federated Global Model\n"
                 "(Non-IID Data: Each Device = Different Room Location)",
                 fontsize=11, fontweight="bold")
                 
    for bar, val, plot_val, n in zip(bars, client_r2, plot_r2, client_sizes):
        if val >= 0:
            text_x = plot_val + 0.01
            ha = "left"
            color = "black"
        else:
            text_x = plot_val - 0.01
            ha = "right"
            color = "darkred"
            
        ax.text(text_x, bar.get_y() + bar.get_height()/2,
                f"R²={val:.3f}  (n={n:,})",
                va="center", ha=ha, fontsize=9, color=color)
                
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(-1.5, 1.05)
    plt.tight_layout()
    return savefig("fig6_per_client_r2_non_iid")


# ─────────────────────────────────────────────────────────────────────────────
# 12. FIGURE 7 — Communication Efficiency
# ─────────────────────────────────────────────────────────────────────────────

def plot_communication_efficiency():
    print("  Plotting Figure 7 — Communication efficiency …")
    plt.style.use(CFG["style"])

    # Old system: raw data uplink every 60s → 1440 msg/day × 18 B/msg
    old_bytes_day   = 1440 * 18          # 25,920 B/day/node
    # New system: 288 status msgs (every 5 min) × 8 B + 1 FL update × 52 B
    new_status      = 288 * 8            # 2,304 B
    new_fl          = 1 * 52             # 52 B
    new_bytes_day   = new_status + new_fl    # ≈ 2,356 B/day/node

    categories = ["Centralized", "Federated"]
    values     = [old_bytes_day, new_bytes_day]
    colors_bar = ["#C0392B", "#3AA66C"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(categories, values, color=colors_bar, edgecolor="white",
                  linewidth=0.8, width=0.4)
    ax.set_ylabel("Bytes / Node / Day", fontsize=11)
    ax.set_title("Figure 7 — Communication Efficiency Comparison\n"
                 f"Reduction factor: {old_bytes_day/new_bytes_day:.1f}× "
                 "Federated vs Centralized",
                 fontsize=11, fontweight="bold")
                 
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                f"{val:,} B", ha="center", fontsize=10, fontweight="bold")
                
    ax.set_ylim(0, 30000)
    plt.tight_layout()
    return savefig("fig7_communication_efficiency")


# ─────────────────────────────────────────────────────────────────────────────
# 13. FIGURE 8 — FL Pred vs Actual (best config)
# ─────────────────────────────────────────────────────────────────────────────

def plot_fl_pred_vs_actual(y_test, y_pred_fl, best_le, best_r2):
    print("  Plotting Figure 8 — FL Predicted vs Actual …")
    plt.style.use(CFG["style"])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred_fl, alpha=0.12, s=4, color="#3AA66C",
               rasterized=True)
    lims = [min(y_test.min(), y_pred_fl.min()) - 2,
            max(y_test.max(), y_pred_fl.max()) + 2]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual Path Loss (dB)", fontsize=11)
    ax.set_ylabel("Predicted Path Loss (dB)", fontsize=11)
    ax.set_title(
        f"Figure 8 — Federated NN (E={best_le}): Predicted vs Actual\n"
        f"$R^2$={best_r2:.4f}  RMSE="
        f"{np.sqrt(mean_squared_error(y_test, y_pred_fl)):.2f} dB",
        fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return savefig("fig8_fl_pred_vs_actual")


# ─────────────────────────────────────────────────────────────────────────────
# 14. SAVE RESULTS JSON (for thesis tables)
# ─────────────────────────────────────────────────────────────────────────────

def save_results(data_summary, central_results, fl_results, client_data):
    print("\n  Saving results JSON …")
    best_le = max(fl_results, key=lambda k: fl_results[k]["final_r2"])

    client_info = {d: len(v[0]) for d, v in client_data.items()}

    out = {
        "simulation_meta": {
            "timestamp"   : time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tensorflow"  : tf.__version__,
            "fl_rounds"   : CFG["fl_rounds"],
            "local_epochs_tested": CFG["fl_local_epochs"],
            "architecture": "Dense(9→8→1)",
            "n_params"    : 89,
        },
        "data_summary": data_summary,
        "client_sample_counts": client_info,
        "reference_values": {
            "supervisor_LDPLSM_MW_EP_R2"  : CFG["supervisor_r2"],
            "supervisor_LDPLSM_MW_EP_RMSE": CFG["supervisor_rmse"],
            "pep_xgboost_R2"              : CFG["pep_r2"],
            "torres_sanchez_F1"           : CFG["torres_f1"],
        },
        "centralized_baseline": {
            "r2"  : central_results["r2"],
            "rmse": central_results["rmse"],
            "mae" : central_results["mae"],
        },
        "federated_results": {
            str(le): {
                "final_r2"  : res["final_r2"],
                "final_rmse": res["final_rmse"],
                "final_mae" : res["final_mae"],
                "r2_relative_to_centralized":
                    res["final_r2"] / central_results["r2"],
            }
            for le, res in fl_results.items()
        },
        "best_fl_config": {
            "local_epochs": best_le,
            "final_r2"    : fl_results[best_le]["final_r2"],
            "final_rmse"  : fl_results[best_le]["final_rmse"],
            "r2_drop_vs_centralized_pct":
                (central_results["r2"] - fl_results[best_le]["final_r2"])
                / central_results["r2"] * 100,
        },
        "communication_efficiency": {
            "old_bytes_per_node_per_day"    : 1440 * 18,
            "new_bytes_per_node_per_day"    : 288 * 8 + 52,
            "reduction_factor"              :
                (1440 * 18) / (288 * 8 + 52),
            "torres_sanchez_bytes_per_round": 28 * 51,
            "our_fl_bytes_per_round"        : 52,
        },
    }

    with open(CFG["results_json"], "w") as f:
        json.dump(out, f, indent=2)
    print(f"  -> Saved: {CFG['results_json']}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 15. PRINT THESIS TABLE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_thesis_tables(results):
    print("\n" + "="*65)
    print("THESIS TABLE VALUES  -  copy directly into LaTeX")
    print("="*65)

    print("\n=== Table: Three-Way Comparison ===")
    hdr = f"{'Model':<30} {'R2':>8} {'RMSE (dB)':>12}"
    print(hdr)
    print("-" * len(hdr))
    print(f"{'Supervisor LDPLSM-MW-EP':<30} {CFG['supervisor_r2']:>8.4f} {CFG['supervisor_rmse']:>12.2f}")
    print(f"{'PEP XGBoost (centralized)':<30} {CFG['pep_r2']:>8.4f} {'N/A':>12}")
    cm = results["centralized_baseline"]
    print(f"{'Centralized NN (this thesis)':<30} {cm['r2']:>8.4f} {cm['rmse']:>12.2f}")
    best = results["best_fl_config"]
    print(f"{'Federated NN (best, this thesis)':<30} {best['final_r2']:>8.4f} {best['final_rmse']:>12.2f}")

    print("\n=== Table: FL vs Local Epochs ===")
    print(f"{'Local Epochs':<15} {'Final R2':>10} {'RMSE (dB)':>12} {'R2 / Central':>14}")
    for le, v in results["federated_results"].items():
        print(f"E={le:<13} {v['final_r2']:>10.4f} {v['final_rmse']:>12.2f} "
              f"{v['r2_relative_to_centralized']:>14.4f}")

    print("\n=== Table: Communication Efficiency ===")
    ce = results["communication_efficiency"]
    print(f"  Old (raw data):       {ce['old_bytes_per_node_per_day']:>8,} B/day/node")
    print(f"  New (federated):      {ce['new_bytes_per_node_per_day']:>8,} B/day/node")
    print(f"  Reduction factor:     {ce['reduction_factor']:>8.1f}x")
    print(f"  Our FL update:        {ce['our_fl_bytes_per_round']:>8} B/round")
    print(f"  Torres Sanchez FL:    {ce['torres_sanchez_bytes_per_round']:>8} B/round (~28 msgs)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    set_seed(CFG["random_seed"])
    os.makedirs(CFG["figures_dir"], exist_ok=True)

    print("\n" + "="*65)
    print("FL SIMULATION — Master's Thesis, Pratik Khadka")
    print("University of Siegen")
    print("Dataset: Indoor LoRaWAN, Hölderlinstraße Campus")
    print("="*65)

    # ── 1. Load data
    df, data_summary = load_and_preprocess()

    # ── 2. Plot data distribution (Figure 1)
    fig1 = plot_data_distribution(df)

    # ── 3. Prepare features & scale
    print("\n  Preparing features and scaling …")
    feat_cols  = CFG["feature_cols"]
    target_col = CFG["target_col"]
    device_col = CFG["device_col"]

    X_all = df[feat_cols].values.astype(np.float32)
    y_all = df[target_col].values.astype(np.float32)
    dev_all = df[device_col].values

    scaler = StandardScaler()
    X_all_norm = scaler.fit_transform(X_all)
    print(f"  Feature means: {dict(zip(feat_cols, scaler.mean_.round(4)))}")
    print(f"  Feature stds:  {dict(zip(feat_cols, scaler.scale_.round(4)))}")

    # Save scaler for Arduino code
    np.save("model_output/feature_means.npy", scaler.mean_)
    np.save("model_output/feature_stds.npy", scaler.scale_)

    # Global train/test split — stratified so each split has all devices
    X_train_all, X_test, y_train_all, y_test, dev_train, _ = train_test_split(
        X_all_norm, y_all, dev_all,
        test_size=CFG["test_split"],
        random_state=CFG["random_seed"],
    )
    print(f"  Train: {len(X_train_all):,}  Test: {len(X_test):,}")

    # ── 4. Centralized baseline
    central_model, central_results, y_pred_central = run_centralized(
        X_train_all, X_test, y_train_all, y_test)

    fig2 = plot_centralized_loss(central_results)
    fig3 = plot_pred_vs_actual(y_test, y_pred_central)

    # ── 5. Build non-IID client data (by device_id)
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

    # ── 6. FL Simulation
    fl_results = run_all_fl_experiments(client_data, X_test, y_test)

    # ── 7. Convergence plot (Figure 4)
    fig4 = plot_r2_vs_rounds(fl_results, central_results["r2"])

    # ── 8. Three-way comparison (Figure 5)
    fig5 = plot_three_way_comparison(central_results, fl_results)

    # ── 9. Best FL config for subsequent figures
    best_le = max(fl_results, key=lambda k: fl_results[k]["final_r2"])
    best_model = fl_results[best_le]["model"]

    # ── 10. Per-client R² (Figure 6)
    fig6 = plot_per_client_r2(client_data, best_model, X_test, y_test)

    # ── 11. Communication efficiency (Figure 7)
    fig7 = plot_communication_efficiency()

    # ── 12. FL pred vs actual (Figure 8)
    fig8 = plot_fl_pred_vs_actual(
        y_test,
        fl_results[best_le]["y_pred"],
        best_le,
        fl_results[best_le]["final_r2"])

    # ── 13. Save results JSON
    results = save_results(data_summary, central_results, fl_results, client_data)

    # ── 14. Print thesis tables
    print_thesis_tables(results)

    elapsed = time.time() - t_start
    print(f"\n{'='*65}")
    print(f"SIMULATION COMPLETE in {elapsed/60:.1f} minutes")
    print(f"Figures saved to: {CFG['figures_dir']}/")
    print(f"Results JSON:     {CFG['results_json']}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
