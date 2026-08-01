# Federated Learning Manual (General System Architecture)

**Thesis Title**: *Decentralized Edge Intelligence for LoRaWAN: Federated Learning for Environment-Driven Path Loss and Link Quality Modeling*  
**Author**: Pratik Khadka  
**Institution**: University of Siegen  
**File Target**: `thesis guideline/Federated Learning Manual.md`  

---

## 1. Executive Summary & Overview

This manual provides a general specification of the **Federated Learning (FL)** system architecture used in this thesis. It details the relationship between calendar timeframe, communication rounds, data sampling, weight/bias parameter updates, server aggregation, and testing protocols.

In Federated Learning:
- **Raw Telemetry Privacy**: Sensor data collected by LoRaWAN end-devices (`ED0` through `ED5`) stays 100% on the local hardware. Raw data is never transmitted to the cloud.
- **Parametric Collaboration**: Instead of sharing data, clients train their local models independently and share only their **89 model parameters** (80 weights + 9 biases) with the central server.

---

## 2. Communication Round Scheduling & Data Volume

The simulation models a continuous multi-client LPWAN deployment over a 1-year timeline:

- **Total Dataset Horizon**: 365 Calendar Days
- **Total Clean Dataset Volume**: 206,957 samples across 6 end-devices (`ED0`–`ED5`)
- **Total FL Communication Rounds**: **50 Rounds**

```
 [365-Day Timeline]  ═══════════════════════════════════════════════════════════════════►
                     ├─── Round 1 ───┼─── Round 2 ───┼─── ... ───┼─── Round 50 ───┤
                       (~7.3 Days)     (~7.3 Days)                 (~7.3 Days)
```

### 2.1 Duration of One Communication Round
The timeline is divided equally across the 50 communication rounds:

$$\text{Duration per FL Round} = \frac{365 \text{ Calendar Days}}{50 \text{ FL Rounds}} = \mathbf{7.3 \text{ Days per Round}}$$

### 2.2 Data Samples Collected per Round
- **System-Wide Samples per Round**:
  $$\text{Total Samples per Round} = \frac{206,957 \text{ Total Clean Samples}}{50 \text{ FL Rounds}} \approx \mathbf{4,139 \text{ Samples per Round}}$$

- **Per-Client Samples per Round**:
  $$\text{Samples per Client per Round} \approx \frac{4,139 \text{ Samples}}{6 \text{ Clients}} \approx \mathbf{690 \text{ Samples per Client per Round}}$$

---

## 3. Privacy-Preserving 4-Step Communication Round Sequence

To guarantee zero data leakage and preserve complete telemetry privacy, testing and evaluation occur across a structured 4-step sequence in every communication round:

```
 ┌───────────────────────────────────────────────────────────────────────────┐
 │ STEP 1: LOCAL TRAINING & WEIGHT UPLINK                                   │
 │ - End Devices (ED0–ED5) train locally for 5 epochs on 80% local data.   │
 │ - EDs send their 89 local weights (89 bytes) to the Server.              │
 └─────────────────────────────────────┬─────────────────────────────────────┘
                                       │
                                       ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │ STEP 2: SERVER AGGREGATION & GLOBAL BROADCAST                            │
 │ - Server averages client weights: W_global = ∑ (Ni / N) * Wi.             │
 │ - Server broadcasts W_global back to all End Devices (89 bytes).         │
 └─────────────────────────────────────┬─────────────────────────────────────┘
                                       │
                                       ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │ STEP 3: LOCAL TESTING ON END DEVICES                                      │
 │ - Each ED replaces its local model: W_local ⟵ W_global.                 │
 │ - Each ED tests W_global locally on its own private 20% local test data. │
 │ - Each ED calculates local R² and local RMSE (just 2 scalar numbers!).   │
 │ - EDs send only their local (R², RMSE) back to the Server.                │
 └─────────────────────────────────────┬─────────────────────────────────────┘
                                       │
                                       ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │ STEP 4: FINAL METRIC AGGREGATION & GRAPH PLOTTING                         │
 │ - Server averages local R² and RMSE scores across all 6 clients.          │
 │ - Server plots 1 point on the FL round convergence curve!                 │
 └───────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Local Parameter Updates & Overwrite Rule

Each client microcontroller maintains a local copy of the **89-parameter Multilayer Perceptron (MLP)**:
- **80 Trainable Weights** ($W$)
- **9 Trainable Biases** ($b$)
- **Total Parameters**: **89 Parameters**

```
           Input Layer (9 Features)
                  │
        ┌─────────┴─────────┐
        │  Dense Hidden (8) │  ──► 9 × 8 Weights + 8 Biases  = 80 Parameters
        └─────────┬─────────┘
        │  Dense Output (1) │  ──► 8 × 1 Weights + 1 Bias    =  9 Parameters
        └───────────────────┘
                                       TOTAL PARAMETERS     = 89 Parameters
```

### 4.1 Local Training Adjustments
During local training epochs, clients update both weights and biases via gradient descent subtraction:

$$W_{i, \text{new}} = W_{i, \text{old}} - \left( \eta \cdot \frac{\partial L_i}{\partial W_i} \right)$$

$$b_{i, \text{new}} = b_{i, \text{old}} - \left( \eta \cdot \frac{\partial L_i}{\partial b_i} \right)$$

*(where $\eta = 0.01$ is the local learning rate)*.

### 4.2 Global Weight Replacement Rule
When an end-device receives the aggregated global model ($W^{\text{global}}, b^{\text{global}}$) from the server, it **replaces (overwrites)** its local model:

$$W_{\text{local}} \longleftarrow W^{\text{global}}, \qquad b_{\text{local}} \longleftarrow b^{\text{global}}$$

> **Key Rule**: The client does **NOT** subtract weights ($W_{\text{local}} - W^{\text{global}}$). The global model simply overwrites the local model so that all clients start Round $t+1$ from the same updated, combined intelligence!

---

## 5. Testing & Evaluation Architecture

### 5.1 Local Held-Out Test Split (20%)
- **Total Testing Volume**: **41,393 held-out test samples** across all 6 clients (`ED0` to `ED5`).
- **Data Location**: Test data is stored 100% locally on edge devices (or partitioned in memory during simulation). Raw test data is **never transmitted to the server**.

### 5.2 Testing Protocol
1. **Round-by-Round Evaluation**: At Step 3 of each round $t$, end-devices test $W^{\text{global}}$ against their **20% testing samples**. Clients send only their local evaluation scalars ($R^2_i, \text{RMSE}_i$) back to the server. The server averages these scalars to plot **1 dot on the convergence curve**.
2. **Final Benchmark Evaluation**: After Round 50 completes, the final global model is evaluated in batch over all 41,393 test samples to generate official thesis performance tables and per-device breakdowns.

---

## 6. General Process Summary

| System Stage | Process Description | Data / Parameter Action |
| :--- | :--- | :--- |
| **Sampling & Duration** | 7.3 Days per Round (~4,139 samples/round) | Environmental telemetry collected locally by nodes |
| **Local Training** | Clients train for $E=5$ local epochs | Weights ($W$) & Biases ($b$) updated via gradient subtraction |
| **Parameter Uplink** | Clients send model parameters over LoRaWAN | 89 parameters (89 bytes) sent in 1 packet |
| **Global Aggregation** | Central server aggregates client models | Combines client weights & biases into global model ($W^{t+1}, b^{t+1}$) |
| **Downlink Broadcast** | Server broadcasts global model to clients | Clients overwrite local weights ($W_{\text{local}} \leftarrow W^{\text{global}}$) |
| **Local Testing** | Clients test $W^{\text{global}}$ on 20% test set | Clients send back only scalar $R^2$ and RMSE evaluation metrics |
| **Graph Plotting** | Server averages $R^2$ & RMSE scalars | Plots 1 point on the FL round convergence curve |
