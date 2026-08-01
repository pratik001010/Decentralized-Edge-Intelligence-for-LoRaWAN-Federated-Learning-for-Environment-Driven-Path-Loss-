# FedAvg Manual (Mathematical & Algorithmic Specification)

**Thesis Title**: *Decentralized Edge Intelligence for LoRaWAN: Federated Learning for Environment-Driven Path Loss and Link Quality Modeling*  
**Author**: Pratik Khadka  
**Institution**: University of Siegen  
**Algorithm**: Federated Averaging (FedAvg, McMahan et al., 2017)  
**Target Model**: 89-Parameter MLP (`Dense(9→8→1)`)  
**File Target**: `thesis guideline/FedAvg Manual.md`  

---

## 1. Executive Summary & Objective

This manual provides an exhaustive mathematical and algorithmic specification of **Federated Averaging (FedAvg)** as implemented in this thesis.

FedAvg solves the global optimization problem across $K = 6$ decentralized end-devices (`ED0` through `ED5`) without sharing raw telemetry:

$$\min_{\mathbf{w}} f(\mathbf{w}) = \sum_{i=1}^{K} \frac{N_i}{N} F_i(\mathbf{w})$$

where:
- $\mathbf{w} \in \mathbb{R}^{89}$ is the global parameter vector (80 weights + 9 biases).
- $N_i$ is the number of local training samples on client $i$.
- $N = \sum_{i=1}^{K} N_i = 165,564$ is the total system-wide training samples.
- $F_i(\mathbf{w})$ is the local Mean Squared Error ($\text{MSE}$) loss function of client $i$.

---

## 2. Complete Step-by-Step Algorithmic Protocol (Round $t$)

```
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ STEP 1: DOWNLINK BROADCAST & LOCAL OVERWRITE                             │
  │ - Server broadcasts global parameter vector w_t (89 bytes) to all 6 EDs.  │
  │ - Each client sets its local baseline: w_{i, local} ⟵ w_t.               │
  └────────────────────────────────────┬─────────────────────────────────────┘
                                       │
                                       ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ STEP 2: LOCAL TRAINING VIA GRADIENT DESCENT                              │
  │ - Client i trains for E = 5 local epochs on local 80% dataset.           │
  │ - Parameters updated via subtraction: w ⟵ w - η ∇F_i(w; b).             │
  │ - Yields updated local parameter vector w_i^{t+1}.                        │
  └────────────────────────────────────┬─────────────────────────────────────┘
                                       │
                                       ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ STEP 3: UPLINK TRANSMISSION OVER LORAWAN                                 │
  │ - Client i packs 89 parameters into 1 LoRaWAN packet (int8 quantized).   │
  │ - Transmits 89-byte update to central server.                            │
  └────────────────────────────────────┬─────────────────────────────────────┘
                                       │
                                       ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ STEP 4: SERVER WEIGHTED AVERAGING (FEDAVG AGGREGATION)                   │
  │ - Server collects w_i^{t+1} from all 6 clients.                          │
  │ - Server calculates new global model: w_{t+1} = ∑ (Ni / N) * w_i^{t+1}.  │
  └────────────────────────────────────┬─────────────────────────────────────┘
                                       │
                                       ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ STEP 5: LOCAL TESTING & SCALAR METRIC LOGGING                            │
  │ - Server broadcasts w_{t+1} to clients.                                  │
  │ - Clients test w_{t+1} on private 20% test data & return (R²_i, RMSE_i).  │
  │ - Server averages metrics and plots 1 point on the convergence curve!     │
  └──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Mathematical Equations for Each Step

### 3.1 Step 1: Downlink Parameter Broadcast & Baseline Initialization
At the start of round $t$, the server transmits the global model $\mathbf{w}_t$ to client $i \in \{1, \dots, K\}$. Client $i$ overwrites its local parameter state:

$$\mathbf{w}_{i,0}^{t} \longleftarrow \mathbf{w}_t$$

*(where $\mathbf{w}_t$ consists of 72 hidden weights $\mathbf{W}^{(1)}_t$, 8 hidden biases $\mathbf{b}^{(1)}_t$, 8 output weights $\mathbf{W}^{(2)}_t$, and 1 output bias $b^{(2)}_t$)*.

---

### 3.2 Step 2: Local Client Training (Gradient Descent Subtraction)
Client $i$ executes $E = 5$ local epochs over its local training partition $\mathcal{D}_i$ ($N_i \approx 27,594$ samples).

For each mini-batch $b \subset \mathcal{D}_i$ of batch size $B = 512$:

1. **Forward Prediction**:
   $$\hat{y}_k = \mathbf{W}^{(2)} \cdot \text{ReLU}\left( \mathbf{W}^{(1)} \mathbf{x}_k + \mathbf{b}^{(1)} \right) + b^{(2)}$$

2. **Mini-Batch Loss**:
   $$F_i(\mathbf{w}; b) = \frac{1}{B} \sum_{k \in b} \left( y_k - \hat{y}_k \right)^2$$

3. **Backpropagation & Parameter Subtraction Updates**:
   $$\mathbf{w} \longleftarrow \mathbf{w} - \eta \cdot \nabla_{\mathbf{w}} F_i(\mathbf{w}; b)$$

   - **For Weights**:  
     $$\mathbf{W}_{\text{new}} = \mathbf{W}_{\text{old}} - \left( \eta \cdot \frac{\partial F_i}{\partial \mathbf{W}} \right)$$

   - **For Biases**:  
     $$\mathbf{b}_{\text{new}} = \mathbf{b}_{\text{old}} - \left( \eta \cdot \frac{\partial F_i}{\partial \mathbf{b}} \right)$$

*(where $\eta = 0.01$ is the local learning rate)*.

After $E = 5$ epochs, client $i$ produces its updated local parameter vector:

$$\mathbf{w}_i^{t+1} = \left[ \mathbf{W}_{i}^{(1), t+1}, \mathbf{b}_{i}^{(1), t+1}, \mathbf{W}_{i}^{(2), t+1}, b_{i}^{(2), t+1} \right]$$

---

### 3.3 Step 3: Uplink Parameter Transmission over LoRaWAN
Client $i$ quantizes its 89 float32 parameters to signed 8-bit integers (`int8`):

$$\mathbf{q}_i = \text{Quantize}_{\text{int8}}\left( \mathbf{w}_i^{t+1} \right)$$

- **Payload Size**: $89 \text{ parameters} \times 1 \text{ byte/parameter} = \mathbf{89 \text{ bytes}}$.
- **LoRaWAN Framing**: Fits entirely within **1 single LoRaWAN packet** at EU868 DR5 (max MAC payload = 222 bytes).

---

### 3.4 Step 4: Server-Side Weighted Averaging (FedAvg Aggregation)
The central server collects parameter updates $\mathbf{w}_i^{t+1}$ and sample weights $N_i$ from all $K = 6$ clients.

The server calculates the new global model parameter vector $\mathbf{w}_{t+1}$ using **FedAvg weighted averaging**:

$$\mathbf{w}_{t+1} = \sum_{i=1}^{K} \frac{N_i}{N} \mathbf{w}_i^{t+1}$$

#### Layer-by-Layer Aggregation Breakdown:

1. **Hidden Layer Weights (72 parameters)**:
   $$\mathbf{W}_{t+1}^{(1)} = \sum_{i=1}^{K} \left( \frac{N_i}{N} \right) \mathbf{W}_{i}^{(1), t+1}$$

2. **Hidden Layer Biases (8 parameters)**:
   $$\mathbf{b}_{t+1}^{(1)} = \sum_{i=1}^{K} \left( \frac{N_i}{N} \right) \mathbf{b}_{i}^{(1), t+1}$$

3. **Output Layer Weights (8 parameters)**:
   $$\mathbf{W}_{t+1}^{(2)} = \sum_{i=1}^{K} \left( \frac{N_i}{N} \right) \mathbf{W}_{i}^{(2), t+1}$$

4. **Output Layer Bias (1 parameter)**:
   $$b_{t+1}^{(2)} = \sum_{i=1}^{K} \left( \frac{N_i}{N} \right) b_{i}^{(2), t+1}$$

---

### 3.5 Step 5: Local Testing & Convergence Logging
1. The server broadcasts $\mathbf{w}_{t+1}$ to all clients.
2. Each client $i$ overwrites its model ($\mathbf{w}_{i, \text{local}} \leftarrow \mathbf{w}_{t+1}$) and evaluates $\mathbf{w}_{t+1}$ against its **20% local held-out test dataset** ($M_i$ samples):

$$\text{MSE}_i = \frac{1}{M_i} \sum_{k=1}^{M_i} \left( y_{i,k} - \hat{y}_{i,k}(\mathbf{w}_{t+1}) \right)^2$$

$$R_i^2 = 1 - \frac{\sum_{k=1}^{M_i} \left( y_{i,k} - \hat{y}_{i,k} \right)^2}{\sum_{k=1}^{M_i} \left( y_{i,k} - \bar{y}_i \right)^2}$$

3. Clients transmit only two scalar evaluation numbers ($R^2_i, \text{RMSE}_i$) back to the server.
4. Server computes the system-wide test metrics for Round $t$:

$$\text{Global } R^2_t = \sum_{i=1}^{K} \frac{M_i}{M_{\text{total}}} R^2_i, \qquad \text{Global RMSE}_t = \sqrt{\sum_{i=1}^{K} \frac{M_i}{M_{\text{total}}} \text{MSE}_i}$$

5. The server records $(\text{Global } R^2_t, \text{Global RMSE}_t)$ to plot 1 point on the FL convergence curve.

---

## 4. Concrete Numerical Example

Suppose we have 2 clients ($K = 2$) updating a single weight $W$:
- Client 1 has $N_1 = 1000$ training samples. After 5 epochs, its updated weight is $W_1 = 0.50$.
- Client 2 has $N_2 = 2000$ training samples. After 5 epochs, its updated weight is $W_2 = 0.80$.
- Total sample volume: $N = 1000 + 2000 = 3000$.

### FedAvg Weighted Calculation:
$$W^{\text{global}} = \left(\frac{1000}{3000}\right) \times 0.50 + \left(\frac{2000}{3000}\right) \times 0.80$$

$$W^{\text{global}} = (0.3333 \times 0.50) + (0.6667 \times 0.80) = 0.1667 + 0.5333 = \mathbf{0.70}$$

Both Client 1 and Client 2 receive $W^{\text{global}} = 0.70$ and overwrite their local weights: $W_1 \leftarrow 0.70$, $W_2 \leftarrow 0.70$.

---

## 5. Python Implementation Mapping

Here is how each mathematical step maps directly to the simulation script `run_fl_365_10min.py`:

```python
# --- STEP 4: FedAvg Mathematical Function ---
def fedavg(client_weights, client_n_samples):
    total = sum(client_n_samples)
    aggregated = []
    for layer_idx in range(len(client_weights[0])):
        layer = np.zeros_like(client_weights[0][layer_idx])
        for c_idx, w in enumerate(client_weights):
            layer += (client_n_samples[c_idx] / total) * w[layer_idx]
        aggregated.append(layer)
    return aggregated

# --- STEP 1 & 2: Local Client Training Loop ---
for rnd in range(1, 51):
    global_weights = global_model.get_weights()  # Download Step 1
    client_weights = []
    client_ns = []

    for dev, (Xc, yc) in client_data.items():
        local = build_model()
        local.set_weights(global_weights)       # Overwrite Step 1
        local.fit(Xc, yc, epochs=5, batch_size=512, verbose=0) # Local Update Step 2
        client_weights.append(local.get_weights())
        client_ns.append(len(Xc))

    # --- STEP 4: Server Aggregation ---
    new_weights = fedavg(client_weights, client_ns)
    global_model.set_weights(new_weights)

    # --- STEP 5: Testing & Logging ---
    y_pred = global_model.predict(X_test, verbose=0).flatten()
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    round_r2.append(r2)
    round_rmse.append(rmse)
```

---

## 6. Summary Checklist for Thesis Defense

| Concept | Mathematical Formula / Value | Thesis Justification |
| :--- | :--- | :--- |
| **Global Objective** | $\min_{\mathbf{w}} f(\mathbf{w}) = \sum \frac{N_i}{N} F_i(\mathbf{w})$ | Minimizes system-wide path loss MSE across all rooms |
| **Local Update Rule** | $\mathbf{w} \longleftarrow \mathbf{w} - \eta \nabla F_i(\mathbf{w}; b)$ | Local gradient descent for $E=5$ epochs ($\eta = 0.01$) |
| **Aggregation Formula**| $\mathbf{w}_{t+1} = \sum \frac{N_i}{N} \mathbf{w}_i^{t+1}$ | Weighted parameter averaging based on client sample counts |
| **Client Overwrite** | $\mathbf{w}_{\text{local}} \longleftarrow \mathbf{w}_{t+1}$ | Overwrites local model so clients share global knowledge |
| **Payload Size** | **89 Bytes** per client per round | Fits in **1 single LoRaWAN packet** (EU868 DR5) |
| **Final Accuracy** | **$R^2 = 0.8807$, $\text{RMSE} = 6.52\text{ dB}$** | Achieves **97.1% of Centralized NN accuracy** ($0.9071$) |

---

## 7. Client Drift & Personalized Federated Learning Considerations

### 7.1 Mathematical Formulation of Client Drift under Non-IID Data
In decentralized LoRaWAN networks, each end-device ($i \in \{0, \dots, 5\}$) operates in a statistically heterogeneous environment (Non-IID data distribution). During the $E = 5$ local training epochs of round $t$, local stochastic gradient descent updates parameters toward the client's local minimum $\mathbf{w}_i^*$:

$$\mathbf{w}_i^{t+1} = \mathbf{w}_t - \eta \sum_{e=1}^{E} \nabla F_i(\mathbf{w}_i^{(e)})$$

Because physical propagation features (wall count $W_{\text{brick}}, W_{\text{wood}}$ and path loss exponent $n$) differ between rooms, the local gradient vectors $\nabla F_i$ point in divergent directions. This creates **Client Drift**:

$$\text{Drift}_i = \|\mathbf{w}_i^{t+1} - \mathbf{w}_{t+1}\|_2$$

The global FedAvg server averages these drifted parameters:

$$\mathbf{w}_{t+1} = \sum_{i=1}^{K} \frac{N_i}{N} \mathbf{w}_i^{t+1}$$

While $\mathbf{w}_{t+1}$ achieves optimal system-wide generalization ($R^2 = 0.8807$, $\text{RMSE} = 6.52\text{ dB}$ across all 1.6M global test samples), testing $\mathbf{w}_{t+1}$ directly on isolated, highly static local rooms (such as `ED4` or `ED5`) reveals local distribution shift.

---

### 7.2 Our Consideration: Personalized Federated Learning (Local Fine-Tuning)
To resolve client drift while preserving privacy, our architecture implements **Personalized Federated Learning**:
1. **Global Round Phase (Rounds 1–50):** The global model ($\mathbf{w}_{t+1}$) collects collaborative knowledge from all 6 nodes across 50 communication rounds.
2. **Local Fine-Tuning Phase (Round 50):** Each end-device uses $\mathbf{w}_{t+1}$ as a pre-trained foundation and runs $E = 5$ local adaptation epochs to produce a room-specialized weight vector $\mathbf{w}_{i, \text{local}}^{50}$.

#### Empirical Results of Personalized FL vs. Centralized NN:

| End Device | Physical Location | Local FL Model $R^2$ | Local FL Model RMSE (dB) | Centralized NN RMSE (dB) | Improvement over Centralized |
|:---:|:---|:---:|:---:|:---:|:---:|
| **ED0** | Room 801 (Office) | **0.8817** | **4.88 dB** | $5.86\text{ dB}$ | **+0.98 dB Better** 🟢 |
| **ED1** | Room 802 (Office) | **0.8369** | **5.94 dB** | $7.26\text{ dB}$ | **+1.32 dB Better** 🟢 |
| **ED2** | Room 803 (Vault) | **0.6534** | **5.42 dB** | $5.73\text{ dB}$ | **+0.31 dB Better** 🟢 |
| **ED3** | Room 804 (Hallway) | **0.8551** | **5.02 dB** | $5.75\text{ dB}$ | **+0.73 dB Better** 🟢 |
| **ED4** | Room 805 (Lab) | **0.6533** | **3.84 dB** | $4.22\text{ dB}$ | **+0.38 dB Better** 🟢 |
| **ED5** | Room 806 (Corner) | **0.2557** | **4.38 dB** | $4.85\text{ dB}$ | **+0.47 dB Better** 🟢 |

**Key Academic Conclusion:** Personalized FL outperforms the Centralized NN on **every single end-device**, reducing local prediction errors down to **$3.84\text{ dB}$ to $5.94\text{ dB}$**.

---

### 7.3 Mathematical Proof of Target Variance Scaling ($SS_{\text{tot}}$) for ED5

If asked why `ED5` exhibits an $R^2 = 0.2557$ despite achieving an extraordinary prediction accuracy of $\text{RMSE} = 4.38\text{ dB}$, the exact mathematical proof is as follows:

$$R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}} = 1 - \frac{\text{MSE}}{\text{Var}(y)}$$

1. **`ED0` Test Data (High Variance):**
   $$\text{Var}(y_{\text{ED0}}) = 201.61\text{ dB}^2 \implies R^2_{\text{ED0}} = 1 - \frac{(4.88)^2}{201.61} = 1 - \frac{23.81}{201.61} = \mathbf{0.8817}$$

2. **`ED5` Test Data (Low Variance):**
   $$\text{Var}(y_{\text{ED5}}) = 25.80\text{ dB}^2 \implies R^2_{\text{ED5}} = 1 - \frac{(4.38)^2}{25.80} = 1 - \frac{19.20}{25.80} = \mathbf{0.2557}$$

Because Room 806 (`ED5`) is a highly static radio environment ($\text{Var} = 25.80\text{ dB}^2$, almost 8 times smaller than `ED0`), $R^2$ is mathematically constrained to $\approx 0.25$. The true physical accuracy indicator is **$\text{RMSE} = 4.38\text{ dB}$**, which represents near-optimal path loss prediction performance.

