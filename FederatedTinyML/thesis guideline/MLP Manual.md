# MLP Manual (89-Parameter Neural Network Architecture)

**Thesis Title**: *Decentralized Edge Intelligence for LoRaWAN: Federated Learning for Environment-Driven Path Loss and Link Quality Modeling*  
**Author**: Pratik Khadka  
**Institution**: University of Siegen  
**Target Architecture**: 89-Parameter Multilayer Perceptron (MLP)  
**File Name**: `MLP Manual.md`  

---

## 1. Complete Architecture & The 89 Parameters

The neural network is an ultra-compact **Multilayer Perceptron (MLP)** specifically designed for TinyML deployment on microcontrollers (e.g. Arduino MKR WAN 1310 with Arm Cortex-M0+ @ 48 MHz, 32 KB SRAM, 256 KB Flash):

$$\text{Input Layer (9 Features)} \longrightarrow \text{Dense Hidden (8 Neurons, ReLU)} \longrightarrow \text{Dense Output (1 Neuron, Linear)}$$

```
                   ┌─────────────────────────────────────────┐
                   │        FEATURE INPUT VECTOR (D = 9)     │
                   │ log_distance, W_brick, W_wood, co2,     │
                   │ humidity, pm25, pressure, temp, snr     │
                   └────────────────────┬────────────────────┘
                                        │ (72 Weights + 8 Biases)
                   ┌────────────────────▼────────────────────┐
                   │    HIDDEN DENSE LAYER (8 Neurons)       │
                   │    Activation: ReLU(z) = max(0, z)      │
                   └────────────────────┬────────────────────┘
                                        │ (8 Weights + 1 Bias)
                   ┌────────────────────▼────────────────────┐
                   │    OUTPUT DENSE LAYER (1 Neuron)        │
                   │    Activation: Linear (exp_pl in dB)    │
                   └─────────────────────────────────────────┘
```

### Exact Breakdown of Weights & Biases (89 Parameters)

| Layer | Layer Type | Connections | Weights Calculation | Biases Calculation | Total Layer Parameters |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Layer 1** | Hidden Dense | 9 Input Features $\to$ 8 Neurons | $9 \times 8 = 72$ Weights | $8 \text{ Neurons} = 8$ Biases | **80 Parameters** |
| **Layer 2** | Output Dense | 8 Hidden Neurons $\to$ 1 Output | $8 \times 1 = 8$ Weights | $1 \text{ Output} = 1$ Bias | **9 Parameters** |
| **TOTAL** | **Full MLP** | **9 $\to$ 8 $\to$ 1 Architecture** | **80 Total Weights** | **9 Total Biases** | **89 Parameters** |

> **Memory Footprint on Edge Hardware**:  
> 89 parameters $\times$ 4 bytes (float32) = **356 bytes** of RAM. Quantized to 8-bit integers (int8), the entire model requires only **89 bytes**, fitting effortlessly inside microcontrollers with 32 KB SRAM.

---

## 2. Forward Propagation

Forward propagation calculates path loss predictions ($\hat{y}$) from environmental telemetry:

### 2.1 Ground Truth Path Loss Target Calculation
Ground truth path loss ($\text{PL}_{\text{exp}}$) is derived from Received Signal Strength Indicator ($\text{RSSI}$) measured at the gateway:

$$\text{PL}_{\text{exp}} = P_{\text{tx}} - \text{RSSI} = 14 - \text{RSSI}$$

*(where $P_{\text{tx}} = +14\text{ dBm}$ is the EU868 LoRaWAN transmit power)*.

### 2.2 Step 1: Hidden Layer Matrix Operations
For an input feature vector $\mathbf{x} = [x_1, x_2, \dots, x_9]^T$, the linear combination $z_j^{(1)}$ for hidden neuron $j \in \{1, \dots, 8\}$ is:

$$z_j^{(1)} = \sum_{i=1}^{9} W_{ij}^{(1)} x_i + b_j^{(1)}$$

Applying the Rectified Linear Unit ($\text{ReLU}$) activation function yields hidden activation $h_j$:

$$h_j = \text{ReLU}\left(z_j^{(1)}\right) = \max\left(0, z_j^{(1)}\right)$$

### 2.3 Step 2: Output Layer Computation
The predicted experimental path loss $\hat{y}$ ($\text{exp\_pl}$ in $\text{dB}$) is computed by the linear output neuron:

$$\hat{y} = \sum_{j=1}^{8} W_{j1}^{(2)} h_j + b_1^{(2)}$$

---

## 3. Backward Propagation & Loss Gradients

### 3.1 Loss Function
Training minimizes Mean Squared Error ($\text{MSE}$) across a mini-batch of $B = 512$ samples:

$$L = \text{MSE} = \frac{1}{B} \sum_{k=1}^{B} \left( y_k - \hat{y}_k \right)^2$$

### 3.2 Definition of Loss Gradients ($\nabla L$)
- The gradient $\nabla L = \frac{\partial L}{\partial W}$ is the partial derivative (slope) of the loss function relative to each weight or bias parameter.
- It specifies both the **direction of maximum error increase** and the **steepness** of the error surface.

### 3.3 Backpropagation Chain Rule Derivatives
1. **Output Layer Error Gradient**:
   $$\delta^{(2)} = \frac{\partial L}{\partial z^{(2)}} = -2 \cdot \left( y - \hat{y} \right)$$

2. **Output Weights & Bias Gradients**:
   $$\frac{\partial L}{\partial W_{j1}^{(2)}} = \delta^{(2)} \cdot h_j, \qquad \frac{\partial L}{\partial b_1^{(2)}} = \delta^{(2)}$$

3. **Hidden Layer Error Gradient**:
   $$\delta_j^{(1)} = \left( \delta^{(2)} \cdot W_{j1}^{(2)} \right) \cdot f'\left(z_j^{(1)}\right), \quad \text{where } f'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \le 0 \end{cases}$$

4. **Hidden Weights & Bias Gradients**:
   $$\frac{\partial L}{\partial W_{ij}^{(1)}} = \delta_j^{(1)} \cdot x_i, \qquad \frac{\partial L}{\partial b_j^{(1)}} = \delta_j^{(1)}$$

### 3.4 Weight and Bias Subtraction Rule (Gradient Descent)
Weights and biases are **NOT** aggregated sample-by-sample. After every mini-batch of 512 samples, the optimizer updates all 89 parameters simultaneously via subtraction:

$$W_{\text{new}} = W_{\text{old}} - \left( \eta \cdot \frac{\partial L}{\partial W} \right)$$

$$b_{\text{new}} = b_{\text{old}} - \left( \eta \cdot \frac{\partial L}{\partial b} \right)$$

> **Fundamental Update Rule**:  
> $$\mathbf{\text{New Parameter} = \text{Old Parameter} - \left( \text{Learning Rate} \times \text{Gradient} \right)}$$

---

## 4. The Learning Rate ($\eta = 0.01$)

The **Learning Rate ($\eta = 0.01$)** acts as a scaling multiplier on the gradient vector, controlling the step size taken downhill toward zero error:

- **Intuitive Mountain Analogy**:
  - $\eta = 1.0$ (Too Large): Giant leaps that overshoot the valley, causing weight oscillations and gradient explosion.
  - $\eta = 0.0001$ (Too Small): Tiny baby steps taking thousands of epochs, draining computational energy.
  - $\eta = 0.01$ (Optimal): Steady strides achieving **97.5% convergence in 10 epochs**.

### Impact on Microcontroller Execution & Battery Life
On a 48 MHz microcontroller (Arduino MKR WAN 1310):
- Running backpropagation drains battery.
- Setting $\eta = 0.01$ allows full convergence in **5–10 local epochs**.
- Saves **80% of CPU execution time and battery energy**, enabling the node to return to deep-sleep mode quickly.

---

## 5. How Centralized Learning Works with the 89-Parameter MLP

In **Centralized Learning**:
1. **Data Pooling**: Telemetry from all 6 end-devices (`ED0` to `ED5`, 206,957 rows) is collected in a central cloud server.
2. **Data Splitting**:
   - **80% Global Train / 20% Held-Out Test** ($41,393$ test samples).
   - **85% Model Train ($140,729$ samples) / 15% Validation ($24,835$ samples)** carved from the 80% train set.
3. **Training Dynamics**:
   - The single central MLP updates its 89 parameters **275 times per epoch** ($140,729 / 512 \approx 275$).
   - Across 50 epochs, the 89 parameters are updated **13,750 times** in RAM.
4. **Single-Pass Test Evaluation**:
   - After Epoch 50, the final locked-in 89 parameters are evaluated **ONLY ONCE** on the 41,393 held-out test samples.
5. **Centralized Benchmark Results**:
   - **Held-Out Test $R^2$**: **0.9089** (Gold standard upper bound)
   - **Held-Out Test RMSE**: **5.6872 dB**
   - **Overfitting**: **Zero Overfitting** ($\Delta R^2 = 0.0018$).

---

## 6. How Federated Learning Works with the 89-Parameter MLP

In **Federated Learning (FedAvg)**:
1. **Data Privacy**: Telemetry stays on local end-devices (`ED0` to `ED5`). Raw data is never sent to the server.
2. **Local Model Training**:
   - Each device maintains its own copy of the 89-parameter MLP.
   - In Round $t$, each device receives global parameters $W^t, b^t$ from the server.
   - Each device trains locally on its 80% local dataset for $E = 5$ local epochs using gradient descent subtraction:
     $$W_{i, \text{new}} = W_{i, \text{old}} - \left( \eta \cdot \frac{\partial L_i}{\partial W_i} \right)$$
3. **Weight Upload & Server Aggregation**:
   - Devices transmit only their updated 89 parameters ($W_i, b_i$) to the FL server (requiring only **356 bytes** of uplink bandwidth).
   - The central FL server averages the 89 parameters across all 6 clients:
     $$W^{\text{global}} = \frac{N_0 W_0 + N_1 W_1 + N_2 W_2 + N_3 W_3 + N_4 W_4 + N_5 W_5}{N_{\text{total}}}$$
     $$b^{\text{global}} = \frac{N_0 b_0 + N_1 b_1 + N_2 b_2 + N_3 b_3 + N_4 b_4 + N_5 b_5}{N_{\text{total}}}$$
4. **Broadcast & Repeat**: The aggregated global 89 parameters ($W^{\text{global}}, b^{\text{global}}$) are broadcast back to the clients for Round $t+1$.

---

## 7. Master Comparison Summary: Centralized vs. Federated MLP

| Feature / Aspect | Centralized MLP | Federated Learning MLP (FedAvg) |
| :--- | :--- | :--- |
| **Model Parameters** | 89 Parameters (80 Weights, 9 Biases) | 89 Parameters (80 Weights, 9 Biases) |
| **Data Location** | Pooled centrally on cloud server | Retained locally on edge devices |
| **Local Training (Gradient Descent)** | Performed centrally ($13,750$ updates) | Performed locally on MCU for $E=5$ epochs |
| **Weight Subtraction Rule** | $W \leftarrow W - \eta \nabla L$ | $W_i \leftarrow W_i - \eta \nabla L_i$ |
| **Server Aggregation** | **None** (1 global model) | **Weighted Average** ($W^{\text{global}} = \sum \frac{N_i}{N} W_i$) |
| **Data Transmission Per Round** | High (transmits all telemetry rows) | Minimal (transmits only 356 bytes of weights) |
| **Held-Out Test $R^2$ Score** | **0.9089** (Gold Standard Benchmark) | **~0.9010 – 0.9050** (Near-Optimal) |
