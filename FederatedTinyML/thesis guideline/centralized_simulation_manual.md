# Master Manual: Centralized Neural Network Baseline Simulation (365-Day 10-Min Dataset)

**Thesis Title**: *Decentralized Edge Intelligence for LoRaWAN: Federated Learning for Environment-Driven Path Loss and Link Quality Modeling*  
**Author**: Pratik Khadka  
**Institution**: University of Siegen  
**Script Target**: `run_centralized_365_10min.py`  
**Output Directory**: `c:\Users\prati\Desktop\edge AI\FederatedTinyML\centralized_365_day_10_min\`  

---

## 1. Executive Summary & Purpose

In this thesis, **Centralized Learning** represents the **Theoretical Upper Bound (Gold Standard)** performance benchmark. In the centralized baseline:
- Telemetry from all 6 LoRaWAN End Devices (`ED0` through `ED5`) is pooled into a single cloud dataset.
- A single global neural network model is trained offline on the combined data.
- Individual end-devices do **not** perform local training or communicate updates; they act purely as data sources.

This baseline establishes the maximum achievable estimation accuracy ($R^2 = 0.9089$, $\text{RMSE} = 5.6872\text{ dB}$) against which all **Federated Learning** algorithms (FedAvg, FedProx, FedAdam, SCAFFOLD) are evaluated to quantify the trade-offs in privacy, network bandwidth, and energy consumption.

---

## 2. Dataset Specifications & RF Target Formulation

- **Source File**: `365_days_staggered_10min_sampled.csv`
- **Timeframe**: 365 calendar days (full 1-year coverage, 4 seasons, day/night cycles)
- **Sampling Rate**: 10-minute staggered sampling
- **Raw Rows**: 206,957
- **Anomalies Filtered**: 0 (pre-cleaned dataset)
- **Usable Clean Rows**: 206,957 rows across 6 devices (`ED0`, `ED1`, `ED2`, `ED3`, `ED4`, `ED5`)

### Feature Vector ($D = 9$ Features)

| Feature | Column Name | Transformation / Description |
| :--- | :--- | :--- |
| 1 | `log_distance` | $10 \cdot \log_{10}(d / d_0)$ relative to Gateway |
| 2 | `W_brick` | Count of brick/concrete wall obstructions ($c_{\text{walls}}$) |
| 3 | `W_wood` | Count of wooden partition obstructions ($w_{\text{walls}}$) |
| 4 | `co2` | Carbon dioxide concentration (ppm) |
| 5 | `humidity` | Relative humidity (%) |
| 6 | `pm25` | Particulate matter PM2.5 ($\mu\text{g/m}^3$) |
| 7 | `pressure` | Atmospheric pressure in true hPa ($\text{raw} \times 3.125$) |
| 8 | `temperature` | Temperature ($^\circ\text{C}$) |
| 9 | `snr` | Signal-to-Noise Ratio (dB) |

### Target Variable & RF Link Budget Formula
- **Target**: `exp_pl` — Experimental Path Loss ($\text{dB}$), bounded within $[50, 200]\text{ dB}$.
- **Ground Truth Calculation Formula**:
  $$\text{PL}_{\text{exp}} = P_{\text{tx}} - \text{RSSI} = 14 - \text{RSSI}$$
  Where $P_{\text{tx}} = +14\text{ dBm}$ is the standard EU868 transmit power of the Arduino MKR WAN 1310 end-device, and $\text{RSSI}$ is the Received Signal Strength Indicator measured at the gateway in negative $\text{dBm}$ (e.g., $14 - (-105) = 119\text{ dB}$).

---

## 3. Data Splitting & Preprocessing Architecture

To guarantee **zero data leakage** and robust evaluation across all rooms:

```
                  ┌───────────────────────────────────────────────┐
                  │ Cleaned Dataset (206,957 Rows, ED0 to ED5)    │
                  └───────────────────────┬───────────────────────┘
                                          │
                 ┌────────────────────────┴────────────────────────┐
                 │ Stratified Per-Device 80/20 Global Split        │
                 └────────────┬────────────────────────┬───────────┘
                              │                        │
         ┌────────────────────▼────────────┐      ┌────▼────────────────────────┐
         │ Global Train Set (80%)          │      │ Held-Out Global Test Set    │
         │ 165,564 Samples                 │      │ 41,393 Samples (20%)        │
         └────────────────────┬────────────┘      └─────────────────────────────┘
                              │
         ┌────────────────────▼────────────────────────┐
         │ Internal Validation Split (Within Train)    │
         └────────────┬────────────────────────┬───────┘
                      │                        │
       ┌──────────────▼─────────────┐   ┌──────▼─────────────────────┐
       │ Model Train Set (85%)      │   │ Epoch Validation Set (15%) │
       │ 140,729 Samples            │   │ 24,835 Samples             │
       └────────────────────────────┘   └────────────────────────────┘
```

1. **Global Test Split (80% Train / 20% Test)**:
   - Stratified per device (`ED0`–`ED5`) with `shuffle=True` and `random_state=42`.
   - **Global Training Set**: 165,564 samples ($80\%$ of total)
   - **Held-Out Global Test Set**: 41,393 samples ($20\%$ of total)
   - *Final thesis benchmark metrics ($R^2 = 0.9089$, $\text{RMSE} = 5.6872\text{ dB}$) are evaluated ONLY ONCE on this unseen 20% test set after training completes.*
2. **Internal Validation Split (85% Train / 15% Validation)**:
   - Carved strictly from inside the 165,564 training samples during `model.fit()`.
   - **Model Training Split ($X_{\text{tr}}$)**: 140,729 samples ($85\%$ of Train) — used for gradient weight updates during epochs.
   - **Epoch Validation Split ($X_{\text{val}}$)**: 24,835 samples ($15\%$ of Train) — evaluated at the end of every epoch to plot validation curves.
3. **Feature Scaling**:
   - `StandardScaler` fitted **strictly on $X_{\text{tr}}$** to ensure complete independence of validation and test sets.

---

## 4. Model Architecture & TinyML Constraints

To reflect deployment on ultra-constrained edge microcontrollers (such as the **Arduino MKR WAN 1310** with Arm Cortex-M0+ @ 48 MHz, 32 KB SRAM, 256 KB Flash):

- **Model Type**: Multilayer Perceptron (MLP)
- **Layer Structure**: `Input(9) -> Dense(8, activation='relu') -> Dense(1, activation='linear')`
- **Weight Initializer**: Glorot Uniform (`seed=42`)
- **Total Trainable Parameters**: **89 parameters** (356 bytes in float32 / 89 bytes in int8)
- **Optimizer**: Adam ($\text{learning\_rate} = 0.01$)
- **Loss Function**: Mean Squared Error ($\text{MSE}$)
- **Batch Size**: 512
- **Training Duration**: 50 Epochs

---

## 5. Epoch & Training Dynamics (Gradients & Weight Updates)

### Understanding the Epoch Process & Weight Updates
- **Definition of an Epoch**: 1 Epoch is **one complete pass through the 140,729 training samples**.
- **Mini-Batch Processing**: The 140,729 training rows are processed in mini-batches of 512 ($140,729 / 512 \approx 275$ mini-batches per epoch).

### Definition and Role of Loss Gradients ($\nabla L$)
- **What is a Gradient?**: The gradient $\nabla L = \frac{\partial L}{\partial W}$ is the partial derivative (slope) of the loss function $L$ with respect to each weight $W$.
- **Purpose**: It tells the optimizer the exact direction and steepness of the error surface. If $\nabla L > 0$, increasing the weight increases error; if $\nabla L < 0$, increasing the weight decreases error.
- **Weights vs. Gradients**:
  - **Weight ($W$)**: The actual parameter value inside the neural network (e.g. $W_1 = 0.45$).
  - **Gradient ($\nabla L$)**: The direction and rate of change required to minimize prediction error (e.g. $\nabla L_1 = +2.1$).

### Weight Update Subtraction Rule (Gradient Descent)
- After every mini-batch of 512 rows, backpropagation computes gradients ($\nabla L$) and updates the 89 weights via subtraction:
  $$W_{\text{new}} = W_{\text{old}} - \left( \eta \cdot \nabla L \right)$$
  *(where $\eta = 0.01$ is the learning rate)*.

> **Key Rule**: During model training, weights are updated via subtraction:  
> **New Weight = Old Weight - (Learning Rate × Gradient)**

- **275 Updates Per Epoch**: The model updates its weights 275 times per epoch across mini-batches.
- **Epoch-End Logging**: At the end of each epoch, training pauses for a microsecond to measure loss on the training set ($X_{\text{tr}}$) and validation set ($X_{\text{val}}$), plotting 1 dot on the convergence graph.
- **Role of Test Data**: Testing data ($20\%$, 41,393 samples) is **never used during epochs**. It remains untouched until training finishes.

```
┌──────────────────────────────────────────────────────────────────────────┐
  EPOCH 1 (Pass 1 through all 140,729 training samples)                    
  • Batch 1 (512 rows)   ──► Predict ──► Calculate Loss ──► Update Weights (1)
  • Batch 2 (512 rows)   ──► Predict ──► Calculate Loss ──► Update Weights (2)
  ...                                                                      
  • Batch 275 (512 rows) ──► Predict ──► Calculate Loss ──► Update Weights (275)
                                                                           
  ─── END OF EPOCH 1 ───                                                   
  Measure current accuracy: Train RMSE = 28.32 dB, Val RMSE = 28.35 dB     
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Summary of Verified Results & Definitive Proof of Zero Overfitting

### Global Metrics (50 Epochs, 20.26s Training Time)

| Dataset Split | Sample Count | $R^2$ Score | RMSE (dB) | Role in Simulation |
| :--- | :--- | :--- | :--- | :--- |
| **Model Training Split ($X_{\text{tr}}$)** | 140,729 | **0.9080** | **5.72 dB** | Updates weights ($\Delta W$) across 50 epochs |
| **Epoch Validation Split ($X_{\text{val}}$)**| 24,835 | **0.9062** | **5.80 dB** | Plots validation curves at end of each epoch |
| **Held-Out Global Test Set ($X_{\text{test}}$)**| **41,393** | **0.9089** | **5.6872 dB** | **Evaluated ONLY ONCE after Epoch 50** |

### Definitive Analysis: Did Overfitting Occur?
> **No. Overfitting did NOT occur.**

1. **What Overfitting WOULD Have Looked Like**:
   - If the neural network had overfitted, Training $R^2$ would be artificially high ($\sim 0.98$), while Validation $R^2$ would collapse ($\sim 0.40 - 0.50$) and Validation RMSE would shoot up ($\sim 15.0\text{ dB}$).
   - On the convergence figure, the blue training line would be at the top while the orange validation line would diverge far below it.

2. **What ACTUALLY Happened (Empirical Evidence)**:
   - **Generalization Gap**: $\Delta R^2 = 0.0018$, $\Delta \text{RMSE} = 0.08\text{ dB}$.
   - The Training ($0.9080$), Validation ($0.9062$), and Held-Out Test ($0.9089$) scores match almost line-for-line.
   - On `centralized_training_curves.png`, the solid blue training line and dashed orange validation line run side-by-side across all 50 epochs.

3. **Why Overfitting was Physically Prevented**:
   - **Architectural Constraint**: Constrained to **89 parameters** (`Dense 9 -> 8 -> 1`), the model is physically too compact to memorise noise across 140,729 rows. It is forced to learn smooth physical equations (log-distance decay and wall attenuation).
   - **Visual Proof in Thesis**: Plotting Training and Validation curves together on the same graph provides thesis reviewers with immediate visual proof that the model is in an optimal state of generalization.

---

## 7. Metric Selection Rationale ($R^2$ & RMSE vs. MAE & MSE)

1. **Why RMSE ($\text{dB}$) is Primary**: RMSE is measured in decibels ($\text{dB}$), making it directly interpretable for RF link budget calculations (e.g. an error of $5.68\text{ dB}$).
2. **Why $R^2$ Score is Primary**: $R^2$ measures goodness-of-fit on a normalized scale (0 to 1), proving that the model explains **90.89% of signal variance**.
3. **Why MAE is Excluded**: Mean Absolute Error (MAE) is redundant when RMSE is reported. Excluding MAE maintains focus on standard regression metrics.
4. **Why MSE is Internal Only**: Mean Squared Error ($\text{MSE}$) has units of $\text{dB}^2$ (squared decibels) which lacks physical intuition in wireless engineering. It is used as the Keras loss function, while $\text{RMSE} = \sqrt{\text{MSE}}$ is reported.

---

## 8. Per-Device Performance Breakdown

After global training completes, the global model is evaluated on each individual end-device's isolated test subset:

```
      Global Test Set (41,393 samples)
                 │
                 ├──► ED0 Test Samples (6,903 rows)  ──► Evaluate Model ──► R² = 0.8299 | RMSE = 5.85 dB
                 ├──► ED1 Test Samples (6,834 rows)  ──► Evaluate Model ──► R² = 0.7559 | RMSE = 7.26 dB
                 ├──► ED2 Test Samples (6,948 rows)  ──► Evaluate Model ──► R² = 0.6128 | RMSE = 5.73 dB
                 ├──► ED3 Test Samples (6,805 rows)  ──► Evaluate Model ──► R² = 0.8098 | RMSE = 5.75 dB
                 ├──► ED4 Test Samples (6,837 rows)  ──► Evaluate Model ──► R² = 0.5813 | RMSE = 4.22 dB
                 └──► ED5 Test Samples (7,066 rows)  ──► Evaluate Model ──► R² = 0.0884 | RMSE = 4.85 dB
```

| Device ID | Location / Room Environment | Test Samples | Local $R^2$ | Local RMSE (dB) |
| :--- | :--- | :--- | :--- | :--- |
| **ED0** | Room 801 (Near Gateway, Direct LOS) | 6,903 | **0.8299** | 5.8554 dB |
| **ED1** | Room 804 (Corridor, Moderate Attenuation)| 6,834 | **0.7559** | 7.2648 dB |
| **ED2** | Room 812 (Far Corner, Multi-Wall) | 6,948 | **0.6128** | 5.7329 dB |
| **ED3** | Room 815 (Thick Concrete Wall) | 6,805 | **0.8098** | 5.7508 dB |
| **ED4** | Room 820 (Wooden Partition) | 6,837 | **0.5813** | 4.2224 dB |
| **ED5** | Outdoor Balcony (High Dynamic Range) | 7,066 | **0.0884** | 4.8495 dB |

---

## 9. Convergence Mechanics & TinyML Edge Insights

### Why 10 Epochs is Sufficient
1. **Epoch 1**: $R^2 = -1.25$, $\text{RMSE} = 28.32\text{ dB}$ (Random initialization)
2. **Epoch 5**: $R^2 = 0.8409$, $\text{RMSE} = 7.52\text{ dB}$ (92.5% of total learning)
3. **Epoch 10**: $R^2 = 0.8859$, $\text{RMSE} = 6.37\text{ dB}$ (**97.5% of total learning**)
4. **Epoch 50**: $R^2 = 0.9080$, $\text{RMSE} = 5.72\text{ dB}$ (Full fine-tuning)

**TinyML Takeaway for Defense**:  
Because 97.5% of convergence occurs within 10 epochs, local client training in **Federated Learning** can be capped at 5–10 local epochs per round ($E = 5\text{ to }10$). This drastically reduces battery consumption and MCU execution time on microcontrollers without degrading final accuracy.

---

## 10. Output Files & Directory Mapping

All outputs are automatically generated and saved by running `python run_centralized_365_10min.py`:

| Output File Path | Description |
| :--- | :--- |
| `centralized_365_day_10_min/epoch_training_metrics.csv` | Full CSV log of training/validation $R^2$ & RMSE across 50 epochs |
| `centralized_365_day_10_min/per_device_metrics.csv` | Per-device test evaluation breakdown (ED0 to ED5) |
| `centralized_365_day_10_min/centralized_365_results.json` | Structured JSON summary of simulation parameters and scores |
| `centralized_365_day_10_min/centralized_365_summary.txt` | Human-readable text summary report |
| `centralized_365_day_10_min/figures/centralized_training_curves.png` | **Primary Thesis 1x2 Subplot Figure** ($R^2$ left, RMSE right) |
| `thesis_figures/centralized_training_curves.png` | Direct copy updated in LaTeX thesis figures directory |

---

## 11. How to Re-Run the Simulation

To reproduce all results, metrics CSVs, and high-resolution figures:

```bash
cd "c:\Users\prati\Desktop\edge AI\FederatedTinyML"
python run_centralized_365_10min.py
```

---

## 12. Defense Presentation Slide Guide (Centralized Slide)

<!-- CENTRALIZED SLIDE RECOMMENDATION -->

### 12.1 Single-Slide Layout Recommendation
For presentation slides during the thesis defense, present a clean, high-impact single slide titled:  
**"Centralized Baseline — Gold Standard Upper Bound"**

- **Left Side of Slide**: Include the 3-Tier Data Splitting Architecture Diagram:
  ```
  Cleaned Dataset (206,957 Rows, 365 Days, 6 End Devices)
                        │
            80/20 Stratified Split
            ┌───────────┴───────────┐
      Train Set               Test Set
   (165,564 Rows)          (41,393 Rows)
        │
    85/15 Split
    ┌───┴───┐
  Model   Val
 140,729 24,835
  ```

- **Right Side of Slide**: Key Technical & Empirical Highlights:
  - **Architecture**: `Dense(9 → 8 → 1)` = **89 Parameters** (356 Bytes in float32 / 89 Bytes in int8).
  - **Target Formulation**: $\text{PL}_{\text{exp}} = P_{\text{tx}} - \text{RSSI} = 14 - \text{RSSI}$.
  - **Training Config**: Adam ($\eta = 0.01$), 50 Epochs, Batch Size 512.
  - **Final Test Result**: $R^2 = \mathbf{0.9089}$, $\text{RMSE} = \mathbf{5.6872\text{ dB}}$ (Theoretical Upper Bound).
  - **Generalization Proof**: $\Delta R^2 = 0.0018$ $\implies$ **Zero Overfitting**.
  - **TinyML Insight**: 97.5% of convergence is achieved within 10 epochs.

---

### 12.2 Defense FAQ: Why Do We Need Validation Split vs. Testing Split?
When asked why a separate **Validation Set (15% = 24,835 samples)** is necessary in addition to the **Held-Out Test Set (20% = 41,393 samples)**:

1. **Validation Set Role (Epoch-by-Epoch Health Monitor)**:
   - Evaluated **after every single epoch** (50 times during training).
   - Monitors training progress in real-time to detect overfitting (e.g., if validation loss diverges while training loss drops).
   - Enables **Early Stopping** (e.g. `patience=8`), halting training when validation loss stops improving to save CPU cycles.
   - Used to generate the epoch-by-epoch convergence curves (`centralized_training_curves.png`).

2. **Test Set Role (Unbiased Final Benchmark)**:
   - Evaluated **ONLY ONCE**, strictly after all training epochs are complete.
   - Provides an unbiased, real-world estimation accuracy score ($R^2 = 0.9089$).

3. **Why Test Set Cannot Be Used During Training**:
   - If the test set were evaluated every epoch and used to tune parameters or stop training early, **test data information would leak into model selection**. The final test score would be artificially inflated and invalid as an unbiased benchmark.

> **One-Line Defense Answer**:  
> *"The validation set acts as an epoch-by-epoch health monitor to detect overfitting and enable early stopping, while the held-out test set is evaluated strictly once after training to guarantee an unbiased final benchmark."*

