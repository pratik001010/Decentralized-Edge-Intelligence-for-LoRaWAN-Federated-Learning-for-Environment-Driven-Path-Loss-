# Federated Learning — Master Reference Document
## Thesis: Decentralized Edge Intelligence for LoRaWAN

> **Golden Rule:** We are doing exactly what the supervisor's paper did (centralized path loss
> modeling), but replacing centralized training with Federated Learning across 6 devices.
> Every input variable, every target, and every modeling choice must trace back directly
> to the supervisor's paper: *"Environment-aware indoor LoRaWAN path loss: parametric
> regression comparisons, shadow fading, and calibrated fade margins"*
> — Obiri & Van Laerhoven, EURASIP J. Wirel. Commun. Netw. 2026:66

---

## 1. The Dataset

| Property | Value |
|----------|-------|
| **File** | `C:\Users\prati\Desktop\thesis new\3.cleaned_dataset_per_device.csv` |
| **Total rows** | 2,079,534 |
| **Total columns** | 20 |
| **Time span** | 2024-10-01 → 2025-09-30 (exactly 12 months — matches paper's campaign) |
| **Missing values** | ZERO — dataset is already clean |
| **Devices** | 6 (ED0, ED1, ED2, ED3, ED4, ED5) |
| **Rows per device** | ~340,000–355,000 (roughly balanced) |

### Device Row Counts

| Device | Rows |
|--------|------|
| ED0 | 347,154 |
| ED1 | 343,565 |
| ED2 | 348,283 |
| ED3 | 341,472 |
| ED4 | 343,893 |
| ED5 | 355,167 |

---

## 2. All 20 Columns — What Each One Is

| # | Column | What it is | Use in FL |
|---|--------|-----------|-----------|
| 0 | `time` | Measurement timestamp | ID / sorting only |
| 1 | `device_id` | Device label (ED0–ED5) | FL client identity — used to split data |
| 2 | `co2` | CO₂ concentration (ppm) | ✅ **MODEL INPUT** |
| 3 | `humidity` | Relative humidity (%) | ✅ **MODEL INPUT** |
| 4 | `pm25` | PM2.5 particulate matter (µg/m³) | ✅ **MODEL INPUT** |
| 5 | `pressure` | Pressure — RAW (needs ×3.125 correction) | ✅ **MODEL INPUT (after correction)** |
| 6 | `temperature` | Temperature (°C) | ✅ **MODEL INPUT** |
| 7 | `rssi` | RSSI at gateway (dBm) | ❌ NOT AN INPUT — used only to compute exp_pl |
| 8 | `snr` | Signal-to-noise ratio (dB) | ✅ **MODEL INPUT** (γ term in paper's Eq.12) |
| 9 | `SF` | LoRa spreading factor (7–10) | ❌ Not in paper's feature set |
| 10 | `frequency` | Carrier frequency (867.1–868.5 MHz) | ❌ In paper's formula but absorbed into intercept β₀ as a constant |
| 11 | `f_count` | Frame counter | ❌ Metadata only |
| 12 | `p_count` | Packet counter | ❌ Metadata only |
| 13 | `toa` | Time on air (seconds) | ❌ Not in paper's feature set |
| 14 | `distance` | Distance device → gateway (m) | ✅ **MODEL INPUT** |
| 15 | `c_walls` | Number of concrete/brick walls in path | ✅ **MODEL INPUT** |
| 16 | `w_walls` | Number of wooden walls/partitions in path | ✅ **MODEL INPUT** |
| 17 | `exp_pl` | Expected path loss (dB) | ✅ **REGRESSION TARGET — what we predict** |
| 18 | `n_power` | Normalized received power (derived) | ❌ Derived quantity, not used |
| 19 | `esp` | Effective signal power (derived) | ❌ Derived quantity, not used |

---

## 3. The 9 Model Inputs + 1 Target (Paper-Faithful)

This is the **definitive feature set**, derived directly from the supervisor's paper Equation 12:

```
L_ℓ,i = β₀ + 10n·log₁₀(d/d₀) + 20·log₁₀(f) + Σ ωₖ·Wₖ + Σ εⱼ·Eⱼ + kᵧ·γ + ψ
```

Where `20·log₁₀(f)` is a constant at EU868 and is absorbed into β₀.
The paper explicitly states: *"In our single-band EU868 deployment, it is constant
and is absorbed into β₀ during fitting."*

### INPUT FEATURES (9 total)

| # | Column | Paper term | Type | Notes |
|---|--------|-----------|------|-------|
| 1 | `distance` | `d` (linearized as 10·log₁₀(d/d₀)) | Geometric — static per device | Core LDPL term |
| 2 | `c_walls` | `W_brick` | Geometric — static per device | Brick/concrete wall count |
| 3 | `w_walls` | `W_wood` | Geometric — static per device | Wooden partition/door count |
| 4 | `pressure` | `E_BP` | Environmental — dynamic | **Must apply ×3.125 correction first** |
| 5 | `co2` | `E_C` | Environmental — dynamic | Occupancy proxy |
| 6 | `temperature` | `E_T` | Environmental — dynamic | |
| 7 | `humidity` | `E_RH` | Environmental — dynamic | |
| 8 | `pm25` | `E_PM` | Environmental — dynamic | |
| 9 | `snr` | `γ` | Link-state indicator — dynamic | Gateway-reported SNR |

### TARGET (1)

| Column | Paper term | Description |
|--------|-----------|-------------|
| `exp_pl` | `L_ℓ,i` | Expected path loss in dB — continuous regression target |

---

## 4. Why Frequency is NOT an Active Input

- The `frequency` column has only **8 unique values**: 867.1, 867.3, 867.5, 867.7, 867.9, 868.1, 868.3, 868.5 MHz
- These are the EU868 sub-channels that LoRaWAN rotates through automatically
- Standard deviation < 1 MHz — it is essentially a constant band
- Correlation with exp_pl = **−0.019** (negligible)
- The paper's formula includes `20·log₁₀(f)` physically, but **explicitly absorbs it into the intercept**
- We treat it the same way: **not included as a column in the feature matrix**

---

## 5. Why RSSI is NOT an Input

- `rssi` has correlation r = **−1.0000** with `exp_pl`
- This is because `exp_pl` is **defined as**: `L = P_tx − L_tx + G_tx + G_rx − L_rx − RSSI`
- Using RSSI as an input is circular — the model would just learn `y = C − x`
- This was the **fatal error** in the previous simulation (which gave fake R² = 0.9999)
- RSSI is only used to *compute* exp_pl — it is not a predictor

---

## 6. Pressure Correction (Critical)

Raw `pressure` values in the dataset are stored compressed (~287–348):

```
pressure_hPa = pressure_raw × 3.125
→ Result: ~897 to ~1086 hPa (physically realistic sea-level range)
```

This correction **must be applied before any normalization or training**.

---

## 7. Geometric Variables — Static Per Device (Confirmed)

These are measured once for the building layout and never change:

| Device | distance (m) | c_walls (brick) | w_walls (wood) | Notes |
|--------|-------------|-----------------|----------------|-------|
| ED0 | 10 | 0 | 0 | **Only LoS device** |
| ED1 | 8 | 1 | 0 | |
| ED2 | 23 | 0 | 2 | |
| ED3 | 18 | 1 | 2 | |
| ED4 | 37 | 0 | 5 | |
| ED5 | 40 | 2 | 2 | |

Verified: `nunique = 1` for every device for all three columns. Perfectly static.

---

## 8. Target Variable (exp_pl) Distribution

| Statistic | Global | ED0 | ED1 | ED2 | ED3 | ED4 | ED5 |
|-----------|--------|-----|-----|-----|-----|-----|-----|
| **mean** | 88.5 dB | 68.0 | 73.2 | 87.8 | 87.2 | 101.5 | 112.5 |
| **std** | 18.8 dB | 14.0 | 14.2 | 9.3 | 12.6 | 6.7 | 5.0 |
| **min** | 46.3 dB | 46.3 | 48.3 | 56.3 | 60.3 | 65.3 | 82.3 |
| **max** | 145.3 dB | 145.3 | 145.3 | 140.3 | 140.3 | 141.3 | 140.3 |

> **This is genuinely non-IID.** Each device has a different mean path loss driven by
> its geometry (distance + walls). ED0 (LoS, 10m) averages 68 dB; ED5 (40m, 2 brick + 2 wood)
> averages 112 dB — a 44 dB difference. This non-IID character is exactly what makes
> federated learning meaningful and challenging.

---

## 9. Correlations of All Features with exp_pl

| Feature | Correlation r | Interpretation |
|---------|--------------|----------------|
| `rssi` | **−1.0000** | Perfect inverse — circular (NOT input) |
| `distance` | **+0.7917** | Strong — core LDPL geometric term |
| `w_walls` | **+0.5840** | Strong — wooden wall attenuation |
| `snr` | **−0.4747** | Moderate — link-state indicator |
| `c_walls` | **+0.3874** | Moderate — concrete wall attenuation |
| `co2` | −0.1092 | Weak but meaningful — occupancy proxy |
| `temperature` | −0.0697 | Weak |
| `humidity` | +0.0504 | Weak |
| `pressure` | +0.0299 | Very weak |
| `SF` | +0.0292 | Negligible — not in paper |
| `frequency` | −0.0194 | Negligible — constant |
| `toa` | +0.0186 | Negligible — not in paper |
| `pm25` | −0.0143 | Very weak |

> Note: The paper uses all 9 features despite the weak environmental correlations.
> These weak correlations are cross-device (pooled). Within a device + controlled for
> geometry, the environmental signals contribute meaningful variance reduction (~44%
> unexplained variance reduction per the paper's ANOVA). Do NOT drop them.

---

## 10. The Supervisor's Centralized Result (Benchmark to Match/Approach)

From reference [6] in the paper (their earlier dataset descriptor):
- **Structure-only model** (distance + walls only): RMSE = 10.577 dB, R² = 0.691
- **Environment-augmented** (+ environmental covariates + SNR): RMSE = **8.034 dB**, R² = **0.822**

From the current paper (extended dataset, best model):
- **Selective polynomial (POLY2)**: RMSE = **7.38 dB**, R² = **0.84**
- **Linear MLR baseline**: RMSE = **8.23 dB**, R² = **0.81**

> **The thesis FL target:** Achieve R² and RMSE as close as possible to the
> centralized MLR baseline (R² ≈ 0.82, RMSE ≈ 8 dB) using federated learning.
> The research question is: **how much does R² drop when we move from centralized
> to federated?** And is the bandwidth/privacy benefit worth that drop?

---

## 11. Spreading Factor (SF) — Present but Not a Model Input

SF in this dataset cycles through values **7, 8, 9, 10** (matching the paper's stated SF7–SF10 range):

| SF | Count | % |
|----|-------|---|
| 7 | 536,266 | 25.8% |
| 8 | 533,178 | 25.6% |
| 9 | 515,908 | 24.8% |
| 10 | 494,182 | 23.8% |

SF is a **protocol configuration variable**, not an environmental or geometric predictor.
The paper excludes it from the regression model. We do the same.

---

## 12. Data Preprocessing Checklist (For When We Start)

When we begin implementing, the following must happen in order:

- [ ] **1. Load** `3.cleaned_dataset_per_device.csv`
- [ ] **2. Apply pressure correction:** `pressure = pressure × 3.125`
- [ ] **3. Select 9 input features:** `distance, c_walls, w_walls, pressure, co2, temperature, humidity, pm25, snr`
- [ ] **4. Select target:** `exp_pl`
- [ ] **5. Split by device_id** → 6 client datasets (non-IID, exactly as deployed)
- [ ] **6. Train/test split** (time-ordered — use chronological split, not random, per paper's protocol)
- [ ] **7. Normalize** inputs with StandardScaler — fit on training data only, apply to test
- [ ] **8. DO NOT include:** `rssi`, `frequency`, `SF`, `toa`, `f_count`, `p_count`, `n_power`, `esp`

---

## 13. Model Architecture (Follows Directly from Features)

```
Input Layer  (9 neurons):
  distance
  c_walls      →  [Hidden Layer: 8 neurons, ReLU]  →  [Output: 1 neuron, Linear]  →  exp_pl (dB)
  w_walls
  pressure
  co2
  temperature
  humidity
  pm25
  snr

Parameters: (9×8 + 8) + (8×1 + 1) = 80 + 9 = 89 total parameters
Architecture name: Dense(9→8→1)
Task: Regression
Loss: MSE (Mean Squared Error)
Metric: R², RMSE (dB), MAE (dB)
```

---

## 14. The Federated Learning Setup (To Be Designed)

These are the questions to answer before we write a single line of FL code:

| Question | Status |
|----------|--------|
| FL algorithm | FedAvg (McMahan et al., 2017) — matches CE-FedAvg paper base |
| Number of clients | 6 (ED0–ED5), one per physical device |
| Client data split | By `device_id` — genuinely non-IID |
| Target centralized R² | ≈ 0.82 (paper's MLR result) |
| Expected FL degradation | Unknown — this is what the thesis measures |
| Number of FL rounds | To be determined |
| Local epochs per round | To be determined (paper tested 1, 3, 5 — CE-FedAvg found 3 optimal) |
| Communication cost | 89 params × 1 byte (int8) = 89 bytes per FL update |
| LoRaWAN compliance | 89 bytes < 222 bytes (DR5 max) — fits in 1 frame |

---

*Document created: 2026-06-19*
*Dataset: 3.cleaned_dataset_per_device.csv (2,079,534 rows, 12 months, 6 devices)*
*Primary paper: Obiri & Van Laerhoven, EURASIP J. Wirel. Commun. Netw. 2026:66*
*DO NOT proceed with any implementation until explicitly instructed.*
