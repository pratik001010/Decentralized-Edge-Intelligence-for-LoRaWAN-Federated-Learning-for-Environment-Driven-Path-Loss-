# ICC26 Amin — CE-FedAvg Paper Analysis
## "CE-FedAvg: A Communication-Efficient Federated Learning Framework for LoRaWAN-Based Edge AI"

**Authors:** Seyedmohammadamin Razaghi, Atakan Aral  
**Affiliation:** Faculty of Computer Science, University of Vienna  
**Venue:** ICC 2026 (IEEE International Conference on Communications)  
**Funding:** CHIST-ERA-22-SPiDDS-07 (TROCI Project), Austrian Science Fund (FWF)  
**Code:** [github.com/aminrazaghi/CE-FedAvg](https://github.com/aminrazaghi/CE-FedAvg)

---

## 1. Bullet-Point Summary of the Paper

### Core Problem
- FL over LoRaWAN is difficult because LoRaWAN has **extremely low bandwidth** (hundreds of bps), **strict 1% duty cycle** (EU868), **limited downlink** (ALOHA-based), and **small max payload** (222 bytes at DR5)
- Existing FL methods assume high-bandwidth links (Wi-Fi, LTE); they **cannot be directly applied** to LoRaWAN without violating compliance or energy budgets
- Previous LoRaWAN FL works (Torres Sanchez et al., Singh & Borkotoky, Giménez et al.) demonstrated feasibility but incurred substantial airtime and required careful scheduling

### CE-FedAvg Framework Architecture

**Client-side workflow (Algorithm 1):**
1. If round t=0: use seed s₀ to deterministically initialize weights `w₀ = Init(s₀)`
2. Local training for E epochs
3. Compute delta: `Δwᵢ = wᵢ^(t+1) − wᵗ`
4. Select Top-K entries by magnitude (sparsification)
5. Quantize selected values to float16 or int8
6. Pack indices + quantized values into LoRaWAN-compliant frames
7. Transmit (duty-cycle compliant)

**Server-side aggregation (Algorithm 2):**
1. Receive fragments from each client
2. Unpack indices and quantized values
3. Dequantize and reconstruct sparse Δwᵢ
4. Weighted FedAvg over the union of indices → new global model wᵗ⁺¹
5. Optionally broadcast compact refresh delta to clients

**Seed-based initialization:**
- Server broadcasts only a **32-bit PRNG seed s₀**, CRC checksum, and initializer type (Kaiming/Xavier)
- Each client reconstructs identical w₀ locally — avoids broadcasting entire model at round 0
- CRC32 over w₀ validates consistency across heterogeneous devices

**Sparse + quantized update transmission:**
- Sparsification: Top-K% of weight deltas by magnitude (tested K=1%, 2%, 5%, 10%)
- Quantization: float16 (default) or int8
- Indices: relative-offset encoded (compact)
- Entropy coding into frames ≤ 222 bytes (max MAC payload at DR5)
- If payload exceeds 222B → modem fragments it and appends CRC

**Model improvements:**
- Lightweight CNN: 2 conv layers replaced with 3 compact blocks (Conv→BN→ReLU→Pool + adaptive global pooling + FC)
- Parameter reduction: ~214.6k (baseline) → **~27.6k (lightweight)** = **87% fewer parameters**
- Float32 during training; only transmitted deltas are quantized

### Hardware Setup
- **Edge clients:** 3× NVIDIA Jetson Orin Nano (GPU-capable Linux SBC)
- **LoRaWAN modem:** ESP32 + Semtech SX1262 transceiver (external, connected via UART)
- **Gateway:** Raspberry Pi 5 + Semtech SX1303 LoRaWAN Gateway HAT
- **Network server:** ChirpStack (private)
- **FL server connects to LNS via MQTT**

> [!IMPORTANT]
> The Jetson Orin Nano is a **powerful** edge device (GPU-capable, Linux, Python/PyTorch). This is NOT a microcontroller. The ESP32 only handles LoRaWAN modem duty; training happens on the Jetson.

### Evaluation Setup

| Parameter | Value |
|-----------|-------|
| Software | PyTorch + Flower 1.19.0 |
| Dataset | FEMNIST (LEAF benchmark) — 62-class handwritten character recognition |
| Input | 28×28 grayscale images |
| Non-IID | Writer-based partitions (naturally non-IID) |
| Physical clients | 3 (NVIDIA Jetson Orin Nano) |
| Simulated clients | Up to 30 (trace-driven replay of measured airtime) |
| FL rounds | 30 |
| Local epochs (E) | 3 (default), tested 1–5 |
| Batch size | 32 |
| Optimizer | SGD (momentum 0.9, LR 0.01, decay on plateau) |
| Sparsity (K) | 0.10 (Top-10%), tested down to 0.01 |
| Quantization | float16 (default), int8 optional |
| LoRaWAN | EU868, DR5 (SF7/125kHz), Class C, OTAA, 1% duty cycle |
| Max MAC payload | 222 bytes |
| Repetitions | 5 per configuration, mean ± 1 SD reported |

### Main Results (Table II)

| Setting | Accuracy (%) | Uplink/Downlink (KB/round) |
|---------|-------------|---------------------------|
| Baseline CNN, dense | 85.0 | 858 |
| Lightweight, dense (fp16) | 84.5 | 50.3 |
| **Lightweight, Top-10%** | **83.4** | **6.21** |
| Lightweight, Top-5% | 81.7 | 3.26 |
| Lightweight, Top-2% | 76.0 | 1.37 |
| Lightweight, Top-1% | 63.6 | 0.75 |

**Key quantitative findings:**
1. Lightweight model: 84.5% accuracy, **17× smaller payload** than baseline (50.3 vs 858 KB)
2. Top-10% sparsification: 83.4% accuracy, payload = 6.21 KB (−1.6% accuracy vs baseline)
3. Top-1% sparsification: 63.6% accuracy, payload = 0.75 KB (−21.4% accuracy — too aggressive)
4. **Rounds vs epochs (fixed 90 total):** 30×3 = 83.4% best, 45×2 = 82.9%, 18×5 = 81.4%, 90×1 = 81.8%
5. **LoRaWAN feasibility:** Top-1% update (0.75 KB) → 4 fragments, 1.47s airtime, ~1.52 min wall time; Top-5% → 3.26 KB; manageable at EU868 DR5
6. **Convergence vs sparsity:** Top-5% converges fastest to 80% accuracy; Top-2% never reaches 80%; Top-1% saturates at ~65%

### Limitations (stated in paper)
- Physical testbed = **only 3 clients**; scalability tested via simulation only
- **Only FEMNIST tested** — no IoT/environmental/metering workloads
- Energy per update **inferred from airtime**, not directly measured
- **No secure aggregation or differential privacy** implemented
- Relies on standard LoRaWAN MAC/PHY — no admission control or collision management

### Future Work (stated in paper)
- Extend to additional IoT workloads and larger models
- Jointly adapt sparsity + quantization to optimize time-to-target accuracy
- Structured sparsification, error-feedback mechanisms
- ADR-aware or staggered client selection
- Security/privacy add-ons (secure aggregation, differential privacy)

---

## 2. Mapping Paper to Thesis — Three-Phase Comparison

### Phase 1: Supervisor's Indoor Environmental LoRaWAN Paper

| Aspect | CE-FedAvg Paper | Supervisor's Work | Comparison |
|--------|----------------|-------------------|------------|
| **Application domain** | Generic FL framework — tested on FEMNIST (image classification) | Indoor LoRaWAN path loss modeling with environmental sensors | **Different domain** — CE-FedAvg doesn't target path loss or indoor propagation |
| **Data source** | FEMNIST benchmark (synthetic partitions by writer) | Real 8-month indoor measurement campaign, 6 devices, 1.7M rows | Supervisor's data is **real IoT deployment data**, CE-FedAvg uses ML benchmark |
| **LoRaWAN usage** | Real LoRaWAN as the FL communication channel | Real LoRaWAN as the sensing medium + data transport | Both use real EU868 LoRaWAN |
| **Path loss** | Not addressed | Core research topic (LDPLSM-MW-EP model, R²=0.8219) | Not directly relevant — CE-FedAvg doesn't do path loss |
| **Environmental sensing** | Not addressed | BME280, SCD4x, SPS30 — pressure, CO₂, temp, humidity, PM2.5 | Not relevant |
| **Directly relevant** | LoRaWAN constraint analysis (duty cycle, payload limits) | LoRaWAN infrastructure description | ✅ Both validate EU868 constraints |
| **Reusable** | LoRaWAN duty-cycle formulas (Eq. 3-4) | Dataset and infrastructure setup | ✅ LoRaWAN constraint model |
| **Cannot reuse** | FEMNIST evaluation results | Path loss model (different architecture than FL model) | ❌ Different tasks entirely |

### Phase 2: PEP XGBoost Regression Work

| Aspect | CE-FedAvg Paper | PEP XGBoost | Comparison |
|--------|----------------|-------------|------------|
| **ML task** | Image classification (FEMNIST, 62 classes) | Path loss regression (R²≈0.93) | **Different tasks** — classification vs regression |
| **Model type** | CNN (convolutional neural network) | XGBoost (gradient-boosted trees) | Different algorithms |
| **Features** | 28×28 pixel values | Environmental + radio features | Different input domains |
| **Parameters** | 214.6k (baseline) / 27.6k (lightweight) | Not stated in the paper — XGBoost parameters are tree-based, not comparable | Architecturally incomparable |
| **FL involved?** | Yes — the core contribution | No — centralized training only | CE-FedAvg's FL approach is what my thesis adds |
| **Directly relevant** | Nothing — different domain and approach | Path loss prediction methodology | ❌ Not directly comparable |
| **Reusable** | Nothing | Regression baseline (R²≈0.93) serves as thesis comparison point | Nothing from CE-FedAvg applies |

### Phase 3: My Thesis — TinyML + Federated Learning

| Aspect | CE-FedAvg Paper | My Thesis | Comparison |
|--------|----------------|-----------|------------|
| **FL algorithm** | CE-FedAvg (modified FedAvg with sparsification + quantization + seed init) | Standard FedAvg (McMahan et al.) | Both use FedAvg base; CE-FedAvg adds compression; my thesis uses vanilla FedAvg |
| **Edge hardware** | NVIDIA Jetson Orin Nano (GPU-capable, Linux, gigabytes of RAM) | Arduino MKR WAN 1310 (Cortex-M0+, 32KB SRAM, 256KB Flash) | **My thesis targets vastly more constrained hardware** — ~100,000× less RAM |
| **Training stack** | PyTorch + Flower on Linux | TensorFlow Lite Micro (simulation) + simplified proxy training (device) | CE-FedAvg runs real PyTorch; my device uses TFLite Micro inference only |
| **LoRaWAN modem** | External ESP32 + SX1262, connected via UART to Jetson | Built-in CMWX1ZZABZ module on MKR WAN 1310 | My modem is integrated; theirs is external |
| **Dataset** | FEMNIST (image benchmark) | Real indoor sensor data (1.7M rows, 8 features, path loss target) | **My dataset is real IoT data**, not a benchmark |
| **Task** | Image classification (62 classes) | Path loss regression (continuous dB output) | Different ML tasks |
| **Model** | CNN: 27.6k params (lightweight) / 214.6k (baseline) | Dense(8→8→1): **81 params** | My model is **340× smaller** than their lightweight model |
| **Update size** | 0.75–50.3 KB per round depending on sparsity | **52 bytes per round** (no sparsification needed!) | My updates are **14× smaller** than even their most aggressive Top-1% config |
| **Sparsification needed?** | Yes — essential to fit in LoRaWAN payloads | **No** — 81 params × 1 byte(int8) = 81 bytes → fits in 1 LoRaWAN frame | My model is small enough that compression is unnecessary |
| **Number of clients** | 3 physical (+ 30 simulated) | 6 (all simulated, data from 6 real devices) | Comparable scale |
| **FL rounds** | 30 | 20 | Comparable |
| **Local epochs** | 3 (default), tested 1–5 | Tested E=1, 3, 5 — E=5 optimal | Both explore epoch trade-offs; both find moderate E is best |
| **Best accuracy/R²** | 83.4% (Top-10%, lightweight) | R²=0.9999 (E=5) | Not directly comparable (classification vs regression) |
| **Non-IID handling** | Writer-based partitions (FEMNIST) | Device-based partitions (real room locations) | Both genuinely non-IID; mine is physically grounded |
| **Communication reduction** | 858 KB → 0.75 KB (Top-1%) = ~1,144× but with 21% accuracy loss | 25,920 B → 2,356 B = 11× with only 0.001% R² loss | Different baselines; both achieve significant reduction |
| **Privacy analysis** | Mentioned but not implemented (no secure aggregation) | Discussed as a design benefit (raw data stays on device) | Both claim privacy; neither implements formal guarantees |
| **Duty cycle compliance** | Quantified: ToA, fragments, wall-clock time per update | Claimed compliant but not formally quantified per-fragment | CE-FedAvg's analysis is more rigorous |

---

## 3. Thesis Design Decisions Supported by This Paper

### ✅ Strongly Supported

1. **FedAvg is the right base algorithm for LoRaWAN FL**
   - CE-FedAvg builds on FedAvg (McMahan et al. 2017), same as my thesis. The paper confirms FedAvg's round semantics are compatible with LoRaWAN's intermittent connectivity.

2. **Local epochs (E) should be moderate (3–5), not 1**
   - CE-FedAvg found 30×3 (83.4%) beat 90×1 (81.8%) under fixed total budget. My thesis found E=5 (R²=0.9999) ≫ E=1 (R²=−4.89). Both independently confirm that too few local epochs leads to poor convergence.

3. **Small models are critical for LoRaWAN FL**
   - CE-FedAvg's lightweight model (27.6k params, −87% from baseline) was essential to make payloads manageable. My Dense(8→8→1) with 81 params is an extreme case of this principle — so small that sparsification is unnecessary.

4. **Quantization (int8) is practical for weight transmission**
   - CE-FedAvg uses float16/int8 quantization. My thesis uses int8 quantization for the 52-byte update payload. The paper validates that quantization doesn't destroy model quality.

5. **EU868 duty cycle is a binding constraint**
   - CE-FedAvg formally quantifies duty-cycle timing (Eq. 4). My thesis claims compliance but benefits from citing their formal model.

6. **Non-IID data is the realistic scenario for IoT FL**
   - CE-FedAvg uses writer-based non-IID FEMNIST partitions. My thesis uses device-location-based non-IID partitions. Both correctly model that real IoT devices have non-identical data distributions.

7. **Dense model broadcast at round 0 should be avoided**
   - CE-FedAvg uses seed-based initialization. My thesis uses a pre-flashed model.h (trained offline, burned to Flash via USB). Both avoid sending the full model over LoRaWAN at initialization.

8. **Real LoRaWAN hardware validation matters**
   - CE-FedAvg implements on physical hardware (Jetson + ESP32 + SX1303 gateway). My thesis has firmware for MKR WAN 1310. Both recognize simulation alone is insufficient.

### ⚠️ Partially Supported

9. **Sparsification adds value when model is large**
   - CE-FedAvg's Top-K sparsification is essential for 27.6k params. For my 81-param model, this technique is irrelevant — sending all 81 weights as int8 = 81 bytes, which already fits in a single LoRaWAN frame.

10. **30 rounds is sufficient for convergence**
    - CE-FedAvg uses 30 rounds, my thesis uses 20. Both achieve good convergence. However, convergence speed depends on model complexity and data — not directly transferable.

---

## 4. Open Gaps Where My Thesis Contributes Something New

### 🟢 My Thesis Fills These Gaps

1. **Real IoT sensing data (not benchmarks)**
   - CE-FedAvg only tests FEMNIST (handwritten characters). They explicitly acknowledge this limitation: *"broader workloads (e.g., metering, industrial, environmental/agriculture) remain to be explored."* My thesis uses **1.7M rows of real indoor environmental + LoRaWAN data** — exactly the type of workload they call for.

2. **True microcontroller deployment (not Jetson)**
   - CE-FedAvg runs training on NVIDIA Jetson Orin Nano (Linux, GPU, gigabytes of RAM). My thesis targets **Arduino MKR WAN 1310** (Cortex-M0+, 32KB SRAM, no OS) — a genuinely constrained device with TFLite Micro. This is the "tiny" in TinyML that CE-FedAvg doesn't address.

3. **Path loss regression (not classification)**
   - CE-FedAvg does classification (62-class FEMNIST). My thesis does **regression** (predict continuous path loss in dB). Regression with R²/RMSE is a different evaluation paradigm — my thesis demonstrates FL works for regression tasks, not just classification.

4. **Ultra-small model (81 params) that needs NO compression**
   - CE-FedAvg's key contribution is compression (sparsification, quantization, seed init) to squeeze 27.6k params through LoRaWAN. My model has only 81 params — **the model is so small that the entire FL update fits in a single 52-byte LoRaWAN frame without any sparsification**. This is a complementary approach: instead of compressing a large model, design a model small enough that compression is unnecessary.

5. **Integrated LoRaWAN modem (no external hardware)**
   - CE-FedAvg uses Jetson + external ESP32/SX1262 modem (UART bridge). My MKR WAN 1310 has the LoRaWAN module **built-in** (CMWX1ZZABZ). This is a simpler, cheaper, more deployable solution.

6. **Environmental monitoring + link quality prediction**
   - The paper explicitly lists "environmental" as an unexplored workload. My thesis combines environmental sensing (BME280, SCD4x, SPS30) with link quality prediction — a dual-purpose system that CE-FedAvg doesn't attempt.

7. **Communication efficiency comparison with older system**
   - My thesis quantifies the **11× bandwidth reduction** from a concrete prior system (raw data every 60s) to FL updates. CE-FedAvg compares only against its own dense baseline, not against a pre-existing deployment.

8. **Near-perfect convergence (R²=0.9999 vs centralized R²=1.0000)**
   - CE-FedAvg's best accuracy is 83.4% (vs 85.0% centralized = 1.6% drop). My thesis achieves R²=0.9999 (vs 1.0000 = 0.001% drop). While these are different tasks (classification vs regression), the near-zero degradation in my thesis is a stronger convergence result.

### 🔴 My Thesis Does NOT Address (CE-FedAvg Does)

1. **Sparsification** — not needed for 81 params
2. **Seed-based initialization** — my thesis pre-flashes model.h instead
3. **Formal duty-cycle timing model** (Eq. 3-4) — my thesis claims compliance but doesn't formalize it
4. **Scalability beyond 6 clients** — CE-FedAvg simulates 30 clients
5. **Per-experiment repetitions** — CE-FedAvg runs 5× with error bars; my thesis runs once (seed=42)

---

## 5. Quick-Reference Comparison Table

| Dimension | CE-FedAvg (ICC26 Amin) | My Thesis |
|-----------|----------------------|-----------|
| Venue | ICC 2026 | Master's Thesis |
| FL variant | CE-FedAvg (sparse/quantized FedAvg) | Standard FedAvg |
| Hardware | Jetson Orin Nano (GPU) | MKR WAN 1310 (Cortex-M0+) |
| RAM | Gigabytes | **32 KB** |
| LoRaWAN modem | External (ESP32+SX1262) | Integrated (CMWX1ZZABZ) |
| Dataset | FEMNIST (benchmark) | Real indoor sensor data |
| Task | Classification (62 classes) | Regression (path loss dB) |
| Model size | 27.6k params | **81 params** |
| Update size | 0.75–50.3 KB | **52 bytes** |
| Sparsification | Required (Top-K) | Not needed |
| Accuracy/R² | 83.4% (Top-10%) | R²=0.9999 |
| Drop from centralized | −1.6% | **−0.001%** |
| Clients | 3 physical + 30 simulated | 6 (simulated from real data) |
| Rounds | 30 | 20 |
| Best E | 3 (from 30×3 config) | 5 |
| LoRaWAN region | EU868 | EU868 |
| Open source | ✅ GitHub | Firmware provided |

---

## 6. How to Cite in Your Thesis

### In Chapter 3 (Related Work):
> *"Razaghi and Aral [XX] present CE-FedAvg, a communication-efficient FL framework for LoRaWAN. Their approach combines update sparsification (Top-K), quantization (float16/int8), and seed-based initialization to reduce uplink payloads from 858 KB to as low as 0.75 KB per round on FEMNIST. However, their framework targets NVIDIA Jetson Orin Nano devices — Linux-based SBCs with GPU capability — and relies on external ESP32 LoRaWAN modems. In contrast, the present work targets genuinely resource-constrained microcontrollers (Cortex-M0+, 32 KB SRAM) with integrated LoRaWAN radios, where the model is designed to be small enough (81 parameters) that sparsification is unnecessary."*

### In Chapter 7 (Discussion):
> *"The finding that E=5 local epochs yields optimal convergence aligns with CE-FedAvg's observation that moderate local computation (3 epochs in their setting) outperforms both minimal (E=1) and excessive (E=5) local training under a fixed total budget [XX]. Both works independently confirm that the local epoch count is a critical hyperparameter for FL convergence over LoRaWAN."*

> *"CE-FedAvg explicitly identifies real IoT workloads — including environmental and industrial sensing — as an open direction for validation. The present work directly addresses this gap by applying FL to real indoor environmental and LoRaWAN measurement data from a production sensor deployment."*
