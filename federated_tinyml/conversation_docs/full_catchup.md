# 🧠 Full Catch-Up: Your Federated TinyML Thesis

*Last worked on: April 14, 2026 — You're picking up after ~3 weeks away*

---

## Part 1: What Is This Project? (The 30-Second Version)

You're writing a **Master's thesis** titled:

> **"Decentralized Edge Intelligence for LoRaWAN: Federated Learning for Environment-Driven Path Loss and Link Quality Modeling"**

**In plain English:** You have 6 Arduino sensor nodes in an office building. In your old project, they were "dumb" — they just sent raw data to a server. In your new thesis, each device has a **tiny brain** (a neural network) that learns locally and shares only tiny weight updates (52 bytes) instead of raw data. A server combines these updates from all 6 devices using **Federated Averaging (FedAvg)**.

---

## Part 2: The Big Picture — Old vs New

```
OLD PROJECT ("train once, deploy, forget"):
  Sensors → Raw data → LoRaWAN → Server → Train XGBoost centrally → R² ≈ 0.93
  Problems: 25.9 KB/day bandwidth, no privacy, can't adapt

NEW THESIS ("learn everywhere, share smartly"):
  Sensors → On-device ML → Only 52-byte weight updates → Server aggregates → Send improved model back
  Benefits: 2.4 KB/day bandwidth (11× reduction), privacy preserved, continuous adaptation
```

---

## Part 3: Your Three Main Code Files

```
┌─────────────────────────────────────────────────────────────────┐
│                         YOUR PC                                  │
│                                                                  │
│   train_model.py ──→ model.h ──→ flash to Arduino via USB       │
│   (trains initial               (C byte array of the            │
│    model from CSV)               neural network)                │
│                                                                  │
│   fl_server.py ←──── TTN webhooks ←──── LoRaWAN uplinks         │
│   (Flask server,     (receives 52-byte   (from all 6 nodes)     │
│    does FedAvg)       weight updates)                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                    LoRa radio (868 MHz)
                              │
┌─────────────────────────────────────────────────────────────────┐
│  6× Arduino MKR WAN 1310 running FederatedTinyML.ino            │
│                                                                  │
│  Each device every cycle:                                        │
│   1. Reads sensors (pressure, CO₂, temp, humidity, PM2.5)       │
│   2. Normalizes → runs TinyML inference → predicts link quality │
│   3. Buffers samples, trains locally every 24h                   │
│   4. Sends 52-byte weight update via LoRaWAN                    │
│   5. Receives improved global model as downlink                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part 4: What Happened in Our Last Session (Apr 5–14)

### The Big Discovery: 5 Features → 8 Features

We discovered that the original 5 environmental features (pressure, CO₂, temp, humidity, PM2.5) had **near-zero correlation** with path loss (R² ≈ 0.07). The model was useless!

The fix: add **3 radio features** — RSSI, SNR, and Spreading Factor. These have very high correlation with path loss (because `exp_pl = TX_power - RSSI`). With 8 features, the model achieves **R² = 0.9999**.

### What We Built / Updated

1. **`fl_simulation.py`** (903 lines) — Complete FL simulation script:
   - Loads 1.7M real rows, cleans data, trains centralized baseline
   - Runs FedAvg with 6 clients × 20 rounds × E={1,3,5}
   - Generates 8 publication-quality figures
   - Architecture: **Dense(8→8→1)** = 81 parameters

2. **Simulation ran successfully** — produced `fl_simulation_results.json` with these key results:

   | Configuration | R² | RMSE (dB) | Verdict |
   |---|---|---|---|
   | **Centralized baseline** | 1.0000 | 0.0007 | Perfect (because RSSI is an input) |
   | **Federated E=1** | −4.89 | 54.87 | ❌ Diverged |
   | **Federated E=3** | 0.9606 | 4.49 | ⚠️ Partial convergence |
   | **Federated E=5** | 0.9999 | 0.083 | ✅ Near-perfect, only 0.001% drop |

3. **8 figures generated** in `thesis_figures/`:
   - fig1: Data distribution per client
   - fig2: Centralized training curves
   - fig3: Predicted vs actual (centralized)
   - fig4: FL convergence R²/RMSE over rounds
   - fig5: Three-way comparison bar chart
   - fig6: Per-client R² (non-IID analysis)
   - fig7: Communication efficiency
   - fig8: FL predicted vs actual

4. **Updated firmware** (`.ino`) and `train_model.py` to 8 features

5. **Built the thesis LaTeX structure** — originally as `thesis_overleaf.tex`, now split into modular chapter files

### What Was Broken When We Stopped
- The simulation had hung for 48 hours (fixed by increasing batch_size to 2048)
- Eventually ran successfully and produced results

---

## Part 5: Current State of the Thesis LaTeX

Your thesis is now structured as **modular chapter files**:

```
thesis/
├── main.tex                    ← Master file (compiles everything)
├── references.bib              ← Bibliography
├── thesis_overleaf.tex         ← OLD monolithic version (backup)
└── chapters/
    ├── titlepage.tex           ← Title page
    ├── declaration.tex         ← Declaration of originality
    ├── abstract.tex            ← Abstract (has [XX] placeholders)
    ├── ch1_introduction.tex    ← Introduction (has [XX] in prior results table)
    ├── ch2_background.tex      ← Background & theory
    ├── ch3_related_work.tex    ← Related work (has [XX] in comparison table)
    ├── ch4_system_design.tex   ← System design & architecture
    ├── ch5_implementation.tex  ← Implementation (has [XX] in data tables)
    ├── ch6_evaluation.tex      ← Evaluation & results (MOSTLY COMMENTS - needs writing)
    ├── ch7_discussion.tex      ← Discussion (ALL COMMENTS - needs writing)
    ├── ch8_conclusion.tex      ← Conclusion (has [XX] placeholders)
    └── appendix.tex            ← Source code excerpts
```

### Chapter Status at a Glance

| Chapter | Tables | Prose | Figures | Status |
|---------|--------|-------|---------|--------|
| Abstract | — | ⚠️ Has `[XX]` | — | Needs numbers filled |
| Ch1 Introduction | 1 (has `[XX]`) | ✅ Written | — | Needs prior project numbers |
| Ch2 Background | — | ✅ Written | — | Done |
| Ch3 Related Work | 1 (has `[XX]`) | ⚠️ Partial | — | Needs our R²/RMSE + more prose |
| Ch4 System Design | — | ✅ Written | — | Done |
| Ch5 Implementation | 2 (have `[XX]`) | ❌ Comments only | — | Needs prose + data filled |
| **Ch6 Evaluation** | 3 (have `[XX]`) | ❌ Comments only | 0/8 referenced | **Most critical — needs full write** |
| **Ch7 Discussion** | — | ❌ Comments only | — | **Needs full write** |
| Ch8 Conclusion | — | ❌ Comments only | — | Needs prose + numbers |
| Appendix | — | ✅ Code excerpts | — | Done |

---

## Part 6: Known Issues to Fix

### 🔴 Critical: Architecture Inconsistency
The chapter files say **Dense(5→8→1)** with **57 parameters**, but the actual simulation uses **Dense(8→8→1)** with **81 parameters** (because we added RSSI, SNR, SF as features). This mismatch appears in:
- ch3 comparison table (line 87)
- ch4 model design sections
- ch5 implementation sections  
- ch6 tables (line 85: says "57" params)
- ch7 limitations (line 105: says "Dense 5→8→1")

### 🟡 Minor Inconsistencies
- Communication cost: ch6 says 2,404 B/day, but JSON says 2,356 B/day
- `main.tex` has placeholder metadata: `[Supervisor Name]`, `[University Name]`, `[Matriculation Number]`, `[Month Year]`

### 🟡 Hardware Values Unknown
- Flash/SRAM usage (needs compiling firmware on real hardware)
- Tensor arena actual usage
- Inference latency measurement

---

## Part 7: What Needs to Happen Next

### Priority 1: Fix + Fill (mechanical, I can do this)
- [ ] Fix 5→8 architecture references everywhere
- [ ] Fill 22 `[XX]` placeholders from `fl_simulation_results.json`
- [ ] Add `\includegraphics` for all 8 figures in appropriate chapters

### Priority 2: Write Prose (needs thinking, I can draft)
- [ ] **Ch6 Evaluation** — write §6.1 methodology, §6.2 centralized results, §6.3 FL results, §6.5 comm efficiency analysis
- [ ] **Ch7 Discussion** — write §7.1 answering RQs, §7.2 trade-off analysis, §7.3 env influence, §7.4 Torres comparison, §7.5 prior project comparison
- [ ] **Ch5 Implementation** — flesh out all subsections from comment outlines
- [ ] **Ch8 Conclusion** — write §8.1 summary, fill §8.2 key findings

### Priority 3: Your Input Needed
- [ ] Your supervisor name, university, department, matrikelnr, submission date
- [ ] Device locations (which room is ED0, ED1, etc.?)
- [ ] Prior project MLR/RF RMSE values (for Ch1 table)
- [ ] Hardware measurements (Flash/SRAM — requires compiling the firmware)

---

## Part 8: Key Numbers Reference Card

Keep these handy — they come from `fl_simulation_results.json`:

| Metric | Value |
|--------|-------|
| Total dataset rows (after cleaning) | 1,714,379 |
| Model architecture | Dense(8→8→1) |
| Total parameters | 81 |
| Centralized R² | 1.0000 |
| Centralized RMSE | 0.0007 dB |
| Centralized MAE | 0.00007 dB |
| FL (E=5) R² | 0.9999 |
| FL (E=5) RMSE | 0.083 dB |
| FL (E=5) MAE | 0.002 dB |
| R² drop (centralized→federated) | 0.001% |
| Old bandwidth/node/day | 25,920 B |
| New bandwidth/node/day | 2,356 B |
| Bandwidth reduction | 11× |
| Our FL update size | 52 bytes (1 LoRaWAN message) |
| Torres Sanchez model size | 1,428 bytes (7+ messages needed) |
| ED0 samples | 228,627 |
| ED1 samples | 226,727 |
| ED2 samples | 229,404 |
| ED3 samples | 226,237 |
| ED4 samples | 225,399 |
| ED5 samples | 235,109 |
