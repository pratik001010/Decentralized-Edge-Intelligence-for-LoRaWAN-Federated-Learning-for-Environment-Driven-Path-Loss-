# Data Budget & Communication Efficiency Manual

**Thesis Title**: *Decentralized Edge Intelligence for LoRaWAN: Federated Learning for Environment-Driven Path Loss and Link Quality Modeling*  
**Author**: Pratik Khadka  
**Institution**: University of Siegen  
**File Target**: `thesis guideline/data_budget_analysis_manual.md`  

---

## 1. Executive Summary & Overview

This manual provides an exhaustive, step-by-step mathematical breakdown of the **data consumption budget** and **communication efficiency** for our decentralized LoRaWAN system. 

It compares the naive **Centralized Raw Telemetry Streaming Scheme** against our compact **Federated Learning (FL) Update Scheme** over a single communication round window ($7.1$ active sensing days, corresponding to $1/50\text{th}$ of a 365-calendar-day deployment timeline).

### Key Findings:
- **Centralized Raw Telemetry Data (1 Round / 7.1 Days):** **$123,300 \text{ Bytes}$ ($123.30 \text{ KB}$)** per node.
- **Federated Learning Data Budget (1 Round / 7.1 Days):** **$19,332 \text{ Bytes}$ ($19.33 \text{ KB}$)** per node.
- **Achieved Communication Reduction Ratio:** **$6.37\times$ Bandwidth Savings per Round** ($\approx 84.3\%$ overall traffic reduction).
- **Pure Model Weight Exchange Payload:** Exchanging 89 parameters takes only **$204 \text{ Bytes}$** per round ($102 \text{ Bytes}$ uplink $+ 102 \text{ Bytes}$ downlink), yielding a **$604.4\times$ reduction** over raw data streaming.

---

## 2. Federated Learning Communication Budget (1 Round = 7.1 Days)

In a 50-round annual schedule, each round spans **$7.1 \text{ active sensing days}$**. At a 10-minute packet interval, each node experiences:

$$\text{Packet Slots per Round} = 96 \text{ slots/day} \times 7.1 \text{ days} = 681.6 \approx \mathbf{682 \text{ Slots/Round}}$$

### 2.1 Step-by-Step FL Protocol Traffic Breakdown

1. **Step 1: Initial Model Seeding (Deployment Startup)**  
   The node receives its initial 4-byte seed configuration over the air:
   $$\text{Data}_{\text{Seed}} = 4 \text{ Bytes Payload} + 13 \text{ Bytes MAC Header} = \mathbf{17 \text{ Bytes}}$$

2. **Step 2a: Telemetry Uplink Heartbeats (over 7.1 Days)**  
   Each 10-minute slot, the node transmits a 13-byte dummy telemetry heartbeat packet:
   $$\text{Data}_{\text{Heartbeat}} = 682 \text{ Packets} \times 13 \text{ Bytes MAC Header} = \mathbf{8,866 \text{ Bytes}}$$

3. **Step 2b: Downlink Link Quality Feedback (RSSI/SNR)**  
   The gateway responds in the RX1 receive window with 2-byte RSSI/SNR feedback:
   $$\text{Data}_{\text{Feedback}} = 682 \text{ Packets} \times (13 \text{ Bytes Header} + 2 \text{ Bytes Payload}) = 682 \times 15 = \mathbf{10,230 \text{ Bytes}}$$

4. **Step 3: Local Model Weight Uplink (at Day 7.1)**  
   The node packs its 89 `int8` local model weights into 1 LoRaWAN packet:
   $$\text{Data}_{\text{Uplink\_Weights}} = 89 \text{ Bytes Payload} + 13 \text{ Bytes MAC Header} = \mathbf{102 \text{ Bytes}}$$

5. **Step 4: Global Model Weight Downlink (at Day 7.1)**  
   The server broadcasts the 89 `int8` aggregated global model weights back to the node:
   $$\text{Data}_{\text{Downlink\_Weights}} = 89 \text{ Bytes Payload} + 13 \text{ Bytes MAC Header} = \mathbf{102 \text{ Bytes}}$$

6. **Step 5: Global Metric Calculation Report ($R^2$, RMSE)**  
   The server transmits final scalar performance evaluation indicators ($R^2, \text{RMSE}$):
   $$\text{Data}_{\text{Report}} = 2 \text{ Bytes Payload} + 13 \text{ Bytes MAC Header} = \mathbf{15 \text{ Bytes}}$$

---

### 2.2 Per-Round FL Data Budget Summary

$$\text{Data}_{\text{FL Total}} = 17 + 8,866 + 10,230 + 102 + 102 + 15 = \mathbf{19,332 \text{ Bytes}} \quad (\mathbf{19.33 \text{ KB}})$$

$$\text{Data}_{\text{FL Pure Model Exchange}} = 102 + 102 = \mathbf{204 \text{ Bytes}} \quad (\mathbf{0.20 \text{ KB}})$$

---

## 3. Centralized Raw Telemetry Streaming Baseline (Actualize)

In a conventional centralized deployment, nodes stream uncompressed sensor measurements continuously to the cloud.

### 3.1 Raw Telemetry Transmission Volume
- **1-Minute Baseline Sampling Rate:** $24 \text{ hours} \times 60 \text{ mins} = 1,440 \text{ slots/day}$
- **Active Sensing Subsampling Ratio ($0.66$):** $1,440 \times 0.66 = 964.8 \text{ active slots/day}$
- **Total Raw Packets per 7.1-Day Round:**
  $$\text{Total Centralized Packets} = 964.8 \text{ slots/day} \times 7.1 \text{ days} = \mathbf{6,850 \text{ Packets}}$$

### 3.2 Centralized Data Volume Calculation
With a standard $18\text{-Byte}$ PHYPayload per raw telemetry packet:

$$\text{Data}_{\text{Centralized}} = 6,850 \text{ Packets} \times 18 \text{ Bytes/Packet} = \mathbf{123,300 \text{ Bytes}} \quad (\mathbf{123.30 \text{ KB}})$$

---

## 4. Side-by-Side Data Budget Comparison Table

| Communication Component / Metric | Centralized Raw Streaming | Federated Learning Scheme | Budget Savings / Impact |
|:---|:---:|:---:|:---:|
| **Initial Seeding (Step 1)** | $0 \text{ Bytes}$ | **$17 \text{ Bytes}$** | $17 \text{ Bytes}$ one-time startup |
| **Telemetry Uplink Heartbeats (Step 2a)** | $123,300 \text{ Bytes}$ | **$8,866 \text{ Bytes}$** | $92.8\%$ uplink volume reduction |
| **Downlink Link Feedback (Step 2b)** | $0 \text{ Bytes}$ | **$10,230 \text{ Bytes}$** | Provides real-time RSSI/SNR feedback |
| **Local Weight Uplink (Step 3)** | $0 \text{ Bytes}$ | **$102 \text{ Bytes}$** | Fits in **1 single LoRaWAN packet** |
| **Global Weight Downlink (Step 4)** | $0 \text{ Bytes}$ | **$102 \text{ Bytes}$** | Fits in **1 single LoRaWAN packet** |
| **Metric Report Feedback (Step 5)** | $0 \text{ Bytes}$ | **$15 \text{ Bytes}$** | 2-byte evaluation scalars |
| **TOTAL DATA PER NODE (1 Round / 7.1 Days)** | **$123,300 \text{ Bytes}$ ($123.30\text{ KB}$)** | **$19,332 \text{ Bytes}$ ($19.33\text{ KB}$)** | **Saves $103.97\text{ KB}$ per node per round** |
| **PURE MODEL WEIGHT EXCHANGE ONLY** | N/A | **$204 \text{ Bytes}$ ($0.20\text{ KB}$)** | **$604.4\times$ reduction over raw data** |

---

## 5. Communication Reduction Ratio Analysis

### 5.1 Overall Network Traffic Reduction Ratio
Comparing total traffic per node per round ($123,300 \text{ Bytes}$ vs. $19,332 \text{ Bytes}$):

$$\text{Reduction Ratio} = \frac{\text{Data}_{\text{Centralized}}}{\text{Data}_{\text{FL Total}}} = \frac{123,300 \text{ Bytes}}{19,332 \text{ Bytes}} = \mathbf{6.3779\times \approx \mathbf{6.37\times \text{ Reduction}}}$$

$$\text{Percentage Savings} = \left( 1 - \frac{19,332}{123,300} \right) \times 100\% = \mathbf{84.32\% \text{ Traffic Reduction}}$$

### 5.2 Pure Algorithmic Model Exchange Efficiency
Comparing raw telemetry streaming ($123,300 \text{ Bytes}$) directly against the weight parameters exchanged during FL ($204 \text{ Bytes}$):

$$\text{Pure Model Reduction Ratio} = \frac{123,300 \text{ Bytes}}{204 \text{ Bytes}} = \mathbf{604.41\times \text{ Reduction}}$$

---

## 6. System-Wide Multi-Node Scaling (6 Devices & 50 Rounds / 1 Year)

| Deployment Scope | Centralized Raw Upload | Federated Learning System | Total Data Saved |
|:---|:---:|:---:|:---:|
| **1 Node / 1 Round (7.1 Days)** | $123.30 \text{ KB}$ | **$19.33 \text{ KB}$** | **$103.97 \text{ KB}$** |
| **6 Nodes / 1 Round (7.1 Days)** | $739.80 \text{ KB}$ | **$116.00 \text{ KB}$** | **$623.80 \text{ KB}$** |
| **1 Node / 50 Rounds (1 Year)** | $6.165 \text{ MB}$ | **$0.967 \text{ MB}$** | **$5.198 \text{ MB}$** |
| **6 Nodes / 50 Rounds (1 Year)** | **$36.99 \text{ MB}$** | **$5.80 \text{ MB}$** | **$31.19 \text{ MB}$** |

---

## 7. Thesis Defense Summary & Justification

> *"By replacing continuous uncompressed raw telemetry streaming ($123.30\text{ KB}$ per node per round) with a compact Federated Learning scheme ($19.33\text{ KB}$ per node per round), our deployment achieves a **$6.37\times$ overall reduction in bandwidth consumption** while preserving **100% data privacy**. Furthermore, the algorithmic parameter exchange itself requires only **204 Bytes per node per round**, demonstrating an extraordinary **$604.4\times$ transmission efficiency** for edge model synchronization over LoRaWAN."*
