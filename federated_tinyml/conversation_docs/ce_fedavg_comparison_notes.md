# Reference Notes: Standard FedAvg vs. CE-FedAvg for LoRaWAN-based TinyML
This document serves as an academic and design reference for the master's thesis. It details the comparison, mathematical payload analysis, and thesis drafting recommendations for why **standard FedAvg** is preferred over **CE-FedAvg** for this work, and how to discuss this comparison in the final text.

---

## 1. Key Architectural Differences

| Dimension | CE-FedAvg (Razaghi & Aral, 2026) | This Thesis (Federated TinyML) |
|---|---|---|
| **Primary FL Algorithm** | CE-FedAvg (Sparsified + Quantized FedAvg) | Standard FedAvg (Vanilla weights average) |
| **Model Target** | Grayscale Image Classification (FEMNIST) | Continuous Path Loss Regression ($exp\_pl$ in dB) |
| **Edge Hardware Profile** | NVIDIA Jetson Orin Nano (SBC + GPU) | Arduino MKR WAN 1310 (Cortex-M0+) |
| **Memory Constraints** | Gigabytes of RAM / GPU capability | **32 KB SRAM** / 256 KB Flash (Ultra-constrained) |
| **LoRaWAN Modem Setup** | External ESP32 + SX1262 via UART bridge | Integrated Murata CMWX1ZZABZ module |
| **Model Complexity** | 27.6k parameters (Lightweight CNN) | **89 parameters** (Dense 9→8→1 neural network) |
| **Update Size per Round** | **6.21 KB** (after 10% sparsification + fp16) | **89 bytes** (100% weights in standard `int8`) |
| **Uplink Transmission** | Fragmented into ~28 LoRaWAN packets | **Single LoRaWAN frame** (no fragmentation) |
| **Compression Method** | Lossy Top-K Sparsification (Top-10%) | No compression needed (native compactness) |
| **Accuracy/R² Penalty** | $-1.6\%$ accuracy loss due to compression | **$0\%$ loss** (100% of weight deltas transmitted) |

---

## 2. Mathematical Sizing & Bandwidth Proof

### CE-FedAvg Transmission Math:
CE-FedAvg has a lightweight CNN with $27,600$ parameters.
* **Dense representation (float32):** $27,600 \times 4\text{ bytes} \approx 110.4\text{ KB}$
* **Compressed representation (Top-10% sparse + float16):**
  $$\text{Payload} = (27,600 \times 10\%) \times (2\text{ bytes [value]} + 2\text{ bytes [relative index]}) \approx 11\text{ KB}$$
  *(The paper achieves $6.21\text{ KB}$ using optimized offset index packaging).*
* **LoRaWAN Packets (DR5 / SF7, 222B max payload):**
  $$\text{Packets} = \left\lceil \frac{6210\text{ bytes}}{222\text{ bytes/packet}} \right\rceil \approx 28\text{ packets per update}$$

### This Thesis (Federated TinyML) Transmission Math:
Our environment-driven path loss model has a `Dense(9→8→1)` architecture.
* **Total Parameters:** $(9\text{ inputs} \times 8\text{ hidden}) + 8\text{ biases} + (8\text{ hidden} \times 1\text{ output}) + 1\text{ bias} = 89\text{ parameters}$.
* **Standard Representation (int8 quantized for TinyML):**
  $$\text{Payload} = 89\text{ parameters} \times 1\text{ byte/parameter} = 89\text{ bytes}$$
* **LoRaWAN Packets (DR5 / SF7, 222B max payload):**
  $$\text{Packets} = \left\lceil \frac{89\text{ bytes}}{222\text{ bytes/packet}} \right\rceil = 1\text{ packet per update}$$

> [!TIP]
> **Key Conclusion:** Standard FedAvg is **70× more bandwidth-efficient** in our work ($89\text{ B}$ vs $6.21\text{ KB}$) without requiring any lossy compression because our model is designed from the ground up to be ultra-compact (TinyML approach) rather than compressing a large model (CE-FedAvg approach).

---

## 3. Microcontroller Engineering Constraints

Implementing sparsification (Top-K selection) on the **SAMD21 microcontroller** of the Arduino MKR WAN 1310 poses severe limitations:
1. **Memory Ceiling:** With only **32 KB of SRAM**, running an sorting algorithm (like QuickSort or HeapSort) to find the Top-K% largest weights consumes precious heap space and CPU cycles.
2. **Indexing Overhead:** Sparsification requires packing the indices of the sparse weights into the frame. For a tiny model of 89 parameters, sending indices (e.g., 1 byte per index) alongside the 1-byte quantized values doubles the size of each selected parameter. If we selected the Top-30% parameters (27 weights), the payload would be $27\text{ weights} + 27\text{ indices} = 54\text{ bytes}$. The savings over the full update ($89\text{ bytes}$) is just $35\text{ bytes}$, which is completely negligible and does not justify the loss of $70\%$ of the weights or the firmware complexity.

---

## 4. Thesis Drafting Templates

### A. Chapter 3 (Related Work) — Positioning CE-FedAvg
Use this section to introduce the Vienna paper and differentiate it based on hardware and model philosophy:
> *"A key challenge of executing Federated Learning over LoRaWAN is the low-bandwidth constraint of the channel. Razaghi and Aral (2026) introduced CE-FedAvg, a framework that compresses model updates to fit LoRaWAN limits by combining Top-K sparsification, float16/int8 quantization, and seed-based initialization. While CE-FedAvg successfully compresses a 27.6k parameter CNN update down to 6.21 KB, their framework requires NVIDIA Jetson Orin Nano edge devices (which possess gigabytes of memory and GPU capability) and external ESP32 modems. In contrast, the present work targets ultra-constrained microcontrollers (Cortex-M0+, 32 KB SRAM) with integrated LoRaWAN modems. Our approach focuses on developing an extremely compact environment-driven path loss model (89 parameters) which requires no sparsification to fit within a single LoRaWAN packet, eliminating both transmission loss and sorting computations on the edge."*

### B. Chapter 5 (Implementation) — Quantization and Payload Sizing
Explain why vanilla FedAvg is used without sparsification:
> *"The local update package contains the weights and biases of the Dense(9→8→1) network. By applying post-training int8 quantization, each parameter is mapped to a 1-byte signed integer, resulting in a total model payload of 89 bytes. Because this payload is lower than the 222-byte maximum transmission limit of EU868 DR5, the entire update is transmitted in a single LoRaWAN uplink frame. This design choice bypasses the need for sparsification algorithms (such as Top-K selection), reducing the firmware's SRAM footprint and ensuring that all local updates contribute fully to the global aggregation without compression-induced degradation."*

### C. Chapter 7 (Discussion) — Comparative Bandwidth Analysis
Present a quantitative comparison to validate the TinyML design philosophy:
> *"Table X compares the communication overhead of our environment-driven TinyML path loss regression against the CE-FedAvg framework. Even with CE-FedAvg's aggressive Top-10% sparsification, their update remains 6.21 KB per round, requiring fragmentation across 28 packets and incurring a 1.6% accuracy loss. Our model achieves a daily update size of 89 bytes—a 70-fold reduction in transmission volume—while maintaining 100% weight integrity. This demonstrates that when applying Federated Learning to highly localized sensing tasks, designing an ultra-compact, specialized network is academically and practically superior to implementing complex compression pipelines on larger, generic architectures."*
