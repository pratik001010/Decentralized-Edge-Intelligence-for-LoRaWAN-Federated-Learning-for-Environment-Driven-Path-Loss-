# Decentralized Edge Intelligence for LoRaWAN
### Federated Learning for Environment-Driven Path Loss and Link Quality Modeling

### Abstract
The rapid expansion of the Internet of Things has pushed intelligence away from centralized clouds and toward the edge, a shift made possible by **Tiny Machine Learning (TinyML)**: running compact machine learning models directly on microcontrollers with only a few kilobytes of memory to spare. TinyML matters most for networks like **Long Range Wide Area Network (LoRaWAN)**. It gives long range and battery-friendly connectivity, but the tradeoff is steep: a 1% duty-cycle ceiling, uplink payloads capped at a few hundred bytes, and battery budgets built for years, not months. Indoor path loss prediction runs straight into these limits.
Radio signals indoors do not behave the way they do outdoors. Walls absorb and scatter them, humidity and temperature shift how materials interact with the signal, and CO<sub>2</sub> levels, rising and falling with how many people are actually in the room, add a human-driven layer of attenuation that distance-only formulas were never built to handle. Worse, the channel itself keeps shifting: reflections bounce unpredictably, signal strength drops without warning, and shadow fading breaks the assumption that propagation stays steady over time. Prior centralized regression work tried to fix this by folding environmental covariates into the model, but it still depends on streaming raw sensor data to a central server, which quickly runs into LoRaWAN's duty-cycle ceiling, drains node batteries, and exposes occupancy-revealing environmental readings that raise real privacy concerns.
This thesis tackles that gap with a **Federated Learning (FL)** framework paired with on-device TinyML inference. A compact **Dense(9-8-1) Multilayer Perceptron** with **89** trainable parameters runs across six edge nodes, each predicting expected path loss (dB) locally from nine covariates, log-distance, wall counts, environmental readings, and gateway Signal-to-Noise Ratio (SNR), while raw sensor data never leaves the device. Using Federated Averaging (**FedAvg**) with five local epochs, the model trains over 50 communication rounds on a 12-month, six-room indoor dataset comprising **206,957** observations sampled at 10-minute intervals, naturally partitioned non-Independent and Identically Distributed (IID) by physical room location.
The federated model reaches an R<sup>2</sup> of **0.8807** and RMSE of **6.52 dB**, retaining **96.90%** of the centralized baseline's accuracy (R<sup>2</sup> **0.9089**, RMSE **5.6872 dB**), a gap of just **0.83 dB** (**+14.6%** relative). Looking at individual devices rather than the aggregate, **FedAvg** improves cross-room model robustness and spatial generalization across all deployment sites. On the communication side, quantized `int8` weight updates of just **89 bytes** fit inside a single LoRaWAN packet. Incorporating live gateway Received Signal Strength Indicator (RSSI) feedback downlinks and single-packet model updates reduces application data volume per node from **18.40 KB** down to **1.11 KB** per 7.1-day cycle (a **16.6×** data reduction) and active transceiver energy by **3.21×** (from **4,375.2 mJ** to **1,362.8 mJ**). The embedded implementation demonstrates TensorFlow Lite Micro inference for the compact path-loss model on the Arduino MKR WAN 1310, running inference in under **0.15 ms** with Flash memory usage of **28.4 KB** (**11%** of the **256 KB** budget). Federated local optimization and sample-weighted aggregation were evaluated in an offline Python simulation. The memory and LoRaWAN communication analyses provide analytical feasibility assessments for a future embedded-training implementation. Together, these results demonstrate the feasibility of privacy-conscious, localized edge intelligence for indoor path loss modeling over constrained LPWAN links.

**Keywords:** Federated Learning, TinyML, LoRaWAN, LPWAN, Path Loss Modeling, Edge Intelligence, Indoor Propagation, Internet of Things.

---
 **Prior work:** The centralized ML baseline (MLR, Random Forest, XGBoost) and environmental data collection underpinning this work are documented in the predecessor repository : [LoRaWAN Indoor Office Environments: Environmental Effects on Path Loss & Signal Reliability](https://github.com/pratik001010/LoRaWAN-in-Indoor-Office-Environments-Environmental-Effects-on-Path-Loss-and-Signal-Reliability). This repository extends that foundation toward a fully federated, on-device learning framework.

---

## 1. Motivation: Why Decentralize?

IoT device growth (18B → 29B by 2030) with TinyML growth (1.8B → 18.2B  by 2035) and LoRaWAN's 1% duty-cycle limit make raw telemetry streaming impractical; Federated Learning keeps raw data on-device and transmits only compact model weights.

![Why Decentralize](https://raw.githubusercontent.com/pratik001010/Decentralized-Edge-Intelligence-for-LoRaWAN-Federated-Learning-for-Environment-Driven-Path-Loss-/26bb4ce307565a519ec256e37cc09648febfdb44/pics/1.png)

## 2. Data Collection Campaign

A one-year, 2M+ row dataset was collected from 6 end devices and 1 Kerlink gateway across a multi-room indoor floor plan, logging temperature, humidity, CO2, pressure, PM2.5, and SNR every 60 seconds.

![Data Collection Campaign](https://raw.githubusercontent.com/pratik001010/Decentralized-Edge-Intelligence-for-LoRaWAN-Federated-Learning-for-Environment-Driven-Path-Loss-/26bb4ce307565a519ec256e37cc09648febfdb44/pics/2.png)


## 3. Research Questions

Four research questions guide the study: accuracy parity with centralized training, non-IID convergence behavior, communication/energy savings, and on-device hardware feasibility.

![Research Questions](https://raw.githubusercontent.com/pratik001010/Decentralized-Edge-Intelligence-for-LoRaWAN-Federated-Learning-for-Environment-Driven-Path-Loss-/26bb4ce307565a519ec256e37cc09648febfdb44/pics/3.png)

## 4. Related Work: Positioning This Study

This thesis is positioned against prior work in indoor path-loss modeling, FL over LoRaWAN, and non-IID FL optimizers, distinguishing itself via real non-IID deployment feasibility analysis with compact 89-byte updates.

![Related Work](https://raw.githubusercontent.com/pratik001010/Decentralized-Edge-Intelligence-for-LoRaWAN-Federated-Learning-for-Environment-Driven-Path-Loss-/26bb4ce307565a519ec256e37cc09648febfdb44/pics/4.png)

## 5. TinyML Model Architecture

The core model is a compact Dense 9-8-1 MLP (89 trainable parameters) taking 9 normalized features as input and predicting path loss (dB) via one ReLU hidden layer and a linear output.

![TinyML Architecture — z1 formula](https://raw.githubusercontent.com/pratik001010/Decentralized-Edge-Intelligence-for-LoRaWAN-Federated-Learning-for-Environment-Driven-Path-Loss-/26bb4ce307565a519ec256e37cc09648febfdb44/pics/8.png)

## 6. Overall Methodology Workflow

The pipeline runs in three phases: data preprocessing and model initialization, centralized/federated model training, and post-simulation benchmarking (convergence, per-client evaluation, energy, and hardware feasibility).

![Methodology Workflow](https://raw.githubusercontent.com/pratik001010/Decentralized-Edge-Intelligence-for-LoRaWAN-Federated-Learning-for-Environment-Driven-Path-Loss-/26bb4ce307565a519ec256e37cc09648febfdb44/pics/9.png)

## 7. Centralized Baseline: Implementation

The centralized model uses an 80/20 per-device split with an internal 85/15 train/validation split, trained with Adam (lr=0.01), batch size 512, for 50 epochs.

![Centralized Baseline](https://raw.githubusercontent.com/pratik001010/Decentralized-Edge-Intelligence-for-LoRaWAN-Federated-Learning-for-Environment-Driven-Path-Loss-/26bb4ce307565a519ec256e37cc09648febfdb44/pics/10.png)

## 8. Centralized Baseline: Results

Training and validation R² and RMSE curves converge closely with no overfitting, reaching a final held-out test R² of 0.9089 and RMSE of 5.6872 dB.

![Centralized Training Curves](https://raw.githubusercontent.com/pratik001010/Decentralized-Edge-Intelligence-for-LoRaWAN-Federated-Learning-for-Environment-Driven-Path-Loss-/26bb4ce307565a519ec256e37cc09648febfdb44/pics/11.png)

## 9. FedAvg Baseline: Implementation

Six clients train locally (5 local epochs per round) and send weight updates to a server, which aggregates via FedAvg over 50 communication rounds; the ground-truth label is path loss derived from RSSI via the link-budget equation.

![FedAvg Baseline — build step 4](https://raw.githubusercontent.com/pratik001010/Decentralized-Edge-Intelligence-for-LoRaWAN-Federated-Learning-for-Environment-Driven-Path-Loss-/26bb4ce307565a519ec256e37cc09648febfdb44/pics/18.png)

## 10. FedAvg Convergence Results

FedAvg with 5 local epochs reaches R²=0.8807 and RMSE=6.52 dB by round 50, approaching the centralized benchmark of R²=0.9089.

![FedAvg Convergence — chart only variant](https://raw.githubusercontent.com/pratik001010/Decentralized-Edge-Intelligence-for-LoRaWAN-Federated-Learning-for-Environment-Driven-Path-Loss-/26bb4ce307565a519ec256e37cc09648febfdb44/pics/21.png)

## 11. Per-Client Performance (Non-IID)

FedAvg (mean R²=0.689) outperforms the centralized model (mean R²=0.613) on average across all six end devices under non-IID room-based data partitioning.

![Per-Client Mean Comparison](https://raw.githubusercontent.com/pratik001010/Decentralized-Edge-Intelligence-for-LoRaWAN-Federated-Learning-for-Environment-Driven-Path-Loss-/26bb4ce307565a519ec256e37cc09648febfdb44/pics/22.png)

![Per-Device Breakdown](https://raw.githubusercontent.com/pratik001010/Decentralized-Edge-Intelligence-for-LoRaWAN-Federated-Learning-for-Environment-Driven-Path-Loss-/26bb4ce307565a519ec256e37cc09648febfdb44/pics/23.png)

## 12. Communication & Energy Efficiency

Federated Learning reduces data volume per node by 1.64x (123,300 B → 75,183 B) and radio energy consumption by 11.36x (6,795.78 mJ → 597.99 mJ) over 7.1 days.

![FL vs CL — data volume detail](https://github.com/pratik001010/Decentralized-Edge-Intelligence-for-LoRaWAN-Federated-Learning-for-Environment-Driven-Path-Loss-/blob/3cfedbdeb1db060d6699d5b6438fabb19c975581/pics/25.png)

## 13. Hardware Feasibility

The 89-parameter (int8-quantized) model update fits in a single LoRaWAN packet, uses under 7% of the hourly duty-cycle budget even at SF12, and runs comfortably within MKR WAN 1310 memory limits (11% flash utilization).

![Hardware Feasibility — Duty Cycle](https://raw.githubusercontent.com/pratik001010/Decentralized-Edge-Intelligence-for-LoRaWAN-Federated-Learning-for-Environment-Driven-Path-Loss-/26bb4ce307565a519ec256e37cc09648febfdb44/pics/26.png)

![Hardware Feasibility — Memory & Compute](https://github.com/pratik001010/Decentralized-Edge-Intelligence-for-LoRaWAN-Federated-Learning-for-Environment-Driven-Path-Loss-/blob/3cfedbdeb1db060d6699d5b6438fabb19c975581/pics/27.png)

## 14. Conclusion

FedAvg retains 96.9% of centralized accuracy while cutting radio energy use over 11x and data volume by 1.64x, confirming the feasibility of federated TinyML path-loss modeling on constrained LoRaWAN hardware.

![Conclusion](https://github.com/pratik001010/Decentralized-Edge-Intelligence-for-LoRaWAN-Federated-Learning-for-Environment-Driven-Path-Loss-/blob/3cfedbdeb1db060d6699d5b6438fabb19c975581/pics/28.png)

## 15. Future Work

Planned next steps include testing advanced optimization algorithms (FedProx, SCAFFOLD) and real-world deployment with pilot testing.

![Future Work](https://github.com/pratik001010/Decentralized-Edge-Intelligence-for-LoRaWAN-Federated-Learning-for-Environment-Driven-Path-Loss-/blob/3cfedbdeb1db060d6699d5b6438fabb19c975581/pics/29.png)

---

![End of Presentation](https://raw.githubusercontent.com/pratik001010/Decentralized-Edge-Intelligence-for-LoRaWAN-Federated-Learning-for-Environment-Driven-Path-Loss-/26bb4ce307565a519ec256e37cc09648febfdb44/pics/30.png)

## 16. Appendices


 ![Appendice1](https://github.com/pratik001010/Decentralized-Edge-Intelligence-for-LoRaWAN-Federated-Learning-for-Environment-Driven-Path-Loss-/blob/3cfedbdeb1db060d6699d5b6438fabb19c975581/pics/31.png)
 
  ![Appendice2](https://github.com/pratik001010/Decentralized-Edge-Intelligence-for-LoRaWAN-Federated-Learning-for-Environment-Driven-Path-Loss-/blob/3cfedbdeb1db060d6699d5b6438fabb19c975581/pics/32.png)

 ![Appendice3](https://raw.githubusercontent.com/pratik001010/Decentralized-Edge-Intelligence-for-LoRaWAN-Federated-Learning-for-Environment-Driven-Path-Loss-/26bb4ce307565a519ec256e37cc09648febfdb44/pics/33.png)

  ![Appendice4](https://raw.githubusercontent.com/pratik001010/Decentralized-Edge-Intelligence-for-LoRaWAN-Federated-Learning-for-Environment-Driven-Path-Loss-/26bb4ce307565a519ec256e37cc09648febfdb44/pics/34.png)

   ![Appendice5](https://raw.githubusercontent.com/pratik001010/Decentralized-Edge-Intelligence-for-LoRaWAN-Federated-Learning-for-Environment-Driven-Path-Loss-/26bb4ce307565a519ec256e37cc09648febfdb44/pics/35.png)


**Author:** Pratik Khadka ([Pratik.Khadka@student.uni-siegen.de](mailto:Pratik.Khadka@student.uni-siegen.de))([Eng.pratikkhadka@meetpratik.me](mailto:eng.pratikkhadka@meetpratik.me))
