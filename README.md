# Decentralized Edge Intelligence for LoRaWAN
### Federated Learning for Environment-Driven Path Loss and Link Quality Modeling


Indoor LoRaWAN deployments are notoriously hard to model accurately : walls, humidity, PM2.5, temp, Baro pressure, and CO2 levels all distort signal propagation in ways static path-loss formulas can't capture. Centralized machine learning can learn these patterns, but it demands streaming raw sensor data to a server, which quickly collides with LoRaWAN's strict 1% duty-cycle limit, drains battery-powered nodes, and raises privacy concerns for occupancy-revealing environmental data.

This research asks whether a **federated, on-device approach** can close that gap: can a lightweight 89-parameter TinyML model, trained collaboratively across distributed end devices via FedAvg, match centralized accuracy without ever transmitting raw data? **The results say yes**. The federated model retains **96.9% of centralized accuracy (R²=0.8807 vs. 0.9089)** with RMSE increasing only slightly from **5.69 dB to 6.52 dB** (**+0.83 dB**, a modest tradeoff for the efficiency gains below), while cutting per-node data volume by **1.64x** and radio energy consumption by **11.36x**. A **hardware feasibility study further confirms** the approach runs comfortably on an **Arduino MKR WAN 1310**, fitting the full model update into a single LoRaWAN packet and staying well within duty-cycle and energy budgets.

 **Prior work:** The centralized ML baseline (MLR, Random Forest, XGBoost) and environmental data collection underpinning this thesis are documented in the predecessor repository : [LoRaWAN Indoor Office Environments: Environmental Effects on Path Loss & Signal Reliability](https://github.com/pratik001010/LoRaWAN-in-Indoor-Office-Environments-Environmental-Effects-on-Path-Loss-and-Signal-Reliability). This repository extends that foundation toward a fully federated, on-device learning framework.

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

![FL vs CL — data volume detail](https://raw.githubusercontent.com/pratik001010/Decentralized-Edge-Intelligence-for-LoRaWAN-Federated-Learning-for-Environment-Driven-Path-Loss-/26bb4ce307565a519ec256e37cc09648febfdb44/pics/25.png)

## 13. Hardware Feasibility

The 89-parameter (int8-quantized) model update fits in a single LoRaWAN packet, uses under 7% of the hourly duty-cycle budget even at SF12, and runs comfortably within MKR WAN 1310 memory limits (11% flash utilization).

![Hardware Feasibility — Duty Cycle](https://raw.githubusercontent.com/pratik001010/Decentralized-Edge-Intelligence-for-LoRaWAN-Federated-Learning-for-Environment-Driven-Path-Loss-/26bb4ce307565a519ec256e37cc09648febfdb44/pics/26.png)

![Hardware Feasibility — Memory & Compute](https://raw.githubusercontent.com/pratik001010/Decentralized-Edge-Intelligence-for-LoRaWAN-Federated-Learning-for-Environment-Driven-Path-Loss-/26bb4ce307565a519ec256e37cc09648febfdb44/pics/27.png)

## 14. Conclusion

FedAvg retains 96.9% of centralized accuracy while cutting radio energy use over 11x and data volume by 1.64x, confirming the feasibility of federated TinyML path-loss modeling on constrained LoRaWAN hardware.

![Conclusion](https://raw.githubusercontent.com/pratik001010/Decentralized-Edge-Intelligence-for-LoRaWAN-Federated-Learning-for-Environment-Driven-Path-Loss-/26bb4ce307565a519ec256e37cc09648febfdb44/pics/28.png)

## 15. Future Work

Planned next steps include testing advanced optimization algorithms (FedProx, SCAFFOLD) and real-world deployment with pilot testing.

![Future Work](https://raw.githubusercontent.com/pratik001010/Decentralized-Edge-Intelligence-for-LoRaWAN-Federated-Learning-for-Environment-Driven-Path-Loss-/26bb4ce307565a519ec256e37cc09648febfdb44/pics/29.png)

---

![End of Presentation](https://raw.githubusercontent.com/pratik001010/Decentralized-Edge-Intelligence-for-LoRaWAN-Federated-Learning-for-Environment-Driven-Path-Loss-/26bb4ce307565a519ec256e37cc09648febfdb44/pics/30.png)

## 16. Appendices


 ![Appendice1](https://raw.githubusercontent.com/pratik001010/Decentralized-Edge-Intelligence-for-LoRaWAN-Federated-Learning-for-Environment-Driven-Path-Loss-/26bb4ce307565a519ec256e37cc09648febfdb44/pics/31.png)
 
  ![Appendice2](https://raw.githubusercontent.com/pratik001010/Decentralized-Edge-Intelligence-for-LoRaWAN-Federated-Learning-for-Environment-Driven-Path-Loss-/26bb4ce307565a519ec256e37cc09648febfdb44/pics/32.png)

 ![Appendice3](https://raw.githubusercontent.com/pratik001010/Decentralized-Edge-Intelligence-for-LoRaWAN-Federated-Learning-for-Environment-Driven-Path-Loss-/26bb4ce307565a519ec256e37cc09648febfdb44/pics/33.png)

  ![Appendice4](https://raw.githubusercontent.com/pratik001010/Decentralized-Edge-Intelligence-for-LoRaWAN-Federated-Learning-for-Environment-Driven-Path-Loss-/26bb4ce307565a519ec256e37cc09648febfdb44/pics/34.png)

   ![Appendice5](https://raw.githubusercontent.com/pratik001010/Decentralized-Edge-Intelligence-for-LoRaWAN-Federated-Learning-for-Environment-Driven-Path-Loss-/26bb4ce307565a519ec256e37cc09648febfdb44/pics/35.png)


**Author:** Pratik Khadka ([Pratik.Khadka@student.uni-siegen.de](mailto:Pratik.Khadka@student.uni-siegen.de))([Eng.pratikkhadka@meetpratik.me](mailto:eng.pratikkhadka@meetpratik.me))
