# Decentralized Edge Intelligence for LoRaWAN: Federated Learning for Environment-Driven Path Loss and Link Quality Modeling

## Abstract
This work presents an upgrade from centralized LoRaWAN path-loss modeling to decentralized edge intelligence for environment-driven path-loss and link-quality estimation. In the previous system, six Arduino MKR WAN 1310 nodes collected multimodal environmental and radio-context data in an indoor office setting, and model training was performed offline at the backend. Although strong predictive baselines were achieved, the deployed intelligence remained largely centralized and static.

The updated architecture introduces federated TinyML, where each LoRaWAN node performs on-device inference, maintains local learning context, and sends compact model updates instead of continuous raw-data streams. A backend federated orchestrator aggregates distributed updates into a global model and periodically redistributes model information under constrained downlink availability. To remain compatible with practical LoRaWAN operation, the design combines compact payload encoding, event-triggered reporting, and round-based federated optimization.

The resulting framework enables continuous adaptation to local environmental and radio variations while reducing communication burden and preserving data locality. This establishes a practical path from train-once centralized modeling to communication-aware, distributed learning for long-running LoRaWAN deployments.

**Keywords:** LoRaWAN, Federated Learning, TinyML, Path Loss Modeling, Link Quality Modeling, Edge Intelligence, Event-Driven Communication

---

## 1. Introduction

### 1.1 Context and Upgrade Rationale
LoRaWAN deployments are widely used for low-power, long-range sensing but operate under strict bandwidth and airtime constraints. In this setting, continuously uploading full sensor streams to a centralized backend is expensive and does not naturally support long-term adaptation. For environment-driven radio behavior, this limitation is critical: link conditions change with occupancy, obstacles, air quality, and local dynamics that evolve after deployment.

The previous project established a strong centralized baseline by collecting a large multi-sensor dataset and training server-side models for path-loss estimation. However, the deployed system remained primarily centralized: nodes sensed and transmitted, while model intelligence and adaptation stayed in the backend. The new thesis upgrades this architecture to distributed learning at the edge.

### 1.2 Problem Definition
The research problem is how to design a LoRaWAN-compatible learning system that can:
1. Preserve strong predictive performance for path loss and link quality.
2. Reduce dependency on continuous raw-data upload.
3. Adapt over time to local environmental and radio drift.
4. Respect constrained uplink/downlink behavior in practical LoRaWAN operation.

This requires a shift from static centralized training to a federated edge-learning pipeline.

### 1.3 Thesis Direction and Scope
This thesis is titled:

**Decentralized Edge Intelligence for LoRaWAN: Federated Learning for Environment-Driven Path Loss and Link Quality Modeling**

The scope is centered on:
1. On-device TinyML inference for real-time local decisions.
2. Local update generation from node-specific data history.
3. Federated aggregation of node updates into a shared global model.
4. Communication-aware design (event-triggered reporting, compact updates, low downlink dependence).

### 1.4 System-Level Concept
The system is organized into three interacting layers:
1. **Model preparation layer (offline):** centralized pretraining and model compression to initialize a small edge model.
2. **Node intelligence layer (online):** per-node sensing, feature construction, TinyML inference, and local update generation.
3. **Federated orchestration layer:** server-side aggregation and model-version management for periodic global redistribution.

In this design, each node acts as an independent learner with local context, while the federated server acts as the coordinator that combines distributed learning signals.

### 1.5 Feature and Labeling Strategy in Deployment
The deployed node model uses features that are available locally at runtime, including environmental measurements, radio configuration/context, and short local history signals. Optional coarse network feedback (e.g., sparse downlink summaries) is treated as slow calibration context rather than dense supervision.

A key practical principle is that the historical 1.3M-row dataset is used offline for pretraining, distillation, and baseline benchmarking. During live federated operation, nodes rely on online proxy-label logic and local behavior outcomes. Therefore, continuous streaming of the old dataset is not required for deployment.

### 1.6 Role of 2410.11612v1 in This Thesis
The paper 2410.11612v1 is used here as a methodological reference for:
1. FL client-server orchestration concepts.
2. Aggregation under LoRaWAN communication limits.
3. Message/time budgeting under spreading-factor payload constraints.
4. Round/epoch configuration trade-offs for constrained networks.

It is not used to redefine the thesis application domain. The application focus remains environment-driven path-loss and link-quality modeling in LoRaWAN edge nodes.

### 1.7 Expected Contributions
This thesis is expected to contribute:
1. A reproducible migration path from centralized LoRaWAN modeling to federated TinyML edge learning.
2. A practical FL communication design tailored to LoRaWAN constraints.
3. A node-level intelligence framework for adaptive, environment-aware path-loss and link-quality estimation.
4. An empirical comparison between static centralized baselines and adaptive federated operation.

---

## Notes for Supervisor Review
This draft intentionally follows scientific-paper structure while staying aligned to the discussed project scope and slide content. Terminology and framing are constrained to the thesis title and LoRaWAN-FL methodology, with 2410.11612v1 treated strictly as literature support for FL protocol design in constrained networks.
