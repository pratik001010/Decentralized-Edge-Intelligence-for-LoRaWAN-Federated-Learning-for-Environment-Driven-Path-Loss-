# Federated TinyML for LoRaWAN Edge Intelligence

## Master's Thesis Project
**Author:** Pratik Khadka  
**Date:** March 2026

---

# Table of Contents

1. [Project Overview](#project-overview)
2. [Previous Project (PEP): Centralized ML](#previous-project-pep-centralized-ml)
3. [New Thesis: Federated TinyML](#new-thesis-federated-tinyml)
4. [Side-by-Side Comparison](#side-by-side-comparison)
5. [System Architecture](#system-architecture)
6. [Hardware Requirements](#hardware-requirements)
7. [Software Requirements](#software-requirements)
8. [File Structure](#file-structure)
9. [Step-by-Step Setup Guide](#step-by-step-setup-guide)
10. [Code Walkthrough](#code-walkthrough)
11. [Payload Formats](#payload-formats)
12. [Configuration Parameters](#configuration-parameters)
13. [Troubleshooting](#troubleshooting)
14. [References](#references)

---

# Project Overview

This project evolves a LoRaWAN-based environmental monitoring system from a **centralized data collection approach** to a **distributed federated learning system** where each edge device (Arduino MKR WAN 1310) becomes an intelligent node capable of on-device machine learning.

---

# Previous Project (PEP): Centralized ML

## What It Was

The original project (`End_Device_Arduino_Sketch.ino`) was a **data logging system** where:

- 6 MKR WAN 1310 nodes collected environmental sensor data
- ALL raw data was uploaded to a backend server via LoRaWAN
- Machine learning was performed **centrally on a server** after data collection
- The trained model (XGBoost → distilled to TFLite) was static - "train once, deploy, forget"

## Old Code: End_Device_Arduino_Sketch.ino

### Libraries Used
```cpp
#include <MKRWAN.h>           // LoRaWAN communication
#include <Wire.h>             // I2C bus
#include <Adafruit_Sensor.h>  // Sensor abstraction
#include <Adafruit_BME280.h>  // Pressure sensor
#include <SensirionI2CScd4x.h> // CO2 sensor
#include <sps30.h>            // PM2.5 sensor
```

### Sensors Connected
| Sensor | Data Collected | I2C Address |
|--------|----------------|-------------|
| BME280 | Pressure (hPa) | 0x77 |
| SCD4x | CO2 (ppm), Temperature (°C), Humidity (%) | Default |
| SPS30 | PM2.5 (µg/m³) | Default |

### Data Flow (OLD)
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Sensors   │────▶│  MKR WAN    │────▶│   Server    │
│  BME280     │     │  1310       │     │  (Backend)  │
│  SCD4x      │     │             │     │             │
│  SPS30      │     │ Raw Data    │     │ Train ML    │
└─────────────┘     │ Upload      │     │ Model       │
                    └─────────────┘     └─────────────┘
```

### What the Old Code Did

1. **setup()**: Initialize I2C, sensors, LoRaWAN modem, join network via OTAA
2. **loop()**: Every 60 seconds:
   - Read pressure from BME280
   - Read CO2, temperature, humidity from SCD4x
   - Read PM2.5 from SPS30
   - Package ALL data into 18-byte payload
   - Send via LoRaWAN uplink
   - Cycle through data rates (DR0-DR5)

### Old Payload Structure (18 bytes)
| Bytes | Data | Encoding |
|-------|------|----------|
| 0-1 | Pressure × 100 | int16, big-endian |
| 2-3 | CO2 | uint16, big-endian |
| 4-5 | Temperature × 100 | int16, big-endian |
| 6-7 | Humidity × 100 | int16, big-endian |
| 8-9 | PM2.5 × 100 | int16, big-endian |
| 10-13 | Packet count | uint32, big-endian |
| 14-17 | Unused | - |

### Limitations of Old Approach

| Issue | Description |
|-------|-------------|
| **Bandwidth** | All raw data uploaded (1.3M rows total) |
| **Privacy** | Sensitive data leaves the device |
| **Adaptability** | Static model, no learning after deployment |
| **Energy** | Continuous transmission every 60s |
| **Intelligence** | Device is "dumb" - just collects and sends |

---

# New Thesis: Federated TinyML

## What It Is Now

The new system (`FederatedTinyML.ino`) transforms each node into an **intelligent edge device**:

- Each node runs a **TinyML model** locally
- Nodes **train on their own data** (local training)
- Only **model weight updates** are sent (not raw data)
- A backend server **aggregates updates** using FedAvg
- Nodes receive a **global model** via rare downlinks
- Transmission is **event-driven** based on ML predictions

## New Code: FederatedTinyML.ino

### Additional Libraries Required
```cpp
// Original libraries (same as before)
#include <MKRWAN.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BME280.h>
#include <SensirionI2CScd4x.h>
#include <sps30.h>

// NEW: TinyML libraries
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/schema/schema_generated.h>

// NEW: Trained model as C array
#include "model.h"
```

### New Data Flow
```
┌─────────────────────────────────────────────────────────────┐
│                    FL Server (Backend)                       │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │ Receive     │───▶│ FedAvg      │───▶│ Compress        │  │
│  │ Updates     │    │ Aggregation │    │ Global Model    │  │
│  └─────────────┘    └─────────────┘    └─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
        ▲  │                                    │
        │  │ Weight Deltas (Uplink)            │ Global Model (Downlink)
        │  │ ~50 bytes/round                   │ ~50 bytes (rare)
        │  │                                    ▼
┌───────┴──┴────────────────────────────────────────────────┐
│                   LoRaWAN Network (TTN)                    │
└────────────────────────────────────────────────────────────┘
        ▲                   ▲                   ▲
        │                   │                   │
┌───────┴────┐      ┌───────┴────┐      ┌───────┴────┐
│ MKR WAN 1  │      │ MKR WAN 2  │      │ MKR WAN 6  │
│            │      │            │      │            │
│ ┌────────┐ │      │ ┌────────┐ │      │ ┌────────┐ │
│ │TinyML  │ │      │ │TinyML  │ │      │ │TinyML  │ │
│ │Inference│ │      │ │Inference│ │      │ │Inference│ │
│ │+Training│ │      │ │+Training│ │      │ │+Training│ │
│ └────────┘ │      │ └────────┘ │      │ └────────┘ │
│ ┌────────┐ │      │ ┌────────┐ │      │ ┌────────┐ │
│ │Buffer  │ │      │ │Buffer  │ │      │ │Buffer  │ │
│ │32 smpls│ │      │ │32 smpls│ │      │ │32 smpls│ │
│ └────────┘ │      │ └────────┘ │      │ └────────┘ │
│ ┌────────┐ │      │ ┌────────┐ │      │ ┌────────┐ │
│ │Sensors │ │      │ │Sensors │ │      │ │Sensors │ │
│ └────────┘ │      │ └────────┘ │      │ └────────┘ │
└────────────┘      └────────────┘      └────────────┘
  Stairwell            Corridor            Kitchen
```

### What the New Code Does

#### setup()
1. Initialize Serial for debugging
2. Initialize I2C and all sensors (same as before)
3. **NEW:** Initialize TinyML interpreter and load model
4. Initialize LoRaWAN and join network
5. **NEW:** Zero-initialize weight arrays for FL

#### loop() - 8 Steps
| Step | Action | NEW? |
|------|--------|------|
| 1 | Read sensor data | Same |
| 2 | Normalize features for ML | **NEW** |
| 3 | Run TinyML inference → predict link state | **NEW** |
| 4 | Compute proxy label from actual PDR | **NEW** |
| 5 | Add sample to circular buffer | **NEW** |
| 6 | Event-driven transmission (based on prediction) | **NEW** |
| 7 | Federated Learning round (if due) | **NEW** |
| 8 | Adaptive delay (based on link state) | **NEW** |

### New Functions Added

| Function | Purpose |
|----------|---------|
| `initTinyML()` | Load TFLite model, create interpreter, allocate tensors |
| `normalizeFeatures()` | Scale features using mean/std from training data |
| `runInference()` | Feed features to model, get link state prediction |
| `argmax()` | Find index of maximum value in output tensor |
| `addSampleToBuffer()` | Store sample in circular buffer for training |
| `computeProxyLabel()` | Derive ground truth from packet delivery ratio |
| `localTraining()` | Train model locally using buffered samples |
| `computeWeightDeltas()` | Calculate difference: local - global weights |
| `sendModelUpdate()` | Send quantized weight deltas via LoRaWAN |
| `receiveGlobalModel()` | Process downlink with aggregated global model |
| `eventDrivenTransmission()` | Smart transmission based on ML predictions |

---

# Side-by-Side Comparison

| Aspect | OLD (End_Device_Arduino_Sketch.ino) | NEW (FederatedTinyML.ino) |
|--------|-------------------------------------|---------------------------|
| **Device role** | Data logger | Intelligent edge node |
| **ML location** | Server only | Device + Server |
| **Data sent** | ALL raw sensor data (14 bytes/msg) | Weight deltas only (52 bytes/round) |
| **Transmission** | Fixed 60s interval | Event-driven (5min normal, 1min urgent) |
| **Model updates** | None | Receives global model via downlink |
| **Adaptability** | Static ("train once") | Continuous learning |
| **Privacy** | Data leaves device | Data stays on device |
| **Bandwidth** | High (every minute) | Low (deltas only, daily rounds) |
| **Energy** | Constant drain | Adaptive based on link state |
| **TinyML** | None | TFLite Micro inference + training |
| **Libraries** | 6 | 6 + TensorFlowLite |
| **RAM usage** | ~2 KB | ~12 KB (8KB tensor arena) |
| **Code size** | ~150 lines | ~700 lines |

---

# System Architecture

## Complete System Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                         YOUR PC / CLOUD                          │
│  ┌────────────────┐    ┌─────────────────────────────────────┐   │
│  │ train_model.py │    │           fl_server.py              │   │
│  │                │    │                                     │   │
│  │ - Train model  │    │ - Flask API (port 5000)            │   │
│  │ - Convert to   │    │ - Receive TTN webhooks             │   │
│  │   TFLite       │    │ - FedAvg aggregation               │   │
│  │ - Generate     │    │ - Compress global model            │   │
│  │   model.h      │    │ - Schedule downlinks               │   │
│  └───────┬────────┘    └──────────────┬──────────────────────┘   │
│          │ model.h                    │ HTTP webhooks            │
│          ▼                            ▼                          │
└──────────────────────────────────────────────────────────────────┘
                                        │
                                        │ Internet
                                        ▼
┌──────────────────────────────────────────────────────────────────┐
│                    The Things Network (TTN)                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐        │
│  │ Network      │    │ Application  │    │ Webhook      │        │
│  │ Server       │◀──▶│ Server       │───▶│ Integration  │        │
│  └──────────────┘    └──────────────┘    └──────────────┘        │
└──────────────────────────────────────────────────────────────────┘
        ▲                                          │
        │ LoRaWAN (uplink/downlink)               │
        ▼                                          │
┌──────────────────┐                               │
│ LoRaWAN Gateway  │◀──────────────────────────────┘
└──────────────────┘
        ▲
        │ LoRa radio (868 MHz EU)
        ▼
┌──────────────────────────────────────────────────────────────────┐
│                    6x MKR WAN 1310 NODES                          │
│                                                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │
│  │   Node 1    │  │   Node 2    │  │   Node 6    │               │
│  │  Stairwell  │  │  Corridor   │  │  Kitchen    │               │
│  │             │  │             │  │             │               │
│  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │               │
│  │ │ BME280  │ │  │ │ BME280  │ │  │ │ BME280  │ │               │
│  │ │ SCD4x   │ │  │ │ SCD4x   │ │  │ │ SCD4x   │ │               │
│  │ │ SPS30   │ │  │ │ SPS30   │ │  │ │ SPS30   │ │               │
│  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │               │
│  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │               │
│  │ │ TinyML  │ │  │ │ TinyML  │ │  │ │ TinyML  │ │               │
│  │ │ Model   │ │  │ │ Model   │ │  │ │ Model   │ │               │
│  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │               │
│  └─────────────┘  └─────────────┘  └─────────────┘               │
└──────────────────────────────────────────────────────────────────┘
```

## Federated Learning Cycle

```
Daily FL Round:
                                                    
    Node 1          Node 2          Node 6          Server
       │               │               │               │
       │──── Local Training ──────────────────────────│
       │               │               │               │
       │    Δw₁        │    Δw₂        │    Δw₆        │
       │───────────────┼───────────────┼──────────────▶│
       │               │               │               │
       │               │               │         ┌─────┴─────┐
       │               │               │         │  FedAvg   │
       │               │               │         │ w = Σwᵢ/n │
       │               │               │         └─────┬─────┘
       │               │               │               │
       │◀──────────────┼───────────────┼───────────────│
       │          Global Model w                       │
       │               │               │               │
```

---

# Hardware Requirements

## Per Node (6 nodes total)

| Component | Model | Purpose |
|-----------|-------|---------|
| Microcontroller | Arduino MKR WAN 1310 | Main board with LoRaWAN |
| Pressure Sensor | BME280 | Atmospheric pressure |
| CO2 Sensor | SCD4x (SCD40/SCD41) | CO2, temp, humidity |
| PM Sensor | SPS30 | Particulate matter PM2.5 |
| Antenna | 868 MHz | LoRaWAN communication |

## MKR WAN 1310 Specifications

| Spec | Value | FL Usage |
|------|-------|----------|
| MCU | SAMD21 Cortex-M0+ | Runs TinyML |
| Clock | 48 MHz | Fast enough for inference |
| SRAM | 32 KB | 8KB for tensor arena, rest for buffers |
| Flash | 256 KB | Model (~500B) + code (~50KB) |
| LoRa | Murata CMWX1ZZABZ | EU868 band |

## Network Infrastructure

| Component | Description |
|-----------|-------------|
| LoRaWAN Gateway | Connected to TTN |
| TTN Account | Application created |
| Backend Server | Runs fl_server.py (any PC/cloud) |

---

# Software Requirements

## Arduino IDE Libraries

Install via **Sketch → Include Library → Manage Libraries**:

| Library | Version | Purpose |
|---------|---------|---------|
| MKRWAN | 1.1.0+ | LoRaWAN communication |
| Adafruit BME280 Library | 2.2.2+ | BME280 sensor |
| Adafruit Unified Sensor | 1.1.9+ | Sensor abstraction |
| Sensirion I2C SCD4x | 0.4.0+ | SCD4x CO2 sensor |
| sensirion-sps | 1.2.0+ | SPS30 PM sensor |
| **Arduino_TensorFlowLite** | **2.4.0+** | **TinyML (NEW)** |

### Arduino CLI Installation
```bash
arduino-cli lib install "MKRWAN"
arduino-cli lib install "Adafruit BME280 Library"
arduino-cli lib install "Adafruit Unified Sensor"
arduino-cli lib install "Sensirion I2C SCD4x"
arduino-cli lib install "sensirion-sps"
arduino-cli lib install "Arduino_TensorFlowLite"
```

## Python Requirements (for training & server)

```bash
pip install -r requirements.txt
```

**requirements.txt** contains:
```
tensorflow>=2.13.0
numpy>=1.24.0
scipy>=1.11.0
scikit-learn>=1.3.0
pandas>=2.0.0
flask>=3.0.0
requests>=2.31.0
matplotlib>=3.7.0
paho-mqtt>=1.6.0
```

---

# File Structure

```
c:\Users\prati\Desktop\edge AI\
│
├── End_Device_Arduino_Sketch.ino    # OLD: Original data logger
│
└── FederatedTinyML/                  # NEW: Federated learning project
    │
    ├── FederatedTinyML.ino           # Main Arduino sketch
    │   - Sensor reading
    │   - TinyML inference
    │   - Local training
    │   - FL communication
    │
    ├── model.h                       # TFLite model as C array
    │   - Placeholder (replace after training)
    │   - ~500 bytes when trained
    │
    ├── train_model.py                # Python training script
    │   - Define neural network
    │   - Train on dataset
    │   - Convert to TFLite
    │   - Generate model.h
    │
    ├── fl_server.py                  # Backend FL server
    │   - Flask API
    │   - Receive TTN webhooks
    │   - FedAvg aggregation
    │   - Downlink scheduling
    │
    ├── requirements.txt              # Python dependencies
    │
    ├── README.md                     # Quick start guide
    │
    └── THESIS_README.md              # This file (detailed documentation)
```

---

# Step-by-Step Setup Guide

## Phase 1: Train the Model (On Your PC)

### Step 1.1: Install Python Dependencies
```bash
cd "c:\Users\prati\Desktop\edge AI\FederatedTinyML"
pip install -r requirements.txt
```

### Step 1.2: Prepare Your Dataset
Option A: Use your existing 1.3M row dataset
```python
# In train_model.py, modify:
X, y = load_real_data("path/to/your/dataset.csv")
```

Option B: Use synthetic data for testing
```python
# Default behavior - generates synthetic data
X, y = generate_synthetic_data(5000)
```

### Step 1.3: Train the Model
```bash
python train_model.py
```

**Output:**
```
============================================================
TinyML Model Training for Federated Learning
============================================================

[1/6] Loading data...
  - Samples: 5000
  - Features: 5
  - Class distribution: [2500 1500 1000]

[2/6] Normalizing features...
  - Feature means: [1013.0, 800.0, 22.0, 50.0, 15.0]
  - Feature stds: [10.0, 400.0, 5.0, 20.0, 10.0]

[3/6] Splitting data...
  - Training samples: 4000
  - Test samples: 1000

[4/6] Training model...
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 hidden (Dense)              (None, 8)                 48
 output (Dense)              (None, 3)                 27
=================================================================
Total params: 75 (300.00 Byte)

[5/6] Evaluating model...
  - Test accuracy: 92.50%
  - Test loss: 0.2134

[6/6] Converting to TFLite...
TFLite model saved: model_output/model.tflite
Model size: 492 bytes
C header saved: model_output/model.h

============================================================
COPY THIS TO YOUR ARDUINO SKETCH:
============================================================

const float featureMeans[5] = {
    1013.0000, 800.0000, 22.0000, 50.0000, 15.0000
};
const float featureStds[5] = {
    10.0000, 400.0000, 5.0000, 20.0000, 10.0000
};

============================================================
DONE! Files generated:
============================================================
  - model_output/model.keras
  - model_output/model.tflite
  - model_output/model.h

Copy model_output/model.h to your Arduino sketch folder.
```

### Step 1.4: Copy Model to Arduino Folder
```bash
copy model_output\model.h FederatedTinyML\model.h
```

### Step 1.5: Update Normalization Parameters
Copy the printed means/stds to `FederatedTinyML.ino`:
```cpp
const float featureMeans[NUM_FEATURES] = {1013.0, 800.0, 22.0, 50.0, 15.0};
const float featureStds[NUM_FEATURES] = {10.0, 400.0, 5.0, 20.0, 10.0};
```

---

## Phase 2: Deploy to MKR WAN 1310

### Step 2.1: Configure LoRaWAN Credentials
Edit `FederatedTinyML.ino`:
```cpp
String appEui = "YOUR_TTN_APP_EUI";      // From TTN Console
String appKey = "YOUR_TTN_APP_KEY";      // From TTN Console
```

### Step 2.2: Open in Arduino IDE
1. Open `FederatedTinyML/FederatedTinyML.ino`
2. Select **Tools → Board → Arduino MKR WAN 1310**
3. Select **Tools → Port → COMx** (your device)

### Step 2.3: Install Libraries
If not already installed:
1. **Sketch → Include Library → Manage Libraries**
2. Search and install each library from the list above

### Step 2.4: Compile and Upload
1. Click **Verify** (✓) to compile
2. Click **Upload** (→) to flash

### Step 2.5: Monitor Serial Output
1. **Tools → Serial Monitor**
2. Set baud rate to **115200**

**Expected output:**
```
===========================================
Federated TinyML for MKR WAN 1310
Master's Thesis Project
===========================================
[INIT] Initializing sensors...
  - SCD4x: OK
  - BME280: OK
  - SPS30: OK
[INIT] Initializing TinyML...
  - Model loaded: OK
  - Tensor allocation: OK
  - Input tensor: 1x5
  - Output tensor: 1x3
  - Arena used: 3456/8192 bytes
[INIT] Initializing LoRaWAN...
  - Modem: OK (EU868)
  - Joining network (OTAA)...
  - Network joined: OK
  - ADR enabled: OK

[SETUP] Initialization complete!
===========================================

--- Sensor Readings ---
Pressure: 1015.32 hPa
CO2: 823 ppm
Temperature: 22.4 °C
Humidity: 48.2 %
PM2.5: 12.5 µg/m³

--- TinyML Prediction ---
  Probabilities: [0.8234, 0.1521, 0.0245]
Predicted Link State: GOOD

--- Buffer Status ---
Samples in buffer: 1/32

[LOOP] Next cycle in 300 seconds...
```

### Step 2.6: Repeat for All 6 Nodes
Repeat steps 2.1-2.5 for each MKR WAN 1310 node.

---

## Phase 3: Start FL Server

### Step 3.1: Run the Server
```bash
cd "c:\Users\prati\Desktop\edge AI\FederatedTinyML"
python fl_server.py
```

**Output:**
```
============================================================
Federated Learning Server for LoRaWAN Edge Devices
============================================================

Endpoints:
  GET  /health         - Health check
  GET  /status         - FL server status
  POST /uplink         - Receive model update (TTN webhook)
  GET  /downlink/<id>  - Get global model for device
  POST /schedule_downlink/<id> - Schedule downlink

Starting server on http://0.0.0.0:5000
============================================================

[FL Server] Starting Round 1
 * Running on http://0.0.0.0:5000
```

### Step 3.2: Test the Server
```bash
# Health check
curl http://localhost:5000/health

# Get status
curl http://localhost:5000/status
```

### Step 3.3: Simulate FL Round (Optional)
```bash
python fl_server.py simulate
```

---

## Phase 4: Configure TTN Webhook

### Step 4.1: Login to TTN Console
Go to: https://console.cloud.thethings.network/

### Step 4.2: Navigate to Your Application
**Applications → Your App → Integrations → Webhooks**

### Step 4.3: Add Webhook
1. Click **+ Add Webhook**
2. Select **Custom Webhook**
3. Configure:
   - **Webhook ID:** `fl-server`
   - **Base URL:** `http://YOUR_SERVER_IP:5000`
   - **Downlink API Key:** Generate one
   - **Enabled messages:** Check **Uplink Message**
   - **Path:** `/uplink`

### Step 4.4: Save Webhook
Click **Create Webhook**

---

# Code Walkthrough

## FederatedTinyML.ino - Key Sections

### 1. TinyML Initialization
```cpp
void initTinyML() {
    // Load model from model.h
    model = tflite::GetModel(g_model);
    
    // Create interpreter with tensor arena
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
    interpreter = &static_interpreter;
    
    // Allocate memory for tensors
    interpreter->AllocateTensors();
    
    // Get input/output tensor pointers
    input = interpreter->input(0);
    output = interpreter->output(0);
}
```

### 2. Running Inference
```cpp
uint8_t runInference(float* features) {
    // Copy features to input tensor
    for (int i = 0; i < NUM_FEATURES; i++) {
        input->data.f[i] = features[i];
    }
    
    // Run the model
    interpreter->Invoke();
    
    // Get prediction (argmax)
    return argmax(output->data.f, NUM_CLASSES);
}
```

### 3. Local Training
```cpp
void localTraining() {
    for (int epoch = 0; epoch < LOCAL_EPOCHS; epoch++) {
        for (int i = 0; i < bufferCount; i++) {
            // Get sample from buffer
            float* features = sampleBuffer[i].features;
            uint8_t actual = sampleBuffer[i].label;
            
            // Run inference
            uint8_t predicted = runInference(features);
            
            // Update weights based on error
            float error = (predicted != actual) ? 1.0 : 0.0;
            for (int w = 0; w < MODEL_WEIGHTS_SIZE; w++) {
                localWeights[w] -= learningRate * error * 0.01;
            }
        }
    }
}
```

### 4. Sending Model Update
```cpp
void sendModelUpdate() {
    uint8_t payload[51];
    payload[0] = 0x01;  // Message type: model update
    payload[1] = 0x01;  // Model version
    
    // Quantize weight deltas to int8
    for (int i = 0; i < 48; i++) {
        int8_t quantized = constrain(weightDeltas[i] * 127, -128, 127);
        payload[4 + i] = quantized;
    }
    
    // Send via LoRaWAN
    modem.beginPacket();
    modem.write(payload, 52);
    modem.endPacket(true);
}
```

### 5. Event-Driven Transmission
```cpp
void eventDrivenTransmission(uint8_t linkState) {
    bool shouldTransmit = false;
    
    // Trigger if link degraded to POOR
    if (linkState == LINK_STATE_POOR && lastLinkState != LINK_STATE_POOR) {
        shouldTransmit = true;
    }
    
    // Or if scheduled interval reached
    unsigned long interval = (linkState == LINK_STATE_GOOD) 
        ? NORMAL_TX_INTERVAL : URGENT_TX_INTERVAL;
    if (millis() - lastTransmission >= interval) {
        shouldTransmit = true;
    }
    
    if (shouldTransmit) {
        // Send minimal status packet (8 bytes vs 18 bytes raw data)
        sendStatusPacket();
    }
}
```

---

# Payload Formats

## Uplink Messages (Device → Server)

### Type 0x00: Status Packet (8 bytes)
```
Byte 0:     0x00 (message type)
Byte 1:     Link state (0=good, 1=degraded, 2=poor)
Byte 2:     PDR (0-100%)
Byte 3:     Current data rate (0-5)
Bytes 4-7:  Packet count (uint32, big-endian)
```

### Type 0x01: Model Update (52 bytes)
```
Byte 0:     0x01 (message type)
Byte 1:     Model version
Bytes 2-3:  Number of weights (uint16, big-endian)
Bytes 4-51: Quantized weight deltas (int8, 48 values)
```

## Downlink Messages (Server → Device)

### Type 0x02: Global Model (52 bytes)
```
Byte 0:     0x02 (message type)
Byte 1:     Model version
Bytes 2-3:  Number of weights (uint16, big-endian)
Bytes 4-51: Quantized global weights (int8, 48 values)
```

---

# Configuration Parameters

## Arduino (FederatedTinyML.ino)

```cpp
// TinyML
#define TENSOR_ARENA_SIZE     8192      // Memory for TFLite (bytes)
#define NUM_FEATURES          5         // Input features
#define NUM_CLASSES           3         // Output classes
#define MODEL_WEIGHTS_SIZE    128       // Trainable parameters

// Federated Learning
#define BUFFER_SIZE           32        // Samples stored for training
#define LOCAL_EPOCHS          3         // Training epochs per round
#define LOCAL_BATCH_SIZE      4         // Batch size
#define FL_ROUND_INTERVAL     86400000  // 24 hours (milliseconds)
#define MIN_SAMPLES_FOR_TRAIN 16        // Min samples before training

// Transmission
#define NORMAL_TX_INTERVAL    300000    // 5 minutes (good link)
#define URGENT_TX_INTERVAL    60000     // 1 minute (poor link)
```

## Server (fl_server.py)

```python
FL_CONFIG = {
    "num_clients": 6,                    # Total MKR WAN 1310 nodes
    "min_clients_per_round": 3,          # Minimum for aggregation
    "fl_round_duration_hours": 24,       # Round duration
    "model_weights_size": 128,           # Must match Arduino
    "quantization_scale": 127,           # int8 scaling factor
    "downlink_port": 3,                  # LoRaWAN port
}
```

---

# Troubleshooting

## Compilation Errors

### "TensorFlowLite.h not found"
**Solution:** Install Arduino_TensorFlowLite library
```
Sketch → Include Library → Manage Libraries → Search "TensorFlowLite"
```

### "model.h not found"
**Solution:** Run `python train_model.py` and copy `model_output/model.h` to sketch folder

### "Not enough memory"
**Solution:** Reduce `TENSOR_ARENA_SIZE` to 4096 or use smaller model

## Runtime Issues

### "Model schema mismatch"
**Solution:** Regenerate model with matching TFLite version

### "Tensor allocation failed"
**Solution:** Increase `TENSOR_ARENA_SIZE` or reduce model complexity

### "Network join failed"
**Solution:** 
- Verify appEui and appKey match TTN console
- Check gateway is online
- Ensure antenna is connected

### "Inference failed"
**Solution:** 
- Check input tensor dimensions
- Verify feature normalization

## FL Issues

### "No updates received on server"
**Solution:**
- Verify TTN webhook is configured
- Check server is accessible from internet
- Test with `curl http://your-server:5000/health`

### "Global model not received"
**Solution:**
- TTN limits downlinks to ~10/day
- Ensure uplink was successful (confirmed)
- Check downlink is scheduled in TTN console

---

# References

## Papers

1. Ramadan, M. N. A., et al. (2025). "Federated learning and TinyML on IoT edge devices: Challenges, advances, and future directions." *ICT Express*, 11(4), 754-768.

2. Torres Sanchez, O., et al. (2024). "Federated Learning framework for LoRaWAN-enabled IIoT communication: A case study." *arXiv:2410.11612*.

3. Puppala, S., et al. (2025). "A Comprehensive Survey of Federated Learning for Edge AI: Recent Trends and Future Directions." *Preprints*, 202512.0118.v1.

## Documentation

- Arduino MKR WAN 1310: https://docs.arduino.cc/hardware/mkr-wan-1300/
- TensorFlow Lite Micro: https://www.tensorflow.org/lite/microcontrollers
- The Things Network: https://www.thethingsnetwork.org/docs/

## Libraries

- MKRWAN: https://github.com/arduino-libraries/MKRWAN
- Arduino_TensorFlowLite: https://github.com/tensorflow/tflite-micro-arduino-examples

---

# License

MIT License

Copyright (c) 2026 Pratik Khadka

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software.

---

**End of Documentation**
