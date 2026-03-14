# Federated TinyML for LoRaWAN Edge Intelligence

## Master's Thesis Project
**Author:** Pratik Khadka

This project implements a federated learning system for LoRaWAN-connected edge devices (Arduino MKR WAN 1310) with TinyML capabilities.

---

## 📁 Project Structure

```
FederatedTinyML/
├── FederatedTinyML.ino    # Main Arduino sketch (device code)
├── model.h                # TFLite Micro model as C array
├── train_model.py         # Python script for model training
├── fl_server.py           # Federated learning server/orchestrator
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## 🔧 Hardware Requirements

- **6x Arduino MKR WAN 1310** (LoRaWAN-enabled MCU)
- **Sensors per node:**
  - BME280 (pressure, temperature, humidity)
  - SCD4x (CO2, temperature, humidity)
  - SPS30 (PM2.5 particulate matter)
- **LoRaWAN Gateway** connected to TTN

---

## 📚 Arduino Libraries (Install via Library Manager)

| Library | Version | Purpose |
|---------|---------|---------|
| MKRWAN | 1.1.0+ | LoRaWAN communication |
| Adafruit BME280 | 2.2.2+ | BME280 sensor |
| Sensirion I2C SCD4x | 0.4.0+ | SCD4x CO2 sensor |
| Sensirion SPS30 | 1.2.0+ | SPS30 PM sensor |
| Arduino_TensorFlowLite | 2.4.0+ | TFLite Micro inference |

### Install libraries via Arduino IDE:
1. Go to **Sketch** → **Include Library** → **Manage Libraries**
2. Search and install each library listed above

### Or via Arduino CLI:
```bash
arduino-cli lib install "MKRWAN"
arduino-cli lib install "Adafruit BME280 Library"
arduino-cli lib install "Sensirion I2C SCD4x"
arduino-cli lib install "sensirion-sps"
arduino-cli lib install "Arduino_TensorFlowLite"
```

---

## 🚀 Quick Start

### Step 1: Train the Model (PC)

```bash
cd FederatedTinyML
pip install -r requirements.txt
python train_model.py
```

This generates:
- `model_output/model.keras` - Keras model
- `model_output/model.tflite` - TFLite model
- `model_output/model.h` - C header for Arduino

### Step 2: Copy Model to Arduino

Copy `model_output/model.h` to the `FederatedTinyML/` folder (replace the placeholder).

### Step 3: Configure LoRaWAN Credentials

Edit `FederatedTinyML.ino`:
```cpp
String appEui = "YOUR_APP_EUI";
String appKey = "YOUR_APP_KEY";
```

### Step 4: Upload to MKR WAN 1310

1. Open `FederatedTinyML.ino` in Arduino IDE
2. Select **Board:** Arduino MKR WAN 1310
3. Select your **Port**
4. Click **Upload**

### Step 5: Start FL Server

```bash
python fl_server.py
```

Server runs at `http://localhost:5000`

### Step 6: Configure TTN Webhook

In TTN Console:
1. Go to your Application → Integrations → Webhooks
2. Add webhook with URL: `http://your-server:5000/uplink`
3. Enable uplink messages

---

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FL Server (Cloud/Edge)                    │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │ Receive     │───▶│ FedAvg     │───▶│ Compress       │  │
│  │ Updates     │    │ Aggregation │    │ Global Model   │  │
│  └─────────────┘    └─────────────┘    └─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
        ▲  │                                    │
        │  │ Model Updates (Uplink)            │ Global Model (Downlink)
        │  │                                    ▼
┌───────┴──┴────────────────────────────────────────────────┐
│                   LoRaWAN Network (TTN)                     │
└─────────────────────────────────────────────────────────────┘
        ▲                   ▲                   ▲
        │                   │                   │
┌───────┴────┐      ┌───────┴────┐      ┌───────┴────┐
│ MKR WAN 1  │      │ MKR WAN 2  │      │ MKR WAN 6  │
│            │      │            │      │            │
│ ┌────────┐ │      │ ┌────────┐ │      │ ┌────────┐ │
│ │TinyML  │ │      │ │TinyML  │ │      │ │TinyML  │ │
│ │Model   │ │      │ │Model   │ │      │ │Model   │ │
│ └────────┘ │      │ └────────┘ │      │ └────────┘ │
│ ┌────────┐ │      │ ┌────────┐ │      │ ┌────────┐ │
│ │Sensors │ │      │ │Sensors │ │      │ │Sensors │ │
│ └────────┘ │      │ └────────┘ │      │ └────────┘ │
└────────────┘      └────────────┘      └────────────┘
  Stairwell            Corridor            Kitchen
```

---

## 🔄 Federated Learning Workflow

1. **Local Inference:** Each node predicts link state from sensor data
2. **Data Buffering:** Recent samples stored in circular buffer
3. **Local Training:** Few epochs of training on buffered data
4. **Delta Computation:** Calculate weight differences from global model
5. **Uplink Update:** Send compressed deltas via LoRaWAN
6. **Server Aggregation:** FedAvg combines updates from all nodes
7. **Downlink Model:** Compressed global model sent back (rare, ≤10/day)

---

## 📡 LoRaWAN Payload Formats

### Uplink: Model Update (Type 0x01)
```
Byte 0:     0x01 (message type)
Byte 1:     Model version
Bytes 2-3:  Number of weights (uint16, big-endian)
Bytes 4+:   Quantized weight deltas (int8)
```

### Downlink: Global Model (Type 0x02)
```
Byte 0:     0x02 (message type)
Byte 1:     Model version
Bytes 2-3:  Number of weights (uint16, big-endian)
Bytes 4+:   Quantized global weights (int8)
```

### Uplink: Status (Type 0x00)
```
Byte 0:     0x00 (message type)
Byte 1:     Link state (0=good, 1=degraded, 2=poor)
Byte 2:     PDR (0-100%)
Byte 3:     Current data rate
Bytes 4-7:  Packet count (uint32, big-endian)
```

---

## 🧠 TinyML Model Specifications

| Property | Value |
|----------|-------|
| Architecture | Dense(8, ReLU) → Dense(3, Softmax) |
| Input | 5 float32 features |
| Output | 3 float32 probabilities |
| Parameters | ~67 |
| Size (quantized) | ~500 bytes |
| Tensor Arena | 8 KB |
| Inference Time | ~1 ms |

### Features (Inputs)
1. Pressure (hPa)
2. CO2 (ppm)
3. Temperature (°C)
4. Humidity (%)
5. PM2.5 (µg/m³)

### Classes (Outputs)
- 0: Good link state
- 1: Degraded link state
- 2: Poor link state

---

## ⚙️ Configuration Parameters

### Arduino (`FederatedTinyML.ino`)
```cpp
#define BUFFER_SIZE           32        // Sample buffer size
#define LOCAL_EPOCHS          3         // Training epochs per round
#define LOCAL_BATCH_SIZE      4         // Batch size for training
#define FL_ROUND_INTERVAL     86400000  // 24 hours between FL rounds
#define NORMAL_TX_INTERVAL    300000    // 5 min normal transmission
#define URGENT_TX_INTERVAL    60000     // 1 min urgent transmission
```

### Server (`fl_server.py`)
```python
FL_CONFIG = {
    "num_clients": 6,
    "min_clients_per_round": 3,
    "fl_round_duration_hours": 24,
    "model_weights_size": 128,
}
```

---

## 📈 Expected Performance

Based on literature and similar deployments:

| Metric | Centralized | Federated |
|--------|-------------|-----------|
| Accuracy | ~93% | ~92% |
| F1 Score | ~94% | ~93% |
| Bandwidth | 1.3M rows | ~500 bytes/round |
| Privacy | Data leaves device | Data stays local |
| Adaptation | Static | Continuous |

---

## 🐛 Troubleshooting

### Model doesn't fit in memory
- Reduce `TENSOR_ARENA_SIZE` if using less than 8KB
- Use smaller model (fewer hidden units)
- Ensure quantization is applied

### LoRaWAN join fails
- Check appEui and appKey match TTN settings
- Verify gateway is online
- Check antenna connection

### Sensors not reading
- Verify I2C connections (SDA to SDA, SCL to SCL)
- Check sensor addresses (BME280: 0x77 or 0x76)
- Run I2C scanner sketch to detect devices

### FL not converging
- Increase `MIN_SAMPLES_FOR_TRAIN`
- Reduce learning rate in local training
- Ensure proxy labels are accurate

---

## 📄 License

MIT License - See LICENSE file

---

## 📚 References

1. Ramadan et al. (2025). "Federated learning and TinyML on IoT edge devices"
2. Torres Sanchez et al. (2024). "Federated Learning framework for LoRaWAN-enabled IIoT"
3. TensorFlow Lite Micro documentation
4. The Things Network documentation
