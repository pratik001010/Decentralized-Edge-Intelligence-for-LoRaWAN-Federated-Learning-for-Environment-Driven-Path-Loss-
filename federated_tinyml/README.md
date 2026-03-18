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

## Dataset Audit Summary

This section summarizes the full profiling and consistency checks performed on:

- `1.unsorted_combined_measurements_data.csv`
- `2.aggregated_measurements_data.csv`

### File-level comparison

| Property | Unsorted raw file | Aggregated file |
|----------|-------------------|-----------------|
| Path | `1.unsorted_combined_measurements_data.csv` | `2.aggregated_measurements_data.csv` |
| Rows | 2,313,903 | 1,715,869 |
| Columns | 81 | 20 |
| Size | 1,952,502,329 bytes | 297,698,840 bytes |
| Character | TTN-style raw export with metadata | Curated ML-ready table |

### Key differences

1. **Schema and purpose**
  - Unsorted file contains expanded TTN metadata (`rx_metadata_*`, IDs, payload internals, network fields).
  - Aggregated file contains compact modeling columns (`co2`, `humidity`, `pm25`, `pressure`, `temperature`, `rssi`, `snr`, `distance`, `exp_pl`, etc.).

2. **Device identity semantics**
  - In unsorted raw, `device_id` is not the per-node identity (1 unique value).
  - True per-node identity in raw data is `end_device_ids_device_id` (7 unique values).
  - Aggregated file uses normalized device IDs (`ED0` to `ED5`, 6 devices).

3. **Duplicates**
  - Raw unsorted has 1,032 duplicate rows on practical key `[time, end_device_ids_device_id, uplink_message_f_cnt]`.
  - Aggregated has 0 duplicates on `[time, device_id, f_count]`.

4. **Ordering**
  - Both files cover the same overall period.
  - Aggregated file is not globally monotonic by time, so explicit sort is required before training splits.

### Data quality findings

1. **Pressure scale issue (both files)**
  - Stored pressure values are mostly around ~299 to ~342 (not realistic hPa).
  - Corrected pressure is: `pressure_hPa = pressure * 3.125`.
  - After scaling, 99.998% of aggregated rows are in 800 to 1200 hPa.

2. **Deterministic anomalous rows in aggregated file**
  - Total anomalous rows: 33
  - Pattern A (19 rows): `co2=21547`, `humidity=156.65`, `temperature=174.90`, `pressure=3.21`, `pm25=33.93`
  - Pattern B (2 rows): `co2=16724`, `humidity=210.53`, `temperature=110.76`, `pressure=317.45`, `pm25=125.57`
  - Pattern C (12 rows): `co2=0`, `humidity=0`, `temperature=0`, `pressure=508.90`, `pm25=0`
  - Recommendation: drop all 33 rows (do not impute).

3. **Missing values**
  - Aggregated: `snr` missing in 1,427 rows, `f_count` missing in 19 rows.
  - Raw unsorted: `snr` missing in 2,044 rows, `uplink_message_f_cnt` missing in 20 rows.
  - Time parse failures: 0 (aggregated), 2 (unsorted).

### Time coverage

- Start: `2024-09-26 11:00:52.541686+00:00`
- End: `2025-05-22 14:56:11.322763+00:00`

### Mandatory preprocessing rules for model training

1. Use `2.aggregated_measurements_data.csv` as the primary training source.
2. Drop the 33 known bad rows by exact signature checks.
3. Apply pressure correction: `pressure = pressure * 3.125`.
4. Keep only rows with non-null `snr` and `f_count` for supervised link-quality labels.
5. Sort by `time` before train/validation/test splitting.
6. Log row counts before/after each cleaning step for reproducibility.

### Recommended usage split

- Use `2.aggregated_measurements_data.csv` for training/inference pipeline input.
- Keep `1.unsorted_combined_measurements_data.csv` for auditability, trace-back, and reconstruction checks.

---

## 📄 License

MIT License - See LICENSE file

---

## 📚 References

1. Ramadan et al. (2025). "Federated learning and TinyML on IoT edge devices"
2. Torres Sanchez et al. (2024). "Federated Learning framework for LoRaWAN-enabled IIoT"
3. TensorFlow Lite Micro documentation
4. The Things Network documentation
