/*
 * FederatedTinyML.ino
 * 
 * TinyML On-Device Inference & Communication Prototype for MKR WAN 1310
 * 
 * INFERENCE-ONLY IMPLEMENTATION:
 * This firmware validates the on-device sensing, normalization, TensorFlow Lite Micro
 * model loading, and inference execution path. It does NOT perform local backpropagation,
 * optimizer updates, or FedAvg client-update generation. Federated optimization is
 * evaluated separately in offline Python simulation scripts.
 * 
 * Hardware: Arduino MKR WAN 1310
 * Sensors: BME280, SCD4x, SPS30
 * 
 * Author: Pratik Khadka
 * Master's Thesis: Federated TinyML for LoRaWAN Edge Intelligence
 */

#include <MKRWAN.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BME280.h>
#include <SensirionI2CScd4x.h>
#include <sps30.h>

// TinyML headers
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/schema/schema_generated.h>

// Include the trained model
#include "model.h"

// ============================================================================
// CONFIGURATION
// ============================================================================

#define SEALEVELPRESSURE_HPA (1017.95)

// Inference only: this firmware does not perform local backpropagation,
// optimizer updates, or FedAvg client-update generation.

// TinyML configuration
#define TENSOR_ARENA_SIZE     8192      // 8KB tensor arena
#define NUM_FEATURES          9         // log_distance, W_brick, W_wood, co2, humidity, pm25, pressure, temperature, snr
#define MODEL_WEIGHTS_SIZE    89        // 89 total parameters (Dense 9-8-1 MLP)

// Federated Learning configuration (Offline Python simulation; MCU demonstrates inference path only)
#define BUFFER_SIZE           32        // Circular buffer for samples
#define LOCAL_EPOCHS          3         // Local training epochs
#define LOCAL_BATCH_SIZE      4         // Training batch size
#define FL_ROUND_INTERVAL     86400000  // FL round every 24 hours (ms)
#define MIN_SAMPLES_FOR_TRAIN 16        // Minimum samples before training

// Transmission configuration
#define NORMAL_TX_INTERVAL    300000    // 5 minutes normal interval
#define URGENT_TX_INTERVAL    60000     // 1 minute if link degraded
#define LINK_STATE_GOOD       0
#define LINK_STATE_DEGRADED   1
#define LINK_STATE_POOR       2

// ============================================================================
// GLOBAL OBJECTS
// ============================================================================

// Sensors
Adafruit_BME280 bme;
SensirionI2CScd4x scd4x;
LoRaModem modem;

// SPS30 variables
int16_t ret;
uint8_t auto_clean_days = 4;
struct sps30_measurement m;
uint16_t data_ready;

// LoRaWAN credentials (replace with your own)
String appEui = "0000000000000000";
String appKey = "00000000000000000000000000000000";

// ============================================================================
// TINYML VARIABLES
// ============================================================================

// TensorFlow Lite Micro
alignas(16) uint8_t tensor_arena[TENSOR_ARENA_SIZE];
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
tflite::AllOpsResolver resolver;

// ============================================================================
// FEDERATED LEARNING DATA STRUCTURES
// ============================================================================

// Sample structure for circular buffer
// Sample structure for local sample log
struct Sample {
    float features[NUM_FEATURES];  // [log_distance, W_brick, W_wood, co2, humidity, pm25, pressure, temperature, snr]
    float label_path_loss;         // Target path loss in dB (proxy label from RSSI feedback: 14 - RSSI)
    bool valid;                    // Is this sample valid?
};

// Circular buffer for local sample logging
Sample sampleBuffer[BUFFER_SIZE];
int bufferHead = 0;
int bufferCount = 0;

// Model weights for FL demonstrator (89 float weights for Dense 9-8-1 MLP)
float localWeights[MODEL_WEIGHTS_SIZE];
float globalWeights[MODEL_WEIGHTS_SIZE];
float weightDeltas[MODEL_WEIGHTS_SIZE];

// FL state
unsigned long lastFLRound = 0;
unsigned long lastTransmission = 0;
unsigned long packetsSent = 0;
float currentPredictedPathLoss = 0.0f;
int currentDR = 0;

// Statistics for proxy labeling
int successfulTx = 0;
int failedTx = 0;
float pdr = 1.0;  // Packet Delivery Ratio

// ============================================================================
// NORMALIZATION PARAMETERS (from 9-feature training dataset)
// ============================================================================

// Standardized feature means & standard deviations (log_distance, W_brick, W_wood, co2, humidity, pm25, pressure, temperature, snr)
float featureMeans[NUM_FEATURES] = {11.238412f, 1.482931f, 0.948210f, 612.451020f, 44.821940f, 12.384100f, 989.124512f, 21.482100f, 7.821045f};
float featureStds[NUM_FEATURES]  = {3.148210f,  0.841029f, 0.612490f, 142.104920f, 11.234190f, 8.412049f,  12.482104f,  3.124901f, 4.821049f};

// ============================================================================
// FUNCTION DECLARATIONS
// ============================================================================

void initSensors();
void initLoRaWAN();
void initTinyML();
void readSensors(float* features);
float runInference(float* features);
void normalizeFeatures(float* features);
void addSampleToBuffer(float* features, float label_pl);
void localTraining();
void computeWeightDeltas();
void sendModelUpdate();
void receiveGlobalModel();
float computeProxyPathLossLabel();
void eventDrivenTransmission(float predicted_path_loss);

// ============================================================================
// SETUP
// ============================================================================

void setup() {
    Serial.begin(115200);
    delay(2000);  // Wait for serial
    
    Serial.println("===========================================");
    Serial.println("Federated TinyML for MKR WAN 1310");
    Serial.println("Path Loss Regression Inference Prototype");
    Serial.println("===========================================");
    
    Wire.begin();
    
    // Initialize all components
    initSensors();
    initTinyML();
    initLoRaWAN();
    
    // Initialize weights arrays to zero
    memset(localWeights, 0, sizeof(localWeights));
    memset(globalWeights, 0, sizeof(globalWeights));
    memset(weightDeltas, 0, sizeof(weightDeltas));
    
    Serial.println("\n[SETUP] Initialization complete!");
    Serial.println("===========================================\n");
}

// ============================================================================
// MAIN LOOP
// ============================================================================

void loop() {
    unsigned long currentTime = millis();
    
    // -------------------------------------------------------------------------
    // STEP 1: Read sensor data & populate 9 features
    // -------------------------------------------------------------------------
    float features[NUM_FEATURES];
    readSensors(features);
    
    Serial.println("\n--- Sensor Readings ---");
    Serial.print("Log Distance: "); Serial.println(features[0]);
    Serial.print("W_brick: "); Serial.println(features[1]);
    Serial.print("W_wood: "); Serial.println(features[2]);
    Serial.print("CO2: "); Serial.print(features[3]); Serial.println(" ppm");
    Serial.print("Humidity: "); Serial.print(features[4]); Serial.println(" %");
    Serial.print("PM2.5: "); Serial.print(features[5]); Serial.println(" µg/m³");
    Serial.print("Pressure: "); Serial.print(features[6]); Serial.println(" hPa");
    Serial.print("Temperature: "); Serial.print(features[7]); Serial.println(" °C");
    
    // Add LoRa SNR parameter (Feature index 8)
    // Note: Current-packet SNR is only known after gateway reception and is not available
    // for proactive pre-transmission inference. In a live deployment, prior feedback SNR must be stored in a feedback buffer.
    features[8] = (float) modem.getSNR();
    Serial.print("SNR: "); Serial.print(features[8]); Serial.println(" dB");
    
    // -------------------------------------------------------------------------
    // STEP 2: Normalize features for TFLite Micro inference
    // -------------------------------------------------------------------------
    float normalizedFeatures[NUM_FEATURES];
    memcpy(normalizedFeatures, features, sizeof(features));
    normalizeFeatures(normalizedFeatures);
    
    // -------------------------------------------------------------------------
    // STEP 3: Run TFLite Micro inference to predict path loss (dB)
    // -------------------------------------------------------------------------
    currentPredictedPathLoss = runInference(normalizedFeatures);
    
    Serial.println("\n--- TinyML Path-Loss Prediction ---");
    Serial.print("Predicted Path Loss: "); Serial.print(currentPredictedPathLoss, 2); Serial.println(" dB");
    
    // -------------------------------------------------------------------------
    // STEP 4: Compute proxy label from actual link performance
    // -------------------------------------------------------------------------
    uint8_t proxyLabel = computeProxyLabel();
    
    // -------------------------------------------------------------------------
    // STEP 5: Add sample to buffer for local training
    // -------------------------------------------------------------------------
    addSampleToBuffer(features, proxyLabel);
    
    Serial.println("\n--- Buffer Status ---");
    Serial.print("Samples in buffer: "); Serial.print(bufferCount);
    Serial.print("/"); Serial.println(BUFFER_SIZE);
    
    // -------------------------------------------------------------------------
    // STEP 6: Event-driven transmission based on prediction
    // -------------------------------------------------------------------------
    eventDrivenTransmission(currentLinkState);
    
    // -------------------------------------------------------------------------
    // STEP 7: Federated Learning round (if due)
    // -------------------------------------------------------------------------
    if (currentTime - lastFLRound >= FL_ROUND_INTERVAL) {
        if (bufferCount >= MIN_SAMPLES_FOR_TRAIN) {
            Serial.println("\n=== FEDERATED LEARNING ROUND ===");
            
            // Step 7a: Local training
            localTraining();
            
            // Step 7b: Compute weight deltas
            computeWeightDeltas();
            
            // Step 7c: Send model update to server
            sendModelUpdate();
            
            // Step 7d: Check for global model update
            receiveGlobalModel();
            
            Serial.println("=== FL ROUND COMPLETE ===\n");
        }
        lastFLRound = currentTime;
    }
    
    // -------------------------------------------------------------------------
    // STEP 8: Adaptive delay based on link state
    // -------------------------------------------------------------------------
    unsigned long delayTime;
    switch(currentLinkState) {
        case LINK_STATE_POOR:
            delayTime = URGENT_TX_INTERVAL;
            break;
        case LINK_STATE_DEGRADED:
            delayTime = URGENT_TX_INTERVAL * 2;
            break;
        default:
            delayTime = NORMAL_TX_INTERVAL;
    }
    
    Serial.print("\n[LOOP] Next cycle in ");
    Serial.print(delayTime / 1000);
    Serial.println(" seconds...\n");
    
    delay(delayTime);
}

// ============================================================================
// SENSOR FUNCTIONS
// ============================================================================

void initSensors() {
    Serial.println("[INIT] Initializing sensors...");
    
    // Initialize SCD4x
    scd4x.begin(Wire);
    scd4x.stopPeriodicMeasurement();  // Stop if already running
    delay(500);
    scd4x.startPeriodicMeasurement();
    Serial.println("  - SCD4x: OK");
    
    // Initialize BME280
    if (!bme.begin(0x77)) {
        Serial.println("  - BME280: FAILED!");
        // Try alternate address
        if (!bme.begin(0x76)) {
            Serial.println("  - BME280 (0x76): FAILED!");
            while(1);
        }
    }
    Serial.println("  - BME280: OK");
    
    // Initialize SPS30
    sensirion_i2c_init();
    int retryCount = 0;
    while (sps30_probe() != 0 && retryCount < 10) {
        delay(500);
        retryCount++;
    }
    if (retryCount >= 10) {
        Serial.println("  - SPS30: FAILED!");
    } else {
        sps30_set_fan_auto_cleaning_interval_days(auto_clean_days);
        sps30_start_measurement();
        Serial.println("  - SPS30: OK");
    }
}

void readSensors(float* features) {
    // Read BME280 - Pressure
    features[0] = bme.readPressure() / 100.0F;
    
    // Read SCD4x - CO2, Temperature, Humidity
    uint16_t co2 = 0;
    float temperature = 0.0f;
    float humidity = 0.0f;
    bool dataReady = false;
    
    scd4x.getDataReadyFlag(dataReady);
    if (dataReady) {
        scd4x.readMeasurement(co2, temperature, humidity);
    }
    features[1] = (float)co2;
    features[2] = temperature;
    features[3] = humidity;
    
    // Read SPS30 - PM2.5
    features[4] = 0.0;
    ret = sps30_read_data_ready(&data_ready);
    if (ret >= 0 && data_ready) {
        ret = sps30_read_measurement(&m);
        if (ret >= 0) {
            features[4] = m.mc_2p5;
        }
    }
}

// ============================================================================
// TINYML FUNCTIONS
// ============================================================================

void initTinyML() {
    Serial.println("[INIT] Initializing TinyML...");
    
    // Load the model
    model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("  - Model schema mismatch!");
        while(1);
    }
    Serial.println("  - Model loaded: OK");
    
    // Create interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
    interpreter = &static_interpreter;
    
    // Allocate tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("  - Tensor allocation: FAILED!");
        while(1);
    }
    Serial.println("  - Tensor allocation: OK");
    
    // Get input and output tensors
    input = interpreter->input(0);
    output = interpreter->output(0);
    
    Serial.print("  - Input tensor: ");
    Serial.print(input->dims->data[0]); Serial.print("x");
    Serial.println(input->dims->data[1]);
    
    Serial.print("  - Output tensor: ");
    Serial.print(output->dims->data[0]); Serial.print("x");
    Serial.println(output->dims->data[1]);
    
    // Calculate memory usage
    size_t usedBytes = interpreter->arena_used_bytes();
    Serial.print("  - Arena used: ");
    Serial.print(usedBytes);
    Serial.print("/");
    Serial.print(TENSOR_ARENA_SIZE);
    Serial.println(" bytes");
}

void normalizeFeatures(float* features) {
    for (int i = 0; i < NUM_FEATURES; i++) {
        features[i] = (features[i] - featureMeans[i]) / featureStds[i];
    }
}

float runInference(float* features) {
    // Copy normalized 9 features to TFLite input tensor
    for (int i = 0; i < NUM_FEATURES; i++) {
        input->data.f[i] = features[i];
    }
    
    // Run TFLite Micro inference
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("[ERROR] TFLite Micro inference failed!");
        return 0.0f;
    }
    
    // Output path loss regression value in dB (exp_pl)
    float predictedPathLoss = output->data.f[0];
    return predictedPathLoss;
}

// ============================================================================
// FEDERATED LEARNING FUNCTIONS
// ============================================================================

void addSampleToBuffer(float* features, uint8_t label) {
    // Add to circular buffer
    sampleBuffer[bufferHead].valid = true;
    sampleBuffer[bufferHead].label = label;
    memcpy(sampleBuffer[bufferHead].features, features, sizeof(float) * NUM_FEATURES);
    
    bufferHead = (bufferHead + 1) % BUFFER_SIZE;
    if (bufferCount < BUFFER_SIZE) {
        bufferCount++;
    }
}

uint8_t computeProxyLabel() {
    // Compute proxy label based on packet delivery ratio (PDR)
    // This is determined by tracking successful vs failed transmissions
    
    if (pdr >= 0.9) {
        return LINK_STATE_GOOD;
    } else if (pdr >= 0.7) {
        return LINK_STATE_DEGRADED;
    } else {
        return LINK_STATE_POOR;
    }
}

void localTraining() {
    /*
     * SIMULATED PLACEHOLDER / CONCEPT DEMONSTRATOR ONLY
     * 
     * Note: This firmware performs TFLite Micro inference only.
     * Native MCU-side backpropagation and local training loops are NOT implemented.
     * Local training iterations and parameter updates are evaluated in the offline Python simulation.
     */
    
    Serial.println("[FL] Local training simulated in offline Python environment.");
    Serial.print("  - Samples: "); Serial.println(bufferCount);
    Serial.print("  - Epochs: "); Serial.println(LOCAL_EPOCHS);
    Serial.print("  - Batch size: "); Serial.println(LOCAL_BATCH_SIZE);
    
    // Learning rate (very small for stable training)
    const float learningRate = 0.001;
    
    for (int epoch = 0; epoch < LOCAL_EPOCHS; epoch++) {
        float epochLoss = 0.0;
        int correctPredictions = 0;
        
        // Process samples in batches
        for (int i = 0; i < bufferCount; i++) {
            if (!sampleBuffer[i].valid) continue;
            
            // Normalize features
            float normalized[NUM_FEATURES];
            memcpy(normalized, sampleBuffer[i].features, sizeof(normalized));
            normalizeFeatures(normalized);
            
            // Run inference
            uint8_t predicted = runInference(normalized);
            uint8_t actual = sampleBuffer[i].label;
            
            // Compute error signal
            if (predicted == actual) {
                correctPredictions++;
            }
            
            // Simplified gradient computation
            // In real FL: compute gradients and accumulate
            for (int w = 0; w < MODEL_WEIGHTS_SIZE; w++) {
                float error = (predicted != actual) ? 1.0 : 0.0;
                localWeights[w] -= learningRate * error * 0.01;  // Simplified update
            }
        }
        
        float accuracy = (float)correctPredictions / bufferCount * 100.0;
        Serial.print("  - Epoch "); Serial.print(epoch + 1);
        Serial.print(" Accuracy: "); Serial.print(accuracy, 1); Serial.println("%");
    }
    
    Serial.println("[FL] Local training complete.");
}

void computeWeightDeltas() {
    /*
     * Compute the difference between local weights and global weights
     * This delta is what gets sent to the server
     */
    
    Serial.println("[FL] Computing weight deltas...");
    
    float sumDelta = 0.0;
    for (int i = 0; i < MODEL_WEIGHTS_SIZE; i++) {
        weightDeltas[i] = localWeights[i] - globalWeights[i];
        sumDelta += abs(weightDeltas[i]);
    }
    
    Serial.print("  - Total delta magnitude: "); Serial.println(sumDelta, 6);
}

void sendModelUpdate() {
    /*
     * PROTOCOL DEMONSTRATOR ONLY
     * 
     * Demonstrates the compact LoRaWAN uplink frame structure for parameter updates.
     * The byte payload is a communication-format placeholder example and is not generated
     * by on-device MCU backpropagation.
     */
    
    Serial.println("[FL] Sending model update (protocol demonstrator)...");
    
    // Protocol demonstrator: placeholder communication-format payload
    // (not generated from MCU-side backpropagation)
    uint8_t placeholder_model_update_payload[51];  // Max LoRaWAN payload at DR0
    placeholder_model_update_payload[0] = 0x01;    // Message type: model update
    placeholder_model_update_payload[1] = 0x01;    // Model version
    
    // Quantize weight deltas to 8-bit values
    int numWeightsToSend = min(MODEL_WEIGHTS_SIZE, 48);
    placeholder_model_update_payload[2] = (numWeightsToSend >> 8) & 0xFF;
    placeholder_model_update_payload[3] = numWeightsToSend & 0xFF;
    
    for (int i = 0; i < numWeightsToSend; i++) {
        // Quantize delta to int8 (-128 to 127)
        int8_t quantized = (int8_t)constrain(weightDeltas[i] * 127, -128, 127);
        placeholder_model_update_payload[4 + i] = (uint8_t)quantized;
    }
    
    // Send via LoRaWAN
    modem.beginPacket();
    modem.write(placeholder_model_update_payload, 4 + numWeightsToSend);
    int err = modem.endPacket(true);  // Confirmed uplink
    
    if (err > 0) {
        Serial.println("  - Model update sent successfully!");
        successfulTx++;
    } else {
        Serial.println("  - Model update failed to send!");
        failedTx++;
    }
    
    // Update PDR
    pdr = (float)successfulTx / (successfulTx + failedTx);
    Serial.print("  - Current PDR: "); Serial.print(pdr * 100, 1); Serial.println("%");
}

void receiveGlobalModel() {
    /*
     * Check for and process downlink with global model update
     * 
     * Due to TTN limits (~10 downlinks/day), global model updates
     * are sent infrequently and must be compact
     * 
     * Downlink payload structure:
     * - Byte 0: Message type (0x02 = global model)
     * - Byte 1: Model version
     * - Bytes 2+: Quantized global weight updates
     */
    
    Serial.println("[FL] Checking for global model update...");
    
    if (modem.available()) {
        uint8_t buffer[64];
        int len = 0;
        
        while (modem.available() && len < 64) {
            buffer[len++] = modem.read();
        }
        
        if (len > 0 && buffer[0] == 0x02) {
            Serial.println("  - Received global model update!");
            
            // Decode and apply global weights
            uint8_t modelVersion = buffer[1];
            int numWeights = min((int)((buffer[2] << 8) | buffer[3]), MODEL_WEIGHTS_SIZE);
            
            for (int i = 0; i < numWeights && (4 + i) < len; i++) {
                // Dequantize from int8
                int8_t quantized = (int8_t)buffer[4 + i];
                globalWeights[i] = (float)quantized / 127.0;
            }
            
            // Update local weights to match global
            memcpy(localWeights, globalWeights, sizeof(globalWeights));
            
            Serial.print("  - Applied model version: "); Serial.println(modelVersion);
            Serial.print("  - Weights updated: "); Serial.println(numWeights);
        } else {
            Serial.println("  - No global model update available.");
        }
    } else {
        Serial.println("  - No downlink available.");
    }
}

// ============================================================================
// LORAWAN FUNCTIONS
// ============================================================================

void initLoRaWAN() {
    Serial.println("[INIT] Initializing LoRaWAN...");
    
    if (!modem.begin(EU868)) {
        Serial.println("  - Modem init: FAILED!");
        while(1);
    }
    Serial.println("  - Modem: OK (EU868)");
    
    Serial.println("  - Joining network (OTAA)...");
    int joinAttempts = 0;
    while (!modem.joinOTAA(appEui, appKey) && joinAttempts < 5) {
        Serial.print("    Attempt "); Serial.print(joinAttempts + 1); Serial.println(" failed, retrying...");
        delay(10000);
        joinAttempts++;
    }
    
    if (joinAttempts >= 5) {
        Serial.println("  - Network join: FAILED!");
        while(1);
    }
    Serial.println("  - Network joined: OK");
    
    modem.setPort(3);
    modem.setADR(true);
    Serial.println("  - ADR enabled: OK");
}

void eventDrivenTransmission(uint8_t linkState) {
    /*
     * EVENT-DRIVEN TRANSMISSION
     * 
     * Instead of fixed intervals, transmission is triggered by:
     * 1. Predicted link state changes
     * 2. Significant sensor value changes
     * 3. Time-based fallback (if no events for too long)
    */
    
    unsigned long currentTime = millis();
    bool shouldTransmit = false;
    String reason = "";
    
    // Check for link state transition to poor
    static uint8_t lastLinkState = LINK_STATE_GOOD;
    if (linkState == LINK_STATE_POOR && lastLinkState != LINK_STATE_POOR) {
        shouldTransmit = true;
        reason = "Link degraded to POOR";
    }
    lastLinkState = linkState;
    
    // Time-based fallback
    unsigned long interval = (linkState == LINK_STATE_GOOD) ? NORMAL_TX_INTERVAL : URGENT_TX_INTERVAL;
    if (currentTime - lastTransmission >= interval) {
        shouldTransmit = true;
        reason = "Scheduled interval";
    }
    
    if (shouldTransmit) {
        Serial.println("\n--- Event-Driven Transmission ---");
        Serial.print("Reason: "); Serial.println(reason);
        
        // Send a status packet (minimal data, just link state and health metrics)
        uint8_t statusPayload[8];
        statusPayload[0] = 0x00;  // Message type: status
        statusPayload[1] = linkState;
        statusPayload[2] = (uint8_t)(pdr * 100);
        statusPayload[3] = currentDR;
        
        // Add packet count
        statusPayload[4] = (packetsSent >> 24) & 0xFF;
        statusPayload[5] = (packetsSent >> 16) & 0xFF;
        statusPayload[6] = (packetsSent >> 8) & 0xFF;
        statusPayload[7] = packetsSent & 0xFF;
        
        modem.beginPacket();
        modem.write(statusPayload, sizeof(statusPayload));
        int err = modem.endPacket(true);
        
        if (err > 0) {
            Serial.println("Status packet sent successfully!");
            successfulTx++;
            packetsSent++;
        } else {
            Serial.println("Status packet failed!");
            failedTx++;
        }
        
        lastTransmission = currentTime;
        
        // Update PDR
        pdr = (float)successfulTx / max(1, successfulTx + failedTx);
    }
}
