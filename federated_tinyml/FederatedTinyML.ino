/*
 * FederatedTinyML.ino
 * 
 * Federated Learning + TinyML for MKR WAN 1310
 * 
 * This sketch:
 * - Reads environmental sensors (BME280, SCD4x, SPS30)
 * - Runs TinyML inference to predict link state (good/degraded/poor)
 * - Stores samples in a circular buffer for local training
 * - Performs local training and computes model weight deltas
 * - Sends only model updates via LoRaWAN (not raw data)
 * - Receives global model updates via downlink
 * - Uses event-driven transmission based on predictions
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

// TinyML configuration
#define TENSOR_ARENA_SIZE     8192      // 8KB tensor arena
#define NUM_FEATURES          5         // pressure, co2, temp, humidity, pm25
#define NUM_CLASSES           3         // good, degraded, poor
#define MODEL_WEIGHTS_SIZE    128       // Approximate model weight count

// Federated Learning configuration
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
struct Sample {
    float features[NUM_FEATURES];  // [pressure, co2, temp, humidity, pm25]
    uint8_t label;                 // Link state label (proxy label)
    bool valid;                    // Is this sample valid?
};

// Circular buffer for local training
Sample sampleBuffer[BUFFER_SIZE];
int bufferHead = 0;
int bufferCount = 0;

// Model weights for FL (simplified representation)
// In practice, these would be extracted from TFLite model
float localWeights[MODEL_WEIGHTS_SIZE];
float globalWeights[MODEL_WEIGHTS_SIZE];
float weightDeltas[MODEL_WEIGHTS_SIZE];

// FL state
unsigned long lastFLRound = 0;
unsigned long lastTransmission = 0;
unsigned long packetsSent = 0;
uint8_t currentLinkState = LINK_STATE_GOOD;
int currentDR = 0;

// Statistics for proxy labeling
int successfulTx = 0;
int failedTx = 0;
float pdr = 1.0;  // Packet Delivery Ratio

// ============================================================================
// NORMALIZATION PARAMETERS (from training dataset)
// ============================================================================

// Synced with the latest train_model.py export output.
const float featureMeans[NUM_FEATURES] = {1009.2605, 542.0334, 21.9533, 36.1344, 1.8987};
const float featureStds[NUM_FEATURES] = {30.9770, 132.9057, 2.9012, 6.6737, 2.3336};

// ============================================================================
// FUNCTION DECLARATIONS
// ============================================================================

void initSensors();
void initLoRaWAN();
void initTinyML();
void readSensors(float* features);
uint8_t runInference(float* features);
void normalizeFeatures(float* features);
void addSampleToBuffer(float* features, uint8_t label);
void localTraining();
void computeWeightDeltas();
void sendModelUpdate();
void receiveGlobalModel();
uint8_t computeProxyLabel();
void eventDrivenTransmission(uint8_t linkState);
int argmax(float* arr, int size);

// ============================================================================
// SETUP
// ============================================================================

void setup() {
    Serial.begin(115200);
    delay(2000);  // Wait for serial
    
    Serial.println("===========================================");
    Serial.println("Federated TinyML for MKR WAN 1310");
    Serial.println("Master's Thesis Project");
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
    // STEP 1: Read sensor data
    // -------------------------------------------------------------------------
    float features[NUM_FEATURES];
    readSensors(features);
    
    Serial.println("\n--- Sensor Readings ---");
    Serial.print("Pressure: "); Serial.print(features[0]); Serial.println(" hPa");
    Serial.print("CO2: "); Serial.print(features[1]); Serial.println(" ppm");
    Serial.print("Temperature: "); Serial.print(features[2]); Serial.println(" °C");
    Serial.print("Humidity: "); Serial.print(features[3]); Serial.println(" %");
    Serial.print("PM2.5: "); Serial.print(features[4]); Serial.println(" µg/m³");
    
    // -------------------------------------------------------------------------
    // STEP 2: Normalize features for inference
    // -------------------------------------------------------------------------
    float normalizedFeatures[NUM_FEATURES];
    memcpy(normalizedFeatures, features, sizeof(features));
    normalizeFeatures(normalizedFeatures);
    
    // -------------------------------------------------------------------------
    // STEP 3: Run TinyML inference to predict link state
    // -------------------------------------------------------------------------
    currentLinkState = runInference(normalizedFeatures);
    
    Serial.println("\n--- TinyML Prediction ---");
    Serial.print("Predicted Link State: ");
    switch(currentLinkState) {
        case LINK_STATE_GOOD:     Serial.println("GOOD"); break;
        case LINK_STATE_DEGRADED: Serial.println("DEGRADED"); break;
        case LINK_STATE_POOR:     Serial.println("POOR"); break;
    }
    
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

uint8_t runInference(float* features) {
    // Copy normalized features to input tensor
    for (int i = 0; i < NUM_FEATURES; i++) {
        input->data.f[i] = features[i];
    }
    
    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("[ERROR] Inference failed!");
        return LINK_STATE_GOOD;  // Default fallback
    }
    
    // Get prediction (argmax of output)
    float predictions[NUM_CLASSES];
    for (int i = 0; i < NUM_CLASSES; i++) {
        predictions[i] = output->data.f[i];
    }
    
    // Debug output
    Serial.print("  Probabilities: [");
    for (int i = 0; i < NUM_CLASSES; i++) {
        Serial.print(predictions[i], 4);
        if (i < NUM_CLASSES - 1) Serial.print(", ");
    }
    Serial.println("]");
    
    return argmax(predictions, NUM_CLASSES);
}

int argmax(float* arr, int size) {
    int maxIdx = 0;
    float maxVal = arr[0];
    for (int i = 1; i < size; i++) {
        if (arr[i] > maxVal) {
            maxVal = arr[i];
            maxIdx = i;
        }
    }
    return maxIdx;
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
     * LOCAL TRAINING (Simplified for MCU)
     * 
     * On a real implementation, you would:
     * 1. Extract weights from TFLite model (complex on MCU)
     * 2. Run forward/backward pass on buffered samples
     * 3. Update weights using SGD
     * 
     * For MKR WAN 1310 with limited resources:
     * - Use quantization-aware training
     * - Very small batch sizes (1-4)
     * - Few epochs (1-3)
     * 
     * Here we simulate the process by computing gradients
     * based on prediction errors (simplified approach)
     */
    
    Serial.println("[FL] Starting local training...");
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
     * Send model weight deltas to server via LoRaWAN uplink
     * 
     * Payload structure (compressed):
     * - Byte 0: Message type (0x01 = model update)
     * - Byte 1: Model version
     * - Bytes 2-3: Number of weights
     * - Bytes 4+: Quantized weight deltas (8-bit each)
     * 
     * For larger models, use compression (RLE, Huffman)
     * and split across multiple uplinks
     */
    
    Serial.println("[FL] Sending model update...");
    
    // Prepare payload
    uint8_t payload[51];  // Max LoRaWAN payload at DR0
    payload[0] = 0x01;    // Message type: model update
    payload[1] = 0x01;    // Model version
    
    // Quantize weight deltas to 8-bit values
    int numWeightsToSend = min(MODEL_WEIGHTS_SIZE, 48);
    payload[2] = (numWeightsToSend >> 8) & 0xFF;
    payload[3] = numWeightsToSend & 0xFF;
    
    for (int i = 0; i < numWeightsToSend; i++) {
        // Quantize delta to int8 (-128 to 127)
        int8_t quantized = (int8_t)constrain(weightDeltas[i] * 127, -128, 127);
        payload[4 + i] = (uint8_t)quantized;
    }
    
    // Send via LoRaWAN
    modem.beginPacket();
    modem.write(payload, 4 + numWeightsToSend);
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
