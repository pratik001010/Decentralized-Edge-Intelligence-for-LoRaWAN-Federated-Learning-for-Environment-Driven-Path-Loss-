"""
train_model.py

TinyML Model Training for Federated Learning on MKR WAN 1310

This script:
1. Trains a compact neural network for link state classification
2. Converts to TensorFlow Lite with int8 quantization
3. Generates C header file for Arduino deployment

Features (inputs):
- Pressure (hPa)
- CO2 (ppm)
- Temperature (°C)
- Humidity (%)
- PM2.5 (µg/m³)

Labels (outputs):
- 0: Good link state
- 1: Degraded link state
- 2: Poor link state

Author: Pratik Khadka
Master's Thesis: Federated TinyML for LoRaWAN Edge Intelligence
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import struct

# ============================================================================
# CONFIGURATION
# ============================================================================

NUM_FEATURES = 5
NUM_CLASSES = 3
HIDDEN_UNITS = 8
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001

MODEL_NAME = "link_state_classifier"
OUTPUT_DIR = "model_output"

# ============================================================================
# SYNTHETIC DATA GENERATION (Replace with your actual dataset)
# ============================================================================

def generate_synthetic_data(num_samples=5000):
    """
    Generate synthetic sensor data for testing.
    Replace this with your actual 1.3M row dataset.
    """
    np.random.seed(42)
    
    # Feature ranges based on typical indoor environment
    pressure = np.random.normal(1013, 10, num_samples)      # hPa
    co2 = np.random.normal(800, 300, num_samples)           # ppm
    temperature = np.random.normal(22, 5, num_samples)      # °C
    humidity = np.random.normal(50, 15, num_samples)        # %
    pm25 = np.random.exponential(15, num_samples)           # µg/m³
    
    # Combine features
    X = np.column_stack([pressure, co2, temperature, humidity, pm25])
    
    # Generate labels based on environmental conditions
    # This simulates how link quality might relate to environment
    y = np.zeros(num_samples, dtype=np.int32)
    
    for i in range(num_samples):
        # Poor link conditions: high humidity + high CO2 + high PM2.5
        poor_score = (humidity[i] > 70) + (co2[i] > 1200) + (pm25[i] > 35)
        # Degraded conditions: moderate values
        degraded_score = (60 < humidity[i] <= 70) + (800 < co2[i] <= 1200) + (20 < pm25[i] <= 35)
        
        if poor_score >= 2:
            y[i] = 2  # Poor
        elif degraded_score >= 2 or poor_score == 1:
            y[i] = 1  # Degraded
        else:
            y[i] = 0  # Good
    
    return X, y


def load_real_data(filepath=None):
    """
    Load your actual dataset from the previous project.
    
    Expected CSV format:
    pressure,co2,temperature,humidity,pm25,rssi,snr,link_state
    
    Returns:
        X: Feature array (num_samples, 5)
        y: Label array (num_samples,)
    """
    if filepath is None or not os.path.exists(filepath):
        print("No real data found, using synthetic data...")
        return generate_synthetic_data()
    
    import pandas as pd
    df = pd.read_csv(filepath)
    
    # Extract features
    feature_cols = ['pressure', 'co2', 'temperature', 'humidity', 'pm25']
    X = df[feature_cols].values
    
    # If you have RSSI/SNR, compute link state from them
    if 'link_state' in df.columns:
        y = df['link_state'].values
    elif 'rssi' in df.columns and 'snr' in df.columns:
        # Derive link state from RSSI and SNR
        y = np.zeros(len(df), dtype=np.int32)
        for i in range(len(df)):
            rssi = df.iloc[i]['rssi']
            snr = df.iloc[i]['snr']
            if rssi > -100 and snr > 0:
                y[i] = 0  # Good
            elif rssi > -115 and snr > -5:
                y[i] = 1  # Degraded
            else:
                y[i] = 2  # Poor
    else:
        raise ValueError("Dataset must have 'link_state' or 'rssi'/'snr' columns")
    
    return X, y


# ============================================================================
# MODEL DEFINITION
# ============================================================================

def create_model():
    """
    Create a TinyML-compatible neural network.
    
    Architecture designed for MKR WAN 1310:
    - Very small (< 10KB weights)
    - Simple operations (Dense + ReLU)
    - Quantization-friendly
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(NUM_FEATURES,), name='input'),
        keras.layers.Dense(HIDDEN_UNITS, activation='relu', name='hidden'),
        keras.layers.Dense(NUM_CLASSES, activation='softmax', name='output')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ============================================================================
# CONVERSION TO TFLITE
# ============================================================================

def convert_to_tflite(model, X_train, output_path):
    """
    Convert Keras model to TensorFlow Lite with int8 quantization.
    """
    def representative_dataset():
        """Generate representative data for quantization."""
        for i in range(min(500, len(X_train))):
            yield [X_train[i:i+1].astype(np.float32)]
    
    # Convert with full integer quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.float32   # Keep float32 input for easier use
    converter.inference_output_type = tf.float32  # Keep float32 output
    
    tflite_model = converter.convert()
    
    # Save TFLite model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved: {output_path}")
    print(f"Model size: {len(tflite_model)} bytes")
    
    return tflite_model


def generate_c_header(tflite_model, output_path):
    """
    Convert TFLite model to C header file for Arduino.
    """
    with open(output_path, 'w') as f:
        f.write("/*\n")
        f.write(" * model.h\n")
        f.write(" * \n")
        f.write(" * Auto-generated TensorFlow Lite Micro model\n")
        f.write(" * Link State Classification for MKR WAN 1310\n")
        f.write(" * \n")
        f.write(f" * Model size: {len(tflite_model)} bytes\n")
        f.write(" * Input: 5 float32 features\n")
        f.write(" * Output: 3 float32 probabilities\n")
        f.write(" */\n\n")
        f.write("#ifndef MODEL_H\n")
        f.write("#define MODEL_H\n\n")
        f.write("alignas(8) const unsigned char g_model[] = {\n")
        
        # Write bytes in rows of 12
        for i in range(0, len(tflite_model), 12):
            chunk = tflite_model[i:i+12]
            hex_values = ', '.join(f'0x{b:02x}' for b in chunk)
            f.write(f"    {hex_values},\n")
        
        f.write("};\n\n")
        f.write(f"const unsigned int g_model_len = {len(tflite_model)};\n\n")
        f.write("#endif // MODEL_H\n")
    
    print(f"C header saved: {output_path}")


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def train_and_export():
    """
    Main training pipeline.
    """
    print("=" * 60)
    print("TinyML Model Training for Federated Learning")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    print("\n[1/6] Loading data...")
    # Replace with: X, y = load_real_data("your_dataset.csv")
    X, y = generate_synthetic_data(5000)
    print(f"  - Samples: {len(X)}")
    print(f"  - Features: {X.shape[1]}")
    print(f"  - Class distribution: {np.bincount(y)}")
    
    # Normalize features
    print("\n[2/6] Normalizing features...")
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    
    # Save normalization parameters for Arduino
    print(f"  - Feature means: {scaler.mean_}")
    print(f"  - Feature stds: {scaler.scale_}")
    
    # Save normalization params
    np.save(os.path.join(OUTPUT_DIR, "feature_means.npy"), scaler.mean_)
    np.save(os.path.join(OUTPUT_DIR, "feature_stds.npy"), scaler.scale_)
    
    # Split data
    print("\n[3/6] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Test samples: {len(X_test)}")
    
    # Create and train model
    print("\n[4/6] Training model...")
    model = create_model()
    model.summary()
    
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate
    print("\n[5/6] Evaluating model...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  - Test accuracy: {test_acc * 100:.2f}%")
    print(f"  - Test loss: {test_loss:.4f}")
    
    # Save Keras model
    model.save(os.path.join(OUTPUT_DIR, "model.keras"))
    
    # Convert to TFLite
    print("\n[6/6] Converting to TFLite...")
    tflite_path = os.path.join(OUTPUT_DIR, "model.tflite")
    tflite_model = convert_to_tflite(model, X_train, tflite_path)
    
    # Generate C header
    header_path = os.path.join(OUTPUT_DIR, "model.h")
    generate_c_header(tflite_model, header_path)
    
    # Print normalization code for Arduino
    print("\n" + "=" * 60)
    print("COPY THIS TO YOUR ARDUINO SKETCH:")
    print("=" * 60)
    print(f"\nconst float featureMeans[{NUM_FEATURES}] = {{")
    print(f"    {', '.join(f'{m:.4f}' for m in scaler.mean_)}")
    print("};")
    print(f"const float featureStds[{NUM_FEATURES}] = {{")
    print(f"    {', '.join(f'{s:.4f}' for s in scaler.scale_)}")
    print("};")
    
    print("\n" + "=" * 60)
    print("DONE! Files generated:")
    print("=" * 60)
    print(f"  - {os.path.join(OUTPUT_DIR, 'model.keras')}")
    print(f"  - {tflite_path}")
    print(f"  - {header_path}")
    print(f"\nCopy {header_path} to your Arduino sketch folder.")
    
    return model, scaler


# ============================================================================
# FEDERATED LEARNING SIMULATION
# ============================================================================

def simulate_federated_learning(num_clients=6, num_rounds=10):
    """
    Simulate federated learning with multiple clients.
    This demonstrates how the server-side aggregation works.
    """
    print("\n" + "=" * 60)
    print("FEDERATED LEARNING SIMULATION")
    print("=" * 60)
    
    # Generate data for each client (non-IID distribution)
    X, y = generate_synthetic_data(6000)
    
    # Split data among clients (simulating different locations)
    clients_data = []
    samples_per_client = len(X) // num_clients
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client
        clients_data.append((X[start_idx:end_idx], y[start_idx:end_idx]))
        print(f"  Client {i+1}: {samples_per_client} samples")
    
    # Initialize global model
    global_model = create_model()
    
    # Federated learning rounds
    for round_num in range(num_rounds):
        print(f"\n--- FL Round {round_num + 1}/{num_rounds} ---")
        
        # Collect client updates
        client_weights = []
        client_samples = []
        
        for client_id, (X_client, y_client) in enumerate(clients_data):
            # Each client trains on their local data
            local_model = create_model()
            local_model.set_weights(global_model.get_weights())
            
            # Local training (few epochs)
            local_model.fit(X_client, y_client, epochs=3, verbose=0)
            
            client_weights.append(local_model.get_weights())
            client_samples.append(len(X_client))
        
        # FedAvg aggregation
        total_samples = sum(client_samples)
        new_weights = []
        
        for layer_idx in range(len(client_weights[0])):
            layer_weights = np.zeros_like(client_weights[0][layer_idx])
            for client_idx in range(num_clients):
                weight = client_samples[client_idx] / total_samples
                layer_weights += weight * client_weights[client_idx][layer_idx]
            new_weights.append(layer_weights)
        
        global_model.set_weights(new_weights)
        
        # Evaluate global model
        _, global_acc = global_model.evaluate(X, y, verbose=0)
        print(f"  Global accuracy: {global_acc * 100:.2f}%")
    
    print("\n[FL Simulation Complete]")
    return global_model


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Train and export model
    model, scaler = train_and_export()
    
    # Optional: Simulate federated learning
    # simulate_federated_learning(num_clients=6, num_rounds=10)
