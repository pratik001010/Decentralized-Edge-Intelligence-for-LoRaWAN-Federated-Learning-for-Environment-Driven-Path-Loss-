"""
fl_server.py

Federated Learning Server/Orchestrator for LoRaWAN Edge Devices

This server:
1. Receives model updates from MKR WAN 1310 nodes via TTN/LoRaWAN backend
2. Aggregates updates using FedAvg
3. Compresses and sends global model back to nodes
4. Manages FL rounds and scheduling

Integration options:
- The Things Network (TTN) webhook
- ChirpStack integration
- Custom MQTT broker

Author: Pratik Khadka
Master's Thesis: Federated TinyML for LoRaWAN Edge Intelligence
"""

import numpy as np
import json
import time
import base64
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from flask import Flask, request, jsonify
import threading
import struct

# ============================================================================
# CONFIGURATION
# ============================================================================

FL_CONFIG = {
    "num_clients": 6,                    # Number of MKR WAN 1310 nodes
    "min_clients_per_round": 3,          # Minimum clients to start aggregation
    "fl_round_duration_hours": 24,       # FL round duration
    "model_weights_size": 128,           # Number of model parameters
    "quantization_scale": 127,           # Scale for int8 quantization
    "downlink_port": 3,                  # LoRaWAN downlink port
}

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ClientUpdate:
    """Model update from a single client."""
    device_id: str
    timestamp: datetime
    model_version: int
    weight_deltas: np.ndarray
    num_samples: int = 1
    pdr: float = 1.0

@dataclass
class FLRound:
    """State of a federated learning round."""
    round_id: int
    start_time: datetime
    updates: List[ClientUpdate] = field(default_factory=list)
    is_complete: bool = False
    global_weights: Optional[np.ndarray] = None

# ============================================================================
# FEDERATED LEARNING SERVER
# ============================================================================

class FederatedLearningServer:
    """
    Orchestrator for federated learning with LoRaWAN devices.
    
    Handles:
    - Receiving model updates from devices
    - FedAvg aggregation
    - Global model management
    - Downlink scheduling
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.current_round: Optional[FLRound] = None
        self.round_history: List[FLRound] = []
        self.global_weights = np.zeros(config["model_weights_size"])
        self.model_version = 1
        self.client_states: Dict[str, dict] = defaultdict(dict)
        self.lock = threading.Lock()
        
        # Start new round
        self._start_new_round()
    
    def _start_new_round(self):
        """Initialize a new FL round."""
        round_id = len(self.round_history) + 1
        self.current_round = FLRound(
            round_id=round_id,
            start_time=datetime.now()
        )
        print(f"[FL Server] Starting Round {round_id}")
    
    def receive_update(self, device_id: str, payload: bytes) -> dict:
        """
        Process incoming model update from a device.
        
        Payload structure:
        - Byte 0: Message type (0x01 = model update)
        - Byte 1: Model version
        - Bytes 2-3: Number of weights
        - Bytes 4+: Quantized weight deltas (int8)
        """
        with self.lock:
            if len(payload) < 4:
                return {"status": "error", "message": "Invalid payload"}
            
            msg_type = payload[0]
            if msg_type != 0x01:
                return {"status": "ignored", "message": "Not a model update"}
            
            model_version = payload[1]
            num_weights = (payload[2] << 8) | payload[3]
            
            # Dequantize weight deltas
            weight_deltas = np.zeros(self.config["model_weights_size"])
            for i in range(min(num_weights, len(payload) - 4)):
                quantized = struct.unpack('b', bytes([payload[4 + i]]))[0]
                weight_deltas[i] = quantized / self.config["quantization_scale"]
            
            # Create client update
            update = ClientUpdate(
                device_id=device_id,
                timestamp=datetime.now(),
                model_version=model_version,
                weight_deltas=weight_deltas
            )
            
            # Add to current round
            self.current_round.updates.append(update)
            self.client_states[device_id]["last_update"] = datetime.now()
            
            print(f"[FL Server] Received update from {device_id}")
            print(f"  - Model version: {model_version}")
            print(f"  - Weights received: {num_weights}")
            print(f"  - Total updates this round: {len(self.current_round.updates)}")
            
            # Check if we should aggregate
            if len(self.current_round.updates) >= self.config["min_clients_per_round"]:
                self._try_aggregation()
            
            return {
                "status": "success",
                "updates_received": len(self.current_round.updates),
                "round_id": self.current_round.round_id
            }
    
    def _try_aggregation(self):
        """
        Perform FedAvg aggregation if conditions are met.
        """
        if self.current_round.is_complete:
            return
        
        updates = self.current_round.updates
        print(f"\n[FL Server] === AGGREGATION ===")
        print(f"  - Round: {self.current_round.round_id}")
        print(f"  - Participating clients: {len(updates)}")
        
        # FedAvg: weighted average of updates
        total_samples = sum(u.num_samples for u in updates)
        aggregated_deltas = np.zeros(self.config["model_weights_size"])
        
        for update in updates:
            weight = update.num_samples / total_samples
            aggregated_deltas += weight * update.weight_deltas
        
        # Apply aggregated deltas to global model
        self.global_weights += aggregated_deltas
        self.model_version += 1
        
        # Mark round as complete
        self.current_round.is_complete = True
        self.current_round.global_weights = self.global_weights.copy()
        self.round_history.append(self.current_round)
        
        print(f"  - New model version: {self.model_version}")
        print(f"  - Aggregated delta norm: {np.linalg.norm(aggregated_deltas):.6f}")
        
        # Start new round
        self._start_new_round()
    
    def get_global_model_payload(self) -> bytes:
        """
        Generate compressed global model payload for downlink.
        
        Payload structure:
        - Byte 0: Message type (0x02 = global model)
        - Byte 1: Model version
        - Bytes 2-3: Number of weights
        - Bytes 4+: Quantized weights (int8)
        """
        payload = bytearray()
        payload.append(0x02)  # Message type
        payload.append(self.model_version & 0xFF)
        
        # Number of weights (limited by LoRaWAN payload)
        max_weights = min(47, self.config["model_weights_size"])  # 51 - 4 header bytes
        payload.append((max_weights >> 8) & 0xFF)
        payload.append(max_weights & 0xFF)
        
        # Quantize weights
        for i in range(max_weights):
            quantized = int(np.clip(
                self.global_weights[i] * self.config["quantization_scale"],
                -128, 127
            ))
            payload.append(quantized & 0xFF)
        
        return bytes(payload)
    
    def get_status(self) -> dict:
        """Return server status."""
        return {
            "model_version": self.model_version,
            "current_round": self.current_round.round_id if self.current_round else 0,
            "updates_this_round": len(self.current_round.updates) if self.current_round else 0,
            "rounds_completed": len(self.round_history),
            "active_clients": len(self.client_states),
            "global_weights_norm": float(np.linalg.norm(self.global_weights))
        }


# ============================================================================
# FLASK API
# ============================================================================

app = Flask(__name__)
fl_server = FederatedLearningServer(FL_CONFIG)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})


@app.route('/status', methods=['GET'])
def status():
    """Get FL server status."""
    return jsonify(fl_server.get_status())


@app.route('/uplink', methods=['POST'])
def uplink():
    """
    Handle uplink webhook from TTN/ChirpStack.
    
    Expected JSON format (TTN v3):
    {
        "end_device_ids": {"device_id": "device-01"},
        "uplink_message": {
            "frm_payload": "<base64 encoded payload>",
            "f_port": 3
        }
    }
    """
    try:
        data = request.get_json()
        
        # Parse TTN webhook format
        device_id = data.get("end_device_ids", {}).get("device_id", "unknown")
        uplink_msg = data.get("uplink_message", {})
        payload_b64 = uplink_msg.get("frm_payload", "")
        
        if not payload_b64:
            return jsonify({"status": "error", "message": "No payload"}), 400
        
        # Decode payload
        payload = base64.b64decode(payload_b64)
        
        # Process update
        result = fl_server.receive_update(device_id, payload)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/downlink/<device_id>', methods=['GET'])
def downlink(device_id: str):
    """
    Get downlink payload for a specific device.
    
    Returns the global model in base64 format for TTN downlink.
    """
    try:
        payload = fl_server.get_global_model_payload()
        payload_b64 = base64.b64encode(payload).decode('utf-8')
        
        return jsonify({
            "device_id": device_id,
            "f_port": FL_CONFIG["downlink_port"],
            "frm_payload": payload_b64,
            "model_version": fl_server.model_version,
            "payload_size": len(payload)
        })
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/schedule_downlink/<device_id>', methods=['POST'])
def schedule_downlink(device_id: str):
    """
    Schedule a downlink to a specific device via TTN API.
    
    Note: This is a placeholder. In production, you would use
    the TTN Application Server API to schedule the downlink.
    """
    payload = fl_server.get_global_model_payload()
    
    # TODO: Integrate with TTN API
    # Example for TTN v3 API:
    # POST https://eu1.cloud.thethings.network/api/v3/as/applications/{app_id}/devices/{device_id}/down/push
    # Headers: Authorization: Bearer {api_key}
    # Body: {"downlinks": [{"f_port": 3, "frm_payload": "<base64>", "confirmed": true}]}
    
    return jsonify({
        "status": "scheduled",
        "device_id": device_id,
        "model_version": fl_server.model_version,
        "payload_size": len(payload),
        "note": "Integration with TTN API required"
    })


# ============================================================================
# SIMULATION MODE
# ============================================================================

def simulate_fl_round():
    """
    Simulate a complete FL round for testing.
    """
    print("\n" + "=" * 60)
    print("SIMULATING FEDERATED LEARNING ROUND")
    print("=" * 60)
    
    # Simulate updates from 6 clients
    for i in range(6):
        device_id = f"mkrwan-{i+1:02d}"
        
        # Create fake model update
        payload = bytearray()
        payload.append(0x01)  # Model update
        payload.append(0x01)  # Model version
        payload.append(0x00)  # Num weights high byte
        payload.append(0x20)  # Num weights low byte (32)
        
        # Random weight deltas
        for j in range(32):
            delta = np.random.randn() * 0.1
            quantized = int(np.clip(delta * 127, -128, 127))
            payload.append(quantized & 0xFF)
        
        result = fl_server.receive_update(device_id, bytes(payload))
        print(f"  Client {device_id}: {result['status']}")
        time.sleep(0.5)
    
    # Get global model
    print("\nGlobal model payload:")
    payload = fl_server.get_global_model_payload()
    print(f"  Size: {len(payload)} bytes")
    print(f"  Base64: {base64.b64encode(payload).decode()[:50]}...")
    
    print("\nServer status:")
    print(json.dumps(fl_server.get_status(), indent=2))


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "simulate":
        simulate_fl_round()
    else:
        print("=" * 60)
        print("Federated Learning Server for LoRaWAN Edge Devices")
        print("=" * 60)
        print("\nEndpoints:")
        print("  GET  /health         - Health check")
        print("  GET  /status         - FL server status")
        print("  POST /uplink         - Receive model update (TTN webhook)")
        print("  GET  /downlink/<id>  - Get global model for device")
        print("  POST /schedule_downlink/<id> - Schedule downlink")
        print("\nStarting server on http://0.0.0.0:5000")
        print("=" * 60 + "\n")
        
        app.run(host='0.0.0.0', port=5000, debug=True)
