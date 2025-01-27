from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pickle
import numpy as np
import pandas as pd
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime
import json
import os
from config import FIREBASE_CONFIG, DEVICE_UUID

app = Flask(__name__)
CORS(app)

# Enhanced Firebase initialization with error handling
try:
    cred_dict = json.loads(os.getenv('FIREBASE_SERVICE_ACCOUNT_KEY'))
    print("Firebase credentials loaded successfully")
    cred = credentials.Certificate(cred_dict)
    
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://playground-bd796-default-rtdb.firebaseio.com'
    })
    print("Firebase initialized successfully")
except Exception as e:
    print(f"Firebase initialization error: {str(e)}")
    raise

# Update model loading paths to use absolute paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(base_dir, 'models')

# Load models
with open(os.path.join(model_dir, "anomaly_model.pkl"), 'rb') as f:
    anomaly_model = pickle.load(f)
print("loaded model 1 ...done!")

with open(os.path.join(model_dir, "logistic_model.pkl"), 'rb') as f:
    fire_model = pickle.load(f)
print("loaded model 2 ...done!")

# Feature mapping
FEATURE_MAPPING = {
    "temp": "Temperature[C]",
    "smoke": "Smoke",
    "lpg": "LPG",
    "humidity": "Humidity[%]",
    "co": "CO"
}

def get_latest_sensor_readings():
    """
    Fetch the latest sensor readings from Firebase
    Returns a dictionary with sensor values
    """
    try:
        # First, check the root to see what data exists
        root_ref = db.reference('/')
        root_data = root_ref.get()
        print(f"Available paths at root: {list(root_data.keys()) if root_data else 'None'}")

        # Now try to access RTSensorData
        ref = db.reference('RTSensorData')
        all_sensor_data = ref.get()
        print(f"Data at RTSensorData: {all_sensor_data}")

        if not all_sensor_data:
            print("No data found at RTSensorData path")
            return None

        # List all devices if any
        if isinstance(all_sensor_data, dict):
            print(f"Available devices: {list(all_sensor_data.keys())}")

        # Try to get specific device data
        # device_id = 'Device uuid:iqD78eGmo7LompLJHfZwm2'
        device_id = 'Device'
        # Try both with and without the 'Device uuid:' prefix
        device_data = None
        
        # First try: Full path
        device_data = ref.child(device_id).get()
        
        # Second try: Just the UUID
        if not device_data:
            uuid_only = device_id.split(':')[-1].strip()
            print(f"Trying with UUID only: {uuid_only}")
            device_data = ref.child(uuid_only).get()

        if not device_data:
            print(f"No data found for device ID using either path")
            return None
            
        print(f"Retrieved device data: {device_data}")
            
        # Extract and convert sensor values
        readings = {}
        for sensor_key in FEATURE_MAPPING.keys():
            if sensor_key in device_data:
                try:
                    readings[sensor_key] = float(device_data[sensor_key])
                except ValueError:
                    print(f"Could not convert {sensor_key} value to float: {device_data[sensor_key]}")
                    continue
        
        return readings
    
    except Exception as e:
        print(f"Error fetching sensor data: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        # Print the full traceback for debugging
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return None
    
def make_predictions(features_dict):
    """
    Make predictions using both models
    """
    try:
        # Map incoming feature names to match model training
        mapped_features = {FEATURE_MAPPING[key]: value for key, value in features_dict.items()}
        
        # Convert to array format
        features = np.array([
            mapped_features["Temperature[C]"],
            mapped_features["Smoke"],
            mapped_features["LPG"],
            mapped_features["Humidity[%]"],
            mapped_features["CO"]
        ]).reshape(1, -1)

        # Make predictions
        anomaly_prediction = anomaly_model.predict(features)
        fire_prediction = fire_model.predict(features)
        
        # Map predictions
        anomaly_result = "Anomaly" if anomaly_prediction[0] == -1 else "Normal"
        fire_result = "High Risk" if fire_prediction[0] == 1 else "Low Risk"
        
        return {
            "anomaly": anomaly_result,
            "fire_risk": fire_result
        }
    except Exception as e:
        print(f"Error making predictions: {str(e)}")
        return None

@app.route("/")
def home():
    return jsonify({
        "message": "Fire Prediction API",
        "endpoints": {
            "GET /predict/latest": "Get latest predictions from sensor data",
            "POST /predict": "Make predictions from provided data",
            "GET /sensors/latest": "Get latest sensor readings",
            "GET /health": "Health check"
        }
    })

@app.route("/predict/latest", methods=["GET"])
def predict_latest():
    try:
        # Get latest sensor readings
        sensor_data = get_latest_sensor_readings()
        
        if not sensor_data:
            return jsonify({"error": "Failed to fetch sensor readings"}), 400

        # Make predictions
        predictions = make_predictions(sensor_data)
        
        if not predictions:
            return jsonify({"error": "Failed to make predictions"}), 400

        # Get timestamp from Firebase data or use current time
        timestamp = datetime.now().isoformat()
        
        # Return results as JSON
        return jsonify({
            "timestamp": timestamp,
            "sensor_readings": sensor_data,
            "predictions": predictions
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Make predictions
        predictions = make_predictions(data)
        
        if not predictions:
            return jsonify({"error": "Failed to make predictions"}), 400

        # Return results as JSON
        return jsonify(predictions)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Add a route to check the latest sensor readings without making predictions
@app.route("/sensors/latest", methods=["GET"])
def get_latest():
    try:
        sensor_data = get_latest_sensor_readings()
        if not sensor_data:
            return jsonify({"error": "Failed to fetch sensor readings"}), 400
            
        return jsonify({
            "timestamp": datetime.now().isoformat(),
            "sensor_readings": sensor_data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

if __name__ == "__main__":
    # app.run()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))