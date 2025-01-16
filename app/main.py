from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pickle
import numpy as np
import pandas as pd
# import pyrebase
import firebase_admin
from datetime import datetime
import os
from config import FIREBASE_CONFIG, DEVICE_UUID

app = Flask(__name__)
CORS(app)

# Initialize Firebase with config from environment variables
firebase = firebase_admin.initialize_app(FIREBASE_CONFIG)
db = firebase.database()

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

app = Flask(__name__)

def get_latest_sensor_readings():
    """
    Fetch the latest sensor readings from Firebase
    Returns a dictionary with sensor values
    """
    try:
        # Get all sensor data at once
        sensor_ref = db.child("RTSensorData").child("Device uuid:iqD78eGmo7LompLJHfZwm2").get()
        data = sensor_ref.val()
        
        if not data:
            print("No data found")
            return None
            
        # Extract and convert sensor values
        readings = {}
        for sensor_key in FEATURE_MAPPING.keys():
            if sensor_key in data:
                try:
                    readings[sensor_key] = float(data[sensor_key])
                except ValueError:
                    print(f"Could not convert {sensor_key} value to float: {data[sensor_key]}")
                    continue
        
        return readings
    
    except Exception as e:
        print(f"Error fetching sensor data: {str(e)}")
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
    app.run()
    # app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))