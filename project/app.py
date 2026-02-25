from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import os
import logging

# Flask API Definition
app = Flask(__name__, template_folder="templates", static_folder="static")

# Setup Logging
logging.basicConfig(
    filename="flask_app.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Project Folder Structure
PROJECT_PATH = r"C:\Users\hp\1.CI_CIS6005-II"
DATA_PATH = os.path.join(PROJECT_PATH, 'data')
MODEL_PATH = os.path.join(PROJECT_PATH, 'models')
OUTPUT_PATH = os.path.join(PROJECT_PATH, 'output')

print("PROJECT_PATH:", PROJECT_PATH)
print("DATA_PATH   :", DATA_PATH)
print("MODEL_PATH  :", MODEL_PATH)
print("OUTPUT_PATH :", OUTPUT_PATH)



# Load Crash Severity System
SYSTEM_FILE = os.path.join(MODEL_PATH, "crash_severity_system.pkl")
ARTIFACTS_FILE = os.path.join(MODEL_PATH, "preprocessing_artifacts.pkl")

model, target_encoder, encoders, features = None, None, {}, []

# Load model
try:
    system = joblib.load(SYSTEM_FILE)
    model = system.get("model")
    target_encoder = system.get("target_encoder")
    if model is not None and target_encoder is not None:
        print(f"Crash severity system loaded successfully from {SYSTEM_FILE}")
        logging.info(f"Crash severity system loaded successfully from {SYSTEM_FILE}")
    else:
        print(f"Model or target encoder is missing in {SYSTEM_FILE}")
        logging.warning(f"Model or target encoder is missing in {SYSTEM_FILE}")
except FileNotFoundError:
    print(f"{SYSTEM_FILE} not found.")
    logging.error(f"{SYSTEM_FILE} not found.")
except Exception as e:
    print(f"Error loading crash severity system: {e}")
    logging.error(f"Error loading crash severity system: {e}")

# Load preprocessing artifacts
try:
    artifacts = joblib.load(ARTIFACTS_FILE)
    encoders = artifacts.get("encoders", {})
    features = artifacts.get("features", [])
    if encoders and features:
        print(f"Preprocessing artifacts loaded successfully from {ARTIFACTS_FILE}")
        logging.info(f"Preprocessing artifacts loaded successfully from {ARTIFACTS_FILE}")
    else:
        print(f"Preprocessing artifacts are incomplete in {ARTIFACTS_FILE}")
        logging.warning(f"Preprocessing artifacts are incomplete in {ARTIFACTS_FILE}")
except FileNotFoundError:
    print(f" {ARTIFACTS_FILE} not found.")
    logging.error(f"{ARTIFACTS_FILE} not found.")
except Exception as e:
    print(f"Error loading preprocessing artifacts: {e}")
    logging.error(f"Error loading preprocessing artifacts: {e}")

# Preprocess Input Function
def preprocess_input(data):
    """
    Preprocesses raw data
    """
    try:
        df = pd.DataFrame([data])
        for col, le in encoders.items():
            if col in df.columns:
                df[col] = df[col].astype(str)
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                df[col] = le.transform(df[col])
        df = df.reindex(columns=features, fill_value=0)
        return df
    except Exception as e:
        logging.error(f"Error in preprocessing input: {e}")
        raise

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or target_encoder is None:
        msg = "Model or target encoder not loaded"
        print(f"[WARNING] {msg}")
        logging.warning(msg)
        return jsonify({"error": msg}), 500
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        X = preprocess_input(data)
        proba = model.predict_proba(X)[0]
        pred_idx = int(np.argmax(proba))
        confidence = float(proba[pred_idx])
        label = target_encoder.inverse_transform([pred_idx])[0]

        print(f"[INFO] Prediction made: {label} (confidence: {confidence:.3f})")
        logging.info(f"Prediction made: {label} with confidence {confidence:.3f}")

        return jsonify({"prediction": label, "confidence": round(confidence, 3)})
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        logging.error(f"Error during prediction: {e}")
        return jsonify({"error": "Prediction failed, see server logs"}), 500

# Error Handlers
@app.errorhandler(404)
def handle_404(e):
    msg = f"404 Error: {request.path} not found"
    print(f"[WARNING] {msg}")
    logging.warning(msg)
    return jsonify({
        "error": "Resource not found",
        "details": f"The requested URL {request.path} was not found on the server."
    }), 404

@app.errorhandler(Exception)
def handle_exception(e):
    print(f"[ERROR] Unhandled exception: {e}")
    logging.error(f"Unhandled Exception: {e}", exc_info=True)
    return jsonify({
        "error": "An unexpected error occurred",
        "details": str(e)
    }), 500

# Run Flask App
if __name__ == "__main__":
    print("Starting Flask server at http://127.0.0.1:5000/")
    app.run(debug=True)
