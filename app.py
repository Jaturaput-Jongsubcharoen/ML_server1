# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 15:38:46 2025

@author: Jaturaput Jongsubcharoen | 301391425 | COMP247001
"""

# -----------------------------------------------------------------------------
# 5. Deploying the Model
# -----------------------------------------------------------------------------

# 5.1 Flask API to serve the trained model
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib

# 5.2 Load the serialized models
models = {
    "random_forest": joblib.load("random_forest_model.pkl"),
    "svm": joblib.load("svm_model.pkl"),
    "neural_network": joblib.load("neural_network_model.pkl"),
    "logistic_regression": joblib.load("logistic_regression_model.pkl"),
    "knn": joblib.load("knn_model.pkl")
}

# Define the expected one-hot encoded feature names
expected_features = [
    'STREET1_LAWRENCE AVE E',
    'STREET2_E OF DVP ON RAMP Aven',
    'OFFSET_10 m West of',
    'DISTRICT_Toronto and East York',
    'IMPACTYPE_Pedestrian Collisions',
    'INJURY_Fatal',
    'INJURY_Major',
    'PEDCOND_Unknown',
    'PEDCOND_Normal',
    'DRIVCOND_Unknown'
]

EXPECTED_NUM_FEATURES = len(expected_features)

# Create the Flask app
app = Flask(__name__)
CORS(app)

# 5.3 Home route
@app.route('/')
def home():
    return "Multiple ML Models API is running!"

# 5.4 /predict route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    model_name = data.get("model_name")
    input_data = data.get("input", {})

    if model_name not in models:
        return jsonify({"error": "Invalid model name"}), 400

    # Manual one-hot encoding
    encoded_vector = []
    for feature in expected_features:
        try:
            col, val = feature.split("_", 1)
        except ValueError:
            return jsonify({"error": f"Feature format error in: {feature}"}), 400

        encoded_vector.append(1 if input_data.get(col) == val else 0)

    if len(encoded_vector) != EXPECTED_NUM_FEATURES:
        return jsonify({
            "error": f"Expected {EXPECTED_NUM_FEATURES} features, but got {len(encoded_vector)}",
            "received_input": list(input_data.keys())
        }), 400

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            features_array = np.array([encoded_vector])
            prediction = models[model_name].predict(features_array)[0]
            prediction = int(prediction)

        return jsonify({
            "model": model_name,
            "prediction": prediction
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# 5.5 Run Flask server
if __name__ == '__main__':
    app.run(debug=True)