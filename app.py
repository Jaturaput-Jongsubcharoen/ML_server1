#app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import traceback
import warnings
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load models
models = {
    "random_forest": joblib.load("random_forest_model.pkl"),
    "svm": joblib.load("svm_model.pkl"),
    "neural_network": joblib.load("neural_network_model.pkl"),
    "logistic_regression": joblib.load("logistic_regression_model.pkl"),
    "knn": joblib.load("knn_model.pkl")
}

@app.route("/", methods=["GET"])
def home():
    return "Machine Learning Model API is running."

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        response = jsonify({"message": "Preflight OK"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response, 200

    try:
        data = request.get_json(force=True)
        model_name = data.get("model_name")
        input_data = data.get("input", {})

        if model_name not in models:
            return jsonify({"error": f"Invalid model name: {model_name}"}), 400

        model_bundle = models[model_name]
        model = model_bundle["model"]
        expected_features = model_bundle["features"]
        
        print("Expected features:", expected_features)
        print("Received fields:", list(input_data.keys()))

        missing = [col for col in expected_features if col not in input_data]
        if missing:
            return jsonify({"error": f"Missing required input fields: {missing}"}), 400

        row = [input_data[col] for col in expected_features]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prediction = int(model.predict([row])[0])

        return jsonify({
            "model": model_name,
            "prediction": prediction
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)