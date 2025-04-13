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
import joblib  # 5.2 we'll use this to load the model

# 5.2 Load the serialized models
# Make sure the model files are in the same folder as this script
models = {
    "random_forest": joblib.load("random_forest_model.pkl"),
    "svm": joblib.load("svm_model.pkl"),
    "neural_network": joblib.load("neural_network_model.pkl"),
    "logistic_regression": joblib.load("logistic_regression_model.pkl"),
    "linear_regression": joblib.load("linear_regression_model.pkl")
}

# Define the expected number of features for safety check (e.g., 10 categorical features after OneHotEncoding)
# Replace this with the actual number if known (e.g., 43 if one-hot created 43 columns)
EXPECTED_NUM_FEATURES = models["logistic_regression"].n_features_in_

# Create the Flask app instance
app = Flask(__name__)
CORS(app)

# 5.3 Home route to confirm the server is running
@app.route('/')
def home():
    return "Multiple ML Models API is running!"

# 5.4 /predict route to receive input and return prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Expect JSON data in the form: { "model_name": "svm", "features": [val1, val2, ..., valN] }
    data = request.get_json(force=True)
    model_name = data.get("model_name")
    features = data.get("features", [])

    # Input validation
    if model_name not in models:
        return jsonify({"error": "Invalid model name"}), 400

    if not isinstance(features, list) or len(features) != EXPECTED_NUM_FEATURES:
        return jsonify({"error": f"Expected {EXPECTED_NUM_FEATURES} features, but got {len(features)}"}), 400

    # Reshape input into 2D array
    features_array = np.array(features).reshape(1, -1)
    model = models[model_name]

    # Generate prediction
    prediction = model.predict(features_array)[0]

    # Special handling for Linear Regression to convert float output to class
    if model_name == "linear_regression":
        prediction = int(prediction > 0.5)
    else:
        prediction = int(prediction) if not isinstance(prediction, np.ndarray) else prediction.tolist()

    # Return prediction result
    return jsonify({
        "model": model_name,
        "prediction": prediction
    })

# 5.5 Deploy model on localhost
if __name__ == '__main__':
    # Run the Flask app
    # Open your browser at http://127.0.0.1:5000 to see it working
    app.run(debug=True)
    