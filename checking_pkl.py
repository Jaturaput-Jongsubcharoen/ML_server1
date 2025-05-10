# -*- coding: utf-8 -*-
"""
Created on Fri May  9 03:26:16 2025

@author: MonaMac
"""

import joblib
model_bundle = joblib.load("logistic_regression_model.pkl")

print("Model object:", type(model_bundle["model"]))
print("Saved feature names:")
print(model_bundle["features"])

model_bundle = joblib.load("logistic_regression_model.pkl")
print(model_bundle["features"])

import joblib

bundle = joblib.load("knn_model.pkl")
print("Expected features in saved model:")
print(bundle["features"])


import joblib

bundle = joblib.load("logistic_regression_model.pkl")  # or any other .pkl
print("Expected features:", bundle["features"])