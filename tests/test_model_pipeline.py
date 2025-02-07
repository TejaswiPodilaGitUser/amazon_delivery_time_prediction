import pandas as pd
import numpy as np
import sys
import os

# Get the absolute path of the 'src' directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from model_training import load_trained_model  # Function to load the trained model
from data_preparation import preprocess_data  # Function to preprocess input data
from app import preprocess_input  # Import function from your Streamlit app

# Load the trained model
model = load_trained_model("best_model.pkl")  # Ensure this function exists in model_training.py

# Define feature columns based on the model
feature_columns = model.feature_names_in_

# Define test cases
test_cases = [
    {"name": "Valid Input", "features": [5.2, 14], "expected": "Valid"},  # Normal case
    {"name": "Missing Values", "features": [None, 14], "expected": "Handle Missing"},
    {"name": "Extreme Values", "features": [99999, -99999], "expected": "Handle Extreme"},
    {"name": "Negative Distance", "features": [-10, 14], "expected": "Handle Negative"},
    {"name": "Non-Numeric Values", "features": ["five", 14], "expected": "Handle Non-Numeric"},
]

# Run test cases
for test in test_cases:
    try:
        print(f"Running Test: {test['name']}")
        # Convert features to DataFrame
        input_data = pd.DataFrame([test["features"]], columns=feature_columns)
        
        # Preprocess data
        processed_input = preprocess_input(test["features"], feature_columns, model)
        
        # Make prediction
        prediction = model.predict(processed_input)
        
        print(f"✅ {test['name']} Passed | Prediction: {prediction[0]}")
    except Exception as e:
        print(f"❌ {test['name']} Failed | Error: {str(e)}")
