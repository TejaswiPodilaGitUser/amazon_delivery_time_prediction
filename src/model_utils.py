# model_utils.py
import os
import joblib
import mlflow
import pandas as pd
import numpy as np

def get_latest_model_path(model_dir):
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
    model_files.sort(reverse=True)
    if model_files:
        return os.path.join(model_dir, model_files[0])
    else:
        raise FileNotFoundError("No model file found in the specified directory.")

def load_model(model_dir):
    try:
        model_path = get_latest_model_path(model_dir)
        model = joblib.load(model_path)
        return model
    except Exception as e:
        return None, str(e)

def preprocess_input(features, df_columns, model, preprocess_data_fn):
    # Convert features to a DataFrame (assuming features only contains a few required values)
    input_data = pd.DataFrame([features], columns=df_columns[:len(features)])

    # Ensure all model-required columns are included, filling missing ones with 0
    if hasattr(model, 'feature_names_in_'):
        input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

    # Apply preprocessing
    input_data = preprocess_data_fn(input_data)

    return input_data

def make_prediction(model, input_data):
    return model.predict(input_data)
