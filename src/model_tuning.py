import pandas as pd
import numpy as np
import logging
import pickle
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# File path to the trained model
file_path = "models/best_model.pkl"

logging.info(f"📂 Loading file: {file_path}")

try:
    with open(file_path, "rb") as file:
        model_data = joblib.load(file)

    # Ensure model_data is a dictionary
    if not isinstance(model_data, dict) or "model" not in model_data or "feature_names" not in model_data:
        logging.error("❌ Invalid model file format. Expected a dictionary with 'model' and 'feature_names'.")
        exit()

    pipeline = model_data["model"]
    feature_names = model_data["feature_names"]

    logging.info(f"✅ Model loaded successfully.")
    logging.info(f"🔢 Expected Features: {len(feature_names)} - {feature_names}")

    # Ensure feature names match
    if len(feature_names) != pipeline.named_steps["model"].n_features_in_:
        logging.error(f"❌ Feature count mismatch: Model expects {pipeline.named_steps['model'].n_features_in_}, but got {len(feature_names)}.")
        exit()

    logging.info("✅ Feature names validated successfully.")

except FileNotFoundError:
    logging.error(f"❌ File not found: {file_path}")
except Exception as e:
    logging.error(f"❌ Unexpected error: {e}")
