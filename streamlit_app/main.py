import streamlit as st
import pandas as pd
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Visualization.data_visualization import generate_plots

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import sidebar  # Import sidebar logic

# Initialize logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load the trained model
try:
    model_data = joblib.load('models/best_model.pkl')

    if isinstance(model_data, dict):  # If saved as a dictionary
        model = model_data['model']
        feature_names = model_data.get('feature_names', [])
    else:  # If saved directly as a Pipeline
        model = model_data
        feature_names = getattr(model.named_steps['model'], 'feature_names_in_', [])
    
    feature_names = [str(f) for f in feature_names]  # Ensure feature names are strings
    logger.info("✅ Model loaded successfully")
    logger.debug(f"Extracted feature names: {feature_names}")
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")
    model = None
    feature_names = []

# Define feature mappings
WEATHER_MAP = {'Sunny': 0, 'Cloudy': 1, 'Stormy': 2, 'Sandstorms': 3}
TRAFFIC_MAP = {'Low': 0, 'Medium': 1, 'High': 2, 'Jam': 3}
# Define vehicle mapping
VEHICLE_MAP = {'motorcycle': 0, 'scooter': 1, 'car': 2, 'bicycle': 3, 'truck': 4}
AREA_MAP = {'Urban': 0, 'Suburban': 1, 'Rural': 2}  # Add more categories if needed

CATEGORY_MAP = {
    'Clothing': 0, 'Electronics': 1, 'Sports': 2, 'Cosmetics': 3, 'Toys': 4, 
    'Shoes': 5, 'Apparel': 6, 'Snacks': 7, 'Outdoor': 8, 'Jewelry': 9, 
    'Kitchen': 10, 'Groceries': 11, 'Books': 12, 'Others': 13, 'Home': 14, 
    'Pet supplies': 15, 'Skin care': 16
}


def preprocess_data(input_data):
    """Preprocess input data based on the selected options"""
    try:
        logger.debug(f"Raw Input Data: {input_data}")

        # Convert categorical values
        input_data['Weather'] = WEATHER_MAP.get(input_data.get('Weather', 'Sunny'), 0)
        input_data['Traffic'] = TRAFFIC_MAP.get(input_data.get('Traffic', 'Medium'), 1)
        logger.debug(f"Mapped categorical values: Weather={input_data['Weather']}, Traffic={input_data['Traffic']}")
        input_data['Vehicle'] = VEHICLE_MAP.get(input_data.get('Vehicle', 'motorcycle'), 0)  # Default to 'motorcycle'
        input_data['Area'] = AREA_MAP.get(input_data.get('Area', 'Urban'), 0)  # Default to 'Urban'
        input_data['Category'] = CATEGORY_MAP.get(input_data.get('Category', 'Others'), 13)  # Default to 'Others'



        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Ensure correct feature order and fill missing columns with default values
        if feature_names:
            for feature in feature_names:
                if feature not in input_df.columns:
                    input_df[feature] = 0  # Fill missing features with default value
            input_df = input_df[feature_names]  # Reorder columns
        else:
            logger.error("Model feature names not available.")
            raise ValueError("Feature names are missing in the trained model.")

        logger.debug(f"Final Preprocessed Data (Correct Feature Order):\n{input_df}")
        return input_df
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        return None

# Streamlit UI
st.title("Amazon Delivery Time Prediction")

# Get user input from sidebar
input_data, selected_options = sidebar.get_input_data()

# Prediction
if st.sidebar.button("Predict Delivery Time"):
    if model is None:
        st.sidebar.error("Model is not loaded. Check the model file path.")
    else:
        input_df = preprocess_data(input_data)
        if input_df is not None:
            try:
                logger.debug(f"Making prediction with DataFrame columns: {input_df.columns.tolist()}")
                predicted_time = model.predict(input_df)[0]
                st.sidebar.success(f"Predicted Delivery Time: {predicted_time:.2f} minutes")
                st.write(f"### Expected Delivery Time: {predicted_time:.2f} minutes")
                
                # Generate plots based on selected features
                generate_plots("data/processed/engineered_data.csv", selected_options)
            except Exception as e:
                logger.error(f"Error during prediction: {e}")
                st.error(f"Error during prediction: {e}")
