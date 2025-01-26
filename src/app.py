import streamlit as st
import pickle
import numpy as np

# Load model
model_path = "../models/trained_models/linear_regression.pkl"
with open(model_path, 'rb') as f:
    model = pickle.load(f)

st.title("Amazon Delivery Time Prediction")

# Input fields
distance = st.number_input("Distance (km)", min_value=0.0, step=0.1)
order_hour = st.number_input("Order Hour", min_value=0, max_value=23)

if st.button("Predict Delivery Time"):
    # Predict
    input_features = np.array([[distance, order_hour]])
    prediction = model.predict(input_features)[0]
    st.success(f"Estimated Delivery Time: {prediction:.2f} hours")
