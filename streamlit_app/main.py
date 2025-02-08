import os
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.charts import plot_predictions_vs_actual, plot_feature_importance, plot_delivery_time_distribution
from src.model_tuning import train_and_tune_model
import joblib

st.set_page_config(layout="wide")

model_dir = "models/trained_models"

def get_latest_model_path(model_dir):
    model_files = [f for f in os.listdir(model_dir) if f.startswith("best_model_") and f.endswith(".pkl")]
    model_files.sort(reverse=True)
    if model_files:
        return os.path.join(model_dir, model_files[0])
    else:
        raise FileNotFoundError("No model file found in the specified directory.")

try:
    model_path = get_latest_model_path(model_dir)
    model = joblib.load(model_path)
    st.success("Model loaded successfully.")
except Exception as e:
    model = None
    st.error(f"Error loading model: {e}")

data_path = "data/processed/cleaned_data.csv"
df = pd.read_csv(data_path)
target_column = "Delivery_Time"

def preprocess_data(df):
    df = df.copy()
    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_columns:
        df[col] = df[col].astype('category').cat.codes
    df = df.astype(np.float64)
    return df

df_processed = preprocess_data(df)
X = df_processed.drop(columns=[target_column])
y = df_processed[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def preprocess_input(features, df_columns, model):
    # Convert features to a DataFrame (assuming features only contains a few required values)
    input_data = pd.DataFrame([features], columns=df_columns[:len(features)])

    # Ensure all model-required columns are included, filling missing ones with 0
    if hasattr(model, 'feature_names_in_'):
        input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

    # Apply preprocessing
    input_data = preprocess_data(input_data)

    return input_data


st.title("Amazon Delivery Time Prediction")
distance = st.number_input("Distance (km)", min_value=0.0, step=0.1)
order_hour = st.number_input("Order Hour", min_value=0, max_value=23)

if st.button("Train & Tune Model"):
    with st.spinner("Training and Tuning Model..."):
        best_model, best_params = train_and_tune_model(X_train, y_train)
        st.success("Model trained and tuned successfully!")
        st.write(f"Best Hyperparameters: {best_params}")
        mlflow.sklearn.log_model(best_model, "best_model")
        mlflow.log_params(best_params)
        best_model_path = os.path.join(model_dir, "best_model.pkl")
        joblib.dump(best_model, best_model_path)
        st.write(f"Best model saved to: {best_model_path}")

if st.button("Predict Delivery Time"):
    input_features = preprocess_input([distance, order_hour], X.columns.tolist(), model)
    if model:
        try:
            prediction = model.predict(input_features)
            st.success(f"Estimated Delivery Time: {prediction[0]:.2f} hours")
            avg_time = y_test.mean()
            max_time = y_test.max()
            min_time = y_test.min()
            st.subheader("Key Insights")
            st.markdown(f"""
            - **Predicted Delivery Time:** {prediction[0]:.2f} hours
            - **Average Delivery Time:** {avg_time:.2f} hours
            - **Fastest Delivery Time:** {min_time:.2f} hours
            - **Slowest Delivery Time:** {max_time:.2f} hours
            - **Distance Impact:** Longer distances increase delivery time.
            - **Peak Hours:** Orders during peak hours may be delayed.
            """)
        except ValueError as e:
            st.error(f"Error in prediction: {e}")
    else:
        st.error("Model not loaded.")

if model:
    predictions = model.predict(X_test.reindex(columns=model.feature_names_in_, fill_value=0))
    st.subheader("Visualizations")
    col1, spacer, col2 = st.columns([1, 0.1, 1])
    with col1:
        fig, ax = plot_predictions_vs_actual(predictions, y_test)
        st.pyplot(fig)
    with col2:
        fig, ax = plot_feature_importance(model, X_train.columns)
        st.pyplot(fig)
    with col1:
        fig, ax = plot_delivery_time_distribution(y_test)
        st.pyplot(fig)