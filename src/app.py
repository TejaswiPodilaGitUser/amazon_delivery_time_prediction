import os
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from charts import plot_predictions_vs_actual, plot_feature_importance, plot_delivery_time_distribution
import matplotlib.pyplot as plt

# Set the page layout to wide
st.set_page_config(layout="wide")

# Directory where the models are saved
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
    model = mlflow.sklearn.load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# Load and preprocess the dataset
data_path = "data/processed/cleaned_data.csv"
df = pd.read_csv(data_path)
target_column = "Delivery_Time"

def preprocess_data(df):
    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_columns:
        df[col] = df[col].astype('category').cat.codes

    integer_columns = df.select_dtypes(include=["int64", "int32"]).columns.tolist()
    df[integer_columns] = df[integer_columns].astype(np.float64)
    return df

df_processed = preprocess_data(df)
X = df_processed.drop(columns=[target_column])
y = df_processed[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to preprocess input features for prediction
def preprocess_input(distance, order_hour, df_columns):
    input_data = pd.DataFrame([[distance, order_hour]], columns=['Distance', 'Order_Hour'])
    
    # Ensure that the columns of the input match the training data
    missing_cols = set(df_columns) - set(input_data.columns)
    for col in missing_cols:
        input_data[col] = 0  # Add missing columns and set them to a default value (e.g., 0)
    
    # Ensure columns are in the same order as the training data
    input_data = input_data[df_columns]
    
    # Preprocess the categorical and integer columns
    categorical_columns = input_data.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_columns:
        input_data[col] = input_data[col].astype('category').cat.codes

    integer_columns = input_data.select_dtypes(include=["int64", "int32"]).columns.tolist()
    input_data[integer_columns] = input_data[integer_columns].astype(np.float64)
    
    return input_data


st.title("Amazon Delivery Time Prediction")
distance = st.number_input("Distance (km)", min_value=0.0, step=0.1)
order_hour = st.number_input("Order Hour", min_value=0, max_value=23)

if st.button("Predict Delivery Time"):
    input_features = preprocess_input(distance, order_hour, X.columns.tolist())
    if model:
        try:
            prediction = model.predict(input_features)
            st.success(f"Estimated Delivery Time: {prediction[0]:.2f} hours")

            # Key Insights for the given input
            st.subheader("Key Insights")
            avg_time = y_test.mean()
            max_time = y_test.max()
            min_time = y_test.min()

            st.markdown(f"""
            - **For Distance = {distance} km and Order Hour = {order_hour}:**
                - **Predicted Delivery Time:** {prediction[0]:.2f} hours
                - **Average Delivery Time in the Dataset:** {avg_time:.2f} hours
                - **Fastest Delivery Time:** {min_time:.2f} hours
                - **Slowest Delivery Time:** {max_time:.2f} hours
            - **Distance Impact:** The farther the distance, the longer the delivery time.
            - **Peak Hours:** Orders placed during peak hours might show slight delays.
            """)

        except ValueError as e:
            st.error(f"Error in prediction: {e}")
    else:
        st.error("Model not loaded.")

if model:
    predictions = model.predict(X_test)
    st.subheader("Visualizations")

    # Create three columns with spacing
    col1, spacer, col2 = st.columns([1, 0.1, 1])  # Adjust widths as needed


    # Plot predictions vs actual in the first column
    with col1:
        fig, ax = plot_predictions_vs_actual(predictions, y_test)
        st.pyplot(fig)

    # Plot feature importance in the second column
    with col2:
        fig, ax = plot_feature_importance(model, X_train.columns)
        st.pyplot(fig)

    # Plot delivery time distribution
    with col1:
        fig, ax = plot_delivery_time_distribution(y_test)
        st.pyplot(fig)
