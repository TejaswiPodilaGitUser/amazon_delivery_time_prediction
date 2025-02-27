import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import streamlit as st

def plot_predictions_vs_actual(predictions, y_test):
    """Plot Predictions vs Actual values."""
    fig, ax = plt.subplots(figsize=(12, 8))  # Create a new figure and axis object
    ax.scatter(y_test, predictions)
    ax.set_xlabel("Actual Delivery Time", fontsize=14, labelpad=15)
    ax.set_ylabel("Predicted Delivery Time", fontsize=14, labelpad=15)
    ax.set_title("Prediction vs Actual Delivery Time", fontsize=20, pad=25)
    ax.tick_params(axis='both', labelsize=14)
    return fig, ax  # Return fig and ax

def plot_feature_importance(model, feature_names):
    """Plot the feature importance from the model, ensuring alignment."""
    
    # Check if model has feature importances
    if not hasattr(model, "feature_importances_"):
        st.warning("The provided model does not have feature importances. Skipping feature importance visualization.")
        return None, None  # Return None if no feature importance

    feature_importances = model.feature_importances_

    # Ensure feature names and importances match in length
    min_length = min(len(feature_importances), len(feature_names))
    feature_importances = feature_importances[:min_length]
    feature_names = feature_names[:min_length]

    # Sort feature importances for better visualization
    sorted_indices = np.argsort(feature_importances)
    feature_names = np.array(feature_names)[sorted_indices]
    feature_importances = feature_importances[sorted_indices]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(feature_names, feature_importances, color=sns.color_palette("viridis", len(feature_names)))
    ax.set_xlabel("Importance", fontsize=16, labelpad=15)
    ax.set_ylabel("Features", fontsize=16, labelpad=15)
    ax.set_title("Feature Importance", fontsize=20, pad=25)
    ax.tick_params(axis='both', labelsize=14, rotation=20)

    return fig, ax

def plot_delivery_time_distribution(y_test):
    """Plot the distribution of delivery times."""
    fig, ax = plt.subplots(figsize=(12, 8))  # Create a new figure and axis object
    sns.histplot(y_test, bins=30, kde=True, color='teal', edgecolor='black', alpha=0.6)  # Using seaborn's histplot with advanced colors
    ax.set_xlabel("Delivery Time", fontsize=14, labelpad=15)
    ax.set_ylabel("Frequency", fontsize=14, labelpad=15)
    ax.set_title("Distribution of Delivery Times", fontsize=20, pad=25)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig, ax  # Return fig and ax
