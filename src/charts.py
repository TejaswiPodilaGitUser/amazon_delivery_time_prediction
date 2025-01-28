import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st

def plot_predictions_vs_actual(predictions, y_test):
    """Plot Predictions vs Actual values."""
    fig, ax = plt.subplots(figsize=(12, 8))  # Create a new figure and axis object
    ax.scatter(y_test, predictions)
    ax.set_xlabel("Actual Delivery Time", fontsize=14)  # Increase fontsize here
    ax.set_ylabel("Predicted Delivery Time", fontsize=14)  # Increase fontsize here
    ax.set_title("Prediction vs Actual Delivery Time", fontsize=16, fontweight='bold')  # Make title bold
    ax.tick_params(axis='both', labelsize=12)  # Rotate tick labels
    return fig, ax  # Return fig and ax

def plot_feature_importance(model, feature_names):
    """Plot the feature importance from the model."""
    fig, ax = plt.subplots(figsize=(12, 8))  # Create a new figure and axis object
    feature_importances = model.feature_importances_
    ax.barh(feature_names, feature_importances, color=sns.color_palette("viridis", len(feature_names)))  # Color palette
    ax.set_xlabel("Importance", fontsize=14, color='darkblue')  # Advanced color
    ax.set_ylabel("Features", fontsize=14, color='darkblue')  # Advanced color
    ax.set_title("Feature Importance", fontsize=16, fontweight='bold', color='darkgreen')  # Make title bold
    ax.tick_params(axis='both', labelsize=12, rotation=20)  # Rotate tick labels
    return fig, ax  # Return fig and ax

def plot_delivery_time_distribution(y_test):
    """Plot the distribution of delivery times."""
    fig, ax = plt.subplots(figsize=(12, 8))  # Create a new figure and axis object
    sns.histplot(y_test, bins=30, kde=True, color='teal', edgecolor='black', alpha=0.6)  # Using seaborn's histplot with advanced colors
    ax.set_xlabel("Delivery Time", fontsize=14, color='darkblue')  # Advanced color
    ax.set_ylabel("Frequency", fontsize=14, color='darkblue')  # Advanced color
    ax.set_title("Distribution of Delivery Times", fontsize=16, fontweight='bold', color='darkgreen')  # Make title bold
    ax.tick_params(axis='both', labelsize=12)  # Rotate tick labels
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig, ax  # Return fig and ax
