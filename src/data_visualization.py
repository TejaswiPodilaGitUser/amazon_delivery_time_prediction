import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_delivery_time_distribution(df):
    """
    Plot the histogram and KDE of delivery time distribution.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(df["Delivery_Time"], bins=30, kde=True, color="blue", ax=ax)
    ax.set_xlabel("Delivery Time (minutes)")
    ax.set_ylabel("Frequency")
    ax.set_title("Delivery Time Distribution")
    return fig, ax

def plot_outlier_detection(df):
    """
    Plot a boxplot to detect outliers in the delivery time.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x=df["Delivery_Time"], color="red", ax=ax)
    ax.set_xlabel("Delivery Time (minutes)")
    ax.set_title("Outlier Detection in Delivery Time")
    return fig, ax

def plot_feature_correlation(df):
    """
    Plot a heatmap of feature correlations in the dataset.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_title("Feature Correlation Heatmap")
    return fig, ax

def plot_distance_vs_delivery_time(df):
    """
    Plot a scatter plot of Distance vs. Delivery Time.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=df["Distance"], y=df["Delivery_Time"], color="green", alpha=0.5, s=50, ax=ax)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Delivery Time (minutes)")
    ax.set_title("Distance vs. Delivery Time")
    return fig, ax
