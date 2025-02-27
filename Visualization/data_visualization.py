import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np

def load_data(file_path):
    """Load dataset and compute additional features."""
    print("Loading dataset...")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

    print("Dataset Loaded Successfully!")
    print("Columns in Dataset:", df.columns)
    print("First 5 rows:\n", df.head())
    print("Missing Values:\n", df.isnull().sum())

    # Drop rows with missing values
    df = df.dropna()

    # Ensure required columns exist before calculation
    required_columns = ['Store_Latitude', 'Store_Longitude', 'Drop_Latitude', 'Drop_Longitude', 'Delivery_Time']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Missing column {col}")
            return None

    # Compute Euclidean distance
    df['Distance'] = np.sqrt((df['Store_Latitude'] - df['Drop_Latitude'])**2 + 
                             (df['Store_Longitude'] - df['Drop_Longitude'])**2)

    print("Data processing completed!")
    return df

def plot_delivery_time_distribution(df, ax):
    print("Plotting: Delivery Time Distribution")
    sns.histplot(df['Delivery_Time'], bins=30, kde=True, ax=ax, color="blue")
    ax.set_title("Distribution of Delivery Times", fontsize=14)
    ax.set_xlabel("Delivery Time (minutes)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.tick_params(axis='x', rotation=25)

def plot_weather_impact(df, ax):
    if 'Weather' in df.columns:
        sns.boxplot(x=df['Weather'], y=df['Delivery_Time'], ax=ax, palette="coolwarm")
        ax.set_title("Impact of Weather on Delivery Time", fontsize=14)
        ax.set_xlabel("Weather Condition", fontsize=12)
        ax.set_ylabel("Delivery Time (minutes)", fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")

def plot_traffic_impact(df, ax):
    if 'Traffic' in df.columns:
        sns.boxplot(x=df['Traffic'], y=df['Delivery_Time'], ax=ax, palette="coolwarm")
        ax.set_title("Impact of Traffic on Delivery Time", fontsize=14)
        ax.set_xlabel("Traffic Level", fontsize=12)
        ax.set_ylabel("Delivery Time (minutes)", fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")

def plot_category_wise_delivery(df, ax):
    if 'Category' in df.columns:
        sns.barplot(x=df['Category'], y=df['Delivery_Time'], ax=ax, palette="viridis")
        ax.set_title("Delivery Time by Product Category", fontsize=14)
        ax.set_xlabel("Product Category", fontsize=12)
        ax.set_ylabel("Delivery Time (minutes)", fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")

def plot_agent_performance(df, ax):
    if 'Agent_Rating' in df.columns:
        sns.barplot(x=df['Agent_Rating'], y=df['Delivery_Time'], ax=ax, palette="coolwarm")
        ax.set_title("Agent Rating vs. Delivery Time", fontsize=14)
        ax.set_xlabel("Agent Rating", fontsize=12)
        ax.set_ylabel("Delivery Time (minutes)", fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")

def plot_distance_vs_delivery(df, ax):
    print("Plotting: Distance vs Delivery Time")
    sns.scatterplot(x=df['Distance'], y=df['Delivery_Time'], ax=ax, color="green")
    ax.set_title("Distance vs. Delivery Time", fontsize=14)
    ax.set_xlabel("Distance (km)", fontsize=12)
    ax.set_ylabel("Delivery Time (minutes)", fontsize=12)
    ax.tick_params(axis='x', rotation=25)

def plot_correlation_heatmap(df, ax):
    print("Plotting: Correlation Heatmap")

    # Select only numerical columns
    numeric_df = df.select_dtypes(include=['number'])

    if numeric_df.empty:
        print("No numerical columns found for correlation heatmap.")
        return

    corr = numeric_df.corr()

    # Increase figure size for readability
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="magma", cbar=True, ax=ax, 
                annot_kws={"size": 10}, linewidths=0.5, linecolor="gray")

    ax.set_title("Correlation Matrix", fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)

def plot_sla_breach_distribution(df, ax):
    if 'Delivery_SLA_Breach' in df.columns:
        print("Plotting: SLA Breach Distribution")
        pie_data = df['Delivery_SLA_Breach'].value_counts()
        if len(pie_data) < 2:
            pie_data = pie_data.reindex([0, 1], fill_value=0)
        ax.pie(pie_data, labels=["No Breach", "Breach"], autopct='%1.1f%%', startangle=90, colors=["green", "red"])
        ax.set_title("SLA Breach Distribution", fontsize=14)

def plot_agent_age_vs_delivery(df, ax):
    if 'Agent_Age' in df.columns:
        sns.lineplot(x=df['Agent_Age'], y=df['Delivery_Time'], ax=ax, color="blue")
        ax.set_title("Agent Age vs. Delivery Time", fontsize=14)
        ax.set_xlabel("Agent Age", fontsize=12)
        ax.set_ylabel("Delivery Time (minutes)", fontsize=12)
        ax.tick_params(axis='x', rotation=25)

def plot_order_hour_vs_delivery(df, ax):
    if 'Order_Hour' in df.columns:
        sns.lineplot(x=df['Order_Hour'], y=df['Delivery_Time'], ax=ax, color="purple")
        ax.set_title("Order Hour vs. Delivery Time", fontsize=14)
        ax.set_xlabel("Order Hour", fontsize=12)
        ax.set_ylabel("Delivery Time (minutes)", fontsize=12)
        ax.tick_params(axis='x', rotation=25)

def generate_plots(file_path, selected_options):
    df = load_data(file_path)
    if df is None:
        print("Error: Dataset could not be loaded or is missing required columns.")
        return

    plot_map = {
        "Agent Details": [plot_agent_performance, plot_agent_age_vs_delivery],
        "Weather & Traffic": [plot_weather_impact, plot_traffic_impact],
        "Order Details": [plot_category_wise_delivery, plot_order_hour_vs_delivery, plot_distance_vs_delivery],
        "SLA Analysis": [plot_sla_breach_distribution, plot_correlation_heatmap],
        "All": []  # "All" will include all plots
    }

    if "All" in selected_options:
        selected_plots = [func for plots in plot_map.values() for func in plots]
    else:
        selected_plots = []
        for key in selected_options:
            if key in plot_map:
                selected_plots.extend(plot_map[key])

    if not selected_plots:
        selected_plots = [plot_delivery_time_distribution]

    num_plots = len(selected_plots)
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(10, 6 * num_plots))

    if num_plots == 1:
        selected_plots[0](df, axes)
    else:
        for func, ax in zip(selected_plots, axes):
            func(df, ax)

    plt.tight_layout()
    st.pyplot(fig)

if __name__ == "__main__":
    generate_plots("data/processed/engineered_data.csv", ["All"])
