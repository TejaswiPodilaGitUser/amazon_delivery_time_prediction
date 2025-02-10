import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
#df = pd.read_csv("data/raw/amazon_delivery.csv")
df = pd.read_csv("data/processed/cleaned_data.csv")


# Select only numeric columns
numeric_df = df.select_dtypes(include=[np.number])

# Create a single figure with adjusted layout
fig, axes = plt.subplots(2, 2, figsize=(16, 12), 
                         gridspec_kw={'width_ratios': [1.5, 1], 'height_ratios': [1, 1.5]})

# 1️⃣ Histogram - Delivery Time Distribution (First row, first column - Wider)
sns.histplot(df["Delivery_Time"], bins=30, kde=True, color="blue", ax=axes[0, 0])
axes[0, 0].set_xlabel("Delivery Time (minutes)")
axes[0, 0].set_ylabel("Frequency")
axes[0, 0].set_title("Distribution of Delivery Time")

# 2️⃣ Boxplot - Detecting Outliers (First row, second column - Narrower)
sns.boxplot(x=df["Delivery_Time"], color="red", ax=axes[0, 1])
axes[0, 1].set_xlabel("Delivery Time (minutes)")
axes[0, 1].set_title("Boxplot for Outlier Detection in Delivery Time")

# 3️⃣ Correlation Heatmap - Full-width (Second row spanning both columns)
corr_matrix = numeric_df.corr()
heatmap = sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5, ax=axes[1, 0])

# Rotate x-axis labels by 45 degrees for better readability
axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45, ha='right')

# Remove the empty second subplot in the second row
fig.delaxes(axes[1, 1])

# Adjust layout
plt.tight_layout()
plt.show()
