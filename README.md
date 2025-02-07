# Amazon Delivery Time Prediction

## Overview

This project aims to predict the delivery time for Amazon e-commerce orders based on various factors such as product size, shipping distance, traffic conditions, and shipping method. The prediction model is built using machine learning techniques, and the results are presented interactively through a web app built with **Streamlit**.

## Technologies Used

- **Python**: For data processing, machine learning model building, and web app development.
- **Machine Learning**: Algorithms like **Random Forest** to predict delivery times based on the input features.
- **MLflow**: For experiment tracking, model management, and evaluation.
- **Streamlit**: For creating an interactive web interface to visualize predictions.
- **Git/GitHub**: For version control and collaboration.

## Project Structure

```
amazon_delivery_time_prediction/
│── src/
│   ├── app.py                  # Streamlit app for user interaction
│   ├── charts.py               # Visualization functions
│   ├── data_preparation.py     # Data loading and preprocessing
│   ├── feature_engineering.py  # Feature extraction & transformation
│   ├── model_tracking.py       # MLflow model tracking
│   ├── model_training.py       # Training machine learning models
│   ├── model_tuning.py         # Hyperparameter tuning
│── streamlit_app/
│   ├── main.py                 # Streamlit main entry point
│── tests/
│   ├── test_model_pipeline.py  # Unit tests for model pipeline
│── models/                     # Stores trained models (e.g., best_model.pkl)
│── mlruns/                     # MLflow logs & experiment tracking
│── requirements.txt            # Dependencies required for the project
│── README.md                   # Project documentation
│── .gitignore                   # Files to be ignored by Git
```

## Visualizations

This section presents the visualizations that showcase the model performance and predictions.

<img width="1696" alt="Visualization 1" src="https://github.com/user-attachments/assets/b0cdb760-0b66-43f6-a80c-642533162506" />

<img width="1706" alt="Visualization 2" src="https://github.com/user-attachments/assets/c64116c4-ea0e-4441-a9ba-596ab8938b34" />

### Streamlit UI

<img width="837" alt="Streamlit UI 1" src="https://github.com/user-attachments/assets/99ed6d11-c3e6-4ccf-9677-4810a2e21054" />

<img width="1703" alt="Streamlit UI 2" src="https://github.com/user-attachments/assets/6fabf145-5e3c-4a6a-920f-27b5c2c2249e" />

<img width="811" alt="Streamlit UI 3" src="https://github.com/user-attachments/assets/a17c2609-5c0e-4a87-9cf5-331d00586e38" />

## Installation Guide

1. **Clone the Repository**
   ```bash
   git clone https://github.com/TejaswiPodilaGitUser/amazon_delivery_time_prediction.git
   cd amazon_delivery_time_prediction
   ```

2. **Create a Virtual Environment** (Optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # For macOS/Linux
   venv\Scripts\activate  # For Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

### 1. Train the Model (if needed)

If you want to retrain the model, run the `model_training.py` script, which preprocesses the data, trains the model, and saves the best-performing model as a `.pkl` file.

```bash
python src/model_training.py
```

### 2. Run the Streamlit App

Once the model is trained, start the Streamlit app to interact with the predictions.

```bash
streamlit run src/app.py
```

The app will open in your browser, allowing you to input various delivery parameters and view the predicted delivery time.

## Testing

To test the model pipeline, run:

```bash
PYTHONPATH=src pytest tests/test_model_pipeline.py
```

This runs unit tests to validate the preprocessing, model predictions, and edge cases.

## Learnings

- **Machine Learning**: Built an ML model using **Random Forest** for delivery time prediction.
- **Data Preprocessing**: Cleaned and prepared data for training.
- **MLflow**: Used MLflow for model tracking and experiment management.
- **Streamlit**: Developed an interactive web app to visualize predictions.
- **Software Engineering**: Followed best practices in structuring the project, testing, and version control with Git/GitHub.

## License

This project is open source and licensed under the MIT License - see the LICENSE file for details.

