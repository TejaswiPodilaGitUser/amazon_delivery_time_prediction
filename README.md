
# Amazon Delivery Time Prediction

## Overview

This project aims to predict the delivery time for Amazon e-commerce orders based on various factors such as product size, shipping distance, traffic conditions, and shipping method. The prediction model is built using machine learning techniques, and the results are presented interactively through a web app built with **Streamlit**.

## Technologies Used

- **Python**: For data processing, machine learning model building, and web app development.
- **Machine Learning**: Algorithms like **Random Forest** to predict delivery times based on the input features.
- **MLflow**: For experiment tracking, model management, and evaluation.
- **Streamlit**: For creating an interactive web interface to visualize predictions.
- **Git/GitHub**: For version control and collaboration.

## Files in the Project

- **data/**: Contains raw and cleaned data used for model training and evaluation.
- **models/**: Stores trained machine learning models in `.pkl` format.
  - `best_model.pkl`: The best-performing machine learning model.
- **mlruns/**: Stores logs from MLflow experiments.
- **app.py**: Streamlit app for the user interface.
- **requirements.txt**: Lists the Python packages required to run the project.
- **README.md**: This file, which provides an overview of the project.
- **.gitignore**: Specifies files and directories that should not be tracked by Git (e.g., large model files).

## Installation

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/TejaswiPodilaGitUser/amazon_delivery_time_prediction.git
   cd amazon_delivery_time_prediction
   ```

2. Create a virtual environment (optional, but recommended):

   ```bash
    python -m venv venv
    source venv/bin/activate  # For macOS/Linux
    venv\\Scripts\\activate  # For Windows
    ```
3. Install the required dependencies:
    ```bash
   pip install -r requirements.txt
    ```

## How to Run

1. **Train the model (if needed)**:
   If you want to retrain the model, run the `train_model.py` script which will preprocess the data, train the machine learning model, and save the best model as a `.pkl` file.

   ```bash
   python train_model.py
   ```

2. **Run the Streamlit app**:
   After training the model, run the Streamlit app for an interactive user interface:

   ```bash
   streamlit run app.py
   ```

   The app will open in your browser, allowing you to input various delivery parameters and view the predicted delivery time. """


## Learnings

- **Machine Learning**: Gained hands-on experience in building machine learning models, especially using **Random Forest**, to predict delivery times based on various features.
- **Data Preprocessing**: Learned how to clean and preprocess data, handle missing values, and transform data to make it suitable for model training.
- **MLflow**: Familiarized with **MLflow** for tracking experiments, managing models, and evaluating results.
- **Streamlit**: Learned to create interactive web applications with **Streamlit** to visualize predictions and results in a user-friendly manner.
- **Version Control**: Worked with **Git/GitHub** for version control and collaboration.
"""
## License

This project is open source and licensed under the MIT License - see the LICENSE file for details.
