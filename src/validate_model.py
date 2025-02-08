import joblib
import os

model_path = "models/trained_models/best_model.pkl"

# Load the model
try:
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("✅ Model loaded successfully!")
        print("Model type:", type(model))
    else:
        print("❌ Model file not found!")
except Exception as e:
    print("❌ Error loading model:", e)
