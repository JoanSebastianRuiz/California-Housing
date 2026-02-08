from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent.parent / 'artifacts' / 'model.pkl'
model = joblib.load(MODEL_PATH)

def predict(input_data: dict):
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)
    return prediction[0]