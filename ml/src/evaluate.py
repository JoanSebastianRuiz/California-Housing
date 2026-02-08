from pathlib import Path
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from data import load_data

def evaluate():
    MODEL_PATH = Path(__file__).resolve().parent.parent / 'artifacts' / 'model.pkl'
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Train the model first.")
    
    model = joblib.load(MODEL_PATH)
    
    _, X_test, _, y_test = load_data()
    
    y_pred = model.predict(X_test)
    
    print('Test R2: ', r2_score(y_test, y_pred))
    print('Test MAE: ', mean_absolute_error(y_test, y_pred))
    print('Test MSE: ', mean_squared_error(y_test, y_pred))
    
if __name__ == '__main__':
    evaluate()