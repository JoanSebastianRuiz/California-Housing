import joblib
from pathlib import Path
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
from model import build_model
from data import load_data

def train():
    model = build_model()
    X_train, _, y_train, _ = load_data()
    
    param_distributions = {
        'model__n_estimators': randint(100, 500),
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': randint(2, 10),
        'model__min_samples_leaf': randint(1, 5),
        'model__max_features': ['sqrt', 'log2']
    }
    
    search = RandomizedSearchCV(
        model,
        param_distributions=param_distributions,
        n_iter=30,
        scoring='r2',
        cv=5,
        n_jobs=-1,
        random_state=42
    )
    
    search.fit(X_train, y_train)
    print('Best CV R2: ', search.best_score_)
    
    best_model = search.best_estimator_
    
    ARTIFACTS_PATH = Path(__file__).resolve().parent.parent / 'artifacts'
    ARTIFACTS_PATH.mkdir(exist_ok=True)
    joblib.dump(best_model, ARTIFACTS_PATH / 'model.pkl')
    
if __name__ == '__main__':
    train()