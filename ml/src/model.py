from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

def build_model(**params):
    return Pipeline([
        ('model', RandomForestRegressor(**params))
    ])