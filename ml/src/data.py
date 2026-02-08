from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

def load_data():
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    
    X = df.drop(columns=['MedHouseVal'])
    y = df['MedHouseVal']
    
    return train_test_split(
        X, y, test_size=0.2, random_state=42
    )