from fastapi import FastAPI
from pydantic import BaseModel
from ml.src.predict import predict

app = FastAPI(title="California Housing Model")

class HousingInput(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def make_prediction(data: HousingInput):
    result = predict(data.model_dump())
    return {"predict": result}
