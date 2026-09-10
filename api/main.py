from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path

app = FastAPI()

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "crop_recommendation_model.pkl"
model = joblib.load(MODEL_PATH)


class CropRequest(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


@app.post("/predict-crop")
def predict_crop(data: CropRequest):

    features = [[
        data.N,
        data.P,
        data.K,
        data.temperature,
        data.humidity,
        data.ph,
        data.rainfall
    ]]

    prediction = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]

    results = sorted(
        zip(model.classes_, probabilities),
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = [
        {
            "crop": crop,
            "confidence": round(float(score), 4)
        }
        for crop, score in results[:3]
    ]

    return {
        "recommended_crop": prediction,
        "recommendations": recommendations
    }