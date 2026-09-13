from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "model"
    / "crop_recommendation_model.pkl"
)

ENCODER_PATH = (
    PROJECT_ROOT
    / "model"
    / "label_encoder.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(MODEL_PATH)

label_encoder = joblib.load(ENCODER_PATH)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="AI Smart Agriculture - Crop Recommendation API",
    description=(
        "Machine Learning API for crop recommendation "
        "using soil and environmental parameters."
    ),
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class CropRequest(BaseModel):

    N: float = Field(
        ge=0,
        description="Nitrogen content"
    )

    P: float = Field(
        ge=0,
        description="Phosphorus content"
    )

    K: float = Field(
        ge=0,
        description="Potassium content"
    )

    temperature: float = Field(
        description="Temperature in Celsius"
    )

    humidity: float = Field(
        ge=0,
        le=100,
        description="Humidity percentage"
    )

    ph: float = Field(
        ge=0,
        le=14,
        description="Soil pH"
    )

    rainfall: float = Field(
        ge=0,
        description="Rainfall"
    )


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": (
            "AI Smart Agriculture "
            "Crop Recommendation API"
        ),
        "status": "running",
        "version": "1.0.0"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "encoder_loaded": label_encoder is not None
    }


# ============================================================
# CROP PREDICTION
# ============================================================

@app.post("/predict-crop")
def predict_crop(data: CropRequest):

    # --------------------------------------------------------
    # Create input dataframe
    # --------------------------------------------------------

    features = pd.DataFrame([{
        "N": data.N,
        "P": data.P,
        "K": data.K,
        "temperature": data.temperature,
        "humidity": data.humidity,
        "ph": data.ph,
        "rainfall": data.rainfall
    }])


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction_encoded = model.predict(features)[0]

    prediction_encoded = int(
        prediction_encoded
    )


    # --------------------------------------------------------
    # Convert encoded class back to crop name
    # --------------------------------------------------------

    predicted_crop = label_encoder.inverse_transform(
        [prediction_encoded]
    )[0]


    # --------------------------------------------------------
    # Prediction probabilities
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        features
    )[0]


    # --------------------------------------------------------
    # Create crop-probability pairs
    # --------------------------------------------------------

    results = []

    for encoded_class, probability in zip(
        model.classes_,
        probabilities
    ):

        encoded_class = int(encoded_class)

        crop_name = label_encoder.inverse_transform(
            [encoded_class]
        )[0]

        results.append({
            "crop": str(crop_name),
            "confidence": round(
                float(probability) * 100,
                2
            )
        })


    # --------------------------------------------------------
    # Sort by confidence
    # --------------------------------------------------------

    results.sort(
        key=lambda item: item["confidence"],
        reverse=True
    )


    # --------------------------------------------------------
    # Return Top 3
    # --------------------------------------------------------

    return {
        "recommended_crop": str(
            predicted_crop
        ),

        "recommendations": results[:3]
    }