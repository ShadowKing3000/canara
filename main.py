from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = FastAPI()

# Load model and scaler at startup
autoencoder = load_model("autoencoder_model.h5", compile=False)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
scaler = joblib.load("scaler.pkl")
THRESHOLD = 0.01  # Your empirical threshold


class InputSample(BaseModel):
    typing_speed: float
    tap_pressure: float
    swipe_velocity: float
    gesture_duration: float
    orientation_variance: float


class PredictionResult(BaseModel):
    reconstruction_error: float
    classification: str


@app.post("/detect-anomaly/", response_model=list[PredictionResult])
async def detect_anomaly(samples: list[InputSample]):
    # Convert input to DataFrame
    input_data = pd.DataFrame([s.dict() for s in samples])

    # Preprocess input
    scaled_input = scaler.transform(input_data)

    # Make prediction
    reconstructed = autoencoder.predict(scaled_input)
    reconstruction_error = np.mean((scaled_input - reconstructed) ** 2, axis=1)

    # Prepare results
    results = []
    for error in reconstruction_error:
        results.append(PredictionResult(
            reconstruction_error=float(error),
            classification="Anomaly" if error > THRESHOLD else "Normal"
        ))

    return results


@app.get("/")
def health_check():
    return {"status": "active", "message": "Anomaly detection API is running"}