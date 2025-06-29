from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = FastAPI()

# Load models at startup
autoencoder = load_model("autoencoder_model.h5", compile=False)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
scaler = joblib.load("scaler.pkl")
THRESHOLD = 0.01


class InputSample(BaseModel):
    typing_speed: float
    tap_pressure: float
    swipe_velocity: float
    gesture_duration: float
    orientation_variance: float


class PredictionResult(BaseModel):
    reconstruction_error: float
    classification: str


@app.post("/detect-anomaly/")
def detect_anomaly(input: InputSample):
    # Convert input to numpy array directly (no pandas)
    X = np.array([[
        input.typing_speed,
        input.tap_pressure,
        input.swipe_velocity,
        input.gesture_duration,
        input.orientation_variance
    ]])

    # Scale and predict
    scaled_input = scaler.transform(X)
    reconstructed = autoencoder.predict(scaled_input)
    error = np.mean((scaled_input - reconstructed) ** 2)

    return PredictionResult(
        reconstruction_error=float(error),
        classification="Anomaly" if error > THRESHOLD else "Normal"
    )


@app.get("/")
def health_check():
    return {"status": "active", "model_loaded": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
