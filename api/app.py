from fastapi import FastAPI
from pydantic import BaseModel

from src.infer import load_model, load_scaler_npz, predict_stress
from src.utils import get_device

MODEL_PATH = "artifacts/stress316l_model.pt"
SCALER_PATH = "artifacts/stress316l_scaler.npz"

app = FastAPI(title="Stress316L Flow Stress Predictor", version="1.0.0")

device = get_device()
model = load_model(MODEL_PATH, device=device)
scaler = load_scaler_npz(SCALER_PATH)

class PredictRequest(BaseModel):
    strain: float
    strain_rate: float

class PredictResponse(BaseModel):
    stress: float


@app.get("/")
def home():
    return {
        "message": "Stress316L Flow Stress Predictor API is running.",
        "routes": {
            "health": "GET /health",
            "predict": "POST /predict"
        }
    }


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    stress = predict_stress(req.strain, req.strain_rate, model=model, scaler=scaler, device=device)
    return PredictResponse(stress=stress)
