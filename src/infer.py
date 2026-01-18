import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.models import MLPRegressor
from src.utils import get_device


def load_scaler_npz(npz_path: str) -> StandardScaler:
    data = np.load(npz_path)

    scaler = StandardScaler()
    scaler.mean_ = data["mean"]
    scaler.scale_ = data["scale"]
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)

    return scaler


def load_model(model_path: str, device=None, hidden_dim=256) -> MLPRegressor:
    if device is None:
        device = get_device()

    model = MLPRegressor(input_dim=2, hidden_dim=hidden_dim, dropout=0.0)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_stress(strain: float, strain_rate: float, model, scaler, device=None) -> float:
    if device is None:
        device = get_device()

    x = np.array([[strain, strain_rate]], dtype=np.float32)
    x_s = scaler.transform(x)

    with torch.no_grad():
        pred = model(torch.tensor(x_s, dtype=torch.float32).to(device)).cpu().numpy().reshape(-1)[0]

    return float(pred)
