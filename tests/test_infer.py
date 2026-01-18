import os
import pytest

from src.infer import load_model, load_scaler_npz, predict_stress
from src.utils import get_device

@pytest.mark.skipif(
    not os.path.exists("artifacts/stress316l_model.pt"),
    reason="Train the model first to generate artifacts."
)
def test_inference_runs():
    device = get_device()
    model = load_model("artifacts/stress316l_model.pt", device=device)
    scaler = load_scaler_npz("artifacts/stress316l_scaler.npz")

    pred = predict_stress(0.10, 1.0, model=model, scaler=scaler, device=device)
    assert isinstance(pred, float)
