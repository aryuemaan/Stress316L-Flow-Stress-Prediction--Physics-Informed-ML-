import torch
import torch.nn as nn

class MLPRegressor(nn.Module):
    """
    Tabular regression model to predict stress given:
      x = [strain, strain_rate]
    """
    def __init__(self, input_dim: int = 2, hidden_dim: int = 256, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),

            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)

def physics_regularization(strain_scaled: torch.Tensor, pred_stress: torch.Tensor):
    """
    Physics-informed regularization:
    1) Monotonic constraint: stress should not decrease with increasing strain
    2) Smoothness constraint: avoid high-frequency oscillations

    strain_scaled: (N,)
    pred_stress:  (N,)
    """
    idx = torch.argsort(strain_scaled)
    s = strain_scaled[idx]
    f = pred_stress[idx]

    ds = s[1:] - s[:-1]
    df = f[1:] - f[:-1]

    slope = df / (ds + 1e-8)
    monotonic_penalty = torch.relu(-slope).mean()

    if len(f) >= 3:
        d2 = f[2:] - 2 * f[1:-1] + f[:-2]
        smoothness_penalty = (d2 ** 2).mean()
    else:
        smoothness_penalty = torch.tensor(0.0, device=f.device)

    return monotonic_penalty, smoothness_penalty
