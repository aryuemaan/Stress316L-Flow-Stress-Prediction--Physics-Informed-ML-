import os
import argparse
import yaml
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src.data import load_dataset
from src.models import MLPRegressor, physics_regularization
from src.metrics import regression_metrics
from src.utils import seed_everything, get_device, ensure_dir


class StressDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


@torch.no_grad()
def predict_torch(model, X_np, device, batch_size=2048):
    model.eval()
    preds = []
    for i in range(0, len(X_np), batch_size):
        xb = torch.tensor(X_np[i:i+batch_size], dtype=torch.float32).to(device)
        p = model(xb).detach().cpu().numpy().reshape(-1)
        preds.append(p)
    return np.concatenate(preds)


def save_scaler_npz(scaler, path: str):
    np.savez(path, mean=scaler.mean_, scale=scaler.scale_)


def train_one_fold(
    X_train, y_train, X_val, y_val,
    cfg, device
):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    train_ds = StressDataset(X_train_s, y_train)
    val_ds = StressDataset(X_val_s, y_val)

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    model = MLPRegressor(
        input_dim=2,
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["train"]["epochs"])
    loss_fn = nn.SmoothL1Loss()

    best_rmse = float("inf")
    best_state = None

    w_mono = cfg["physics"]["w_monotonic"]
    w_smooth = cfg["physics"]["w_smoothness"]

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            data_loss = loss_fn(pred, yb)

            strain_scaled = xb[:, 0].view(-1)
            pred_flat = pred.view(-1)

            mono_pen, smooth_pen = physics_regularization(strain_scaled, pred_flat)
            loss = data_loss + w_mono * mono_pen + w_smooth * smooth_pen

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * len(xb)

        total_loss /= len(train_ds)
        scheduler.step()

        # validation
        val_pred = predict_torch(model, X_val_s, device=device)
        mae, rmse, r2 = regression_metrics(y_val, val_pred)

        if rmse < best_rmse:
            best_rmse = rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 25 == 0:
            print(
                f"Epoch {epoch:3d} | TrainLoss={total_loss:.6f} | "
                f"Val RMSE={rmse:.4f} | MAE={mae:.4f} | R2={r2:.4f}"
            )

    model.load_state_dict(best_state)
    return model, scaler


def main(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg["train"]["seed"])
    device = get_device()
    print("Device:", device)

    df = load_dataset(cfg["paths"]["features_csv"], cfg["paths"]["labels_csv"])
    print("Dataset loaded:", df.shape)

    X = df[["strain", "strain_rate"]].values.astype(np.float32)
    y = df["stress"].values.astype(np.float32)
    groups = df["strain_rate"].values

    gkf = GroupKFold(n_splits=cfg["train"]["n_splits"])

    oof_pred = np.zeros(len(df), dtype=np.float32)
    fold_scores = []

    for fold, (tr, va) in enumerate(gkf.split(X, y, groups=groups), 1):
        print("\n==============================")
        print(f"Fold {fold}")
        print("Validation strain_rate:", sorted(np.unique(groups[va]).tolist()))
        print("==============================")

        model, scaler = train_one_fold(X[tr], y[tr], X[va], y[va], cfg, device=device)

        X_val_s = scaler.transform(X[va])
        val_pred = predict_torch(model, X_val_s, device=device)
        oof_pred[va] = val_pred

        mae, rmse, r2 = regression_metrics(y[va], val_pred)
        fold_scores.append((mae, rmse, r2))
        print(f"[Fold {fold}] MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f}")

    # Results Table summary
    maes = [s[0] for s in fold_scores]
    rmses = [s[1] for s in fold_scores]
    r2s = [s[2] for s in fold_scores]

    print("\nFinal GroupKFold Summary:")
    print(f"MAE  = {np.mean(maes):.4f} ± {np.std(maes):.4f}")
    print(f"RMSE = {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
    print(f"R2   = {np.mean(r2s):.4f} ± {np.std(r2s):.4f}")

    # Train final model on full dataset
    print("\nTraining FINAL model on full dataset for deployment...")
    ensure_dir(cfg["paths"]["artifacts_dir"])

    scaler_final = StandardScaler()
    X_full_s = scaler_final.fit_transform(X)

    train_ds = StressDataset(X_full_s, y)
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)

    final_model = MLPRegressor(
        input_dim=2,
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"]
    ).to(device)

    opt = torch.optim.AdamW(
        final_model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"]
    )
    loss_fn = nn.SmoothL1Loss()

    w_mono = cfg["physics"]["w_monotonic"]
    w_smooth = cfg["physics"]["w_smoothness"]

    for epoch in range(1, 201):
        final_model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = final_model(xb)
            data_loss = loss_fn(pred, yb)

            strain_scaled = xb[:, 0].view(-1)
            pred_flat = pred.view(-1)
            mono_pen, smooth_pen = physics_regularization(strain_scaled, pred_flat)

            loss = data_loss + w_mono * mono_pen + w_smooth * smooth_pen

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * len(xb)

        total_loss /= len(train_ds)
        if epoch == 1 or epoch % 25 == 0:
            print(f"[Final Train] Epoch {epoch:3d} | Loss={total_loss:.6f}")

    model_path = os.path.join(cfg["paths"]["artifacts_dir"], "stress316l_model.pt")
    scaler_path = os.path.join(cfg["paths"]["artifacts_dir"], "stress316l_scaler.npz")

    torch.save(final_model.state_dict(), model_path)
    save_scaler_npz(scaler_final, scaler_path)

    print("\nSaved artifacts:")
    print("Model :", model_path)
    print("Scaler:", scaler_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
