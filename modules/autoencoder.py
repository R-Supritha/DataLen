"""
autoencoder.py
PyTorch Autoencoder trained on numeric dataset features.
Detects anomalies via reconstruction error.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────
#  Model Definition
# ─────────────────────────────────────────────

class TabularAutoencoder(nn.Module):
    """
    Symmetric encoder-decoder for tabular data.
    Bottleneck forces the model to learn a compressed representation;
    high reconstruction error → anomalous sample.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        h1 = max(64, input_dim * 4)
        h2 = max(32, input_dim * 2)
        bottleneck = max(8, input_dim)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, bottleneck),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, h2),
            nn.ReLU(),
            nn.Linear(h2, h1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(h1, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


# ─────────────────────────────────────────────
#  Training & Inference
# ─────────────────────────────────────────────

def _prepare_tensor(df: pd.DataFrame):
    """Scale numeric data and return tensor + scaler."""
    num_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    X = num_df.fillna(num_df.median()).values.astype(np.float32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    tensor = torch.from_numpy(X_scaled)
    return tensor, scaler, num_df.columns.tolist()


def train_autoencoder(
    df: pd.DataFrame,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
    threshold_percentile: float = 95.0,
) -> dict:
    """
    Train a TabularAutoencoder on numeric features.
    Returns anomaly scores, a per-row flag, and a summary.
    """
    tensor, scaler, feature_cols = _prepare_tensor(df)
    n, input_dim = tensor.shape

    if input_dim == 0:
        return {
            "error": "No numeric columns found for autoencoder training.",
            "anomaly_scores": [],
            "anomaly_indices": [],
            "threshold": None,
            "feature_columns": [],
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = tensor.to(device)

    model = TabularAutoencoder(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(tensor)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    # ── Training loop ──
    model.train()
    loss_history = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
        loss_history.append(round(epoch_loss / n, 6))

    # ── Reconstruction errors ──
    model.eval()
    with torch.no_grad():
        recon = model(tensor)
        errors = torch.mean((recon - tensor) ** 2, dim=1).cpu().numpy()

    threshold = float(np.percentile(errors, threshold_percentile))
    anomaly_mask = errors > threshold
    anomaly_indices = np.where(anomaly_mask)[0].tolist()

    # Normalise scores to [0, 1] for display
    norm_scores = (errors - errors.min()) / (errors.max() - errors.min() + 1e-9)

    return {
        "anomaly_scores": norm_scores.round(4).tolist(),
        "raw_errors": errors.round(6).tolist(),
        "anomaly_indices": anomaly_indices,
        "anomaly_count": int(anomaly_mask.sum()),
        "anomaly_percentage": round(float(anomaly_mask.mean()) * 100, 2),
        "threshold": round(threshold, 6),
        "threshold_percentile": threshold_percentile,
        "feature_columns": feature_cols,
        "epochs_trained": epochs,
        "final_loss": loss_history[-1] if loss_history else None,
        "loss_history": loss_history,
    }