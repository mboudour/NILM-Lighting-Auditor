"""
Ablation Study and Computational Efficiency Analysis
Reviewer Revision: Action B (ablation) and Action D (latency)
"""
import torch
import torch.nn as nn
import numpy as np
import time
import pickle
import os

# ─── Load real preprocessed data ──────────────────────────────────────────────
print("Loading preprocessed profiles...")
with open("/home/ubuntu/computations_run/computations/outputs/preprocessed_daily_profiles.pkl", "rb") as f:
    daily_profiles = pickle.load(f)

# daily_profiles is a DataFrame; extract the 24-hour columns
profile_cols = [c for c in daily_profiles.columns if str(c).isdigit() or isinstance(c, int)]
if not profile_cols:
    # try numeric column detection
    profile_cols = list(range(24))
    data_array = daily_profiles.iloc[:, :24].values.astype(np.float32)
else:
    data_array = daily_profiles[profile_cols].values.astype(np.float32)

print(f"Loaded {len(data_array)} profiles of shape {data_array.shape}")

# Use a subsample for speed
np.random.seed(42)
idx = np.random.choice(len(data_array), size=5000, replace=False)
X = torch.tensor(data_array[idx], dtype=torch.float32)

# ─── N-BEATS architecture (matching stage2_disagg.py) ─────────────────────────
class NBEATSBlock(nn.Module):
    def __init__(self, input_size=24, hidden=128, with_basis=True):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.backcast_fc = nn.Linear(hidden, input_size)
        self.forecast_fc = nn.Linear(hidden, input_size)
        self.with_basis = with_basis

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        h = torch.relu(self.fc3(h))
        backcast = self.backcast_fc(h)
        forecast = self.forecast_fc(h)
        return backcast, forecast


class NBEATSFull(nn.Module):
    """Full model: Trend Block + Seasonality Block"""
    def __init__(self):
        super().__init__()
        self.trend_block = NBEATSBlock()
        self.season_block = NBEATSBlock()

    def forward(self, x):
        trend_back, trend_fore = self.trend_block(x)
        residual = x - trend_back
        season_back, season_fore = self.season_block(residual)
        recon = trend_fore + season_fore
        return recon


class NBEATSNoTrend(nn.Module):
    """Ablation: Seasonality Block only (no Trend Block)"""
    def __init__(self):
        super().__init__()
        self.season_block = NBEATSBlock()

    def forward(self, x):
        _, season_fore = self.season_block(x)
        return season_fore


class NBEATSNoSeason(nn.Module):
    """Ablation: Trend Block only (no Seasonality Block)"""
    def __init__(self):
        super().__init__()
        self.trend_block = NBEATSBlock()

    def forward(self, x):
        _, trend_fore = self.trend_block(x)
        return trend_fore


# ─── Training function ─────────────────────────────────────────────────────────
def train_model(model, X, epochs=50, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, X)
        loss.backward()
        optimizer.step()
    return loss.item()


# ─── Run ablation ──────────────────────────────────────────────────────────────
print("\n--- Ablation Study ---")
results = {}

for name, ModelClass in [
    ("Full N-BEATS (Trend + Seasonality)", NBEATSFull),
    ("Ablation: No Trend Block", NBEATSNoTrend),
    ("Ablation: No Seasonality Block", NBEATSNoSeason),
]:
    model = ModelClass()
    t0 = time.time()
    mse = train_model(model, X)
    elapsed = time.time() - t0
    results[name] = {"mse": mse, "train_time_s": elapsed}
    print(f"{name}: MSE={mse:.4f}, Training time={elapsed:.1f}s")

# ─── Inference latency ─────────────────────────────────────────────────────────
print("\n--- Inference Latency ---")
full_model = NBEATSFull()
full_model.eval()
with torch.no_grad():
    # Warmup
    _ = full_model(X[:10])
    # Timed run
    t0 = time.time()
    _ = full_model(X)
    elapsed_inf = (time.time() - t0)

profiles_per_sec = len(X) / elapsed_inf
ms_per_profile = elapsed_inf / len(X) * 1000
print(f"Inference: {profiles_per_sec:.0f} profiles/sec, {ms_per_profile:.4f} ms/profile")

# ─── Save results ──────────────────────────────────────────────────────────────
import json
output = {
    "ablation": results,
    "inference_latency": {
        "profiles_per_second": profiles_per_sec,
        "ms_per_profile": ms_per_profile
    }
}
with open("/home/ubuntu/computations_run/computations/outputs/ablation_results.json", "w") as f:
    json.dump(output, f, indent=2)
print("\nResults saved to ablation_results.json")
