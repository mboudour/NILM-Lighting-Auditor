# stage2_disagg.py
#
# CHECKPOINTING: Every expensive step saves its result to disk on first run.
#                Re-running skips any step whose output already exists.
#
# Checkpoints:
#   outputs/stage2/nbeats_disaggregator.pt   N-BEATS model weights
#   outputs/stage2/disaggregated_signals.pkl disaggregated trend + lighting
#   outputs/baselines/baseline_comparison.csv STL + GB baseline results
#   outputs/baselines/cross_climate_results.csv cross-climate generalisation
#   outputs/sensitivity/sensitivity_results.csv sensitivity analysis
#   outputs/case_study/building_<id>_*.png/csv  case study figures
#   outputs/plots/final_lighting_profiles.png   main paper figure
#
# Run after: stage1_clustering.py

import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from rich.console import Console
from rich.table import Table
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import STL
from tqdm import tqdm

CONSOLE = Console()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CLUSTERED_DATA_PATH = "outputs/stage1/clustered_daily_profiles.pkl"
OUTPUT_DIR          = "outputs/stage2"
PLOTS_DIR           = "outputs/plots"
BASELINES_DIR       = "outputs/baselines"
SENSITIVITY_DIR     = "outputs/sensitivity"
CASE_STUDY_DIR      = "outputs/case_study"

MODEL_PATH          = os.path.join(OUTPUT_DIR,    "nbeats_disaggregator.pt")
RESULTS_PATH        = os.path.join(OUTPUT_DIR,    "disaggregated_signals.pkl")
BASELINE_CSV        = os.path.join(BASELINES_DIR, "baseline_comparison.csv")
CROSSCLIMATE_CSV    = os.path.join(BASELINES_DIR, "cross_climate_results.csv")
SENSITIVITY_CSV     = os.path.join(SENSITIVITY_DIR, "sensitivity_results.csv")
SENSITIVITY_PLOT    = os.path.join(PLOTS_DIR,     "sensitivity_analysis.png")
FINAL_PLOT          = os.path.join(PLOTS_DIR,     "final_lighting_profiles.png")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# ---------------------------------------------------------------------------
# N-BEATS Model
# ---------------------------------------------------------------------------
class NBeatsBlock(nn.Module):
    def __init__(self, in_features, out_features, hidden_units=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_units), nn.ReLU(True),
            nn.Linear(hidden_units, hidden_units), nn.ReLU(True),
            nn.Linear(hidden_units, out_features)
        )

    def forward(self, x):
        return self.fc(x)


class NBeats(nn.Module):
    def __init__(self, window=24):
        super().__init__()
        self.trend_block       = NBeatsBlock(window, window)
        self.seasonality_block = NBeatsBlock(window, window)

    def forward(self, x):
        trend       = self.trend_block(x)
        residual    = x - trend
        seasonality = self.seasonality_block(residual)
        return trend, seasonality


# ---------------------------------------------------------------------------
# Helper: train N-BEATS with live tqdm MSE postfix
# ---------------------------------------------------------------------------
def train_nbeats(tensor_data, n_epochs=50, lr=1e-4, desc="Training N-BEATS"):
    model     = NBeats().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loss_val  = float("inf")
    pbar = tqdm(
        range(n_epochs), desc=desc, unit="epoch",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} epochs "
                   "[{elapsed}<{remaining}, MSE={postfix[0]:.4f}]",
        postfix=[loss_val]
    )
    for epoch in pbar:
        optimizer.zero_grad()
        trend, seas = model(tensor_data)
        loss = criterion(trend + seas, tensor_data)
        loss.backward()
        optimizer.step()
        loss_val        = loss.item()
        pbar.postfix[0] = loss_val
        pbar.update(0)
    pbar.close()
    return model, loss_val


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def run_stage2():
    for d in [OUTPUT_DIR, PLOTS_DIR, BASELINES_DIR, SENSITIVITY_DIR, CASE_STUDY_DIR]:
        os.makedirs(d, exist_ok=True)

    CONSOLE.print(
        f"\n[bold green]Stage 2: Disaggregation & Analysis[/] "
        f"— device: [cyan]{DEVICE}[/cyan]"
    )

    # ------------------------------------------------------------------
    # 1. Load clustered data
    # ------------------------------------------------------------------
    try:
        profiles_df = pd.read_pickle(CLUSTERED_DATA_PATH)
    except FileNotFoundError:
        CONSOLE.print(
            "[bold red]Error:[/] Run stage1_clustering.py first."
        )
        return

    hour_cols   = list(range(24))
    data_values = profiles_df[hour_cols].values.astype(np.float32)
    tensor_data = torch.FloatTensor(data_values).to(DEVICE)
    CONSOLE.log(f"Loaded {len(profiles_df):,} clustered daily profiles.")

    # ------------------------------------------------------------------
    # 2. N-BEATS — train or load checkpoint
    # ------------------------------------------------------------------
    if os.path.exists(MODEL_PATH):
        CONSOLE.print(
            f"[bold yellow]SKIP[/] N-BEATS training — "
            f"loading [cyan]{MODEL_PATH}[/cyan]"
        )
        model = NBeats().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        with torch.no_grad():
            t, s = model(tensor_data)
            final_mse = nn.MSELoss()(t + s, tensor_data).item()
        CONSOLE.log(f"N-BEATS loaded. MSE: [magenta]{final_mse:.4f}[/]")
    else:
        model, final_mse = train_nbeats(
            tensor_data, n_epochs=50, desc="Training N-BEATS (main model)"
        )
        torch.save(model.state_dict(), MODEL_PATH)
        CONSOLE.log(f"Saved N-BEATS checkpoint → [cyan]{MODEL_PATH}[/cyan]")

    # ------------------------------------------------------------------
    # 3. Disaggregate and save results — or load checkpoint
    # ------------------------------------------------------------------
    if os.path.exists(RESULTS_PATH):
        CONSOLE.print(
            f"[bold yellow]SKIP[/] Disaggregation — "
            f"loading [cyan]{RESULTS_PATH}[/cyan]"
        )
        results_df = pd.read_pickle(RESULTS_PATH)
    else:
        model.eval()
        with torch.no_grad():
            trend_final, light_final = model(tensor_data)
        results_df = profiles_df.copy()
        results_df[[f"trend_{h}" for h in range(24)]] = trend_final.cpu().numpy()
        results_df[[f"light_{h}" for h in range(24)]] = light_final.cpu().numpy()
        results_df.to_pickle(RESULTS_PATH)
        CONSOLE.log(f"Saved disaggregated signals → [cyan]{RESULTS_PATH}[/cyan]")

    # ------------------------------------------------------------------
    # 4. Baseline comparison — compute or skip
    # ------------------------------------------------------------------
    if os.path.exists(BASELINE_CSV):
        CONSOLE.print(
            f"[bold yellow]SKIP[/] Baselines — "
            f"loading [cyan]{BASELINE_CSV}[/cyan]"
        )
        tbl = Table(title="Baseline Comparison [cached]")
        tbl.add_column("Method", style="cyan")
        tbl.add_column("MSE",    style="magenta")
        for _, row in pd.read_csv(BASELINE_CSV).iterrows():
            tbl.add_row(row["Method"], f"{row['MSE']:.4f}")
        CONSOLE.print(tbl)
    else:
        run_baselines(data_values, final_mse)

    # ------------------------------------------------------------------
    # 5. Cross-climate generalisation — compute or skip
    # ------------------------------------------------------------------
    if os.path.exists(CROSSCLIMATE_CSV):
        CONSOLE.print(
            f"[bold yellow]SKIP[/] Cross-climate test — "
            f"loading [cyan]{CROSSCLIMATE_CSV}[/cyan]"
        )
        CONSOLE.log(pd.read_csv(CROSSCLIMATE_CSV).to_string(index=False))
    else:
        run_cross_climate_test(profiles_df)

    # ------------------------------------------------------------------
    # 6. Sensitivity analysis — compute or skip
    # ------------------------------------------------------------------
    if os.path.exists(SENSITIVITY_CSV):
        CONSOLE.print(
            f"[bold yellow]SKIP[/] Sensitivity analysis — "
            f"loading [cyan]{SENSITIVITY_CSV}[/cyan]"
        )
    else:
        run_sensitivity_analysis(tensor_data, model)

    # ------------------------------------------------------------------
    # 7. Case study — compute or skip
    # ------------------------------------------------------------------
    if any(f.endswith("_case_study.png") for f in os.listdir(CASE_STUDY_DIR)):
        CONSOLE.print("[bold yellow]SKIP[/] Case study — already exists.")
    else:
        run_case_study(results_df)

    # ------------------------------------------------------------------
    # 8. Final plot — generate or skip
    # ------------------------------------------------------------------
    if os.path.exists(FINAL_PLOT):
        CONSOLE.print(f"[bold yellow]SKIP[/] Final plot — already exists.")
    else:
        plot_final_results(results_df)

    CONSOLE.print("\n[bold green]Stage 2 Complete![/]")


# ---------------------------------------------------------------------------
# 4. Baseline Comparison
# ---------------------------------------------------------------------------
def run_baselines(data, nbeats_mse):
    CONSOLE.print("\n[bold underline]Baseline Comparison[/]")
    n_samples = min(2000, len(data))
    subset    = data[:n_samples]

    # STL — tqdm per series
    stl_recon = []
    for i in tqdm(range(n_samples), desc="STL Baseline", unit="series"):
        res = STL(subset[i], period=24, robust=True).fit()
        stl_recon.append(res.trend + res.seasonal)
    mse_stl = mean_squared_error(subset, np.array(stl_recon))

    # Gradient Boosting
    hours    = np.tile(np.arange(24), n_samples).reshape(-1, 1)
    readings = subset.flatten()
    gbr      = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
    )
    with tqdm(total=1, desc="Gradient Boosting fit",
              bar_format="{l_bar}{bar}| {elapsed}") as pb:
        gbr.fit(hours, readings)
        pb.update(1)
    mse_gbr = mean_squared_error(subset, gbr.predict(hours).reshape(n_samples, 24))

    tbl = Table(title="Disaggregation Baselines (MSE — lower is better)")
    tbl.add_column("Method",  style="cyan")
    tbl.add_column("MSE",     style="magenta")
    tbl.add_column("Notes")
    tbl.add_row("N-BEATS (Ours)",    f"{nbeats_mse:.4f}", "Two-stack neural basis decomposition")
    tbl.add_row("STL Decomposition", f"{mse_stl:.4f}",   "Classical seasonal-trend decomposition")
    tbl.add_row("Gradient Boosting", f"{mse_gbr:.4f}",   "Hour-of-day regression baseline")
    CONSOLE.print(tbl)

    pd.DataFrame({
        "Method": ["N-BEATS (Ours)", "STL Decomposition", "Gradient Boosting"],
        "MSE":    [nbeats_mse, mse_stl, mse_gbr]
    }).to_csv(BASELINE_CSV, index=False)
    CONSOLE.log(f"Saved baseline comparison → [cyan]{BASELINE_CSV}[/cyan]")


# ---------------------------------------------------------------------------
# 5. Cross-Climate Generalisation
# ---------------------------------------------------------------------------
def run_cross_climate_test(df):
    CONSOLE.print("\n[bold underline]Cross-Climate Generalisation Test[/]")
    site_counts = df.index.get_level_values("site_id").value_counts()
    if len(site_counts) < 2:
        CONSOLE.print("[bold yellow]Warning:[/] Fewer than 2 sites. Skipping.")
        return

    site_train_id = site_counts.index[0]
    site_test_id  = site_counts.index[1]
    hour_cols     = list(range(24))

    train_t = torch.FloatTensor(
        df[df.index.get_level_values("site_id") == site_train_id][hour_cols]
        .values.astype(np.float32)
    ).to(DEVICE)
    test_t  = torch.FloatTensor(
        df[df.index.get_level_values("site_id") == site_test_id][hour_cols]
        .values.astype(np.float32)
    ).to(DEVICE)

    model_cc, train_mse = train_nbeats(
        train_t, n_epochs=30,
        desc=f"Cross-climate: site {site_train_id} → {site_test_id}"
    )
    model_cc.eval()
    with torch.no_grad():
        t, s     = model_cc(test_t)
        test_mse = nn.MSELoss()(t + s, test_t).item()

    CONSOLE.log(
        f"Train site {site_train_id} MSE={train_mse:.4f} "
        f"→ Test site {site_test_id} MSE={test_mse:.4f}"
    )
    pd.DataFrame({
        "train_site": [site_train_id], "test_site": [site_test_id],
        "train_mse":  [train_mse],     "test_mse":  [test_mse]
    }).to_csv(CROSSCLIMATE_CSV, index=False)
    CONSOLE.log(f"Saved cross-climate results → [cyan]{CROSSCLIMATE_CSV}[/cyan]")


# ---------------------------------------------------------------------------
# 6. Sensitivity Analysis
# ---------------------------------------------------------------------------
def run_sensitivity_analysis(tensor_data, model):
    CONSOLE.print("\n[bold underline]Sensitivity Analysis[/]")
    noise_levels = [0.0, 0.01, 0.05, 0.10, 0.15, 0.20]
    mses         = []
    model.eval()
    for noise_std in tqdm(noise_levels, desc="Sensitivity sweep", unit="level"):
        noisy = tensor_data + torch.randn_like(tensor_data) * noise_std
        with torch.no_grad():
            t, s = model(noisy)
            mses.append(nn.MSELoss()(t + s, tensor_data).item())

    pd.DataFrame({"noise_std": noise_levels, "mse": mses}).to_csv(
        SENSITIVITY_CSV, index=False
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(noise_levels, mses, marker="o", linewidth=2, color="#1f77b4")
    ax.fill_between(noise_levels, mses, alpha=0.15, color="#1f77b4")
    ax.set_title("Model Robustness: MSE vs. Input Noise", fontsize=13)
    ax.set_xlabel("Std of Added Gaussian Noise", fontsize=11)
    ax.set_ylabel("Reconstruction MSE", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(SENSITIVITY_PLOT, dpi=300)
    plt.close(fig)
    CONSOLE.log(f"Saved sensitivity results → [cyan]{SENSITIVITY_CSV}[/cyan]")
    CONSOLE.log(f"Saved sensitivity plot    → [cyan]{SENSITIVITY_PLOT}[/cyan]")


# ---------------------------------------------------------------------------
# 7. Practical Case Study
# ---------------------------------------------------------------------------
def run_case_study(results_df):
    CONSOLE.print("\n[bold underline]Practical Case Study[/]")
    edu = results_df[results_df.index.get_level_values("primary_use") == "Education"]
    if edu.empty:
        edu = results_df[results_df.index.get_level_values("primary_use") == "Office"]
    if edu.empty:
        CONSOLE.print("[bold yellow]Warning:[/] No suitable building found. Skipping.")
        return

    bld_id      = edu.index.get_level_values("building_id").value_counts().index[0]
    building_df = results_df[
        results_df.index.get_level_values("building_id") == bld_id
    ].head(7)
    sector = building_df.index.get_level_values("primary_use")[0]
    CONSOLE.log(f"Case study: building {bld_id} ({sector}), {len(building_df)} days.")

    hour_cols = list(range(24))
    original  = building_df[hour_cols].values.flatten()
    trend     = building_df[[f"trend_{h}" for h in range(24)]].values.flatten()
    light     = building_df[[f"light_{h}" for h in range(24)]].values.flatten()
    hours     = np.arange(len(original))

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    axes[0].plot(hours, original, color="black",  alpha=0.85, linewidth=1.5, label="Aggregate Signal")
    axes[0].plot(hours, trend,    color="#1f77b4", linestyle="--", linewidth=1.5, label="Trend (HVAC/Base Load)")
    axes[0].plot(hours, light,    color="#ff7f0e", linewidth=2,   label="Isolated Lighting Component")
    axes[0].set_title(f"Case Study: Building {bld_id} ({sector}) — 7-Day Disaggregation", fontsize=14)
    axes[0].set_ylabel("Normalised Meter Reading", fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].fill_between(hours, light, alpha=0.4, color="#ff7f0e")
    axes[1].plot(hours, light, color="#ff7f0e", linewidth=1.5)
    for day in range(1, 7):
        axes[1].axvline(day * 24, color="grey", linestyle=":", alpha=0.7)
    axes[1].set_title("Isolated Lighting Load Only", fontsize=13)
    axes[1].set_xlabel("Hour of Week", fontsize=11)
    axes[1].set_ylabel("Lighting Load (Normalised)", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    plot_path = os.path.join(CASE_STUDY_DIR, f"building_{bld_id}_case_study.png")
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    building_df.to_csv(os.path.join(CASE_STUDY_DIR, f"building_{bld_id}_data.csv"))
    CONSOLE.log(f"Saved case study → [cyan]{plot_path}[/cyan]")


# ---------------------------------------------------------------------------
# 8. Final Visualisation
# ---------------------------------------------------------------------------
def plot_final_results(results_df):
    light_cols  = [f"light_{h}" for h in range(24)]
    palette     = sns.color_palette("tab10", n_colors=3)
    cluster_ids = sorted(results_df["cluster"].unique())

    fig, ax = plt.subplots(figsize=(12, 7))
    for cid in tqdm(cluster_ids, desc="Plotting cluster profiles", unit="cluster"):
        data  = results_df[results_df["cluster"] == cid][light_cols].values
        mean  = data.mean(axis=0)
        std   = data.std(axis=0)
        color = palette[int(cid)]
        ax.plot(range(24), mean, label=f"Cluster {cid} Mean", color=color, linewidth=2)
        ax.fill_between(range(24), mean - std, mean + std, alpha=0.15, color=color)

    ax.set_title("Isolated Lighting Pulses by Control Regime (N-BEATS Output)", fontsize=15)
    ax.set_xlabel("Hour of Day", fontsize=12)
    ax.set_ylabel("Disaggregated Lighting Load (Normalised)", fontsize=12)
    ax.set_xticks(range(24))
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(FINAL_PLOT, dpi=300)
    plt.close(fig)
    CONSOLE.log(f"Saved final results plot → [cyan]{FINAL_PLOT}[/cyan]")


if __name__ == "__main__":
    run_stage2()
