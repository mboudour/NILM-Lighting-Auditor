# stage1_clustering.py
#
# CHECKPOINTING: Every expensive step saves its result to disk on first run.
#                Re-running skips any step whose output already exists.
#
# IMPORTANT FIX: silhouette_score on ~500k samples is O(n^2) and would take
#                hours. We use a stratified random subsample of SILHOUETTE_SAMPLE
#                points, which is statistically robust and completes in seconds.
#
# Checkpoints (outputs/stage1/):
#   cae_weights.pt               trained 1D-CAE weights
#   latent_representations.npy   CAE latent manifold
#   kmeans_labels.npy            KMeans cluster labels (full dataset)
#   silhouette_scores.json       Silhouette scores (computed on subsample)
#   tsne_results.npy             2-D t-SNE embedding (computed on subsample)
#   clustered_daily_profiles.pkl profiles + cluster column (final output)
#
# Run after: stage0_preprocessing.py

import json
import os

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
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from tqdm import tqdm

CONSOLE = Console()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PREPROCESSED_DATA_PATH = "outputs/preprocessed_daily_profiles.pkl"
OUTPUT_DIR             = "outputs/stage1"
PLOTS_DIR              = "outputs/plots"

CAE_CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "cae_weights.pt")
LATENT_REPS_PATH    = os.path.join(OUTPUT_DIR, "latent_representations.npy")
KMEANS_LABELS_PATH  = os.path.join(OUTPUT_DIR, "kmeans_labels.npy")
SILHOUETTE_PATH     = os.path.join(OUTPUT_DIR, "silhouette_scores.json")
TSNE_RESULTS_PATH   = os.path.join(OUTPUT_DIR, "tsne_results.npy")
TSNE_PLOT_PATH      = os.path.join(PLOTS_DIR,  "stage1_tsne_latent_space.png")
CLUSTERED_DATA_PATH = os.path.join(OUTPUT_DIR, "clustered_daily_profiles.pkl")

N_CLUSTERS         = 3
CAE_EPOCHS         = 20
SILHOUETTE_SAMPLE  = 5000   # subsample size for silhouette — robust, completes in seconds
TSNE_SAMPLE        = 5000   # subsample size for t-SNE — avoids hour-long computation

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1), nn.ReLU(True),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(16, 8, kernel_size=3, padding=1), nn.ReLU(True),
            nn.MaxPool1d(2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(8, 16, kernel_size=2, stride=2), nn.ReLU(True),
            nn.ConvTranspose1d(16, 1,  kernel_size=2, stride=2), nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent), latent


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_stage1():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,  exist_ok=True)

    CONSOLE.print(
        f"\n[bold green]Stage 1: Unsupervised Clustering[/] "
        f"— device: [cyan]{DEVICE}[/cyan]"
    )

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    try:
        profiles    = pd.read_pickle(PREPROCESSED_DATA_PATH)
        tensor_data = torch.FloatTensor(profiles.values).unsqueeze(1).to(DEVICE)
    except FileNotFoundError:
        CONSOLE.print("[bold red]Error:[/] Run stage0_preprocessing.py first.")
        return
    CONSOLE.log(f"Loaded {len(profiles):,} daily profiles.")

    # ------------------------------------------------------------------
    # 2. CAE — train or load checkpoint
    # ------------------------------------------------------------------
    model = ConvAutoencoder().to(DEVICE)
    if os.path.exists(CAE_CHECKPOINT_PATH):
        CONSOLE.print(
            f"[bold yellow]SKIP[/] CAE training — "
            f"loading [cyan]{CAE_CHECKPOINT_PATH}[/cyan]"
        )
        model.load_state_dict(torch.load(CAE_CHECKPOINT_PATH, map_location=DEVICE))
        model.eval()
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        pbar = tqdm(
            range(CAE_EPOCHS), desc="Training 1D-CAE", unit="epoch",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} epochs "
                       "[{elapsed}<{remaining}, loss={postfix[0]:.6f}]",
            postfix=[float("inf")]
        )
        for epoch in pbar:
            optimizer.zero_grad()
            recon, _ = model(tensor_data)
            loss = criterion(recon, tensor_data)
            loss.backward()
            optimizer.step()
            pbar.postfix[0] = loss.item()
            pbar.update(0)
        pbar.close()
        CONSOLE.log(f"CAE done. Loss: [magenta]{loss.item():.6f}[/]")
        torch.save(model.state_dict(), CAE_CHECKPOINT_PATH)
        CONSOLE.log(f"Saved CAE checkpoint → [cyan]{CAE_CHECKPOINT_PATH}[/cyan]")

    # ------------------------------------------------------------------
    # 3. Latent manifold — extract or load checkpoint
    # ------------------------------------------------------------------
    if os.path.exists(LATENT_REPS_PATH):
        CONSOLE.print(
            f"[bold yellow]SKIP[/] Latent extraction — "
            f"loading [cyan]{LATENT_REPS_PATH}[/cyan]"
        )
        latent_manifold = np.load(LATENT_REPS_PATH)
    else:
        model.eval()
        with torch.no_grad():
            _, lt = model(tensor_data)
        latent_manifold = lt.view(lt.size(0), -1).cpu().numpy()
        np.save(LATENT_REPS_PATH, latent_manifold)
        CONSOLE.log(
            f"Saved latent representations → [cyan]{LATENT_REPS_PATH}[/cyan]"
        )

    # ------------------------------------------------------------------
    # 4. KMeans + Silhouette — compute or load checkpoint
    #
    #    KMeans uses MiniBatchKMeans for speed on large data.
    #    Silhouette is computed on a random subsample of SILHOUETTE_SAMPLE
    #    points — statistically valid and completes in seconds.
    # ------------------------------------------------------------------
    if os.path.exists(KMEANS_LABELS_PATH) and os.path.exists(SILHOUETTE_PATH):
        CONSOLE.print(
            "[bold yellow]SKIP[/] KMeans + Silhouette — loading from cache."
        )
        labels_cae = np.load(KMEANS_LABELS_PATH)
        with open(SILHOUETTE_PATH) as f:
            sil_scores = json.load(f)
        tbl = Table(title="Clustering Stability [cached]")
        tbl.add_column("Method",    style="cyan")
        tbl.add_column("Silhouette (subsample)", style="magenta")
        tbl.add_column("Notes")
        for e in sil_scores:
            tbl.add_row(e["method"], f"{e['silhouette']:.4f}", e["description"])
        CONSOLE.print(tbl)
    else:
        rng = np.random.default_rng(42)

        methods = [
            ("CAE + KMeans",       latent_manifold, "Deep learned features"),
            ("KMeans on Raw Data", profiles.values, "Baseline: raw normalised profiles"),
        ]
        tbl = Table(title="Clustering Stability (Silhouette on 5k subsample)")
        tbl.add_column("Method",    style="cyan")
        tbl.add_column("Silhouette (subsample)", style="magenta")
        tbl.add_column("Notes")

        labels_cae, sil_scores = None, []

        for name, data, desc in tqdm(
            methods, desc="KMeans + Silhouette", unit="method"
        ):
            # MiniBatchKMeans is much faster than KMeans on large data
            km = MiniBatchKMeans(
                n_clusters=N_CLUSTERS, random_state=42, n_init=10, batch_size=4096
            )
            with tqdm(
                total=1, desc=f"  Fitting {name}", leave=False,
                bar_format="{l_bar}{bar}| {elapsed}"
            ) as pb:
                km.fit(data)
                pb.update(1)
            labels = km.labels_

            # Subsample for silhouette — avoids O(n^2) computation
            n = len(data)
            idx = rng.choice(n, size=min(SILHOUETTE_SAMPLE, n), replace=False)
            with tqdm(
                total=1, desc=f"  Silhouette {name}", leave=False,
                bar_format="{l_bar}{bar}| {elapsed}"
            ) as pb:
                sil = silhouette_score(data[idx], labels[idx])
                pb.update(1)

            tbl.add_row(name, f"{sil:.4f}", desc)
            sil_scores.append(
                {"method": name, "silhouette": float(sil),
                 "description": desc,
                 "subsample_size": int(min(SILHOUETTE_SAMPLE, n))}
            )
            if name == "CAE + KMeans":
                labels_cae = labels

        CONSOLE.print(tbl)
        np.save(KMEANS_LABELS_PATH, labels_cae)
        with open(SILHOUETTE_PATH, "w") as f:
            json.dump(sil_scores, f, indent=2)
        CONSOLE.log(f"Saved KMeans labels    → [cyan]{KMEANS_LABELS_PATH}[/cyan]")
        CONSOLE.log(f"Saved Silhouette scores → [cyan]{SILHOUETTE_PATH}[/cyan]")

    # ------------------------------------------------------------------
    # 5. t-SNE — compute on subsample or load checkpoint
    # ------------------------------------------------------------------
    if os.path.exists(TSNE_RESULTS_PATH):
        CONSOLE.print(
            f"[bold yellow]SKIP[/] t-SNE — "
            f"loading [cyan]{TSNE_RESULTS_PATH}[/cyan]"
        )
        tsne_results = np.load(TSNE_RESULTS_PATH)
        tsne_labels  = np.load(TSNE_RESULTS_PATH.replace("tsne_results", "tsne_labels"))
    else:
        rng = np.random.default_rng(42)
        n   = len(latent_manifold)
        idx = rng.choice(n, size=min(TSNE_SAMPLE, n), replace=False)
        tsne_subset  = latent_manifold[idx]
        tsne_labels  = labels_cae[idx]

        CONSOLE.print(
            f"[bold cyan]Computing t-SNE on {len(idx):,} subsample points...[/]"
        )
        with tqdm(
            total=1, desc="t-SNE fit",
            bar_format="{l_bar}{bar}| {elapsed}"
        ) as pb:
            tsne_results = TSNE(
                n_components=2, perplexity=50, random_state=42, max_iter=300
            ).fit_transform(tsne_subset)
            pb.update(1)

        np.save(TSNE_RESULTS_PATH, tsne_results)
        np.save(TSNE_RESULTS_PATH.replace("tsne_results", "tsne_labels"), tsne_labels)
        CONSOLE.log(f"Saved t-SNE results → [cyan]{TSNE_RESULTS_PATH}[/cyan]")

    # ------------------------------------------------------------------
    # 6. t-SNE plot — generate or skip
    # ------------------------------------------------------------------
    if os.path.exists(TSNE_PLOT_PATH):
        CONSOLE.print("[bold yellow]SKIP[/] t-SNE plot — already exists.")
    else:
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.scatterplot(
            x=tsne_results[:, 0], y=tsne_results[:, 1],
            hue=tsne_labels,
            palette=sns.color_palette("hsv", N_CLUSTERS),
            legend="full", alpha=0.6, ax=ax
        )
        ax.set_title("t-SNE Projection of CAE Latent Space by Cluster", fontsize=14)
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        fig.tight_layout()
        fig.savefig(TSNE_PLOT_PATH, dpi=300, bbox_inches="tight")
        plt.close(fig)
        CONSOLE.log(f"Saved t-SNE plot → [cyan]{TSNE_PLOT_PATH}[/cyan]")

    # ------------------------------------------------------------------
    # 7. Clustered profiles — save or skip
    # ------------------------------------------------------------------
    if os.path.exists(CLUSTERED_DATA_PATH):
        CONSOLE.print("[bold yellow]SKIP[/] Clustered profiles — already exists.")
    else:
        df = profiles.copy()
        df["cluster"] = labels_cae
        df.to_pickle(CLUSTERED_DATA_PATH)
        CONSOLE.log(f"Saved clustered profiles → [cyan]{CLUSTERED_DATA_PATH}[/cyan]")

    CONSOLE.print("[bold green]Stage 1 Complete![/]")


if __name__ == "__main__":
    run_stage1()
