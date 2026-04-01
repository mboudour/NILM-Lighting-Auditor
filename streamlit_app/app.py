"""
Closing the Verification Gap: Non-Intrusive Auditing of Lighting Efficiency
in Commercial Buildings

Interactive Companion Demo
"""

import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lighting Efficiency Auditor",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded",
)

ASSETS = os.path.join(os.path.dirname(__file__), "assets")
DATA   = os.path.join(ASSETS, "data")
PLOTS  = os.path.join(ASSETS, "plots")

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        border-radius: 12px;
        padding: 18px 22px;
        color: white;
        text-align: center;
        margin: 4px;
    }
    .metric-card .value { font-size: 2.2rem; font-weight: 700; }
    .metric-card .label { font-size: 0.85rem; opacity: 0.85; margin-top: 4px; }
    .metric-card .delta { font-size: 0.8rem; color: #7ecf7e; margin-top: 2px; }
    .section-header {
        border-left: 4px solid #2d6a9f;
        padding-left: 12px;
        margin: 24px 0 12px 0;
    }
    .highlight-box {
        background: #f0f7ff;
        border: 1px solid #b3d4f5;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 10px 0;
    }
    .paper-ref {
        background: #fffbf0;
        border-left: 3px solid #f0a500;
        padding: 10px 16px;
        border-radius: 0 8px 8px 0;
        font-size: 0.88rem;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# N-BEATS Model (identical architecture to training)
# ─────────────────────────────────────────────────────────────────────────────
class NBeatsBlock(nn.Module):
    def __init__(self, in_features, out_features, hidden_units=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_units), nn.ReLU(True),
            nn.Linear(hidden_units, hidden_units), nn.ReLU(True),
            nn.Linear(hidden_units, out_features),
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

@st.cache_resource
def load_model():
    device    = torch.device("cpu")
    model     = NBeats(window=24).to(device)
    model_path = os.path.join(os.path.dirname(__file__), "nbeats_disaggregator.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

model, device = load_model()

# ─────────────────────────────────────────────────────────────────────────────
# Data loaders (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_baselines():
    return pd.read_csv(os.path.join(DATA, "baseline_comparison.csv"))

@st.cache_data
def load_cross_climate():
    return pd.read_csv(os.path.join(DATA, "cross_climate_results.csv"))

@st.cache_data
def load_sensitivity():
    return pd.read_csv(os.path.join(DATA, "sensitivity_results.csv"))

@st.cache_data
def load_silhouette():
    with open(os.path.join(DATA, "silhouette_scores.json")) as f:
        return json.load(f)

@st.cache_data
def load_case_study():
    return pd.read_csv(os.path.join(DATA, "building_0_data.csv"))

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────
def normalize(arr):
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)

def disaggregate(profile_24h):
    x = torch.FloatTensor(profile_24h).unsqueeze(0).to(device)
    with torch.no_grad():
        trend, lighting = model(x)
    return trend.squeeze().numpy(), lighting.squeeze().numpy()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💡 Lighting Efficiency Auditor")
    st.divider()
    st.markdown("""
    **Paper:** *Closing the Verification Gap: Non-Intrusive Auditing of Lighting
    Efficiency in Commercial Buildings*

    This app is the interactive companion to the research paper.
    """)
    st.divider()
    st.markdown("**Navigate to:**")
    page = st.radio(
        "Section",
        [
            "🏠 Overview & Key Results",
            "📊 Stage 1: Clustering Analysis",
            "⚡ Stage 2: Disaggregation & Baselines",
            "🌍 Cross-Climate Generalization",
            "🔬 Sensitivity & Robustness",
            "🏢 Case Study: Educational Building",
            "🔧 Live Demo: Disaggregate Your Data",
            "📖 Methodology & Dataset",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("""
    **Dataset:** [ASHRAE GEPIII](https://www.kaggle.com/competitions/ashrae-energy-prediction)
    (public, Kaggle)

    **Model:** Pre-trained N-BEATS weights included.
    No raw data is bundled in this app.
    """)

# ─────────────────────────────────────────────────────────────────────────────
# Load all data
# ─────────────────────────────────────────────────────────────────────────────
baselines_df    = load_baselines()
cross_df        = load_cross_climate()
sensitivity_df  = load_sensitivity()
silhouette_data = load_silhouette()
case_df         = load_case_study()

hours = list(range(24))

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Overview & Key Results
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview & Key Results":
    st.title("💡 Non-Intrusive Auditing of Lighting Efficiency in Commercial Buildings")
    st.markdown("""
    <div class="paper-ref">
    <strong>Paper:</strong> <em>Closing the Verification Gap: Non-Intrusive Auditing of Lighting Efficiency
    in Commercial Buildings</em><br>
    <strong>Dataset:</strong> ASHRAE Great Energy Predictor III (public) —
    <a href="https://www.kaggle.com/competitions/ashrae-energy-prediction" target="_blank">Kaggle</a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    A significant **verification gap** exists in sustainable architecture: buildings designed to be
    energy-efficient routinely consume far more energy in practice than predicted. This app demonstrates
    a fully **unsupervised, two-stage deep learning framework** that audits lighting efficiency using
    only standard hourly aggregate electricity meter data — **no hardware sub-meters required**.
    """)

    st.divider()
    st.markdown("### Key Results at a Glance")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown("""<div class="metric-card">
        <div class="value">1,448</div>
        <div class="label">Buildings Analysed</div>
        <div class="delta">16 climate zones</div>
    </div>""", unsafe_allow_html=True)
    c2.markdown("""<div class="metric-card">
        <div class="value">~500K</div>
        <div class="label">Daily Load Profiles</div>
        <div class="delta">ASHRAE GEPIII dataset</div>
    </div>""", unsafe_allow_html=True)
    c3.markdown("""<div class="metric-card">
        <div class="value">0.3728</div>
        <div class="label">Silhouette Score (CAE)</div>
        <div class="delta">vs 0.3480 baseline</div>
    </div>""", unsafe_allow_html=True)
    c4.markdown("""<div class="metric-card">
        <div class="value">0.0628</div>
        <div class="label">N-BEATS MSE</div>
        <div class="delta">vs 0.1112 Grad. Boosting</div>
    </div>""", unsafe_allow_html=True)
    c5.markdown("""<div class="metric-card">
        <div class="value">3</div>
        <div class="label">Operational Regimes</div>
        <div class="delta">Discovered unsupervised</div>
    </div>""", unsafe_allow_html=True)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Pipeline Architecture")
        st.markdown("""
        The framework operates in two sequential stages:

        **Stage 1 — Unsupervised Clustering**
        - A **1D Convolutional Autoencoder (1D-CAE)** learns a compact latent
          representation of daily 24-hour load profiles
        - **MiniBatch KMeans** (k=3) clusters buildings into distinct operational
          regimes without any labels
        - Validated by Silhouette Score comparison against raw-data baseline

        **Stage 2 — N-BEATS Disaggregation**
        - An **N-BEATS** model with specialized Trend and Seasonality blocks
          decomposes the aggregate signal
        - The **Trend Block** captures the slow-varying base load (HVAC, servers)
        - The **Seasonality Block** isolates the high-frequency, occupancy-driven
          lighting pulses
        - Evaluated against STL decomposition and Gradient Boosting baselines
        """)

    with col2:
        st.markdown("### The Verification Gap Problem")
        st.markdown("""
        Commercial buildings account for roughly **40% of global energy consumption**,
        with lighting representing **15–20%** of a facility's total electricity use.
        Despite stringent design-phase efficiency mandates (LEED, BREEAM), actual
        operational consumption routinely exceeds predictions by **20–60%**.

        The root cause is suboptimal operational control — static lighting schedules
        that fail to adapt to dynamic occupancy, illuminating empty floors during
        evenings and weekends.

        **The barrier to fixing this:** hardware sub-metering costs £500–£5,000 per
        circuit, making comprehensive auditing prohibitive for most building stock.

        **Our solution:** extract the same information from the smart meter data
        already being collected, using only deep learning.
        """)

    st.divider()
    st.markdown("### Paper Figures")
    col1, col2 = st.columns(2)
    with col1:
        st.image(os.path.join(PLOTS, "stage1_tsne_latent_space.png"),
                 caption="Fig. 1: t-SNE projection of the CAE latent space — three distinct operational regimes emerge unsupervised",
                 use_container_width=True)
    with col2:
        st.image(os.path.join(PLOTS, "final_lighting_profiles.png"),
                 caption="Fig. 2: Mean isolated lighting profiles per cluster — Scheduled Weekday, Reactive/Weekend, Continuous 24/7",
                 use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Stage 1 — Clustering Analysis
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Stage 1: Clustering Analysis":
    st.title("📊 Stage 1: Unsupervised Clustering via 1D-CAE")

    st.markdown("""
    The first stage of the pipeline learns a compact, noise-robust representation of
    daily load profiles using a **1D Convolutional Autoencoder (1D-CAE)**. The encoder
    compresses each 24-hour profile into a 16-dimensional latent vector, filtering out
    high-frequency noise and emphasizing the structural differences between building
    operational regimes. MiniBatch KMeans is then applied to this latent space.
    """)

    st.divider()

    # Architecture description
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### 1D-CAE Architecture")
        arch_data = {
            "Layer": ["Input", "Conv1D (32 filters, k=3)", "MaxPool1D (2)", "Conv1D (64 filters, k=3)",
                      "MaxPool1D (2)", "Flatten", "Dense (16) — Latent", "Dense (64)",
                      "Upsample (2)", "ConvTranspose1D (32, k=3)", "Upsample (2)",
                      "ConvTranspose1D (1, k=3) — Output"],
            "Shape": ["(24,)", "(22, 32)", "(11, 32)", "(9, 64)", "(4, 64)", "(256,)",
                      "(16,)", "(64,)", "(8, 64)", "(10, 32)", "(20, 32)", "(24,)"],
            "Activation": ["—", "ReLU", "—", "ReLU", "—", "—", "Linear", "ReLU",
                           "—", "ReLU", "—", "Sigmoid"],
        }
        st.dataframe(pd.DataFrame(arch_data), use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### Clustering Stability Results")
        sil_df = pd.DataFrame(silhouette_data)
        sil_df.columns = ["Method", "Silhouette Score", "Description", "Subsample Size"]
        st.dataframe(sil_df[["Method", "Silhouette Score", "Description"]],
                     use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="highlight-box">
        The CAE + KMeans approach achieves a Silhouette Score of <strong>0.3728</strong>,
        compared to <strong>0.3480</strong> for KMeans applied directly to raw profiles —
        a <strong>7.1% relative improvement</strong>. Both scores are computed on a
        stratified random subsample of 5,000 profiles to manage the O(N²) complexity.
        </div>
        """, unsafe_allow_html=True)

        # Bar chart of silhouette scores
        fig, ax = plt.subplots(figsize=(6, 3))
        methods = [d["method"] for d in silhouette_data]
        scores  = [d["silhouette"] for d in silhouette_data]
        colors  = ["#2d6a9f", "#aac4e0"]
        bars    = ax.barh(methods, scores, color=colors, edgecolor="white", height=0.5)
        ax.set_xlim(0.30, 0.40)
        ax.set_xlabel("Silhouette Score")
        ax.set_title("Clustering Quality Comparison")
        for bar, score in zip(bars, scores):
            ax.text(score + 0.001, bar.get_y() + bar.get_height()/2,
                    f"{score:.4f}", va="center", fontsize=10, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    st.markdown("#### t-SNE Projection of the CAE Latent Space")
    st.image(os.path.join(PLOTS, "stage1_tsne_latent_space.png"),
             caption="t-SNE projection of the 16-dimensional CAE latent space (5,000-point subsample, coloured by KMeans cluster assignment). The clear separation between clusters validates the unsupervised feature learning.",
             use_container_width=True)

    st.markdown("""
    The t-SNE projection reveals three well-separated clusters corresponding to distinct
    building operational regimes:

    | Cluster | Regime | Characteristics |
    |---|---|---|
    | **0 (Blue)** | Scheduled Weekday | Sharp morning ramp-up at 08:00, peak around noon, decline after 18:00 |
    | **1 (Orange)** | Reactive / Weekend | Lower overall magnitude, higher variance, irregular occupancy patterns |
    | **2 (Green)** | Continuous 24/7 | Elevated baseline throughout the night, characteristic of data centres and hospitals |
    """)

    st.divider()
    st.markdown("#### Isolated Lighting Profiles by Cluster")
    st.image(os.path.join(PLOTS, "final_lighting_profiles.png"),
             caption="Mean isolated lighting load per operational cluster. Shaded areas represent the intra-cluster variance of individual daily profiles.",
             use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Stage 2 — Disaggregation & Baselines
# ═════════════════════════════════════════════════════════════════════════════
elif page == "⚡ Stage 2: Disaggregation & Baselines":
    st.title("⚡ Stage 2: N-BEATS Disaggregation & Baseline Comparison")

    st.markdown("""
    The second stage applies an **N-BEATS (Neural Basis Expansion Analysis for
    Interpretable Time Series)** model to decompose the aggregate load signal into
    its constituent components. N-BEATS was originally designed for time series
    forecasting; here it is repurposed as a **blind source separator** by training
    it to reconstruct the input signal as the sum of two interpretable components.
    """)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### N-BEATS Architecture (Disaggregation Mode)")
        st.markdown("""
        The model consists of two specialized blocks operating in sequence:

        **Trend Block** — Three fully-connected layers (256 hidden units, ReLU)
        mapping the 24-hour input to a smooth, low-frequency trend component
        representing the base load (HVAC, servers, always-on equipment).

        **Seasonality Block** — Identical architecture applied to the residual
        (input minus trend), capturing the high-frequency, occupancy-driven
        lighting pulses.

        The model is trained end-to-end to minimize the reconstruction MSE:
        ```
        Loss = MSE(Trend + Seasonality, Input)
        ```
        Training: 50 epochs, Adam optimizer (lr=1e-4), batch = full dataset (~500K profiles).
        """)

    with col2:
        st.markdown("#### Baseline Comparison Results")

        # Format the baselines table nicely
        bl = baselines_df.copy()
        bl["MSE"] = bl["MSE"].apply(lambda x: f"{x:.4f}" if x > 1e-10 else "≈ 0 (trivial)")
        bl["Improvement over N-BEATS"] = ["—", "Trivial (mathematical artifact)", "+77.1% worse"]
        bl.columns = ["Method", "Reconstruction MSE", "Note vs. N-BEATS"]
        st.dataframe(bl, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="highlight-box">
        <strong>N-BEATS achieves MSE = 0.0628</strong>, a <strong>43.6% reduction</strong>
        over Gradient Boosting (0.1112). While STL Decomposition achieves near-zero
        reconstruction error, this is a mathematical artifact — STL perfectly reconstructs
        the input by definition but fails to physically separate the underlying loads,
        bleeding lighting pulses into the trend component.
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Bar chart of baselines
    st.markdown("#### Visual Comparison: Reconstruction MSE")
    fig, ax = plt.subplots(figsize=(10, 4))
    methods = ["N-BEATS\n(Ours)", "Gradient\nBoosting", "STL\nDecomposition"]
    mse_vals = [0.0628, 0.1112, 0.0]
    display_vals = [0.0628, 0.1112, 0.001]  # STL shown as small bar for visibility
    colors = ["#2d6a9f", "#e05c5c", "#aaaaaa"]
    bars = ax.bar(methods, display_vals, color=colors, edgecolor="white", width=0.5)
    ax.set_ylabel("Reconstruction MSE (lower is better)")
    ax.set_title("Disaggregation Baseline Comparison")
    for bar, val, display in zip(bars, mse_vals, display_vals):
        label = f"{val:.4f}" if val > 1e-10 else "≈ 0\n(trivial)"
        ax.text(bar.get_x() + bar.get_width()/2, display + 0.002,
                label, ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 0.14)
    ax.grid(True, axis="y", alpha=0.3)
    ax.annotate("43.6% improvement\nover Gradient Boosting",
                xy=(0, 0.0628), xytext=(0.5, 0.09),
                arrowprops=dict(arrowstyle="->", color="#2d6a9f"),
                color="#2d6a9f", fontsize=10, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()
    st.markdown("""
    #### Why STL is Not a Meaningful Baseline for Physical Disaggregation

    STL (Seasonal-Trend Decomposition using LOESS) achieves near-zero reconstruction
    error because it is designed to perfectly reconstruct the input signal by
    construction. However, in the context of commercial building load disaggregation,
    this mathematical property is a liability rather than an asset:

    - STL allocates high-frequency lighting pulses to the **seasonal component** based
      purely on their periodicity, not their physical origin
    - It cannot distinguish between a lighting pulse at 09:00 and a regular HVAC
      morning ramp-up occurring at the same time
    - The resulting "lighting" component from STL is contaminated by HVAC patterns,
      making it unsuitable for actionable efficiency auditing

    N-BEATS, by contrast, learns the structural distinction between slow-varying
    base loads and sharp occupancy-driven pulses through its specialized block
    architecture, producing physically interpretable decompositions.
    """)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Cross-Climate Generalization
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🌍 Cross-Climate Generalization":
    st.title("🌍 Cross-Climate Generalization")

    st.markdown("""
    A critical requirement for any scalable building energy auditing tool is the ability
    to **generalize across different geographies and climate zones** without requiring
    site-specific retraining. The ASHRAE GEPIII dataset spans 16 diverse climate zones,
    each with drastically different heating and cooling degree days, which fundamentally
    alters the shape of the continuous base load (HVAC).
    """)

    st.divider()

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### Generalization Experiment Results")
        cc = cross_df.copy()
        cc.columns = ["Train Site", "Test Site", "Train MSE", "Test MSE"]
        cc["Train MSE"] = cc["Train MSE"].apply(lambda x: f"{x:.4f}")
        cc["Test MSE"]  = cc["Test MSE"].apply(lambda x: f"{x:.4f}")
        cc["Δ MSE"] = ["−0.0053 (improvement)"]
        st.dataframe(cc, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="highlight-box">
        The model trained exclusively on <strong>Site 3</strong> achieves a
        <strong>Test MSE of 0.1253</strong> on the completely held-out
        <strong>Site 13</strong> — a <em>different climate zone</em>.
        The test MSE is actually slightly <em>lower</em> than the training MSE (0.1306),
        confirming that the N-BEATS architecture learns <strong>universal structural
        representations</strong> of commercial electrical loads rather than
        climate-specific HVAC patterns.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### Visual: Train vs. Test MSE")
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = ["Train MSE\n(Site 3)", "Test MSE\n(Site 13, unseen)"]
        vals   = [0.1306, 0.1253]
        colors = ["#2d6a9f", "#27ae60"]
        bars   = ax.bar(labels, vals, color=colors, edgecolor="white", width=0.4)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.002,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 0.17)
        ax.set_ylabel("Reconstruction MSE")
        ax.set_title("Cross-Climate Generalization\n(N-BEATS trained on Site 3, tested on Site 13)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.annotate("No degradation\nacross climate zones",
                    xy=(1, 0.1253), xytext=(0.5, 0.145),
                    arrowprops=dict(arrowstyle="->", color="#27ae60"),
                    color="#27ae60", fontsize=10, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    st.markdown("""
    #### Interpretation

    The cross-climate generalization result has important practical implications.
    It demonstrates that the N-BEATS model is learning the **fundamental temporal
    structure** of lighting loads — the sharp morning onset, the midday plateau,
    and the evening decline — rather than the specific magnitude or shape of the
    HVAC base load, which varies dramatically between a hot-humid climate (Site 3)
    and a cold-dry climate (Site 13).

    This means that a single pre-trained model can be deployed across an entire
    building portfolio spanning multiple geographies **without retraining**, dramatically
    reducing the barrier to large-scale adoption.

    | Climate Zone Property | Site 3 | Site 13 |
    |---|---|---|
    | Climate Type | Hot-Humid | Cold-Dry |
    | HVAC Dominance | Cooling-dominated | Heating-dominated |
    | Load Profile Shape | High summer peaks | High winter peaks |
    | N-BEATS MSE | 0.1306 (train) | 0.1253 (test) |
    """)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Sensitivity & Robustness
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Sensitivity & Robustness":
    st.title("🔬 Sensitivity Analysis & Robustness to Noise")

    st.markdown("""
    Real-world smart meter data is frequently corrupted by **sensor noise, transmission
    dropouts, and irregular usage spikes**. To evaluate the framework's resilience to
    such perturbations, we performed a comprehensive sensitivity analysis by injecting
    Gaussian noise of varying standard deviations (σ) into the normalized input profiles.
    """)

    st.divider()

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### Sensitivity Analysis Results")
        sens = sensitivity_df.copy()
        sens.columns = ["Noise Std (σ)", "Reconstruction MSE"]
        sens["Noise Std (σ)"] = sens["Noise Std (σ)"].apply(lambda x: f"{x:.2f}")
        sens["Reconstruction MSE"] = sens["Reconstruction MSE"].apply(lambda x: f"{x:.4f}")
        sens["Change from σ=0"] = ["—", "−0.0002", "−0.0007", "−0.0011", "−0.0014", "−0.0013"]
        st.dataframe(sens, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="highlight-box">
        The reconstruction MSE remains <strong>remarkably stable</strong> across all
        noise levels — from 0.0611 at zero noise to 0.0598 at σ=0.20 (a severe
        perturbation representing 20% of the normalized signal range). The slight
        <em>decrease</em> in MSE with increasing noise is a known <strong>regularization
        effect</strong> in deep neural networks, where moderate input noise prevents
        overfitting to minor fluctuations.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### Sensitivity Plot")
        st.image(os.path.join(PLOTS, "sensitivity_analysis.png"),
                 caption="Reconstruction MSE vs. injected Gaussian noise standard deviation. The near-flat curve confirms the model's robustness.",
                 use_container_width=True)

    st.divider()

    # Interactive noise simulation
    st.markdown("#### Interactive Noise Simulation")
    st.markdown("Adjust the noise level to see how the N-BEATS model responds to corrupted input:")

    noise_level = st.slider("Noise Standard Deviation (σ)", 0.0, 0.30, 0.05, 0.01)

    # Use a representative profile
    profile_clean = np.array([
        0.12, 0.10, 0.09, 0.09, 0.10, 0.13,
        0.28, 0.55, 0.78, 0.88, 0.92, 0.95,
        0.90, 0.93, 0.91, 0.89, 0.82, 0.65,
        0.42, 0.30, 0.22, 0.18, 0.15, 0.13,
    ], dtype=np.float32)

    np.random.seed(42)
    noisy_profile = np.clip(profile_clean + np.random.normal(0, noise_level, 24).astype(np.float32), 0, 1)
    trend_clean, light_clean = disaggregate(profile_clean)
    trend_noisy, light_noisy = disaggregate(noisy_profile)

    mse_clean = float(np.mean((trend_clean + light_clean - profile_clean)**2))
    mse_noisy = float(np.mean((trend_noisy + light_noisy - noisy_profile)**2))

    c1, c2, c3 = st.columns(3)
    c1.metric("Clean Profile MSE", f"{mse_clean:.4f}")
    c2.metric("Noisy Profile MSE", f"{mse_noisy:.4f}", delta=f"{mse_noisy - mse_clean:+.4f}")
    c3.metric("Noise Level σ", f"{noise_level:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    ax = axes[0]
    ax.plot(hours, profile_clean, "k--", lw=1.5, alpha=0.5, label="Clean input")
    ax.plot(hours, noisy_profile, "gray", lw=1, alpha=0.7, label=f"Noisy input (σ={noise_level:.2f})")
    ax.plot(hours, trend_noisy, "#1f77b4", lw=2, label="Base load (trend)")
    ax.plot(hours, light_noisy, "#ff7f0e", lw=2, label="Lighting (isolated)")
    ax.set_xlabel("Hour of Day"); ax.set_ylabel("Normalized Load")
    ax.set_title(f"Disaggregation with σ={noise_level:.2f} Noise")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_xticks(range(0, 24, 2))

    ax2 = axes[1]
    ax2.plot(hours, light_clean, "#ff7f0e", lw=2.5, label="Lighting (clean input)", zorder=3)
    ax2.plot(hours, light_noisy, "#ff7f0e", lw=1.5, ls="--", alpha=0.7, label=f"Lighting (σ={noise_level:.2f})")
    ax2.fill_between(hours, light_clean, light_noisy, alpha=0.2, color="red", label="Deviation")
    ax2.set_xlabel("Hour of Day"); ax2.set_ylabel("Isolated Lighting Load")
    ax2.set_title("Lighting Component: Clean vs. Noisy")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3); ax2.set_xticks(range(0, 24, 2))

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Case Study
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🏢 Case Study: Educational Building":
    st.title("🏢 Case Study: Educational Facility (Building 0, Site 0)")

    st.markdown("""
    To demonstrate the practical utility of the framework for facility managers,
    we analysed a continuous **7-day period** for Building 0 — an educational
    facility from Site 0 of the ASHRAE GEPIII dataset. This case study illustrates
    how the disaggregated lighting profile can reveal specific, actionable
    energy-saving opportunities.
    """)

    st.divider()
    st.markdown("#### 7-Day Disaggregation: Full Case Study Figure")
    st.image(os.path.join(PLOTS, "building_0_case_study.png"),
             caption="Case study of an educational facility over 7 days. Top: aggregate signal with decomposed trend and lighting components. Bottom: isolated lighting load revealing operational patterns.",
             use_container_width=True)

    st.divider()
    st.markdown("#### Interactive: Day-by-Day Analysis")

    # Load actual case study data
    hour_cols   = [str(h) for h in range(24)]
    trend_cols  = [f"trend_{h}" for h in range(24)]
    light_cols  = [f"light_{h}" for h in range(24)]

    available_dates = case_df["date"].tolist()
    selected_date   = st.selectbox("Select a day to analyse:", available_dates)

    row = case_df[case_df["date"] == selected_date].iloc[0]
    agg_profile   = row[hour_cols].values.astype(float)
    trend_profile = row[trend_cols].values.astype(float)
    light_profile = row[light_cols].values.astype(float)
    cluster       = int(row["cluster"])

    cluster_names = {0: "Scheduled Weekday", 1: "Reactive / Weekend", 2: "Continuous 24/7"}
    cluster_colors = {0: "#2d6a9f", 1: "#e07b39", 2: "#27ae60"}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Date", selected_date)
    c2.metric("Cluster", f"{cluster} — {cluster_names.get(cluster, 'Unknown')}")
    lighting_frac = float(np.mean(np.abs(light_profile)) / (np.mean(np.abs(agg_profile)) + 1e-9))
    c3.metric("Lighting Fraction", f"{lighting_frac*100:.1f}%")
    peak_h = int(np.argmax(light_profile))
    c4.metric("Peak Lighting Hour", f"{peak_h:02d}:00")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax = axes[0]
    ax.fill_between(hours, agg_profile, alpha=0.1, color="gray")
    ax.plot(hours, agg_profile,   "gray",   lw=1.5, ls="--", label="Aggregate (input)")
    ax.plot(hours, trend_profile, "#1f77b4", lw=2.5, label="Base load (trend)")
    ax.plot(hours, light_profile, "#ff7f0e", lw=2.5, label="Lighting (isolated)")
    ax.set_ylabel("Normalized Load")
    ax.set_title(f"Signal Decomposition — {selected_date} | Cluster {cluster}: {cluster_names.get(cluster, '')}")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    bar_colors = [cluster_colors.get(cluster, "#ff7f0e") if v > np.percentile(light_profile, 60)
                  else "#ffd1a3" for v in light_profile]
    ax2.bar(hours, light_profile, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax2.set_xlabel("Hour of Day"); ax2.set_ylabel("Isolated Lighting Load")
    ax2.set_title("Isolated Lighting Profile")
    ax2.set_xticks(range(0, 24, 2)); ax2.grid(True, axis="y", alpha=0.3)

    # Annotate off-hours
    off_hours_vals = [light_profile[h] for h in range(24) if h < 6 or h > 21]
    if np.mean(off_hours_vals) > 0.05:
        ax2.axvspan(0, 6, alpha=0.08, color="red", label="Off-hours (risk zone)")
        ax2.axvspan(21, 23, alpha=0.08, color="red")
        ax2.legend(fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Actionable insight
    off_mean = np.mean([light_profile[h] for h in range(24) if h < 6 or h > 21])
    if off_mean > 0.05:
        st.warning(
            f"⚠️ **Potential inefficiency detected on {selected_date}:** "
            f"The isolated lighting load averages **{off_mean:.3f}** (normalized) "
            f"during off-hours (21:00–06:00). For an educational facility, this "
            f"likely indicates lighting left on in unoccupied spaces or a "
            f"misconfigured automated control schedule."
        )
    else:
        st.success(
            f"✅ **Lighting schedule appears efficient on {selected_date}:** "
            f"Off-hour lighting load is within expected bounds for an educational facility."
        )

    st.divider()
    st.markdown("#### Raw Disaggregated Data for This Day")
    display_df = pd.DataFrame({
        "Hour": [f"{h:02d}:00" for h in hours],
        "Aggregate Load": agg_profile.round(4),
        "Base Load (Trend)": trend_profile.round(4),
        "Lighting (Isolated)": light_profile.round(4),
    })
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    csv_bytes = display_df.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download disaggregated data (CSV)",
        data=csv_bytes,
        file_name=f"building_0_{selected_date}_disaggregated.csv",
        mime="text/csv",
    )

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Live Demo
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔧 Live Demo: Disaggregate Your Data":
    st.title("🔧 Live Demo: Disaggregate Your Own Building Data")

    st.markdown("""
    Upload a CSV file containing hourly electricity readings for your building,
    or select one of the representative profiles below. The pre-trained N-BEATS
    model will decompose the aggregate signal into base load and lighting components
    in real time.

    > **Note:** No raw ASHRAE data is bundled in this app. The pre-trained model
    > weights are included. The original dataset is publicly available on
    > [Kaggle](https://www.kaggle.com/competitions/ashrae-energy-prediction).
    """)

    st.divider()

    input_mode = st.radio(
        "Input source:",
        ["Use a representative ASHRAE profile", "Upload my own CSV"],
        horizontal=True,
    )

    profiles_to_show = []

    if input_mode == "Use a representative ASHRAE profile":
        sample_profiles = {
            "Educational Building — Weekday (Cluster 0)": np.array([
                0.12, 0.10, 0.09, 0.09, 0.10, 0.13, 0.28, 0.55,
                0.78, 0.88, 0.92, 0.95, 0.90, 0.93, 0.91, 0.89,
                0.82, 0.65, 0.42, 0.30, 0.22, 0.18, 0.15, 0.13,
            ], dtype=np.float32),
            "Office Building — Weekday (Cluster 0)": np.array([
                0.08, 0.07, 0.07, 0.07, 0.08, 0.10, 0.22, 0.48,
                0.72, 0.85, 0.90, 0.88, 0.84, 0.87, 0.89, 0.88,
                0.80, 0.60, 0.38, 0.25, 0.18, 0.14, 0.11, 0.09,
            ], dtype=np.float32),
            "Retail Building — Weekend (Cluster 1)": np.array([
                0.15, 0.14, 0.13, 0.13, 0.14, 0.16, 0.20, 0.30,
                0.50, 0.68, 0.78, 0.82, 0.80, 0.81, 0.79, 0.75,
                0.70, 0.62, 0.50, 0.40, 0.30, 0.24, 0.20, 0.17,
            ], dtype=np.float32),
            "Data Centre / 24-7 Facility (Cluster 2)": np.array([
                0.72, 0.71, 0.71, 0.70, 0.71, 0.72, 0.75, 0.80,
                0.85, 0.88, 0.89, 0.90, 0.89, 0.90, 0.90, 0.89,
                0.87, 0.85, 0.82, 0.79, 0.77, 0.75, 0.74, 0.73,
            ], dtype=np.float32),
            "Hospital — Mixed Occupancy (Cluster 2)": np.array([
                0.55, 0.53, 0.52, 0.51, 0.52, 0.55, 0.62, 0.72,
                0.82, 0.88, 0.90, 0.91, 0.89, 0.90, 0.88, 0.86,
                0.83, 0.78, 0.72, 0.66, 0.62, 0.60, 0.58, 0.56,
            ], dtype=np.float32),
        }
        selected = st.multiselect(
            "Select profiles to analyse:",
            list(sample_profiles.keys()),
            default=["Educational Building — Weekday (Cluster 0)"],
        )
        for name in selected:
            profiles_to_show.append((name, sample_profiles[name]))

    else:
        st.markdown("""
        **Accepted CSV formats:**
        - **Single day:** One row with 24 comma-separated values (hours 0–23)
        - **Multi-day:** A `meter_reading` column with 24 rows per day, and optionally a `timestamp` column
        """)
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                if df.shape[1] >= 24 and df.shape[0] <= 10:
                    for idx, row in df.iterrows():
                        vals = row.values[:24].astype(float)
                        profiles_to_show.append((f"Row {idx+1}", normalize(vals).astype(np.float32)))
                elif "meter_reading" in df.columns:
                    readings = df["meter_reading"].values.astype(float)
                    n_days   = len(readings) // 24
                    if n_days == 0:
                        st.error("Need at least 24 hourly readings.")
                    else:
                        for i in range(min(n_days, 7)):
                            prof = normalize(readings[i*24:(i+1)*24]).astype(np.float32)
                            profiles_to_show.append((f"Day {i+1}", prof))
                else:
                    for idx, row in df.iterrows():
                        vals = row.values[:24].astype(float)
                        profiles_to_show.append((f"Row {idx+1}", normalize(vals).astype(np.float32)))
            except Exception as e:
                st.error(f"Could not parse CSV: {e}")

    if not profiles_to_show:
        st.info("Select or upload a building profile to begin.")
    else:
        for label, raw_profile in profiles_to_show:
            st.subheader(f"📊 {label}")
            trend, lighting = disaggregate(raw_profile)

            c1, c2, c3, c4 = st.columns(4)
            lighting_frac = float(np.mean(np.abs(lighting)) / (np.mean(np.abs(raw_profile)) + 1e-9))
            peak_h        = int(np.argmax(lighting))
            recon_mse     = float(np.mean((trend + lighting - raw_profile)**2))
            off_mean      = np.mean([lighting[h] for h in range(24) if h < 6 or h > 21])

            c1.metric("Lighting Fraction",  f"{lighting_frac*100:.1f}%")
            c2.metric("Peak Lighting Hour", f"{peak_h:02d}:00")
            c3.metric("Reconstruction MSE", f"{recon_mse:.4f}")
            c4.metric("Off-Hour Lighting",  f"{off_mean:.3f}", delta="⚠️ High" if off_mean > 0.1 else "✅ OK")

            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            ax = axes[0]
            ax.fill_between(hours, raw_profile, alpha=0.12, color="gray")
            ax.plot(hours, raw_profile, "gray",   lw=1.5, ls="--", label="Aggregate (input)")
            ax.plot(hours, trend,       "#1f77b4", lw=2.5, label="Base load (trend)")
            ax.plot(hours, lighting,    "#ff7f0e", lw=2.5, label="Lighting (isolated)")
            ax.set_xlabel("Hour of Day"); ax.set_ylabel("Normalized Load")
            ax.set_title("Signal Decomposition"); ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3); ax.set_xticks(range(0, 24, 2))

            ax2 = axes[1]
            bar_colors = ["#ff7f0e" if v > np.percentile(lighting, 70) else "#ffd1a3" for v in lighting]
            ax2.bar(hours, lighting, color=bar_colors, edgecolor="white", linewidth=0.5)
            ax2.set_xlabel("Hour of Day"); ax2.set_ylabel("Isolated Lighting Load")
            ax2.set_title("Isolated Lighting Profile")
            ax2.set_xticks(range(0, 24, 2)); ax2.grid(True, axis="y", alpha=0.3)
            ax2.annotate(f"Peak\n{peak_h:02d}:00",
                         xy=(peak_h, lighting[peak_h]),
                         xytext=(min(peak_h + 3, 21), lighting[peak_h] * 1.1),
                         arrowprops=dict(arrowstyle="->", color="black"), fontsize=9)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            if off_mean > 0.1:
                st.warning(f"⚠️ **Off-hour lighting detected:** Average isolated lighting load during "
                           f"22:00–06:00 is {off_mean:.3f}. This may indicate unoccupied spaces "
                           f"remaining lit or a misconfigured automated schedule.")
            else:
                st.success("✅ **Lighting schedule appears efficient:** Off-hour lighting is within expected bounds.")

            out_df = pd.DataFrame({
                "hour": [f"{h:02d}:00" for h in hours],
                "aggregate": raw_profile.round(4),
                "base_load_trend": trend.round(4),
                "lighting_isolated": lighting.round(4),
            })
            st.download_button(f"⬇️ Download disaggregated data — {label}",
                               data=out_df.to_csv(index=False).encode(),
                               file_name=f"{label.replace(' ', '_')}_disaggregated.csv",
                               mime="text/csv")
            st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Methodology & Dataset
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📖 Methodology & Dataset":
    st.title("📖 Methodology & Dataset")

    tab1, tab2, tab3 = st.tabs(["Dataset", "Stage 1: 1D-CAE Clustering", "Stage 2: N-BEATS Disaggregation"])

    with tab1:
        st.markdown("""
        ### ASHRAE Great Energy Predictor III Dataset

        The **ASHRAE Great Energy Predictor III (GEPIII)** competition dataset is one of
        the largest publicly available building energy datasets. It is hosted on
        [Kaggle](https://www.kaggle.com/competitions/ashrae-energy-prediction) and
        freely accessible under the competition terms.

        | Property | Value |
        |---|---|
        | **Buildings** | 1,448 non-residential buildings |
        | **Climate Zones (Sites)** | 16 |
        | **Meter Types** | Electricity, chilled water, steam, hot water |
        | **Temporal Resolution** | Hourly |
        | **Coverage Period** | 2016–2017 (training), 2017–2018 (test) |
        | **Total Rows (train.csv)** | ~20 million |
        | **Building Types** | Education, Office, Entertainment, Public Services, Lodging, etc. |

        #### Preprocessing Pipeline (Stage 0)

        1. **Meter filtering:** Retain only electricity meters (meter type 0)
        2. **Missing value handling:** Forward-fill gaps ≤ 3 hours; drop buildings with
           >10% missing data
        3. **Outlier removal:** Remove readings beyond 5× the building's interquartile range
        4. **Pivoting:** Reshape from long format to wide daily profiles (one row per
           building-day, 24 columns for hours 0–23)
        5. **Normalization:** Min-Max normalization per building to [0, 1], preserving
           the intra-day shape while removing inter-building scale differences
        6. **Output:** ~500,000 normalized daily 24-hour profiles

        #### How to Access the Dataset

        ```bash
        # Install Kaggle API
        pip install kaggle

        # Download (requires Kaggle account and competition acceptance)
        kaggle competitions download -c ashrae-energy-prediction

        # Place files in:
        # ashrae-energy-prediction/train.csv
        # ashrae-energy-prediction/building_metadata.csv
        ```
        """)

    with tab2:
        st.markdown("""
        ### Stage 1: 1D Convolutional Autoencoder + KMeans Clustering

        #### Motivation
        Raw 24-hour load profiles contain substantial high-frequency noise from
        measurement error, irregular occupancy events, and equipment transients.
        Applying KMeans directly to raw profiles clusters on these noise components
        rather than the underlying operational regime. The 1D-CAE learns to filter
        this noise in its encoder, producing a compact latent representation that
        emphasizes the structural differences between regimes.

        #### Architecture Details

        **Encoder:**
        - `Conv1D(32, kernel=3, padding=1)` → ReLU → `MaxPool1D(2)`
        - `Conv1D(64, kernel=3, padding=1)` → ReLU → `MaxPool1D(2)`
        - `Flatten` → `Dense(16)` [latent space]

        **Decoder:**
        - `Dense(64)` → ReLU → `Upsample(2)`
        - `ConvTranspose1D(32, kernel=3, padding=1)` → ReLU → `Upsample(2)`
        - `ConvTranspose1D(1, kernel=3, padding=1)` → Sigmoid

        **Training:** Adam optimizer, lr=1e-3, 30 epochs, MSE reconstruction loss,
        batch size=256.

        #### Clustering
        - **Algorithm:** MiniBatch KMeans (k=3, batch_size=10,000)
        - **k selection:** Elbow method on within-cluster sum of squares
        - **Validation:** Silhouette Score on 5,000-point stratified subsample
        - **Result:** Three clusters corresponding to Scheduled Weekday,
          Reactive/Weekend, and Continuous 24/7 operational regimes
        """)

    with tab3:
        st.markdown("""
        ### Stage 2: N-BEATS Disaggregation

        #### Motivation
        N-BEATS (Oreshkin et al., 2020) was designed for univariate time series
        forecasting. We repurpose it as a **blind source separator** by training it
        to decompose the input signal into two physically interpretable components
        rather than to forecast future values.

        #### Key Insight
        Commercial building electricity loads have a natural two-component structure:
        - **Base load:** Slow-varying, temperature-driven (HVAC, servers, refrigeration)
          — well-captured by a *trend* model
        - **Lighting load:** Sharp, occupancy-driven pulses with strong daily periodicity
          — well-captured by a *seasonality* model

        This structural prior maps directly onto the N-BEATS architecture.

        #### Architecture Details

        **Trend Block:**
        - `Dense(256)` → ReLU → `Dense(256)` → ReLU → `Dense(24)`
        - Input: 24-hour aggregate profile
        - Output: smooth base load estimate

        **Seasonality Block:**
        - Identical architecture
        - Input: residual (aggregate − trend)
        - Output: isolated lighting component

        **Loss:** `MSE(Trend + Seasonality, Input)`

        **Training:** Adam optimizer, lr=1e-4, 50 epochs, full dataset (~500K profiles).

        #### Evaluation Protocol
        - **Primary metric:** Reconstruction MSE (lower = better decomposition quality)
        - **Baselines:** STL Decomposition (classical), Gradient Boosting Regressor (ML)
        - **Generalization:** Cross-climate test (train Site 3, test Site 13)
        - **Robustness:** Sensitivity analysis with injected Gaussian noise (σ ∈ [0, 0.20])

        #### References
        - Oreshkin, B.N. et al. (2020). N-BEATS: Neural basis expansion analysis for
          interpretable time series forecasting. *ICLR 2020*.
        - Miller, C. et al. (2020). The ASHRAE Great Energy Predictor III competition.
          *Science and Technology for the Built Environment*.
        """)
