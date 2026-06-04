# Closing the Verification Gap: Non-Intrusive Auditing of Lighting Efficiency in Commercial Buildings

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-green)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This repository contains the full source code, pre-trained model weights, and all computation outputs for the paper *"Closing the Verification Gap: Non-Intrusive Auditing of Lighting Efficiency in Commercial Buildings"* (Moses Boudourides and Eleni Savvidou, 2026).

We propose a fully **unsupervised, two-stage deep learning framework** that audits lighting efficiency in commercial buildings using only standard hourly aggregate electricity meter data — **no hardware sub-meters required**.

### The Problem

Commercial buildings routinely consume 20–60% more energy than predicted at design time — the **verification gap**. A primary driver is suboptimal lighting control: static schedules that leave lights on in unoccupied spaces. Fixing this requires knowing *when* and *where* lighting is being wasted, but hardware sub-metering costs £500–£5,000 per circuit.

### Our Solution

Extract the same information from the smart meter data already being collected, using deep learning.

---

## Pipeline

```
Raw Hourly Smart Meter Data (ASHRAE GEPIII)
            |
            v
+-----------------------------+
|  Stage 0: Preprocessing     |  scripts/stage0_preprocessing.py
|  - Filter electricity meters|
|  - Handle missing values    |
|  - Pivot to daily profiles  |
|  - Min-Max normalize        |
+-------------+---------------+
              |  ~500,000 normalized 24-hour profiles
              v
+-----------------------------+
|  Stage 1: 1D-CAE Clustering |  scripts/stage1_clustering.py
|  - 1D Convolutional AE      |
|  - 16-dim latent space      |
|  - MiniBatch KMeans (k=3)   |
|  - Silhouette validation    |
+-------------+---------------+
              |  3 operational regime clusters
              v
+-----------------------------+
|  Stage 2: N-BEATS Disagg.   |  scripts/stage2_disagg.py
|  - Trend block (base load)  |
|  - Seasonality block (light)|
|  - Cross-climate eval       |
|  - Sensitivity analysis     |
+-------------+---------------+
              |
              v
    Isolated High-Frequency Components
    Consistent with Lighting Behavior
    + Actionable Efficiency Audit
```

---

## Key Results

| Metric | Value |
|---|---|
| Buildings analysed | 1,448 (16 climate zones) |
| Daily profiles processed | ~500,000 |
| Clustering Silhouette Score (CAE features) | **0.3728** (95% CI: [0.3226, 0.3992]) |
| Clustering Silhouette Score (raw features, baseline) | 0.3480 |
| Optimal number of clusters (k) | **3** (peak Silhouette across k=2,3,4,5) |
| Disaggregation MSE (N-BEATS) | **0.0628** |
| Disaggregation MSE (Gradient Boosting baseline) | 0.1112 |
| Cross-climate generalization (train → test MSE) | 0.1306 → **0.1253** |
| Robustness to noise (σ=0.20) | MSE **0.0598** (stable) |
| Ablation: removing Trend Block | +46.3% MSE degradation |
| Ablation: removing Seasonality Block | +62.0% MSE degradation |
| Inference throughput | ~567,000 profiles/second (CPU) |
| Full pipeline training time | ~47 minutes (CPU, ~500K profiles) |

---

## Repository Structure

```
NILM-Lighting-Auditor/
├── computations/
│   ├── scripts/
│   │   ├── stage0_preprocessing.py       # Data loading, cleaning, normalization
│   │   ├── stage1_clustering.py          # 1D-CAE training, KMeans, t-SNE
│   │   ├── stage2_disagg.py              # N-BEATS disaggregation, evaluation, case study
│   │   ├── ablation_study.py             # N-BEATS ablation study (reviewer addition)
│   │   ├── reviewer_revisions.py         # Clustering sensitivity & bootstrap CI (reviewer addition)
│   │   ├── plot_ablation.py              # Ablation bar chart figure
│   │   ├── plot_clustering_sensitivity.py  # Clustering sensitivity figure
│   │   └── pipeline_diagram.mmd          # Mermaid source for architecture diagram (Fig. 1)
│   ├── outputs/
│   │   ├── figures/                      # All paper figures (PNG, 300 dpi)
│   │   ├── tables/                       # All paper tables (LaTeX .tex files)
│   │   ├── models/                       # Pre-trained model weights (.pt)
│   │   └── logs/                         # Pipeline execution log, ablation results JSON
│   └── requirements.txt                  # Python dependencies
├── manuscript/
│   ├── paper.pdf                         # Camera-ready paper (PDF)
│   └── README.md
├── .gitignore
├── LICENSE                               # MIT License
└── README.md
```

---

## Dataset

The pipeline uses the **ASHRAE Great Energy Predictor III** dataset, publicly available on Kaggle:

> https://www.kaggle.com/competitions/ashrae-energy-prediction

The raw dataset is **not included** in this repository. Download it from Kaggle and place the files in a local `data/ashrae-energy-prediction/` directory before running the pipeline scripts.

Required files:

```
data/ashrae-energy-prediction/
├── train.csv              (~678 MB)
├── building_metadata.csv  (~46 KB)
└── weather_train.csv      (~7.5 MB)
```

---

## Reproducing Results

```bash
# 1. Clone the repository
git clone https://github.com/mboudour/NILM-Lighting-Auditor.git
cd NILM-Lighting-Auditor

# 2. Install dependencies
pip install -r computations/requirements.txt

# 3. Download the ASHRAE dataset from Kaggle (see above)
#    Place files in: data/ashrae-energy-prediction/

# 4. Run the pipeline
python computations/scripts/stage0_preprocessing.py   # ~10 min
python computations/scripts/stage1_clustering.py      # ~30 min
python computations/scripts/stage2_disagg.py          # ~20 min

# 5. (Optional) Reproduce reviewer-response analyses
python computations/scripts/reviewer_revisions.py     # Clustering sensitivity + bootstrap CI
python computations/scripts/ablation_study.py         # N-BEATS ablation study
```

All output figures, tables, model weights, and logs will be saved under `computations/outputs/`.

---

## Paper

The camera-ready paper is available in [`manuscript/paper.pdf`](manuscript/paper.pdf).

**Citation** (to be updated with full proceedings reference upon publication):

```bibtex
@inproceedings{boudourides2026verification,
  title     = {Closing the Verification Gap: Non-Intrusive Auditing of
               Lighting Efficiency in Commercial Buildings},
  author    = {Boudourides, Moses},
  year      = {2026},
  note      = {Manuscript under review}
}
```

---

## License

Copyright (c) 2026 Moses Boudourides. This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
