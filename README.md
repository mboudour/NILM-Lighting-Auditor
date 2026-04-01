# Closing the Verification Gap: Non-Intrusive Auditing of Lighting Efficiency in Commercial Buildings

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-green)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This repository contains the full code, trained model weights, and interactive Streamlit demo for the paper
*"Closing the Verification Gap: Non-Intrusive Auditing of Lighting Efficiency in Commercial Buildings"*.

We propose a fully **unsupervised, two-stage deep learning framework** that audits lighting efficiency
in commercial buildings using only standard hourly aggregate electricity meter data — **no hardware
sub-meters required**.

### The Problem

Commercial buildings routinely consume 20--60% more energy than predicted at design time — the
**verification gap**. A primary driver is suboptimal lighting control: static schedules that leave
lights on in unoccupied spaces. Fixing this requires knowing *when* and *where* lighting is being
wasted, but hardware sub-metering costs £500--£5,000 per circuit.

### Our Solution

Extract the same information from the smart meter data already being collected, using deep learning.

---

## Pipeline

```
Raw Hourly Smart Meter Data (ASHRAE GEPIII)
            |
            v
+-----------------------------+
|  Stage 0: Preprocessing     |  stage0_preprocessing.py
|  - Filter electricity meters|
|  - Handle missing values    |
|  - Pivot to daily profiles  |
|  - Min-Max normalize        |
+-------------+---------------+
              |  ~500,000 normalized 24-hour profiles
              v
+-----------------------------+
|  Stage 1: 1D-CAE Clustering |  stage1_clustering.py
|  - 1D Convolutional AE      |
|  - 16-dim latent space      |
|  - MiniBatch KMeans (k=3)   |
|  - Silhouette validation    |
+-------------+---------------+
              |  3 operational regime clusters
              v
+-----------------------------+
|  Stage 2: N-BEATS Disagg.   |  stage2_disagg.py
|  - Trend block (base load)  |
|  - Seasonality block (light)|
|  - Cross-climate eval       |
|  - Sensitivity analysis     |
+-------------+---------------+
              |
              v
    Isolated Lighting Profiles
    + Actionable Efficiency Audit
```

---

## Key Results

| Metric | Value |
|---|---|
| Buildings analysed | 1,448 (16 climate zones) |
| Daily profiles processed | ~500,000 |
| Clustering: CAE + KMeans Silhouette | **0.3728** (vs. 0.3480 baseline) |
| Disaggregation: N-BEATS MSE | **0.0628** (vs. 0.1112 Gradient Boosting) |
| Cross-climate generalization | Train MSE 0.1306 -> Test MSE **0.1253** |
| Robustness to noise (sigma=0.20) | MSE **0.0598** (stable) |

---

## Repository Structure

```
.
+-- stage0_preprocessing.py      # Data loading, cleaning, normalization
+-- stage1_clustering.py         # 1D-CAE training + KMeans + t-SNE
+-- stage2_disagg.py             # N-BEATS disaggregation + all evaluations
+-- requirements.txt             # Python dependencies
+-- README.md
+-- LICENSE
|
+-- manuscript/
|   +-- paper.tex                # Main LaTeX manuscript
|   +-- section_introduction.tex
|   +-- section_related_work.tex
|   +-- section_methodology.tex
|   +-- section_experiments.tex
|   +-- section_conclusion.tex
|   +-- svproc.cls               # Springer proceedings class
|   +-- figures/                 # All paper figures (PNG, 300 dpi)
|
+-- streamlit_app/
    +-- app.py                   # Interactive demo application (8 sections)
    +-- nbeats_disaggregator.pt  # Pre-trained N-BEATS weights
    +-- requirements.txt
    +-- assets/
        +-- plots/               # Pre-computed figures from the paper
        +-- data/                # Pre-computed result CSVs and JSON
```

---

## Installation

```bash
git clone https://github.com/mboudour/NILM-Lighting-Auditor.git
cd NILM-Lighting-Auditor
pip install -r requirements.txt
```

---

## Dataset

The pipeline uses the **ASHRAE Great Energy Predictor III** dataset, publicly available on Kaggle:

```
https://www.kaggle.com/competitions/ashrae-energy-prediction
```

Download and place the files in an `ashrae-energy-prediction/` directory:

```
ashrae-energy-prediction/
+-- train.csv              (~678 MB)
+-- test.csv               (~1.46 GB)
+-- building_metadata.csv  (~46 KB)
+-- weather_train.csv      (~7.5 MB)
+-- weather_test.csv       (~14.8 MB)
```

> **Note:** The raw dataset is NOT included in this repository. Only the pre-trained model weights
> and pre-computed result files are provided. The Streamlit demo runs entirely from the
> pre-trained weights and does not require the raw data.

---

## Reproducing Results

```bash
# Step 1: Preprocess the raw ASHRAE data (~10 min)
python stage0_preprocessing.py

# Step 2: Train the 1D-CAE and cluster (~30 min on GPU)
python stage1_clustering.py

# Step 3: Train N-BEATS and run all evaluations (~20 min on GPU)
python stage2_disagg.py
```

All outputs (figures, CSVs, model weights, logs) will be saved to `outputs/`.

---

## Interactive Streamlit Demo

Run the demo locally without needing the raw dataset:

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

The app includes 8 sections:

| Section | Content |
|---|---|
| Overview & Key Results | Headline metrics, pipeline description, paper figures |
| Stage 1: Clustering | t-SNE visualization, silhouette score comparison |
| Stage 2: Disaggregation | Baseline comparison table and annotated charts |
| Cross-Climate Generalization | Train/test MSE across climate zones |
| Sensitivity & Robustness | Interactive noise slider with live N-BEATS inference |
| Case Study | Day-by-day analysis of an educational building |
| Live Demo | Upload your own building CSV for real-time disaggregation |
| Methodology & Dataset | Full technical documentation and dataset access guide |

---

## Citation

If you use this code or the associated paper, please cite:

```bibtex
@article{boudour2026verification,
  title   = {Closing the Verification Gap: Non-Intrusive Auditing of
             Lighting Efficiency in Commercial Buildings},
  author  = {Boudour, M.},
  year    = {2026},
  note    = {Manuscript under review}
}
```

---

## License

Copyright (c) 2026 M. Boudour. This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
