# Corporación Favorita Enterprise Demand Forecasting Engine

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org)
[![Machine Learning](https://img.shields.io/badge/Framework-LightGBM-green.svg)](https://lightgbm.readthedocs.io/)
[![Notebook Viewer](https://img.shields.io/badge/Render-Jupyter%20Nbviewer-orange.svg)](https://nbviewer.org/github/Petra-aou/Retail-Demand-Forecasting-Engine/blob/main/retail_demand_forecasting.ipynb)

An end-to-end scalable machine learning and automated ETL pipeline engineered to forecast large-scale, high-dimensional retail demand across multi-million-row transactional datasets.
An end-to-end scalable machine learning and automated ETL pipeline engineered to forecast large-scale, high-dimensional retail demand across multi-million-row transactional datasets.

## Project Architecture & Methodology

### 1. Data Alignment & Heterogeneous Integration (ETL)
* Built automated pipelines to ingest and align **1,782 parallel time-series** panel datasets with dynamic macro economic indicators (e.g., crude oil prices via forward-filling `ffill`) and national/regional calendar holidays.
* Resolved missing warehouse transactional timestamps using customized statistical imputation, enhancing real-time localized tracking inventory alignment.

### 2. Time-Series Diagnostics & Feature Engineering
* **Stationarity Analysis:** Deployed Augmented Dickey-Fuller (**ADF**) statistical tests to detect non-stationarity (P-Value approximately 0.0897). 
* **Temporal Dependency:** Utilized Auto-Correlation (**ACF**) and Partial Auto-Correlation (**PACF**) diagnostics to isolate Lag-0 operational artifacts and capture rigid 7-day cyclical seasonality.
* **Target Stabilization:** Implemented a logarithmic target transformation ln(y + 1) to structurally stabilize variances across long-tailed SKU sales distributions, mitigating Root Mean Squared Logarithmic Error (RMSLE) penalties.
* **Feature Matrix:** Constructed multi-granularity cross-week lag vectors (Lag 16/17/23) to train temporal momentum.

### 3. Predictive Modeling & Scaling via LightGBM
* Deployed a high-performance **LightGBM Regressor** using native categorical encoding to completely bypass traditional high-dimensional One-Hot memory bottlenecks.
* Fine-tuned gradient boosting hyperparameters via structured validation splits to prevent overfitting on historical operational drift, outputting an optimized **28,512-row demand blueprint** for enterprise execution.

---

## Repository Structure
* `retail_demand_forecasting.ipynb`: Core end-to-end pipeline containing automated ETL, statistical tests, feature engineering, and LightGBM model training.
* `README.md`: Professional executive summary of methodologies and engineering metrics.

---

## Core Stack Used
* **Languages:** Python (Pandas, NumPy, Scikit-Learn)
* **Algorithms:** LightGBM (Gradient Boosting Decision Trees)
* **Statistical Models:** Statsmodels (ADF Test, ACF/PACF Plots), SciPy
