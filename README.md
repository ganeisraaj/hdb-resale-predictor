# HDB Resale Price Predictor
**Singapore Public Housing Resale Price Estimation**

---

## 🚀 Live App

👉 **[HDB Resale Price Predictor — Live App](https://ganeisraaj-hdb-resale-predictor.streamlit.app/)**

Select your flat details and get an instant price estimate with town context.

---

## Overview

Singapore's HDB resale market transacts billions of dollars every year. This project builds a machine learning model to predict resale prices from flat characteristics using 238,000+ real transactions from 2017 to 2025.

The model achieves R² = 0.933 and MAE ≈ S$37,000 on held-out test data — meaning typical predictions are within 7% of the actual transaction price.

---

## Data

| | |
|---|---|
| **Source** | data.gov.sg — HDB Resale Flat Prices |
| **Period** | January 2017 to 2025 |
| **Observations** | 238,815 transactions |
| **Target** | Resale price (SGD) |

---

## Features

| Feature | Description |
|---|---|
| `town` | HDB town (26 towns, one-hot encoded) |
| `flat_type` | Flat size category (2 ROOM to EXECUTIVE) |
| `floor_area_sqm` | Floor area in square metres |
| `storey_mid` | Midpoint of storey range |
| `lease_remaining` | Remaining lease in decimal years |
| `year` | Transaction year (captures price inflation) |

---

## Key Findings

- Bukit Timah is the most expensive town (median S$788k), Ang Mo Kio the least (median S$420k)
- Flat type and floor area are the strongest predictors (correlation ~0.58 and 0.56 with price)
- Transaction year has correlation 0.41 — prices have risen significantly since 2017
- Gradient Boosting outperforms Linear Regression (R² 0.933 vs 0.886)
- Log-transforming the target variable improves model fit due to right-skewed price distribution

---

## Methods

- Feature engineering: storey midpoint extraction, remaining lease parsing, flat type encoding
- Baseline: Linear Regression on log(price) — R² = 0.886, MAE = S$48,354
- Final model: Gradient Boosting Regressor — R² = 0.933, MAE = S$37,203
- Town effect captured via one-hot encoding (26 dummy variables)

---

## Software

Python 3.12 · `pandas` · `numpy` · `scikit-learn` · `streamlit`
