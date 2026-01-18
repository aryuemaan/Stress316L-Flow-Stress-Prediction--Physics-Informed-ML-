# Stress316L Flow Stress Prediction (Physics-Informed ML)

## Overview
This project provides an end-to-end machine learning solution for predicting the **true flow stress** of **Austenitic Stainless Steel 316L (ASS316L)** during cold deformation using uniaxial tensile test data. The model predicts stress as a function of:

- **True Strain (ε)**
- **Strain Rate (ε̇)**

The system is designed with a production mindset and includes:
- Reproducible training with group-aware cross-validation
- Physics-informed regularization for physically meaningful curves
- Exportable artifacts (trained model + scaler) for deployment
- REST API deployment using FastAPI
- Optional graphical visualization via a web dashboard

---

## Problem Statement
The stress–strain behavior of ASS316L is known to show complex and sometimes unpredictable patterns, especially across different strain rates. Accurate modeling of flow stress is essential for:

- Constitutive material modeling
- Metal forming simulations (FEA/FEM)
- Process optimization in manufacturing
- Data-driven digital twins

This project frames the task as a **supervised regression problem**:

**Inputs**
- `strain` (true strain)
- `strain_rate` (true strain rate)

**Target**
- `stress` (true stress, MPa)

---

## Dataset
The dataset contains **15,858** samples at **4 different strain rates**. It is derived from uniaxial tensile tests performed at room temperature and converted to true stress/true strain.

Dataset source:
- Stress316L Kaggle Dataset

Files used:
- `features.csv` → input features (`strain`, `strain_rate`)
- `labels.csv` → regression target (`stress`)
- `x_y_initial.csv` → additional reference curve data (optional)

Expected columns:
- `features.csv`: `strain`, `strain_rate`
- `labels.csv`: `stress`

---

## Key Design Decisions

### 1) Group-Aware Validation (No Data Leakage)
Stress–strain points sampled from the same strain-rate curve are highly correlated. A random train/test split would artificially inflate metrics by leaking curve information.

This project uses **GroupKFold cross-validation** where:
- The group key is `strain_rate`
- Validation folds contain strain rates not used during training

This better represents real-world use where the model must generalize to unseen operating conditions.

### 2) Physics-Informed Training (Monotonicity + Smoothness)
To improve physical realism and stability, the neural network incorporates physics-inspired regularization:

- **Monotonic constraint:** penalizes negative slope (stress decreasing with increasing strain)
- **Smoothness constraint:** penalizes oscillations using a second-difference penalty

These constraints improve curve quality, reduce noise sensitivity, and increase generalization robustness.

---

## Project Structure

```
stress316l_flow_model/
├── requirements.txt
├── configs/
│   └── config.yaml
├── artifacts/
│   ├── stress316l_model.pt
│   └── stress316l_scaler.npz
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── metrics.py
│   ├── models.py
│   ├── utils.py
│   ├── train.py
│   └── infer.py
├── api/
│   └── app.py
└── tests/
    └── test_infer.py



Maintainer

Aryuemaan Kumar Chowdhury
Project: Stress316L Flow Stress Prediction (Physics-Informed ML)
