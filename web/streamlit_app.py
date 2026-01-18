import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Stress316L Predictor", layout="wide")

st.title("Stress316L Flow Stress Prediction Dashboard")
st.write("Generate and visualize stress–strain curves using your trained model API.")

# Sidebar controls
st.sidebar.header("Controls")

strain_rate = st.sidebar.selectbox("Strain Rate", [1.0, 0.052, 0.0052, 0.00052])
strain_min = st.sidebar.number_input("Strain Min", value=0.00, step=0.01)
strain_max = st.sidebar.number_input("Strain Max", value=0.50, step=0.01)
n_points = st.sidebar.slider("Number of points", min_value=50, max_value=500, value=200)

plot_true_curve = st.sidebar.checkbox("Overlay true experimental curve (from CSV)", value=True)

# Load true curve if needed
df_true = None
if plot_true_curve:
    FEATURES_PATH = "C:/Users/aryuemaan/Videos/Projects/stress316l_flow_model/data/raw/features.csv"
    LABELS_PATH   = "C:/Users/aryuemaan/Videos/Projects/stress316l_flow_model/data/raw/labels.csv"

    features = pd.read_csv(FEATURES_PATH)
    labels = pd.read_csv(LABELS_PATH)

    df_true = features.copy()
    df_true["stress"] = labels["stress"].values

    df_true = df_true[df_true["strain_rate"] == strain_rate].sort_values("strain")

# Generate strains
strain_values = np.linspace(strain_min, strain_max, n_points)

# Call API for each strain (simple but reliable)
pred_stress = []
for s in strain_values:
    payload = {"strain": float(s), "strain_rate": float(strain_rate)}
    r = requests.post(f"{API_URL}/predict", json=payload)
    pred_stress.append(r.json()["stress"])

pred_stress = np.array(pred_stress)

# Plot
fig = plt.figure(figsize=(10, 6))
plt.plot(strain_values, pred_stress, linewidth=2, label="Predicted Curve (API)")

if df_true is not None:
    plt.plot(df_true["strain"], df_true["stress"], linewidth=2, linestyle="--", label="True Experimental Curve")

plt.xlabel("True Strain")
plt.ylabel("True Stress (MPa)")
plt.title(f"Stress–Strain Curve at strain_rate = {strain_rate}")
plt.grid(True, alpha=0.25)
plt.legend()

st.pyplot(fig)

# Show table
st.subheader("Predicted Values")
pred_df = pd.DataFrame({
    "strain": strain_values,
    "predicted_stress": pred_stress
})
st.dataframe(pred_df, use_container_width=True)
