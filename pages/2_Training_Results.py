import json
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    recall_score,
    precision_score,
    f1_score,
    roc_curve,
    auc
)

# --- Load Model Scores from testing (see colab notebook) ---
@st.cache_data
def load_scores():
    with open("vae_model_utils/vae_scores.json", "r") as f:
        scores = json.load(f)

    # --- Extract scores for normal and anomaly ---
    normal_scores = scores["normal"]
    anomaly_scores = scores["anomaly"]

    return normal_scores, anomaly_scores


normal_scores, anomaly_scores = load_scores()


# --- Page Setup ---
st.set_page_config(layout="wide")
st.title("Training & Testing Results")

# --------------------------------------------------
# Methods
# --------------------------------------------------
st.markdown("## Methods Evaluated")

st.markdown("""
Several anomaly scoring strategies were evaluated during training and validation:

**1. Image-level reconstruction loss**  
Mean squared reconstruction error aggregated across all pixels.

**2. Pixel-level anomaly scoring**  
Per-pixel reconstruction error used to localize anomalous regions.

**3. KL divergence–based scoring**  
Deviation of latent posterior from the prior distribution.

**4. Combined score (Reconstruction + KL)**  
Weighted sum of reconstruction loss and KL divergence.

Among these, **image-level reconstruction loss** provided the most stable and interpretable results
and was selected as the primary decision metric.
""")

# --- Threshold Selection ---
st.markdown("## Threshold Selection")

st.markdown("""
Since the model was trained exclusively on **normal samples**, anomaly thresholds were derived
using reconstruction error statistics from the normal validation set.

A **percentile-based approach** was used to control the false positive rate.
""")

# --- Prepare Labels & Scores ---
y_scores = np.concatenate([
    normal_scores["recon"],
    anomaly_scores["recon"]
])

y_true = np.concatenate([
    np.zeros(len(normal_scores["recon"])),   # normal = 0
    np.ones(len(anomaly_scores["recon"]))    # anomaly = 1
])

# --- Threshold Sensitivity Analysis ---
st.markdown("## Threshold Sensitivity Analysis")

percentiles = [70, 75, 80, 85, 90, 95, 99]
rows = []

for p in percentiles:
    t = np.percentile(normal_scores["recon"], p)
    y_pred = (y_scores > t).astype(int)

    rows.append({
        "Percentile": p,
        "Threshold": round(t, 6),
        "Recall": recall_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "F1-score": f1_score(y_true, y_pred)
    })

st.dataframe(rows, use_container_width=True)

# --- Selected Operating Point ---
st.markdown("## Selected Operating Point")

selected_p = st.slider(
    "Percentile of normal reconstruction error",
    min_value=70,
    max_value=99,
    step=1,
    value=95
)

recon_thresh = np.percentile(normal_scores["recon"], selected_p)
y_pred = (y_scores > recon_thresh).astype(int)

st.markdown(
    f"**Reconstruction threshold:** `{recon_thresh:.6f}` "
    f"(Percentile = {selected_p})"
)

#  --- Classification Report ---
# 
st.markdown("## Classification Report")

report = classification_report(
    y_true,
    y_pred,
    target_names=["Normal", "Anomalous"]
)

st.code(report)

# --- Plots ---

cm = confusion_matrix(y_true, y_pred)
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)


col1, col2 = st.columns(2)

with col1:
    st.markdown("### Confusion Matrix")
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cbar=False, ax=ax)
    ax.set_title("Confusion Matrix", fontsize=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    st.pyplot(fig)

with col2:
    st.markdown("### ROC Curve")
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)



# --- Evaluation Summary ---
st.markdown("## Evaluation Methodology")

st.markdown("""
- The VAE was trained exclusively on **normal samples**
- Reconstruction error was used as the anomaly score
- Decision thresholds were derived **only from normal data**
- Percentile-based thresholding avoids tuning on anomalous samples
- Final evaluation was performed on a held-out test set containing both normal and anomalous scans
            
- Best results where found with a percentile of 70% or 0.003821 reconstruction threshold

""")

st.markdown("## Improvements")

st.markdown("""
- The model was relatively shallow, a deeper model may serve to learn the latent distribution of normal images. Which results in a better separation of reconstruction scores between normal and anomaly data
- Greater fine tuning of the models hyper parameters (learning rate, batch size, latent dimension, etc) could also serve to improve performance here.
            
""")
