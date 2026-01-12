import torch
import streamlit as st
import matplotlib.pyplot as plt
import torch.nn.functional as F

from utils.load import load_model
from utils.preprocess_image import preprocess_image


@st.cache_resource
def get_model(device):
    vae = load_model()
    return vae.to(device)

def compute_recon_scores(recon, original):
    recon_err = F.mse_loss(recon, original, reduction="none")
    recon_err = recon_err.flatten(1).mean(dim=1)
    return recon_err


def is_anomaly(recon_loss, recon_thresh):
    return bool(recon_loss > recon_thresh)


def error_and_mask(orig, recon, thresh):
    err = (orig - recon).pow(2).mean(dim=1)
    mask = err > thresh
    return err, mask


st.set_page_config(layout="wide")
st.title("Interactive Anomaly Detection")

# --- Sidebar Controls ---
st.sidebar.header("Settings")

# Best threshold from training / evaluation
BEST_RECON_THRESH = 0.003821  # 70th percentile (validation set)

# Initialize session state
if "recon_thresh" not in st.session_state:
    st.session_state.recon_thresh = BEST_RECON_THRESH

# Reset button
if st.sidebar.button("Reset to Best Threshold"):
    st.session_state.recon_thresh = BEST_RECON_THRESH

# Fully movable slider
recon_thresh = st.sidebar.slider(
    "Reconstruction Threshold",
    min_value=0.0,
    max_value=1.0,
    value=float(st.session_state.recon_thresh),
    step=0.001,
    help="Default value comes from training; move freely to explore sensitivity."
)

# Sync slider → session state
st.session_state.recon_thresh = recon_thresh

# Explanation
st.sidebar.info(
    f"""
**Best Threshold from Training:** `{BEST_RECON_THRESH:.6f}`  

• Selected using validation data (normal samples only)  
• 70th percentile of image-level reconstruction error  
• Slider allows exploration beyond training optimum  

Lower values -> higher sensitivity  
Higher values -> more conservative detection
"""
)

overlay_alpha = st.sidebar.slider(
    "Overlay Opacity",
    0.0, 1.0, 0.35, 0.05
)

show_mask = st.sidebar.checkbox("Show Anomaly Mask", True)

# --- Upload ---
image_file = st.file_uploader(
    "Upload a Brain MRI scan",
    type=["jpg", "png"]
)

if image_file is None:
    st.info("Upload a MRI scan to begin")
    st.stop()

# ---Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.write(f"Device: **{device}**")

# --- Load Model ---
vae = get_model(device)
vae.eval()

# --- Preprocess ---
image = preprocess_image(image_file).to(device)

# --- Forward ---
with torch.no_grad():
    recon, _, _ = vae(image)

image = image.clamp(0, 1)
recon = recon.clamp(0, 1)

# --- Reconstruction Score ---
recon_score = compute_recon_scores(recon, image)
recon_score_val = recon_score.item()

prediction = is_anomaly(recon_score_val, recon_thresh)

st.markdown("### Model Prediction")

if prediction:
    st.error("**Prediction: Anomalous Scan**")
else:
    st.success("**Prediction: Normal Scan**")


# --- Error ---
err, mask = error_and_mask(image, recon, recon_thresh)
err_norm = (err - err.min()) / (err.max() - err.min() + 1e-8)

# --- To NumPy ---
orig_img = image.squeeze(0).cpu().permute(1, 2, 0).numpy()
recon_img = recon.squeeze(0).cpu().permute(1, 2, 0).numpy()
err_img = err_norm.squeeze(0).cpu().numpy()
mask_img = mask.squeeze(0).cpu().numpy()

# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(24, 7))

axes[0].imshow(orig_img)
axes[0].set_title("Original MRI")
axes[0].axis("off")

axes[1].imshow(recon_img)
axes[1].set_title("Reconstruction")
axes[1].axis("off")

axes[2].imshow(err_img, cmap="hot")
if show_mask:
    axes[2].imshow(mask_img, cmap="Reds", alpha=overlay_alpha)
axes[2].set_title("Error + Anomaly Overlay")
axes[2].axis("off")

st.pyplot(fig, use_container_width=True)

# --- Metrics ---
st.markdown("### 🔍 Anomaly Statistics")
col1, col2, col3 = st.columns(3)

col1.metric("Mean Pixel Error", f"{err.mean().item():.6f}")
col2.metric("Anomalous Pixel %", f"{mask.float().mean().item()*100:.2f}%")
col3.metric("Reconstruction Score", f"{recon_score_val:.6f}")
st.caption(
    "The prediction is based on the image-level reconstruction score. "
    "If the average reconstruction error exceeds the selected threshold, "
    "the scan is classified as anomalous, indicating deviation from the "
    "learned normal data distribution."
)


st.markdown(
    """
###  Reconstruction Threshold Explained

The **reconstruction threshold** controls how sensitive the model is when identifying anomalous regions.

Each pixel is assigned a reconstruction error based on how well the VAE can reproduce it.  
Pixels with an error **above the selected threshold** are classified as anomalous.

**Effect of the threshold value:**
- **Lower threshold** -> more sensitive detection  
  - Captures subtle abnormalities  
  - May introduce false positives (noise)
- **Higher threshold** -> more conservative detection  
  - Highlights only strong deviations from normal anatomy  
  - May miss small or low-contrast anomalies

In practice, this threshold represents a trade-off between **sensitivity** and **specificity**, and is typically selected using validation data containing only normal samples.
"""
)
