import torch
import numpy as np
import streamlit as st
from utils.load import load_model
from utils.preprocess_image import preprocess_image
import matplotlib.pyplot as plt
import plotly.express as px
# --- Load Model ---
@st.cache_resource
def get_model(device):
    vae = load_model()
    return vae.to(device)

# --- Streamlit App ---
def main():
    st.set_page_config(layout="wide")
    st.title("Brain Tumor MRI Anomaly Detection")

    # --- Prompt user to load a Brain MRI Scan ---
    image = st.file_uploader("Upload a MRI scan (JPG)", type=["jpg"])
    if image is None:
        st.info("Upload a MRI scan to begin")
        return
    
    # --- Initialize Device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load Model ---
    vae = get_model(device)

    # --- Transform Image ---
    image = preprocess_image(image)
    image = image.to(device)

    # --- Pass Image to Model ---
    with torch.no_grad():
        recon, mu, logvar = vae(image)

    # --- Plot Reconstruction ---
    recon_img = recon.squeeze(0).detach().cpu()   # [3, H, W]
    recon_img = recon_img.permute(1, 2, 0)        # [H, W, 3]



if __name__ == '__main__':
    main()