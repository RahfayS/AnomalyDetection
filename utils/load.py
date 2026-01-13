import torch
from model import VAE
from huggingface_hub import hf_hub_download
import streamlit as st

@st.cache_resource(show_spinner="Loading VAE model...")
def load_model(device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = hf_hub_download(
        repo_id="Rahfay/vae-anomaly-detection",
        filename="vae-anomaly-v1.pth"
    )

    vae = VAE(channel_in=1, z=32).to(device)
    vae.load_state_dict(torch.load(model_path, map_location=device))
    vae.eval()
    return vae
