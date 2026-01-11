import torch
import os
from model import VAE

def load_model(pth_path = 'vae_model_utils/vae_model_best.pth', device = None):
    """
    Loads state dict of Variational AutoEncoder
    
    Args:
        pth_path (str): Path to scratch model
        device (torch.device): CPU or GPU. If None, auto-detect.
    
    Returns:
        model (nn.Module): Ready-to-use model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- Initialize Model
    vae = VAE(channel_in=1, z = 32).to(device)
    vae.load_state_dict(
        torch.load(pth_path,map_location=device)
    )

    print('Model Loaded')
    vae.eval()

    return vae
