import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from utils.load import load_model
from utils.preprocess_image import preprocess_image


# --- Load Model ---
@st.cache_resource
def get_model(device):
    vae = load_model()
    return vae.to(device)



st.set_page_config(
    page_title="Brain Tumor MRI Anomaly Detection",
    layout="wide"
)

st.title("Brain Tumor MRI Anomaly Detection")
st.markdown("""
### Variational Autoencoder (VAE)–based Anomaly Detection

This application demonstrates:
- Unsupervised anomaly detection on **brain MRI scans**
- A **Variational Autoencoder (VAE)** trained on normal tissue
- Tumor detection via **reconstruction error**

---

### Pages
- **Training Results** -> Model performance, losses, reconstructions
- **Anomaly Detection** -> Upload MRI & detect anomalies interactively

---

### Method Summary
1. Train VAE on healthy MRI scans
2. Reconstruct input image
3. Compute pixel-wise reconstruction error
4. Threshold error to identify anomalous regions

---
  
### Dataset Reference

The dataset used for all model training and testing comes from
         
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset?select=Training
            

---

### Why VAE For Anomaly Detection?
            
The motivation for using Variational Autoencoders (VAEs) for anomaly detection is based on two key ideas:

- **Scarcity of anomalous data**: In most real-world scenarios, anomalous samples are rare or unavailable. VAEs can be trained exclusively on normal data, learning a compact probabilistic representation of the normal data distribution without requiring labeled anomalies.

- **Exploiting reconstruction error**: After training, the VAE can accurately reconstruct normal inputs. Samples that deviate from the learned distribution (i.e., anomalies are reconstructed poorly, resulting in higher reconstruction error, which serves as an effective anomaly score).

---
            
### Training Methodology
            
The core idea is to train a Variational Autoencoder (VAE) using only normal (no-tumor) images. By doing so, the model learns the underlying structure of healthy data without being exposed to anomalies. Specifically, the VAE learns to:

    - Model the latent distribution of normal data by regularizing the encoder output toward a known prior (typically a standard normal distribution).
    - Accurately reconstruct normal inputs through the decoder by minimizing reconstruction error.

Once trained, the VAE is evaluated on previously unseen data. When anomalous (tumor-containing) images are passed through the model, they deviate from the learned normal data distribution, leading to:

    - Higher KL divergence, indicating that the latent representation of the anomalous input lies far from the learned prior.
    - Increased reconstruction error, as the decoder struggles to faithfully reproduce features that were not present during training.

By combining these signals—reconstruction loss and KL divergence—we obtain a reliable anomaly score that enables effective discrimination between normal and anomalous cases.

--- 

### Model Architecture                     

#### Encoder
            
The Encoder is tasked with learning a **compressed representation of the input data** into parameters of some distribution (mean and variance) in the latent space.
This feature of the variational auto encoder is what enables the generation of new data (from sampling the learned latent space). Essentially what the encoder does
            
```
Input Data -> Feature Compression -> Probabilistic Latent Representation
```


In my implementation, the encoder compresses the original MRI scans from (1, 1, 256, 256) to (1, 32, 4, 4), where the dimensions correspond to (batch size, channels, height, width).

Within the encoder, there is 3 Convolutional blocks that preforms the input compression.

The encoder also utilizes a ResNet style skip connection, allowing for a direct connection from previous layers to the output of the current layer. 

This operation significantly reduces the spatial dimensionality of the input while increasing the number of feature channels, enabling the network to learn increasingly abstract representations.
            
The resulting tensor is then flattened and mapped to the parameters of a latent probability distribution, forcing the VAE to retain only the most salient features necessary to reconstruct the input image.




#### Decoder

The Decoder is tasked with reconstruction the original image, given a sample from the latent. 

Essentially the idea is that once a latent space distribution is learned, the decoder should be able to take some sample from this latent space and reconstruct the original input.        

In my implementation the Decoder consists of 6 blocks, used to upsample the compressed representation created from the encoder to the original dimension of the input.
            
Importantly, the decoder does not simply learn to invert the encoder.

Instead, it learns a probabilistic generative mapping that, when combined with the KL-divergence regularization imposed on the latent space, encourages smooth transitions and meaningful interpolation between latent samples.
            
This property enables the VAE to generate plausible new MRI scans by sampling from the latent prior rather than relying solely on encoded inputs.
            
""")



