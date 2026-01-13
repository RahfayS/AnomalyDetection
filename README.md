# VAE-Based Anomaly Detection

This repository implements an **anomaly detection pipeline using a Variational Autoencoder (VAE)**. The system is designed to learn the distribution of *normal* data and flag samples as anomalous when their reconstruction error exceeds a calibrated threshold.

The project includes:

* A PyTorch-based VAE model
* Training and evaluation loops with KL and reconstruction loss tracking
* Threshold selection for anomaly detection
* Visualization tools (original vs reconstruction, PCA of latent space)
* An optional **Streamlit** interface for interactive inference and visualization

---

## Motivation

In many real-world scenarios, anomalous samples are rare or poorly labeled. VAEs are well-suited for this setting because they:

* Learn a compact latent representation of normal data
* Reconstruct in-distribution samples well
* Produce higher reconstruction errors for out-of-distribution (anomalous) inputs

This project leverages reconstruction error as the primary anomaly score.

---

## Model Overview

### Variational Autoencoder (VAE)

The VAE consists of:

* **Encoder**: Maps input images to latent mean (μ) and log-variance (logσ²)
* **Latent space**: Regularized via KL-divergence toward a unit Gaussian
* **Decoder**: Reconstructs the input from sampled latent vectors

The loss function is:

```
Total Loss = Reconstruction Loss + KL Divergence
```

Where:

* Reconstruction loss measures how well the input is reconstructed
* KL divergence enforces latent space regularization

---

## Project Structure

```
.
├── main.py                      # Entry point (training or Streamlit app)
├── utils/
│   ├── load.py                  # Model loading utilities
│   ├── preprocess_image.py      # Image preprocessing utils
├── vae_model_utils/
│   └── vae_model_best.pth   
│   ├── vae_scores.json
│ 
├── notebooks/
│   └── AnomalyDetection.ipynb  # EDA & modelling
visualization
└── README.md
```

---

## Training

### Data Assumption

* The VAE is trained **only on normal (non-anomalous) data**
* Images are normalized and resized consistently before training

### Training Loop

During training, the following are tracked per epoch:

* Total loss
* Reconstruction loss
* KL divergence

These metrics are saved to disk for later visualization and diagnostics.

---

## Anomaly Detection

### Reconstruction Error

At inference time:

1. An input image is passed through the VAE
2. The reconstruction error is computed
3. The error is compared against a predefined threshold

```
Anomaly if: reconstruction_error > threshold
```

### Threshold Selection

The threshold was empirically determined during evaluation. The best-performing threshold was:

* **Reconstruction threshold:** `0.003821` (≈ 70%)

This value balances false positives and false negatives on the validation set.

---

## Visualization

### Original vs Reconstruction

Side-by-side visualizations allow qualitative inspection of reconstruction quality. Normal samples reconstruct cleanly, while anomalies exhibit visible artifacts or structural loss.

### Latent Space Analysis (PCA)

* Latent vectors are extracted from the encoder
* PCA is applied to project them into 2D or 3D
* Helps assess clustering and separation between normal and anomalous samples

---

## 🖥 Streamlit App (Optional)

The Streamlit interface provides:

* Image upload and preprocessing
* Real-time reconstruction
* Reconstruction error display
* Anomaly / normal classification
* Interactive visualizations

To run:

```
streamlit run main.py
```

---

##  Results & Observations

* The VAE learns a smooth latent space for normal samples
* Reconstruction error is an effective anomaly score
* PCA projections show tighter clusters for normal data
* Threshold tuning is critical for reliable detection

---

## Limitations

* Performance depends heavily on the quality and diversity of normal training data
* VAEs may reconstruct certain anomalies if they are visually similar to normal samples
* Thresholds may not generalize across datasets without recalibration

---

##  Future Work

* Incorporate perceptual or SSIM-based reconstruction loss
* Compare against Autoencoders, β-VAE, or VQ-VAE
* Use latent likelihood as an additional anomaly score
* Add ROC / PR curve evaluation
* Extend to temporal or multimodal anomaly detection

---

## License

This project is for research and educational purposes. Add a license file if redistribution is intended.

---

## Acknowledgements

Inspired by research on unsupervised anomaly detection using autoencoders and VAEs.


### Dataset

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset?select=Training

## How It works

![Demo](test_video/demo.gif)
