import torch
import torchvision.transforms as T
from PIL import Image

def preprocess_image(image):
    """
    Preprocess MRI image for VAE inference
    Output shape: (1, 1, 256, 256)
    Output range: [-1, 1]
    """
    image = Image.open(image)

    # --- Define Image Transform ---
    transform = T.Compose([
        T.ToTensor(),                         # [0,1]
        T.Lambda(lambda x: x[:3, :, :]),      # RGBA → RGB
        T.Grayscale(num_output_channels=1),   # RGB → 1 channel
        T.Resize((256, 256)),
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=[0.5], std=[0.5])    # [0,1] → [-1,1]
    ])

    # --- Transform Image ---
    image = transform(image)

    # --- Add batch dimension ---
    image = image.unsqueeze(0)  # (1, 1, 256, 256)

    return image
