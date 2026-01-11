import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
def preprocess_image(image):
    """
    Docstring for preprocess_image
    
    :param image: Description
    """
    image = Image.open(image)
    
    # --- Define Image Transform ---
    transform = T.Compose([
        T.ToTensor(),
        T.Lambda(lambda image: image[:3, :, :]),  # RGBA → RGB
        T.Grayscale(num_output_channels=1),
        T.Resize((256, 256)),
        T.ConvertImageDtype(torch.float32),
    ])

    # --- Transform Image ---
    image = transform(image)
    image = image.detach().unsqueeze(1)
    return image


    