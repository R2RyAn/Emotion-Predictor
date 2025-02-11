import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image  # Import PIL's Image module

# Define transformation for input image
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def preprocess_image(image):
    # Convert to grayscale using OpenCV
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize image using OpenCV
    resized = cv2.resize(gray, (48, 48))

    # Convert the NumPy array (resized) to a PIL Image
    pil_img = Image.fromarray(resized)

    # Apply the torchvision transforms on the PIL image
    img_tensor = transform(pil_img)

    # Add a batch dimension: shape becomes [1, C, H, W]
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor
