import torch
import cv2
import sys
from models.load_model import load_trained_model
from utils.preprocess import preprocess_image

# Load the model
model = load_trained_model()
class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Load image from argument
image_path = sys.argv[1]  # Run script with `python test/test_image.py image.jpg`
image = cv2.imread(image_path)

# Preprocess image
img_tensor = preprocess_image(image)

# Make prediction
with torch.no_grad():
    output = model(img_tensor)
    _, predicted = torch.max(output, 1)
    emotion = class_names[predicted.item()]

print(f"Predicted Emotion: {emotion}")
