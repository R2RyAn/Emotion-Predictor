import cv2
import torch
import numpy as np
import time
from models.load_model import load_trained_model
from utils.preprocess import preprocess_image

# Load the trained model
model = load_trained_model()
class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Initialize webcam
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Store last predictions
last_update_time = time.time()
last_predictions = {}  # Dictionary to store predictions per face

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    current_time = time.time()

    for i, (x, y, w, h) in enumerate(faces):
        face_id = i  # Unique ID for each detected face in frame

        # Only update predictions every 1 second
        if current_time - last_update_time >= 1:
            face = frame[y:y + h, x:x + w]
            img_tensor = preprocess_image(face)

            with torch.no_grad():
                output = model(img_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)  # Convert to probabilities
                percentages = probabilities.squeeze().tolist()  # Convert to list
                predicted_index = torch.argmax(output, 1).item()
                predicted_emotion = class_names[predicted_index]

            # Store latest prediction for this face
            last_predictions[face_id] = (predicted_emotion, percentages)

        # Retrieve stored prediction
        if face_id in last_predictions:
            predicted_emotion, percentages = last_predictions[face_id]

            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display top emotion
            cv2.putText(frame, f"Pred: {predicted_emotion}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Display all class probabilities
            y_offset = y + h + 20
            for j, (emotion, prob) in enumerate(zip(class_names, percentages)):
                text = f"{emotion}: {prob * 100:.2f}%"
                cv2.putText(frame, text, (x, y_offset + (j * 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Update time only when predictions are made
    if current_time - last_update_time >= 1:
        last_update_time = current_time

    # Show webcam feed
    cv2.imshow("Emotion Detector", frame)

    # Break loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
