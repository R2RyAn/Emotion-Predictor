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

# For storing predictions and detections
last_update_time = time.time()
last_predictions = {}  # Dictionary to store predictions per face

# Use a detection interval to reduce flickering:
detection_interval = 5  # Detect faces every 5 frames instead of every frame
frame_count = 0
last_faces = []  # Store the most recent detections

# FPS calculation variables
prev_frame_time = time.time()
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Calculate FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection only every 'detection_interval' frames
    if frame_count % detection_interval == 0:
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        last_faces = faces  # Save these detections for later frames
    else:
        faces = last_faces

    current_time = time.time()

    for i, (x, y, w, h) in enumerate(faces):
        face_id = i  # Use index as temporary ID for the detected face

        # Update predictions only once per second
        if current_time - last_update_time >= 1:
            face = frame[y:y+h, x:x+w]
            img_tensor = preprocess_image(face)

            with torch.no_grad():
                output = model(img_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                percentages = probabilities.squeeze().tolist()
                predicted_index = torch.argmax(output, 1).item()
                predicted_emotion = class_names[predicted_index]

            # Save the prediction for this face
            last_predictions[face_id] = (predicted_emotion, percentages)

        # Retrieve stored prediction if available
        if face_id in last_predictions:
            predicted_emotion, percentages = last_predictions[face_id]

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Pred: {predicted_emotion}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Display all class probabilities below the face
            y_offset = y + h + 20
            for j, (emotion, prob) in enumerate(zip(class_names, percentages)):
                text = f"{emotion}: {prob * 100:.2f}%"
                cv2.putText(frame, text, (x, y_offset + (j * 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Update the prediction update timer once per second
    if current_time - last_update_time >= 1:
        last_update_time = current_time

    # Draw FPS on the frame
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the video feed
    cv2.imshow("Emotion Detector", frame)

    # Break the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
