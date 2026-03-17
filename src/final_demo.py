import cv2
import numpy as np
from keras.models import load_model
import pygame
import os
import time

# ----------------------------
# Paths
# ----------------------------
MODEL_PATH = "../models/drowsiness_model.h5" # ensure this file exists here
ALARM_PATH = "alarm.wav"            # place a working alarm.wav in this folder

# ----------------------------
# Load Model
# ----------------------------
model = load_model(MODEL_PATH)

# ----------------------------
# Alarm Setup (SAFE)
# ----------------------------
pygame.mixer.init()
alarm_loaded = False
if os.path.exists(ALARM_PATH):
    try:
        pygame.mixer.music.load(ALARM_PATH)
        alarm_loaded = True
    except pygame.error:
        print("Warning: Alarm file exists but cannot be loaded!")
else:
    print("Warning: Alarm file not found! Alarm disabled.")

# ----------------------------
# Webcam Setup
# ----------------------------
cap = cv2.VideoCapture(0)
start_time = time.time()
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

font = cv2.FONT_HERSHEY_SIMPLEX

# ----------------------------
# Prediction Labels
# ----------------------------
classes = ["Closed", "Open", "Yawn", "No_Yawn"]  # update according to your model

# ----------------------------
# Main Loop
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    img = cv2.resize(frame, (64, 64))  # resize to your model's expected input
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    if model is None:
      print("Model not loaded!")
      break

    preds = model.predict(img)
    label = classes[np.argmax(preds)]
 
    elapsed_time = time.time() - start_time

    color = (0, 255, 0)  # green by default
    if label in ["Closed", "Yawn"]:
        color = (0, 0, 255)  # red
        if alarm_loaded and not pygame.mixer.music.get_busy():
            try:
                pygame.mixer.music.play()
            except:
                print("Failed to play alarm")

    cv2.putText(frame, f"Status: {label}", (10, 40), font, 1.2, color, 3)
    cv2.putText(frame, f"Time: {elapsed_time:.2f} sec", (10,30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    cv2.imshow("Driver Drowsiness Detection", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------------
# Cleanup
# ----------------------------
cap.release()
cv2.destroyAllWindows()
if alarm_loaded:
    pygame.mixer.music.stop()