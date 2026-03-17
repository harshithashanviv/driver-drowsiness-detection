import cv2
import numpy as np
import pygame
from tensorflow.keras.models import load_model
import os

# ----------------------------
# Paths
# ----------------------------
MODEL_PATH = "drowsiness_model.h5"  # put in src
ALARM_PATH = "alarm.wav"  # make sure this matches the file in src           # put in src

# ----------------------------
# Load Model
# ----------------------------
model = load_model(MODEL_PATH, compile=False)

# ----------------------------
# Alarm Setup
# ----------------------------
pygame.mixer.init()
pygame.mixer.music.load(ALARM_PATH)

# ----------------------------
# Webcam Setup
# ----------------------------
cap = cv2.VideoCapture(0)  # change to 1 if 0 doesn't work
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

font = cv2.FONT_HERSHEY_SIMPLEX

# ----------------------------
# Labels
# ----------------------------
classes = ["Closed", "Open", "Yawn", "No_Yawn"]

# ----------------------------
# Run Loop
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot read frame, trying next...")
        continue

    # Resize and normalize
    img = cv2.resize(frame, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, 0)  

    # Predict
    preds = model.predict(img)
    label = classes[np.argmax(preds)]

        # -------------------------
    # Display status and alarm
    # -------------------------
    cv2.putText(frame, f"Status: {label}", (10, 40), font, 1.2, (0,255,0), 3)

    # Alarm triggers only for drowsy signs
    if label in ["Closed", "Yawn"]:  
        if not pygame.mixer.music.get_busy():  # avoids overlapping alarms
            pygame.mixer.music.play()
            
        cv2.imshow("Driver Drowsiness Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display and trigger alarm once
    color = (0, 255, 0)
    if label in ["Closed", "Yawn"]:
        color = (0, 0, 255)
        if not pygame.mixer.music.get_busy():
            if pygame.mixer.get_init():       # check mixer is initialized
               if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play()

    cv2.putText(frame, f"Status: {label}", (10, 40), font, 1.2, color, 3)
    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------------
# Cleanup
# ----------------------------
cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()