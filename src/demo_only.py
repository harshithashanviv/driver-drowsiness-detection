import cv2
import numpy as np
import pygame
import random

# ----------------------------
# Alarm Setup
# ----------------------------
ALARM_PATH = "alarm.mp3"
pygame.mixer.init()
pygame.mixer.music.load(ALARM_PATH)

# ----------------------------
# Webcam Setup
# ----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

font = cv2.FONT_HERSHEY_SIMPLEX
alarm_playing = False

# ----------------------------
# Fake classes for demo
# ----------------------------
classes = ["Closed", "Open", "Yawn", "No_Yawn"]

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Random label for demo
    label = random.choice(classes)

    if label in ["Closed", "Yawn"]:
        color = (0,0,255)
        if not alarm_playing:
            pygame.mixer.music.play()
            alarm_playing = True
    else:
        color = (0,255,0)
        alarm_playing = False

    cv2.putText(frame, f"Status: {label}", (10,40), font, 1.2, color, 3)
    cv2.imshow("Driver Drowsiness Demo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()