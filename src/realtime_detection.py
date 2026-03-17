import cv2
import numpy as np
import threading
import os
import winsound
from keras.models import load_model

# ======================
# Alarm function
# ======================
def play_alarm():
    duration = 1000
    freq = 1000
    winsound.Beep(freq, duration)

# ======================
# Load model
# ======================
MODEL_PATH = "../models/drowsiness_model.h5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found")

model = load_model(MODEL_PATH, compile=False)

class_names = ['closed', 'no_yawn', 'open', 'yawn']

# ======================
# Load face cascade
# ======================
cascade_path = os.path.join(cv2.__path__[0], "data", "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_path)

# ======================
# Start webcam
# ======================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# ======================
# Main loop
# ======================
while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    status = "Alert"

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        face = face / 255.0
        face = np.expand_dims(face, axis=0)

        if model is not None:
         prediction = model.predict(face, verbose=0)

        confidence = float(np.max(prediction)) * 100
        label = class_names[np.argmax(prediction)]

        print("Prediction:", prediction)
        print("Label:", label)

        # ⭐ ADD THIS HERE
        if label in ["closed", "yawn"]:
        status = "DROWSY!"
        threading.Thread(target=play_alarm, daemon=True).start()
    else:
        status = "Alert"

        if label == "closed" or label == "yawn":
            status = "DROWSY!"
            threading.Thread(target=play_alarm, daemon=True).start()
            color = (0,0,255)
        else:
            status = "Alert"
            color = (0,255,0)

        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)

        text = f"State: {label}  Conf: {confidence:.1f}%"

        cv2.putText(frame, text, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.putText(frame, status, (30,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()