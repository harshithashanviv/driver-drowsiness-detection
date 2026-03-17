# 🚗 Driver Drowsiness Detection System

## 📌 Project Overview

The **Driver Drowsiness Detection System** is a real-time computer vision application that detects whether a driver is alert or drowsy using a webcam.
It helps in preventing road accidents by alerting the driver when signs of drowsiness are detected.

---

## 🎯 Objective

* To monitor driver alertness in real-time
* To detect eye closure and yawning
* To trigger an alarm when drowsiness is detected

---

## 🛠️ Technologies Used

* Python
* OpenCV
* TensorFlow / Keras
* NumPy
* Pygame / Winsound (for alarm)

---

## ⚙️ How It Works

1. Webcam captures live video
2. Face is detected using Haar Cascade
3. Face image is preprocessed
4. Deep Learning model predicts the state:

   * Open Eyes
   * Closed Eyes
   * Yawning
   * No Yawn
5. If **Closed Eyes or Yawning** is detected:

   * System marks **DROWSY**
   * Alarm is triggered

---

## 🧠 Model Details

* Model Type: Convolutional Neural Network (CNN)
* Input Size: 224x224 images
* Classes:

  * `open`
  * `closed`
  * `yawn`
  * `no_yawn`

---

## 🚨 Drowsiness Logic

```python
if label in ["closed", "yawn"]:
    status = "DROWSY!"
else:
    status = "Alert"
```

---

## ▶️ How to Run

1. Clone the repository:

```bash
git clone https://github.com/harshithashanviv/driver-drowsiness-detection.git
```

2. Navigate to project:

```bash
cd driver-drowsiness-detection/src
```

3. Activate virtual environment:

```bash
venv\Scripts\activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Run the project:

```bash
python final_demo.py
```

---

## ⏱️ Performance

* Real-time detection using webcam
* Displays prediction instantly
* Shows elapsed time on screen
* Triggers alert within seconds

---

## 📷 Output

* Webcam window opens
* Displays:

  * Face detection box
  * Prediction label
  * Status (Alert / Drowsy)
  * Time taken

---

## 📂 Project Structure

```
driver-drowsiness-detection/
│── src/
│   ├── final_demo.py
│   ├── realtime_detection.py
│
│── models/
│   └── drowsiness_model.h5
│
│── README.md
│── .gitignore
```

---

## 🔮 Future Enhancements

* Mobile app integration
* Night vision support
* Driver monitoring system in cars
* Integration with IoT devices

---

## 👩‍💻 Author

**Harshitha Shanvi Vempalli**

---

## ⭐ Acknowledgment

This project was developed as part of an academic final year project to demonstrate real-time AI-based safety systems.

---
