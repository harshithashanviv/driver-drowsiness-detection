import cv2

# Try different camera indexes if needed: 0, 1, 2
cap = cv2.VideoCapture(0)  

ret, frame = cap.read()

if ret:
    print("Camera opened successfully!")
    cv2.imshow("Camera Test", frame)
    cv2.waitKey(0)  # Press any key to close
else:
    print("Failed to open camera. Try a different index (0, 1, 2).")

cap.release()
cv2.destroyAllWindows()