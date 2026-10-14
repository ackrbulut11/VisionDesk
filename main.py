
import cv2
from ultralytics import YOLO

# Pretrained object detection model
model = YOLO("yolo11n.pt")

# Open default camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Camera could not be opened")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Run object detection
    results = model(frame, verbose=False)

    # Draw bounding boxes and labels
    annotated_frame = results[0].plot()

    # Display live output
    cv2.imshow("VisionDesk", annotated_frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
