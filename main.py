
import cv2
from ultralytics import YOLO
import time

# Pretrained object detection model
model = YOLO("yolo26n.pt")

# Open default camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Camera could not be opened")

prev_time = time.perf_counter()

while True:
    ret, frame = cap.read()
    # ret -> true, false
    # frame -> image numPy array

    if not ret:
        break

    # object detection
    results = model(frame, conf=0.6, verbose=True) 
    # verbose=True to print detection results
    # default conf = 0.25

    # read detected objects
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        # draw bounding box
        cv2.rectangle( frame,
                       (x1, y1), (x2, y2),
                       color=(155, 255, 185), 
                       thickness=2)

        label = f"{class_name}: {confidence:.2f}"

        cv2.putText(frame, label,
                    (x1, y1 - 10),  # position above the bounding box
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,            # font scale
                    (179, 53, 252), 
                    1)

        curr_time = time.perf_counter()
        elapsed_time = curr_time - prev_time
        fps = 1 / elapsed_time if elapsed_time > 0 else 0
        prev_time = curr_time

        cv2.putText(frame, 
                    f"FPS: {fps:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, 
                    (255, 250, 45), 
                    1)



    # Display live output
    cv2.imshow("VisionDesk", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release() 
cv2.destroyAllWindows()
