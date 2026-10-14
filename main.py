
import cv2
from ultralytics import YOLO
import time
from collections import Counter

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
    
    result = results[0]  
    inference_ms = result.speed["inference"]

    class_counts = Counter()
    total_objects = len(result.boxes)

    # read detected objects
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        class_counts[class_name] += 1

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

    # application fps: calculate one time per frame
    curr_time = time.perf_counter()
    elapsed_time = curr_time - prev_time
    fps = 1 / elapsed_time if elapsed_time > 0 else 0
    prev_time = curr_time

    panel_lines = [
        "VisionDesk",
        f"FPS: {fps:.1f}",
        f"Inference time: {inference_ms:.1f} ms",
        f"Objects in frame: {total_objects}",
        "",
        "Detection Statistics:"
    ]

    if class_counts: 
        for class_name, count in sorted(class_counts.items()):
            panel_lines.append(f"{class_name}: {count}")
    else:
        panel_lines.append("No objects detected")

    image_width = frame.shape[1]
    panel_width = 340
    line_height = 28
    required_height = 30 + len(panel_lines) * line_height
    bottom_padding = max(0, required_height - frame.shape[0])

    display_frame = cv2.copyMakeBorder(
        frame,
        0,
        bottom_padding,
        0,
        panel_width,
        cv2.BORDER_CONSTANT,
        value=(25, 25, 25),
    )

    for index, text in enumerate(panel_lines):
        cv2.putText(
            display_frame,
            text,
            (image_width + 15, 30 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

    # Display live output
    cv2.imshow("VisionDesk", display_frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release() 
cv2.destroyAllWindows()
