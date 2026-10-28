import math
import os
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import cv2
# Keep Ultralytics settings inside the writable project folder.
YOLO_CONFIG_DIR = Path(".ultralytics").resolve()
YOLO_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))

import mediapipe as mp
from ultralytics import YOLO

# Models and camera
model = YOLO("yolo26n.pt")
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera could not be opened")

# Keep the last 7 raw gesture predictions; 5 matching votes make a gesture stable.
gesture_history = deque(maxlen=7)
STABLE_VOTES = 5

# Application states
object_detection_paused = False
show_detections = True
previous_stable_gesture = "Unknown"
prev_time = time.perf_counter()
screenshot_dir = Path("C:\\Users\\ackrb\\Desktop\\python\\Learn-ComputerVision\\Projects\\VisionDesk\\screenshots")
screenshot_dir.mkdir(exist_ok=True)


def get_finger_states(hand_landmarks, frame_shape):
    landmarks = hand_landmarks.landmark if hasattr(hand_landmarks, "landmark") else hand_landmarks
    height, width = frame_shape[:2]

    # Convert normalized landmarks to pixels before measuring angles, because
    # MediaPipe normalizes x by image width and y by image height separately.
    def point(index):
        landmark = landmarks[index]
        return landmark.x * width, landmark.y * height

    def joint_angle(first, joint, last):
        ax, ay = point(first)
        bx, by = point(joint)
        cx, cy = point(last)
        vector_a = (ax - bx, ay - by)
        vector_b = (cx - bx, cy - by)
        length_a = math.hypot(*vector_a)
        length_b = math.hypot(*vector_b)
        if not length_a or not length_b:
            return 0.0
        cosine = sum(a * b for a, b in zip(vector_a, vector_b)) / (length_a * length_b)
        return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

    # MCP (örn 5)-> PIP(örn 6) -> fingertip (örn 8): straight fingers are near 180 degrees.
    finger_chains = [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]
    finger_angles = [joint_angle(*chain) for chain in finger_chains]
    fingers_open = [angle >= 150 for angle in finger_angles]
    fingers_closed = [angle <= 130 for angle in finger_angles]

    # The thumb must be straight at its IP joint AND extended away from the palm.
    thumb_angle = joint_angle(2, 3, 4)
    thumb_tip = point(4)
    index_base = point(5)
    wrist = point(0)
    middle_base = point(9)
    thumb_distance = math.hypot(thumb_tip[0] - index_base[0], thumb_tip[1] - index_base[1])
    hand_size = math.hypot(wrist[0] - middle_base[0], wrist[1] - middle_base[1])
    normalized_thumb_distance = thumb_distance / hand_size if hand_size else 0
    thumb_open = thumb_angle >= 150 and normalized_thumb_distance >= 0.35

    thumb_vertical_delta = (
        (thumb_tip[1] - point(2)[1]) / hand_size if hand_size else 0
    )
    return thumb_open, thumb_vertical_delta, fingers_open, fingers_closed


def classify_gesture(hand_landmarks, frame_shape):
    thumb_open, thumb_vertical_delta, fingers_open, fingers_closed = get_finger_states(
        hand_landmarks, frame_shape
    )

    if thumb_open and all(fingers_open):
        return "Open Palm"
    if not thumb_open and all(fingers_closed):
        return "Fist"
    if thumb_open and all(fingers_closed):
        if thumb_vertical_delta < -0.15:
            return "Thumbs Up"
        if thumb_vertical_delta > 0.15:
            return "Thumbs Down"
    return "Unknown"


def stable_gesture(history):
    """Return a gesture only when it has at least STABLE_VOTES in the window."""
    if not history:
        return "Unknown"
    counts = Counter(history)
    gesture, votes = counts.most_common(1)[0]
    return gesture if gesture != "Unknown" and votes >= STABLE_VOTES else "Unknown"


def draw_hand_landmarks(frame, landmarks):
    mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read a frame from the camera")
            break

        # Mirror both the camera display and the hand coordinates users see.
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(rgb_frame)

        current_gesture = "Unknown"
        if hand_results.multi_hand_landmarks:
            hand_landmarks = hand_results.multi_hand_landmarks[0]
            draw_hand_landmarks(frame, hand_landmarks)
            current_gesture = classify_gesture(hand_landmarks, frame.shape)
        # Add Unknown when no hand is visible, so an old command cannot remain active.
        gesture_history.append(current_gesture)
        gesture = stable_gesture(gesture_history)

        # Commands fire on entry into a stable gesture, once per gesture hold.
        if gesture != previous_stable_gesture:
            if gesture == "Thumbs Up":
                date = datetime.now().strftime("%Y%m%d")
                number = 1
                screenshot_path = screenshot_dir / f"visiondesk_{date}_{number:02d}.png"
                while screenshot_path.exists():
                    number += 1
                    screenshot_path = screenshot_dir / f"visiondesk_{date}_{number:02d}.png"
                cv2.imwrite(str(screenshot_path), frame)
                print(f"Screenshot saved: {screenshot_path}")
            elif gesture == "Open Palm":
                object_detection_paused = True
            elif previous_stable_gesture == "Open Palm":
                object_detection_paused = False
            elif gesture == "Fist":
                show_detections = not show_detections
            elif gesture == "Thumbs Down":
                print("Thumbs Down - break.")
                break
            previous_stable_gesture = gesture

        class_counts = Counter()
        total_objects = 0
        inference_ms = 0.0

        if not object_detection_paused:
            result = model(frame, conf=0.6, verbose=False)[0]
            inference_ms = result.speed["inference"]
            total_objects = len(result.boxes)

            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                class_counts[class_name] += 1

                if show_detections:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (155, 255, 185), 2)
                    cv2.putText(
                        frame,
                        f"{class_name}: {confidence:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (179, 53, 252),
                        1,
                    )

        curr_time = time.perf_counter()
        elapsed_time = curr_time - prev_time
        fps = 1 / elapsed_time if elapsed_time > 0 else 0
        prev_time = curr_time

        panel_lines = [
            "VisionDesk",
            f"FPS: {fps:.1f}",
            f"Inference time: {inference_ms:.1f} ms",
            f"Objects in frame: {total_objects}",
            f"Detection: {'Paused' if object_detection_paused else 'Running'}",
            f"Boxes and labels: {'Visible' if show_detections else 'Hidden'}",
            f"Gesture: {gesture}",
            "",
            "Detection Statistics:",
        ]
        if object_detection_paused:
            panel_lines.append("Paused by Open Palm")
        elif class_counts:
            panel_lines.extend(
                f"{name}: {count}" for name, count in sorted(class_counts.items())
            )
        else:
            panel_lines.append("No objects detected")

        image_width = frame.shape[1]
        panel_width = 340
        line_height = 28
        required_height = 30 + len(panel_lines) * line_height
        bottom_padding = max(0, required_height - frame.shape[0])
        display_frame = cv2.copyMakeBorder(
            frame, 0, bottom_padding, 0, panel_width,
            cv2.BORDER_CONSTANT, value=(25, 25, 25)
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

        cv2.imshow("VisionDesk", display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    hands.close()
    cv2.destroyAllWindows()
