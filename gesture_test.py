import math
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def get_finger_states(hand_landmarks):
    landmarks = hand_landmarks.landmark

    finger_pairs = [
        (8, 6),
        (12, 10),
        (16, 14),
        (20, 18),
    ]

    non_thumb_fingers_open = [
        landmarks[tip_id].y < landmarks[pip_id].y
        for tip_id, pip_id in finger_pairs
    ]

    thumb_tip = landmarks[4]
    index_base = landmarks[5]
    thumb_distance = math.hypot(  # √((x₂ - x₁)² + (y₂ - y₁)²)
        thumb_tip.x - index_base.x,
        thumb_tip.y - index_base.y,
    )

    wrist = landmarks[0]
    middle_base = landmarks[9]
    hand_size = math.hypot(
        wrist.x - middle_base.x,
        wrist.y - middle_base.y,
    )

    normalized_thumb_distance = thumb_distance / hand_size if hand_size else 0
    thumb_open = normalized_thumb_distance > 0.45

    return thumb_open, non_thumb_fingers_open


def classify_gesture(hand_landmarks):
    thumb_open, non_thumb_fingers_open = get_finger_states(hand_landmarks)

    if thumb_open and all(non_thumb_fingers_open):
        return "Open Palm"

    if not thumb_open and not any(non_thumb_fingers_open):
        return "Fist"

    if thumb_open and not any(non_thumb_fingers_open):
        return "Thumbs Up"

    return ""


hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    hands.close()
    raise RuntimeError("Camera could not be opened")

try:
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Could not read a frame from the camera")
            break

        # Mirror webcam image
        frame = cv2.flip(frame, 1)

        # OpenCV uses BGR, MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            handedness_results = results.multi_handedness or []

            for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,     # landmark dots
                    mp_hands.HAND_CONNECTIONS  # connection between dots
                )

                gesture = classify_gesture(hand_landmarks)
                hand_label = ""

                if index < len(handedness_results):
                    hand_label = handedness_results[index].classification[0].label


                thumb_open, non_thumb_fingers_open = get_finger_states(hand_landmarks)
                open_finger_count = sum(non_thumb_fingers_open) + int(thumb_open)

                cv2.putText(
                    frame,
                    f"{hand_label}: {open_finger_count} fingers",
                    (10, 30 + index * 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (179, 53, 252),
                    2,
                )
                
                text = f"{hand_label}: {gesture}"
                font_scale = 0.7
                thickness = 2
                font = cv2.FONT_HERSHEY_SIMPLEX

                (text_width, _), _ = cv2.getTextSize(text, font, font_scale, thickness)

                frame_height, frame_width = frame.shape[:2]
                text_y = frame_height - 20

                if hand_label == "Left":
                    text_x = 10
                else:
                    text_x = frame_width - text_width - 10

                cv2.putText(
                    frame,
                    text,
                    (text_x, text_y),
                    font,
                    font_scale,
                    (179, 53, 252),
                    thickness,
                )

        cv2.imshow("Gesture Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    hands.close()
    cv2.destroyAllWindows()
