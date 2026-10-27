from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

SEQUENCE_LENGTH = 30
NUM_HANDS = 2
LANDMARKS_PER_HAND = 21
FEATURES = NUM_HANDS * LANDMARKS_PER_HAND * 3


def extract_landmarks(results, previous: np.ndarray | None = None) -> np.ndarray:
    """Return fixed left/right hand coordinates, or hold the last frame if hands are absent."""
    frame = np.zeros((NUM_HANDS, LANDMARKS_PER_HAND, 3), dtype=np.float32)
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            side = handedness.classification[0].label
            # MediaPipe labels are from the image's mirrored-camera convention.
            index = 0 if side == "Left" else 1
            frame[index] = np.asarray([[p.x, p.y, p.z] for p in hand.landmark], dtype=np.float32)
    if not np.any(frame) and previous is not None:
        return previous.copy()
    return frame.reshape(FEATURES)


def capture_sequence(hands, camera, draw: bool = True) -> np.ndarray | None:
    """Capture a rolling sequence; press SPACE to accept, Q to cancel."""
    sequence: list[np.ndarray] = []
    previous = None
    while True:
        ok, image = camera.read()
        if not ok:
            raise RuntimeError("อ่านภาพจากกล้องไม่ได้ ตรวจสอบ camera index หรือสิทธิ์กล้อง")
        image = cv2.flip(image, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        current = extract_landmarks(results, previous)
        previous = current
        sequence.append(current)
        sequence = sequence[-SEQUENCE_LENGTH:]

        if draw and results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    image, hand, mp.solutions.hands.HAND_CONNECTIONS
                )
        ready = len(sequence) == SEQUENCE_LENGTH
        cv2.putText(image, f"Frames {len(sequence)}/{SEQUENCE_LENGTH} | SPACE save | Q cancel",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (40, 230, 40), 2)
        cv2.imshow("Basic Sign Language Translator", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return None
        if key == 32 and ready:
            return np.asarray(sequence, dtype=np.float32)


def open_hands():
    return mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=NUM_HANDS,
        model_complexity=0,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.5,
    )
