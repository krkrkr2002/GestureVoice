import copy
import itertools
import string
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pyttsx3
from tensorflow import keras


MODEL_PATH = Path(__file__).with_name("model.h5")

# Define alphabet classes (1-9 and A-Z) - must match training
ALPHABET = ['1', '2', '3', '4', '5', '6', '7', '8', '9'] + list(string.ascii_uppercase)


def calc_landmark_list(image: np.ndarray, landmarks) -> list:
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list: list) -> list:
    temp_landmark_list = copy.deepcopy(landmark_list)

    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    max_value = max(list(map(abs, temp_landmark_list))) if temp_landmark_list else 0

    def normalize_(n):
        return n / max_value if max_value != 0 else 0

    return list(map(normalize_, temp_landmark_list))


def speak(engine: pyttsx3.Engine, text: str):
    engine.stop()
    engine.say(text)
    engine.runAndWait()


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    print(f"Loading model from: {MODEL_PATH}")
    model = keras.models.load_model(str(MODEL_PATH))
    print("Model loaded.")

    tts = pyttsx3.init()
    tts.setProperty("rate", 165)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (VideoCapture(0)).")

    conf_threshold = 0.75
    stable_frames_required = 6
    cooldown_s = 1.2

    last_label = None
    stable_label = None
    stable_count = 0
    last_spoken_at = 0.0

    print("Camera started. Press 'q' to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            label = "No hand"
            confidence = 0.0

            if results and results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                for lm in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame,
                        lm,
                        mp_hands.HAND_CONNECTIONS,
                    )

                landmark_list = calc_landmark_list(frame, hand_landmarks)
                pre_processed = pre_process_landmark(landmark_list)
                df = pd.DataFrame([pre_processed])

                preds = model.predict(df, verbose=0)
                idx = int(np.argmax(preds, axis=1)[0])
                confidence = float(preds[0][idx])
                label = ALPHABET[idx] if 0 <= idx < len(ALPHABET) else "Unknown"

            # stability logic: only speak when same label holds for N frames
            if label == stable_label:
                stable_count += 1
            else:
                stable_label = label
                stable_count = 1

            now = time.time()
            should_speak = (
                label not in ("No hand", "Unknown")
                and confidence >= conf_threshold
                and stable_count >= stable_frames_required
                and label != last_label
                and (now - last_spoken_at) >= cooldown_s
            )

            if should_speak:
                speak(tts, label)
                last_label = label
                last_spoken_at = now

            overlay = f"Pred: {label}  Conf: {confidence:.2f}  Stable: {stable_count}/{stable_frames_required}"
            cv2.rectangle(frame, (10, 10), (620, 50), (0, 0, 0), -1)
            cv2.putText(frame, overlay, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("GestureVoice - Live", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == "__main__":
    main()

