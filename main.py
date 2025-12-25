import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import time
from collections import deque, Counter

import pyautogui

from core.config import (
    WIDTH, HEIGHT, SMOOTHING,
    CONFIDENCE_THRESHOLD, MODEL_FILE,
    CALIBRATION_FRAMES, GESTURES,
    SWIPE_HISTORY_LENGTH
)
from core.features import extract_features
from core.model import HybridModel
from core.actions import execute_action, screen_w, screen_h


# ================= INITIALIZATION =================
cap = cv2.VideoCapture(0)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

model = HybridModel()

# ================= STATE =================
X_data, y_data = [], []
plocX, plocY = 0, 0

pred_buffer = deque(maxlen=5)
swipe_history = deque(maxlen=SWIPE_HISTORY_LENGTH)

last_times = {"click": 0, "swipe": 0}
prev_action_name = "IDLE"

frame_count = 0
calibration_active = True
box_coords = [
    int(WIDTH * 0.45), int(WIDTH * 0.55),
    int(HEIGHT * 0.45), int(HEIGHT * 0.55)
]

acc_correct = 0
acc_total = 0


# ================= LOAD SAVED DATA =================
if os.path.exists(MODEL_FILE):
    try:
        with open(MODEL_FILE, "rb") as f:
            data = pickle.load(f)
            X_data = data.get("X", [])
            y_data = data.get("y", [])
            model.train_knn(X_data, y_data)
            model.train_rf(X_data, y_data)
        print(f">> Loaded {len(X_data)} samples")
    except Exception as e:
        print(">> Failed to load model:", e)


# ================= MAIN LOOP =================
while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    frame_count += 1

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    key = cv2.waitKey(1) & 0xFF

    status_text = "NO HAND"
    brain_used = "NONE"
    confidence = 0.0
    color = (0, 0, 255)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            lm = hand_lms.landmark
            feats = extract_features(lm)
            if feats is None:
                continue

            x1 = int(lm[8].x * WIDTH)
            y1 = int(lm[8].y * HEIGHT)

            # -------- CALIBRATION --------
            if frame_count <= CALIBRATION_FRAMES:
                box_coords[0] = min(box_coords[0], x1)
                box_coords[1] = max(box_coords[1], x1)
                box_coords[2] = min(box_coords[2], y1)
                box_coords[3] = max(box_coords[3], y1)

            # -------- PREDICTION --------
            pred, confidence, brain_used = model.predict(
                feats, X_data, y_data, pred_buffer
            )

            if confidence > CONFIDENCE_THRESHOLD:
                pred_buffer.append(pred)
                stable_pred = Counter(pred_buffer).most_common(1)[0][0]

                status_text = GESTURES.get(stable_pred, "IDLE")
                color = (0, 255, 0)

                result = execute_action(
                    stable_pred,
                    (x1, y1),
                    prev_action_name,
                    box_coords,
                    swipe_history,
                    last_times
                )

                if result and result["type"] == "move":
                    nx, ny = result["coords"]
                    clocX = plocX + (nx - plocX) / SMOOTHING
                    clocY = plocY + (ny - plocY) / SMOOTHING
                    pyautogui.moveTo(clocX, clocY)
                    plocX, plocY = clocX, clocY

                prev_action_name = status_text

            # -------- TRAINING (KEYS 1–4) --------
            if key in [ord("1"), ord("2"), ord("3"), ord("4")]:
                label = int(chr(key))

                if model.knn_trained and confidence > CONFIDENCE_THRESHOLD:
                    acc_total += 1
                    if stable_pred == label:
                        acc_correct += 1

                X_data.append(feats)
                y_data.append(label)

                model.train_knn(X_data, y_data)
                if len(X_data) % 20 == 0:
                    model.train_rf(X_data, y_data)

            mp_draw.draw_landmarks(
                img, hand_lms, mp_hands.HAND_CONNECTIONS
            )

    else:
        pred_buffer.clear()
        swipe_history.clear()
        prev_action_name = "IDLE"

    # -------- SAVE / RESET --------
    if key == ord("s"):
        with open(MODEL_FILE, "wb") as f:
            pickle.dump({"X": X_data, "y": y_data}, f)
        print(">> Model Saved")

    if key == ord("r"):
        frame_count = 0
        box_coords = [
            int(WIDTH * 0.45), int(WIDTH * 0.55),
            int(HEIGHT * 0.45), int(HEIGHT * 0.55)
        ]
        calibration_active = True
        print(">> Calibration Reset")

    # -------- UI --------
    cv2.putText(img, status_text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    cv2.putText(img, f"{brain_used} | {confidence:.2f}",
                (20, 90), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200, 200, 200), 1)

    if acc_total > 0:
        acc = (acc_correct / acc_total) * 100
        cv2.putText(
            img,
            f"Accuracy: {acc:.1f}%",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0) if acc > 80 else (0, 165, 255),
            2
        )

    cv2.imshow("AI Mouse", img)
    if key == 27:
        break


# ================= CLEANUP =================
cap.release()
hands.close()
cv2.destroyAllWindows()
