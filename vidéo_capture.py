import os



os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Connexions entre les points (les "os" de la main)
BONES = [(0, 1), (1, 2), (2, 3), (3, 4),          # pouce
         (0, 5), (5, 6), (6, 7), (7, 8),          # index
         (5, 9), (9, 10), (10, 11), (11, 12),     # majeur
         (9, 13), (13, 14), (14, 15), (15, 16),   # annulaire
         (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]  # auriculaire

ARTICULATION = [(2,3,4),                     #pouce
                (5,6,7), (6,7,8),            #index
                (9,10,11), (10,11,12),       #majeur
                (13,14,15), (14,15,16),      #annulaire
                (17,18,19), (18,19,20),]     #auriculaire

def draw_hands(frame, result):
    """Dessine toutes les mains détectées sur frame (modifie l'image en place)."""
    h, w = frame.shape[:2]
    for i, handed in enumerate(result.handedness):
        label = "Left" if handed[0].category_name == "Right" else "Right"
        landmarks = result.hand_landmarks[i]

        pts = [(int(p.x * w), int(p.y * h)) for p in landmarks]
        color = (0, 200, 0) if label == "Right" else (220, 120, 0)

        for a, b in BONES:
            cv2.line(frame, pts[a], pts[b], color, 3)

        for idx, (x, y) in enumerate(pts):
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(frame, str(idx), (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        wx, wy = pts[0]
        cv2.putText(frame, f"{label} {handed[0].score:.2f}", (wx - 40, wy + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        for a, b, c in ARTICULATION:
            theta = angle(pts[a], pts[b], pts[c])
            cv2.putText(frame, f"{theta:.0f}", (pts[b][0] - 30, pts[b][1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)


options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,   # mode vidéo : suit la main d'une image à l'autre
    num_hands=4,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)



import numpy as np

def angle(a, b, c):
    """Angle en degrés au point b, formé par les segments b→a et b→c."""
    a, b, c = np.array(a, dtype=float), np.array(b, dtype=float), np.array(c, dtype=float)
    ba = a - b
    bc = c - b
    norms = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norms == 0:          # deux points confondus : angle indéfini
        return 0.0
    cos_theta = np.dot(ba, bc) / norms
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


# 0 = webcam par défaut. CAP_DSHOW accélère l'ouverture sous Windows.
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    raise RuntimeError("Impossible d'ouvrir la webcam")

start = time.perf_counter()
prev = start

with vision.HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Effet miroir : plus naturel, et les labels Left/Right deviennent justes
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # En mode VIDEO, chaque image a besoin d'un timestamp en ms, strictement croissant
        timestamp_ms = int((time.perf_counter() - start) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        draw_hands(frame, result)

        # Compteur d'images par seconde
        now = time.perf_counter()
        fps = 1.0 / (now - prev) if now > prev else 0.0
        prev = now
        cv2.putText(frame, f"{fps:.0f} FPS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Main - temps reel (q pour quitter)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()



