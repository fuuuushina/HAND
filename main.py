import mediapipe as mp
import time
import cv2
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.metadata.metadata_writers import model_asset_bundle_utils
from mediapipe.tasks.python.vision import HandLandmarkerOptions
"""
option = HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode = vision.RunningMode.IMAGE,
    num_hands = 1,
)

with vision.HandLandmarker.create_from_options(option) as Landmarker:
    image = mp.Image.create_from_file("main2.jpg")
    result = Landmarker.detect(image)

    for i, handed in enumerate(result.handedness):
        label=handed[0].category_name
        score=handed[0].score
        wrist=result.hand_world_landmarks[i][0]
        print(f"Main {i} : {label} : ({score:.2f}) | poignet(m) :"
              f" {wrist.x:.3f}, {wrist.y:.3f}, {wrist.z:.3f}")



""""------------------------------------------------------------------  """

IMAGE_PATH = "main2.jpg"

# Connexions entre les points (les "os" de la main)
BONES = [(0, 1), (1, 2), (2, 3), (3, 4),          # pouce
         (0, 5), (5, 6), (6, 7), (7, 8),          # index
         (5, 9), (9, 10), (10, 11), (11, 12),     # majeur
         (9, 13), (13, 14), (14, 15), (15, 16),   # annulaire
         (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]  # auriculaire

options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.2,
    min_hand_presence_confidence=0.2
)

# 1. Lire la photo avec OpenCV (format BGR) pour pouvoir dessiner dessus
frame = cv2.imread(IMAGE_PATH)
if frame is None:
    raise FileNotFoundError(f"Impossible de lire {IMAGE_PATH}")
h, w = frame.shape[:2]

# 2. Convertir en RGB pour MediaPipe
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

with vision.HandLandmarker.create_from_options(options) as landmarker:
    result = landmarker.detect(mp_image)

# 3. Dessiner chaque main détectée
for i, handed in enumerate(result.handedness):
    """
    result.handedness = [ [Category(category_name="Right", score=0.97)],
                          [Category(category_name="Left",  score=0.91)] ]
    result.hand_landmarks = [ [21 points de la main 0],
                          [21 points de la main 1] ]   
    """
    label = handed[0].category_name
    landmarks = result.hand_landmarks[i]

    # Conversion coordonnées normalisées (0..1) -> pixels
    pts = [(int(p.x * w), int(p.y * h)) for p in landmarks]

    # Couleur selon la main : vert = droite, bleu = gauche (en BGR)
    color = (0, 200, 0) if label == "Right" else (220, 120, 0)

    for a, b in BONES:
        cv2.line(frame, pts[a], pts[b], color, 3)

    for idx, (x, y) in enumerate(pts):
        cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(frame, str(idx), (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # Étiquette près du poignet
    wx, wy = pts[0]
    cv2.putText(frame, f"{label} {handed[0].score:.2f}", (wx - 40, wy + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

# 4. Afficher (fenêtre redimensionnable pour les grandes photos)
cv2.namedWindow("Main", cv2.WINDOW_NORMAL)
cv2.imshow("Main", frame)
cv2.waitKey(0)            # attend une touche
cv2.destroyAllWindows()

cv2.imwrite("main_annotee.jpg", frame)  # sauvegarde le résultat