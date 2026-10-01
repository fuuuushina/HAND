import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ============================================================
# CONFIG
# ============================================================
DRY_RUN = True            # True = pas de main branchée, on print seulement. False = envoie aux servos.
SERIAL_PORT = "COM3"
MAIN_SUIVIE = "Right"     # la main humaine qui pilote le robot
INVERSER_LABEL = True     # ton code d'origine inversait Left/Right ; mets False si le label affiché est faux

# Main robot (tes valeurs)
IDS = [1, 11, 21, 31, 41, 51, 61, 71]   # index1, index2, majeur1, majeur2, annulaire1, annulaire2, pouce1, pouce2
MiddlePos = [-0.3, 0.3, -0.3, 0.0, -0.6, -0.3, 0.3, -0.9]
MaxSpeed = 7

# Consigne robot pour Angle_1 (Angle_2 = -Angle_1), comme dans OpenHand / CloseHand
OUVERT = 10
FERME = -100

# Doigt robot -> (servo 1, servo 2, articulations humaines (a,b,c), flexion humaine max en degrés)
# La flexion = somme des (180 - angle) aux articulations MCP + PIP.
# Ajuste flexion max si le robot ne ferme jamais complètement (baisse) ou ferme trop tôt (monte).
DOIGTS = {
    "Index":     (0, 1, [(0, 5, 6),   (5, 6, 7)],    160),
    "Majeur":    (2, 3, [(0, 9, 10),  (9, 10, 11)],  160),
    "Annulaire": (4, 5, [(0, 13, 14), (13, 14, 15)], 160),
    "Pouce":     (6, 7, [(1, 2, 3),   (2, 3, 4)],    100),
}
BOUTS = {"Index": 8, "Majeur": 12, "Annulaire": 16, "Pouce": 4}   # pour placer le texte à l'écran

ALPHA = 0.3            # lissage : 0 = figé, 1 = aucun lissage
ENVOI_HZ = 20          # fréquence d'envoi des consignes au robot
PRINT_PERIODE = 0.2    # secondes entre deux affichages console

BONES = [(0, 1), (1, 2), (2, 3), (3, 4),
         (0, 5), (5, 6), (6, 7), (7, 8),
         (5, 9), (9, 10), (10, 11), (11, 12),
         (9, 13), (13, 14), (14, 15), (15, 16),
         (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


# ============================================================
# MAIN ROBOT
# ============================================================
class MainRobot:
    def __init__(self, dry_run):
        self.dry_run = dry_run
        self.c = None
        if not dry_run:
            from rustypot import Scs0009PyController
            self.c = Scs0009PyController(serial_port=SERIAL_PORT, baudrate=115200, timeout=0.5)
            for id_ in IDS:
                self.c.write_torque_enable(id_, 1)   # 1 = On

    def bouger_doigt(self, s1, s2, angle_1, angle_2, speed):
        if self.dry_run:
            return
        c = self.c
        c.write_goal_speed(IDS[s1], speed)
        time.sleep(0.0002)
        c.write_goal_speed(IDS[s2], speed)
        time.sleep(0.0002)
        c.write_goal_position(IDS[s1], np.deg2rad(MiddlePos[s1] + angle_1))
        c.write_goal_position(IDS[s2], np.deg2rad(MiddlePos[s2] + angle_2))

    def arreter(self):
        if self.dry_run:
            return
        for _, (s1, s2, _, _) in DOIGTS.items():
            self.bouger_doigt(s1, s2, OUVERT, -OUVERT, 3)   # on rouvre la main
        time.sleep(1.0)
        for id_ in IDS:
            self.c.write_torque_enable(id_, 2)   # 2 = Off


# ============================================================
# CALCULS
# ============================================================
def angle(a, b, c):
    """Angle en degrés au point b, formé par b→a et b→c (marche en 2D et en 3D)."""
    a, b, c = np.array(a, dtype=float), np.array(b, dtype=float), np.array(c, dtype=float)
    ba, bc = a - b, c - b
    norms = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norms == 0:
        return 180.0
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / norms, -1.0, 1.0))))


def flexion_doigt(pts3d, articulations):
    return sum(180.0 - angle(pts3d[a], pts3d[b], pts3d[c]) for a, b, c in articulations)


def consigne(t):
    """t = 0 (ouvert) … 1 (fermé) -> (Angle_1, Angle_2) pour le robot."""
    a1 = OUVERT + t * (FERME - OUVERT)
    return a1, -a1


def trouver_main(result):
    for i, handed in enumerate(result.handedness):
        label = handed[0].category_name
        if INVERSER_LABEL:
            label = "Left" if label == "Right" else "Right"
        if label == MAIN_SUIVIE:
            return i
    return None


def barre(t, n=10):
    k = int(round(t * n))
    return "#" * k + "-" * (n - k)


def afficher_console(etat):
    print(f"--- {time.strftime('%H:%M:%S')} ---")
    for nom, (s1, s2, _, _) in DOIGTS.items():
        t = etat[nom]
        a1, a2 = consigne(t)
        print(f"{nom:<10}[{barre(t)}] {t * 100:3.0f}%   "
              f"ID{IDS[s1]:<3}{a1:+7.1f}°   ID{IDS[s2]:<3}{a2:+7.1f}°")


def dessiner(frame, landmarks, etat, fps):
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in landmarks]
    for a, b in BONES:
        cv2.line(frame, pts[a], pts[b], (0, 200, 0), 3)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
    for nom, tip in BOUTS.items():
        x, y = pts[tip]
        cv2.putText(frame, f"{nom[:3]} {etat[nom] * 100:.0f}%", (x - 30, y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)


# ============================================================
# BOUCLE PRINCIPALE
# ============================================================
def main():
    robot = MainRobot(DRY_RUN)
    print("Mode SIMULATION (print seulement)" if DRY_RUN else f"Mode ROBOT sur {SERIAL_PORT}")

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la webcam")

    etat = {nom: 0.0 for nom in DOIGTS}   # fermeture lissée de chaque doigt (0..1)
    start = prev = time.perf_counter()
    dernier_envoi = dernier_print = 0.0
    main_vue = False

    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_image, int((time.perf_counter() - start) * 1000))

                now = time.perf_counter()
                fps = 1.0 / (now - prev) if now > prev else 0.0
                prev = now

                idx = trouver_main(result)
                if idx is not None:
                    if not main_vue:
                        print(">>> Main détectée")
                        main_vue = True

                    # Angles en 3D (coordonnées monde, en mètres) -> vrais angles articulaires
                    pts3d = [(p.x, p.y, p.z) for p in result.hand_world_landmarks[idx]]
                    for nom, (_, _, arts, fmax) in DOIGTS.items():
                        t = float(np.clip(flexion_doigt(pts3d, arts) / fmax, 0.0, 1.0))
                        etat[nom] = ALPHA * t + (1 - ALPHA) * etat[nom]

                    # Envoi au robot (limité en fréquence)
                    if now - dernier_envoi >= 1.0 / ENVOI_HZ:
                        for nom, (s1, s2, _, _) in DOIGTS.items():
                            a1, a2 = consigne(etat[nom])
                            robot.bouger_doigt(s1, s2, a1, a2, MaxSpeed)
                        dernier_envoi = now

                    # Affichage console (limité aussi, sinon ça défile trop vite)
                    if now - dernier_print >= PRINT_PERIODE:
                        afficher_console(etat)
                        dernier_print = now

                    dessiner(frame, result.hand_landmarks[idx], etat, fps)
                elif main_vue:
                    print(">>> Main perdue (le robot garde sa dernière position)")
                    main_vue = False

                cv2.putText(frame, f"{fps:.0f} FPS", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("Main -> robot (q pour quitter)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        robot.arreter()


if __name__ == "__main__":
    main()