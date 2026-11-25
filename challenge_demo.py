"""
Webcam demo of the random-challenge liveness engine.
Keys:  r = restart with new random challenges   q = quit
Live values (EAR / MAR / smile / turn) are shown so you can tune thresholds
in challenges.py if a challenge is too easy or too hard on your face/camera.
"""
import cv2
import numpy as np
import mediapipe as mp

from challenges import ChallengeEngine, metrics, PROMPTS

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open the webcam.")

face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

engine = ChallengeEngine(n_challenges=3, timeout=8.0)
GREEN, RED, YELLOW, WHITE = (0, 200, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255)


def put(img, text, y, color=WHITE, scale=0.8, thick=2):
    cv2.putText(img, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2)
    cv2.putText(img, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)                       # mirror: what the user sees
    h, w = frame.shape[:2]
    res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    pts = None
    if res.multi_face_landmarks:
        lms = res.multi_face_landmarks[0].landmark
        pts = np.array([[p.x * w, p.y * h] for p in lms])

    st = engine.update(pts)

    if st["state"] == "calibrating":
        put(frame, st["message"], 40, YELLOW)
    elif st["state"] == "running":
        put(frame, f"Challenge {st['index'] + 1}/{st['total']}: {st['prompt']}", 40, YELLOW, 0.9)
        if st["time_left"] is not None:
            put(frame, f"Time left: {st['time_left']:.1f}s", 75, WHITE)
        if st["message"]:
            put(frame, st["message"], 110, WHITE, 0.7)
    elif st["state"] == "passed":
        put(frame, "LIVENESS PASSED  (press r to retry)", 40, GREEN, 0.9)
    else:
        put(frame, "LIVENESS FAILED  (press r to retry)", 40, RED, 0.9)

    for i, r in enumerate(st["results"]):
        put(frame, f"{'OK ' if r['passed'] else 'X  '}{PROMPTS[r['name']]} ({r['seconds']}s)",
            h - 20 - 28 * (len(st["results"]) - 1 - i), GREEN if r["passed"] else RED, 0.6, 1)

    if pts is not None:
        m = metrics(pts)
        put(frame, f"EAR {m['ear']:.2f}  MAR {m['mar']:.2f}  smile {m['smile']:.2f}  turn {m['turn']:.2f}",
            h - 5, WHITE, 0.5, 1)

    cv2.imshow("Liveness challenge demo (q = quit, r = restart)", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("r"):
        engine = ChallengeEngine(n_challenges=3, timeout=8.0)

cap.release()
cv2.destroyAllWindows()
