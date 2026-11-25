"""Webcam + MediaPipe Face Mesh test. Press q to quit."""
import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   # Windows webcam
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open the webcam. Close other apps using it (Zoom, Teams, Camera) and try again.")

face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

while True:
    ok, frame = cap.read()
    if not ok:
        print("Could not read a frame from the webcam.")
        break
    frame = cv2.flip(frame, 1)                       # mirror view
    h, w = frame.shape[:2]
    result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if result.multi_face_landmarks:
        for lm in result.multi_face_landmarks[0].landmark:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 1, (0, 255, 0), -1)
        text = "Face found"
    else:
        text = "No face found"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.imshow("Webcam test (press q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
