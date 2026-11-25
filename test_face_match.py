"""
Test identity matching.

  python test_face_match.py id_photo.jpg               -> captures your live face from the webcam
  python test_face_match.py id_photo.jpg selfie.jpg    -> compares two image files

Webcam mode: press SPACE to capture (it takes 5 frames), q to cancel.
"""
import sys
import cv2
from face_match import FaceMatcher

if len(sys.argv) < 2:
    raise SystemExit(__doc__)

id_img = cv2.imread(sys.argv[1])
if id_img is None:
    raise SystemExit(f"Could not read image: {sys.argv[1]}")

matcher = FaceMatcher()

if len(sys.argv) >= 3:
    live = cv2.imread(sys.argv[2])
    if live is None:
        raise SystemExit(f"Could not read image: {sys.argv[2]}")
    frames = [live]
else:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            raise SystemExit("Webcam read failed")
        show = f.copy()
        cv2.putText(show, "SPACE = capture, q = cancel", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Face match test", show)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            raise SystemExit("Cancelled")
        if k == 32:
            frames.append(f)
            for _ in range(4):
                for _ in range(5):
                    cap.read()
                frames.append(cap.read()[1])
            break
    cap.release()
    cv2.destroyAllWindows()

print(matcher.compare(id_img, frames))
