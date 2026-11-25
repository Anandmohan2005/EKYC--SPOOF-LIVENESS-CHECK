"""
Record your OWN labelled face data for training the spoof model (v3).

  python collect_data.py --person p01 --label real
  python collect_data.py --person p01 --label screen
  python collect_data.py --person p01 --label print

Labels:  real   = the live person's face
         screen = the person's face shown on a phone/tablet screen (photo OR video replay)
         print  = the person's face on a printed photo

Keys:  SPACE = start / stop recording      q = quit

Face crops are saved to  own_data/<person>/<label>/  using the SAME crop as the
app and the training notebook (Haar face, square, 128x128, 80% margin).
Use a new person id (p01, p02, ...) for every different person.
"""
import argparse
import os

import cv2

from face_match import FaceMatcher, box_iou

HERE = os.path.dirname(os.path.abspath(__file__))
IMG, PAD = 128, 0.8
HINTS = {
    "real": "Move your head a little, blink, vary the light",
    "screen": "Hold the phone so the face on it is the biggest face. Vary distance/angle",
    "print": "Hold the printed photo in front of the camera. Vary distance/angle/bending",
}


def crop_face(frame, haar, matcher):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(g, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        return None, None
    # keep only Haar boxes that agree with the (more reliable) YuNet detector,
    # so cushions, pillows and patterns are never saved as faces
    ref = matcher.face_box(frame)
    if ref is None:
        return None, None
    faces = [f for f in faces if box_iou(f, ref) >= 0.3]
    if not faces:
        return None, None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    cx, cy, half = x + w // 2, y + h // 2, int(w * (1 + 2 * PAD) / 2)
    H, W = frame.shape[:2]
    x0, x1, y0, y1 = max(0, cx - half), min(W, cx + half), max(0, cy - half), min(H, cy + half)
    return cv2.resize(frame[y0:y1, x0:x1], (IMG, IMG)), (x, y, w, h)


def put(img, text, org, color=(255, 255, 255), scale=0.6, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", required=True, help="person id, e.g. p01")
    ap.add_argument("--label", required=True, choices=["real", "screen", "print"])
    ap.add_argument("--every", type=int, default=4, help="save every Nth frame while recording")
    ap.add_argument("--max", type=int, default=300, help="stop automatically after this many images")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    out = os.path.join(HERE, "own_data", args.person, args.label)
    os.makedirs(out, exist_ok=True)
    existing = len([f for f in os.listdir(out) if f.lower().endswith(".jpg")])

    haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    matcher = FaceMatcher()
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam.")

    recording, saved, i = False, 0, 0
    while True:
        ok, raw = cap.read()                  # raw (un-mirrored) frames are what we crop and save
        if not ok:
            break
        i += 1
        view = cv2.flip(raw, 1)
        crop, box = crop_face(raw, haar, matcher)

        if recording and crop is not None and i % args.every == 0:
            cv2.imwrite(os.path.join(out, f"{args.person}_{args.label}_{existing + saved:05d}.jpg"), crop)
            saved += 1
            if saved >= args.max:
                recording = False

        W = view.shape[1]
        if box is not None:
            x, y, w, h = box
            cv2.rectangle(view, (W - x - w, y), (W - x, y + h),
                          (0, 0, 255) if recording else (0, 200, 0), 2)
        put(view, f"person {args.person} | label {args.label}", (10, 25), (0, 255, 255), 0.7, 2)
        put(view, "RECORDING" if recording else "paused (SPACE to start)", (10, 55),
            (0, 0, 255) if recording else (255, 255, 255), 0.7, 2)
        put(view, f"saved this run: {saved}   total in folder: {existing + saved}", (10, 85))
        put(view, "No face found" if box is None else HINTS[args.label], (10, view.shape[0] - 15),
            (0, 0, 255) if box is None else (255, 255, 255), 0.5)
        cv2.imshow("Collect data (SPACE = start/stop, q = quit)", view)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == 32:
            recording = not recording

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {saved} images to {out}")


if __name__ == "__main__":
    main()
