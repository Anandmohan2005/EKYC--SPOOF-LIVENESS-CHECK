"""
Identity verification: compare the face on an ID card photo with the live face.

Uses OpenCV's built-in YuNet (face detection) + SFace (face embedding).
No extra pip packages are needed. The two small model files are downloaded
automatically on first use into a 'models' folder next to this file.

Decision rule: cosine similarity >= THRESHOLD  ->  same person.
(0.363 is the value recommended by the SFace authors; tune it on your own photos.)
"""
import os
import urllib.request

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
MODELS = {
    "face_detection_yunet_2023mar.onnx": (
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx", 200_000),
    "face_recognition_sface_2021dec.onnx": (
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
        "face_recognition_sface/face_recognition_sface_2021dec.onnx", 30_000_000),
}
COSINE = 0      # cv2.FaceRecognizerSF_FR_COSINE


def _ensure_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for name, (url, min_size) in MODELS.items():
        path = os.path.join(MODEL_DIR, name)
        if os.path.exists(path) and os.path.getsize(path) >= min_size:
            continue
        print(f"Downloading {name} ...")
        urllib.request.urlretrieve(url, path)
        if os.path.getsize(path) < min_size:
            os.remove(path)
            raise RuntimeError(
                f"Download of {name} failed (file too small). Download it manually from\n{url}\n"
                f"and save it into {MODEL_DIR}")
    return [os.path.join(MODEL_DIR, n) for n in MODELS]


def box_iou(a, b):
    """Overlap (0..1) of two boxes given as (x, y, w, h)."""
    ax, ay, aw, ah = a[:4]
    bx, by, bw, bh = b[:4]
    iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def pick_tracked(faces, mesh_box, other_min_score=0.8, other_ratio=0.35):
    """
    Bind every check to ONE face. 'mesh_box' is the box (x, y, w, h) of the face the liveness
    tracker follows. Returns (main, others):
      main   - the detector face whose centre lies inside the tracked box (largest such), or None
      others - confident faces of ANOTHER person that are not tiny background faces:
               width >= other_ratio * (width of the main face)
               (if the tracked face is not found by the detector: width >= other_ratio * 0.6 * tracked width)
    If 'others' is empty, the tracked face is the largest face in the frame, so identity matching
    and the spoof crop (which both use the largest face) look at the same person as liveness.
    """
    mx, my, mw, mh = mesh_box
    inside = [f for f in faces
              if mx <= f[0] + f[2] / 2 <= mx + mw and my <= f[1] + f[3] / 2 <= my + mh]
    main = max(inside, key=lambda f: f[2] * f[3]) if inside else None
    ref_w = main[2] if main is not None else 0.6 * mw
    others = [f for f in faces if f is not main and f not in inside
              and f[4] >= other_min_score and f[2] >= other_ratio * ref_w]
    return main, others


class FaceMatcher:
    def __init__(self, threshold=0.363, max_side=1280):
        det_path, rec_path = _ensure_models()
        self.threshold = threshold
        self.max_side = max_side
        self.detector = cv2.FaceDetectorYN.create(det_path, "", (320, 320), 0.7, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(rec_path, "")

    def _prep(self, img):
        h, w = img.shape[:2]
        s = self.max_side / max(h, w)
        return cv2.resize(img, (int(w * s), int(h * s))) if s < 1 else img

    def detect(self, img):
        """Return the largest face (YuNet row) or None."""
        img = self._prep(img)
        self.detector.setInputSize((img.shape[1], img.shape[0]))
        _, faces = self.detector.detect(img)
        if faces is None or len(faces) == 0:
            return None, img
        return max(faces, key=lambda f: f[2] * f[3]), img

    def all_faces(self, img):
        """Every face found, as (x, y, w, h, score) in ORIGINAL image coordinates, largest first."""
        prepped = self._prep(img)
        self.detector.setInputSize((prepped.shape[1], prepped.shape[0]))
        _, faces = self.detector.detect(prepped)
        if faces is None or len(faces) == 0:
            return []
        k = img.shape[1] / prepped.shape[1]
        out = [(float(f[0]) * k, float(f[1]) * k, float(f[2]) * k, float(f[3]) * k, float(f[14])) for f in faces]
        return sorted(out, key=lambda f: f[2] * f[3], reverse=True)

    def face_box(self, img):
        """(x, y, w, h) of the largest face in the ORIGINAL image coordinates, or None."""
        face, prepped = self.detect(img)
        if face is None:
            return None
        k = img.shape[1] / prepped.shape[1]
        return tuple(float(v) * k for v in face[:4])

    def embed(self, img):
        """128-d face embedding, or None if no face is found."""
        face, img = self.detect(img)
        if face is None:
            return None
        return self.recognizer.feature(self.recognizer.alignCrop(img, face))

    def similarity(self, e1, e2):
        return float(self.recognizer.match(e1, e2, COSINE))

    def compare(self, id_img, live_imgs):
        """
        id_img    : BGR image of the ID card (or the face photo from it)
        live_imgs : one BGR frame or a list of frames from the live session
        Uses the MEDIAN similarity over the live frames, so one bad frame
        cannot decide the result.
        """
        if isinstance(live_imgs, np.ndarray):
            live_imgs = [live_imgs]
        e_id = self.embed(id_img)
        if e_id is None:
            return {"match": False, "score": None, "threshold": self.threshold,
                    "reason": "No face found in the ID image"}
        scores = [self.similarity(e_id, e) for e in map(self.embed, live_imgs) if e is not None]
        if not scores:
            return {"match": False, "score": None, "threshold": self.threshold,
                    "reason": "No face found in the live frames"}
        score = float(np.median(scores))
        return {"match": score >= self.threshold, "score": round(score, 3),
                "threshold": self.threshold, "frames_used": len(scores),
                "reason": "ok"}
