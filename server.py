"""Server entry point for deployment (gunicorn server:app).

Same setup as web_app.py main(), but it builds the app at import time, listens
on all addresses, and does NOT keep reports on disk (they would hold people's
faces/scores, and a free host wipes them anyway).

Settings can be changed with environment variables:
  MODEL_FILE  spoof model file in this folder   (default best_model_v3.pth)
  SPOOF_THR   spoof threshold                   (default 0.5)
  MATCH_THR   identity threshold                (default 0.363)
  SAVE_REPORTS=1 to save reports to ./reports   (default: off)
"""
import os

import mediapipe as mp

from ekyc_app import SpoofDetector
from face_match import FaceMatcher
from web_app import VerificationService, create_app

HERE = os.path.dirname(os.path.abspath(__file__))

model_file = os.environ.get("MODEL_FILE", "best_model_v3.pth")
model_path = os.path.join(HERE, model_file)
if not os.path.exists(model_path):
    raise SystemExit(f"Spoof model not found: {model_path}")

match_thr = float(os.environ.get("MATCH_THR", "0.363"))
spoof_thr = float(os.environ.get("SPOOF_THR", "0.5"))

matcher = FaceMatcher(threshold=match_thr)
spoof = SpoofDetector(model_path, verifier=matcher)
mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                       refine_landmarks=True, min_detection_confidence=0.5)
service = VerificationService(
    matcher, spoof, mesh,
    spoof_thr=spoof_thr, n_challenges=3, timeout=8.0,
    report_dir=os.path.join(HERE, "reports") if os.environ.get("SAVE_REPORTS") == "1" else None,
    model_name=model_file, step_up=True, gray_margin=0.2)

app = create_app(service)

if __name__ == "__main__":  # local test:  python server.py
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "7860")), threaded=True)
