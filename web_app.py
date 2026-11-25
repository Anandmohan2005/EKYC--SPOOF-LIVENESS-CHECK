"""
eKYC web app (runs on your laptop):  ID upload -> random-challenge liveness -> spoof check -> verdict.

  pip install flask                       (one-time, inside your venv)
  python web_app.py                       then open  http://localhost:5000  in Chrome/Edge

Files needed in the same folder: web_app.py, index.html, ekyc_app.py, challenges.py,
face_match.py and best_model_v2.pth.

The browser sends camera frames to this local server (nothing leaves your laptop). The server
runs exactly the same pipeline as ekyc_app.py.
"""
import argparse
import datetime
import json
import os
import threading
import time
import uuid

import cv2
import numpy as np

from challenges import ChallengeEngine, PROMPTS, turn_ratio
from ekyc_app import decide, needs_step_up
from face_match import pick_tracked

HERE = os.path.dirname(os.path.abspath(__file__))
SPOOF_EVERY = 2        # score every 2nd frame with the spoof model
MATCH_EVERY = 2        # keep every 2nd frontal frame for identity matching
MAX_MATCH_FRAMES = 8
EXTRA_FACE_LIMIT = 3.0     # seconds another person may be visible before the attempt fails


class Session:
    def __init__(self, id_img, id_name):
        self.id_img, self.id_name = id_img, id_name
        self.reset(None)

    def reset(self, engine):
        self.engine = engine
        self.spoof, self.frames = [], []
        self.i, self.final, self.last_spoof = 0, None, None
        self.stepped_up, self.step_up_info, self.max_frames = False, None, MAX_MATCH_FRAMES
        self.attempt_type = getattr(self, 'attempt_type', '')
        self.extra_secs, self.last_t, self.warning = 0.0, None, ""


class VerificationService:
    """All the logic, independent of Flask (so it can be tested without a server)."""

    def __init__(self, matcher, spoof, mesh, spoof_thr=0.68, n_challenges=3, timeout=8.0,
                 report_dir=None, model_name="", step_up=True, step_up_challenges=2, gray_margin=0.2):
        self.matcher, self.spoof, self.mesh = matcher, spoof, mesh
        self.spoof_thr, self.n_challenges, self.timeout = spoof_thr, n_challenges, timeout
        self.report_dir, self.model_name = report_dir, model_name
        self.step_up, self.step_up_challenges, self.gray_margin = step_up, step_up_challenges, gray_margin
        self.match_thr = matcher.threshold
        self.sessions = {}
        self.lock = threading.Lock()          # one frame at a time (MediaPipe / torch are shared)

    # ---------------------------------------------------------------- sessions
    def create_session(self, id_img, id_name="id"):
        if id_img is None:
            return None, "Could not read the image. Upload a JPG or PNG."
        with self.lock:
            if self.matcher.embed(id_img) is None:
                return None, "No face found on the ID photo. Use a clear, front-facing photo."
            faces = self.matcher.all_faces(id_img)
            big = [f for f in faces if f[4] >= 0.8 and f[2] >= 0.7 * faces[0][2]]
            if len(big) >= 2:
                return None, ("More than one face of similar size was found on the ID photo. "
                              "Upload a photo that shows only the ID holder.")
        sid = uuid.uuid4().hex
        self.sessions[sid] = Session(id_img, id_name)
        return sid, None

    def start(self, sid, attempt_type=""):
        s = self.sessions.get(sid)
        if s is None:
            return False
        s.reset(ChallengeEngine(n_challenges=self.n_challenges, timeout=self.timeout,
                                min_closed_frames=1))
        s.attempt_type = "".join(c for c in str(attempt_type).lower() if c.isalnum() or c == "_")[:30]
        return True

    # ---------------------------------------------------------------- frames
    def process(self, sid, raw, now=None):
        s = self.sessions.get(sid)
        if s is None or s.engine is None:
            return {"error": "Session not started"}
        if raw is None:
            return {"error": "Bad frame"}
        if s.final is not None:
            return self._out(s, s.engine.update(None, now), True)

        with self.lock:
            t = time.time() if now is None else now
            view = cv2.flip(raw, 1)            # mirrored: what the user sees (challenge logic)
            h, w = view.shape[:2]
            res = self.mesh.process(cv2.cvtColor(view, cv2.COLOR_BGR2RGB))
            pts = None
            if res.multi_face_landmarks:
                pts = np.array([[p.x * w, p.y * h] for p in res.multi_face_landmarks[0].landmark])

            # --- one-person rule: the tracked face must be the only real face in view
            competing = False
            s.warning = ""
            if pts is not None:
                x0, x1 = float(pts[:, 0].min()), float(pts[:, 0].max())
                y0, y1 = float(pts[:, 1].min()), float(pts[:, 1].max())
                mesh_box = (w - x1, y0, x1 - x0, y1 - y0)       # back to un-mirrored coordinates
                _, others = pick_tracked(self.matcher.all_faces(raw), mesh_box)
                competing = bool(others)
            if competing:
                dt = 0.0 if s.last_t is None else min(max(t - s.last_t, 0.0), 0.5)
                s.extra_secs += dt
                s.warning = "Only one person should be in view"
            s.last_t = t
            use = None if competing else pts       # while another face is visible nothing counts

            if s.extra_secs >= EXTRA_FACE_LIMIT:
                st = s.engine.status(now, "More than one person in view")
                self._finalize(s, st)
                return self._out(s, st, pts is not None)

            st = s.engine.update(use, now)
            s.i += 1
            if st["state"] in ("calibrating", "running") and use is not None:
                if s.i % SPOOF_EVERY == 0:
                    p, _ = self.spoof.score(raw)       # models use the un-mirrored frame
                    if p is not None:
                        s.spoof.append(p)
                        s.last_spoof = p
                if (s.i % MATCH_EVERY == 0 and len(s.frames) < s.max_frames
                        and 0.42 < turn_ratio(use) < 0.58):
                    s.frames.append(raw.copy())

            if st["state"] == "passed" and self.step_up and not s.stepped_up:
                st = self._maybe_step_up(s, st, now)
            if st["state"] in ("passed", "failed"):
                self._finalize(s, st)
        return self._out(s, st, pts is not None)

    def _maybe_step_up(self, s, st, now):
        """Risk-adaptive step-up: borderline evidence -> a few more random challenges."""
        match = (self.matcher.compare(s.id_img, s.frames) if s.frames else
                 {"match": False, "score": None, "threshold": self.match_thr,
                  "reason": "No front-facing live frames captured"})
        need, why = needs_step_up(match, s.spoof, self.spoof_thr, self.match_thr, self.gray_margin)
        s.step_up_info = {"triggered": need, "reason": why,
                          "extra_challenges": self.step_up_challenges if need else 0,
                          "initial_spoof_avg": float(np.mean(s.spoof)) if s.spoof else None,
                          "initial_match_score": match.get("score")}
        if not need:
            return st
        s.stepped_up, s.max_frames = True, 2 * MAX_MATCH_FRAMES
        s.engine.extend(self.step_up_challenges, now)
        return s.engine.status(now, "Extra verification needed")

    def _finalize(self, s, st):
        if s.frames:
            match = self.matcher.compare(s.id_img, s.frames)
        else:
            match = {"match": False, "score": None, "threshold": self.matcher.threshold,
                     "reason": "No front-facing live frames captured"}
        final = decide(match, st["results"], st["total"], s.spoof, self.spoof_thr)
        final["match_raw"] = match
        final["challenges"] = st["results"]
        sf_ok = s.extra_secs < EXTRA_FACE_LIMIT
        final["checks"]["single_face"] = {
            "pass": sf_ok,
            "detail": ("only one person in view" if sf_ok else
                       f"another face was in view for {s.extra_secs:.1f} s (limit {EXTRA_FACE_LIMIT} s)")}
        final["extra_face_seconds"] = round(s.extra_secs, 2)
        final["verified"] = bool(final["verified"] and sf_ok)
        final["step_up"] = s.step_up_info or {"triggered": False, "reason": "not evaluated"}
        s.final = final
        if self.report_dir:
            os.makedirs(self.report_dir, exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(os.path.join(self.report_dir, f"web_report_{stamp}.json"), "w") as f:
                json.dump({"time": stamp, "source": "web", "attempt_type": s.attempt_type, "id_file": s.id_name,
                           "spoof_model": self.model_name, "spoof_threshold": self.spoof_thr,
                           **final}, f, indent=2, default=float)

    def _out(self, s, st, face_found):
        return {"state": st["state"], "challenge": st["challenge"], "prompt": st["prompt"],
                "index": st["index"], "total": st["total"], "time_left": st["time_left"],
                "timeout": s.engine.timeout, "message": st["message"], "face_found": face_found,
                "results": [{**r, "label": PROMPTS[r["name"]]} for r in st["results"]],
                "spoof_live": s.last_spoof, "spoof_thr": self.spoof_thr, "final": s.final,
                "step_up": s.stepped_up, "warning": s.warning}


# -------------------------------------------------------------------- Flask layer
def create_app(service):
    from flask import Flask, Response, request

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

    def jr(obj, code=200):
        return Response(json.dumps(obj, default=float), status=code, mimetype="application/json")

    def decode(file_storage):
        data = np.frombuffer(file_storage.read(), np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None

    @app.get("/")
    def index():
        with open(os.path.join(HERE, "index.html"), encoding="utf-8") as f:
            return Response(f.read(), mimetype="text/html", headers={"Cache-Control": "no-store"})

    @app.post("/api/upload_id")
    def upload_id():
        f = request.files.get("id_image")
        if f is None:
            return jr({"error": "No file uploaded"}, 400)
        sid, err = service.create_session(decode(f), os.path.basename(f.filename or "id"))
        return jr({"error": err}, 400) if err else jr({"session_id": sid})

    @app.post("/api/start")
    def start():
        data = request.get_json(silent=True) or {}
        ok = service.start(data.get("session_id"), data.get("attempt_type", ""))
        return jr({"ok": True}) if ok else jr({"error": "Unknown session"}, 404)

    @app.post("/api/frame")
    def frame():
        f = request.files.get("frame")
        if f is None:
            return jr({"error": "No frame"}, 400)
        out = service.process(request.form.get("session_id"), decode(f))
        return jr(out, 400 if "error" in out else 200)

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="spoof model checkpoint (.pth)")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--spoof-thr", type=float, default=0.68)
    ap.add_argument("--match-thr", type=float, default=0.363)
    ap.add_argument("--challenges", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=8.0)
    ap.add_argument("--gray-margin", type=float, default=0.2, help="spoof gray zone = threshold +- this")
    ap.add_argument("--no-step-up", action="store_true", help="turn off the risk-adaptive extra challenges")
    args = ap.parse_args()

    model_path = args.model
    if not model_path:
        for n in ("best_model_v2.pth", "best_model.pth"):
            if os.path.exists(os.path.join(HERE, n)):
                model_path = os.path.join(HERE, n)
                break
    if not model_path or not os.path.exists(model_path):
        raise SystemExit("Spoof model not found. Put best_model_v2.pth in this folder or use --model <file>.")

    from face_match import FaceMatcher
    from ekyc_app import SpoofDetector
    import mediapipe as mp

    matcher = FaceMatcher(threshold=args.match_thr)
    spoof = SpoofDetector(model_path, verifier=matcher)
    print(f"Spoof model: {os.path.basename(model_path)} (epoch {spoof.epoch})")
    mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                           refine_landmarks=True, min_detection_confidence=0.5)
    service = VerificationService(matcher, spoof, mesh,
                                  spoof_thr=args.spoof_thr, n_challenges=args.challenges,
                                  timeout=args.timeout, report_dir=os.path.join(HERE, "reports"),
                                  model_name=os.path.basename(model_path),
                                  step_up=not args.no_step_up, gray_margin=args.gray_margin)
    print(f"\nOpen  http://localhost:{args.port}  in Chrome or Edge.  Press Ctrl+C here to stop.\n")
    create_app(service).run(host="127.0.0.1", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
