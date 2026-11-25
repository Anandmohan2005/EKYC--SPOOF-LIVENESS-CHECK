"""
eKYC verification app: ID match + random-challenge liveness + spoof detection.

Run:   python ekyc_app.py                      (a file dialog asks for the ID photo)
       python ekyc_app.py --id id_photo.jpg --model best_model_v2.pth

Keys:  r = new attempt (new random challenges)    q = quit

VERIFIED only if ALL three checks pass:
  1. Identity  : face on the ID matches the live face (SFace cosine similarity)
  2. Liveness  : all random challenges completed in time
  3. Spoof     : average P(real) from the trained MobileNetV2 >= threshold
"""
import argparse
import datetime
import json
import os

import cv2
import numpy as np

from challenges import ChallengeEngine, PROMPTS, turn_ratio
from face_match import FaceMatcher, box_iou

HERE = os.path.dirname(os.path.abspath(__file__))
IMG, PAD = 128, 0.8          # MUST match the crop used when training the spoof model (v2)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
GREEN, RED, YELLOW, WHITE = (0, 200, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255)


# ------------------------------------------------------------------ spoof model
class SpoofDetector:
    """MobileNetV2 trained on NUAA (Colab notebook). Output: P(real) in [0, 1]."""

    def __init__(self, path, verifier=None):
        # verifier: a FaceMatcher. Haar boxes that do not overlap the face found by its
        # (more reliable) detector are ignored, so pillows / patterns are never cropped.
        self.verifier = verifier
        import torch
        import torch.nn as nn
        from torchvision import models
        self.torch = torch
        m = models.mobilenet_v2(weights=None)
        m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, 2))
        ck = torch.load(path, map_location="cpu")
        m.load_state_dict(ck["model"])
        m.eval()
        self.model, self.epoch = m, ck.get("epoch")
        self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def crop(self, frame):
        """Same face crop as in training: Haar face, square, side = w * (1 + 2*PAD), 128x128."""
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.haar.detectMultiScale(g, 1.1, 5, minSize=(60, 60))
        if len(faces) == 0:
            return None, None
        if self.verifier is not None:
            ref = self.verifier.face_box(frame)
            if ref is None:
                return None, None
            faces = [f for f in faces if box_iou(f, ref) >= 0.3]
            if not faces:
                return None, None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        cx, cy, half = x + w // 2, y + h // 2, int(w * (1 + 2 * PAD) / 2)
        H, W = frame.shape[:2]
        x0, x1, y0, y1 = max(0, cx - half), min(W, cx + half), max(0, cy - half), min(H, cy + half)
        c = cv2.resize(frame[y0:y1, x0:x1], (IMG, IMG))
        _, buf = cv2.imencode(".jpg", c)                 # training images were JPEGs
        return cv2.imdecode(buf, 1), (x0, y0, x1, y1)

    def score(self, frame):
        c, box = self.crop(frame)
        if c is None:
            return None, None
        rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = ((rgb - MEAN) / STD).transpose(2, 0, 1)[None]
        with self.torch.no_grad():
            p = self.torch.softmax(self.model(self.torch.from_numpy(x)), 1)[0, 1].item()
        return p, box


# ------------------------------------------------------------------ decision logic
def decide(match, results, total, spoof_scores, spoof_thr, min_spoof_frames=5):
    """Combine the three checks into one verdict. Pure function (easy to test)."""
    identity_ok = bool(match.get("match"))
    id_detail = (f"similarity {match['score']} (needs >= {match['threshold']})"
                 if match.get("score") is not None else match.get("reason", ""))

    passed = sum(1 for r in results if r["passed"])
    live_ok = total > 0 and len(results) == total and passed == total
    live_detail = f"{passed}/{total} challenges completed"

    if len(spoof_scores) >= min_spoof_frames:
        avg = float(np.mean(spoof_scores))
        spoof_ok = avg >= spoof_thr
        spoof_detail = f"average real-probability {avg:.2f} (needs >= {spoof_thr}) over {len(spoof_scores)} frames"
    else:
        avg, spoof_ok = None, False
        spoof_detail = f"only {len(spoof_scores)} frames analysed (need {min_spoof_frames})"

    return {"verified": identity_ok and live_ok and spoof_ok,
            "checks": {"identity": {"pass": identity_ok, "detail": id_detail},
                       "liveness": {"pass": live_ok, "detail": live_detail},
                       "spoof": {"pass": spoof_ok, "detail": spoof_detail, "avg": avg}}}


def needs_step_up(match, spoof_scores, spoof_thr, match_thr, gray_margin=0.2,
                  id_below=0.05, id_above=0.10, min_spoof_frames=5):
    """
    Risk-adaptive step-up. Called after the normal challenges were completed.
    Returns (step_up, reason).
      * clear accept or clear reject  -> no extra friction  (False)
      * uncertain evidence            -> ask for extra random challenges (True)
    Uncertain means: spoof score within +-gray_margin of its threshold, identity score
    in [match_thr - id_below, match_thr + id_above), or too little evidence to judge.
    """
    if len(spoof_scores) < min_spoof_frames:
        return True, "too few frames analysed"
    avg = float(np.mean(spoof_scores))
    if avg < spoof_thr - gray_margin:
        return False, f"clearly fake (spoof {avg:.2f})"
    score = match.get("score")
    if score is None:
        return True, "identity could not be judged yet"
    if score < match_thr - id_below:
        return False, f"identity clearly different ({score:.2f})"
    if avg < spoof_thr + gray_margin:
        return True, f"spoof score borderline ({avg:.2f})"
    if score < match_thr + id_above:
        return True, f"identity match borderline ({score:.2f})"
    return False, "clear"


# ------------------------------------------------------------------ drawing helpers
def put(img, text, org, color=WHITE, scale=0.6, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def draw_result(view, final):
    img = (view * 0.25).astype(np.uint8)
    h, w = img.shape[:2]
    ok = final["verified"]
    put(img, "VERIFIED" if ok else "REJECTED", (20, 55), GREEN if ok else RED, 1.4, 3)
    y = 110
    names = {"identity": "1. Identity match", "liveness": "2. Liveness challenges", "spoof": "3. Spoof check"}
    for k, label in names.items():
        c = final["checks"][k]
        put(img, f"{label}: {'PASS' if c['pass'] else 'FAIL'}", (20, y), GREEN if c["pass"] else RED, 0.7, 2)
        put(img, c["detail"][: w // 9], (40, y + 26), WHITE, 0.5, 1)
        y += 70
    put(img, "r = try again    q = quit", (20, h - 20), YELLOW, 0.6, 1)
    return img


def pick_id_file():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(title="Select the ID card photo",
                                      filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
    root.destroy()
    return path


def read_image(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", help="path to the ID card photo")
    ap.add_argument("--model", help="spoof model checkpoint (.pth)")
    ap.add_argument("--spoof-thr", type=float, default=0.68, help="min average P(real); 0.68 = EER point of v2")
    ap.add_argument("--match-thr", type=float, default=0.363, help="min cosine similarity for identity")
    ap.add_argument("--challenges", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=8.0)
    ap.add_argument("--camera", type=int, default=0)
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

    id_path = args.id or pick_id_file()
    if not id_path:
        raise SystemExit("No ID photo selected.")
    id_img = read_image(id_path)
    if id_img is None:
        raise SystemExit(f"Could not read the image: {id_path}")

    matcher = FaceMatcher(threshold=args.match_thr)
    if matcher.embed(id_img) is None:
        raise SystemExit("No face found on the ID photo. Use a clearer, front-facing ID photo.")
    spoof = SpoofDetector(model_path, verifier=matcher)
    print(f"Spoof model: {os.path.basename(model_path)} (epoch {spoof.epoch})")

    import mediapipe as mp
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                                min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam.")

    th = 110
    id_thumb = cv2.resize(id_img, (int(id_img.shape[1] * th / id_img.shape[0]), th))
    os.makedirs(os.path.join(HERE, "reports"), exist_ok=True)

    def new_session():
        return {"engine": ChallengeEngine(n_challenges=args.challenges, timeout=args.timeout),
                "spoof": [], "frames": [], "i": 0, "final": None, "last_spoof": None, "result_img": None,
                "stepped": False, "step_info": None, "max_frames": 10}

    S = new_session()
    while True:
        ok, raw = cap.read()                   # raw = un-mirrored: used for the models
        if not ok:
            break
        view = cv2.flip(raw, 1)                # mirrored: what the user sees + challenge logic
        h, w = view.shape[:2]
        res = face_mesh.process(cv2.cvtColor(view, cv2.COLOR_BGR2RGB))
        pts = None
        if res.multi_face_landmarks:
            pts = np.array([[p.x * w, p.y * h] for p in res.multi_face_landmarks[0].landmark])

        if S["final"] is None:
            st = S["engine"].update(pts)
            S["i"] += 1
            if st["state"] in ("calibrating", "running") and pts is not None:
                if S["i"] % 3 == 0:
                    p, box = spoof.score(raw)
                    if p is not None:
                        S["spoof"].append(p)
                        S["last_spoof"] = p
                if S["i"] % 5 == 0 and len(S["frames"]) < S["max_frames"] and 0.42 < turn_ratio(pts) < 0.58:
                    S["frames"].append(raw.copy())

            if st["state"] == "passed" and not args.no_step_up and not S["stepped"]:
                m0 = (matcher.compare(id_img, S["frames"]) if S["frames"] else
                      {"match": False, "score": None, "threshold": args.match_thr,
                       "reason": "No front-facing live frames captured"})
                need, why = needs_step_up(m0, S["spoof"], args.spoof_thr, args.match_thr, args.gray_margin)
                S["step_info"] = {"triggered": need, "reason": why, "extra_challenges": 2 if need else 0,
                                  "initial_spoof_avg": float(np.mean(S["spoof"])) if S["spoof"] else None,
                                  "initial_match_score": m0.get("score")}
                if need:
                    S["stepped"], S["max_frames"] = True, 20
                    S["engine"].extend(2)
                    st = S["engine"].status(None, "Extra verification needed")
                    print("Step-up:", why)

            if st["state"] in ("passed", "failed"):
                match = (matcher.compare(id_img, S["frames"]) if S["frames"] else
                         {"match": False, "score": None, "threshold": args.match_thr,
                          "reason": "No front-facing live frames captured"})
                S["final"] = decide(match, st["results"], st["total"], S["spoof"], args.spoof_thr)
                S["final"]["step_up"] = S["step_info"] or {"triggered": False, "reason": "not evaluated"}
                S["final"]["match_raw"] = match
                S["final"]["challenges"] = st["results"]
                S["result_img"] = draw_result(view, S["final"])
                stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                with open(os.path.join(HERE, "reports", f"report_{stamp}.json"), "w") as f:
                    json.dump({"time": stamp, "id_file": os.path.basename(id_path),
                               "spoof_model": os.path.basename(model_path), "spoof_threshold": args.spoof_thr,
                               **S["final"]}, f, indent=2, default=float)
                cv2.imwrite(os.path.join(HERE, "reports", f"report_{stamp}.jpg"), S["result_img"])
                print("VERIFIED" if S["final"]["verified"] else "REJECTED", S["final"]["checks"])

        # ---- on-screen display
        if S["final"] is not None:
            frame_out = S["result_img"]
        else:
            frame_out = view.copy()
            frame_out[8:8 + th, w - id_thumb.shape[1] - 8:w - 8] = id_thumb
            put(frame_out, "ID", (w - id_thumb.shape[1] - 8, 8 + th + 18), WHITE, 0.5)
            if st["state"] == "calibrating":
                put(frame_out, st["message"], (15, 35), YELLOW, 0.7, 2)
            else:
                put(frame_out, f"Challenge {st['index'] + 1}/{st['total']}: {st['prompt']}", (15, 35), YELLOW, 0.8, 2)
                if st["time_left"] is not None:
                    put(frame_out, f"Time left: {st['time_left']:.1f}s", (15, 65), WHITE, 0.7, 2)
                if st["message"]:
                    put(frame_out, st["message"], (15, 95), WHITE, 0.6)
            for i, r in enumerate(st["results"]):
                put(frame_out, f"{'OK' if r['passed'] else 'X'}  {PROMPTS[r['name']]}",
                    (15, h - 15 - 24 * (len(st["results"]) - 1 - i)), GREEN if r["passed"] else RED, 0.55)
            if S["stepped"]:
                put(frame_out, "EXTRA VERIFICATION (borderline result)", (15, 125), YELLOW, 0.6, 2)
            if S["last_spoof"] is not None:
                c = GREEN if S["last_spoof"] >= args.spoof_thr else RED
                put(frame_out, f"real-score {S['last_spoof']:.2f}", (w - 170, h - 15), c, 0.55)

        cv2.imshow("eKYC verification (r = retry, q = quit)", frame_out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            S = new_session()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
