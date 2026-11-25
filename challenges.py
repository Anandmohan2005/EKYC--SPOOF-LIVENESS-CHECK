"""
Random-challenge liveness engine for the eKYC project.

Input  : MediaPipe Face Mesh landmarks in PIXEL coordinates, taken from the
         frame the user SEES on screen (the mirrored frame).
Output : a status dict every frame (state, prompt, countdown, results).

Challenges: blink, turn_left, turn_right, smile, open_mouth
Flow      : calibrate on a neutral face -> random challenges one by one,
            each with a time limit -> PASSED or FAILED.
"""
import random
import time
from statistics import median

import numpy as np

# ---- MediaPipe Face Mesh landmark indices -------------------------------
RIGHT_EYE = [33, 160, 158, 133, 153, 144]     # p1..p6 for EAR
LEFT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_TOP, MOUTH_BOTTOM, MOUTH_L, MOUTH_R = 13, 14, 78, 308   # inner lips
MOUTH_OUT_L, MOUTH_OUT_R = 61, 291                            # outer corners
FACE_EDGE_A, FACE_EDGE_B = 234, 454
NOSE_TIP = 1

PROMPTS = {
    "blink": "Blink your eyes",
    "turn_left": "Turn your head LEFT",
    "turn_right": "Turn your head RIGHT",
    "smile": "Smile",
    "open_mouth": "Open your mouth",
}


# ---- Feature functions ---------------------------------------------------
def _d(a, b):
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))


def _eye_ar(pts, idx):
    p1, p2, p3, p4, p5, p6 = [pts[i] for i in idx]
    return (_d(p2, p6) + _d(p3, p5)) / (2.0 * _d(p1, p4) + 1e-9)


def ear(pts):
    """Eye Aspect Ratio, averaged over both eyes. Drops when eyes close."""
    return (_eye_ar(pts, RIGHT_EYE) + _eye_ar(pts, LEFT_EYE)) / 2.0


def mar(pts):
    """Mouth Aspect Ratio (inner lips). Rises when the mouth opens."""
    return _d(pts[MOUTH_TOP], pts[MOUTH_BOTTOM]) / (_d(pts[MOUTH_L], pts[MOUTH_R]) + 1e-9)


def smile_ratio(pts):
    """Outer mouth width / face width. Rises when smiling."""
    return _d(pts[MOUTH_OUT_L], pts[MOUTH_OUT_R]) / (_d(pts[FACE_EDGE_A], pts[FACE_EDGE_B]) + 1e-9)


def turn_ratio(pts):
    """Nose position between the face edges: ~0.5 frontal,
    <0.5 nose toward screen-left, >0.5 toward screen-right."""
    xl, xr = sorted([pts[FACE_EDGE_A][0], pts[FACE_EDGE_B][0]])
    return (pts[NOSE_TIP][0] - xl) / (xr - xl + 1e-9)


def metrics(pts):
    """Live values, useful for the on-screen debug display and tuning."""
    return {"ear": ear(pts), "mar": mar(pts),
            "smile": smile_ratio(pts), "turn": turn_ratio(pts)}


# ---- Engine --------------------------------------------------------------
class ChallengeEngine:
    def __init__(self, n_challenges=3, timeout=8.0, calib_seconds=1.5,
                 hold=0.25, gap=1.0, seed=None, sequence=None,
                 ear_closed_frac=0.75,     # eye "closed" below 75% of the open-eye baseline
                 smile_gain=1.10,          # smile = mouth width 10% above neutral
                 mar_open=0.28,            # mouth "open" above this MAR
                 turn_left_thr=0.35, turn_right_thr=0.65,
                 min_closed_frames=2):     # closed-eye frames needed for a blink (use 1 at low frame rates)
        rng = random.Random(seed)
        self.sequence = list(sequence) if sequence else rng.sample(list(PROMPTS), n_challenges)
        self.timeout, self.calib_seconds = timeout, calib_seconds
        self.hold, self.gap = hold, gap
        self.ear_closed_frac, self.smile_gain, self.mar_open = ear_closed_frac, smile_gain, mar_open
        self.turn_left_thr, self.turn_right_thr = turn_left_thr, turn_right_thr
        self.min_closed_frames = min_closed_frames

        self.state = "calibrating"          # calibrating | running | passed | failed
        self.idx = 0
        self.extended = False               # True after a risk-adaptive step-up
        self.results = []                   # [{name, passed, seconds}]
        self.base = {}
        self._calib_start = None
        self._calib = {"ear": [], "smile": []}
        self._challenge_start = self._deadline = None
        self._reset_detectors()

    # -- internals --
    def _reset_detectors(self):
        self._hold_start = None
        self._closed_frames = 0
        self._closed_since = None

    def _begin_challenge(self, start_time):
        self._challenge_start = start_time
        self._deadline = start_time + self.timeout
        self._reset_detectors()

    def _detect(self, name, pts, now):
        if name == "blink":
            closed = ear(pts) < self.ear_closed_frac * self.base["ear"]
            if closed:
                self._closed_frames += 1
                if self._closed_since is None:
                    self._closed_since = now
                return False
            # eyes are open now: was there a proper blink just before?
            ok = (self._closed_frames >= self.min_closed_frames and self._closed_since is not None
                  and (now - self._closed_since) <= 1.5)
            self._closed_frames, self._closed_since = 0, None
            return ok

        if name == "turn_left":
            cond = turn_ratio(pts) < self.turn_left_thr
        elif name == "turn_right":
            cond = turn_ratio(pts) > self.turn_right_thr
        elif name == "smile":
            cond = (smile_ratio(pts) > self.base["smile"] * self.smile_gain
                    and mar(pts) < self.mar_open)
        elif name == "open_mouth":
            cond = mar(pts) > self.mar_open
        else:
            raise ValueError(name)

        if cond:                            # must be held briefly, not a 1-frame glitch
            if self._hold_start is None:
                self._hold_start = now
            return (now - self._hold_start) >= self.hold
        self._hold_start = None
        return False

    def _status(self, now, message=""):
        current = self.sequence[self.idx] if self.idx < len(self.sequence) else None
        left = None
        if self.state == "running" and self._deadline is not None:
            left = max(0.0, self._deadline - max(now, self._challenge_start))
        return {"state": self.state, "challenge": current,
                "prompt": PROMPTS.get(current, ""), "index": self.idx,
                "total": len(self.sequence), "time_left": left,
                "results": list(self.results), "message": message}

    # -- public --
    def status(self, now=None, message=""):
        """Current status dict without feeding a frame."""
        return self._status(time.time() if now is None else now, message)

    def extend(self, n=2, now=None, seed=None):
        """Risk-adaptive step-up: add n more random challenges after the sequence was passed.
        Prefers actions that were not used yet; never repeats the last one back to back."""
        now = time.time() if now is None else now
        rng = random.Random(seed)
        unused = [c for c in PROMPTS if c not in self.sequence]
        new = rng.sample(unused, min(n, len(unused)))
        if len(new) < n:
            fallback = [c for c in PROMPTS if c not in new and c != self.sequence[-1]]
            new += rng.sample(fallback, n - len(new))
        self.sequence += new
        self.state, self.extended = "running", True
        self._begin_challenge(now + self.gap)
        return new

    def update(self, pts, now=None):
        """Call once per frame. pts: (N,2) pixel landmarks, or None if no face."""
        now = time.time() if now is None else now
        if self.state in ("passed", "failed"):
            return self._status(now)

        # 1) calibration on a neutral, frontal face
        if self.state == "calibrating":
            if pts is None:
                self._calib_start, self._calib = None, {"ear": [], "smile": []}
                return self._status(now, "Look at the camera")
            if self._calib_start is None:
                self._calib_start = now
            if 0.40 < turn_ratio(pts) < 0.60:        # only frontal samples
                self._calib["ear"].append(ear(pts))
                self._calib["smile"].append(smile_ratio(pts))
            if now - self._calib_start >= self.calib_seconds:
                if len(self._calib["ear"]) >= 5:
                    self.base = {"ear": median(self._calib["ear"]),
                                 "smile": median(self._calib["smile"])}
                    self.state = "running"
                    self._begin_challenge(now + self.gap)
                else:                                 # not enough good samples: retry
                    self._calib_start, self._calib = None, {"ear": [], "smile": []}
            return self._status(now, "Keep a neutral face, look straight")

        # 2) running challenges
        if now < self._challenge_start:               # short pause between challenges
            return self._status(now, "Get ready")
        name = self.sequence[self.idx]
        if now > self._deadline:
            self.results.append({"name": name, "passed": False, "seconds": self.timeout})
            self.state = "failed"
            return self._status(now, "Time out")
        if pts is not None and self._detect(name, pts, now):
            self.results.append({"name": name, "passed": True,
                                 "seconds": round(now - self._challenge_start, 2)})
            self.idx += 1
            if self.idx >= len(self.sequence):
                self.state = "passed"
            else:
                self._begin_challenge(now + self.gap)
            return self._status(now, "Done")
        return self._status(now, "" if pts is not None else "No face found")
