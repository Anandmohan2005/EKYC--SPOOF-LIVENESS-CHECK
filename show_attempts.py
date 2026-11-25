"""
List individual attempts from the 'reports' folder, so you can see WHICH ones were rejected / verified.

  python show_attempts.py                      -> genuine attempts that were rejected + 'other' attempts that were verified
  python show_attempts.py --type genuine --status rejected
  python show_attempts.py --type other --status any
  python show_attempts.py --type genuine --status verified

--type   : the label you chose in the website dropdown (genuine, print, screen, replay, wrong_person, other, ...)
--status : verified | rejected | any
"""
import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def line(r):
    c = r.get("checks", {})
    p = lambda k: "pass" if c.get(k, {}).get("pass") else "FAIL"
    sp = (c.get("spoof") or {}).get("avg")
    sc = (r.get("match_raw") or {}).get("score")
    sf = c.get("single_face")
    t = str(r.get("time", ""))
    when = f"{t[0:4]}-{t[4:6]}-{t[6:8]} {t[9:11]}:{t[11:13]}:{t[13:15]}" if len(t) >= 15 else t
    su = r.get("step_up") or {}
    return (f"{when}  {str(r.get('attempt_type') or 'unlabeled'):12s} "
            f"{'VERIFIED' if r.get('verified') else 'rejected':8s} "
            f"id {p('identity')} ({'-' if sc is None else f'{sc:.2f}'})  "
            f"live {p('liveness')}  spoof {p('spoof')} ({'-' if sp is None else f'{sp:.2f}'})  "
            f"one-person {('-' if sf is None else ('pass' if sf.get('pass') else 'FAIL'))}"
            f" ({r.get('extra_face_seconds', '-')}s)  "
            f"step-up {'yes' if su.get('triggered') else 'no'}  id file: {r.get('id_file', '')}")


def show(reps, typ, status):
    rows = [r for r in reps
            if (r.get("attempt_type") or "unlabeled") == typ
            and (status == "any" or (status == "verified") == bool(r.get("verified")))]
    rows.sort(key=lambda r: str(r.get("time")))
    print(f"\n=== type '{typ}', status '{status}': {len(rows)} attempts ===")
    for r in rows:
        print(line(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type")
    ap.add_argument("--status", default="any", choices=["verified", "rejected", "any"])
    ap.add_argument("--dir", default=os.path.join(HERE, "reports"))
    a = ap.parse_args()
    reps = []
    for fp in glob.glob(os.path.join(a.dir, "*.json")):
        try:
            reps.append(json.load(open(fp)))
        except Exception:
            pass
    if a.type:
        show(reps, a.type, a.status)
    else:
        show(reps, "genuine", "rejected")
        show(reps, "other", "verified")


if __name__ == "__main__":
    main()
