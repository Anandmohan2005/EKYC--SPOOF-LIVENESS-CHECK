"""
Turn the saved attempt reports into a results table for your report.

  python summarize_reports.py                 (reads the 'reports' folder next to this file)
  python summarize_reports.py --dir reports --out results_summary.csv

Use the dropdown "This attempt is (for testing)" on the website before each attempt, so every
report is labelled: genuine / screen / replay / print / wrong_person / other.
Attempts without a label are grouped as 'unlabeled'.

Output (printed + saved as CSV):
  * per attempt type: attempts, VERIFIED / REJECTED, and which checks failed
  * defence in depth: how often EACH check alone would have accepted the attempt
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ATTACKS = {"screen", "replay", "print", "wrong_person"}
NAMES = {"genuine": "Genuine live face", "screen": "Photo on phone screen", "replay": "Video replay on phone",
         "print": "Printed photo", "wrong_person": "Different person", "other": "Other", "unlabeled": "Unlabeled"}


def mean(v):
    return sum(v) / len(v) if v else None


def fmt(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(HERE, "reports"))
    ap.add_argument("--out", default=os.path.join(HERE, "results_summary.csv"))
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*report_*.json")))
    if not files:
        raise SystemExit(f"No report files found in {args.dir}")

    groups = defaultdict(list)
    for f in files:
        r = json.load(open(f))
        groups[r.get("attempt_type") or "unlabeled"].append(r)

    rows = []
    order = [k for k in ["genuine", "screen", "replay", "print", "wrong_person", "other", "unlabeled"] if k in groups]
    order += [k for k in groups if k not in order]
    for k in order:
        rs = groups[k]
        n = len(rs)
        ver = sum(r["verified"] for r in rs)
        c = lambda name: [r["checks"][name] for r in rs]
        ident, live, spoof = c("identity"), c("liveness"), c("spoof")
        id_scores = [r["match_raw"]["score"] for r in rs if r.get("match_raw", {}).get("score") is not None]
        sp_scores = [x["avg"] for x in spoof if x.get("avg") is not None]
        rows.append({
            "attempt_type": NAMES.get(k, k), "attempts": n, "verified": ver, "rejected": n - ver,
            "verified_%": round(100 * ver / n, 1),
            "identity_pass": sum(x["pass"] for x in ident),
            "liveness_pass": sum(x["pass"] for x in live),
            "spoof_pass": sum(x["pass"] for x in spoof),
            "avg_identity_score": None if mean(id_scores) is None else round(mean(id_scores), 3),
            "avg_spoof_score": None if mean(sp_scores) is None else round(mean(sp_scores), 3),
            "kind": "attack" if k in ATTACKS else ("genuine" if k == "genuine" else "other"),
        })

    cols = ["attempt_type", "attempts", "verified", "rejected", "verified_%", "identity_pass",
            "liveness_pass", "spoof_pass", "avg_identity_score", "avg_spoof_score"]
    w = [max(len(c), *(len(str(r[c])) for r in rows)) for c in cols]
    print("\nRESULTS PER ATTEMPT TYPE   (xxx_pass = attempts in which that check passed)\n")
    print("  ".join(c.ljust(wi) for c, wi in zip(cols, w)))
    for r in rows:
        print("  ".join(str(r[c]).ljust(wi) for c, wi in zip(cols, w)))

    att = [r for r in rows if r["kind"] == "attack"]
    gen = [r for r in rows if r["kind"] == "genuine"]
    if att:
        n = sum(r["attempts"] for r in att)
        print(f"\nAttack success rate (attacks accepted):  {sum(r['verified'] for r in att)}/{n} = "
              f"{100 * sum(r['verified'] for r in att) / n:.1f}%")
        print("Each check ALONE would have accepted this share of the attacks:")
        for key, label in [("identity_pass", "identity only"), ("liveness_pass", "liveness only"),
                           ("spoof_pass", "spoof detection only")]:
            print(f"  {label:22s} {sum(r[key] for r in att)}/{n} = {100 * sum(r[key] for r in att) / n:.1f}%")
        print("  all three together     "
              f"{sum(r['verified'] for r in att)}/{n} = {100 * sum(r['verified'] for r in att) / n:.1f}%")
    if gen:
        n = sum(r["attempts"] for r in gen)
        print(f"Genuine acceptance rate:  {sum(r['verified'] for r in gen)}/{n} = "
              f"{100 * sum(r['verified'] for r in gen) / n:.1f}%  "
              f"(genuine users wrongly rejected: {n - sum(r['verified'] for r in gen)})")

    allr = [r for rs in groups.values() for r in rs]
    stepped = [r for r in allr if (r.get("step_up") or {}).get("triggered")]
    print(f"\nRisk-adaptive step-up triggered in {len(stepped)}/{len(allr)} attempts "
          f"(verified after step-up: {sum(bool(r.get('verified')) for r in stepped)})")

    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows({c: r[c] for c in cols} for r in rows)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
