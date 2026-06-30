"""Held-out eval: baseline vs fine-tuned early-fusion on an unseen iPhone session.

Uses the pseudo-3D GT (cap_*_gt.json) as the reference: its projected 2D == the
MediaPipe landmarks, and its `joint_valid` mask restricts the comparison to the
reliable joints. For each model's projected joints (`uv` in cap_*_pred_<tag>.json)
we remove a similarity (Procrustes: translation+rotation+uniform scale) and report:

  * pose-shape error = aligned residual / hand-size  -> is the CONFIGURATION right?
  * scale s          = what alignment had to undo (s>1 => hand projects too small)

Lower pose-shape = less finger-cramping. Compare --tags on the held-out folder.

    python eval_iphone_heldout.py --dir rgbd_captures_04 --tags base,ft
"""
import argparse
import glob
import json
import os

import numpy as np


def umeyama_sim(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    var_s = (s0 ** 2).sum() / len(src)
    cov = (d0.T @ s0) / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    s = (D * np.diag(S)).sum() / var_s
    aligned = (s * (R @ src.T)).T + (mu_d - s * (R @ mu_s))
    return aligned, float(s)


def project(j3d, K):
    uv = (np.asarray(K) @ np.asarray(j3d).T).T
    return uv[:, :2] / uv[:, 2:3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="rgbd_captures_04")
    ap.add_argument("--tags", default="base,ft")
    ap.add_argument("--min_valid", type=int, default=8)
    args = ap.parse_args()
    tags = [t.strip() for t in args.tags.split(",")]

    stems = sorted(g[:-len("_gt.json")] for g in glob.glob(os.path.join(args.dir, "cap_*_gt.json")))
    print(f"[eval] {len(stems)} GT frames in {args.dir}; tags={tags}\n")

    res = {t: {"pose": [], "raw": [], "scale": []} for t in tags}
    n_common = 0
    for stem in stems:
        g = json.load(open(stem + "_gt.json"))
        valid = np.array(g["joint_valid"], bool)
        if valid.sum() < args.min_valid:
            continue
        gt_uv = project(g["joints3d_cam"], g["K"])              # == MediaPipe uv
        preds = {}
        for t in tags:
            p = stem + f"_pred_{t}.json"
            if not os.path.exists(p):
                break
            d = json.load(open(p))
            if not d.get("detected") or "uv" not in d:
                break
            preds[t] = np.asarray(d["uv"], np.float64)
        if len(preds) != len(tags):
            continue
        gv = gt_uv[valid]
        sc = float(np.sqrt(((gv - gv.mean(0)) ** 2).sum() / len(gv)))
        if sc < 1e-6:
            continue
        n_common += 1
        for t in tags:
            mv = preds[t][valid]
            aligned, s = umeyama_sim(mv, gv)
            res[t]["pose"].append(np.sqrt(((aligned - gv) ** 2).sum(1)).mean() / sc)
            res[t]["raw"].append(np.sqrt(((mv - gv) ** 2).sum(1)).mean() / sc)
            res[t]["scale"].append(s)

    print(f"[eval] {n_common} frames common to all tags (>= {args.min_valid} valid joints)\n")
    print(f"{'model(tag)':12s} {'POSE-shape (med)':>17} {'p25':>6} {'p75':>6}  {'raw(med)':>9} {'scale s':>8}")
    for t in tags:
        pe = np.array(res[t]["pose"]); rw = np.array(res[t]["raw"]); sc = np.array(res[t]["scale"])
        if len(pe) == 0:
            print(f"{t:12s}  (no data)"); continue
        print(f"{t:12s} {np.median(pe)*100:15.1f}% {np.percentile(pe,25)*100:5.0f}% "
              f"{np.percentile(pe,75)*100:5.0f}%  {np.median(rw)*100:7.0f}% {np.median(sc):8.2f}")


if __name__ == "__main__":
    main()
