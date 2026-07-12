"""Split the iPhone-capture error into POSE-shape vs ROOT/scale, without 3D GT.

We have no 3-D ground truth for the captures, so we use MediaPipe's 21 2-D
landmarks as a pseudo-GT for the hand's image-plane configuration and compare each
model's PROJECTED 2-D joints (`uv`, saved in the *_pred*.json) against them.

A similarity-Procrustes fit (translation + rotation + uniform scale) is removed
before measuring the residual:
  * pose-shape error  = Procrustes-aligned residual / hand-scale  -> is the joint
        CONFIGURATION right? (independent of where/how big the hand is placed)
  * root/scale error  = what the alignment had to remove: the fitted scale s
        (s>1 => model hand projects too small => root placed too far) and the
        translation magnitude (root x,y offset).

If pose-shape error is large and similar across models, the inaccuracy is in the
shared POSE backbone (retraining the dual-stream root head won't help). If pose is
small but scale/translation are off, it's a ROOT problem.

Run:  python diag_pose_vs_root.py
"""

import os
import re
import glob
import json

import numpy as np

from infer_rgbd_captures import load_capture, rotate_frame
from infer_rgbd_femtobolt import HandDetector


def umeyama_sim(src, dst):
    """Best similarity (s,R,t) mapping src->dst (both (N,2)); proper rotation only."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    var_s = (s0 ** 2).sum() / len(src)
    cov = (d0.T @ s0) / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1                       # keep a proper rotation (no reflection)
    R = U @ S @ Vt
    s = (D * np.diag(S)).sum() / var_s
    t = mu_d - s * (R @ mu_s)
    aligned = (s * (R @ src.T)).T + t
    return aligned, float(s), t


def hand_scale(uv):
    return float(np.sqrt(((uv - uv.mean(0)) ** 2).sum() / len(uv)))


def eval_model(stems, k, mp_by_stem, suffix):
    """For one prediction set (suffix like '_pred_ef'), return per-frame metrics."""
    pose_e, raw_e, scales, trans_e = [], [], [], []
    for stem in stems:
        jp = stem + suffix + ".json"
        if stem not in mp_by_stem or not os.path.exists(jp):
            continue
        d = json.load(open(jp))
        if not d.get("detected") or "uv" not in d:
            continue
        model_uv = np.asarray(d["uv"], np.float64)        # projected model joints
        mp_uv = mp_by_stem[stem]                           # pseudo-GT (21,2)
        if model_uv.shape != mp_uv.shape:
            continue
        sc = hand_scale(mp_uv)
        if sc < 1e-6:
            continue
        aligned, s, t = umeyama_sim(model_uv, mp_uv)
        pose_e.append(np.sqrt(((aligned - mp_uv) ** 2).sum(1)).mean() / sc)
        raw_e.append(np.sqrt(((model_uv - mp_uv) ** 2).sum(1)).mean() / sc)
        scales.append(s)
        trans_e.append(np.linalg.norm(t) / sc)
    return (np.array(pose_e), np.array(raw_e), np.array(scales), np.array(trans_e))


def main():
    d = "rgbd_captures"
    base_re = re.compile(r"^cap_\d+\.json$")
    stems = sorted(s[:-5] for s in glob.glob(os.path.join(d, "cap_*.json"))
                   if base_re.match(os.path.basename(s)))
    # global rotation: read it back from a prediction sidecar (all share the same k)
    k = 1
    for stem in stems:
        p = stem + "_pred.json"
        if os.path.exists(p):
            k = int(json.load(open(p)).get("rotation_deg_cw", 90)) // 90 % 4
            break
    print(f"[diag] {len(stems)} captures, rotation k={k} ({k*90} deg CW)")

    det = HandDetector()
    mp_by_stem = {}
    for stem in stems:
        cap = load_capture(stem)
        if cap is None:
            continue
        bgr0, depth0, K0 = cap
        bgr, _, _ = rotate_frame(bgr0, depth0, K0, k)
        pts = det.detect(bgr)
        if pts is not None:
            mp_by_stem[stem] = np.asarray(pts, np.float64)[:, :2]
    det.close()
    print(f"[diag] MediaPipe pseudo-GT on {len(mp_by_stem)} frames\n")

    sets = [("dual-stream orig   ", "_pred"),
            ("dual-stream MASKfix", "_pred_mask"),
            ("early-fusion        ", "_pred_ef")]
    print(f"{'model':20s} {'n':>4}  {'POSE-shape':>22}  {'scale s (root_z)':>18}  {'trans(root_xy)':>14}")
    print(f"{'':20s} {'':>4}  {'aligned resid / size':>22}  {'>1 => too far':>18}  {'/ hand size':>14}")
    for name, suf in sets:
        pe, re_, sc, te = eval_model(stems, k, mp_by_stem, suf)
        if len(pe) == 0:
            print(f"{name:20s}  (no data)")
            continue
        print(f"{name:20s} {len(pe):>4}  "
              f"med {np.median(pe)*100:5.1f}%  (raw {np.median(re_)*100:4.0f}%)   "
              f"med {np.median(sc):6.2f}        "
              f"med {np.median(te)*100:5.1f}%")


if __name__ == "__main__":
    main()
