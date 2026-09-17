"""Pose-shape error for a ZED early-fusion run, comparable to the iPhone captures.

Reads <save_dir>/metrics.jsonl (written by scripts/infer/infer_rgbd_zed.py --save), similarity-
Procrustes-aligns each frame's projected model joints to MediaPipe's 21 landmarks,
and reports the aligned residual / hand-size = POSE-shape error -- the SAME metric
scripts/diag/diag_pose_vs_root.py computes on the iPhone set (iPhone early-fusion median ~51%).

Run:  python scripts/diag/diag_zed_metrics.py <save_dir>
"""

import _bootstrap  # noqa: F401  (repo-root + sibling script imports)

import sys
import json
import os

import numpy as np

from diag_pose_vs_root import umeyama_sim, hand_scale


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "zed_ef_out"
    path = os.path.join(d, "metrics.jsonl")
    if not os.path.exists(path):
        print(f"[diag] no metrics.jsonl in {d} (run scripts/infer/infer_rgbd_zed.py --save {d})")
        return
    pose_e, rzs = [], []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        muv = np.asarray(r["uv"], float)
        mp = np.asarray(r["mp"], float)
        if muv.shape != mp.shape:
            continue
        sc = hand_scale(mp)
        if sc < 1e-6:
            continue
        al, s, t = umeyama_sim(muv, mp)
        pose_e.append(np.sqrt(((al - mp) ** 2).sum(1)).mean() / sc)
        rzs.append(r.get("root_z_m", np.nan))
    pose_e, rzs = np.array(pose_e), np.array(rzs)
    if len(pose_e) == 0:
        print(f"[diag] no usable frames in {path}")
        return
    print(f"[ZED early-fusion]  n={len(pose_e)} frames")
    print(f"  POSE-shape error : median {np.median(pose_e)*100:5.1f}%   "
          f"best {np.min(pose_e)*100:.1f}%   p25 {np.percentile(pose_e,25)*100:.1f}%   "
          f"p75 {np.percentile(pose_e,75)*100:.1f}%")
    print(f"  wrist root_z     : median {np.nanmedian(rzs)*100:5.1f} cm")
    print(f"\n  (iPhone early-fusion ref: pose median 51.4%, best 13%, p25 36% @ ~43cm)")


if __name__ == "__main__":
    main()
