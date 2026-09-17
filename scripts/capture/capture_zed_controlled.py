"""Controlled ZED capture to QUANTIFY absolute root-z error of ds_anchor_gate.

Hold the hand STILL at a tape-measured distance from the ZED front face. The tool
captures a few seconds and reports, over the steady high-confidence frames:
  * predicted root z (== abs wrist z)         -- the deploy number
  * geometric anchor z                        -- depth-only anchor before the residual
  * ZED sensor depth at the wrist pixel        -- a per-frame on-rig depth reference
  * errors vs the user-measured GT (--gt_z) and vs the ZED wrist depth

Two independent GTs:
  (1) --gt_z   : your tape measurement (absolute truth, but coarse / palm-vs-wrist).
  (2) zed_wrist: ZED NEURAL depth sampled at the MediaPipe wrist landmark (dense,
                 ~1-2% accurate <1m, but it's the sensor so it cross-checks the
                 LEARNED residual, not the anchor which is also depth-derived).

Run (repeat at several distances for a bias-vs-scale read):
  python scripts/capture/capture_zed_controlled.py --gt_z 0.50 --label 50cm --seconds 8
  python scripts/capture/capture_zed_controlled.py --gt_z 0.40 --label 40cm --seconds 8
"""

import _bootstrap  # noqa: F401  (repo-root + sibling script imports)

import os
import csv
import time
import argparse

import numpy as np
import cv2

from infer_zed_dualstream import DualStreamHandPose
from infer_rgbd_zed import build_source
from infer_rgbd_femtobolt import HandDetector, landmarks_to_bbox, project, draw_skeleton


def wrist_depth(depth_m, uv, win=3):
    """Robust ZED depth (m) at a pixel: median of valid depth in a (2win+1)^2 box."""
    h, w = depth_m.shape[:2]
    u, v = int(round(uv[0])), int(round(uv[1]))
    if not (0 <= u < w and 0 <= v < h):
        return 0.0
    patch = depth_m[max(0, v - win):v + win + 1, max(0, u - win):u + win + 1]
    valid = patch[patch > 0]
    return float(np.median(valid)) if valid.size else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="mobrecon_ckpt/ds_anchor_gate/best.pt")
    ap.add_argument("--gt_z", default=0.0, type=float, help="tape-measured hand distance (m)")
    ap.add_argument("--label", default="run")
    ap.add_argument("--seconds", default=8.0, type=float)
    ap.add_argument("--max_frames", default=400, type=int)
    ap.add_argument("--conf_min", default=0.5, type=float, help="keep frames with anchor conf >= this")
    ap.add_argument("--save", default="out/zed_controlled")
    ap.add_argument("--margin", default=0.50, type=float)
    ap.add_argument("--hand_model", default=None)
    ap.add_argument("--source", default="zed")
    ap.add_argument("--svo", default=None)
    ap.add_argument("--zed_resolution", default="HD720")
    ap.add_argument("--zed_fps", default=30, type=int)
    ap.add_argument("--zed_depth_mode", default="NEURAL")
    ap.add_argument("--zed_min_depth", default=0.3, type=float)
    ap.add_argument("--zed_max_depth", default=2.0, type=float)
    args = ap.parse_args()

    os.makedirs(args.save, exist_ok=True)
    est = DualStreamHandPose(args.ckpt)
    src = build_source(args)
    detector = HandDetector(model_path=args.hand_model)

    rows = []
    n_total = n_hand = 0
    t0 = time.time()
    print(f"[capture:{args.label}] gt_z={args.gt_z*100:.1f}cm -- hold the hand STILL "
          f"at the measured distance for {args.seconds:.0f}s ...")
    try:
        while n_total < args.max_frames and (time.time() - t0) < args.seconds:
            frame = src.read()
            if frame is None:
                break
            bgr, depth_m, K = frame
            n_total += 1
            pts = detector.detect(bgr)
            if pts is None:
                continue
            n_hand += 1
            bbox = landmarks_to_bbox(pts, bgr.shape[:2], margin_frac=args.margin)
            inp, rgb_crop, dn, depth_med, root_anchor, anchor_valid = est.preprocess(
                bgr, depth_m, bbox, K)
            abs_j, root, rel = est.infer(inp, depth_med, root_anchor, anchor_valid)
            zw = wrist_depth(depth_m, pts[0])           # ZED depth at MediaPipe wrist px
            rows.append(dict(
                conf=float(anchor_valid), root_z=float(root[2]),
                anchor_z=float(root_anchor[2]), zed_wrist_z=zw,
                depth_med=depth_med, cov=float((dn != 0).mean()),
            ))
    finally:
        src.close()
        detector.close()

    # save raw rows
    csv_path = os.path.join(args.save, f"{args.label}.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader(); wr.writerows(rows)

    kept = [r for r in rows if r["conf"] >= args.conf_min and r["zed_wrist_z"] > 0]
    print(f"\n[capture:{args.label}] grabbed={n_total} hand={n_hand} "
          f"kept(conf>={args.conf_min} & valid wrist depth)={len(kept)} -> {csv_path}")
    if not kept:
        print("[capture] no usable frames. Hold a clear, still hand at the measured distance.")
        return

    def arr(k): return np.array([r[k] for r in kept], np.float32)
    root_z, anchor_z, zed_z = arr("root_z"), arr("anchor_z"), arr("zed_wrist_z")

    def line(name, v):
        return f"  {name:16s}: mean {v.mean()*100:6.2f}  median {np.median(v)*100:6.2f}  std {v.std()*100:5.2f} cm"
    print("\n--- depths (cm) ---")
    print(line("pred root z", root_z))
    print(line("anchor z", anchor_z))
    print(line("ZED wrist z", zed_z))

    print("\n--- absolute z error (cm), signed = pred - reference ---")
    e_zed = root_z - zed_z
    print(f"  pred root  vs ZED wrist : mean {e_zed.mean()*100:+6.2f}  median {np.median(e_zed)*100:+6.2f}  "
          f"MAE {np.abs(e_zed).mean()*100:5.2f}  std {e_zed.std()*100:5.2f}")
    e_anc = anchor_z - zed_z
    print(f"  anchor     vs ZED wrist : mean {e_anc.mean()*100:+6.2f}  median {np.median(e_anc)*100:+6.2f}  "
          f"MAE {np.abs(e_anc).mean()*100:5.2f}")
    if args.gt_z > 0:
        e_gt = root_z - args.gt_z
        print(f"  pred root  vs GT {args.gt_z*100:.1f}cm: mean {e_gt.mean()*100:+6.2f}  "
              f"median {np.median(e_gt)*100:+6.2f}  MAE {np.abs(e_gt).mean()*100:5.2f}")
        e_gtz = zed_z - args.gt_z
        print(f"  ZED wrist  vs GT {args.gt_z*100:.1f}cm: mean {e_gtz.mean()*100:+6.2f}  "
              f"(how well the ZED sensor itself matches your tape)")
    print(f"\n  conf mean {arr('conf').mean():.2f}  coverage mean {arr('cov').mean()*100:.0f}%")


if __name__ == "__main__":
    main()
