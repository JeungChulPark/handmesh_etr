"""Headless ZED validation for the geo_anchor dual-stream deploy path.

Grabs live ZED frames for a bounded window, runs DualStreamHandPose (ds_anchor_gate),
and for every frame with a detected hand records the deploy-critical signals:
  depth_med, anchor (x,y,z) + confidence, learned root, abs wrist z, depth coverage.
Saves a few overlay PNGs and prints summary stats. No GUI, no infinite loop.

The point: confirm that on REAL ZED passive-stereo depth (vs training lidar_sim)
the geometric_root_anchor produces sane confidence and the learned root tracks it.

Run (hold a hand ~40-60cm in front of the ZED):
  python scripts/diag/diag_zed_anchor.py --ckpt mobrecon_ckpt/ds_anchor_gate/best.pt --seconds 15
"""

import _bootstrap  # noqa: F401  (repo-root + sibling script imports)

import os
import time
import argparse

import numpy as np
import cv2

from infer_zed_dualstream import DualStreamHandPose
from infer_rgbd_zed import build_source
from infer_rgbd_femtobolt import HandDetector, landmarks_to_bbox, project, draw_skeleton


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="mobrecon_ckpt/ds_anchor_gate/best.pt")
    ap.add_argument("--seconds", default=15.0, type=float, help="capture window")
    ap.add_argument("--max_frames", default=400, type=int)
    ap.add_argument("--save", default="out/zed_anchor_test")
    ap.add_argument("--save_every", default=10, type=int, help="save overlay every Nth hand frame")
    ap.add_argument("--margin", default=0.50, type=float)
    ap.add_argument("--hand_model", default=None)
    # ZED knobs (forwarded to build_source)
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

    rows = []          # per-hand-frame diagnostics
    saved = 0
    n_total = 0
    n_hand = 0
    t0 = time.time()
    print(f"[diag] capturing up to {args.seconds:.0f}s / {args.max_frames} frames -- "
          f"hold a hand in front of the ZED ...")
    try:
        while n_total < args.max_frames and (time.time() - t0) < args.seconds:
            frame = src.read()
            if frame is None:
                print("[diag] stream ended")
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
            cov = float((dn != 0).mean())
            rows.append(dict(
                depth_med=depth_med, conf=float(anchor_valid),
                anchor_z=float(root_anchor[2]), root_z=float(root[2]),
                abs_wrist_z=float(abs_j[0, 2]), cov=cov,
                resid_z=float(root[2] - root_anchor[2]),
            ))
            if n_hand % args.save_every == 1 and saved < 20:
                uv = project(abs_j, K)
                ov = draw_skeleton(bgr, uv, bbox)
                cv2.putText(ov, f"conf={float(anchor_valid):.2f} med={depth_med*100:.1f}cm "
                            f"anchorZ={root_anchor[2]*100:.1f} rootZ={root[2]*100:.1f}cm cov={cov*100:.0f}%",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                cv2.imwrite(os.path.join(args.save, f"hand_{saved:02d}.png"), ov)
                cv2.imwrite(os.path.join(args.save, f"depth_{saved:02d}.png"),
                            cv2.applyColorMap(((dn + 1) * 127).clip(0, 255).astype(np.uint8),
                                              cv2.COLORMAP_JET))
                saved += 1
    finally:
        src.close()
        detector.close()

    print(f"\n[diag] frames grabbed={n_total}  with-hand={n_hand}  overlays saved={saved} -> {args.save}")
    if not rows:
        print("[diag] NO hand frames captured. Re-run with a hand in view (40-60cm).")
        return

    def stats(key):
        v = np.array([r[key] for r in rows], np.float32)
        return f"{v.mean():7.3f} | {np.median(v):7.3f} | {v.min():7.3f} | {v.max():7.3f}"

    print(f"\n  metric         |   mean  | median |   min  |   max")
    print(f"  ---------------+---------+--------+--------+--------")
    for k in ["depth_med", "conf", "anchor_z", "root_z", "resid_z", "abs_wrist_z", "cov"]:
        print(f"  {k:14s} | {stats(k)}")

    conf = np.array([r["conf"] for r in rows], np.float32)
    cov = np.array([r["cov"] for r in rows], np.float32)
    print(f"\n[diag] anchor confidence: >0.5 on {(conf > 0.5).mean()*100:.0f}% of hand frames, "
          f">0.8 on {(conf > 0.8).mean()*100:.0f}%")
    print(f"[diag] depth coverage in crop: mean {cov.mean()*100:.0f}% "
          f"(low coverage => ZED stereo dropped depth on the hand)")
    resid = np.array([r["resid_z"] for r in rows], np.float32)
    print(f"[diag] learned root z - anchor z (residual): mean {resid.mean()*1000:.1f}mm "
          f"median {np.median(resid)*1000:.1f}mm  (training residual was ~ -20mm hand half-thickness)")


if __name__ == "__main__":
    main()
