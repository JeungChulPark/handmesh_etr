"""Quick numeric diagnostic for ZED dual-stream inference.

For each frame with a detected hand, prints the metric extents that localize a
collapsed-skeleton failure: relative-pose span (cm), learned scale, learned root,
depth median, anchor + confidence, and the projected 2D skeleton span (px) vs the
MediaPipe hand-bbox span (px). If rel-span is tiny -> pose/scale collapse; if
rel-span is sane but 2D-span << bbox -> root-Z/projection problem.
"""

import _bootstrap  # noqa: F401  (repo-root + sibling script imports)

import argparse, numpy as np, cv2
from infer_zed_dualstream import DualStreamHandPose
from infer_rgbd_zed import build_source
from infer_rgbd_femtobolt import HandDetector, landmarks_to_bbox, project


def span_cm(j):  # max extent across axes, cm
    return (j.max(0) - j.min(0)) * 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="mobrecon_ckpt/ds_anchor_gate/best.pt")
    ap.add_argument("--source", default="zed")
    ap.add_argument("--svo", default=None)
    ap.add_argument("--path", default="captures")
    ap.add_argument("--cam_index", default=0, type=int)
    ap.add_argument("--rgb_only", action="store_true")
    ap.add_argument("--margin", default=0.50, type=float)
    ap.add_argument("--norm_scale", default=0.1, type=float)
    ap.add_argument("--hand_model", default=None)
    ap.add_argument("--zed_resolution", default="HD720")
    ap.add_argument("--zed_fps", default=30, type=int)
    ap.add_argument("--zed_depth_mode", default="NEURAL")
    ap.add_argument("--zed_min_depth", default=0.3, type=float)
    ap.add_argument("--zed_max_depth", default=2.0, type=float)
    ap.add_argument("--n", default=40, type=int, help="frames with a hand to log")
    args = ap.parse_args()

    est = DualStreamHandPose(args.ckpt, rgb_only=args.rgb_only, norm_scale=args.norm_scale)
    src = build_source(args)
    det = HandDetector(model_path=args.hand_model)
    print(f"{'i':>3} {'relXYZ(cm)':>22} {'scale':>6} {'rootZ':>6} {'med':>6} "
          f"{'aZ':>6} {'conf':>5} {'2Dpx':>6} {'bboxpx':>7} {'absZ':>6}")
    logged = 0
    while logged < args.n:
        frame = src.read()
        if frame is None:
            print("[stream ended]"); break
        bgr, depth_m, K = frame
        pts = det.detect(bgr)
        if pts is None:
            continue
        bbox = landmarks_to_bbox(pts, bgr.shape[:2], margin_frac=args.margin)
        inp, rgb_crop, dn, med, ra, va = est.preprocess(bgr, depth_m, bbox, K)
        abs_j, root, rel = est.infer(inp, med, ra, va)
        # learned scale: recover from abs = scale*rel + root  ->  not returned; infer from model
        uv = project(abs_j, K)
        d2 = float(np.linalg.norm(uv.max(0) - uv.min(0)))
        bb = float(np.hypot(bbox[2], bbox[3]))
        rcm = span_cm(rel)
        print(f"{logged:3d} "
              f"[{rcm[0]:5.1f} {rcm[1]:5.1f} {rcm[2]:5.1f}] "
              f"{'':1} {root[2]:6.3f}*  {med*100:5.1f} "
              f"{ra[2] if ra is not None else 0:6.3f} {float(va):5.2f} "
              f"{d2:6.1f} {bb:7.1f} {abs_j[:,2].mean()*100:5.1f}")
        logged += 1
    src.close(); det.close()
    print("[done] '*'=rootZ(m); relXYZ is the metric span of the relative pose; "
          "2Dpx=projected skeleton span; bboxpx=MediaPipe hand bbox span.")


if __name__ == "__main__":
    main()
