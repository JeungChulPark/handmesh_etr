"""Generate pseudo-3D ground truth for iPhone RGB-D captures (Option A: 2D+LiDAR lift).

Pipeline (no rig, no manual labels):

    cap_NNNN_{rgb.png, depth.f32, json}             (Unity RgbdRecorder)
        -> uprightness rotation (shared k over the set)
        -> MediaPipe 21 px landmarks                (better than our model on free hands)
        -> lift each landmark to 3D camera-metres via LiDAR depth + intrinsics
        -> per-joint validity mask (occluded / depth-outlier joints -> valid=0)
        -> surface->joint-centre offset along the viewing ray
        => cap_NNNN_gt.json   { joints3d_cam[21,3], joint_valid[21], bbox[4], K[3,3], rot_k }
        +  cap_NNNN_gtviz.png (RED lifted-GT skeleton vs GREEN MediaPipe 2D, for QA)

The validity mask is the key idea: we do NOT try to reconstruct occluded joints
perfectly. A joint whose LiDAR depth is missing or inconsistent with its bone is
flagged invalid, and the masked training loss (`joint_valid`) simply skips it.
So the GT is "sparse but trustworthy" rather than "dense but wrong".

Joints with reliable depth are exactly the free-hand spread fingers MediaPipe nails
and our model cramps -> training toward them un-cramps the pose backbone.

Usage:
    python make_iphone_gt.py --dir rgbd_captures                 # auto rotation
    python make_iphone_gt.py --dir rgbd_captures_03 --rotate 0   # force k=0
    python make_iphone_gt.py --dir rgbd_captures --no_viz        # skip QA pngs
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

from infer_rgbd_captures import load_capture, rotate_frame, vote_rotation
from infer_rgbd_femtobolt import (HAND_BONES, HandDetector, RGBDHandPose,
                                   landmarks_to_bbox, project)

# Anatomy: longest hand bone (proximal phalanx / metacarpal span) is ~5 cm. A
# lifted bone longer than this almost always means the distal joint's depth fell
# on a different surface (occlusion bleed) -> flag it invalid.
MAX_BONE_M = 0.075
# A frontally-presented hand spans only ~8 cm in depth; a joint whose z is far
# outside the hand-depth band is reading background / another finger.
MAX_Z_DEV_M = 0.12
# Sensor sees skin; push the lifted joint ~one capsule radius away from camera
# to reach the joint centre (same constant used in recover_root_aligned).
SURFACE_OFFSET_M = 0.010


def backproject(u, v, z, K):
    """Pixel + metric depth -> 3D camera-space point (metres)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.array([(u - cx) / fx * z, (v - cy) / fy * z, z], np.float32)


def lift_landmarks(mp_uv, depth_m, K, win=2):
    """Lift 21 MediaPipe pixel landmarks to 3D camera-metres + a per-joint valid mask.

    Returns (joints3d[21,3] float32, valid[21] float32). Invalid joints keep a
    best-effort position (so the skeleton still draws) but valid=0 so the loss
    skips them.
    """
    z = np.array([RGBDHandPose._sample_depth(depth_m, mp_uv[i, 0], mp_uv[i, 1], win)
                  for i in range(21)], np.float32)
    valid = (z > 0).astype(np.float32)

    # Robust hand-depth band: median over the joints that DID get depth.
    zv = z[valid > 0]
    z_med = float(np.median(zv)) if zv.size else 0.0

    # Fill missing joints with the hand-median z so back-projection stays in-frame
    # (position is only a placeholder; valid=0 already excludes them from the loss).
    z_filled = np.where(valid > 0, z, z_med)
    joints = np.stack([backproject(mp_uv[i, 0], mp_uv[i, 1], z_filled[i], K)
                       for i in range(21)], 0)

    # --- outlier rejection on the joints that claimed valid depth ---
    # (1) z far outside the hand band -> background / finger bleed.
    if z_med > 0:
        far = np.abs(z - z_med) > MAX_Z_DEV_M
        valid[far & (valid > 0)] = 0.0
    # (2) implausibly long lifted bone -> distal joint depth is wrong.
    for p, c in HAND_BONES:
        if valid[p] > 0 and valid[c] > 0:
            if np.linalg.norm(joints[c] - joints[p]) > MAX_BONE_M:
                valid[c] = 0.0

    # Re-place every INVALID joint at the hand-median depth so the stored GT stays
    # clean (a depth-outlier joint kept its background z otherwise, blowing up the
    # 3D extent). They are masked out of the loss anyway; this only keeps geometry
    # sane for bbox / root-alignment / inspection.
    for i in range(21):
        if valid[i] == 0 and z_med > 0:
            joints[i] = backproject(mp_uv[i, 0], mp_uv[i, 1], z_med, K)

    # surface -> joint-centre offset along +z (away from camera) for valid joints.
    joints[valid > 0, 2] += SURFACE_OFFSET_M
    return joints.astype(np.float32), valid


def draw_gt(bgr, gt_uv, mp_uv, valid, bbox):
    """QA overlay: RED = lifted-GT reprojection, GREEN = raw MediaPipe 2D."""
    ov = bgr.copy()
    if bbox is not None:
        x, y, s, _ = [int(v) for v in bbox]
        cv2.rectangle(ov, (x, y), (x + s, y + s), (0, 220, 220), 1)
    for p, c in HAND_BONES:
        cv2.line(ov, tuple(mp_uv[p].astype(int)), tuple(mp_uv[c].astype(int)), (0, 200, 0), 1)
    for i in range(21):
        cv2.circle(ov, tuple(mp_uv[i].astype(int)), 2, (0, 200, 0), -1)
        col = (40, 40, 230) if valid[i] > 0 else (120, 120, 120)  # grey = masked
        cv2.circle(ov, tuple(gt_uv[i].astype(int)), 3, col, -1)
    nval = int(valid.sum())
    cv2.putText(ov, f"valid {nval}/21  RED=GT GREEN=MP grey=masked", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return ov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="rgbd_captures", help="folder of cap_*_rgb.png")
    ap.add_argument("--rotate", type=int, default=None, choices=[0, 1, 2, 3],
                    help="force k*90 CW rotation; default = uprightness vote")
    ap.add_argument("--win", type=int, default=2, help="depth sampling half-window (px)")
    ap.add_argument("--no_viz", action="store_true", help="skip QA overlay pngs")
    ap.add_argument("--hand_model", default=None)
    args = ap.parse_args()

    stems = sorted(s[:-len("_rgb.png")] for s in glob.glob(os.path.join(args.dir, "cap_*_rgb.png")))
    if not stems:
        print(f"[make_gt] no cap_*_rgb.png under {args.dir}"); return
    det = HandDetector(model_path=args.hand_model)

    k = args.rotate if args.rotate is not None else vote_rotation(stems, det)
    print(f"[make_gt] {len(stems)} captures, rotation k={k} ({k*90} CW)")

    n_hand = 0
    valid_hist = []
    for stem in stems:
        cap = load_capture(stem)
        if cap is None:
            continue
        bgr0, depth0, K0 = cap
        bgr, depth, K = rotate_frame(bgr0, depth0, K0, k)
        pts = det.detect(bgr)
        if pts is None:
            continue
        n_hand += 1
        bbox = landmarks_to_bbox(pts, bgr.shape[:2])
        joints3d, valid = lift_landmarks(pts, depth, K, win=args.win)
        valid_hist.append(int(valid.sum()))

        out = {
            "joints3d_cam": joints3d.tolist(),     # [21,3] metres, camera space
            "joint_valid": valid.tolist(),         # [21] 1=supervised
            "bbox": [float(v) for v in bbox],      # [x,y,side,side] in rotated frame
            "K": K.astype(float).tolist(),         # 3x3 rotated intrinsics
            "rot_k": int(k),
        }
        with open(stem + "_gt.json", "w") as f:
            json.dump(out, f)

        if not args.no_viz:
            gt_uv = project(joints3d, K)
            cv2.imwrite(stem + "_gtviz.png", draw_gt(bgr, gt_uv, pts, valid, bbox))

    det.close()
    vh = np.array(valid_hist) if valid_hist else np.zeros(1)
    print(f"[make_gt] hands: {n_hand}/{len(stems)} frames | "
          f"valid joints/frame: mean {vh.mean():.1f} median {int(np.median(vh))} "
          f"min {int(vh.min())} max {int(vh.max())}")
    print(f"[make_gt] wrote {n_hand} *_gt.json"
          + ("" if args.no_viz else " + *_gtviz.png (inspect a few for quality)"))


if __name__ == "__main__":
    main()
