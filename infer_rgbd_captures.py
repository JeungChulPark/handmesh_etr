"""Offline inference on iPhone RGB-D captures (the RgbdRecorder dump).

Reads the cap_NNNN_{rgb.png,depth.f32,json} triples written by the Unity
`RgbdRecorder`, runs the DUAL-STREAM model (ds_anchor_gate/best.pt) with the exact
training preprocessing (reusing DualStreamHandPose), and writes results back into
the SAME folder:
  cap_NNNN_pred.png  -- RGB with the 21-joint skeleton + bbox + distance overlay.
  cap_NNNN_pred.json -- detected flag, rotation, abs joints (camera metres), root,
                        2D projection, depth_med, anchor confidence.

The captures are raw camera-SENSOR frames, so the hand is rotated ~90 deg from
upright (see SensorRotation in DualStreamHandProvider). We auto-detect the global
rotation by voting MediaPipe detections over a few frames, then apply it to RGB +
depth + intrinsics consistently so the model sees an upright hand and the
projection stays correct.

Run:
  python infer_rgbd_captures.py                       # dir=rgbd_captures, best.pt
  python infer_rgbd_captures.py --dir rgbd_captures --rotate auto
  python infer_rgbd_captures.py --rotate 90 --limit 20
"""

import os
import re
import csv
import glob
import json
import argparse

import numpy as np
import cv2

from infer_zed_dualstream import DualStreamHandPose
from infer_rgbd_femtobolt import (HandDetector, landmarks_to_bbox, project,
                                  draw_skeleton, RGBDHandPose)


def rotate_frame(bgr, depth, K, k):
    """Rotate RGB + (RGB-resolution) depth + intrinsics by k*90 deg CLOCKWISE."""
    if k % 4 == 0:
        return bgr, depth, K.astype(np.float32).copy()
    H, W = bgr.shape[:2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if k == 1:      # 90 CW
        bgr2 = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
        dep2 = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)
        nfx, nfy, ncx, ncy = fy, fx, (H - 1) - cy, cx
    elif k == 2:    # 180
        bgr2 = cv2.rotate(bgr, cv2.ROTATE_180)
        dep2 = cv2.rotate(depth, cv2.ROTATE_180)
        nfx, nfy, ncx, ncy = fx, fy, (W - 1) - cx, (H - 1) - cy
    else:           # k == 3, 270 CW (= 90 CCW)
        bgr2 = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        dep2 = cv2.rotate(depth, cv2.ROTATE_90_COUNTERCLOCKWISE)
        nfx, nfy, ncx, ncy = fy, fx, cy, (W - 1) - cx
    K2 = np.array([[nfx, 0, ncx], [0, nfy, ncy], [0, 0, 1]], np.float32)
    return bgr2, dep2, K2


def load_capture(stem):
    """stem like '.../cap_0007' -> (bgr, depth_full[m, RGB-res], K) or None."""
    meta_p = stem + ".json"
    if not os.path.exists(meta_p):
        return None
    m = json.load(open(meta_p))
    rgb_p = os.path.join(os.path.dirname(stem), m["rgbFile"])
    bgr = cv2.imread(rgb_p, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    H, W = bgr.shape[:2]
    K = np.array([[m["fx"], 0, m["cx"]], [0, m["fy"], m["cy"]], [0, 0, 1]], np.float32)
    # intrinsics are at intrWidth/Height (= RGB res); scale if the PNG was downscaled.
    if m.get("intrWidth") and m["intrWidth"] != W:
        s = W / float(m["intrWidth"])
        K[0] *= s
        K[1] *= s
    depth = None
    if m.get("depthFile"):
        dp = os.path.join(os.path.dirname(stem), m["depthFile"])
        if os.path.exists(dp):
            d = np.fromfile(dp, dtype="<f4").reshape(m["depthHeight"], m["depthWidth"])
            depth = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)  # -> RGB res
    if depth is None:
        depth = np.zeros((H, W), np.float32)
    return bgr, depth, K


def joint_pixel_anchor(mp_uv, depth_m, K, win=2):
    """Depth-only root anchor sampled AT the 21 hand-landmark pixels (not the
    crop's valid-depth centroid). This is the early-fusion idea ported into the
    dual-stream anchor: read depth where the HAND actually is, so a close-range,
    background-heavy crop can't pull the anchor toward the wall.

    Returns (anchor[3] camera-metres, conf in [0,1] = fraction of landmarks that
    had valid depth). conf=0 -> unusable (zeros), head falls back to its prior.
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    zs = [RGBDHandPose._sample_depth(depth_m, mp_uv[i, 0], mp_uv[i, 1], win)
          for i in range(len(mp_uv))]
    zs = [z for z in zs if z > 0]
    if not zs:
        return np.zeros(3, np.float32), np.float32(0.0)
    z0 = float(np.median(zs))                       # robust hand-surface depth
    u0, v0 = float(mp_uv[0, 0]), float(mp_uv[0, 1])  # wrist landmark for x,y dir
    anchor = np.array([(u0 - cx) / fx * z0, (v0 - cy) / fy * z0, z0], np.float32)
    conf = float(len(zs)) / float(len(mp_uv))
    return anchor, np.float32(conf)


def hand_band_mask(depth_m, z0, bbox, band=0.12, pad=0.15):
    """Hand mask = (valid depth within `band` metres of the hand-surface z0) inside a
    padded bbox. Reproduces the training `depth_mode="hand_masked"` (HO3D z-band /
    DexYCB seg) at DEPLOY, where no segmentation exists, using only the MediaPipe
    bbox + the joint-pixel hand depth. Excludes background AND inter-finger gaps
    (which a convex hull would wrongly include)."""
    H, W = depth_m.shape
    x, y, s, _ = bbox
    x0 = max(0, int(x - pad * s)); y0 = max(0, int(y - pad * s))
    x1 = min(W, int(x + s + pad * s)); y1 = min(H, int(y + s + pad * s))
    m = np.zeros((H, W), bool)
    reg = depth_m[y0:y1, x0:x1]
    m[y0:y1, x0:x1] = (reg > 0) & (np.abs(reg - z0) < band)
    return m


def vote_rotation(stems, detector, sample=12):
    """Pick the global k*90 CW rotation that makes the hand most UPRIGHT.

    Counting hand DETECTIONS is broken here: MediaPipe is rotation-invariant (it
    detects a hand at every k*90), so all rotations tie and the old tie-break wrongly
    forced k=1, rotating already-upright captures sideways. Instead we score how
    upright the detected hand is -- fingertips above the wrist (image y grows down,
    so wrist_y - tip_y > 0 means fingers are higher). The rotation with the most
    upright hand across the sampled frames wins. Override with --rotate if a session
    holds the hand fingers-down."""
    idxs = np.linspace(0, len(stems) - 1, min(sample, len(stems))).astype(int)
    rots = {1: cv2.ROTATE_90_CLOCKWISE, 2: cv2.ROTATE_180, 3: cv2.ROTATE_90_COUNTERCLOCKWISE}
    tips = [8, 12, 16, 20]                  # index/middle/ring/pinky tips (skip thumb)
    score = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    ndet = {0: 0, 1: 0, 2: 0, 3: 0}
    for i in idxs:
        cap = load_capture(stems[i])
        if cap is None:
            continue
        bgr, _, _ = cap
        for k in score:
            rb = bgr if k == 0 else cv2.rotate(bgr, rots[k])
            pts = detector.detect(rb)
            if pts is None:
                continue
            ndet[k] += 1
            pts = np.asarray(pts, np.float32)[:, :2]
            hs = float(np.linalg.norm(pts[9] - pts[0])) + 1e-6   # wrist->mid-MCP, scale
            score[k] += (float(pts[0, 1]) - float(pts[tips, 1].mean())) / hs
    cand = {k: score[k] for k in score if ndet[k] > 0}
    best = max(cand, key=lambda k: cand[k]) if cand else 0
    print(f"[rotate] uprightness {dict((k, round(score[k], 2)) for k in score)} "
          f"dets {ndet} -> k={best} ({best * 90} deg CW)")
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="rgbd_captures")
    ap.add_argument("--arch", default="dualstream",
                    choices=["dualstream", "earlyfusion"],
                    help="dualstream = ds_anchor_gate (learned root); earlyfusion = the model "
                         "that worked on ZED (rgbd_real_0613, geometric root recovery).")
    ap.add_argument("--ckpt", default=None,
                    help="default: ds_anchor_gate/best.pt (dualstream) or "
                         "rgbd_real_0613/best.pt (earlyfusion).")
    ap.add_argument("--tag", default=None,
                    help="output suffix: cap_NNNN_pred_<tag>.png. Default '' (ds) / 'ef'.")
    ap.add_argument("--rotate", default="auto",
                    help="'auto' (vote) or one of 0/90/180/270 (deg CW).")
    ap.add_argument("--anchor", default="centroid",
                    choices=["centroid", "joint"],
                    help="dual-stream root anchor: 'centroid' = original "
                         "geometric_root_anchor (valid-depth centroid of the crop); "
                         "'joint' = depth sampled at the 21 hand-landmark pixels "
                         "(experiment: robust to background in close-range crops).")
    ap.add_argument("--med", default="crop",
                    choices=["crop", "hand"],
                    help="dual-stream depth_med cue: 'crop' = median over the whole "
                         "crop (background-dominated when the hand is close); 'hand' = "
                         "median depth at the 21 hand-landmark pixels (experiment).")
    ap.add_argument("--mask", default="none",
                    choices=["none", "band"],
                    help="dual-stream: 'band' zeroes background depth via a hand "
                         "z-band mask BEFORE preprocess, so the depth channel, "
                         "depth_med AND the anchor all see hand-only depth "
                         "(reproduces the training hand_masked distribution).")
    ap.add_argument("--band", default=0.12, type=float,
                    help="hand z-band half-width (m) for --mask band.")
    ap.add_argument("--margin", default=0.50, type=float)
    ap.add_argument("--norm_scale", default=0.1, type=float)
    ap.add_argument("--limit", default=0, type=int, help="0 = all frames.")
    args = ap.parse_args()

    # only the base capture sidecars cap_NNNN.json (never the *_pred*.json we write)
    base_re = re.compile(r"^cap_\d+\.json$")
    stems = sorted(s[:-5] for s in glob.glob(os.path.join(args.dir, "cap_*.json"))
                   if base_re.match(os.path.basename(s)))
    if args.limit > 0:
        stems = stems[:args.limit]
    if not stems:
        print(f"[main] no captures in {args.dir}")
        return
    print(f"[main] {len(stems)} captures in {args.dir}")

    if args.arch == "dualstream":
        ckpt = args.ckpt or "mobrecon_ckpt/ds_anchor_gate/best.pt"
        est = DualStreamHandPose(ckpt, norm_scale=args.norm_scale)
        _auto = (("mask" if args.mask == "band" else "")
                 + ("ja" if args.anchor == "joint" else "")
                 + ("m" if args.med == "hand" else ""))
        tag = args.tag if args.tag is not None else _auto
    else:
        ckpt = args.ckpt or "mobrecon_ckpt/rgbd_real_0613/best.pt"
        est = RGBDHandPose(ckpt, norm_scale=args.norm_scale)
        tag = args.tag if args.tag is not None else "ef"
    suffix = f"_pred_{tag}" if tag else "_pred"
    detector = HandDetector()

    if args.rotate == "auto":
        k = vote_rotation(stems, detector)
    else:
        k = (int(args.rotate) // 90) % 4

    summary = []
    n_det = 0
    for stem in stems:
        name = os.path.basename(stem)
        cap = load_capture(stem)
        if cap is None:
            print(f"[skip] {name}: load failed")
            continue
        bgr0, depth0, K0 = cap
        bgr, depth, K = rotate_frame(bgr0, depth0, K0, k)

        rec = {"name": name, "arch": args.arch, "rotation_deg_cw": k * 90, "detected": False}
        pts = detector.detect(bgr)
        abs_j = None
        depth_med = float("nan")
        if pts is not None:
            bbox = landmarks_to_bbox(pts, bgr.shape[:2], margin_frac=args.margin)
            if args.arch == "dualstream":
                depth_in = depth
                if args.mask == "band":     # hand-only depth (reproduce training mask)
                    jz, jv0 = joint_pixel_anchor(pts, depth, K)
                    if jv0 > 0:
                        m = hand_band_mask(depth, float(jz[2]), bbox, band=args.band)
                        depth_in = np.where(m, depth, 0.0).astype(np.float32)
                inp, _, dn, depth_med, root_anchor, anchor_valid = est.preprocess(
                    bgr, depth_in, bbox, K)
                if args.anchor == "joint" or args.med == "hand":
                    ja, jv = joint_pixel_anchor(pts, depth, K)  # hand-surface depth
                    if args.anchor == "joint":
                        root_anchor, anchor_valid = ja, jv
                    if args.med == "hand" and jv > 0:
                        depth_med = float(ja[2])  # median depth AT the hand
                abs_j, root, rel = est.infer(inp, depth_med, root_anchor, anchor_valid)
                rec["anchor_valid"] = round(float(anchor_valid), 3)
                rec["anchor"] = [round(float(v), 4) for v in root_anchor]
            else:  # earlyfusion: model gives root-relative pose; recover root from depth
                inp, _, dn = est.preprocess(bgr, depth, bbox, K)
                rel = est.infer(inp)
                abs_j = RGBDHandPose.recover_root_aligned(rel, pts, depth, K)
                if abs_j is None:           # no valid depth at the hand -> skip placement
                    abs_j = rel
                root = abs_j[0]

        if abs_j is not None:
            uv = project(abs_j, K)
            overlay = draw_skeleton(bgr, uv, bbox)          # colored bones + white joints = MODEL
            if pts is not None:                 # MediaPipe landmarks = cyan reference rings
                for u, v in np.asarray(pts, np.float32)[:, :2]:
                    cv2.circle(overlay, (int(u), int(v)), 6, (255, 255, 0), 2)
            cv2.putText(overlay, f"[{args.arch}] z={abs_j[0,2]*100:5.1f}cm  "
                        f"med={depth_med*100:5.1f}cm", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(overlay, "colored = model (depth-projected)   cyan ring = MediaPipe",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            rec.update(detected=True, depth_med_m=round(float(depth_med), 4),
                       root_z_m=round(float(abs_j[0, 2]), 4),
                       root=[round(float(v), 4) for v in root],
                       abs_joints=[[round(float(v), 4) for v in p] for p in abs_j],
                       uv=[[round(float(v), 1) for v in p] for p in uv])
            n_det += 1
        else:
            overlay = bgr.copy()
            cv2.putText(overlay, "no hand", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imwrite(stem + suffix + ".png", overlay)
        json.dump(rec, open(stem + suffix + ".json", "w"), indent=2)
        summary.append(rec)
        print(f"[{name}] " + ("hand  z={:.1f}cm med={:.1f}cm".format(
            rec["root_z_m"] * 100, rec["depth_med_m"] * 100) if rec["detected"] else "no hand"))

    # session summary CSV
    csv_name = f"predictions_summary{('_' + tag) if tag else ''}.csv"
    with open(os.path.join(args.dir, csv_name), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["name", "detected", "rotation_deg_cw", "root_z_m", "depth_med_m", "anchor_valid"])
        for r in summary:
            wr.writerow([r["name"], r["detected"], r["rotation_deg_cw"],
                         r.get("root_z_m", ""), r.get("depth_med_m", ""), r.get("anchor_valid", "")])

    detector.close()
    print(f"\n[done] {args.arch}: {n_det}/{len(stems)} frames had a hand. "
          f"Wrote *{suffix}.png/json + {csv_name} to {args.dir}")


if __name__ == "__main__":
    main()
