"""Deploy inference for the Z-only lifter (MediaPipe 2D + LiDAR depth -> metric 3D).

Flow per frame (exactly the deployment pipeline):
  1. MediaPipe(RGB) -> 21 (u,v) pixel landmarks               [HandDetector]
  2. sample the aligned depth at each (u,v) -> z_sampled + valid, z_ref=median
  3. feature [21,4] = (u_norm, v_norm, z_use - z_ref, valid)  (u,v normalised by W,H)
  4. ZLifter MLP -> per-joint depth correction dZ
  5. reconstruct metric 3D:  Z = z_use + dZ,  X=(u-cx)/fx*Z,  Y=(v-cy)/fy*Z
Because X,Y are rebuilt from MediaPipe's (u,v), the 2D overlay is exactly the
MediaPipe skeleton (the 2D shape is kept); the model's contribution is metric Z.
Joints are coloured by depth so the recovered 3D is visible; abs joints + per-joint
depth are written to cap_NNNN_pred_<tag>.json.

Runs on the iPhone RgbdRecorder captures (same loader as infer_rgbd_captures).

    python scripts/infer/infer_zlifter.py --dir rgbd_captures --ckpt mobrecon_ckpt/zlifter_dexycb/best.pt
"""

import _bootstrap  # noqa: F401  (repo-root + sibling script imports)

import os
import re
import glob
import json
import argparse

import numpy as np
import cv2
import torch

from train_zlifter import ZLifter
from infer_rgbd_captures import load_capture, rotate_frame, vote_rotation
from infer_rgbd_femtobolt import HandDetector, project, HAND_BONES, _FINGER_COLORS


def sample_depth(dm, u, v, win=3):
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < dm.shape[1] and 0 <= vi < dm.shape[0]):
        return 0.0, 0.0
    p = dm[max(0, vi - win):vi + win + 1, max(0, ui - win):ui + win + 1]
    nz = p[p > 0]
    return (float(np.median(nz)), 1.0) if nz.size else (0.0, 0.0)


def draw_depth_skeleton(bgr, uv, z, bbox=None):
    """Skeleton at MediaPipe uv, joints coloured by metric depth (near=red far=blue)."""
    out = bgr.copy()
    for a, b in HAND_BONES:
        cv2.line(out, tuple(uv[a].astype(int)), tuple(uv[b].astype(int)), (200, 200, 200), 2)
    zmin, zmax = float(np.min(z)), float(np.max(z) + 1e-6)
    for j, (u, v) in enumerate(uv.astype(int)):
        t = (z[j] - zmin) / (zmax - zmin)                    # 0=near 1=far
        col = (int(255 * t), 60, int(255 * (1 - t)))         # BGR: near red, far blue
        cv2.circle(out, (u, v), 5, col, -1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="rgbd_captures")
    ap.add_argument("--ckpt", default="mobrecon_ckpt/zlifter_dexycb/best.pt")
    ap.add_argument("--tag", default="zlift")
    ap.add_argument("--rotate", default="auto", help="'auto' or 0/90/180/270 deg CW")
    ap.add_argument("--limit", default=0, type=int)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ZLifter().to(device).eval()
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state.get("model_state_dict", state))
    print(f"[zlifter] loaded {args.ckpt} (epoch {state.get('epoch','?')}) device={device}")

    base_re = re.compile(r"^cap_\d+\.json$")
    stems = sorted(s[:-5] for s in glob.glob(os.path.join(args.dir, "cap_*.json"))
                   if base_re.match(os.path.basename(s)))
    if args.limit > 0:
        stems = stems[:args.limit]
    if not stems:
        print(f"[main] no captures in {args.dir}"); return
    print(f"[main] {len(stems)} captures in {args.dir}")

    det = HandDetector()
    k = vote_rotation(stems, det) if args.rotate == "auto" else (int(args.rotate) // 90) % 4
    suffix = f"_pred_{args.tag}"
    n_det = 0
    for stem in stems:
        name = os.path.basename(stem)
        cap = load_capture(stem)
        if cap is None:
            continue
        bgr, depth, K = rotate_frame(*cap, k)
        H, W = bgr.shape[:2]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        pts = det.detect(bgr)
        rec = {"name": name, "detected": False, "rotation_deg_cw": k * 90}
        if pts is not None:
            uv = np.asarray(pts, np.float32)[:, :2]
            zs = np.zeros(21, np.float32); zv = np.zeros(21, np.float32)
            for j in range(21):
                zs[j], zv[j] = sample_depth(depth, uv[j, 0], uv[j, 1])
            z_ref = float(np.median(zs[zv > 0])) if zv.any() else 0.5
            z_use = np.where(zv > 0, zs, z_ref).astype(np.float32)
            feat = np.stack([uv[:, 0] / W * 2 - 1, uv[:, 1] / H * 2 - 1,
                             z_use - z_ref, zv], axis=1).astype(np.float32)
            with torch.no_grad():
                dZ = model(torch.from_numpy(feat).unsqueeze(0).to(device))[0].cpu().numpy()
            Z = z_use + dZ                                    # metric depth per joint
            X = (uv[:, 0] - cx) / fx * Z
            Y = (uv[:, 1] - cy) / fy * Z
            abs_j = np.stack([X, Y, Z], axis=1).astype(np.float32)   # (21,3) camera metres

            overlay = draw_depth_skeleton(bgr, uv, Z)
            cv2.putText(overlay, f"[zlifter] root_z={Z[0]*100:5.1f}cm  "
                        f"depth_span={np.ptp(Z)*100:4.1f}cm  det_depth={int(zv.sum())}/21",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            rec.update(detected=True, root_z_m=round(float(Z[0]), 4),
                       abs_joints=[[round(float(v), 4) for v in p] for p in abs_j],
                       uv=[[round(float(v), 1) for v in p] for p in uv])
            n_det += 1
        else:
            overlay = bgr.copy()
            cv2.putText(overlay, "no hand (MediaPipe miss)", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite(stem + suffix + ".png", overlay)
        json.dump(rec, open(stem + suffix + ".json", "w"), indent=2)

    det.close()
    print(f"[done] {n_det}/{len(stems)} frames had a hand. Wrote *{suffix}.png/json to {args.dir}")


if __name__ == "__main__":
    main()
