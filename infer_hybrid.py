"""Deploy inference for the BACKBONE-PRIMARY hybrid on iPhone captures.

  MediaPipe(RGB)=uv  +  LiDAR depth  +  MobRecon(hand crop)=p_cnn
     -> head Δp  ->  p = p_cnn + Δp   (metric 3D, root-relative)

Out-of-domain check: rgbd_real_0613's backbone was weak on iPhone; this shows
whether the MediaPipe-2D + depth refinement RESCUES it. Overlays, for the same
frame:
  * cyan rings   = MediaPipe 2D (the accurate reference)
  * grey skeleton = backbone-alone p_cnn reprojected  (where pixels alone land)
  * colour skeleton (depth-tinted joints) = HYBRID reprojected  (after refinement)
If the backbone is confused, the grey skeleton drifts off the cyan rings while the
colour (hybrid) one snaps back onto them.

3D is placed at the LiDAR wrist depth for reprojection (root-relative model output).

    python infer_hybrid.py --dir rgbd_captures --ckpt mobrecon_ckpt/hybrid_bb/best.pt
"""
import os
import re
import glob
import json
import argparse

import numpy as np
import cv2
import torch

from utils import load_cfg
from train_hybrid import HybridLifter
from train_zlifter import build_hand_crop
from infer_rgbd_captures import load_capture, rotate_frame, vote_rotation
from infer_rgbd_femtobolt import HandDetector, project, draw_skeleton, HAND_BONES


def sample_depth(dm, u, v, win=3):
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < dm.shape[1] and 0 <= vi < dm.shape[0]):
        return 0.0, 0.0
    p = dm[max(0, vi - win):vi + win + 1, max(0, ui - win):ui + win + 1]
    nz = p[p > 0]
    return (float(np.median(nz)), 1.0) if nz.size else (0.0, 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="rgbd_captures")
    ap.add_argument("--ckpt", default="mobrecon_ckpt/hybrid_bb/best.pt")
    ap.add_argument("--cfg", default="./configs_rgbd.yaml")
    ap.add_argument("--tag", default="hybrid")
    ap.add_argument("--backbone", default="densestack", choices=["densestack", "fastvit"])
    ap.add_argument("--fastvit_variant", default="fastvit_t8")
    ap.add_argument("--mode", default="backbone", choices=["backbone", "locked"],
                    help="must match the checkpoint's training mode")
    ap.add_argument("--rotate", default="auto")
    ap.add_argument("--limit", default=0, type=int)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_cfg(args.cfg)
    model = HybridLifter(cfg, mode=args.mode, backbone=args.backbone,
                         fastvit_variant=args.fastvit_variant).to(device).eval()
    st = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(st.get("model_state_dict", st))
    print(f"[hybrid] loaded {args.ckpt} (epoch {st.get('epoch','?')})")

    base_re = re.compile(r"^cap_\d+\.json$")
    stems = sorted(s[:-5] for s in glob.glob(os.path.join(args.dir, "cap_*.json"))
                   if base_re.match(os.path.basename(s)))
    if args.limit > 0:
        stems = stems[:args.limit]
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
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        rec = {"name": name, "detected": False}
        pts = det.detect(bgr)
        if pts is not None:
            uv = np.asarray(pts, np.float32)[:, :2]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            zs = np.zeros(21, np.float32); zv = np.zeros(21, np.float32)
            for j in range(21):
                zs[j], zv[j] = sample_depth(depth, uv[j, 0], uv[j, 1])
            z_ref = float(np.median(zs[zv > 0])) if zv.any() else 0.5
            z_use = np.where(zv > 0, zs, z_ref).astype(np.float32)
            feat = np.stack([uv[:, 0] / bgr.shape[1] * 2 - 1, uv[:, 1] / bgr.shape[0] * 2 - 1,
                             z_use - z_ref, zv], axis=1).astype(np.float32)
            crop = build_hand_crop(rgb, depth, uv)
            with torch.no_grad():
                p, p_cnn, p_geom = model(
                    torch.from_numpy(feat).unsqueeze(0).to(device),
                    crop.unsqueeze(0).to(device),
                    torch.from_numpy(uv).unsqueeze(0).to(device),
                    torch.from_numpy(z_use).unsqueeze(0).to(device),
                    torch.tensor([[fx, fy, cx, cy]], dtype=torch.float32).to(device))
            p = p[0].cpu().numpy(); p_cnn = p_cnn[0].cpu().numpy()
            # place root at the LiDAR wrist depth for reprojection
            zw = zs[0] if zv[0] > 0 else z_ref
            root = np.array([(uv[0, 0] - cx) / fx * zw, (uv[0, 1] - cy) / fy * zw, zw], np.float32)
            abs_h = p + root; abs_c = (p_cnn - p_cnn[0]) + root
            uv_h = project(abs_h, K); uv_c = project(abs_c, K)

            overlay = bgr.copy()
            for a, b in HAND_BONES:                          # backbone-alone = grey
                cv2.line(overlay, tuple(uv_c[a].astype(int)), tuple(uv_c[b].astype(int)), (150, 150, 150), 1)
            overlay = draw_skeleton(overlay, uv_h)           # hybrid = colour
            for u, v in uv:                                  # MediaPipe = cyan rings
                cv2.circle(overlay, (int(u), int(v)), 6, (255, 255, 0), 2)
            cv2.putText(overlay, f"[hybrid_bb] root_z={abs_h[0,2]*100:5.1f}cm  "
                        f"span={np.ptp(abs_h[:,2])*100:4.1f}cm", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(overlay, "cyan=MediaPipe  grey=backbone-only  colour=hybrid",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            rec.update(detected=True, root_z_m=round(float(abs_h[0, 2]), 4),
                       abs_joints=[[round(float(v), 4) for v in q] for q in abs_h])
            n_det += 1
        else:
            overlay = bgr.copy()
            cv2.putText(overlay, "no hand", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite(stem + suffix + ".png", overlay)
        json.dump(rec, open(stem + suffix + ".json", "w"), indent=2)

    det.close()
    print(f"[done] {n_det}/{len(stems)} frames had a hand. Wrote *{suffix}.png to {args.dir}")


if __name__ == "__main__":
    main()
