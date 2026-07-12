"""RGB-ONLY inference with the base MobRecon model (pretrain/100.pt) on the iPhone
RGB-D captures -- depth is NOT used at all.

Purpose: check whether the base RGB model (trained on HanCo RGB, never on iPhone)
already tracks the HAND SHAPE from iPhone RGB alone. The base model is
root-relative with no metric root (no depth), so absolute placement is impossible;
instead we Procrustes-fit (similarity: scale + rotation + translation) the predicted
21 joints' XY onto the MediaPipe 21 landmarks and overlay that. This removes global
pose/scale and leaves ONLY shape/articulation -- so a good overlay == the model got
the finger structure right, a bad one == it was confused by the iPhone RGB domain.
The mean post-alignment 2D error (px, normalised by hand size) is printed per frame
and summarised, as a quantitative shape-fidelity score.

    python infer_rgb_base.py --dir rgbd_captures --ckpt pretrain/100.pt
"""
import os
import re
import glob
import json
import argparse

import numpy as np
import cv2
import torch

from models.mobrecon_ds import LargeModel_Extra
from models.fastvit_backbone import FastViTHandSoftArgmax
from utils import load_cfg
from datasets.dataset_utils import augmentation
from infer_rgbd_captures import load_capture, rotate_frame, vote_rotation
from infer_rgbd_femtobolt import HandDetector, landmarks_to_bbox, draw_skeleton


def umeyama_sim(src, dst):
    """Least-squares similarity (s, R, t) mapping s*R@src + t -> dst for (N,2) pts."""
    src_mean, dst_mean = src.mean(0), dst.mean(0)
    sc, dc = src - src_mean, dst - dst_mean
    cov = (dc.T @ sc) / len(src)
    U, S, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, d])
    R = U @ D @ Vt
    var = (sc ** 2).sum() / len(src)
    s = (S * np.array([1.0, d])).sum() / (var + 1e-12)
    t = dst_mean - s * (R @ src_mean)
    return s, R, t


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="rgbd_captures")
    ap.add_argument("--ckpt", default="pretrain/100.pt")
    ap.add_argument("--cfg", default="./configs_rgbd.yaml")
    ap.add_argument("--tag", default="rgbbase")
    ap.add_argument("--backbone", default="densestack", choices=["densestack", "fastvit"])
    ap.add_argument("--fastvit_variant", default="fastvit_t8")
    ap.add_argument("--rotate", default="auto", help="'auto' or 0/90/180/270 deg CW")
    ap.add_argument("--margin", default=0.50, type=float)
    ap.add_argument("--limit", default=0, type=int)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_cfg(args.cfg)
    if args.backbone == "fastvit":
        model = FastViTHandSoftArgmax(variant=args.fastvit_variant, in_chans=3, pretrained=False).to(device).eval()
    else:
        model = LargeModel_Extra(cfg).to(device).eval()
    state = torch.load(args.ckpt, map_location=device)
    sd = state.get("model_state_dict", state) if isinstance(state, dict) else state
    model.load_state_dict(sd)
    print(f"[model] loaded {args.ckpt} ({args.backbone} RGB-only, 3-ch, no depth) device={device}")

    base_re = re.compile(r"^cap_\d+\.json$")
    stems = sorted(s[:-5] for s in glob.glob(os.path.join(args.dir, "cap_*.json"))
                   if base_re.match(os.path.basename(s)))
    if args.limit > 0:
        stems = stems[:args.limit]
    if not stems:
        print(f"[main] no captures in {args.dir}")
        return
    print(f"[main] {len(stems)} captures in {args.dir}")

    detector = HandDetector()
    k = vote_rotation(stems, detector) if args.rotate == "auto" else (int(args.rotate) // 90) % 4

    suffix = f"_pred_{args.tag}"
    n_det = 0
    errs = []
    for stem in stems:
        name = os.path.basename(stem)
        cap = load_capture(stem)
        if cap is None:
            continue
        bgr0, depth0, K0 = cap
        bgr, _, K = rotate_frame(bgr0, depth0, K0, k)

        pts = detector.detect(bgr)
        if pts is not None:
            pts = np.asarray(pts, np.float32)[:, :2]
            bbox = landmarks_to_bbox(pts, bgr.shape[:2], margin_frac=args.margin)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            aug_img, img2bb_trans, _, _, _, _, _, _ = augmentation(
                rgb.astype(np.float32), bbox, "test", exclude_flip=True,
                rotation=True, cam_param=K)
            rgb_crop = np.clip(aug_img, 0, 255).astype(np.uint8)
            inp = (torch.from_numpy(rgb_crop).float().permute(2, 0, 1) / 255.0
                   ).unsqueeze(0).to(device)
            with torch.no_grad():
                rel = model(inp)["keypoints"][0].cpu().numpy()      # (21,3) root-rel

            # Procrustes-fit predicted XY onto MediaPipe landmarks -> shape-only overlay
            s, R, t = umeyama_sim(rel[:, :2], pts)
            uv = (s * (rel[:, :2] @ R.T)) + t
            hand_sz = float(np.linalg.norm(pts[9] - pts[0]) + 1e-6)  # wrist->mid-MCP
            err = float(np.linalg.norm(uv - pts, axis=1).mean() / hand_sz)
            errs.append(err)

            overlay = draw_skeleton(bgr, uv, bbox)          # colored bones + white joints = MODEL
            for u, v in pts:                    # MediaPipe landmarks = cyan reference rings
                cv2.circle(overlay, (int(u), int(v)), 6, (255, 255, 0), 2)
            cv2.putText(overlay, f"[{args.tag}] shape-err={err*100:4.1f}% of hand",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(overlay, f"colored = {args.tag}   cyan ring = MediaPipe",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            n_det += 1
        else:
            overlay = bgr.copy()
            cv2.putText(overlay, "no hand", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite(stem + suffix + ".png", overlay)

    detector.close()
    if errs:
        errs = np.array(errs)
        print(f"\n[done] {n_det}/{len(stems)} frames had a hand. "
              f"shape-err (post-alignment, % of hand size): "
              f"mean={errs.mean()*100:.1f}%  median={np.median(errs)*100:.1f}%  "
              f"p90={np.percentile(errs,90)*100:.1f}%")
    print(f"[done] wrote *{suffix}.png to {args.dir}")


if __name__ == "__main__":
    main()
