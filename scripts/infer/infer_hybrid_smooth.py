"""Hybrid inference with One-Euro temporal smoothing — raw vs smoothed overlays.

Processes the capture frames IN ORDER, keeps a One-Euro filter state on the 3D
joints, and draws BOTH the raw per-frame skeleton and the smoothed one so the
jitter reduction is visible side-by-side. Same model path as scripts/infer/infer_hybrid.py.

    python scripts/infer/infer_hybrid_smooth.py --ckpt mobrecon_ckpt/hybrid_fastvit_bb_uv2d/best.pt \
        --backbone fastvit --mode backbone --tag fvbb2d
"""

import _bootstrap  # noqa: F401  (repo-root + sibling script imports)

import os
import re
import glob
import argparse

import numpy as np
import cv2
import torch

from utils import load_cfg
from train_hybrid import HybridLifter
from train_zlifter import build_hand_crop
from infer_rgbd_captures import load_capture, rotate_frame, vote_rotation
from infer_rgbd_femtobolt import HandDetector, project, draw_skeleton, HAND_BONES
from infer_hybrid import sample_depth


class OneEuro:
    """Vectorised One-Euro filter over an array x (e.g. [21,3]). freq in Hz."""
    def __init__(self, freq=15.0, mincutoff=1.5, beta=0.05, dcutoff=1.0):
        self.freq, self.mincutoff, self.beta, self.dcutoff = freq, mincutoff, beta, dcutoff
        self.x_prev = None; self.dx_prev = None

    @staticmethod
    def _alpha(cutoff, freq):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau * freq)

    def reset(self):
        self.x_prev = None; self.dx_prev = None

    def __call__(self, x):
        x = np.asarray(x, np.float64)
        if self.x_prev is None:
            self.x_prev = x; self.dx_prev = np.zeros_like(x)
            return x.astype(np.float32)
        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.dcutoff, self.freq)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.mincutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff, self.freq)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat; self.dx_prev = dx_hat
        return x_hat.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="rgbd_captures")
    ap.add_argument("--ckpt", default="mobrecon_ckpt/hybrid_fastvit_bb_uv2d/best.pt")
    ap.add_argument("--cfg", default="./configs/configs_rgbd.yaml")
    ap.add_argument("--backbone", default="fastvit", choices=["densestack", "fastvit"])
    ap.add_argument("--fastvit_variant", default="fastvit_t8")
    ap.add_argument("--mode", default="backbone", choices=["backbone", "locked"])
    ap.add_argument("--tag", default="smooth")
    ap.add_argument("--fps", default=15.0, type=float)
    ap.add_argument("--mincutoff", default=1.5, type=float)
    ap.add_argument("--beta", default=0.05, type=float)
    ap.add_argument("--rotate", default="auto")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_cfg(args.cfg)
    model = HybridLifter(cfg, mode=args.mode, backbone=args.backbone,
                         fastvit_variant=args.fastvit_variant).to(device).eval()
    st = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(st.get("model_state_dict", st))
    print(f"[smooth] loaded {args.ckpt} (epoch {st.get('epoch','?')})", flush=True)

    base_re = re.compile(r"^cap_\d+\.json$")
    stems = sorted(s[:-5] for s in glob.glob(os.path.join(args.dir, "cap_*.json"))
                   if base_re.match(os.path.basename(s)))
    det = HandDetector()
    k = vote_rotation(stems, det) if args.rotate == "auto" else (int(args.rotate) // 90) % 4
    euro = OneEuro(freq=args.fps, mincutoff=args.mincutoff, beta=args.beta)
    n_det = 0; miss = 0

    for stem in stems:
        cap = load_capture(stem)
        if cap is None:
            continue
        bgr, depth, K = rotate_frame(*cap, k)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        pts = det.detect(bgr)
        raw = bgr.copy(); smo = bgr.copy()
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
                p, p_cnn, _ = model(
                    torch.from_numpy(feat).unsqueeze(0).to(device), crop.unsqueeze(0).to(device),
                    torch.from_numpy(uv).unsqueeze(0).to(device), torch.from_numpy(z_use).unsqueeze(0).to(device),
                    torch.tensor([[fx, fy, cx, cy]], dtype=torch.float32).to(device))
            p = p[0].cpu().numpy()
            zw = zs[0] if zv[0] > 0 else z_ref
            root = np.array([(uv[0, 0] - cx) / fx * zw, (uv[0, 1] - cy) / fy * zw, zw], np.float32)
            abs_h = p + root
            abs_s = euro(abs_h)                                   # temporal One-Euro
            raw = draw_skeleton(raw, project(abs_h, K))
            smo = draw_skeleton(smo, project(abs_s, K))
            n_det += 1
        else:
            euro.reset(); miss += 1
            for im in (raw, smo):
                cv2.putText(im, "no hand", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(raw, "RAW (per-frame)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
        cv2.putText(smo, "One-Euro SMOOTHED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 120), 2)
        cv2.imwrite(stem + f"_sm_{args.tag}_raw.png", raw)
        cv2.imwrite(stem + f"_sm_{args.tag}_smooth.png", smo)

    det.close()
    print(f"[smooth] {n_det}/{len(stems)} hand frames ({miss} gaps). "
          f"Wrote *_sm_{args.tag}_raw.png / _smooth.png to {args.dir}", flush=True)


if __name__ == "__main__":
    main()
