"""Live ZED hand-pose inference with the HYBRID-B model (`hybrid_B`, locked mode).

Model B = "locked 2D + backbone Z":
  * X,Y  from MediaPipe (u,v)              -> 2D locked to the detector
  * Z    from the MobRecon backbone's per-joint relative depth, anchored at the
         ZED wrist depth Z0                 -> Zabs = Z0 + dz_backbone
This is the same per-frame path as `infer_hybrid.py`, but the frame source is a
LIVE ZED camera (or .svo replay) via `infer_rgbd_zed.ZEDSource` instead of saved
cap_*.json captures.

Overlay, per frame:
  * cyan rings    = MediaPipe 2D (the accurate 2D reference)
  * grey skeleton = backbone-alone p_cnn reprojected
  * colour skeleton = HYBRID-B reprojected (depth-tinted joints)

Examples
--------
  python infer_zed_hybrid.py                                   # live ZED, hybrid_B
  python infer_zed_hybrid.py --source svo --svo rec.svo2
  python infer_zed_hybrid.py --no-gui --save zed_hybrid_out/   # headless dump
  python infer_zed_hybrid.py --source selftest                 # no hardware
"""
import os
import json
import time
import argparse

import numpy as np
import cv2
import torch

from utils import load_cfg
from train_hybrid import HybridLifter
from train_zlifter import build_hand_crop
from infer_rgbd_femtobolt import HandDetector, project, draw_skeleton, HAND_BONES
from infer_rgbd_zed import ZEDSource
from infer_hybrid import sample_depth


def build_source(args):
    if args.source == "zed":
        return ZEDSource(resolution=args.zed_resolution, fps=args.zed_fps,
                         depth_mode=args.zed_depth_mode, min_depth=args.zed_min_depth,
                         max_depth=args.zed_max_depth)
    if args.source == "svo":
        if not args.svo:
            raise SystemExit("[zed] --source svo requires --svo <file.svo|.svo2>")
        return ZEDSource(depth_mode=args.zed_depth_mode, min_depth=args.zed_min_depth,
                         max_depth=args.zed_max_depth, svo=args.svo)
    raise ValueError(args.source)


@torch.no_grad()
def infer_frame(model, det, bgr, depth_m, K, device):
    """Run hybrid-B on one BGR + metric-depth frame. Returns overlay + record."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pts = det.detect(bgr)
    if pts is None:
        overlay = bgr.copy()
        cv2.putText(overlay, "no hand", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return overlay, {"detected": False}

    uv = np.asarray(pts, np.float32)[:, :2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    zs = np.zeros(21, np.float32); zv = np.zeros(21, np.float32)
    for j in range(21):
        zs[j], zv[j] = sample_depth(depth_m, uv[j, 0], uv[j, 1])
    z_ref = float(np.median(zs[zv > 0])) if zv.any() else 0.5
    z_use = np.where(zv > 0, zs, z_ref).astype(np.float32)
    feat = np.stack([uv[:, 0] / bgr.shape[1] * 2 - 1, uv[:, 1] / bgr.shape[0] * 2 - 1,
                     z_use - z_ref, zv], axis=1).astype(np.float32)
    crop = build_hand_crop(rgb, depth_m, uv)

    p, p_cnn, _ = model(
        torch.from_numpy(feat).unsqueeze(0).to(device),
        crop.unsqueeze(0).to(device),
        torch.from_numpy(uv).unsqueeze(0).to(device),
        torch.from_numpy(z_use).unsqueeze(0).to(device),
        torch.tensor([[fx, fy, cx, cy]], dtype=torch.float32).to(device))
    p = p[0].cpu().numpy(); p_cnn = p_cnn[0].cpu().numpy()

    # place root at the ZED wrist depth for reprojection (model output is root-relative)
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
    cv2.putText(overlay, f"[hybrid_B] wrist_z={abs_h[0,2]*100:5.1f}cm  "
                f"z-span={np.ptp(abs_h[:,2])*100:4.1f}cm  depth-cov={(zv>0).mean()*100:4.0f}%",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(overlay, "cyan=MediaPipe  grey=backbone-only  colour=hybrid_B",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
    rec = {"detected": True, "root_z_m": round(float(abs_h[0, 2]), 4),
           "abs_joints": [[round(float(v), 4) for v in q] for q in abs_h]}
    return overlay, rec


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="mobrecon_ckpt/hybrid_B/best.pt")
    ap.add_argument("--cfg", default="./configs_rgbd.yaml")
    ap.add_argument("--source", default="zed", choices=["zed", "svo", "selftest"])
    ap.add_argument("--svo", default=None, help="path to a .svo/.svo2 recording")
    ap.add_argument("--hand_model", default=None,
                    help="path to MediaPipe hand_landmarker.task (auto-downloaded if absent)")
    ap.add_argument("--zed_resolution", default="HD720",
                    choices=["HD2K", "HD1080", "HD720", "VGA"])
    ap.add_argument("--zed_fps", default=30, type=int)
    ap.add_argument("--zed_depth_mode", default="NEURAL",
                    choices=["NEURAL", "ULTRA", "QUALITY", "PERFORMANCE"])
    ap.add_argument("--zed_min_depth", default=0.3, type=float)
    ap.add_argument("--zed_max_depth", default=2.0, type=float)
    ap.add_argument("--no-gui", dest="gui", action="store_false", help="headless")
    ap.add_argument("--save", default=None, help="dir to dump overlay PNGs + metrics.jsonl")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_cfg(args.cfg)
    model = HybridLifter(cfg, mode="locked").to(device).eval()   # B = locked mode
    st = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(st.get("model_state_dict", st))
    print(f"[hybrid_B] loaded {args.ckpt} (epoch {st.get('epoch','?')}, "
          f"combined MPJPE {st.get('best_test_mpjpe', float('nan')):.2f}mm)", flush=True)

    det = HandDetector(model_path=args.hand_model)

    if args.source == "selftest":                 # no hardware: synthetic frame
        bgr = np.full((720, 1280, 3), 40, np.uint8)
        depth_m = np.full((720, 1280), 0.5, np.float32)
        K = np.array([[700, 0, 640], [0, 700, 360], [0, 0, 1]], np.float32)
        overlay, rec = infer_frame(model, det, bgr, depth_m, K, device)
        print(f"[selftest] wiring OK  detected={rec['detected']}")
        det.close()
        return

    src = build_source(args)
    metrics_f = None
    if args.save:
        os.makedirs(args.save, exist_ok=True)
        metrics_f = open(os.path.join(args.save, "metrics.jsonl"), "w")

    t_prev, fps, frame_i, n_det = time.time(), 0.0, 0, 0
    try:
        while True:
            frame = src.read()
            if frame is None:
                print("[main] stream ended")
                break
            bgr, depth_m, K = frame
            overlay, rec = infer_frame(model, det, bgr, depth_m, K, device)
            if rec["detected"]:
                n_det += 1

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now
            cv2.putText(overlay, f"{fps:4.1f} FPS  [{args.source}]", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            if frame_i and frame_i % 30 == 0:
                print(f"[fps] frame {frame_i}  {fps:4.1f} FPS  (hand {n_det}/{frame_i})", flush=True)

            if args.save:
                cv2.imwrite(os.path.join(args.save, f"{frame_i:05d}.png"), overlay)
                if metrics_f is not None:
                    metrics_f.write(json.dumps({"frame": frame_i, **rec}) + "\n")
            if args.gui:
                cv2.imshow("hand pose (ZED, hybrid_B)", overlay)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            frame_i += 1
    finally:
        src.close()
        det.close()
        if metrics_f is not None:
            metrics_f.close()
        if args.gui:
            cv2.destroyAllWindows()
    print(f"[main] processed {frame_i} frames, hand in {n_det}", flush=True)


if __name__ == "__main__":
    main()
