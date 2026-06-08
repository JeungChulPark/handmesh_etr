"""Absolute root-depth experiment: can monocular depth recover the wrist's metric
distance-from-camera — something the RGB hand model does NOT predict at all?

The RGB model outputs root-RELATIVE joints (no absolute scale). FreiHAND's standard
MPJPE is also root-relative, so it can't show depth's value for absolute placement.
Here we isolate that: sample monocular depth at the wrist and compare the estimated
absolute wrist Z to ground truth, against a 'no-depth' constant baseline.

We test:
  - constant      : predict the calibration-set median wrist depth (best guess w/o depth)
  - relative+calib: Depth Anything (relative) inverse-depth, affine-calibrated to meters
  - metric raw    : Depth Anything V2 Metric (Indoor), meters as-is
  - metric+calib  : metric model with a per-dataset affine correction

Calibration is fit on a held-out split and applied to the test split (no per-sample GT).

Usage:
    python -m depth_refine.eval_root_depth --n 2000 --calib 600
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from transformers import pipeline

ROOT_DIR = "../Datasets/Hand Dataset/FreiHAND/FreiHAND_pub_v2"


def project_wrist(xyz, K):
    """xyz (21,3) camera-frame, K (3,3) -> wrist (u,v) pixel."""
    w = np.asarray(xyz)[0]
    K = np.asarray(K)
    u = w[0] / w[2] * K[0, 0] + K[0, 2]
    v = w[1] / w[2] * K[1, 1] + K[1, 2]
    return u, v


def sample(depth, u, v, r=3):
    H, W = depth.shape
    ui, vi = int(round(np.clip(u, 0, W - 1))), int(round(np.clip(v, 0, H - 1)))
    return float(np.median(depth[max(0, vi - r):vi + r + 1, max(0, ui - r):ui + r + 1]))


def depth_maps(pipe, pil_list):
    outs = pipe(pil_list)
    if isinstance(outs, dict):
        outs = [outs]
    maps = []
    for o in outs:
        d = o["predicted_depth"]
        d = d.squeeze().detach().cpu().numpy() if isinstance(d, torch.Tensor) else np.asarray(d)
        maps.append(d.astype(np.float32))
    return maps


def report(name, est, gt):
    mae = np.mean(np.abs(est - gt)) * 1000
    rmse = np.sqrt(np.mean((est - gt) ** 2)) * 1000
    r = np.corrcoef(est, gt)[0, 1] if est.std() > 1e-9 else 0.0
    print(f"{name:>16} | MAE {mae:7.1f} mm | RMSE {rmse:7.1f} mm | r {r:+.3f}")
    return {"mae_mm": mae, "rmse_mm": rmse, "r": float(r)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--calib", type=int, default=600)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--out", default="depth_refine/results_rootdepth.json")
    args = ap.parse_args()

    dev = 0 if torch.cuda.is_available() else -1
    xyz = np.array(json.load(open(f"{ROOT_DIR}/evaluation_xyz.json")))
    K = np.array(json.load(open(f"{ROOT_DIR}/evaluation_K.json")))
    names = np.sort(os.listdir(f"{ROOT_DIR}/evaluation/rgb"))[: len(xyz)]
    n = min(args.n, len(xyz))

    print("loading relative + metric Depth Anything V2 ...")
    pipe_rel = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=dev)
    try:
        pipe_met = pipeline("depth-estimation",
                            model="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", device=dev)
    except Exception as e:
        print("metric model unavailable:", e)
        pipe_met = None

    disp_rel, z_met, z_gt = [], [], []
    with torch.no_grad():
        for s in range(0, n, args.batch):
            idx = range(s, min(s + args.batch, n))
            pil = [Image.open(f"{ROOT_DIR}/evaluation/rgb/{names[i]}").convert("RGB") for i in idx]
            rel = depth_maps(pipe_rel, pil)
            met = depth_maps(pipe_met, pil) if pipe_met else [None] * len(pil)
            for k, i in enumerate(idx):
                u, v = project_wrist(xyz[i], K[i])
                disp_rel.append(sample(rel[k], u, v))
                z_met.append(sample(met[k], u, v) if met[k] is not None else np.nan)
                z_gt.append(xyz[i][0][2])

    disp_rel = np.array(disp_rel); z_met = np.array(z_met); z_gt = np.array(z_gt)
    c = args.calib
    cal, tst = slice(0, c), slice(c, n)

    print(f"\n=== Absolute wrist-depth recovery (FreiHAND eval, n={n}, calib={c}, test={n-c}) ===")
    print(f"   GT wrist Z: mean {z_gt.mean():.3f} m, std {z_gt.std():.3f} m")
    print(f"{'method':>16} | {'MAE':>10} | {'RMSE':>10} | {'corr':>6}")
    print("-" * 56)
    res = {}

    # no-depth constant (calibration median)
    const = np.full(z_gt[tst].shape, np.median(z_gt[cal]))
    res["constant_no_depth"] = report("constant", const, z_gt[tst])

    # relative + affine calib: 1/Z = a*disp + b
    A = np.stack([disp_rel[cal], np.ones(c)], 1)
    a, b = np.linalg.lstsq(A, 1.0 / z_gt[cal], rcond=None)[0]
    inv = a * disp_rel[tst] + b
    z_rel = 1.0 / np.where(inv > 1e-3, inv, np.nan)
    z_rel = np.where(np.isfinite(z_rel), z_rel, np.median(z_gt[cal]))
    res["relative_calibrated"] = report("relative+calib", z_rel, z_gt[tst])

    if pipe_met is not None and np.isfinite(z_met).all():
        res["metric_raw"] = report("metric raw", z_met[tst], z_gt[tst])
        am, bm = np.linalg.lstsq(np.stack([z_met[cal], np.ones(c)], 1), z_gt[cal], rcond=None)[0]
        res["metric_calibrated"] = report("metric+calib", am * z_met[tst] + bm, z_gt[tst])

    json.dump({"n": n, "calib": c, "gt_mean": float(z_gt.mean()),
               "gt_std": float(z_gt.std()), "results": res}, open(args.out, "w"), indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
