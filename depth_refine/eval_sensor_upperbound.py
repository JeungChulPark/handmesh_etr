"""Upper-bound experiment: how much would a REAL depth sensor help?

The refinement corrects only the joint depth (Z). So the ceiling of any depth-based
refinement is: keep the RGB model's (x, y), replace each joint's Z with the sensor reading.
With a *perfect* sensor that reading equals the GT joint depth, giving an exact upper bound.
We then add Gaussian sensor noise (sigma) to map noise level -> achievable accuracy, so a
hardware choice (LiDAR ~5-10 mm, ToF ~10-20 mm) can be read off directly.

This isolates the *value of a good depth source* from the monocular-depth dead end
(see eval_refine.py / eval_root_depth.py).

Usage:
    python -m depth_refine.eval_sensor_upperbound --n 3960 --sigmas 0 2 5 10 20
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.mobrecon_ds import LargeModel_Extra
from datasets.freihand_ty import Freihand
from depth_refine.eval_refine import pa_mpjpe

ROOT = 0


def metrics(pred, gt):
    mpjpe = np.sqrt(((pred - gt) ** 2).sum(-1)).mean()
    pa = pa_mpjpe(pred, gt)
    zerr = np.abs(pred[:, 2] - gt[:, 2]).mean()
    return mpjpe, pa, zerr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3960)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0, 2, 5, 10, 20],
                    help="sensor depth noise std in mm")
    ap.add_argument("--ckpt", default="pretrain/100.pt")
    ap.add_argument("--out", default="depth_refine/results_upperbound.json")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)

    model = LargeModel_Extra(None)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev)["model_state_dict"])
    model.eval().to(dev)

    ds = Freihand(mode="eval")
    n = min(args.n, len(ds))
    dl = DataLoader(Subset(ds, list(range(n))), batch_size=args.batch, shuffle=False, num_workers=4)

    keys = ["baseline"] + [f"sensor σ={s:g}mm" for s in args.sigmas]
    acc = {k: np.zeros(3) for k in keys}
    count = 0

    with torch.no_grad():
        for it in dl:
            img = it["img"].float().to(dev)
            gt = it["align_joint"].numpy().astype(np.float64)      # (B,21,3) root-rel meters
            pred = model(img)["keypoints"].cpu().numpy().astype(np.float64)
            B = img.shape[0]
            for j in range(B):
                acc["baseline"] += metrics(pred[j], gt[j])
                for s in args.sigmas:
                    noise = rng.normal(0, s / 1000.0, size=21)     # per-joint absolute depth noise (m)
                    sensor_relz = (gt[j][:, 2] + noise)            # absolute-ish; re-center to root below
                    sensor_relz = sensor_relz - sensor_relz[ROOT]
                    Jr = pred[j].copy()
                    Jr[:, 2] = sensor_relz                         # keep model x,y, take sensor z
                    acc[f"sensor σ={s:g}mm"] += metrics(Jr, gt[j])
            count += B

    base = acc["baseline"] / count * 1000
    print(f"\n=== Depth-sensor UPPER BOUND (FreiHAND eval, {count} samples) ===")
    print("(model x,y kept; joint Z replaced by simulated sensor reading)")
    print(f"{'source':>16} | {'MPJPE(mm)':>10} | {'PA(mm)':>8} | {'Z-err(mm)':>9} | {'ΔMPJPE':>8}")
    print("-" * 64)
    results = {}
    for k in keys:
        mp, pa, ze = acc[k] / count * 1000
        d = "" if k == "baseline" else f"  {mp - base[0]:+.2f}"
        print(f"{k:>16} | {mp:>10.2f} | {pa:>8.2f} | {ze:>9.2f} |{d:>8}")
        results[k] = {"mpjpe_mm": mp, "pa_mm": pa, "zerr_mm": ze}

    json.dump({"n": count, "ckpt": args.ckpt, "results": results}, open(args.out, "w"), indent=2)
    print(f"\nsaved -> {args.out}")
    print("\nReference sensor noise: iPhone LiDAR ~5-10 mm (close range), ToF ~10-20 mm.")


if __name__ == "__main__":
    main()
