"""Evaluate RGB-only baseline vs depth-refined hand joints on FreiHAND.

Pipeline (no retraining):
    RGB --> MobRecon (frozen)        --> root-relative 3D joints  (baseline)
        --> Depth Anything V2        --> monocular inverse-depth map
        --> refine.refine_sample     --> depth-corrected joints   (refined)

Reports root-relative MPJPE, PA-MPJPE (Procrustes) and Z-only error, for the
baseline and for several fusion weights w. w=0 == baseline (sanity check).

Usage:
    python -m depth_refine.eval_refine --n 800 --weights 0 0.25 0.5 0.75 1.0
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
from depth_refine.monocular_depth import MonocularDepth
from depth_refine import refine


def pa_mpjpe(pred, gt):
    """Procrustes-aligned MPJPE for one sample (21,3) -> scalar (same units as input)."""
    p, g = pred - pred.mean(0), gt - gt.mean(0)
    H = p.T @ g
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    scale = S.sum() / (p ** 2).sum()
    aligned = scale * (p @ R)
    return np.sqrt(((aligned - g) ** 2).sum(-1)).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800, help="number of eval samples (<=3960)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--weights", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--ckpt", default="pretrain/100.pt")
    ap.add_argument("--bone_strength", type=float, default=0.0)
    ap.add_argument("--out", default="depth_refine/results.json")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    model = LargeModel_Extra(None)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev)["model_state_dict"])
    model.eval().to(dev)

    depth = MonocularDepth()

    ds = Freihand(mode="eval")
    n = min(args.n, len(ds))
    dl = DataLoader(Subset(ds, list(range(n))), batch_size=args.batch, shuffle=False, num_workers=4)

    # reference bone lengths from a chunk of GT (used only if bone_strength>0)
    bone_ref = None
    if args.bone_strength > 0:
        gts = np.stack([ds[i]["align_joint"] for i in range(min(500, n))])
        bone_ref = refine.estimate_bone_lengths(gts)

    keys = ["baseline"] + [f"w={w:g}" for w in args.weights]
    acc = {k: {"mpjpe": 0.0, "pa": 0.0, "zerr": 0.0} for k in keys}
    # diagnostic: correlation of relative-Z with GT across the 21 joints
    diag = {"corr_pred": 0.0, "corr_mono": 0.0, "dn": 0}
    count = 0

    with torch.no_grad():
        for it in dl:
            img = it["img"].float().to(dev)
            ori = it["ori_image"].float()
            gt = it["align_joint"].numpy().astype(np.float64)        # (B,21,3) root-rel meters
            root = it["root"].numpy().astype(np.float64)             # (B,3)
            K = it["cam"].numpy().astype(np.float64)                 # (B,3,3)
            pred = model(img)["keypoints"].cpu().numpy().astype(np.float64)  # (B,21,3) root-rel
            disp_maps = depth.infer_batch(ori)                       # list of (224,224)

            B = img.shape[0]
            for j in range(B):
                g = gt[j]
                # baseline
                _accumulate(acc["baseline"], pred[j], g)
                # refined for each weight
                for w in args.weights:
                    Jr = refine.refine_sample(
                        pred[j], root[j], K[j], disp_maps[j], w=w,
                        bone_ref=bone_ref, bone_strength=args.bone_strength,
                    )
                    _accumulate(acc[f"w={w:g}"], Jr, g)
                # diagnostic correlations (does monocular Z track GT better than predicted?)
                z_mono, _ = refine.monocular_rel_z(pred[j], root[j], K[j], disp_maps[j])
                if np.all(np.isfinite(z_mono)) and z_mono.std() > 1e-9:
                    diag["corr_pred"] += _corr(pred[j][:, 2], g[:, 2])
                    diag["corr_mono"] += _corr(z_mono, g[:, 2])
                    diag["dn"] += 1
            count += B

    print(f"\n=== FreiHAND eval: {count} samples (ckpt={args.ckpt}) ===")
    print(f"{'config':>10} | {'MPJPE(mm)':>10} | {'PA-MPJPE(mm)':>12} | {'Z-err(mm)':>9}")
    print("-" * 52)
    results = {}
    base_mpjpe = acc["baseline"]["mpjpe"] / count * 1000
    for k in keys:
        mp = acc[k]["mpjpe"] / count * 1000
        pa = acc[k]["pa"] / count * 1000
        ze = acc[k]["zerr"] / count * 1000
        delta = ""
        if k != "baseline":
            d = mp - base_mpjpe
            delta = f"  ({'+' if d >= 0 else ''}{d:.2f})"
        print(f"{k:>10} | {mp:>10.2f} | {pa:>12.2f} | {ze:>9.2f}{delta}")
        results[k] = {"mpjpe_mm": mp, "pa_mpjpe_mm": pa, "zerr_mm": ze}

    if diag["dn"] > 0:
        cp = diag["corr_pred"] / diag["dn"]
        cm = diag["corr_mono"] / diag["dn"]
        print(f"\nDiagnostic (relative-Z vs GT, mean Pearson r over {diag['dn']} samples):")
        print(f"  predicted-Z  vs GT-Z : r = {cp:+.3f}")
        print(f"  monocular-Z  vs GT-Z : r = {cm:+.3f}")
        print("  -> if monocular r <= predicted r, fusing depth cannot help (it only adds noise).")
        results["diagnostic"] = {"corr_pred_z": cp, "corr_mono_z": cm}

    with open(args.out, "w") as f:
        json.dump({"n": count, "ckpt": args.ckpt, "results": results}, f, indent=2)
    print(f"\nsaved -> {args.out}")


def _corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _accumulate(slot, pred, gt):
    slot["mpjpe"] += np.sqrt(((pred - gt) ** 2).sum(-1)).mean()
    slot["pa"] += pa_mpjpe(pred, gt)
    slot["zerr"] += np.abs(pred[:, 2] - gt[:, 2]).mean()


if __name__ == "__main__":
    main()
