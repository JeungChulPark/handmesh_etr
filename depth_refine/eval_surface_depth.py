"""Realistic depth-sensor gain via proper MANO-mesh SURFACE rendering.

A real sensor reads the hand SKIN SURFACE (not the joint center), at the 2D pixel of the
joint, and a self-occluded joint reads the OCCLUDER's surface. We model this correctly by
ray-casting each joint's 2D pixel against the GT MANO mesh triangles (z-buffer: the front
-most triangle wins), using the real face topology (template/right_faces.npy, 1538 tris).

This captures three real effects the ideal upper bound ignored:
  - surface-vs-joint offset (~finger radius, the skin is closer than the bone joint),
  - self-occlusion (front surface, detected from the mesh itself — not from the prediction),
  - 2D sampling error (sample at PREDICTED vs GT joint pixel).

Occluded joints (front surface well in front of the joint) are detected from mesh truth and
fall back to the model's depth (a correct pipeline rejects them). Visible joints take the
surface reading (optionally with sensor noise).

Usage:
    python -m depth_refine.eval_surface_depth --n 2000 --occ_margin 20 --noise 5
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

ROOT_DIR = "../Datasets/Hand Dataset/FreiHAND/FreiHAND_pub_v2"
ROOT = 0


def project(pts3d, K):
    z = np.where(np.abs(pts3d[:, 2]) < 1e-6, 1e-6, pts3d[:, 2])
    u = pts3d[:, 0] / z * K[0, 0] + K[0, 2]
    v = pts3d[:, 1] / z * K[1, 1] + K[1, 2]
    return np.stack([u, v], 1)


def raycast_front_depth(v2, vz, faces, p):
    """Front-most mesh surface depth at pixel p (z-buffer over triangles). None if no hit."""
    A = v2[faces[:, 0]]; B = v2[faces[:, 1]]; C = v2[faces[:, 2]]
    e0 = B - A; e1 = C - A; e2 = p[None, :] - A
    d00 = (e0 * e0).sum(1); d01 = (e0 * e1).sum(1); d11 = (e1 * e1).sum(1)
    d20 = (e2 * e0).sum(1); d21 = (e2 * e1).sum(1)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    u = 1.0 - v - w
    inside = (u >= -1e-3) & (v >= -1e-3) & (w >= -1e-3)
    if not inside.any():
        return None
    z = u * vz[faces[:, 0]] + v * vz[faces[:, 1]] + w * vz[faces[:, 2]]
    return float(z[inside].min())


def metrics(pred, gt):
    return (np.sqrt(((pred - gt) ** 2).sum(-1)).mean(),
            pa_mpjpe(pred, gt),
            np.abs(pred[:, 2] - gt[:, 2]).mean())


def replace_z(pred_relxyz, abs_z):
    z = np.asarray(abs_z, float)
    out = pred_relxyz.copy()
    out[:, 2] = z - z[ROOT]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--occ_margin", type=float, default=20.0, help="mm; surface this much in front of joint => occluded")
    ap.add_argument("--noise", type=float, default=5.0, help="sensor noise std (mm)")
    ap.add_argument("--ckpt", default="pretrain/100.pt")
    ap.add_argument("--out", default="depth_refine/results_surface.json")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    faces = np.load("template/right_faces.npy").astype(np.int64)
    if faces.ndim == 2 and faces.shape[1] != 3 and faces.shape[0] == 3:
        faces = faces.T

    model = LargeModel_Extra(None)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev)["model_state_dict"])
    model.eval().to(dev)

    verts_all = json.load(open(f"{ROOT_DIR}/evaluation_verts.json"))
    ds = Freihand(mode="eval")
    n = min(args.n, len(ds))
    dl = DataLoader(Subset(ds, list(range(n))), batch_size=args.batch, shuffle=False, num_workers=4)

    occ_m = args.occ_margin / 1000.0
    keys = ["baseline", "ideal (joint Z)",
            "surface @GT-2D (visible)", "surface @pred-2D (visible)",
            f"surface @pred-2D +noise{args.noise:g}mm"]
    acc = {k: np.zeros(3) for k in keys}
    occ_count = 0; jtot = 0; surf_off = 0.0; surf_n = 0
    gidx = 0

    with torch.no_grad():
        for it in dl:
            img = it["img"].float().to(dev)
            gt = it["align_joint"].numpy().astype(np.float64)
            root = it["root"].numpy().astype(np.float64)
            K = it["cam"].numpy().astype(np.float64)
            pred = model(img)["keypoints"].cpu().numpy().astype(np.float64)
            B = img.shape[0]
            for j in range(B):
                verts = np.array(verts_all[gidx]); gidx += 1
                Kj = K[j]
                gt_abs = gt[j] + root[j][None, :]            # GT absolute joints
                gtz_abs = gt_abs[:, 2]
                pred_abs = pred[j] + root[j][None, :]
                v2 = project(verts, Kj)
                vz = verts[:, 2]

                gt2d = project(gt_abs, Kj)
                pr2d = project(pred_abs, Kj)

                surf_gt = np.empty(21); surf_pr = np.empty(21); occluded = np.zeros(21, bool)
                for i in range(21):
                    sg = raycast_front_depth(v2, vz, faces, gt2d[i])
                    sp = raycast_front_depth(v2, vz, faces, pr2d[i])
                    surf_gt[i] = sg if sg is not None else gtz_abs[i]
                    surf_pr[i] = sp if sp is not None else pred_abs[i, 2]
                    # mesh-truth occlusion: front surface clearly in front of this joint
                    occluded[i] = (sg is not None) and (gtz_abs[i] - sg > occ_m)
                    if sg is not None and not occluded[i]:
                        surf_off += abs(gtz_abs[i] - sg); surf_n += 1
                occ_count += int(occluded.sum()); jtot += 21

                acc["baseline"] += metrics(pred[j], gt[j])
                acc["ideal (joint Z)"] += metrics(replace_z(pred[j], gtz_abs), gt[j])

                # visible joints -> surface reading; occluded -> keep model depth
                z_gt = np.where(occluded, pred_abs[:, 2], surf_gt)
                z_pr = np.where(occluded, pred_abs[:, 2], surf_pr)
                acc["surface @GT-2D (visible)"] += metrics(replace_z(pred[j], z_gt), gt[j])
                acc["surface @pred-2D (visible)"] += metrics(replace_z(pred[j], z_pr), gt[j])
                z_no = z_pr + np.where(occluded, 0.0, rng.normal(0, args.noise / 1000.0, 21))
                acc[f"surface @pred-2D +noise{args.noise:g}mm"] += metrics(replace_z(pred[j], z_no), gt[j])

    base = acc["baseline"] / gidx * 1000
    print(f"\n=== Realistic MANO-surface sensor (FreiHAND eval, {gidx} samples) ===")
    print(f"  mesh-truth occluded joints: {occ_count/jtot*100:.1f}%   "
          f"mean surface-vs-joint offset: {surf_off/max(surf_n,1)*1000:.1f} mm")
    print(f"{'source':>34} | {'MPJPE':>8} | {'PA':>7} | {'Z-err':>7} | {'ΔMPJPE':>8}")
    print("-" * 78)
    res = {}
    for k in keys:
        mp, pa, ze = acc[k] / gidx * 1000
        d = "" if k == "baseline" else f"  {mp - base[0]:+.2f}"
        print(f"{k:>34} | {mp:>8.2f} | {pa:>7.2f} | {ze:>7.2f} |{d:>8}")
        res[k] = {"mpjpe_mm": mp, "pa_mm": pa, "zerr_mm": ze}
    json.dump({"n": gidx, "occ_margin_mm": args.occ_margin,
               "occluded_pct": occ_count / jtot * 100,
               "surface_offset_mm": surf_off / max(surf_n, 1) * 1000, "results": res},
              open(args.out, "w"), indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
