"""PA-MPJPE (and friends) evaluation for the dual-stream RGB-D hand model.

Computes the standard hand-pose metrics on the SAME HanCo val split the training
uses (seed-0, 0.95/0.05), so numbers line up with the training logs:

  * PA-MPJPE   -- Procrustes-aligned MPJPE: removes the global similarity
                  (rotation + scale + translation) per sample, then averages the
                  per-joint L2. Isolates pose/shape error from any global mismatch.
  * root-rel MPJPE -- root-aligned per-joint error (== training `test_rel`).
  * abs MPJPE  -- absolute camera-space per-joint error (== training `test_abs`).
  * root err   -- global translation error ||pred_root - gt_root||.
  * PCK-AUC    -- area under the PCK curve over 0..`auc_max` mm (rel & PA).

Works for any checkpoint of models.mobrecon_dualstream.MobRecon_DualStream; the
pose backbone's input-channel count is auto-detected from the stem weight, so the
same script evaluates the RGB-pose (3ch) and hybrid (4ch) variants.

Run:
    python eval_pa_mpjpe.py --ckpt mobrecon_ckpt/ds_hybrid_ft2/best.pt
    python eval_pa_mpjpe.py --ckpt mobrecon_ckpt/ds_hybrid/best.pt --max_eval 5000 --device cpu
"""

import os
import argparse

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

from models.mobrecon_dualstream import MobRecon_DualStream
from datasets.hanco_ty import HanCo_ETRI_jitter


def compute_similarity_transform(S1, S2):
    """Procrustes: align S1 to S2 with a similarity transform (s, R, t).

    S1, S2: (3, N) arrays. Returns S1 mapped onto S2 (the standard SMPL/HMR
    `compute_similarity_transform`). Minimises ||s*R*S1 + t - S2||.
    """
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2
    var1 = np.sum(X1 ** 2)
    K = X1 @ X2.T
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U @ V.T))  # reflection guard
    R = V @ Z @ U.T
    scale = np.trace(R @ K) / (var1 + 1e-12)
    t = mu2 - scale * (R @ mu1)
    return scale * (R @ S1) + t


def pa_mpjpe_mm(pred, gt):
    """pred, gt: (J, 3) metres -> PA-MPJPE in mm."""
    aligned = compute_similarity_transform(pred.T, gt.T).T  # (J,3)
    return np.sqrt(((aligned - gt) ** 2).sum(-1)).mean() * 1000.0


def mpjpe_mm(pred, gt):
    return np.sqrt(((pred - gt) ** 2).sum(-1)).mean() * 1000.0


def detect_pose_in_chans(state):
    """Read the pose backbone stem to recover its input-channel count (3 or 4)."""
    w = state.get("rgb_backbone.pre_layer.0.0.weight")
    return int(w.shape[1]) if w is not None else 3


def main(args):
    torch.manual_seed(0)  # MUST match training so the val split is identical
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    if args.dataset == "hanco":
        ds = HanCo_ETRI_jitter(limit=args.limit, with_depth=True,
                               depth_source=args.depth_source, return_abs_depth=True)
    elif args.dataset == "dexycb":
        from datasets.dexycb import DexYCB_RGBD
        ds = DexYCB_RGBD(root=args.dexycb_root, mode="train", with_depth=True,
                         flip_left=True, return_abs_depth=True)
    elif args.dataset == "ho3d":
        from datasets.ho3d import HO3D_RGBD
        ds = HO3D_RGBD(root=args.ho3d_root, mode="train", with_depth=True,
                       return_abs_depth=True)
    else:
        raise ValueError(args.dataset)
    n_train = int(len(ds) * 0.95)
    _, test_ds = random_split(ds, [n_train, len(ds) - n_train])
    loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=8)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    pose_in_chans = detect_pose_in_chans(state)
    model = MobRecon_DualStream(cfg=None, pose_in_chans=pose_in_chans)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    print(f"[eval] {args.ckpt} (pose_in_chans={pose_in_chans}, "
          f"epoch={ckpt.get('epoch','?')}) on {len(test_ds)} val frames", flush=True)

    rel, pa, ab, rooterr = [], [], [], []
    thresholds = np.linspace(0, args.auc_max, 100)
    pck_rel = np.zeros_like(thresholds)
    pck_pa = np.zeros_like(thresholds)
    n_seen = 0

    with torch.no_grad():
        for item in tqdm(loader, leave=False):
            img = item["image"].float().to(device)
            if args.rgb_only:
                img[:, 3:4] = 0.0
            kps3d = item["keypoints3D"].numpy()                 # [B,21,3] root-rel GT (m)
            root_gt = item["root"].numpy()                      # [B,3]
            depth_med = item["depth_med"].float().to(device)
            if args.rgb_only:
                depth_med = torch.zeros_like(depth_med)
            out = model(img, depth_med)
            pred_rel = out["keypoints"].cpu().numpy()           # [B,21,3]
            pred_root = out["root"].cpu().numpy()               # [B,3]
            pred_abs = out["keypoints_abs"].cpu().numpy()
            gt_abs = kps3d + root_gt[:, None, :]

            for b in range(pred_rel.shape[0]):
                rel.append(mpjpe_mm(pred_rel[b], kps3d[b]))
                pa.append(pa_mpjpe_mm(pred_rel[b], kps3d[b]))
                ab.append(mpjpe_mm(pred_abs[b], gt_abs[b]))
                rooterr.append(np.sqrt(((pred_root[b] - root_gt[b]) ** 2).sum()) * 1000.0)
                # PCK curves (per-joint, root-rel and PA-aligned)
                d_rel = np.sqrt(((pred_rel[b] - kps3d[b]) ** 2).sum(-1)) * 1000.0
                aligned = compute_similarity_transform(pred_rel[b].T, kps3d[b].T).T
                d_pa = np.sqrt(((aligned - kps3d[b]) ** 2).sum(-1)) * 1000.0
                pck_rel += (d_rel[None, :] <= thresholds[:, None]).mean(1)
                pck_pa += (d_pa[None, :] <= thresholds[:, None]).mean(1)
                n_seen += 1
            if args.max_eval and n_seen >= args.max_eval:
                break

    pck_rel /= n_seen
    pck_pa /= n_seen
    trapz = getattr(np, "trapezoid", np.trapz)  # trapezoid (numpy>=2) or legacy trapz
    auc_rel = trapz(pck_rel, thresholds) / args.auc_max
    auc_pa = trapz(pck_pa, thresholds) / args.auc_max

    print(f"\n=== {os.path.basename(os.path.dirname(args.ckpt))} | {n_seen} frames ===")
    print(f"  PA-MPJPE        : {np.mean(pa):6.2f} mm")
    print(f"  root-rel MPJPE  : {np.mean(rel):6.2f} mm")
    print(f"  abs MPJPE       : {np.mean(ab):6.2f} mm")
    print(f"  root error      : {np.mean(rooterr):6.2f} mm")
    print(f"  PCK-AUC (rel)   : {auc_rel:6.4f}  (0-{args.auc_max:.0f}mm)")
    print(f"  PCK-AUC (PA)    : {auc_pa:6.4f}  (0-{args.auc_max:.0f}mm)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="dual-stream checkpoint (best.pt)")
    ap.add_argument("--limit", default=5e6, type=float)
    ap.add_argument("--batch", default=32, type=int)
    ap.add_argument("--depth_source", default="cache", choices=["render", "cache"])
    ap.add_argument("--dataset", default="hanco", choices=["hanco", "dexycb", "ho3d"],
                    help="which val set to evaluate on (real-depth sets show curriculum generalization)")
    ap.add_argument("--dexycb_root", default="/home/jucpark/DeepLearning/Datasets/Hand Dataset/DexYCB_full/data")
    ap.add_argument("--ho3d_root", default="/home/jucpark/DeepLearning/Datasets/Hand Dataset/HO3D_full")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--max_eval", default=0, type=int, help="cap #frames for a quick run (0=all)")
    ap.add_argument("--auc_max", default=50.0, type=float, help="PCK-AUC upper threshold (mm)")
    ap.add_argument("--rgb_only", action="store_true", help="ablation: zero depth + depth_med")
    args = ap.parse_args()
    main(args)
