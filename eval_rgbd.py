"""Inference + evaluation for the RGB-D hand-joint model.

Loads a checkpoint produced by train_mobrecon_rgbd.py and reports root-relative
MPJPE (mm) on the SAME held-out test split the trainer carves out (random_split
with torch seed 0, 95/5). Also dumps a few qualitative overlays (RGB + GT vs pred
2D, and the depth channel) to --out.

Run:
    python eval_rgbd.py --ckpt mobrecon_ckpt/rgbd_1hr/3.pt --depth_source cache
    python eval_rgbd.py --ckpt mobrecon_ckpt/rgbd_1hr/3.pt --rgb_only   # ablation
"""

import os
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms.functional import to_pil_image

from datasets.hanco_ty import HanCo_ETRI_jitter
from models.mobrecon_ds import LargeModel_Extra_RGBD
from utils import load_cfg, draw_joint2D


def cam2pixel_torch(joints, K):
    x = joints[..., 0] / joints[..., 2] * K[:, 0, 0].unsqueeze(1) + K[:, 0, 2].unsqueeze(1)
    y = joints[..., 1] / joints[..., 2] * K[:, 1, 1].unsqueeze(1) + K[:, 1, 2].unsqueeze(1)
    return torch.stack((x, y), 2)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cfg", default="./configs_rgbd.yaml")
    ap.add_argument("--depth_source", default="cache", choices=["render", "cache"])
    ap.add_argument("--rgb_only", action="store_true", help="zero the depth channel (ablation)")
    ap.add_argument("--n", default=2000, type=int, help="max held-out samples to evaluate")
    ap.add_argument("--batch", default=64, type=int)
    ap.add_argument("--out", default="out/rgbd_eval")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # Rebuild the exact train/test split (trainer: torch.manual_seed(0) -> random_split 95/5).
    torch.manual_seed(0)
    ds = HanCo_ETRI_jitter(limit=5e6, with_depth=True, depth_source=args.depth_source)
    n_train = int(len(ds) * 0.95)
    _, test_set = random_split(ds, [n_train, len(ds) - n_train])
    ds.mode = "test"  # deterministic preprocessing on the underlying dataset
    if args.n and len(test_set) > args.n:
        test_set = torch.utils.data.Subset(test_set, list(range(args.n)))
    print(f"[eval] held-out test samples: {len(test_set)}  (full set {len(ds)})")

    dl = DataLoader(test_set, batch_size=args.batch, shuffle=False, num_workers=8)

    cfg = load_cfg(args.cfg)
    model = LargeModel_Extra_RGBD(cfg).to(dev).eval()
    state = torch.load(args.ckpt, map_location=dev)
    sd = state.get("model_state_dict", state)
    model.load_state_dict(sd)
    print(f"[eval] loaded {args.ckpt} (epoch {state.get('epoch', '?')})  rgb_only={args.rgb_only}")

    per_joint_err = []  # mm, (N,21)
    saved = 0
    for bi, b in enumerate(dl):
        img = b["image"].float().to(dev)
        if args.rgb_only:
            img[:, 3:4] = 0.0
        k3 = b["keypoints3D"].float().to(dev)
        out = model(img)["keypoints"]
        err = torch.sqrt(((out - k3) ** 2).sum(-1)) * 1000.0  # (B,21) mm
        per_joint_err.append(err.cpu())

        if saved < 6:  # qualitative overlays
            xy = b["keypoints2D"].float()
            pred_xy = cam2pixel_torch(
                out + b["root"][:, None, :].float().to(dev), b["cam"].float().to(dev)
            ) / 256.0
            rgb = img[:, :3]
            gt_vis = draw_joint2D(rgb, xy, idx=0)
            pred_vis = draw_joint2D(rgb, pred_xy.cpu(), idx=0)
            to_pil_image(gt_vis).save(os.path.join(args.out, f"{bi:03d}_gt.png"))
            to_pil_image(pred_vis).save(os.path.join(args.out, f"{bi:03d}_pred.png"))
            d = img[0, 3:4].cpu().clamp(-1, 1)
            to_pil_image((d + 1) / 2).save(os.path.join(args.out, f"{bi:03d}_depth.png"))
            saved += 1

    err = torch.cat(per_joint_err, 0)  # (N,21)
    mpjpe = err.mean().item()
    print("\n========== RGB-D EVAL ==========")
    print(f"samples            : {err.shape[0]}")
    print(f"MPJPE (root-rel)   : {mpjpe:.2f} mm")
    print(f"  median per-frame : {err.mean(1).median().item():.2f} mm")
    print(f"  PCK@20mm         : {100.0*(err < 20).float().mean().item():.1f} %")
    print(f"  fingertip MPJPE  : {err[:, [4,8,12,16,20]].mean().item():.2f} mm")
    print(f"  wrist  MPJPE     : {err[:, 0].mean().item():.2f} mm")
    print(f"overlays saved to  : {args.out}/")
    print("================================")


if __name__ == "__main__":
    main()
