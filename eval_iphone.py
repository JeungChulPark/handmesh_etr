"""Score a 4-channel RGB-D checkpoint on the FROZEN iPhone-eval holdout.

Uses the exact same preprocessing + masked-MPJPE metric as train_mobrecon_rgbd.py,
on the deterministic iPhone-eval split (idx %% 5 == 0), so any checkpoint --
the original ef_iphone_v2, the no-finetune rgbd_real_0613, or the new
finetune/replay/joint runs -- is measured on one identical, never-trained-on set.
Root-relative MPJPE over the supervised (joint_valid) joints, in mm.

    python eval_iphone.py --ckpt mobrecon_ckpt/ef_iphone_v2/best.pt
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.mobrecon_ds import LargeModel_Extra_RGBD
from datasets.iphone_captures import IPhoneCaptures_RGBD
from train_mobrecon_rgbd import WithJointValid, WithHeatmap
from utils import load_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cfg", default="./configs_rgbd.yaml")
    ap.add_argument("--iphone_root",
                    default="rgbd_captures,rgbd_captures_01,rgbd_captures_02,"
                            "rgbd_captures_03,rgbd_captures_04")
    ap.add_argument("--band", default=0.12, type=float)
    ap.add_argument("--batch", default=32, type=int)
    ap.add_argument("--split", default="eval", choices=["all", "train", "eval"],
                    help="'eval' = idx%%5 holdout on --iphone_root; use 'all' with a "
                         "single held-out session dir for a session-level eval.")
    ap.add_argument("--mp_heatmap", action="store_true",
                    help="model expects 21 MediaPipe 2D-heatmap channels (25ch input).")
    ap.add_argument("--hm_sigma", default=6.0, type=float)
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    cfg.MODEL.set_new_allowed(True)
    cfg.MODEL.IN_CHANS = 25 if args.mp_heatmap else 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LargeModel_Extra_RGBD(cfg)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd)
    model.to(device).eval()

    ds = WithJointValid(IPhoneCaptures_RGBD(root=args.iphone_root, mode="test",
                                            with_depth=True, band=args.band,
                                            split=args.split))
    if args.mp_heatmap:
        ds = WithHeatmap(ds, sigma=args.hm_sigma, jitter_px=0.0, drop_p=0.0, train=False)
    # num_workers=0 + fixed seed -> the small bbox-margin jitter is identical for
    # every checkpoint, so scores differ only by the model.
    np.random.seed(0)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)

    # Exact frame/joint-weighted MPJPE: accumulate summed per-joint distance and
    # valid count across ALL eval frames (not a batch-count average), so the number
    # is estimator-consistent no matter the batch size or last-batch remainder.
    dist_sum, valid_sum, n = 0.0, 0.0, 0
    with torch.no_grad():
        for item in dl:
            img = item["image"].float().to(device)
            kps3d = item["keypoints3D"].float().to(device)
            valid = item["joint_valid"].float().to(device)
            out = model(img)
            d = torch.sqrt(((out["keypoints"] - kps3d) ** 2).sum(dim=-1))  # [B,21] m
            dist_sum += float((d * valid).sum())
            valid_sum += float(valid.sum())
            n += img.shape[0]
    mpjpe = dist_sum / max(valid_sum, 1e-8) * 1000.0
    print(f"[eval_iphone] {args.ckpt}: MPJPE = {mpjpe:.2f} mm  ({n} eval frames)")


if __name__ == "__main__":
    main()
