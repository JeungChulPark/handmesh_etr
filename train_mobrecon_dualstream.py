"""Dual-stream RGB-D training (depth-native redesign).

SEPARATE from `train_mobrecon_rgbd.py` (the early-fusion baseline) so the two can
be compared head-to-head. This trains `models.mobrecon_dualstream.MobRecon_DualStream`,
which predicts root-relative joints (RGB stream) AND the global root translation +
metric scale (depth-driven TranslationHead), then assembles absolute joints
`abs = s * rel + root`. See docs/RGBD_DUAL_STREAM_DESIGN.md.

What differs from the baseline:
  * model    : two-stream + TranslationHead (vs early-fusion 4-channel conv stem)
  * losses   : adds L_root (translation), L_scale, L_abs (absolute MPJPE) on top of
               the baseline's root-relative L_pose + 2D reprojection
  * data     : HanCo with return_abs_depth=True -> also yields depth_med (raw-depth
               median = absolute distance), the one absolute-z signal the centred
               depth channel discards
  * metrics  : logs root-rel MPJPE (comparable to baseline) AND abs MPJPE + root error

The val split uses the same seed(0) + 0.95/0.05 split as the baseline, so the two
runs share the same held-out frames for a fair comparison.

Run:
    python train_mobrecon_dualstream.py --exp ds_v0 --epoch 30 --depth_source cache \
        --pretrain pretrain/100.pt
    tensorboard --logdir runs --port 6006
"""

import os
import argparse

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split, Dataset

from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import ConcatDataset, Dataset

from models.mobrecon_dualstream import MobRecon_DualStream
from datasets.hanco_ty import HanCo_ETRI_jitter
from datasets.dexycb import DexYCB_RGBD
from datasets.ho3d import HO3D_RGBD, H2O3D_RGBD
from utils import *


class WithMeta(Dataset):
    """Ensure every sample carries `joint_valid` [21] and `depth_med` scalar so a
    ConcatDataset of mixed real-depth loaders collates cleanly (PyTorch's default
    collate needs identical keys). Loaders that already emit them are untouched."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        d = self.ds[i]
        if "joint_valid" not in d:
            d["joint_valid"] = np.ones(21, np.float32)
        if "depth_med" not in d:
            d["depth_med"] = np.float32(0.0)
        return d


def build_dataset(args, depth_cfg):
    """Assemble the (possibly concatenated) RGB-D training set from --datasets.

    Every loader is opened with return_abs_depth=True so it emits `depth_med`
    (raw-depth median = absolute distance) for the TranslationHead, and wrapped in
    WithMeta so a mixed batch collates. hanco = synthetic depth; dexycb/ho3d/h2o3d
    = real sensor depth (the real-depth curriculum)."""
    names = [n.strip() for n in args.datasets.split(",") if n.strip()]
    parts = []
    for n in names:
        if n == "hanco":
            ds = HanCo_ETRI_jitter(limit=args.limit, with_depth=True, depth_cfg=depth_cfg,
                                   depth_source=args.depth_source, return_abs_depth=True)
        elif n == "dexycb":
            ds = DexYCB_RGBD(root=args.dexycb_root, mode="train", with_depth=True,
                             flip_left=True, depth_cfg=depth_cfg, return_abs_depth=True)
        elif n == "ho3d":
            ds = HO3D_RGBD(root=args.ho3d_root, mode="train", with_depth=True,
                           depth_cfg=depth_cfg, return_abs_depth=True)
        elif n == "h2o3d":
            ds = H2O3D_RGBD(root=args.h2o3d_root, mode="train", with_depth=True,
                            depth_cfg=depth_cfg, return_abs_depth=True)
        else:
            raise ValueError(f"unknown dataset '{n}' (supported: hanco,dexycb,ho3d,h2o3d)")
        parts.append(WithMeta(ds))
    print(f"[data] datasets={names}  total={sum(len(p) for p in parts)} frames", flush=True)
    return parts[0] if len(parts) == 1 else ConcatDataset(parts)


def cam2pixel_torch(joints, K):
    x = joints[..., 0] / joints[..., 2] * K[:, 0, 0].unsqueeze(1) + K[:, 0, 2].unsqueeze(1)
    y = joints[..., 1] / joints[..., 2] * K[:, 1, 1].unsqueeze(1) + K[:, 1, 2].unsqueeze(1)
    return torch.stack((x, y), 2)


def masked_l2(pred, gt, valid, dim):
    v = valid[..., None]
    return ((pred - gt) * v).pow(2).sum() / (v.sum() * dim + 1e-8)


def masked_mpjpe_mm(pred, gt, valid):
    d = torch.sqrt(((pred - gt) ** 2).sum(dim=-1))  # [B,21]
    return (d * valid).sum() / (valid.sum() + 1e-8) * 1000.0


def _rgb(img):
    """First 3 channels back to a CHW uint8-ish tensor for TensorBoard."""
    return img[:, :3].detach().cpu()


def main(args, log_every=500):
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Current cuda device ", torch.cuda.device_count())

    ckpt_dir = os.path.join("mobrecon_ckpt", args.exp)
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join("runs", args.exp))

    depth_cfg = dict(sigma=args.depth_sigma, dropout_p=args.depth_dropout,
                     n_holes=args.depth_holes, hole_frac=args.depth_hole_frac,
                     lidar_sim=args.lidar_sim, lidar_downscale=args.lidar_downscale,
                     lidar_edge_drop=args.lidar_edge_drop)
    if args.lidar_sim:
        print("[depth] LiDAR-simulation augmentation ON "
              f"(downscale={args.lidar_downscale}, edge_drop={args.lidar_edge_drop})", flush=True)

    # Build the (possibly concatenated) RGB-D set; every loader emits depth_med.
    train_dataset = build_dataset(args, depth_cfg)

    # Same seed(0) + 0.95/0.05 split as the baseline -> identical held-out frames.
    len_dataset = int(len(train_dataset) * 0.95)
    left_len = len(train_dataset) - len_dataset
    train_dataset, test_dataset = random_split(train_dataset, [len_dataset, left_len])

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=False
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False
    )

    model = MobRecon_DualStream(cfg=None, pose_in_chans=args.pose_in_chans)

    # Warm-start: load the RGB backbone weights from an RGB checkpoint by remapping
    # 'backbone.*' -> 'rgb_backbone.*'. The DepthEncoder and TranslationHead start
    # fresh. For a 4-channel hybrid pose backbone the stem conv is grafted 3->4
    # (RGB filters kept on [0:3], depth channel [3] zero-init), exactly like the
    # early-fusion baseline, so the hybrid starts at the RGB baseline.
    if not getattr(args, "resume", "") and getattr(args, "pretrain", "") and os.path.isfile(args.pretrain):
        ckpt = torch.load(args.pretrain, map_location="cpu")
        src = ckpt.get("model_state_dict", ckpt)
        tgt = model.state_dict()
        stem_key = "rgb_backbone.pre_layer.0.0.weight"
        remap = {}
        for k, v in src.items():
            nk = ("rgb_backbone." + k[len("backbone."):]) if k.startswith("backbone.") else k
            if nk == stem_key and nk in tgt and v.shape != tgt[nk].shape:
                w = tgt[nk].clone()      # (out, 4, 3, 3)
                w[:, :3] = v             # graft RGB filters
                w[:, 3:] = 0.0           # depth channel starts as no-op
                remap[nk] = w
            elif nk in tgt and v.shape == tgt[nk].shape:
                remap[nk] = v
        model.load_state_dict(remap, strict=False)
        print(f"[warm-start] {args.pretrain}: loaded {len(remap)}/{len(tgt)} tensors into "
              f"rgb_backbone; DepthEncoder + TranslationHead reinit", flush=True)
    elif not getattr(args, "resume", "") and getattr(args, "pretrain", ""):
        print(f"[warm-start] WARNING: --pretrain {args.pretrain} not found; training from scratch", flush=True)

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    if getattr(args, "cosine", False):
        # Cosine anneal LR -> 0 over the whole run (stepped per-iteration). Use for
        # a final low-LR polish: lets the model settle into a finer minimum instead
        # of oscillating at a constant LR. T_max = total iterations.
        t_max = len(train_dataloader) * args.epoch
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=0.0)
        print(f"[sched] CosineAnnealingLR over {t_max} iters (lr {args.lr:g} -> 0)", flush=True)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100000, gamma=0.9)

    start_epoch = 0
    best_test_abs = float("inf")
    if getattr(args, "resume", "") and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0))
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if ckpt.get("best_test_abs") is not None:
            best_test_abs = float(ckpt["best_test_abs"])
        print(f"[resume] {args.resume}: from epoch {start_epoch} (target {args.epoch}); "
              f"best abs {best_test_abs:.2f}mm", flush=True)

    for epoch in tqdm(range(start_epoch, args.epoch), leave=True, position=1):
        iter = 0
        epoch_loss = 0
        epoch_rel = 0
        epoch_abs = 0
        model.train()

        for i, item_dict in enumerate(tqdm(train_dataloader, leave=False, position=2)):
            steps = len(train_dataloader) * epoch + i
            optimizer.zero_grad()

            img = item_dict["image"].float().to(device)
            if args.rgb_only:
                img[:, 3:4] = 0.0  # ablation: drop the depth signal
            kps3d = item_dict["keypoints3D"].float().to(device)        # root-relative GT [B,21,3]
            xy = item_dict["keypoints2D"].float().to(device)
            root_gt = item_dict["root"].float().to(device)             # [B,3] absolute root
            valid = item_dict.get("joint_valid")
            valid = (valid.float().to(device) if valid is not None
                     else torch.ones(kps3d.shape[0], 21, device=device))
            depth_med = item_dict["depth_med"].float().to(device)      # [B] raw-depth median
            if args.rgb_only:
                depth_med = torch.zeros_like(depth_med)
            abs_gt = kps3d + root_gt[:, None, :]

            out = model(img, depth_med)

            loss_pose = masked_l2(out["keypoints"], kps3d, valid, dim=3)
            loss_root = (out["root"] - root_gt).pow(2).mean()
            loss_scale = (out["scale"] - 1.0).abs().mean()
            loss_abs = masked_l2(out["keypoints_abs"], abs_gt, valid, dim=3)
            pred_xy = cam2pixel_torch(out["keypoints_abs"], item_dict["cam"].float().to(device)) / 256.0
            loss_2d = masked_l2(pred_xy, xy, valid, dim=2)

            iter_loss = (loss_pose + args.w_root * loss_root + args.w_scale * loss_scale
                         + args.w_abs * loss_abs + args.w2d * loss_2d)
            iter_loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += iter_loss.item()
            rel_mpjpe = masked_mpjpe_mm(out["keypoints"], kps3d, valid)
            abs_mpjpe = masked_mpjpe_mm(out["keypoints_abs"], abs_gt, valid)
            epoch_rel += rel_mpjpe
            epoch_abs += abs_mpjpe
            iter += 1

            if steps % log_every == 0:
                root_err = (out["root"] - root_gt).pow(2).sum(-1).sqrt().mean() * 1000.0
                writer.add_scalar("loss/total", iter_loss.item(), steps)
                writer.add_scalar("loss/pose", loss_pose.item(), steps)
                writer.add_scalar("loss/root", loss_root.item(), steps)
                writer.add_scalar("loss/scale", loss_scale.item(), steps)
                writer.add_scalar("loss/abs", loss_abs.item(), steps)
                writer.add_scalar("loss/2d", loss_2d.item(), steps)
                writer.add_scalar("mpjpe/train_rel", float(rel_mpjpe), steps)
                writer.add_scalar("mpjpe/train_abs", float(abs_mpjpe), steps)
                writer.add_scalar("root/train_err_mm", float(root_err), steps)
                writer.add_scalar("scale/mean", float(out["scale"].mean()), steps)
                rgb = _rgb(img)
                writer.add_image("train/gt", draw_joint2D(rgb, xy.detach().cpu(), idx=0),
                                 steps, dataformats="HWC")
                writer.add_image("train/pred", draw_joint2D(rgb, pred_xy.detach().cpu(), idx=0),
                                 steps, dataformats="HWC")

        writer.add_scalar("mpjpe/train_rel_epoch", float(epoch_rel) / iter, steps)
        writer.add_scalar("mpjpe/train_abs_epoch", float(epoch_abs) / iter, steps)

        if (epoch + 1) % 2 == 0 or (epoch + 1) == args.epoch:
            torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_test_abs": best_test_abs},
                       os.path.join(ckpt_dir, f"{epoch+1}.pt"))

        # --- eval ---
        test_iter = 0
        test_rel = 0.0
        test_abs = 0.0
        test_root = 0.0
        model.eval()
        with torch.no_grad():
            for item_dict in tqdm(test_dataloader, leave=False, position=2):
                img = item_dict["image"].float().to(device)
                if args.rgb_only:
                    img[:, 3:4] = 0.0
                kps3d = item_dict["keypoints3D"].float().to(device)
                root_gt = item_dict["root"].float().to(device)
                valid = item_dict.get("joint_valid")
                valid = (valid.float().to(device) if valid is not None
                         else torch.ones(kps3d.shape[0], 21, device=device))
                depth_med = item_dict["depth_med"].float().to(device)
                if args.rgb_only:
                    depth_med = torch.zeros_like(depth_med)
                abs_gt = kps3d + root_gt[:, None, :]
                out = model(img, depth_med)
                test_rel += masked_mpjpe_mm(out["keypoints"], kps3d, valid)
                test_abs += masked_mpjpe_mm(out["keypoints_abs"], abs_gt, valid)
                test_root += (out["root"] - root_gt).pow(2).sum(-1).sqrt().mean() * 1000.0
                test_iter += 1

        test_rel = float(test_rel / max(test_iter, 1))
        test_abs = float(test_abs / max(test_iter, 1))
        test_root = float(test_root / max(test_iter, 1))
        print(f"[epoch {epoch+1:3d}/{args.epoch}] "
              f"test_rel={test_rel:6.2f}mm  test_abs={test_abs:7.2f}mm  "
              f"root_err={test_root:7.2f}mm  best_abs={min(best_test_abs, test_abs):7.2f}mm", flush=True)
        writer.add_scalar("mpjpe/test_rel", test_rel, steps)
        writer.add_scalar("mpjpe/test_abs", test_abs, steps)
        writer.add_scalar("root/test_err_mm", test_root, steps)

        if test_abs < best_test_abs:
            best_test_abs = test_abs
            torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "test_abs": test_abs, "test_rel": test_rel, "test_root": test_root,
                        "best_test_abs": best_test_abs},
                       os.path.join(ckpt_dir, "best.pt"))
            print(f"[epoch {epoch+1:3d}] new best -> best.pt ({test_abs:.2f}mm abs)", flush=True)
        writer.flush()

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, help="experiment name (runs/<exp>, mobrecon_ckpt/<exp>)")
    parser.add_argument("--batch", default=32, type=int, dest="batch_size")
    parser.add_argument("--epoch", default=30, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--limit", default=5e6, type=float, help="max HanCo samples")
    parser.add_argument("--w2d", default=0.1, type=float, help="2D reprojection weight")
    parser.add_argument("--w_root", default=1.0, type=float, help="root translation loss weight")
    parser.add_argument("--w_scale", default=0.01, type=float, help="scale regulariser weight")
    parser.add_argument("--w_abs", default=1.0, type=float, help="absolute-joint loss weight")
    parser.add_argument("--depth_sigma", default=0.005, type=float)
    parser.add_argument("--depth_dropout", default=0.05, type=float)
    parser.add_argument("--depth_holes", default=2, type=int)
    parser.add_argument("--depth_hole_frac", default=0.15, type=float)
    parser.add_argument("--lidar_sim", action="store_true")
    parser.add_argument("--lidar_downscale", default=4, type=int)
    parser.add_argument("--lidar_edge_drop", default=0.5, type=float)
    parser.add_argument("--pose_in_chans", default=3, type=int, choices=[3, 4],
                        help="pose backbone input channels: 3=RGB-only (orig dual-stream), "
                             "4=hybrid RGB-D (depth also aids pose, like the early-fusion baseline)")
    parser.add_argument("--cosine", action="store_true",
                        help="cosine-anneal LR to 0 over the run (for a final low-LR polish)")
    parser.add_argument("--clip", default=0.0, type=float,
                        help="grad-norm clip (0=off). Use ~1.0 to stabilise multi-dataset fine-tuning.")
    parser.add_argument("--rgb_only", action="store_true", help="ablation: zero depth + depth_med")
    parser.add_argument("--depth_source", default="cache", choices=["render", "cache"])
    parser.add_argument("--datasets", default="hanco",
                        help="comma list to concat: hanco,dexycb,ho3d,h2o3d (real-depth curriculum)")
    parser.add_argument("--dexycb_root", default="", help="DexYCB root (…/DexYCB_full/data)")
    parser.add_argument("--ho3d_root", default="", help="HO3D root (…/HO3D_full, contains train/)")
    parser.add_argument("--h2o3d_root", default="", help="H2O-3D root (contains train/)")
    parser.add_argument("--pretrain", default="", type=str, help="RGB ckpt to warm-start rgb_backbone")
    parser.add_argument("--resume", default="", type=str)
    args = parser.parse_args()
    main(args)
