"""RGB-D retraining for the MobRecon-style hand joint regressor.

This mirrors `train_mobrecon_jittering_3d.py` but trains the early-fusion 4-channel
model `LargeModel_Extra_RGBD` on HanCo with a synthetic metric-depth channel
(see datasets/depth_synth.py). Everything else -- losses (3D L2 + 2D reprojection
L2), optimiser, schedule, logging -- is kept identical so the only variable under
study is the depth channel. Metrics/visuals are logged directly to TensorBoard
(no wandb) under runs/<exp>; view with `tensorboard --logdir runs`.

Why synthetic depth: HanCo ships multi-view mocap GT but no sensor depth. We
render a metric depth map from the GT geometry and corrupt it with a tunable
sensor noise model, which lets us (a) validate the RGB-D architecture end-to-end
today and (b) sweep sensor quality (sigma / dropout) to estimate how good a real
sensor must be before committing to a real paired-depth capture. See
RGBD_TRAINING.md for how to swap in real MANO verts or a real sensor depth map.

Multiple datasets can be concatenated via --datasets (each loader emits the same
contract; a per-joint `joint_valid` mask lets the ToF depth-only sets, which only
supervise a subset of the 21 joints, train through a masked loss without polluting
the absent slots). Real-depth (DexYCB/HO3D) and ToF (MSRA/ICVL/NYU) sets bring in
*real* sensor depth instead of HanCo's synthetic render. See datasets/rgbd_datasets.md.

Run:
    python train_mobrecon_rgbd.py --exp rgbd_v0 --cfg ./configs_rgbd.yaml --batch 32
    # warm-start from an RGB checkpoint (much faster to a good model):
    python train_mobrecon_rgbd.py --exp rgbd_v0 --epoch 30 --depth_source cache \
        --pretrain pretrain/100.pt
    # train on real depth, HanCo + DexYCB + HO3D + MSRA(ToF) concatenated:
    python train_mobrecon_rgbd.py --exp rgbd_real --datasets hanco,dexycb,ho3d,msra \
        --dexycb_root /data/DexYCB --ho3d_root /data/HO3D_v3 \
        --msra_root /data/cvpr15_MSRAHandGestureDB --pretrain pretrain/100.pt
    # RGB-only ablation through the exact same pipeline (4th channel zeroed-out):
    python train_mobrecon_rgbd.py --exp rgb_only --cfg ./configs_rgbd.yaml --rgb_only

    tensorboard --logdir runs --port 6006        # http://localhost:6006
"""

import os
import argparse

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split, ConcatDataset, Dataset
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from models.mobrecon_ds import LargeModel_Extra_RGBD
from datasets.hanco_ty import HanCo_ETRI_jitter
from datasets.dexycb import DexYCB_RGBD
from datasets.ho3d import HO3D_RGBD, H2O3D_RGBD
from datasets.tof_depth import (MSRA_RGBD, ICVL_RGBD, NYU_RGBD,
                                 BigHand_RGBD, HANDS17_RGBD, FPHA_RGBD)
from datasets.oakink_contactpose import ContactPose_RGBD, OakInk_RGBD
from utils import *


def cam2pixel_torch(joints, K):
    x = joints[..., 0] / joints[..., 2] * K[:, 0, 0].unsqueeze(1) + K[:, 0, 2].unsqueeze(1)
    y = joints[..., 1] / joints[..., 2] * K[:, 1, 1].unsqueeze(1) + K[:, 1, 2].unsqueeze(1)
    return torch.stack((x, y), 2)


class WithJointValid(Dataset):
    """Wrap a dataset so every sample carries a `joint_valid` [21] mask.

    HanCo/DexYCB/HO3D supervise all 21 joints; the ToF depth-only sets
    (MSRA/ICVL/NYU) supervise a subset and already emit `joint_valid`. PyTorch's
    default collate requires identical keys across a mixed batch, so this injects
    an all-ones mask where it is missing -> a single ConcatDataset can mix any of
    the loaders without a collate error.
    """

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        d = self.ds[i]
        if "joint_valid" not in d:
            d["joint_valid"] = np.ones(21, np.float32)
        return d


def masked_l2(pred, gt, valid, dim):
    """MSE over supervised joints only. With `valid` all-ones this is exactly
    nn.MSELoss() (same numerator, same B*21*dim denominator), so a HanCo-only run
    is numerically unchanged; partial-joint ToF sets simply skip absent slots."""
    v = valid[..., None]
    return ((pred - gt) * v).pow(2).sum() / (v.sum() * dim + 1e-8)


def masked_mpjpe_mm(pred, gt, valid):
    """Mean per-joint position error (mm) over supervised joints; == .mean() when
    valid is all-ones."""
    d = torch.sqrt(((pred - gt) ** 2).sum(dim=-1))  # [B,21]
    return (d * valid).sum() / (valid.sum() + 1e-8) * 1000.0


def build_dataset(args, depth_cfg):
    """Assemble the (possibly concatenated) RGB-D training set from --datasets.

    Every part is wrapped in WithJointValid so a mixed batch collates cleanly.
    Real-depth sets (dexycb/ho3d) and ToF sets (msra/icvl/nyu) load their own
    sensor depth; hanco renders/loads the synthetic depth as before.
    """
    names = [n.strip() for n in args.datasets.split(",") if n.strip()]
    parts = []
    for n in names:
        if n == "hanco":
            ds = HanCo_ETRI_jitter(limit=5e6, with_depth=True, depth_cfg=depth_cfg,
                                   depth_source=args.depth_source)
        elif n == "dexycb":
            ds = DexYCB_RGBD(root=args.dexycb_root, mode="train", with_depth=True,
                             flip_left=True, depth_cfg=depth_cfg)
        elif n == "ho3d":
            ds = HO3D_RGBD(root=args.ho3d_root, mode="train", with_depth=True,
                           depth_cfg=depth_cfg)
        elif n == "h2o3d":
            ds = H2O3D_RGBD(root=args.h2o3d_root, mode="train", with_depth=True,
                            depth_cfg=depth_cfg)
        elif n == "msra":
            ds = MSRA_RGBD(root=args.msra_root, mode="train", with_depth=True)
        elif n == "icvl":
            ds = ICVL_RGBD(root=args.icvl_root, mode="train", with_depth=True)
        elif n == "nyu":
            ds = NYU_RGBD(root=args.nyu_root, mode="train", with_depth=True)
        elif n == "bighand":
            ds = BigHand_RGBD(root=args.bighand_root, mode="train", with_depth=True)
        elif n == "hands17":
            ds = HANDS17_RGBD(root=args.hands17_root, mode="train", with_depth=True)
        elif n == "fpha":
            ds = FPHA_RGBD(root=args.fpha_root, mode="train", with_depth=True)
        elif n == "contactpose":
            ds = ContactPose_RGBD(data_dir=args.contactpose_dir, mode="train", with_depth=True)
        elif n == "oakink":
            # NOTE: oikit exposes no sensor depth -> OakInk depth is RENDERED (synthetic)
            ds = OakInk_RGBD(data_split="train", mode="train", with_depth=True)
        else:
            raise ValueError(f"unknown dataset '{n}' in --datasets")
        parts.append(WithJointValid(ds))
        print(f"[build_dataset] + {n}: {len(ds)} frames", flush=True)
    return parts[0] if len(parts) == 1 else ConcatDataset(parts)


def _rgb(img):
    """First 3 channels for visualisation (img may be [B,4,H,W])."""
    return img[:, :3]


def _depth_vis(img):
    """Map the depth channel (in [-1,1]) to a [0,1] grey image for logging."""
    if img.shape[1] < 4:
        return None
    d = img[:1, 3:4, :, :].detach().cpu()  # first sample, [1,1,H,W]
    return (d.clamp(-1, 1) + 1.0) / 2.0


def main(args, log_every=500):
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Current cuda device ", torch.cuda.device_count())

    ckpt_dir = os.path.join("mobrecon_ckpt", args.exp)
    os.makedirs(ckpt_dir, exist_ok=True)

    # TensorBoard logging (no wandb). View with: tensorboard --logdir runs
    writer = SummaryWriter(os.path.join("runs", args.exp))

    # Depth knobs (CLI overrides every dataset's defaults — hanco, dexycb, ho3d).
    # Stronger dropout/holes simulate a passive-stereo sensor (ZED) dropping depth
    # in low-texture / occluded regions, which trains the model NOT to over-rely on
    # depth (the ablation showed it collapses to 106mm when depth is removed).
    depth_cfg = dict(sigma=args.depth_sigma, dropout_p=args.depth_dropout,
                     n_holes=args.depth_holes, hole_frac=args.depth_hole_frac,
                     lidar_sim=args.lidar_sim, lidar_downscale=args.lidar_downscale,
                     lidar_edge_drop=args.lidar_edge_drop)
    if args.lidar_sim:
        print("[depth] LiDAR-simulation augmentation ON "
              f"(downscale={args.lidar_downscale}, edge_drop={args.lidar_edge_drop}) "
              "-> training depth coarsened to mimic iPhone LiDAR", flush=True)
    with_depth = True  # always produce 4 channels; --rgb_only zeroes the depth
    in_chans = 4

    train_dataset = build_dataset(args, depth_cfg)

    len_dataset = int(len(train_dataset) * 0.95)
    left_len = len(train_dataset) - len_dataset
    train_dataset, test_dataset = random_split(train_dataset, [len_dataset, left_len])
    test_dataset.mode = "test"

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=False
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False
    )

    # Force in_chans=4 on the model regardless of cfg, to match the dataloader.
    cfg.MODEL.set_new_allowed(True)
    cfg.MODEL.IN_CHANS = in_chans
    model = LargeModel_Extra_RGBD(cfg)

    # Warm-start from an RGB checkpoint: load every weight, and adapt the stem
    # conv from 3->4 input channels by keeping the RGB filters on channels [0:3]
    # and zero-initialising the new depth channel [3]. The model therefore starts
    # functionally identical to the RGB baseline and learns to exploit depth.
    if getattr(args, "pretrain", "") and os.path.isfile(args.pretrain):
        ckpt = torch.load(args.pretrain, map_location="cpu")
        src = ckpt.get("model_state_dict", ckpt)
        tgt = model.state_dict()
        stem_key = "backbone.pre_layer.0.0.weight"
        adapted = {}
        for k, v in src.items():
            if k == stem_key and k in tgt and v.shape != tgt[k].shape:
                w = tgt[k].clone()              # (out, 4, 3, 3)
                w[:, :3] = v                    # graft RGB filters
                w[:, 3:] = 0.0                  # depth channel starts as no-op
                adapted[k] = w
            elif k in tgt and v.shape == tgt[k].shape:
                adapted[k] = v
        missing = [k for k in tgt if k not in adapted]
        model.load_state_dict(adapted, strict=False)
        print(f"[warm-start] {args.pretrain}: loaded {len(adapted)}/{len(tgt)} tensors "
              f"(stem 3->4 grafted, depth ch zero-init); {len(missing)} reinit", flush=True)
    elif getattr(args, "pretrain", ""):
        print(f"[warm-start] WARNING: --pretrain {args.pretrain} not found; training from scratch", flush=True)

    model.to(device)

    l2_loss = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), amsgrad=False, eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100000, gamma=0.9)

    best_test_mpjpe = float("inf")
    for epoch in tqdm(range(0, args.epoch), leave=True, position=1):
        iter = 0
        epoch_loss = 0
        epoch_mpjpe = 0
        model.train()

        for i, item_dict in enumerate(tqdm(train_dataloader, leave=False, position=2)):
            steps = len(train_dataloader) * epoch + i
            optimizer.zero_grad()

            img = item_dict["image"].float().to(device)
            if args.rgb_only:
                img[:, 3:4] = 0.0  # ablation: keep 4 channels, drop the depth signal
            kps3d = item_dict["keypoints3D"].float().to(device)
            xy = item_dict["keypoints2D"].float().to(device)
            valid = item_dict["joint_valid"].float().to(device)  # [B,21]

            output_dict = model(img)

            loss_3d = masked_l2(output_dict["keypoints"], kps3d, valid, dim=3)
            pred_xy = (
                cam2pixel_torch(
                    output_dict["keypoints"] + item_dict["root"][:, None, :].float().to(device),
                    item_dict["cam"].float().to(device),
                )
                / 256.0
            )
            loss_2d = masked_l2(pred_xy, xy, valid, dim=2)

            # Down-weight the 2D reprojection term: across mixed datasets its scale
            # varies with each camera's intrinsics, so at w2d=1 it dominates and
            # destabilises the 3D objective we actually deploy (ZED). w2d~0.1 keeps
            # 2D as a light regulariser while 3D MPJPE drives training.
            iter_loss = loss_3d + args.w2d * loss_2d
            iter_loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += iter_loss.item()
            mpjpe = masked_mpjpe_mm(output_dict["keypoints"], kps3d, valid)
            epoch_mpjpe += mpjpe
            iter += 1

            if steps % log_every == 0:
                writer.add_scalar("loss/total", iter_loss.item(), steps)
                writer.add_scalar("loss/3d", loss_3d.item(), steps)
                writer.add_scalar("loss/2d", loss_2d.item(), steps)
                writer.add_scalar("mpjpe/train_iter", float(mpjpe), steps)
                rgb = _rgb(img)
                writer.add_image("train/gt",
                                 draw_joint2D(rgb, xy.detach().cpu(), idx=0), steps, dataformats="HWC")
                writer.add_image("train/pred_25d",
                                 draw_joint2D(rgb, pred_xy.detach().cpu(), idx=0), steps, dataformats="HWC")
                dvis = _depth_vis(img)
                if dvis is not None:
                    writer.add_image("train/depth", dvis[0], steps)  # [1,H,W]

        writer.add_scalar("loss/train_epoch", epoch_loss / iter, steps)
        writer.add_scalar("mpjpe/train_epoch", float(epoch_mpjpe) / iter, steps)

        if (epoch + 1) % 2 == 0 or (epoch + 1) == args.epoch:
            torch.save(
                {"epoch": epoch + 1, "model_state_dict": model.state_dict()},
                os.path.join(ckpt_dir, f"{epoch+1}.pt"),
            )

        # --- eval ---
        test_iter = 0
        test_epoch_mpjpe = 0
        model.eval()
        with torch.no_grad():
            for i, item_dict in enumerate(tqdm(test_dataloader, leave=False, position=2)):
                img = item_dict["image"].float().to(device)
                if args.rgb_only:
                    img[:, 3:4] = 0.0
                kps3d = item_dict["keypoints3D"].float().to(device)
                valid = item_dict["joint_valid"].float().to(device)
                output_dict = model(img)
                mpjpe = masked_mpjpe_mm(output_dict["keypoints"], kps3d, valid)
                test_epoch_mpjpe += mpjpe
                test_iter += 1

        test_mpjpe = float(test_epoch_mpjpe / max(test_iter, 1))
        train_mpjpe = float(epoch_mpjpe / iter)
        print(f"[epoch {epoch+1:3d}/{args.epoch}] train_mpjpe={train_mpjpe:6.2f}mm  "
              f"test_mpjpe={test_mpjpe:6.2f}mm  best={min(best_test_mpjpe, test_mpjpe):6.2f}mm", flush=True)
        if test_mpjpe < best_test_mpjpe:
            best_test_mpjpe = test_mpjpe
            torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(),
                        "test_mpjpe": test_mpjpe}, os.path.join(ckpt_dir, "best.pt"))
            print(f"[epoch {epoch+1:3d}] new best -> saved best.pt ({test_mpjpe:.2f}mm)", flush=True)

        writer.add_scalar("mpjpe/test_epoch", test_mpjpe, steps)
        xy = item_dict["keypoints2D"].float()
        pred_xy = (
            cam2pixel_torch(
                output_dict["keypoints"] + item_dict["root"][:, None, :].float().to(device),
                item_dict["cam"].float().to(device),
            )
            / 256.0
        )
        rgb = _rgb(img)
        writer.add_image("test/gt", draw_joint2D(rgb, xy, idx=0), steps, dataformats="HWC")
        writer.add_image("test/pred_25d",
                         draw_joint2D(rgb, pred_xy.detach().cpu(), idx=0), steps, dataformats="HWC")
        writer.flush()

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, help="Save experiment name")
    parser.add_argument("--batch", default=32, type=int, dest="batch_size")
    parser.add_argument("--epoch", default=200, type=int, dest="epoch")
    parser.add_argument("--cfg", default="./configs_rgbd.yaml")
    parser.add_argument("--lr", default=1e-4, type=float, help="AdamW learning rate (use ~1e-5 to fine-tune a warm-started model)")
    parser.add_argument("--w2d", default=1.0, type=float, help="weight on the 2D reprojection loss (~0.1 for mixed-dataset stability)")
    parser.add_argument("--depth_sigma", default=0.005, type=float, help="sensor depth noise std (m)")
    parser.add_argument("--depth_dropout", default=0.05, type=float, help="fraction of depth pixels dropped (raise for ZED/passive-stereo robustness)")
    parser.add_argument("--depth_holes", default=2, type=int, help="number of rectangular depth dropout holes")
    parser.add_argument("--depth_hole_frac", default=0.15, type=float, help="max hole side as a fraction of the crop")
    parser.add_argument("--lidar_sim", action="store_true", help="simulate iPhone LiDAR (coarse/blocky depth + finger/edge dropout + ToF noise) on all RGB-D sets")
    parser.add_argument("--lidar_downscale", default=4, type=int, help="LiDAR coarse-grid block factor on the 256 crop")
    parser.add_argument("--lidar_edge_drop", default=0.5, type=float, help="prob. of dropping finger/edge depth pixels (LiDAR misses thin geometry)")
    parser.add_argument("--rgb_only", action="store_true", help="ablation: zero the depth channel")
    parser.add_argument("--depth_source", default="render", choices=["render", "cache"],
                        help="render depth on-the-fly, or load the pre-rendered cache (hanco)")
    parser.add_argument("--datasets", default="hanco",
                        help="comma list to concat: hanco,dexycb,ho3d,h2o3d,msra,icvl,nyu,bighand,hands17,fpha,contactpose,oakink")
    parser.add_argument("--dexycb_root", default="", help="DexYCB root ($DEX_YCB_DIR)")
    parser.add_argument("--ho3d_root", default="", help="HO3D root (contains train/)")
    parser.add_argument("--h2o3d_root", default="", help="H2O-3D root (contains train/)")
    parser.add_argument("--msra_root", default="", help="cvpr15_MSRAHandGestureDB root")
    parser.add_argument("--icvl_root", default="", help="ICVL root (Depth/ + labels.txt)")
    parser.add_argument("--nyu_root", default="", help="NYU train/ dir (joint_data.mat)")
    parser.add_argument("--bighand_root", default="", help="BigHand2.2M root (Training_Annotation.txt)")
    parser.add_argument("--hands17_root", default="", help="HANDS17 root (training/)")
    parser.add_argument("--fpha_root", default="", help="FPHA root (Video_files/ + Hand_pose_annotation_v1_1/)")
    parser.add_argument("--contactpose_dir", default="", help="ContactPose data dir (needs the toolkit on PYTHONPATH)")
    parser.add_argument("--pretrain", default="", type=str,
                        help="RGB checkpoint (e.g. pretrain/100.pt) to warm-start from: "
                             "all weights are loaded; the 3->4 channel stem conv keeps the "
                             "RGB filters and the depth channel is zero-initialised so the "
                             "model starts at the RGB baseline and only improves.")
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)

    main(args)
