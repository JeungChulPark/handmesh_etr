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

Run:
    python train_mobrecon_rgbd.py --exp rgbd_v0 --cfg ./configs_rgbd.yaml --batch 32
    # warm-start from an RGB checkpoint (much faster to a good model):
    python train_mobrecon_rgbd.py --exp rgbd_v0 --epoch 30 --depth_source cache \
        --pretrain pretrain/100.pt
    # RGB-only ablation through the exact same pipeline (4th channel zeroed-out):
    python train_mobrecon_rgbd.py --exp rgb_only --cfg ./configs_rgbd.yaml --rgb_only

    tensorboard --logdir runs --port 6006        # http://localhost:6006
"""

import os
import argparse

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from models.mobrecon_ds import LargeModel_Extra_RGBD
from datasets.hanco_ty import HanCo_ETRI_jitter
from utils import *


def cam2pixel_torch(joints, K):
    x = joints[..., 0] / joints[..., 2] * K[:, 0, 0].unsqueeze(1) + K[:, 0, 2].unsqueeze(1)
    y = joints[..., 1] / joints[..., 2] * K[:, 1, 1].unsqueeze(1) + K[:, 1, 2].unsqueeze(1)
    return torch.stack((x, y), 2)


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

    # Depth knobs (CLI overrides the dataset defaults). For the RGB-only ablation
    # we still emit a 4th channel but zero it, so the tensor shapes / model are
    # identical and the comparison is clean.
    depth_cfg = dict(sigma=args.depth_sigma, dropout_p=args.depth_dropout)
    with_depth = True  # always produce 4 channels; --rgb_only zeroes the depth
    in_chans = 4

    train_dataset = HanCo_ETRI_jitter(
        limit=5e6, with_depth=with_depth, depth_cfg=depth_cfg,
        depth_source=args.depth_source,
    )

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
        model.parameters(), lr=1e-4, betas=(0.9, 0.999), amsgrad=False, eps=1e-8
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

            output_dict = model(img)

            loss_3d = l2_loss(output_dict["keypoints"], kps3d)
            pred_xy = (
                cam2pixel_torch(
                    output_dict["keypoints"] + item_dict["root"][:, None, :].float().to(device),
                    item_dict["cam"].float().to(device),
                )
                / 256.0
            )
            loss_2d = l2_loss(pred_xy, xy)

            iter_loss = loss_3d + loss_2d
            iter_loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += iter_loss.item()
            mpjpe = torch.sqrt(((output_dict["keypoints"] - kps3d) ** 2).sum(dim=-1)).mean() * 1000.0
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
                output_dict = model(img)
                mpjpe = torch.sqrt(((output_dict["keypoints"] - kps3d) ** 2).sum(dim=-1)).mean() * 1000.0
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
    parser.add_argument("--depth_sigma", default=0.005, type=float, help="sensor depth noise std (m)")
    parser.add_argument("--depth_dropout", default=0.05, type=float, help="fraction of depth pixels dropped")
    parser.add_argument("--rgb_only", action="store_true", help="ablation: zero the depth channel")
    parser.add_argument("--depth_source", default="render", choices=["render", "cache"],
                        help="render depth on-the-fly, or load the pre-rendered cache")
    parser.add_argument("--pretrain", default="", type=str,
                        help="RGB checkpoint (e.g. pretrain/100.pt) to warm-start from: "
                             "all weights are loaded; the 3->4 channel stem conv keeps the "
                             "RGB filters and the depth channel is zero-initialised so the "
                             "model starts at the RGB baseline and only improves.")
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)

    main(args)
