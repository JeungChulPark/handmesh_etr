"""Option C: 2D->3D lifter.  MediaPipe 21 (u,v) + per-joint LiDAR depth -> metric
root-relative 3D joints, via a small MLP.  Pixels are DISCARDED -- the model leans
fully on MediaPipe 2D + the depth sampled at each landmark, then applies a learned
bone-length/pose prior.  Tiny and fast (mobile-friendly), and it uses the per-joint
depth Z directly, which the heatmap-input model (B) could not exploit.

Trained and evaluated on the SAME session split as the CNN runs (train on 4 iPhone
sessions, hold out rgbd_captures_04), scored with the exact frame/joint-weighted
MPJPE estimator so the number lines up with finetune-S (25.54mm) and B (25.46mm).

    python scripts/train/train_lifter.py --exp lifter_s --epoch 80
"""

import _bootstrap  # noqa: F401  (repo-root + sibling script imports)

import os
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from datasets.iphone_captures import IPhoneCaptures_RGBD
from train_mobrecon_rgbd import WithJointValid, masked_l2, masked_mpjpe_mm


def sample_z(depth_norm, u01, v01, win=3):
    """Median non-zero normalised depth in a small window at (u,v) (both in [0,1]).
    Returns (z, valid). z==0/valid==0 when the landmark falls on a depth hole."""
    H, W = depth_norm.shape
    ui, vi = int(round(u01 * W)), int(round(v01 * H))
    if not (0 <= ui < W and 0 <= vi < H):
        return 0.0, 0.0
    p = depth_norm[max(0, vi - win):vi + win + 1, max(0, ui - win):ui + win + 1]
    nz = p[p != 0]
    if nz.size == 0:
        return 0.0, 0.0
    return float(np.median(nz)), 1.0


class WithLiftInput(Dataset):
    """Add `lift_in` [21,4] = (u,v centred to [-1,1], sampled depth, depth-valid)."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        d = self.ds[i]
        kps = np.asarray(d["keypoints2D"], np.float32)          # [21,2] in [0,1]
        depth = np.asarray(d["image"][3], np.float32)           # [H,W] normalised
        feat = np.zeros((21, 4), np.float32)
        for j in range(21):
            z, valid = sample_z(depth, kps[j, 0], kps[j, 1])
            feat[j] = (kps[j, 0] * 2 - 1, kps[j, 1] * 2 - 1, z, valid)
        d["lift_in"] = feat
        return d


class Lifter2Dto3D(nn.Module):
    def __init__(self, n=21, fin=4, h=512):
        super().__init__()
        self.n = n
        self.net = nn.Sequential(
            nn.Linear(n * fin, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True),
            nn.Linear(h, n * 3),
        )

    def forward(self, x):                                       # x [B,21,4]
        B = x.shape[0]
        out = self.net(x.reshape(B, -1)).reshape(B, self.n, 3)
        out = out - out[:, 0:1]                                 # wrist -> origin (root-relative)
        return {"keypoints": out}


def build_iphone(root, split, train):
    ds = IPhoneCaptures_RGBD(root=root, mode=("train" if train else "test"),
                             with_depth=True, split=split)
    return WithLiftInput(WithJointValid(ds))


@torch.no_grad()
def evaluate(model, dl, device):
    """Exact frame/joint-weighted MPJPE (mm), identical estimator to scripts/eval/eval_iphone.py."""
    model.eval()
    dist_sum, valid_sum = 0.0, 0.0
    for item in dl:
        x = item["lift_in"].float().to(device)
        kps3d = item["keypoints3D"].float().to(device)
        valid = item["joint_valid"].float().to(device)
        out = model(x)["keypoints"]
        d = torch.sqrt(((out - kps3d) ** 2).sum(dim=-1))
        dist_sum += float((d * valid).sum())
        valid_sum += float(valid.sum())
    return dist_sum / max(valid_sum, 1e-8) * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="lifter_s")
    ap.add_argument("--iphone_root",
                    default="rgbd_captures,rgbd_captures_01,rgbd_captures_02,rgbd_captures_03")
    ap.add_argument("--iphone_eval_root", default="rgbd_captures_04")
    ap.add_argument("--epoch", default=80, type=int)
    ap.add_argument("--batch", default=64, type=int)
    ap.add_argument("--lr", default=1e-3, type=float)
    ap.add_argument("--w2d", default=0.0, type=float, help="(unused; lifter has no reprojection here)")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.join("mobrecon_ckpt", args.exp)
    os.makedirs(ckpt_dir, exist_ok=True)

    train_ds = build_iphone(args.iphone_root, "all", train=True)
    eval_ds = build_iphone(args.iphone_eval_root, "all", train=False)
    print(f"[lifter] train={len(train_ds)}  eval={len(eval_ds)} frames", flush=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=8, drop_last=True)
    eval_dl = DataLoader(eval_ds, batch_size=args.batch, shuffle=False, num_workers=4)

    model = Lifter2Dto3D().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[lifter] Lifter2Dto3D params={n_params/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epoch)

    best = float("inf")
    for epoch in tqdm(range(args.epoch)):
        model.train()
        tot, n = 0.0, 0
        for item in train_dl:
            x = item["lift_in"].float().to(device)
            kps3d = item["keypoints3D"].float().to(device)
            valid = item["joint_valid"].float().to(device)
            opt.zero_grad()
            out = model(x)["keypoints"]
            loss = masked_l2(out, kps3d, valid, dim=3)
            loss.backward()
            opt.step()
            tot += float(masked_mpjpe_mm(out, kps3d, valid)); n += 1
        sched.step()
        test_mpjpe = evaluate(model, eval_dl, device)
        print(f"[epoch {epoch+1:3d}/{args.epoch}] train_mpjpe={tot/max(n,1):6.2f}mm  "
              f"test_mpjpe={test_mpjpe:6.2f}mm  best={min(best,test_mpjpe):6.2f}mm", flush=True)
        if test_mpjpe < best:
            best = test_mpjpe
            torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(),
                        "test_mpjpe": test_mpjpe, "best_test_mpjpe": best},
                       os.path.join(ckpt_dir, "best.pt"))
    print(f"[lifter] BEST session-eval MPJPE = {best:.2f} mm", flush=True)


if __name__ == "__main__":
    main()
