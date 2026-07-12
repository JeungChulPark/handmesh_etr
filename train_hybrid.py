"""Hybrid lifter: MobRecon backbone REFINES the MediaPipe-2D + depth lifter.

Reintroduces the pixel backbone with an explicit job:
  * geometric branch: MediaPipe (u,v) + LiDAR depth  -> strong 3D prior (the Z-lifter)
  * pixel branch: MobRecon (LargeModel_Extra_RGBD) on the hand crop -> a monocular
    pose cue p_cnn (shape + relative depth from PIXELS), warm-started from rgbd_real_0613.
  * fusion head: per-joint (du, dv, dZ) so PIXELS can (a) CORRECT MediaPipe's 2D
    (du,dv) and (b) supply a monocular Z cue (dZ) where the sensor depth is bad.
        u' = u+du,  v' = v+dv,  Z = z_sampled+dZ,   X=(u'-cx)/fx*Z ...

Loss = 3D L2 + bone-length + aux(p_cnn vs GT, keeps the backbone honest) +
       small uv-reg (2D correction stays a correction). Real GT (DexYCB+HO3D),
       same subject/sequence holdout as train_zlifter -> directly comparable to the
       pure lifter's 37.43mm.

    python train_hybrid.py --exp hybrid --datasets dexycb,ho3d --epoch 40
"""
import os
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from models.mobrecon_ds import LargeModel_Extra_RGBD
from models.fastvit_backbone import FastViTHandSoftArgmax
from utils import load_cfg
from train_zlifter import (DexYCBLift, HO3DLift, IPhoneLift, bone_lengths, _BONES)


class HybridLifter(nn.Module):
    def __init__(self, cfg, n=21, h=512, mode="backbone", backbone="densestack",
                 fastvit_variant="fastvit_t8"):
        super().__init__()
        cfg.MODEL.set_new_allowed(True); cfg.MODEL.IN_CHANS = 4
        self.backbone_kind = backbone
        if backbone == "fastvit":                              # FastViT-4ch (RGB-D) encoder + soft-argmax
            self.backbone = FastViTHandSoftArgmax(variant=fastvit_variant, in_chans=4, pretrained=True)
        else:
            self.backbone = LargeModel_Extra_RGBD(cfg)         # revived MobRecon (pixels)
        self.n = n
        self.mode = mode                                       # 'backbone' | 'locked'
        # BACKBONE-PRIMARY: p_cnn is the main output; the head predicts a residual Δp
        # from [p_cnn(3) + p_geom(3) + mp_feat(4)] = 10/joint, so MediaPipe 2D + depth
        # REFINE the backbone (correct joints where p_cnn's reprojection/Z is off).
        self.head = nn.Sequential(
            nn.Linear(n * 10, h), nn.BatchNorm1d(h), nn.ReLU(True), nn.Dropout(0.1),
            nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(True),
            nn.Linear(h, n * 3))

    def warm_start(self, ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu")
        sd = ck.get("model_state_dict", ck)
        if self.backbone_kind == "fastvit":
            # FreiHAND FastViT ckpt is 3-ch RGB; inflate the stem conv to 4-ch (RGB copied,
            # depth channel zero-init) and load everything else that matches.
            tgt = self.backbone.state_dict(); new = {}
            n_inf = n_skip = 0
            for k, v in tgt.items():
                sv = sd.get(k)
                if sv is None: new[k] = v; n_skip += 1
                elif sv.shape == v.shape: new[k] = sv
                elif (sv.dim() == 4 and v.dim() == 4 and sv.shape[0] == v.shape[0]
                      and sv.shape[1] == 3 and v.shape[1] == 4 and sv.shape[2:] == v.shape[2:]):
                    w = v.clone(); w[:, :3] = sv; w[:, 3:] = 0; new[k] = w; n_inf += 1
                else: new[k] = v; n_skip += 1
            self.backbone.load_state_dict(new)
            print(f"[hybrid] FastViT backbone warm-started from {ckpt_path} "
                  f"(stem inflated 3->4: {n_inf}, skipped {n_skip})", flush=True)
        else:
            self.backbone.load_state_dict(sd)
            print(f"[hybrid] backbone warm-started from {ckpt_path}", flush=True)

    def forward(self, feat, crop, uv, z_use, K):
        bb = self.backbone(crop)
        p_cnn = bb["keypoints"]                                 # [B,21,3] backbone 3D (pixels)
        self.bb_uv = bb.get("uv")                               # soft-argmax 2D (fastvit only), for 2D aux
        fx, fy, cx, cy = K[:, 0:1], K[:, 1:2], K[:, 2:3], K[:, 3:4]
        Xg = (uv[..., 0] - cx) / fx * z_use
        Yg = (uv[..., 1] - cy) / fy * z_use
        p_geom = torch.stack([Xg, Yg, z_use], dim=-1)
        p_geom = p_geom - p_geom[:, 0:1]                        # MediaPipe+depth geometric prior
        if self.mode == "locked":
            # B: X,Y from MediaPipe uv (2D locked), Z from the backbone's relative depth.
            # Absolute Z = LiDAR wrist depth (Z0) + backbone per-joint relative depth.
            Z0 = z_use[:, 0:1]                                  # [B,1] LiDAR wrist absolute depth
            dz = p_cnn[..., 2] - p_cnn[:, 0:1, 2]              # [B,21] backbone rel depth (wrist=0)
            Zabs = Z0 + dz
            X = (uv[..., 0] - cx) / fx * Zabs
            Y = (uv[..., 1] - cy) / fy * Zabs
            p = torch.stack([X, Y, Zabs], dim=-1)
        else:                                                  # backbone-primary + residual
            x = torch.cat([p_cnn, p_geom, feat], dim=-1)       # [B,21,10]
            dp = self.head(x.reshape(x.shape[0], -1)).reshape(-1, self.n, 3)
            p = p_cnn + dp
        return p - p[:, 0:1], p_cnn, p_geom


def build(args, train):
    parts, tests = [], {}
    dsets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if "dexycb" in dsets:
        subs = sorted(set(np.load(args.cache, allow_pickle=True)["subj"].tolist()))
        sel = subs[:-args.n_test_subj] if train else subs[-args.n_test_subj:]
        parts.append(DexYCBLift(args.cache, args.root, sel, with_crop=True))
    if "ho3d" in dsets:
        seqs = sorted(set(np.load(args.ho3d_cache, allow_pickle=True)["subj"].tolist()))
        k = max(1, len(seqs) // 5)
        sel = seqs[:-k] if train else seqs[-k:]
        parts.append(HO3DLift(args.ho3d_cache, sel, with_crop=True))
    if "iphone" in dsets:
        dirs = sorted(set(np.load(args.iphone_cache, allow_pickle=True)["subj"].tolist()))
        sel = dirs[:-1] if train else dirs[-1:]                 # hold out the last capture dir
        parts.append(IPhoneLift(args.iphone_cache, sel))
    return parts


@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    s_hyb = s_cnn = s_geo = vv = 0.0
    for it in dl:
        feat = it["feat"].float().to(device); crop = it["crop"].float().to(device)
        uv = it["uv"].float().to(device); z_use = it["z_use"].float().to(device)
        K = it["K"].float().to(device); gt = it["gt3d"].float().to(device); gt = gt - gt[:, 0:1]
        v = it["joint_valid"].float().to(device)               # [B,21] (all-ones for real GT)
        p, p_cnn, p_geom = model(feat, crop, uv, z_use, K)
        s_hyb += float((torch.sqrt(((p - gt) ** 2).sum(-1)) * v).sum())
        s_cnn += float((torch.sqrt(((p_cnn - gt) ** 2).sum(-1)) * v).sum())
        s_geo += float((torch.sqrt(((p_geom - gt) ** 2).sum(-1)) * v).sum())
        vv += float(v.sum())
    return s_hyb / vv * 1000, s_cnn / vv * 1000, s_geo / vv * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="dexycb,ho3d")
    ap.add_argument("--cache", default="cache/mp_dexycb.npz")
    ap.add_argument("--ho3d_cache", default="cache/mp_ho3d.npz")
    ap.add_argument("--iphone_cache", default="cache/mp_iphone.npz")
    ap.add_argument("--root", default="/home/jucpark/DeepLearning/Datasets/Hand Dataset/DexYCB_full/data")
    ap.add_argument("--cfg", default="./configs_rgbd.yaml")
    ap.add_argument("--pretrain", default="mobrecon_ckpt/rgbd_real_0613/best.pt")
    ap.add_argument("--backbone", default="densestack", choices=["densestack", "fastvit"],
                    help="fastvit = FastViT-4ch soft-argmax (warm-start from FreiHAND ckpt via --pretrain)")
    ap.add_argument("--fastvit_variant", default="fastvit_t8")
    ap.add_argument("--exp", default="hybrid")
    ap.add_argument("--epoch", default=40, type=int)
    ap.add_argument("--batch", default=64, type=int)
    ap.add_argument("--lr_head", default=1e-3, type=float)
    ap.add_argument("--lr_bb", default=1e-4, type=float)
    ap.add_argument("--lambda_bone", default=1.0, type=float)
    ap.add_argument("--lambda_aux", default=1.0, type=float)
    ap.add_argument("--mode", default="backbone", choices=["backbone", "locked"],
                    help="'locked' (B) = X,Y from MediaPipe uv + Z from backbone; "
                         "'backbone' = p_cnn + residual head.")
    ap.add_argument("--lambda_reproj", default=0.0, type=float,
                    help="(A) weight on the 2D-reprojection-to-MediaPipe loss (px^2).")
    ap.add_argument("--lambda_uv", default=1e-4, type=float)
    ap.add_argument("--lambda_uv2d", default=0.0, type=float,
                    help="weight on the FastViT soft-argmax 2D loss vs detector uv (crop space)")
    ap.add_argument("--n_test_subj", default=2, type=int)
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.join("mobrecon_ckpt", args.exp); os.makedirs(ckpt_dir, exist_ok=True)

    tr = ConcatDataset(build(args, True))
    te_parts = build(args, False)
    names = [d.strip() for d in args.datasets.split(",") if d.strip()]
    te_dls = {nm: DataLoader(p, batch_size=args.batch, shuffle=False, num_workers=6)
              for nm, p in zip(names, te_parts)}
    tr_dl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=8, drop_last=True)

    cfg = load_cfg(args.cfg)
    model = HybridLifter(cfg, mode=args.mode, backbone=args.backbone,
                         fastvit_variant=args.fastvit_variant).to(device)
    print(f"[hybrid] backbone={args.backbone}  mode={args.mode}  lambda_reproj={args.lambda_reproj}", flush=True)
    if args.pretrain and os.path.isfile(args.pretrain):
        model.warm_start(args.pretrain)
    print(f"[hybrid] params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr_bb},
        {"params": model.head.parameters(), "lr": args.lr_head}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epoch)

    best = float("inf")
    for ep in tqdm(range(args.epoch)):
        model.train(); tot = 0.0; nb = 0
        for it in tr_dl:
            feat = it["feat"].float().to(device); crop = it["crop"].float().to(device)
            uv = it["uv"].float().to(device); z_use = it["z_use"].float().to(device)
            K = it["K"].float().to(device); gt = it["gt3d"].float().to(device); gt = gt - gt[:, 0:1]
            v = it["joint_valid"].float().to(device)                             # [B,21]
            bv = v[:, _BONES[:, 0].to(device)] * v[:, _BONES[:, 1].to(device)]    # [B,20] both ends valid
            opt.zero_grad()
            p, p_cnn, p_geom = model(feat, crop, uv, z_use, K)
            def m(e):  # masked per-joint mean
                return (e * v).sum() / (v.sum() + 1e-8)
            loss = (m(((p - gt) ** 2).sum(-1))
                    + args.lambda_bone * ((bone_lengths(p) - bone_lengths(gt)).abs() * bv).sum() / (bv.sum() + 1e-8)
                    + args.lambda_aux * m(((p_cnn - gt) ** 2).sum(-1))           # keep backbone honest
                    + args.lambda_uv * m(((p - p_cnn) ** 2).sum(-1)))            # refinement stays modest
            if args.lambda_reproj > 0:  # (A) pull the reprojection onto MediaPipe uv
                fx_, fy_, cx_, cy_ = K[:, 0:1], K[:, 1:2], K[:, 2:3], K[:, 3:4]
                Z0 = z_use[:, 0:1]
                wx = (uv[:, 0:1, 0] - cx_) / fx_ * Z0; wy = (uv[:, 0:1, 1] - cy_) / fy_ * Z0
                pa = p + torch.cat([wx, wy, Z0], -1)[:, None, :]                  # absolute (wrist-anchored)
                Zc = pa[..., 2].clamp(min=1e-3)
                up = pa[..., 0] / Zc * fx_ + cx_; vp = pa[..., 1] / Zc * fy_ + cy_
                loss = loss + args.lambda_reproj * m((up - uv[..., 0]) ** 2 + (vp - uv[..., 1]) ** 2)
            if args.lambda_uv2d > 0 and model.bb_uv is not None and "uv_crop" in it:
                # tie the FastViT backbone's soft-argmax 2D to the detector's uv (crop space) so it
                # localises cleanly WITHOUT depending on MediaPipe at inference.
                uv_tgt = it["uv_crop"].float().to(device)
                loss = loss + args.lambda_uv2d * m(((model.bb_uv - uv_tgt) ** 2).sum(-1))
            loss.backward(); opt.step()
            tot += float(torch.sqrt(((p.detach() - gt) ** 2).sum(-1)).mean()) * 1000; nb += 1
        sched.step()
        res = {nm: evaluate(model, dl, device) for nm, dl in te_dls.items()}
        tot_n = sum(len(te_parts[i]) for i in range(len(te_parts)))
        comb = sum(res[nm][0] * len(te_parts[i]) for i, nm in enumerate(names)) / tot_n
        msg = "  ".join(f"{nm}={res[nm][0]:.1f}(cnn{res[nm][1]:.0f}/geo{res[nm][2]:.0f})" for nm in res)
        print(f"[ep {ep+1:3d}/{args.epoch}] train={tot/max(nb,1):6.2f}mm  {msg}  "
              f"comb={comb:6.2f}mm  best={min(best,comb):6.2f}mm", flush=True)
        if comb < best:
            best = comb
            torch.save({"epoch": ep + 1, "model_state_dict": model.state_dict(),
                        "best_test_mpjpe": comb,
                        "per_dataset": {nm: res[nm][0] for nm in res}}, os.path.join(ckpt_dir, "best.pt"))
    print(f"[hybrid] BEST combined MPJPE = {best:.2f} mm  "
          f"(final: {', '.join(f'{nm}={res[nm][0]:.1f}[cnn{res[nm][1]:.0f}]' for nm in res)})", flush=True)


if __name__ == "__main__":
    main()
