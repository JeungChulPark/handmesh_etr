"""Train the FastViT hand backbone on FreiHAND (+ optional DexYCB/HO3D mix),
eval with MPJPE.

Directly comparable to the RGB base backbone `pretrain/100.pt` (FreiHAND eval
MPJPE 31.10 mm, root-relative). Same dataset, same target (align_joint), same
metric.

Stage-1 "joint pretrain": FreiHAND alone is green-screen only, so all FastViT
variants share the same real-world robustness ceiling on iPhone captures. Mixing
DexYCB + HO3D (real backgrounds, occlusion) puts that knowledge INTO the weights,
recreating the precondition under which target-dominant replay fine-tuning
(Stage 2) was the winning recipe for the densestack model. Holdout matches
train_zlifter: DexYCB = last 2 subjects, HO3D = last 1/5 sequences, so later
hybrid evals on those splits stay leakage-free. best.pt is selected on the MEAN
of the per-dataset eval MPJPEs.

    python train_fastvit_hand.py --variant fastvit_t8 --epoch 30
    python train_fastvit_hand.py --variant fastvit_sa12 --full 1 \
        --datasets frei,dexycb,ho3d --exp fastvit_sa12_joint
"""
import os
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset
from tqdm import tqdm

from models.fastvit_backbone import FastViTHandBackbone, FastViTHandSoftArgmax
from datasets.freihand_ty import Freihand
from datasets.dexycb import DexYCB_RGBD
from datasets.ho3d import HO3D_RGBD
from datasets.iphone_captures import IPhoneCaptures_RGBD


class HandCropDS(torch.utils.data.Dataset):
    """Unify item contracts across sources: Freihand yields img/align_joint,
    DexYCB_RGBD/HO3D_RGBD/IPhoneCaptures_RGBD yield image/keypoints3D. Emits
    {'img': [3,256,256] float tensor, 'align_joint': (21,3) float32,
     'joint_valid': (21,) float32} so a mixed ConcatDataset batches cleanly
    (float64 frei joints would break collate). joint_valid is all-ones except for
    the iPhone pseudo-GT, whose occluded/outlier joints are masked from the loss."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        it = self.ds[i]
        img = it["img"] if "img" in it else it["image"][:3]
        aj = it["align_joint"] if "align_joint" in it else it["keypoints3D"]
        valid = it.get("joint_valid", np.ones(21, np.float32))
        return {"img": img, "align_joint": np.asarray(aj, np.float32),
                "joint_valid": np.asarray(valid, np.float32)}


def rand_subset(ds, n, seed=0):
    """Seeded random subset (NOT a head-truncation, which would bias to the
    first subjects/sequences)."""
    if n <= 0 or n >= len(ds):
        return ds
    g = torch.Generator().manual_seed(seed)
    return Subset(ds, torch.randperm(len(ds), generator=g)[:n].tolist())

_BONES = torch.tensor([
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)])


def bone_len(p):
    return (p[:, _BONES[:, 0]] - p[:, _BONES[:, 1]]).norm(dim=-1)


def masked_mpjpe(p, gt, v):
    """Mean joint error over VALID joints only ((B,21,3),(B,21,3),(B,21))."""
    d = torch.sqrt(((p - gt) ** 2).sum(-1))
    return (d * v).sum() / v.sum().clamp(min=1.0)


def masked_bone_l1(p, gt, v):
    """Bone-length L1 over bones whose BOTH endpoints are valid."""
    bv = v[:, _BONES[:, 0]] * v[:, _BONES[:, 1]]
    return ((bone_len(p) - bone_len(gt)).abs() * bv).sum() / bv.sum().clamp(min=1.0)


def photometric_aug(img):
    """Cheap per-sample brightness/contrast/noise jitter on RGB [B,3,H,W] in [0,1].
    Geometric aug is already done by the FreiHAND loader; this adds appearance
    robustness (no hflip — that would flip hand chirality)."""
    B = img.shape[0]; dev = img.device
    img = img * (torch.rand(B, 1, 1, 1, device=dev) * 0.4 + 0.8)          # brightness 0.8-1.2
    m = img.mean(dim=(2, 3), keepdim=True)
    img = (img - m) * (torch.rand(B, 1, 1, 1, device=dev) * 0.4 + 0.8) + m  # contrast
    img = img + torch.randn_like(img) * 0.02                              # mild noise
    return img.clamp(0, 1)


@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    tot, n = 0.0, 0.0
    for it in dl:
        img = it["img"].float().to(device)
        gt = it["align_joint"].float().to(device)
        v = it["joint_valid"].float().to(device)
        p = model(img)["keypoints"]
        d = torch.sqrt(((p - gt) ** 2).sum(-1))
        tot += float((d * v).sum()) * 1000.0
        n += float(v.sum())
    return tot / max(n, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="fastvit_t8")
    ap.add_argument("--head", default="softargmax", choices=["mlp", "softargmax"])
    ap.add_argument("--in_chans", default=3, type=int)
    ap.add_argument("--epoch", default=30, type=int)
    ap.add_argument("--batch", default=64, type=int)
    ap.add_argument("--lr_enc", default=1e-4, type=float)
    ap.add_argument("--lr_head", default=1e-3, type=float)
    ap.add_argument("--lambda_bone", default=0.5, type=float)
    ap.add_argument("--aug", default=1, type=int, help="1 = photometric augmentation on")
    ap.add_argument("--full", default=0, type=int, help="1 = use all 130240 FreiHAND images (4x data)")
    ap.add_argument("--datasets", default="frei", help="comma list: frei,dexycb,ho3d")
    ap.add_argument("--dexycb_root",
                    default="/home/jucpark/DeepLearning/Datasets/Hand Dataset/DexYCB_full/data")
    ap.add_argument("--ho3d_root",
                    default="/home/jucpark/DeepLearning/Datasets/Hand Dataset/HO3D_full")
    ap.add_argument("--dex_limit", default=80000, type=int, help="random cap on DexYCB train frames")
    ap.add_argument("--ho3d_limit", default=40000, type=int, help="random cap on HO3D train frames")
    ap.add_argument("--frei_limit", default=0, type=int,
                    help="random cap on FreiHAND train frames (0 = all); the replay slice")
    ap.add_argument("--eval_limit", default=4000, type=int, help="random cap per heldout eval set")
    # ---- Stage-2 replay fine-tune on the iPhone pseudo-GT ----
    ap.add_argument("--iphone_root", default="",
                    help="comma list of capture dirs to TRAIN on (e.g. rgbd_captures,...,_03)")
    ap.add_argument("--iphone_eval_root", default="",
                    help="whole capture session held out for eval (e.g. rgbd_captures_04) -- "
                         "session-level split, no temporally-adjacent-frame leakage")
    ap.add_argument("--iphone_repeat", default=8, type=int,
                    help="oversample factor so the tiny iPhone set stays target-dominant")
    ap.add_argument("--pretrain", default="", help="warm-start checkpoint (best.pt)")
    ap.add_argument("--select", default="comb", choices=["comb", "iphone"],
                    help="best.pt selection metric: mean of all eval sets, or iPhone only")
    ap.add_argument("--exp", default="fastvit_t8_frei")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.join("mobrecon_ckpt", args.exp)
    os.makedirs(ckpt_dir, exist_ok=True)

    dsets = [s.strip() for s in args.datasets.split(",") if s.strip()]
    trains, tests = [], {}
    if "iphone" in dsets:
        assert args.iphone_root and args.iphone_eval_root, \
            "--iphone_root and --iphone_eval_root required for the iphone dataset"
        iph = HandCropDS(IPhoneCaptures_RGBD(root=args.iphone_root, mode="train",
                                             with_depth=False, split="all"))
        trains.extend([iph] * max(1, args.iphone_repeat))       # target-dominant replay
        tests["iphone"] = HandCropDS(IPhoneCaptures_RGBD(
            root=args.iphone_eval_root, mode="test", with_depth=False, split="all"))
    if "frei" in dsets:
        trains.append(HandCropDS(rand_subset(
            Freihand(mode="train", full=bool(args.full)), args.frei_limit)))
        tests["frei"] = HandCropDS(Freihand(mode="eval"))
    if "dexycb" in dsets:
        # same holdout as train_zlifter: last 2 subjects (sorted) never trained on
        subs = sorted(d for d in os.listdir(args.dexycb_root) if "-subject-" in d)
        te_s, tr_s = subs[-2:], subs[:-2]
        print(f"[split] dexycb: {len(tr_s)} train subj / test subj={te_s}", flush=True)
        trains.append(HandCropDS(rand_subset(
            DexYCB_RGBD(root=args.dexycb_root, mode="train", with_depth=False,
                        subjects=tr_s), args.dex_limit)))
        tests["dexycb"] = HandCropDS(rand_subset(
            DexYCB_RGBD(root=args.dexycb_root, mode="test", with_depth=False,
                        subjects=te_s), args.eval_limit, seed=1))
    if "ho3d" in dsets:
        # same holdout as train_zlifter: last 1/5 sequences (sorted)
        seqs = sorted(d for d in os.listdir(os.path.join(args.ho3d_root, "train"))
                      if os.path.isdir(os.path.join(args.ho3d_root, "train", d)))
        k = max(1, len(seqs) // 5)
        te_q, tr_q = seqs[-k:], seqs[:-k]
        print(f"[split] ho3d: {len(tr_q)} train seqs / test seqs={te_q}", flush=True)
        trains.append(HandCropDS(rand_subset(
            HO3D_RGBD(root=args.ho3d_root, mode="train", with_depth=False,
                      sequences=tr_q), args.ho3d_limit)))
        tests["ho3d"] = HandCropDS(rand_subset(
            HO3D_RGBD(root=args.ho3d_root, mode="test", with_depth=False,
                      sequences=te_q), args.eval_limit, seed=1))

    tr = trains[0] if len(trains) == 1 else ConcatDataset(trains)
    print(f"[data] train={len(tr)} frames  eval=" +
          " ".join(f"{k}:{len(v)}" for k, v in tests.items()), flush=True)
    tr_dl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=8, drop_last=True)
    te_dls = {k: DataLoader(v, batch_size=args.batch, shuffle=False, num_workers=4)
              for k, v in tests.items()}

    Net = FastViTHandSoftArgmax if args.head == "softargmax" else FastViTHandBackbone
    model = Net(variant=args.variant, in_chans=args.in_chans, pretrained=True).to(device)
    if args.pretrain:
        state = torch.load(args.pretrain, map_location=device)
        model.load_state_dict(state.get("model_state_dict", state))
        print(f"[fastvit] warm-start from {args.pretrain} "
              f"(ep{state.get('epoch','?')}, eval={state.get('eval_mpjpe',float('nan')):.2f}mm)",
              flush=True)
    nparam = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[fastvit] {args.variant} head={args.head} in_chans={args.in_chans}  params={nparam:.2f}M", flush=True)

    bones = _BONES.to(device)
    globals()["_BONES"] = bones
    enc_ids = {id(p) for p in model.encoder.parameters()}
    head_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": args.lr_enc},
        {"params": head_params, "lr": args.lr_head}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epoch)

    best = float("inf")
    for ep in range(args.epoch):
        model.train(); tot = 0.0; nb = 0
        for it in tqdm(tr_dl, desc=f"ep{ep+1}"):
            img = it["img"].float().to(device)
            gt = it["align_joint"].float().to(device)
            v = it["joint_valid"].float().to(device)
            if args.aug:
                img = photometric_aug(img)
            opt.zero_grad()
            p = model(img)["keypoints"]
            loss = masked_mpjpe(p, gt, v)                                          # MPJPE loss
            loss = loss + args.lambda_bone * masked_bone_l1(p, gt, v)
            loss.backward(); opt.step()
            tot += float(loss); nb += 1
        sched.step()
        per = {k: evaluate(model, dl, device) for k, dl in te_dls.items()}
        # selection metric: mean over sets, or the iPhone session holdout alone
        mpjpe = per["iphone"] if args.select == "iphone" else float(np.mean(list(per.values())))
        per_str = "  ".join(f"{k}={v:5.2f}" for k, v in per.items())
        print(f"[ep {ep+1:3d}/{args.epoch}] train_loss={tot/max(nb,1)*1000:6.2f}  "
              f"{per_str}  sel({args.select})={mpjpe:6.2f}mm  best={min(best,mpjpe):6.2f}mm  "
              f"(frei baseline 100.pt = 31.10mm)", flush=True)
        if mpjpe < best:
            best = mpjpe
            torch.save({"epoch": ep + 1, "model_state_dict": model.state_dict(),
                        "eval_mpjpe": mpjpe, "eval_per_set": per, "variant": args.variant},
                       os.path.join(ckpt_dir, "best.pt"))
    print(f"[fastvit] BEST combined MPJPE = {best:.2f}mm  "
          f"(datasets={args.datasets}, frei 100.pt baseline = 31.10mm)", flush=True)


if __name__ == "__main__":
    main()
