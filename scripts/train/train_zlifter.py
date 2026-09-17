"""Z-only lifter with bone-length prior, trained + validated on DexYCB REAL 3D GT.

Idea (user's): trust MediaPipe 2D, solve only depth.
  * Input per joint: real MediaPipe (u,v) [cached] + LiDAR/sensor depth sampled at
    (u,v) + a valid flag.  Works in the ORIGINAL camera frame (no crop).
  * Head predicts a per-joint depth CORRECTION dZ over the sampled depth:
        Z = z_sampled + dZ
    (LiDAR sees the skin surface / has holes; the net learns the surface->joint
    offset and fills occluded/hole joints from the pose prior.)
  * 3D is reconstructed GEOMETRICALLY from MediaPipe 2D so the 2D shape is kept
    EXACTLY:  X = (u-cx)/fx * Z,  Y = (v-cy)/fy * Z,  Z.
  * Loss = 3D L2 (root-relative) + lambda * bone-length consistency (vs GT bones).

Because the GT is real MANO/mocap 3D (NOT MediaPipe-derived), this is a
circularity-free test.  Train/test split is by SUBJECT (no leakage).

    python scripts/train/train_zlifter.py --cache cache/mp_dexycb.npz --root "$DEXYCB"
"""

import _bootstrap  # noqa: F401  (repo-root + sibling script imports)

import os
import json
import argparse

import numpy as np
import cv2
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm

import pickle

from datasets.dexycb import JOINT_PERM
from datasets.ho3d import COORD_CHANGE, decode_ho3d_depth
from datasets.depth_synth import normalize_depth
from infer_rgbd_femtobolt import HAND_BONES, landmarks_to_bbox

_W, _H = 640, 480

# HO3D handJoints3D groups fingertips at the end -> reorder to MANO/MediaPipe order.
# (Recovered by MediaPipe-vs-GT NN matching: fixes 81.9px -> 16.9px. The stock
# ho3d.py loader uses identity, a latent order bug that is only self-consistent.)
HO3D_TO_MANO = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]


def build_hand_crop(rgb, dm, uv, size=256):
    """256x256 RGB+depth hand crop around the MediaPipe bbox, for the MobRecon
    backbone. rgb: HxWx3 RGB uint8; dm: HxW metric (hand-masked) depth; uv:[21,2]px.
    -> [4,size,size] float tensor (RGB in [0,1], depth median-centred like training)."""
    H, W = dm.shape
    x, y, s, _ = landmarks_to_bbox(uv, (H, W), margin_frac=0.5)
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(W, int(x + s)), min(H, int(y + s))
    if x1 <= x0 or y1 <= y0:
        return torch.zeros(4, size, size)
    rc = cv2.resize(rgb[y0:y1, x0:x1], (size, size)).astype(np.float32) / 255.0
    dc = cv2.resize(dm[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_NEAREST)
    dn = normalize_depth(dc, scale=0.1)[..., None]
    return torch.from_numpy(np.concatenate([rc, dn], axis=2)).permute(2, 0, 1).float()


def uv_to_crop_norm(uv, H, W):
    """Map full-image pixel uv [21,2] into the SAME 256-crop's normalised [-1,1] coords
    that build_hand_crop uses -> the soft-argmax 2D target for the FastViT backbone."""
    x, y, s, _ = landmarks_to_bbox(uv, (H, W), margin_frac=0.5)
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(W, int(x + s)), min(H, int(y + s))
    w = max(1, x1 - x0); h = max(1, y1 - y0)
    nx = (uv[:, 0] - x0) / w * 2 - 1
    ny = (uv[:, 1] - y0) / h * 2 - 1
    return np.stack([nx, ny], axis=1).astype(np.float32)


def _paths(color):
    return (color.replace("color_", "aligned_depth_to_color_").replace(".jpg", ".png"),
            color.replace("color_", "labels_").replace(".jpg", ".npz"))


class DexYCBLift(Dataset):
    """Full-frame Z-lifter samples from the MediaPipe cache + DexYCB raw GT/depth."""

    def __init__(self, cache, root, subjects, win=4, hand_mask=True, with_crop=False):
        self.root, self.win, self.hand_mask, self.with_crop = root, win, hand_mask, with_crop
        c = np.load(cache, allow_pickle=True)
        keep = (c["det"] == 1) & np.isin(c["subj"], list(subjects))
        self.paths = c["paths"][keep]
        self.uv = c["uv"][keep]                                  # [N,21,2] px
        self.subj = c["subj"][keep]
        self._intr = {}
        print(f"[DexYCBLift] {len(self.paths)} detected frames  subjects={sorted(set(subjects))}",
              flush=True)

    def __len__(self):
        return len(self.paths)

    def _K(self, serial):
        if serial not in self._intr:
            cc = yaml.safe_load(open(os.path.join(
                self.root, "calibration", "intrinsics", f"{serial}_{_W}x{_H}.yml")))["color"]
            self._intr[serial] = (cc["fx"], cc["fy"], cc["ppx"], cc["ppy"])
        return self._intr[serial]

    def _sample_depth(self, dm, u, v):
        ui, vi, w = int(round(u)), int(round(v)), self.win
        if not (0 <= ui < dm.shape[1] and 0 <= vi < dm.shape[0]):
            return 0.0, 0.0
        p = dm[max(0, vi - w):vi + w + 1, max(0, ui - w):ui + w + 1]
        nz = p[p > 0]
        return (float(np.median(nz)), 1.0) if nz.size else (0.0, 0.0)

    def __getitem__(self, i):
        cp = str(self.paths[i])
        uv = self.uv[i].astype(np.float32)                       # [21,2] px
        dpath, lpath = _paths(cp)
        lab = np.load(lpath)
        j3 = lab["joint_3d"][0].astype(np.float32)
        med = np.median(np.abs(j3[:, 2][j3[:, 2] != 0])) if np.any(j3[:, 2] != 0) else 0.0
        j3 = (j3 / 1000.0 if med > 10 else j3)[JOINT_PERM]        # [21,3] cam metres
        serial = os.path.basename(os.path.dirname(cp))
        fx, fy, cx, cy = self._K(serial)

        d = cv2.imread(dpath, cv2.IMREAD_UNCHANGED)
        dm = (d.astype(np.float32) / 1000.0) if d is not None else np.zeros((_H, _W), np.float32)
        if self.hand_mask:
            # Keep ONLY hand-segmented depth (seg==255) so a joint's window can't
            # sample the held object / table / silhouette-edge background -- the
            # cause of the 132mm sampled-vs-GT depth error under naive sampling.
            dm = np.where(lab["seg"] == 255, dm, 0.0).astype(np.float32)
        zs = np.zeros(21, np.float32); zv = np.zeros(21, np.float32)
        for j in range(21):
            zs[j], zv[j] = self._sample_depth(dm, uv[j, 0], uv[j, 1])
        z_ref = float(np.median(zs[zv > 0])) if zv.any() else 0.5
        z_use = np.where(zv > 0, zs, z_ref).astype(np.float32)   # base depth for recon

        feat = np.stack([uv[:, 0] / _W * 2 - 1, uv[:, 1] / _H * 2 - 1,
                         z_use - z_ref, zv], axis=1).astype(np.float32)   # [21,4]
        out = {"feat": feat, "uv": uv, "z_use": z_use, "joint_valid": np.ones(21, np.float32),
               "K": np.array([fx, fy, cx, cy], np.float32), "gt3d": j3}
        if self.with_crop:
            rgb = cv2.cvtColor(cv2.imread(cp), cv2.COLOR_BGR2RGB)
            out["crop"] = build_hand_crop(rgb, dm, uv)
            out["uv_crop"] = uv_to_crop_norm(uv, dm.shape[0], dm.shape[1])
        return out


class HO3DLift(Dataset):
    """HO3D Z-lifter samples: pickle meta (handJoints3D OpenGL + camMat), 2-channel
    depth, GT reordered to MANO, z-band hand masking (HO3D has no seg)."""

    def __init__(self, cache, subjects, win=4, band=0.06, hand_mask=True, with_crop=False):
        self.win, self.band, self.hand_mask, self.with_crop = win, band, hand_mask, with_crop
        c = np.load(cache, allow_pickle=True)
        keep = (c["det"] == 1) & np.isin(c["subj"], list(subjects))
        self.paths = c["paths"][keep]; self.uv = c["uv"][keep]
        print(f"[HO3DLift] {len(self.paths)} detected frames  seqs={sorted(set(subjects))}", flush=True)

    def __len__(self):
        return len(self.paths)

    def _sample(self, dm, u, v):
        ui, vi, w = int(round(u)), int(round(v)), self.win
        if not (0 <= ui < dm.shape[1] and 0 <= vi < dm.shape[0]):
            return 0.0, 0.0
        p = dm[max(0, vi - w):vi + w + 1, max(0, ui - w):ui + w + 1]; nz = p[p > 0]
        return (float(np.median(nz)), 1.0) if nz.size else (0.0, 0.0)

    def __getitem__(self, i):
        cp = str(self.paths[i]); uv = self.uv[i].astype(np.float32)
        m = pickle.load(open(cp.replace("/rgb/", "/meta/").rsplit(".", 1)[0] + ".pkl", "rb"),
                        encoding="latin1")
        hj = np.asarray(m["handJoints3D"], np.float32)
        if hj.shape != (21, 3) or not np.isfinite(hj).all():           # eval-split / missing GT
            return self.__getitem__((i + 1) % len(self))
        j3 = (hj @ COORD_CHANGE)[HO3D_TO_MANO].astype(np.float32)       # (21,3) MANO cam m
        K = np.asarray(m["camMat"], np.float32)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        dm = decode_ho3d_depth(cp.replace("/rgb/", "/depth/").rsplit(".", 1)[0] + ".png")
        if dm is None:
            dm = np.zeros((_H, _W), np.float32)
        if self.hand_mask:                                              # z-band mask (no seg)
            z = j3[:, 2]; lo, hi = z.min() - self.band, z.max() + self.band
            dm = np.where((dm >= lo) & (dm <= hi), dm, 0.0).astype(np.float32)
        zs = np.zeros(21, np.float32); zv = np.zeros(21, np.float32)
        for jj in range(21):
            zs[jj], zv[jj] = self._sample(dm, uv[jj, 0], uv[jj, 1])
        z_ref = float(np.median(zs[zv > 0])) if zv.any() else 0.5
        z_use = np.where(zv > 0, zs, z_ref).astype(np.float32)
        feat = np.stack([uv[:, 0] / _W * 2 - 1, uv[:, 1] / _H * 2 - 1,
                         z_use - z_ref, zv], axis=1).astype(np.float32)
        out = {"feat": feat, "uv": uv, "z_use": z_use, "joint_valid": np.ones(21, np.float32),
               "K": np.array([fx, fy, cx, cy], np.float32), "gt3d": j3}
        if self.with_crop:
            rgb = cv2.cvtColor(cv2.imread(cp), cv2.COLOR_BGR2RGB)
            out["crop"] = build_hand_crop(rgb, dm, uv)
            out["uv_crop"] = uv_to_crop_norm(uv, dm.shape[0], dm.shape[1])
        return out


def _iphone_stem(rgb_path):
    """Capture stem from a cached rgb path; session_* dumps use _rgb.jpg, not .png."""
    import re
    return re.sub(r"_rgb\.(png|jpe?g)$", "", rgb_path)


class IPhoneLift(Dataset):
    """iPhone captures for DOMAIN ADAPTATION of the hybrid backbone. Real LiDAR depth
    + pseudo-3D GT (make_iphone_gt: MediaPipe-2D + LiDAR lift, masked by joint_valid).
    Full-frame (rot k=0, verified: pseudo-GT projects to within ~8px of MediaPipe uv).
    Split by capture dir. Depth z-band-masked around the GT hand (no seg)."""

    def __init__(self, cache, subjects, win=4, band=0.10, with_crop=True):
        self.win, self.band, self.with_crop = win, band, with_crop
        c = np.load(cache, allow_pickle=True)
        keep = (c["det"] == 1) & np.isin(c["subj"], list(subjects))
        paths, uv = c["paths"][keep], c["uv"][keep]
        ex = np.array([os.path.isfile(_iphone_stem(str(p)) + "_depth.f32") for p in paths])
        self.paths, self.uv = paths[ex], uv[ex]                        # drop frames without depth
        print(f"[IPhoneLift] {len(self.paths)} frames  dirs={sorted(set(subjects))}", flush=True)

    def __len__(self):
        return len(self.paths)

    def _load(self, stem):
        m = json.load(open(stem + ".json"))
        from datasets.iphone_captures import _rgb_path
        bgr = cv2.imread(_rgb_path(stem)); H, W = bgr.shape[:2]
        K = np.array([[m["fx"], 0, m["cx"]], [0, m["fy"], m["cy"]], [0, 0, 1]], np.float32)
        if m.get("intrWidth") and m["intrWidth"] != W:
            s = W / float(m["intrWidth"]); K[0] *= s; K[1] *= s
        d = np.fromfile(os.path.join(os.path.dirname(stem), m["depthFile"]), dtype="<f4")
        depth = cv2.resize(d.reshape(m["depthHeight"], m["depthWidth"]), (W, H),
                           interpolation=cv2.INTER_NEAREST)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), depth, K

    def _sample(self, dm, u, v):
        ui, vi, w = int(round(u)), int(round(v)), self.win
        if not (0 <= ui < dm.shape[1] and 0 <= vi < dm.shape[0]):
            return 0.0, 0.0
        p = dm[max(0, vi - w):vi + w + 1, max(0, ui - w):ui + w + 1]; nz = p[p > 0]
        return (float(np.median(nz)), 1.0) if nz.size else (0.0, 0.0)

    def __getitem__(self, i):
        stem = _iphone_stem(str(self.paths[i]))
        uv = self.uv[i].astype(np.float32)
        g = json.load(open(stem + "_gt.json"))
        j3 = np.array(g["joints3d_cam"], np.float32)
        valid = np.array(g["joint_valid"], np.float32)
        rgb, depth, K = self._load(stem)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        zv_gt = j3[valid > 0, 2]
        if zv_gt.size:                                                  # z-band hand mask
            lo, hi = zv_gt.min() - self.band, zv_gt.max() + self.band
            depth = np.where((depth >= lo) & (depth <= hi), depth, 0.0).astype(np.float32)
        zs = np.zeros(21, np.float32); zvd = np.zeros(21, np.float32)
        for jj in range(21):
            zs[jj], zvd[jj] = self._sample(depth, uv[jj, 0], uv[jj, 1])
        z_ref = float(np.median(zs[zvd > 0])) if zvd.any() else 0.5
        z_use = np.where(zvd > 0, zs, z_ref).astype(np.float32)
        feat = np.stack([uv[:, 0] / rgb.shape[1] * 2 - 1, uv[:, 1] / rgb.shape[0] * 2 - 1,
                         z_use - z_ref, zvd], axis=1).astype(np.float32)
        out = {"feat": feat, "uv": uv, "z_use": z_use, "joint_valid": valid.astype(np.float32),
               "K": np.array([fx, fy, cx, cy], np.float32), "gt3d": j3.astype(np.float32)}
        if self.with_crop:
            out["crop"] = build_hand_crop(rgb, depth, uv)
            out["uv_crop"] = uv_to_crop_norm(uv, depth.shape[0], depth.shape[1])
        return out


def reconstruct(dZ, uv, z_use, K):
    """dZ[B,21], uv[B,21,2], z_use[B,21], K[B,4] -> 3D[B,21,3] cam, root-relative."""
    fx, fy, cx, cy = K[:, 0:1], K[:, 1:2], K[:, 2:3], K[:, 3:4]
    Z = z_use + dZ
    X = (uv[..., 0] - cx) / fx * Z
    Y = (uv[..., 1] - cy) / fy * Z
    p = torch.stack([X, Y, Z], dim=-1)                           # [B,21,3]
    return p - p[:, 0:1]                                         # root(wrist)-relative


class ZLifter(nn.Module):
    def __init__(self, n=21, fin=4, h=512):
        super().__init__()
        self.n = n
        self.net = nn.Sequential(
            nn.Linear(n * fin, h), nn.BatchNorm1d(h), nn.ReLU(True), nn.Dropout(0.1),
            nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(True), nn.Dropout(0.1),
            nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(True),
            nn.Linear(h, n))
    def forward(self, feat):                                     # [B,21,4] -> [B,21] dZ
        return self.net(feat.reshape(feat.shape[0], -1))


_BONES = torch.tensor(HAND_BONES, dtype=torch.long)              # [20,2]


def bone_lengths(p):                                             # p[B,21,3]
    a, b = _BONES[:, 0].to(p.device), _BONES[:, 1].to(p.device)
    return torch.linalg.norm(p[:, a] - p[:, b], dim=-1)          # [B,20]


def mpjpe_mm(pred, gt):
    return torch.sqrt(((pred - gt) ** 2).sum(-1)).mean() * 1000.0


@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    ds_m, ds_geo, n = 0.0, 0.0, 0
    for it in dl:
        feat = it["feat"].float().to(device); uv = it["uv"].float().to(device)
        z_use = it["z_use"].float().to(device); K = it["K"].float().to(device)
        gt = (it["gt3d"].float().to(device)); gt = gt - gt[:, 0:1]
        dZ = model(feat)
        pred = reconstruct(dZ, uv, z_use, K)
        geo = reconstruct(torch.zeros_like(dZ), uv, z_use, K)    # no-learning baseline
        B = feat.shape[0]
        ds_m += float(torch.sqrt(((pred - gt) ** 2).sum(-1)).sum());
        ds_geo += float(torch.sqrt(((geo - gt) ** 2).sum(-1)).sum())
        n += B * 21
    return ds_m / n * 1000.0, ds_geo / n * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="dexycb", help="comma list: dexycb,ho3d")
    ap.add_argument("--cache", default="cache/mp_dexycb.npz")
    ap.add_argument("--ho3d_cache", default="cache/mp_ho3d.npz")
    ap.add_argument("--root", default="/home/jucpark/DeepLearning/Datasets/Hand Dataset/DexYCB_full/data")
    ap.add_argument("--exp", default="zlifter")
    ap.add_argument("--epoch", default=60, type=int)
    ap.add_argument("--batch", default=256, type=int)
    ap.add_argument("--lr", default=1e-3, type=float)
    ap.add_argument("--lambda_bone", default=1.0, type=float)
    ap.add_argument("--n_test_subj", default=2, type=int, help="hold out last N subjects")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.join("mobrecon_ckpt", args.exp); os.makedirs(ckpt_dir, exist_ok=True)

    dsets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    trains, tests = [], {}
    if "dexycb" in dsets:
        subs = sorted(set(np.load(args.cache, allow_pickle=True)["subj"].tolist()))
        te_s, tr_s = subs[-args.n_test_subj:], subs[:-args.n_test_subj]
        print(f"[split] dexycb: {len(tr_s)} train subj / test subj={te_s}", flush=True)
        trains.append(DexYCBLift(args.cache, args.root, tr_s))
        tests["dexycb"] = DexYCBLift(args.cache, args.root, te_s)
    if "ho3d" in dsets:
        seqs = sorted(set(np.load(args.ho3d_cache, allow_pickle=True)["subj"].tolist()))
        k = max(1, len(seqs) // 5); te_q, tr_q = seqs[-k:], seqs[:-k]
        print(f"[split] ho3d: {len(tr_q)} train seqs / test seqs={te_q}", flush=True)
        trains.append(HO3DLift(args.ho3d_cache, tr_q))
        tests["ho3d"] = HO3DLift(args.ho3d_cache, te_q)
    train_ds = trains[0] if len(trains) == 1 else ConcatDataset(trains)
    tr_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=8, drop_last=True)
    te_dls = {k: DataLoader(v, batch_size=args.batch, shuffle=False, num_workers=4)
              for k, v in tests.items()}

    model = ZLifter().to(device)
    print(f"[zlifter] params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epoch)

    best = float("inf"); geo0 = None
    for ep in tqdm(range(args.epoch)):
        model.train(); tot = 0.0; nb = 0
        for it in tr_dl:
            feat = it["feat"].float().to(device); uv = it["uv"].float().to(device)
            z_use = it["z_use"].float().to(device); K = it["K"].float().to(device)
            gt = it["gt3d"].float().to(device); gt = gt - gt[:, 0:1]
            opt.zero_grad()
            pred = reconstruct(model(feat), uv, z_use, K)
            loss = ((pred - gt) ** 2).sum(-1).mean() + args.lambda_bone * \
                (bone_lengths(pred) - bone_lengths(gt)).abs().mean()
            loss.backward(); opt.step()
            tot += float(mpjpe_mm(pred.detach(), gt)); nb += 1
        sched.step()
        res = {k: evaluate(model, dl, device) for k, dl in te_dls.items()}
        tot_n = sum(len(tests[k]) for k in tests)
        comb = sum(res[k][0] * len(tests[k]) for k in tests) / tot_n     # frame-weighted
        msg = "  ".join(f"{k}={res[k][0]:.1f}(geo{res[k][1]:.0f})" for k in res)
        print(f"[ep {ep+1:3d}/{args.epoch}] train={tot/max(nb,1):6.2f}mm  {msg}  "
              f"comb={comb:6.2f}mm  best={min(best,comb):6.2f}mm", flush=True)
        if comb < best:
            best = comb
            torch.save({"epoch": ep + 1, "model_state_dict": model.state_dict(),
                        "test_mpjpe": comb, "best_test_mpjpe": best,
                        "per_dataset": {k: res[k][0] for k in res}}, os.path.join(ckpt_dir, "best.pt"))
    print(f"[zlifter] BEST combined MPJPE = {best:.2f} mm  "
          f"(final per-dataset: {', '.join(f'{k}={res[k][0]:.1f}' for k in res)})", flush=True)


if __name__ == "__main__":
    main()
