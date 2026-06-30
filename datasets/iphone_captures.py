"""iPhone RGB-D capture loader (pseudo-3D GT from make_iphone_gt.py).

Emits the standard RGB-D training contract (identical to DexYCB_RGBD) so it
concatenates cleanly in train_mobrecon_rgbd.py:

    image       FloatTensor [4,256,256]  RGB[0:3] in [0,1] + masked depth[3]
    keypoints3D [21,3] root(wrist)-aligned camera-space joints (m), crop-rotated
    keypoints2D [21,2] crop-pixel / 256
    root        [3]   camera-space wrist (m)
    cam         [3,3] intrinsics adjusted to the 256 crop
    joint_valid [21]  1 = supervised (occluded/outlier joints from the lift -> 0)

The 3D GT is the MediaPipe-2D + LiDAR-lift pseudo-GT produced by make_iphone_gt.py
(per-frame `cap_*_gt.json`). Run that FIRST. Joints flagged invalid there are
masked out of the loss via `joint_valid` (the trainer already honours it), so the
GT is sparse-but-trustworthy.

Deployment captures have no segmentation, so the depth channel is hand-masked here
with a z-band around the GT hand-surface depth inside a padded bbox -- reproducing
the `depth_mode="hand_masked"` distribution the other datasets train on, and
avoiding the background contamination the dual-stream ablation exposed.

    python -m datasets.iphone_captures --root rgbd_captures --check 8
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset

from datasets.augmentation import color_jitter
from datasets.dataset_utils import augmentation
from datasets.depth_synth import normalize_depth, corrupt_depth
from datasets.dexycb import get_2D_annotation

# MediaPipe landmark order == FreiHAND/MANO order (wrist, thumb..pinky), so no
# permutation is needed (see HAND_BONES note in infer_rgbd_femtobolt.py).

_ROT_CODES = {1: cv2.ROTATE_90_CLOCKWISE, 2: cv2.ROTATE_180, 3: cv2.ROTATE_90_COUNTERCLOCKWISE}


def _rotate(img, k):
    return img if k % 4 == 0 else cv2.rotate(img, _ROT_CODES[k % 4])


def hand_band_mask(depth_m, z0, bbox, band=0.12, pad=0.15):
    """Keep depth within `band` m of the hand-surface z0 inside a padded bbox; zero
    the rest. Reproduces hand_masked depth at deploy using only the GT hand depth."""
    H, W = depth_m.shape
    x, y, s, _ = bbox
    px = int(s * pad)
    x0, y0 = max(int(x) - px, 0), max(int(y) - px, 0)
    x1, y1 = min(int(x + s) + px, W), min(int(y + s) + px, H)
    out = np.zeros_like(depth_m)
    reg = depth_m[y0:y1, x0:x1]
    keep = (reg > 0) & (np.abs(reg - z0) < band)
    out[y0:y1, x0:x1] = np.where(keep, reg, 0.0)
    return out


class IPhoneCaptures_RGBD(Dataset):
    def __init__(self, root, mode="train", with_depth=True, depth_cfg=None,
                 band=0.12, min_valid=6, limit=0, return_abs_depth=False):
        """
        Args:
            root: folder of cap_*_{rgb.png,depth.f32,json} + cap_*_gt.json.
            mode: "train" enables RGB jitter + crop aug; else deterministic.
            band: z-band half-width (m) for the hand depth mask.
            min_valid: drop frames with fewer valid GT joints than this.
            limit: cap index length (0 = all).
        """
        self.root = root
        self.mode = mode
        self.with_depth = with_depth
        self.band = band
        self.min_valid = min_valid
        self.return_abs_depth = return_abs_depth
        self.depth_cfg = {**dict(sigma=0.002, dropout_p=0.0, quant=0.001, n_holes=0,
                                 hole_frac=0.15, norm_scale=0.1), **(depth_cfg or {})}
        self.samples = self._build_index(limit)
        print(f"[IPhoneCaptures_RGBD] {len(self.samples)} frames  mode={mode} "
              f"root={root} band={band}")

    def _build_index(self, limit):
        # root may be a single dir or a comma-separated list of capture folders.
        roots = [r.strip() for r in str(self.root).split(",") if r.strip()]
        stems = []
        for root in roots:
            for gt in sorted(glob.glob(os.path.join(root, "cap_*_gt.json"))):
                stem = gt[:-len("_gt.json")]
                if not os.path.isfile(stem + "_rgb.png"):
                    continue
                try:
                    d = json.load(open(gt))
                    if int(np.sum(d["joint_valid"])) >= self.min_valid:
                        stems.append(stem)
                except Exception:
                    continue
                if limit and len(stems) >= limit:
                    return stems
        return stems

    def __len__(self):
        return len(self.samples)

    def _load_depth_full(self, stem, H, W, k):
        """Load LiDAR depth (m) at RGB res, rotated to the GT frame."""
        cap = stem + ".json"
        if not os.path.isfile(cap):
            return np.zeros((H, W), np.float32)
        m = json.load(open(cap))
        df = m.get("depthFile")
        if not df:
            return np.zeros((H, W), np.float32)
        dp = os.path.join(os.path.dirname(stem), df)
        if not os.path.isfile(dp):
            return np.zeros((H, W), np.float32)
        d = np.fromfile(dp, dtype="<f4").reshape(m["depthHeight"], m["depthWidth"])
        d = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)
        return _rotate(d, k).astype(np.float32)

    def __getitem__(self, idx):
        stem = self.samples[idx]
        g = json.load(open(stem + "_gt.json"))
        joints3d = np.array(g["joints3d_cam"], np.float32)        # (21,3) cam, m
        valid = np.array(g["joint_valid"], np.float32)            # (21,)
        K = np.array(g["K"], np.float32)
        k = int(g.get("rot_k", 0))

        bgr = cv2.imread(stem + "_rgb.png")
        image = cv2.cvtColor(_rotate(bgr, k), cv2.COLOR_BGR2RGB)
        H, W = image.shape[:2]

        depth_full_m = None
        if self.with_depth:
            depth_full_m = self._load_depth_full(stem, H, W, k)
            zv = joints3d[valid > 0, 2]
            z0 = float(np.median(zv)) if zv.size else 0.0
            if z0 > 0:
                depth_full_m = hand_band_mask(depth_full_m, z0, g["bbox"], band=self.band)

        joints2d = get_2D_annotation(joints3d, K)

        # square bbox around the hand (native res), small random margin like DexYCB
        margin = 20.0
        mx = int(np.random.rand() * margin); my = int(np.random.rand() * margin)
        x_min = max(joints2d[:, 0].min() - mx, 0.0)
        y_min = max(joints2d[:, 1].min() - my, 0.0)
        x_max = min(joints2d[:, 0].max() + my, float(W))
        y_max = min(joints2d[:, 1].max() + mx, float(H))
        seg_len = max(x_max - x_min, y_max - y_min)
        bbox = [x_min, y_min, seg_len, seg_len]

        if self.mode == "train":
            pil = Image.fromarray(image)
            pil = pil.filter(ImageFilter.GaussianBlur(np.random.rand() * 0.5))
            pil = color_jitter(pil, brightness=0.5, saturation=0.5, hue=0.15, contrast=0.5)
            image = np.array(pil)

        aug_img, img2bb_trans, bb2img_trans, rot, _, cam, cam_nh, _ = augmentation(
            image, bbox, self.mode, exclude_flip=True, rotation=True, cam_param=K)

        rot = -rot * np.pi / 180.0
        rot_mat = np.array([[np.cos(rot), -np.sin(rot), 0],
                            [np.sin(rot), np.cos(rot), 0],
                            [0, 0, 1]], dtype=np.float32)
        rot_joints = rot_mat.dot(joints3d.T).T
        root_xyz = rot_joints[0].copy()
        align_joints = rot_joints - root_xyz

        ori_2d = np.concatenate((joints2d, np.ones((21, 1))), axis=1)
        kps = (img2bb_trans @ ori_2d.T).T[:, :2] / 256.0

        return_img = torch.from_numpy(
            np.ascontiguousarray(aug_img.astype(np.uint8))).permute(2, 0, 1).float() / 255.0
        depth_med = 0.0
        if self.with_depth:
            depth_crop = cv2.warpAffine(depth_full_m, img2bb_trans, (256, 256),
                                        flags=cv2.INTER_NEAREST)
            is_train = self.mode == "train"
            rng = None if is_train else np.random.default_rng(int(idx))
            depth_crop = corrupt_depth(depth_crop, self.depth_cfg, train=is_train, rng=rng)
            dvalid = depth_crop > 0
            depth_med = float(np.median(depth_crop[dvalid])) if dvalid.any() else 0.0
            depth_norm = normalize_depth(depth_crop, scale=self.depth_cfg["norm_scale"])
            depth_t = torch.from_numpy(depth_norm).float().unsqueeze(0)
            return_img = torch.cat([return_img, depth_t], dim=0)        # [4,256,256]

        new_cam = cam_nh.copy()
        new_cam[:, 2] = cam[:, 2]
        out = {
            "image": return_img,
            "keypoints3D": align_joints.astype(np.float32),
            "keypoints2D": kps.astype(np.float32),
            "root": root_xyz.astype(np.float32),
            "cam": new_cam.astype(np.float32),
            "joint_valid": valid.astype(np.float32),
        }
        if self.return_abs_depth:
            out["depth_med"] = np.float32(depth_med)
            out["root_anchor"], out["anchor_valid"] = np.zeros(3, np.float32), np.float32(0.0)
        return out


# python -m datasets.iphone_captures --root rgbd_captures --check 8
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="rgbd_captures")
    ap.add_argument("--mode", default="train")
    ap.add_argument("--check", default=8, type=int)
    args = ap.parse_args()

    ds = IPhoneCaptures_RGBD(root=args.root, mode=args.mode)
    print("len:", len(ds))
    if len(ds):
        os.makedirs("iphone_check", exist_ok=True)
        step = max(1, len(ds) // max(args.check, 1))
        for i in range(0, len(ds), step):
            d = ds[i]
            img = (d["image"][:3].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            dep = d["image"][3].numpy()
            kps = (d["keypoints2D"] * 256).astype(int)
            ov = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()
            for u, v in kps:
                cv2.circle(ov, (int(u), int(v)), 2, (0, 0, 255), -1)
            cv2.imwrite(f"iphone_check/s{i:04d}_rgb.png", ov)
            cv2.imwrite(f"iphone_check/s{i:04d}_depth.png",
                        (np.clip((dep - dep.min()) / (np.ptp(dep) + 1e-6), 0, 1) * 255).astype(np.uint8))
            print(f"{i}: img{tuple(d['image'].shape)} valid={int(d['joint_valid'].sum())}/21 "
                  f"root_z={d['root'][2]:.3f}m kps3d_span={np.ptp(d['keypoints3D'], 0)}")
