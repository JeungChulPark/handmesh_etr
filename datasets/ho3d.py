"""HO3D -> FreiHAND/HanCo RGB-D adapter.

Emits the *same* per-sample dict as `datasets/hanco_ty.HanCo_ETRI_jitter` (see
`datasets/rgbd_datasets.md` for the full contract) so the RGB-D trainer
(`train_mobrecon_rgbd.py`) and `LargeModel_Extra_RGBD` consume HO3D with no
change. Companion to `datasets/dexycb.py`; only the dataset-specific I/O differs.

Why HO3D: real RealSense depth + MANO GT under heavy hand-object occlusion -- a
good second real-depth source after DexYCB for occlusion robustness.

HO3D facts encoded here (verified against shreyashampali/ho3d toolkit)
---------------------------------------------------------------------
  layout:   <root>/train/<seq>/rgb/<f>.jpg          (or .png)
            <root>/train/<seq>/depth/<f>.png        (2-channel encoded)
            <root>/train/<seq>/meta/<f>.pkl
            <root>/train.txt                        (optional split list)
  meta.pkl: handJoints3D (21,3) metres, **OpenGL coords**; camMat (3,3);
            handPose(48) handTrans(3) handBeta(10); object fields.
            NOTE: only the **train** split has full 21 joints; evaluation gives
            the wrist only -> this adapter uses the train split.
  depth:    dpt = (D[:,:,2] + D[:,:,1]*256) * 0.00012498664727900177  (metres),
            channels read in OpenCV BGR order (so R + G*256).
  coords:   joints are OpenGL -> apply coordChangeMat = diag(1,-1,-1) to get the
            OpenCV camera frame (z forward) used by cam2pixel: u = X*fx/z + cx.
  joints:   HO3D order == MANO/FreiHAND (the toolkit's limb list is the MANO
            connectivity applied directly), so JOINT_PERM is identity.

Depth masking: HO3D has no segmentation mask, so `depth_mode="hand_masked"`
(default) keeps only pixels whose metric depth lies in the hand's z-band
(GT joint z-range +/- `hand_band_m`), dropping the far table -> reproduces the
hand-only depth distribution the model was trained on (the held object, at hand
depth, is kept -- realistic occlusion). `depth_mode="scene"` keeps full depth.

Usage
-----
    from datasets.ho3d import HO3D_RGBD
    train_dataset = HO3D_RGBD(root="/path/HO3D_v3", mode="train", with_depth=True)
    python -m datasets.ho3d --root /path/HO3D_v3 --check 8
"""

import os
import glob
import pickle
import argparse

import numpy as np
import cv2
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset

from datasets.augmentation import color_jitter
from datasets.dataset_utils import augmentation
from datasets.depth_synth import add_sensor_noise, normalize_depth, corrupt_depth

# HO3D handJoints3D order == MANO/FreiHAND; identity unless a check says otherwise.
JOINT_PERM = np.arange(21)

# OpenGL -> OpenCV camera frame (negate y, z). Symmetric, so M == M.T.
COORD_CHANGE = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]], dtype=np.float32)
DEPTH_SCALE = 0.00012498664727900177  # metres per unit


def get_2D_annotation(xyz, K):
    """Project camera-space joints (metres) to pixels: uv = K @ X / z."""
    uv = np.matmul(K, xyz.T).T
    return uv[:, :2] / uv[:, -1:]


def decode_ho3d_depth(depth_path):
    """Decode an HO3D depth PNG to metric depth (metres); 0 where missing."""
    d = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # BGR, HxWx3 uint8/16
    if d is None:
        return None
    if d.ndim == 2:  # already single-channel mm (some redistributions)
        return d.astype(np.float32) * DEPTH_SCALE
    dpt = d[:, :, 2].astype(np.float32) + d[:, :, 1].astype(np.float32) * 256.0
    return dpt * DEPTH_SCALE


class HO3D_RGBD(Dataset):
    def __init__(self, root, mode="train", split="train", img_size=256,
                 with_depth=True, depth_mode="hand_masked", depth_cfg=None,
                 hand_band_m=0.06, sequences=None, limit=0, return_abs_depth=False):
        """
        Args:
            root: HO3D dataset root (contains train/, evaluation/).
            mode: "train" enables aug + sensor-noise jitter; else deterministic.
            split: which HO3D split dir to read ("train" has full 21 joints).
            with_depth: emit the 4th (real depth) channel.
            depth_mode: "hand_masked" (z-band around the hand) or "scene" (full).
            depth_cfg: noise-model / normalisation overrides (see _default_depth_cfg).
            hand_band_m: +/- metres around the GT joint z-range kept when masking.
            sequences: optional list of sequence names to include (else all).
            limit: cap index length for smoke tests (0 = no cap).
        """
        self.root = root
        self.mode = mode
        self.split = split
        self.img_size = img_size
        self.with_depth = with_depth
        self.depth_mode = depth_mode
        self.hand_band_m = hand_band_m
        # return_abs_depth: additive opt-in -> also emit "depth_med" (raw-depth
        # median = absolute distance) for the dual-stream TranslationHead. Default
        # off keeps the dict identical to the early-fusion baseline.
        self.return_abs_depth = return_abs_depth
        self.depth_cfg = {**self._default_depth_cfg(), **(depth_cfg or {})}

        self.samples = self._build_index(sequences, limit)
        print(f"[HO3D_RGBD] {len(self.samples)} frames  split={split} mode={mode} "
              f"depth_mode={depth_mode}")

    @staticmethod
    def _default_depth_cfg():
        # Real sensor depth -> light synthetic corruption only (vs HanCo render).
        return dict(sigma=0.002, dropout_p=0.0, quant=0.001, n_holes=0,
                    hole_frac=0.15, norm_scale=0.1)

    # ------------------------------------------------------------------ index
    def _build_index(self, sequences, limit):
        split_dir = os.path.join(self.root, self.split)
        seqs = sorted(d for d in os.listdir(split_dir)
                      if os.path.isdir(os.path.join(split_dir, d)))
        if sequences is not None:
            seqs = [s for s in seqs if s in set(sequences)]

        samples = []
        for seq in seqs:
            meta_dir = os.path.join(split_dir, seq, "meta")
            if not os.path.isdir(meta_dir):
                continue
            for meta_fp in sorted(glob.glob(os.path.join(meta_dir, "*.pkl"))):
                fid = os.path.splitext(os.path.basename(meta_fp))[0]
                base = os.path.join(split_dir, seq)
                rgb = os.path.join(base, "rgb", fid + ".jpg")
                if not os.path.isfile(rgb):
                    rgb = os.path.join(base, "rgb", fid + ".png")
                samples.append(dict(
                    rgb=rgb,
                    depth=os.path.join(base, "depth", fid + ".png"),
                    meta=meta_fp))
                if limit and len(samples) >= limit:
                    return samples
        return samples

    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------------- depth chan
    def _depth_channel(self, depth_full_m, joints3d, img2bb_trans, idx):
        dc = self.depth_cfg
        if self.depth_mode == "hand_masked":
            z = joints3d[:, 2]
            zlo, zhi = float(z.min()) - self.hand_band_m, float(z.max()) + self.hand_band_m
            band = (depth_full_m >= zlo) & (depth_full_m <= zhi)
            depth_full_m = np.where(band, depth_full_m, 0.0).astype(np.float32)
        depth_crop = cv2.warpAffine(depth_full_m, img2bb_trans, (256, 256),
                                    flags=cv2.INTER_NEAREST)
        is_train = self.mode == "train"
        rng = None if is_train else np.random.default_rng(int(idx))
        depth_crop = corrupt_depth(depth_crop, dc, train=is_train, rng=rng)
        valid = depth_crop > 0
        depth_med = float(np.median(depth_crop[valid])) if valid.any() else 0.0
        depth_norm = normalize_depth(depth_crop, scale=dc["norm_scale"])
        return torch.from_numpy(depth_norm).float().unsqueeze(0), depth_med

    # --------------------------------------------------------------- getitem
    def __getitem__(self, idx):
        s = self.samples[idx]
        with open(s["meta"], "rb") as f:
            meta = pickle.load(f, encoding="latin1")

        hj = meta["handJoints3D"]
        hj = np.asarray(hj, dtype=np.float32)
        if hj.shape != (21, 3):  # evaluation split / missing -> skip
            return self.__getitem__((idx + 1) % len(self.samples))
        # OpenGL -> OpenCV camera frame (z forward), then MANO order.
        joints3d = (hj @ COORD_CHANGE)[JOINT_PERM]
        if not np.isfinite(joints3d).all() or (joints3d[:, 2] <= 0).any():
            return self.__getitem__((idx + 1) % len(self.samples))
        K = np.asarray(meta["camMat"], dtype=np.float32).copy()

        image = cv2.cvtColor(cv2.imread(s["rgb"]), cv2.COLOR_BGR2RGB)
        depth_full_m = None
        if self.with_depth:
            depth_full_m = decode_ho3d_depth(s["depth"])
            if depth_full_m is None:
                depth_full_m = np.zeros(image.shape[:2], np.float32)

        return self._finalize(image, depth_full_m, joints3d, K, idx)

    def _finalize(self, image, depth_full_m, joints3d, K, idx):
        """Shared crop+augment+output assembly given camera-frame joints (metres),
        RGB image, native depth (m), and intrinsics. Reused by H2O3D_RGBD."""
        joints2d = get_2D_annotation(joints3d, K)

        margin = 20.0
        mx = int(np.random.rand() * margin); my = int(np.random.rand() * margin)
        x_min = max(joints2d[:, 0].min() - mx, 0.0)
        y_min = max(joints2d[:, 1].min() - my, 0.0)
        x_max = min(joints2d[:, 0].max() + my, float(image.shape[1]))
        y_max = min(joints2d[:, 1].max() + mx, float(image.shape[0]))
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
            depth_t, depth_med = self._depth_channel(depth_full_m, joints3d, img2bb_trans, idx)
            return_img = torch.cat([return_img, depth_t], dim=0)

        new_cam = cam_nh.copy()
        new_cam[:, 2] = cam[:, 2]
        out = {
            "image": return_img,
            "keypoints3D": align_joints,
            "keypoints2D": kps,
            "root": root_xyz,
            "cam": new_cam,
        }
        if getattr(self, "return_abs_depth", False):
            out["depth_med"] = np.float32(depth_med)
        return out


class H2O3D_RGBD(HO3D_RGBD):
    """H2O-3D: two hands manipulating an object, HOnnotate format -- identical to
    HO3D (OpenGL coords, camMat, 2-channel depth, MANO joint order), but meta has
    per-hand `rightHandJoints3D` / `leftHandJoints3D` (21,3 each).

    Our model is single right-hand, so we pick one hand per frame (default right,
    falling back to left if the right is absent). `flip_left` mirrors a left hand
    into the right-hand convention (image+depth flipped, joint x and cx mirrored),
    so left-hand frames are usable. depth_mode / z-band masking are inherited.
    """
    name = "h2o3d"

    def __init__(self, *a, hand="right", flip_left=True, **k):
        self.hand = hand
        self.flip_left = flip_left
        super().__init__(*a, **k)

    def _hand_joints(self, meta, which):
        hj = meta.get("rightHandJoints3D" if which == "right" else "leftHandJoints3D")
        if hj is None:
            return None
        hj = np.asarray(hj, dtype=np.float32)
        if hj.shape != (21, 3) or not np.isfinite(hj).all():
            return None
        return (hj @ COORD_CHANGE)[JOINT_PERM]  # OpenGL -> camera frame, MANO order

    def __getitem__(self, idx):
        s = self.samples[idx]
        with open(s["meta"], "rb") as f:
            meta = pickle.load(f, encoding="latin1")

        which = self.hand
        joints3d = self._hand_joints(meta, which)
        if joints3d is None and self.hand == "right":     # fall back to the left
            which, joints3d = "left", self._hand_joints(meta, "left")
        if joints3d is None or (joints3d[:, 2] <= 0).any():
            return self.__getitem__((idx + 1) % len(self.samples))
        K = np.asarray(meta["camMat"], dtype=np.float32).copy()

        image = cv2.cvtColor(cv2.imread(s["rgb"]), cv2.COLOR_BGR2RGB)
        depth_full_m = None
        if self.with_depth:
            depth_full_m = decode_ho3d_depth(s["depth"])
            if depth_full_m is None:
                depth_full_m = np.zeros(image.shape[:2], np.float32)

        if self.flip_left and which == "left":            # fold left -> right
            W = image.shape[1]
            image = image[:, ::-1, :].copy()
            joints3d = joints3d.copy(); joints3d[:, 0] *= -1.0
            K[0, 2] = W - 1 - K[0, 2]
            if depth_full_m is not None:
                depth_full_m = depth_full_m[:, ::-1].copy()

        return self._finalize(image, depth_full_m, joints3d, K, idx)


# python -m datasets.ho3d --root /path/HO3D_v3 --check 8
# python -m datasets.ho3d --root /path/H2O3D --dataset h2o3d --check 8
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="HO3D / H2O-3D dataset root")
    ap.add_argument("--dataset", default="ho3d", choices=["ho3d", "h2o3d"])
    ap.add_argument("--mode", default="train")
    ap.add_argument("--split", default="train")
    ap.add_argument("--depth_mode", default="hand_masked", choices=["hand_masked", "scene"])
    ap.add_argument("--check", default=8, type=int)
    args = ap.parse_args()

    _cls = HO3D_RGBD if args.dataset == "ho3d" else H2O3D_RGBD
    ds = _cls(root=args.root, mode=args.mode, split=args.split,
              depth_mode=args.depth_mode, limit=2000)
    print("len:", len(ds))
    od = f"{args.dataset}_check"
    os.makedirs(od, exist_ok=True)
    step = max(1, len(ds) // max(args.check, 1))
    for i in range(0, min(len(ds), step * args.check), step):
        d = ds[i]
        img = d["image"]
        rgb = (img[:3].permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
        kp = (d["keypoints2D"] * 256).astype(np.int32)
        for (x, y) in kp:
            cv2.circle(rgb, (int(x), int(y)), 2, (0, 255, 0), -1)
        out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if img.shape[0] == 4:
            dvis = ((img[3].clamp(-1, 1).numpy() + 1) / 2 * 255).astype(np.uint8)
            dvis = cv2.applyColorMap(dvis, cv2.COLORMAP_JET)
            out = np.concatenate([out, dvis], axis=1)
        cv2.imwrite(f"{od}/sample_{i:05d}.png", out)
        print(f"  sample {i}: img{tuple(img.shape)} z_root={d['root'][2]:.3f}m")
    print(f"wrote {od}/*.png")
