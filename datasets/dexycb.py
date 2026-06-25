"""DexYCB -> FreiHAND/HanCo RGB-D adapter.

Goal
----
Make the DexYCB dataset emit the *exact same* per-sample dict that the project's
canonical RGB-D loader `datasets/hanco_ty.HanCo_ETRI_jitter` produces, so the
RGB-D trainer (`train_mobrecon_rgbd.py`) and `LargeModel_Extra_RGBD` consume it
without any change. The one substantive difference from HanCo is the depth
channel: HanCo has no sensor depth and *renders* a synthetic capsule-mesh depth
(datasets/depth_synth.py), whereas DexYCB ships a **real aligned metric depth**
map from an Intel RealSense, which is exactly the signal we want to train on.

Output contract (identical to HanCo_ETRI_jitter.__getitem__)
-----------------------------------------------------------
    {
      "image":       FloatTensor [4, 256, 256]  RGB in [0,1] (ch 0-2) + depth in
                                                 [-1,1] (ch 3), pixel-aligned.
      "keypoints3D": [21, 3] root-aligned camera-space joints (metres), rotated
                     by the in-plane crop rotation (root = joint 0 at origin).
      "keypoints2D": [21, 2] crop-pixel coords / 256.
      "root":        [3] camera-space root joint (metres), for 2D reprojection.
      "cam":         [3, 3] intrinsics adjusted to the 256 crop (== HanCo "cam").
    }

DexYCB facts encoded here (verified against NVlabs/dex-ycb-toolkit)
------------------------------------------------------------------
  layout:   <root>/<YYYYMMDD-subject-NN>/<seq-timestamp>/<serial>/
                color_{f:06d}.jpg
                aligned_depth_to_color_{f:06d}.png   (uint16, millimetres)
                labels_{f:06d}.npz                   (joint_3d, joint_2d, seg, pose_m)
            <root>/calibration/intrinsics/<serial>_640x480.yml   (key 'color')
            <seq>/meta.yml   (serials[8], mano_sides, num_frames, ...)
  labels:   joint_3d [1,21,3] CAMERA coords (metres); -1 when the hand is absent.
            seg      [H,W] uint8: 0 bg, 1-21 YCB objects, 255 hand.
  joints:   MANO 21-joint order == FreiHAND/HanCo order (wrist, thumb..pinky,
            each MCP->TIP), so no re-permutation is needed. See JOINT_PERM.

Depth handling (why masking matters)
------------------------------------
The RGB-D model was warm-started/trained on HanCo's *hand-only* synthetic depth
(background = 0, per-frame median centring). Real DexYCB depth contains the whole
scene (arm, table, objects), so feeding it raw shifts the depth distribution the
network was tuned on. `depth_mode="hand_masked"` (default) keeps only seg==255
pixels (background -> 0), reproducing the training distribution. `depth_mode=
"scene"` keeps the full sensor depth (closer to a real Femto-Bolt deployment that
has no hand mask) -- useful for a fine-tune/robustness study, with a known
train/test distribution shift. See datasets/rgbd_datasets.md.

Usage
-----
    from datasets.dexycb import DexYCB_RGBD
    ds = DexYCB_RGBD(root="/path/DexYCB", mode="train")       # right hands only
    # drop-in for HanCo_ETRI_jitter in train_mobrecon_rgbd.py:
    #   train_dataset = DexYCB_RGBD(root=..., mode="train", with_depth=True)

    python -m datasets.dexycb --root /path/DexYCB --check 8    # smoke test + viz
"""

import os
import glob
import argparse

import numpy as np
import cv2
import yaml
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset

from datasets.augmentation import color_jitter
from datasets.dataset_utils import augmentation
from datasets.depth_synth import add_sensor_noise, normalize_depth, corrupt_depth

# DexYCB MANO 21-joint order matches FreiHAND/HanCo (and project depth_synth
# HAND_BONES). If a future check shows a mismatch, set this permutation; identity
# means "use as-is".
JOINT_PERM = np.arange(21)

_SERIALS = ('836212060125', '839512060362', '840412060917', '841412060263',
            '932122060857', '932122060861', '932122061900', '932122062010')
_W, _H = 640, 480
HAND_SEG_ID = 255

# DexYCB intrinsics yml files carry an `extrinsics: !!python/tuple` tag that
# yaml.safe_load rejects; teach SafeLoader to read it as a plain tuple so we can
# still grab the clean `color` block.
yaml.SafeLoader.add_constructor(
    'tag:yaml.org,2002:python/tuple',
    lambda loader, node: tuple(loader.construct_sequence(node)))


def get_2D_annotation(xyz, K):
    """Project camera-space joints (metres) to pixels: uv = K @ X / z."""
    uv = np.matmul(K, xyz.T).T
    return uv[:, :2] / uv[:, -1:]


class DexYCB_RGBD(Dataset):
    def __init__(self, root, mode="train", img_size=256,
                 with_depth=True, depth_mode="hand_masked", depth_cfg=None,
                 hands="right", flip_left=False, subjects=None, limit=0,
                 return_abs_depth=False):
        """
        Args:
            root: DexYCB dataset root ($DEX_YCB_DIR).
            mode: "train" enables aug + sensor-noise jitter; anything else is
                deterministic (eval/test), mirroring HanCo_ETRI_jitter.mode.
            with_depth: emit a 4th (depth) channel. False -> 3-channel RGB only.
            depth_mode: "hand_masked" (seg==255 -> keep, else 0; matches the
                synthetic-depth training distribution) or "scene" (full sensor
                depth, deployment-realistic but distribution-shifted).
            depth_cfg: noise-model / normalisation overrides (see _default_depth_cfg).
            hands: "right" | "left" | "both". The RGB pipeline is right-hand-only
                (exclude_flip=True); use flip_left to fold left hands into right.
            flip_left: horizontally mirror left-hand frames into the right-hand
                convention (image+depth flipped, joints x and cx mirrored). Lets
                you train on every hand while honouring the right-hand contract.
            subjects: optional list of subject-dir names to include (else all).
            limit: cap the index length for quick smoke tests (0 = no cap).
        """
        self.root = root
        self.mode = mode
        self.img_size = img_size
        self.with_depth = with_depth
        self.depth_mode = depth_mode
        self.flip_left = flip_left
        # return_abs_depth: additive opt-in for the dual-stream model -> also emit
        # "depth_med" (raw-depth median, metres = absolute distance). Default off
        # keeps the dict identical to the early-fusion baseline.
        self.return_abs_depth = return_abs_depth
        self.depth_cfg = {**self._default_depth_cfg(), **(depth_cfg or {})}

        self._intr_cache = {}
        self.samples = self._build_index(hands, subjects, limit)
        print(f"[DexYCB_RGBD] {len(self.samples)} frames  mode={mode} "
              f"depth_mode={depth_mode} hands={hands} flip_left={flip_left}")

    # ----------------------------------------------------------------- config
    @staticmethod
    def _default_depth_cfg():
        # Real sensor depth is already noisy, so synthetic corruption is *light*
        # (vs HanCo's render which needs more to look like a sensor). dropout/
        # holes default to 0; a small sigma in train mode acts as augmentation.
        return dict(sigma=0.002, dropout_p=0.0, quant=0.001, n_holes=0,
                    hole_frac=0.15, norm_scale=0.1)

    # ------------------------------------------------------------------ index
    def _load_intrinsics(self, serial):
        if serial in self._intr_cache:
            return self._intr_cache[serial]
        fp = os.path.join(self.root, "calibration", "intrinsics",
                          f"{serial}_{_W}x{_H}.yml")
        with open(fp, "r") as f:
            c = yaml.safe_load(f)["color"]
        K = np.array([[c["fx"], 0.0, c["ppx"]],
                      [0.0, c["fy"], c["ppy"]],
                      [0.0, 0.0, 1.0]], dtype=np.float32)
        self._intr_cache[serial] = K
        return K

    def _build_index(self, hands, subjects, limit):
        samples = []
        subj_dirs = sorted(d for d in os.listdir(self.root)
                           if "-subject-" in d and
                           os.path.isdir(os.path.join(self.root, d)))
        if subjects is not None:
            subj_dirs = [d for d in subj_dirs if d in set(subjects)]

        for subj in subj_dirs:
            subj_path = os.path.join(self.root, subj)
            for seq in sorted(os.listdir(subj_path)):
                seq_path = os.path.join(subj_path, seq)
                meta_fp = os.path.join(seq_path, "meta.yml")
                if not os.path.isfile(meta_fp):
                    continue
                with open(meta_fp, "r") as f:
                    meta = yaml.safe_load(f)
                side = meta.get("mano_sides", ["right"])[0]
                if hands != "both" and side != hands:
                    continue
                serials = meta.get("serials", _SERIALS)
                n = int(meta["num_frames"])
                for serial in serials:
                    K = self._load_intrinsics(serial)
                    cam_dir = os.path.join(seq_path, serial)
                    if not os.path.isdir(cam_dir):
                        continue
                    for fidx in range(n):
                        samples.append(dict(
                            color=os.path.join(cam_dir, f"color_{fidx:06d}.jpg"),
                            depth=os.path.join(cam_dir, f"aligned_depth_to_color_{fidx:06d}.png"),
                            label=os.path.join(cam_dir, f"labels_{fidx:06d}.npz"),
                            K=K, side=side))
                        if limit and len(samples) >= limit:
                            return samples
        return samples

    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------------- depth chan
    def _depth_channel(self, depth_full_m, seg, img2bb_trans, idx):
        """Warp native real depth into the 256 crop and normalise -> [1,256,256]."""
        dc = self.depth_cfg
        if self.depth_mode == "hand_masked" and seg is not None:
            depth_full_m = np.where(seg == HAND_SEG_ID, depth_full_m, 0.0).astype(np.float32)
        depth_crop = cv2.warpAffine(depth_full_m, img2bb_trans, (256, 256),
                                    flags=cv2.INTER_NEAREST)
        is_train = self.mode == "train"
        rng = None if is_train else np.random.default_rng(int(idx))
        depth_crop = corrupt_depth(depth_crop, dc, train=is_train, rng=rng)
        valid = depth_crop > 0
        depth_med = float(np.median(depth_crop[valid])) if valid.any() else 0.0
        depth_norm = normalize_depth(depth_crop, scale=dc["norm_scale"])
        return torch.from_numpy(depth_norm).float().unsqueeze(0), depth_med

    @staticmethod
    def _to_metres(joint_3d):
        """DexYCB joint_3d is metres; auto-correct if a copy is in millimetres."""
        z = np.abs(joint_3d[:, 2])
        med = float(np.median(z[z > 0])) if np.any(z > 0) else 0.0
        return joint_3d / 1000.0 if med > 10.0 else joint_3d  # >10 => mm

    # --------------------------------------------------------------- getitem
    def __getitem__(self, idx):
        s = self.samples[idx]
        lab = np.load(s["label"])
        joints3d = lab["joint_3d"][0].astype(np.float32)          # (21,3) cam, m
        if np.any(joints3d <= -1.0 + 1e-6) and (joints3d < 0).all(axis=1).any():
            return self.__getitem__((idx + 1) % len(self.samples))  # hand absent
        joints3d = self._to_metres(joints3d)[JOINT_PERM]
        K = s["K"].copy()

        image = cv2.cvtColor(cv2.imread(s["color"]), cv2.COLOR_BGR2RGB)
        depth_full_m = None
        seg = None
        if self.with_depth:
            d = cv2.imread(s["depth"], cv2.IMREAD_UNCHANGED)
            depth_full_m = (d.astype(np.float32) / 1000.0) if d is not None else \
                np.zeros(image.shape[:2], np.float32)
            if self.depth_mode == "hand_masked":
                seg = lab["seg"]

        # Fold a left hand into the right-hand convention by mirroring x.
        if self.flip_left and s["side"] == "left":
            image = image[:, ::-1, :].copy()
            joints3d[:, 0] *= -1.0
            K[0, 2] = _W - 1 - K[0, 2]
            if depth_full_m is not None:
                depth_full_m = depth_full_m[:, ::-1].copy()
            if seg is not None:
                seg = seg[:, ::-1].copy()

        joints2d = get_2D_annotation(joints3d, K)

        # square bbox around the hand (native res), random margin like HanCo
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
            depth_t, depth_med = self._depth_channel(depth_full_m, seg, img2bb_trans, idx)
            return_img = torch.cat([return_img, depth_t], dim=0)  # [4,256,256]

        new_cam = cam_nh.copy()
        new_cam[:, 2] = cam[:, 2]
        out = {
            "image": return_img,
            "keypoints3D": align_joints,
            "keypoints2D": kps,
            "root": root_xyz,
            "cam": new_cam,
        }
        if self.return_abs_depth:
            out["depth_med"] = np.float32(depth_med)
        return out


# python -m datasets.dexycb --root /path/DexYCB --check 8
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="DexYCB dataset root")
    ap.add_argument("--mode", default="train")
    ap.add_argument("--depth_mode", default="hand_masked", choices=["hand_masked", "scene"])
    ap.add_argument("--check", default=8, type=int, help="num samples to dump")
    args = ap.parse_args()

    ds = DexYCB_RGBD(root=args.root, mode=args.mode, depth_mode=args.depth_mode, limit=2000)
    print("len:", len(ds))
    os.makedirs("dexycb_check", exist_ok=True)
    step = max(1, len(ds) // max(args.check, 1))
    for i in range(0, min(len(ds), step * args.check), step):
        d = ds[i]
        img = d["image"]
        rgb = (img[:3].permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
        # overlay 2D joints to verify alignment
        kp = (d["keypoints2D"] * 256).astype(np.int32)
        for (x, y) in kp:
            cv2.circle(rgb, (int(x), int(y)), 2, (0, 255, 0), -1)
        out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if img.shape[0] == 4:
            dvis = ((img[3].clamp(-1, 1).numpy() + 1) / 2 * 255).astype(np.uint8)
            dvis = cv2.applyColorMap(dvis, cv2.COLORMAP_JET)
            out = np.concatenate([out, dvis], axis=1)
        cv2.imwrite(f"dexycb_check/sample_{i:05d}.png", out)
        print(f"  sample {i}: img{tuple(img.shape)} "
              f"kps3d[root]={d['keypoints3D'][0].round(3)} "
              f"z_root={d['root'][2]:.3f}m")
    print("wrote dexycb_check/*.png")
