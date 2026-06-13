"""ContactPose & OakInk -> RGB-D adapter (via their official toolkits).

Both are large hand-object MANO datasets accessed through their own Python
toolkits (not flat files), so these loaders **wrap the toolkit API** rather than
re-deriving the on-disk schema, and emit the project's standard RGB-D contract
(see datasets/rgbd_datasets.md). Each toolkit is imported lazily with a clear
install hint.

Two important, verified facts that shape these loaders:

* **ContactPose** (`utilities.dataset.ContactPose`) DOES expose real Kinect-v2
  depth (`image_filenames('depth', frame)`), and hand joints are given **w.r.t.
  the object** -> we map them to the camera with `object_pose(cam, frame)` (cTo).
  => a genuine real-depth RGB-D source.

* **OakInk** (`oikit.oi_image.OakInkImage`) does NOT surface sensor depth (there
  is no `get_depth()`); it only provides RGB + MANO + camera-frame joints. So the
  OakInk loader RENDERS a metric depth channel from the joints (datasets/
  depth_synth.py, same path HanCo uses) and flags `depth_is_synthetic=True`. Use
  OakInk for diverse RGB + accurate MANO; it does not add real sensor depth.

Joint order: ContactPose returns OpenPose-21 and OakInk returns MANO-21; both use
the wrist->thumb..pinky (MCP->TIP) convention that matches FreiHAND/HanCo here, so
JOINT_PERM is identity. CONFIRM with the --check viz on real data before training.

Usage:
    from datasets.oakink_contactpose import ContactPose_RGBD, OakInk_RGBD
    ds = ContactPose_RGBD(data_dir="/path/contactpose_data", mode="train")
    ds = OakInk_RGBD(data_split="train", mode_split="default")  # needs OAKINK_DIR
"""

import os
import argparse

import numpy as np
import cv2
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset

from datasets.augmentation import color_jitter
from datasets.dataset_utils import augmentation
from datasets.depth_synth import (render_hand_depth, add_sensor_noise, normalize_depth)

JOINT_PERM = np.arange(21)  # OpenPose-21 / MANO-21 == FreiHAND order here


def _project(xyz_m, K):
    uv = (K @ xyz_m.T).T
    return uv[:, :2] / uv[:, -1:]


def _to_metres(joints):
    z = np.abs(joints[:, 2])
    med = float(np.median(z[z > 0])) if np.any(z > 0) else 0.0
    return joints / 1000.0 if med > 10.0 else joints  # >10 => millimetres


def _default_depth_cfg():
    return dict(base_radius=0.009, palm_radius=0.014, tip_scale=0.55, ring=6,
                sigma=0.003, dropout_p=0.03, quant=0.001, n_holes=1, hole_frac=0.1,
                norm_scale=0.1, hand_band_m=0.08)


def assemble_rgbd(image, joints3d, K, idx, mode, with_depth, depth_cfg,
                  depth_full_m=None, depth_mode="hand_masked"):
    """Shared crop+augment+output assembly (mirrors ho3d/dexycb _finalize).

    joints3d: (21,3) camera-frame metres, MANO order. If `depth_full_m` is given
    (real sensor depth, metres, native res) it is warped into the crop; otherwise
    a metric depth is RENDERED from joints (synthetic, OakInk path).
    """
    dc = depth_cfg
    joints2d = _project(joints3d, K)

    margin = 20.0
    mx = int(np.random.rand() * margin); my = int(np.random.rand() * margin)
    x_min = max(joints2d[:, 0].min() - mx, 0.0)
    y_min = max(joints2d[:, 1].min() - my, 0.0)
    x_max = min(joints2d[:, 0].max() + my, float(image.shape[1]))
    y_max = min(joints2d[:, 1].max() + mx, float(image.shape[0]))
    seg = max(x_max - x_min, y_max - y_min)
    bbox = [x_min, y_min, seg, seg]

    if mode == "train":
        pil = Image.fromarray(image)
        pil = pil.filter(ImageFilter.GaussianBlur(np.random.rand() * 0.5))
        pil = color_jitter(pil, brightness=0.5, saturation=0.5, hue=0.15, contrast=0.5)
        image = np.array(pil)

    aug_img, img2bb_trans, bb2img_trans, rot, _, cam, cam_nh, _ = augmentation(
        image, bbox, mode, exclude_flip=True, rotation=True, cam_param=K)

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

    if with_depth:
        if depth_full_m is None:                                   # render (OakInk)
            depth_full_m = render_hand_depth(
                K, image.shape[0], image.shape[1], joints=joints3d,
                base_radius=dc["base_radius"], palm_radius=dc["palm_radius"],
                tip_scale=dc["tip_scale"], ring=dc["ring"])
        elif depth_mode == "hand_masked":                          # z-band (ContactPose)
            z = joints3d[:, 2]
            lo, hi = z.min() - dc["hand_band_m"], z.max() + dc["hand_band_m"]
            depth_full_m = np.where((depth_full_m >= lo) & (depth_full_m <= hi),
                                    depth_full_m, 0.0).astype(np.float32)
        depth_crop = cv2.warpAffine(depth_full_m, img2bb_trans, (256, 256),
                                    flags=cv2.INTER_NEAREST)
        if mode == "train":
            depth_crop = add_sensor_noise(depth_crop, sigma=dc["sigma"],
                dropout_p=dc["dropout_p"], quant=dc["quant"], n_holes=dc["n_holes"],
                hole_frac=dc["hole_frac"])
        else:
            depth_crop = add_sensor_noise(depth_crop, sigma=0.0, dropout_p=0.0,
                quant=dc["quant"], n_holes=0, rng=np.random.default_rng(int(idx)))
        depth_t = torch.from_numpy(normalize_depth(depth_crop, scale=dc["norm_scale"])).float().unsqueeze(0)
        return_img = torch.cat([return_img, depth_t], dim=0)

    new_cam = cam_nh.copy(); new_cam[:, 2] = cam[:, 2]
    return {
        "image": return_img,
        "keypoints3D": align_joints.astype(np.float32),
        "keypoints2D": kps.astype(np.float32),
        "root": root_xyz.astype(np.float32),
        "cam": new_cam.astype(np.float32),
    }


# ============================================================== ContactPose ==
class ContactPose_RGBD(Dataset):
    """ContactPose (CMU): Kinect-v2 RGB-D, MANO, hand-object grasps.

    Wraps `utilities.dataset.ContactPose`. Real depth + joints-w.r.t-object mapped
    to camera via `object_pose`. Index building instantiates each grasp once
    (loads its annotations) to enumerate valid (camera, frame, hand) -- a one-time
    cost; use `limit`, `p_nums`, `intents` to scope it. Grasp objects are LRU-cached.
    """

    def __init__(self, data_dir=None, mode="train", with_depth=True,
                 depth_mode="hand_masked", depth_cfg=None,
                 p_nums=range(1, 51), intents=("use", "handoff"), limit=0, cache_size=4):
        try:
            from utilities import dataset as cp_ds
            from utilities import misc as cp_misc
        except ImportError as e:
            raise ImportError("ContactPose toolkit not found: clone "
                              "facebookresearch/ContactPose and add it to PYTHONPATH") from e
        self._cp_ds = cp_ds
        if data_dir:
            os.environ.setdefault("CONTACTPOSE_DATA_DIR", data_dir)
        self.mode = mode
        self.with_depth = with_depth
        self.depth_mode = depth_mode
        self.depth_cfg = {**_default_depth_cfg(), **(depth_cfg or {})}
        self._cache = {}
        self._cache_order = []
        self._cache_size = cache_size

        self.samples = self._build_index(p_nums, intents, limit)
        print(f"[ContactPose_RGBD] {len(self.samples)} frames mode={mode}")

    def _get_cp(self, key):
        if key in self._cache:
            return self._cache[key]
        p_num, intent, obj = key
        cp = self._cp_ds.ContactPose(p_num, intent, obj, load_mano=False)
        self._cache[key] = cp
        self._cache_order.append(key)
        if len(self._cache_order) > self._cache_size:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)
        return cp

    def _build_index(self, p_nums, intents, limit):
        samples = []
        for p in p_nums:
            for intent in intents:
                try:
                    objs = self._cp_ds.get_object_names(p, intent)
                except Exception:
                    continue
                for obj in objs:
                    try:
                        cp = self._cp_ds.ContactPose(p, intent, obj, load_mano=False)
                    except Exception:
                        continue
                    cams = list(cp.valid_cameras)
                    for fi in range(len(cp)):
                        hj = cp.hand_joints(fi)
                        for hidx in range(len(hj)):
                            if hj[hidx] is None:
                                continue
                            for cam in cams:
                                samples.append((p, intent, obj, cam, fi, hidx))
                                if limit and len(samples) >= limit:
                                    return samples
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, intent, obj, cam, fi, hidx = self.samples[idx]
        cp = self._get_cp((p, intent, obj))
        hj = cp.hand_joints(fi)[hidx]
        if hj is None:
            return self.__getitem__((idx + 1) % len(self.samples))
        joints_obj = np.asarray(hj, np.float32)[JOINT_PERM]          # object frame
        cTo = np.asarray(cp.object_pose(cam, fi), np.float32)        # 4x4 obj->cam
        joints_cam = (cTo @ np.concatenate([joints_obj, np.ones((21, 1))], 1).T).T[:, :3]
        joints_cam = _to_metres(joints_cam)
        K = np.asarray(cp.K(cam), np.float32)

        files = cp.image_filenames("color", fi)
        image = cv2.cvtColor(cv2.imread(files[cam]), cv2.COLOR_BGR2RGB)
        depth_full_m = None
        if self.with_depth:
            dfile = cp.image_filenames("depth", fi)[cam]
            d = cv2.imread(dfile, cv2.IMREAD_UNCHANGED)
            depth_full_m = (d.astype(np.float32) / 1000.0) if d is not None else \
                np.zeros(image.shape[:2], np.float32)

        return assemble_rgbd(image, joints_cam, K, idx, self.mode, self.with_depth,
                             self.depth_cfg, depth_full_m=depth_full_m,
                             depth_mode=self.depth_mode)


# ================================================================== OakInk ==
class OakInk_RGBD(Dataset):
    """OakInk-Image (via oikit): RGB + MANO, 4 third-person views, 230K frames.

    The oikit `OakInkImage` loader exposes NO sensor depth, so the depth channel
    here is RENDERED from the MANO joints (datasets/depth_synth.py) -- same as
    HanCo. `depth_is_synthetic=True`. Real OakInk depth would need raw RealSense
    streams outside oikit.
    """
    depth_is_synthetic = True

    def __init__(self, data_split="train", mode_split="default", mode="train",
                 with_depth=True, depth_cfg=None, limit=0):
        try:
            from oikit.oi_image.oi_image import OakInkImage
        except ImportError as e:
            raise ImportError("OakInk toolkit not found: pip install the 'oikit' "
                              "package (oakink/OakInk) and set OAKINK_DIR") from e
        self.oi = OakInkImage(data_split=data_split, mode_split=mode_split)
        self.mode = mode
        self.with_depth = with_depth
        self.depth_cfg = {**_default_depth_cfg(), **(depth_cfg or {})}
        n = len(self.oi)
        self.indices = list(range(n if not limit else min(n, limit)))
        print(f"[OakInk_RGBD] {len(self.indices)} frames (synthetic depth) mode={mode}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        image = np.asarray(self.oi.get_image(i), np.uint8)
        if image.ndim == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        joints = _to_metres(np.asarray(self.oi.get_joints_3d(i), np.float32)[JOINT_PERM])
        K = np.asarray(self.oi.get_cam_intr(i), np.float32)
        # depth_full_m=None -> assemble_rgbd renders from joints
        return assemble_rgbd(image, joints, K, idx, self.mode, self.with_depth,
                             self.depth_cfg, depth_full_m=None)


# python -m datasets.oakink_contactpose --dataset contactpose --data_dir <path> --check 8
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["contactpose", "oakink"])
    ap.add_argument("--data_dir", default=None, help="ContactPose data dir / unused for oakink")
    ap.add_argument("--data_split", default="train")
    ap.add_argument("--mode", default="train")
    ap.add_argument("--check", default=8, type=int)
    args = ap.parse_args()

    if args.dataset == "contactpose":
        ds = ContactPose_RGBD(data_dir=args.data_dir, mode=args.mode, limit=2000)
    else:
        ds = OakInk_RGBD(data_split=args.data_split, mode=args.mode, limit=2000)
    print("len:", len(ds))
    od = f"{args.dataset}_check"; os.makedirs(od, exist_ok=True)
    step = max(1, len(ds) // max(args.check, 1))
    for i in range(0, min(len(ds), step * args.check), step):
        d = ds[i]
        img = d["image"]
        rgb = (img[:3].permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
        for (x, y) in (d["keypoints2D"] * 256).astype(np.int32):
            cv2.circle(rgb, (int(x), int(y)), 2, (0, 255, 0), -1)
        out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if img.shape[0] == 4:
            dvis = ((img[3].clamp(-1, 1).numpy() + 1) / 2 * 255).astype(np.uint8)
            out = np.concatenate([out, cv2.applyColorMap(dvis, cv2.COLORMAP_JET)], 1)
        cv2.imwrite(f"{od}/sample_{i:05d}.png", out)
        print(f"  {i}: img{tuple(img.shape)} z_root={d['root'][2]:.3f}m")
    print(f"wrote {od}/*.png")
