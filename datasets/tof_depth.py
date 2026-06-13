"""ToF / depth-only hand datasets (MSRA, ICVL, NYU) -> RGB-D adapter.

Purpose
-------
Bring the classic **depth-only** benchmarks into the project's RGB-D pipeline,
mainly to close the **ToF domain gap**: DexYCB/HO3D depth is stereo RealSense,
but the deployment sensor (Orbbec Femto Bolt) is **Time-of-Flight**. MSRA and
ICVL are ToF; training/pre-training on them teaches the depth branch the ToF
hole/edge noise that stereo data never shows. (NYU is structured-light, not ToF,
and is included only for completeness.)

These differ from `dexycb.py`/`ho3d.py` in two structural ways, both handled here:

1. **No RGB.** The 4-channel model needs 3 colour channels, so we synthesise a
   **pseudo-RGB** = the per-frame-normalised depth replicated to 3 channels. The
   colour branch therefore sees a depth-shaped image; that's expected — use these
   sets for a depth-focused *pre-train / fine-tune*, not as RGB supervision.

2. **Joint sets don't all match MANO-21.** The model outputs 21 MANO joints
   (FreiHAND order). Only MSRA has 21; ICVL has 16 (3/finger), NYU's usable
   subset is fingertips+wrist. Each loader maps its joints into the 21 MANO slots
   and returns a **`joint_valid` [21] mask** (1 = supervised, 0 = absent). Train
   with a *masked* 3D loss so absent slots don't inject noise:

       v = item["joint_valid"].float().to(device)[..., None]   # [B,21,1]
       loss_3d = (((pred - kps3d) * v) ** 2).sum() / (v.sum() * 3 + 1e-8)

   MSRA returns all-ones, so it works unmasked too. See datasets/rgbd_datasets.md.

Output: the standard contract (image[4,256,256], keypoints3D, keypoints2D, root,
cam) PLUS `joint_valid` [21]. RGB-only sources (dexycb/ho3d) implicitly have
joint_valid = ones; add it there if you mix all four in one ConcatDataset loop.

Smoke test (verify joint/depth alignment on real data):
    python -m datasets.tof_depth --dataset msra --root /path/cvpr15_MSRAHandGestureDB --check 8
    python -m datasets.tof_depth --dataset icvl --root /path/ICVL --check 8
    python -m datasets.tof_depth --dataset nyu  --root /path/nyu/train --check 8
"""

import os
import glob
import struct
import argparse

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from datasets.dataset_utils import augmentation
from datasets.depth_synth import add_sensor_noise, normalize_depth

# ----------------------------------------------------------------------------
# Intrinsics (depth camera) per dataset. (fx, fy, cx, cy), image (W, H).
# ----------------------------------------------------------------------------
INTR = {
    "msra": dict(fx=241.42, fy=241.42, cx=160.0, cy=120.0, W=320, H=240),
    "icvl": dict(fx=241.42, fy=241.42, cx=160.0, cy=120.0, W=320, H=240),
    "nyu":  dict(fx=588.036865, fy=587.075073, cx=320.0, cy=240.0, W=640, H=480),
    # BigHand2.2M / HANDS17 / FPHA: Intel RealSense SR300 depth, 640x480.
    "bighand": dict(fx=475.065948, fy=475.065857, cx=315.944855, cy=245.287079, W=640, H=480),
    "hands17": dict(fx=475.065948, fy=475.065857, cx=315.944855, cy=245.287079, W=640, H=480),
    "fpha":    dict(fx=475.065948, fy=475.065857, cx=315.944855, cy=245.287079, W=640, H=480),
}

# FPHA hand skeletons are in a magnetic-tracker WORLD frame; this 4x4 maps world
# -> camera (the official load_example.py constant). We then project with the
# SR300 depth intrinsics above (the standard FPHA-depth convention).
FPHA_CAM_EXTR = np.array([
    [0.999988496304, -0.00468848412856, 0.000982563360594, 25.7],
    [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
    [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902],
    [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

# ----------------------------------------------------------------------------
# Joint mappings into the 21 MANO/FreiHAND slots:
#   0 wrist | thumb 1-4 | index 5-8 | middle 9-12 | ring 13-16 | pinky 17-20
#   (each finger MCP->PIP->DIP->TIP).
# ----------------------------------------------------------------------------
# MSRA source order: wrist, index(4), middle(4), ring(4), little(4), thumb(4),
# each finger MCP->TIP. Full 21 -> MANO is a pure permutation (all valid).
MSRA_TO_MANO = np.array([0, 17, 18, 19, 20, 1, 2, 3, 4, 5, 6, 7, 8,
                         9, 10, 11, 12, 13, 14, 15, 16])

# ICVL source order (16): palm, then per finger root/mid/tip for
# thumb,index,middle,ring,pinky. Map root->MCP, mid->DIP, tip->TIP; PIP absent.
#   MANO slot : ICVL idx
ICVL_MAP = {0: 0,
            1: 1, 3: 2, 4: 3,        # thumb  MCP,DIP,TIP
            5: 4, 7: 5, 8: 6,        # index
            9: 7, 11: 8, 12: 9,      # middle
            13: 10, 15: 11, 16: 12,  # ring
            17: 13, 19: 14, 20: 15}  # pinky  (PIP slots 2,6,10,14,18 absent)

# BigHand2.2M / HANDS17 source order (21):
#   [Wrist, TMCP, IMCP, MMCP, RMCP, PMCP,            (wrist + 5 MCPs T,I,M,R,P)
#    TPIP, TDIP, TTIP, IPIP, IDIP, ITIP,             (then PIP,DIP,TIP per finger)
#    MPIP, MDIP, MTIP, RPIP, RDIP, RTIP, PPIP, PDIP, PTIP]
# -> MANO (wrist, then thumb..pinky each MCP,PIP,DIP,TIP). Pure permutation, all valid.
#   BIGHAND_TO_MANO[mano_slot] = bighand_index
BIGHAND_TO_MANO = np.array([0,
                            1, 6, 7, 8,        # thumb  MCP,PIP,DIP,TIP
                            2, 9, 10, 11,      # index
                            3, 12, 13, 14,     # middle
                            4, 15, 16, 17,     # ring
                            5, 18, 19, 20])    # pinky

# NYU 36-joint set -> we trust only the 5 fingertips + wrist (unambiguous across
# skeletons). These default indices follow the common NYU layout; if the viz
# self-check shows a mismatch, edit them (they are the only NYU-specific guess).
#   MANO slot : NYU 36-idx        (pinky..thumb tips, then wrist/palm)
NYU_MAP = {20: 0, 16: 6, 12: 12, 8: 18, 4: 24, 0: 32}


def backproject(uvz, fx, fy, cx, cy):
    """(u px, v px, z mm) -> camera xyz in metres (z forward)."""
    u, v, z = uvz[:, 0], uvz[:, 1], uvz[:, 2]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=1) / 1000.0


def project(xyz_m, fx, fy, cx, cy):
    """camera xyz (m) -> (u, v) px."""
    z = xyz_m[:, 2]
    return np.stack([xyz_m[:, 0] / z * fx + cx, xyz_m[:, 1] / z * fy + cy], axis=1)


class _ToFDepthBase(Dataset):
    """Shared RGB-D plumbing for depth-only datasets. Subclasses implement
    `_load(idx) -> (depth_full_m[H,W], joints_mano[21,3] cam metres, valid[21])`,
    with absent joints free to be any value (they are masked out)."""

    name = "tof"

    def __init__(self, root, mode="train", img_size=256, with_depth=True,
                 depth_mode="hand_masked", depth_cfg=None, hand_band_m=0.08, limit=0):
        self.root = root
        self.mode = mode
        self.img_size = img_size
        self.with_depth = with_depth
        self.depth_mode = depth_mode
        self.hand_band_m = hand_band_m
        self.depth_cfg = {**self._default_depth_cfg(), **(depth_cfg or {})}
        self.intr = INTR[self.name]
        self.samples = self._build_index(limit)
        print(f"[{self.name.upper()}_RGBD] {len(self.samples)} frames mode={mode} "
              f"depth_mode={depth_mode}")

    @staticmethod
    def _default_depth_cfg():
        # Real ToF depth: add a touch of ToF-like dropout in train mode so a model
        # later sees Femto-Bolt-style holes; tune as needed.
        return dict(sigma=0.003, dropout_p=0.03, quant=0.001, n_holes=1,
                    hole_frac=0.1, norm_scale=0.1)

    # subclasses override
    def _build_index(self, limit):
        raise NotImplementedError

    def _load(self, idx):
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)

    def _pseudo_rgb(self, depth_full_m):
        """Depth -> 3-channel uint8 image (per-frame min-max over valid pixels)."""
        v = depth_full_m > 0
        img = np.zeros(depth_full_m.shape, np.float32)
        if v.any():
            lo, hi = np.percentile(depth_full_m[v], [2, 98])
            img[v] = np.clip((depth_full_m[v] - lo) / max(hi - lo, 1e-6), 0, 1)
        g = (img * 255).astype(np.uint8)
        return np.repeat(g[:, :, None], 3, axis=2)

    def _depth_channel(self, depth_full_m, joints3d, img2bb_trans, idx):
        dc = self.depth_cfg
        if self.depth_mode == "hand_masked":
            z = joints3d[:, 2]
            zlo, zhi = float(z.min()) - self.hand_band_m, float(z.max()) + self.hand_band_m
            depth_full_m = np.where((depth_full_m >= zlo) & (depth_full_m <= zhi),
                                    depth_full_m, 0.0).astype(np.float32)
        depth_crop = cv2.warpAffine(depth_full_m, img2bb_trans, (256, 256),
                                    flags=cv2.INTER_NEAREST)
        if self.mode == "train":
            depth_crop = add_sensor_noise(depth_crop, sigma=dc["sigma"],
                dropout_p=dc["dropout_p"], quant=dc["quant"], n_holes=dc["n_holes"],
                hole_frac=dc["hole_frac"])
        else:
            depth_crop = add_sensor_noise(depth_crop, sigma=0.0, dropout_p=0.0,
                quant=dc["quant"], n_holes=0, rng=np.random.default_rng(int(idx)))
        depth_norm = normalize_depth(depth_crop, scale=dc["norm_scale"])
        return torch.from_numpy(depth_norm).float().unsqueeze(0)

    def __getitem__(self, idx):
        out = self._load(idx)
        if out is None:
            return self.__getitem__((idx + 1) % len(self.samples))
        depth_full_m, joints3d, valid = out
        it = self.intr
        K = np.array([[it["fx"], 0, it["cx"]], [0, it["fy"], it["cy"]], [0, 0, 1]], np.float32)

        image = self._pseudo_rgb(depth_full_m)
        joints2d = project(joints3d, it["fx"], it["fy"], it["cx"], it["cy"])

        vj = joints2d[valid > 0]
        margin = 20.0
        mx = int(np.random.rand() * margin); my = int(np.random.rand() * margin)
        x_min = max(vj[:, 0].min() - mx, 0.0); y_min = max(vj[:, 1].min() - my, 0.0)
        x_max = min(vj[:, 0].max() + my, float(image.shape[1]))
        y_max = min(vj[:, 1].max() + mx, float(image.shape[0]))
        seg_len = max(x_max - x_min, y_max - y_min)
        bbox = [x_min, y_min, seg_len, seg_len]

        aug_img, img2bb_trans, bb2img_trans, rot, _, cam, cam_nh, _ = augmentation(
            image, bbox, self.mode, exclude_flip=True, rotation=True, cam_param=K)

        rot = -rot * np.pi / 180.0
        rot_mat = np.array([[np.cos(rot), -np.sin(rot), 0],
                            [np.sin(rot), np.cos(rot), 0],
                            [0, 0, 1]], dtype=np.float32)
        rot_joints = rot_mat.dot(joints3d.T).T
        root_xyz = rot_joints[0].copy()           # MANO slot 0 = wrist (always set)
        align_joints = rot_joints - root_xyz

        ori_2d = np.concatenate((joints2d, np.ones((21, 1))), axis=1)
        kps = (img2bb_trans @ ori_2d.T).T[:, :2] / 256.0

        return_img = torch.from_numpy(
            np.ascontiguousarray(aug_img.astype(np.uint8))).permute(2, 0, 1).float() / 255.0
        if self.with_depth:
            return_img = torch.cat(
                [return_img, self._depth_channel(depth_full_m, joints3d, img2bb_trans, idx)], 0)

        new_cam = cam_nh.copy(); new_cam[:, 2] = cam[:, 2]
        return {
            "image": return_img,
            "keypoints3D": align_joints.astype(np.float32),
            "keypoints2D": kps.astype(np.float32),
            "root": root_xyz.astype(np.float32),
            "cam": new_cam.astype(np.float32),
            "joint_valid": valid.astype(np.float32),
        }


# ============================================================ MSRA (ToF, 21) ==
class MSRA_RGBD(_ToFDepthBase):
    """cvpr15_MSRAHandGestureDB: P0..P8 / <gesture> / {NNNNNN_depth.bin, joint.txt}.

    Cleanest ToF source: 21 joints -> MANO via permutation, all valid. Joints are
    camera mm; MSRA's y axis points up, so we flip y to the OpenCV convention.
    """
    name = "msra"

    def __init__(self, *a, y_sign=-1.0, **k):
        self.y_sign = y_sign
        super().__init__(*a, **k)

    def _build_index(self, limit):
        samples = []
        for subj in sorted(d for d in os.listdir(self.root) if d.startswith("P")):
            sp = os.path.join(self.root, subj)
            for gesture in sorted(os.listdir(sp)):
                gp = os.path.join(sp, gesture)
                jt = os.path.join(gp, "joint.txt")
                if not os.path.isfile(jt):
                    continue
                with open(jt) as f:
                    n = int(f.readline())
                    joints = np.array([list(map(float, f.readline().split())) for _ in range(n)])
                joints = joints.reshape(n, 21, 3)
                for i in range(n):
                    samples.append((os.path.join(gp, f"{i:06d}_depth.bin"), joints[i]))
                    if limit and len(samples) >= limit:
                        return samples
        return samples

    def _read_bin(self, fp):
        with open(fp, "rb") as f:
            w, h, l, t, r, b = struct.unpack("6i", f.read(24))
            crop = np.frombuffer(f.read(), np.float32).reshape(b - t, r - l)
        full = np.zeros((h, w), np.float32)
        full[t:b, l:r] = crop
        return full / 1000.0  # mm -> m

    def _load(self, idx):
        fp, j = self.samples[idx]
        if not os.path.isfile(fp):
            return None
        depth = self._read_bin(fp)
        j = j.copy(); j[:, 1] *= self.y_sign; j[:, 2] = np.abs(j[:, 2])
        joints = (j[MSRA_TO_MANO] / 1000.0).astype(np.float32)
        return depth, joints, np.ones(21, np.float32)


# ============================================================ ICVL (ToF, 16) ==
class ICVL_RGBD(_ToFDepthBase):
    """ICVL: Depth/ 16-bit-mm PNGs + a labels file (image_path u v z ...x16).

    16 joints (3/finger) -> 16 MANO slots; the 5 PIP slots are invalid.
    """
    name = "icvl"

    def __init__(self, *a, label_file="labels.txt", depth_subdir="Depth", **k):
        self.label_file = label_file
        self.depth_subdir = depth_subdir
        super().__init__(*a, **k)

    def _build_index(self, limit):
        samples = []
        lf = self.label_file if os.path.isabs(self.label_file) else \
            os.path.join(self.root, self.label_file)
        with open(lf) as f:
            for line in f:
                p = line.split()
                if len(p) < 1 + 16 * 3:
                    continue
                vals = np.array(p[1:1 + 48], np.float32).reshape(16, 3)
                samples.append((p[0], vals))
                if limit and len(samples) >= limit:
                    break
        return samples

    def _load(self, idx):
        rel, vals = self.samples[idx]
        fp = os.path.join(self.root, self.depth_subdir, rel)
        if not os.path.isfile(fp):
            fp = os.path.join(self.root, rel)
        d = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        if d is None:
            return None
        depth = d.astype(np.float32) / 1000.0
        it = self.intr
        cam16 = backproject(vals, it["fx"], it["fy"], it["cx"], it["cy"])  # (16,3) m
        joints = np.zeros((21, 3), np.float32)
        valid = np.zeros(21, np.float32)
        for mano_slot, icvl_idx in ICVL_MAP.items():
            joints[mano_slot] = cam16[icvl_idx]; valid[mano_slot] = 1.0
        # ensure wrist (slot 0) present for root alignment
        return depth, joints, valid


# ====================================================== NYU (structured, 14) ==
class NYU_RGBD(_ToFDepthBase):
    """NYU: depth_1_*.png (depth = G*256 + B) + joint_data.mat (joint_uvd 36).

    NOT ToF (structured-light) -- included for completeness. Only fingertips +
    wrist are mapped (6 valid MANO slots); requires scipy to read the .mat.
    """
    name = "nyu"

    def __init__(self, *a, kinect=1, **k):
        self.kinect = kinect
        super().__init__(*a, **k)

    def _build_index(self, limit):
        try:
            from scipy.io import loadmat
        except ImportError as e:
            raise ImportError("NYU needs scipy: pip install scipy") from e
        mat = loadmat(os.path.join(self.root, "joint_data.mat"))
        self.uvd = mat["joint_uvd"][self.kinect - 1]  # (N, 36, 3)
        n = self.uvd.shape[0]
        idxs = list(range(n))
        if limit:
            idxs = idxs[:limit]
        return idxs

    def _load(self, idx):
        i = self.samples[idx]
        fp = os.path.join(self.root, f"depth_{self.kinect}_{i+1:07d}.png")
        d = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        if d is None:
            return None
        depth = (d[:, :, 1].astype(np.float32) * 256.0 + d[:, :, 0].astype(np.float32)) / 1000.0
        it = self.intr
        cam36 = backproject(self.uvd[i], it["fx"], it["fy"], it["cx"], it["cy"])
        joints = np.zeros((21, 3), np.float32)
        valid = np.zeros(21, np.float32)
        for mano_slot, nyu_idx in NYU_MAP.items():
            joints[mano_slot] = cam36[nyu_idx]; valid[mano_slot] = 1.0
        return depth, joints, valid


# ============================================= BigHand2.2M / HANDS17 (21) ==
class _BigHandAnno(_ToFDepthBase):
    """Shared loader for BigHand2.2M and HANDS17 (same SR300 format).

    Annotation: one line per frame = `<image_path> x1 y1 z1 ... x21 y21 z21`
    (camera-space millimetres, BigHand/HANDS17 joint order). Depth: 16-bit
    single-channel PNG in mm. Joints map to MANO via BIGHAND_TO_MANO (all valid).
    Note: SR300 is structured/coded-light (like NYU), not strictly ToF, but it is
    the canonical large-scale real-sensor depth set -- excellent for a depth-branch
    pre-train. Full BigHand is 2.2M lines (~1 GB joints in RAM); use `limit` to cap.
    """

    def __init__(self, root, anno_file=None, image_dir=None, **k):
        self._anno_file = anno_file
        self._image_dir = image_dir
        super().__init__(root, **k)

    def _default_anno(self):
        return os.path.join(self.root, "Training_Annotation.txt")

    def _default_image_dir(self):
        return self.root

    def _build_index(self, limit):
        anno = self._anno_file or self._default_anno()
        self.image_dir = self._image_dir or self._default_image_dir()
        names, rows = [], []
        with open(anno) as f:
            for line in f:
                p = line.split()
                if len(p) < 1 + 21 * 3:
                    continue
                names.append(p[0].replace("\\", os.sep))
                rows.append([float(x) for x in p[1:1 + 63]])
                if limit and len(names) >= limit:
                    break
        self._joints = np.asarray(rows, np.float32).reshape(-1, 21, 3)
        return names

    def _load(self, idx):
        fname = self.samples[idx]
        fp = os.path.join(self.image_dir, fname)
        if not os.path.isfile(fp):
            fp = os.path.join(self.root, fname)
        d = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        if d is None:
            return None
        if d.ndim == 3:                         # just in case of a 3-ch export
            d = d[:, :, 0]
        depth = d.astype(np.float32) / 1000.0   # mm -> m
        joints = (self._joints[idx][BIGHAND_TO_MANO] / 1000.0).astype(np.float32)
        return depth, joints, np.ones(21, np.float32)


class BigHand_RGBD(_BigHandAnno):
    """BigHand2.2M. Default anno `<root>/Training_Annotation.txt`, images under root."""
    name = "bighand"


class HANDS17_RGBD(_BigHandAnno):
    """HANDS17 frame-based. Defaults to `<root>/training/{Training_Annotation.txt,images/}`."""
    name = "hands17"

    def _default_anno(self):
        return os.path.join(self.root, "training", "Training_Annotation.txt")

    def _default_image_dir(self):
        return os.path.join(self.root, "training", "images")


# ===================================================== FPHA (SR300, 21) ======
class FPHA_RGBD(_ToFDepthBase):
    """First-Person Hand Action Benchmark (egocentric SR300 RGB-D, 105k frames).

    Layout:
        <root>/Video_files/<Subj>/<action>/<rep>/depth/depth_{f:04d}.png
        <root>/<anno_dir>/<Subj>/<action>/<rep>/skeleton.txt
    skeleton.txt: one line per frame = `frame_idx x1 y1 z1 ... x21 y21 z21`, world
    coords (mm), BigHand 21-joint order -> BIGHAND_TO_MANO (all valid).

    Coords: skeleton is WORLD -> FPHA_CAM_EXTR gives the camera frame; we project
    with the SR300 *depth* intrinsics (INTR['fpha']). The official load_example.py
    projects to the 1080p *colour* frame; colour and depth are separate SR300
    cameras, so depth-frame projection carries a small (~tens of px worst-case)
    colour/depth baseline offset. The hand-bbox margin + z-band depth mask absorb
    it, but RUN THE --check VIZ once to confirm joints land on the hand; if not,
    override FPHA_CAM_EXTR or switch to the colour stream.
    """
    name = "fpha"

    def __init__(self, root, anno_dir="Hand_pose_annotation_v1_1", **k):
        self.anno_dir = anno_dir
        super().__init__(root, **k)

    def _build_index(self, limit):
        vid_root = os.path.join(self.root, "Video_files")
        anno_root = os.path.join(self.root, self.anno_dir)
        if not os.path.isdir(anno_root):  # tolerate the alternate folder name
            alt = os.path.join(self.root, "Hand_pose_annotation_v1")
            anno_root = alt if os.path.isdir(alt) else anno_root
        samples, self._rows = [], []
        for skel_fp in sorted(glob.glob(os.path.join(anno_root, "*", "*", "*", "skeleton.txt"))):
            rel = os.path.relpath(os.path.dirname(skel_fp), anno_root)  # Subj/action/rep
            depth_dir = os.path.join(vid_root, rel, "depth")
            if not os.path.isdir(depth_dir):
                continue
            arr = np.loadtxt(skel_fp).reshape(-1, 64)
            for r in arr:
                fidx = int(r[0])
                samples.append((os.path.join(depth_dir, f"depth_{fidx:04d}.png"), len(self._rows)))
                self._rows.append(r[1:64].astype(np.float32))
                if limit and len(samples) >= limit:
                    self._rows = np.asarray(self._rows, np.float32).reshape(-1, 21, 3)
                    return samples
        self._rows = np.asarray(self._rows, np.float32).reshape(-1, 21, 3)
        return samples

    def _load(self, idx):
        fp, row = self.samples[idx]
        d = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        if d is None:
            return None
        if d.ndim == 3:
            d = d[:, :, 0]
        depth = d.astype(np.float32) / 1000.0
        world = self._rows[row]                              # (21,3) world mm
        R, t = FPHA_CAM_EXTR[:3, :3], FPHA_CAM_EXTR[:3, 3]
        cam = (R @ world.T).T + t                            # camera mm
        joints = (cam[BIGHAND_TO_MANO] / 1000.0).astype(np.float32)
        return depth, joints, np.ones(21, np.float32)


_REG = {"msra": MSRA_RGBD, "icvl": ICVL_RGBD, "nyu": NYU_RGBD,
        "bighand": BigHand_RGBD, "hands17": HANDS17_RGBD, "fpha": FPHA_RGBD}


# python -m datasets.tof_depth --dataset msra --root <path> --check 8
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(_REG))
    ap.add_argument("--root", required=True)
    ap.add_argument("--mode", default="train")
    ap.add_argument("--depth_mode", default="hand_masked", choices=["hand_masked", "scene"])
    ap.add_argument("--check", default=8, type=int)
    args = ap.parse_args()

    ds = _REG[args.dataset](root=args.root, mode=args.mode, depth_mode=args.depth_mode, limit=3000)
    print("len:", len(ds))
    od = f"{args.dataset}_check"; os.makedirs(od, exist_ok=True)
    step = max(1, len(ds) // max(args.check, 1))
    for i in range(0, min(len(ds), step * args.check), step):
        d = ds[i]
        img = d["image"]
        rgb = (img[:3].permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
        kp = (d["keypoints2D"] * 256).astype(np.int32)
        val = d["joint_valid"]
        for k, (x, y) in enumerate(kp):
            if val[k] > 0:
                cv2.circle(rgb, (int(x), int(y)), 2, (0, 255, 0), -1)
        out = rgb
        if img.shape[0] == 4:
            dvis = ((img[3].clamp(-1, 1).numpy() + 1) / 2 * 255).astype(np.uint8)
            out = np.concatenate([rgb, cv2.applyColorMap(dvis, cv2.COLORMAP_JET)], axis=1)
        cv2.imwrite(f"{od}/sample_{i:05d}.png", out)
        print(f"  {i}: img{tuple(img.shape)} valid={int(val.sum())}/21 z_root={d['root'][2]:.3f}m")
    print(f"wrote {od}/*.png")
