# Copyright (c) 2026. Depth-Aware Hand Pose Initiative.
"""
Real-time RGB-D hand-pose estimation from a Femto Bolt (Orbbec) camera.

This is the *deployment* counterpart of `train_mobrecon_rgbd.py` / `eval_rgbd.py`:
it feeds a live **RGB + metric depth** stream through the early-fusion 4-channel
`LargeModel_Extra_RGBD` and recovers camera-space 3D hand joints.

Pipeline (per frame)
--------------------
  Femto Bolt ──► RGB (BGR)            ──► MediaPipe Hands ─► 21 px landmarks ─► square bbox
            └──► depth (mm, ALIGNED to colour, metres after scale)
                                                                │
        augmentation(bbox, 'test')  ── img2bb_trans (256x256 affine) ───────┤
                                                                │           │
   RGB  ─warpAffine(LINEAR)─► [3,256,256] /255 ─────────────────┐           │
   depth─warpAffine(NEAREST)─► normalize_depth ─► [1,256,256] ──┴─ concat ─► [1,4,256,256]
                                                                            │
                                          LargeModel_Extra_RGBD ─► 21x3 root-relative joints (m)
                                                                            │
   absolute root  = backproject(wrist_px, sensor_depth@wrist, K)           │
   absolute joints = root + root-relative joints  ◄───────────────────────┘

The preprocessing is byte-for-byte the same path the trainer used (it reuses the
project's `augmentation()` and `normalize_depth`), so a checkpoint trained by
`train_mobrecon_rgbd.py` runs here without any conversion. The only difference
from training is the depth *source*: a real sensor instead of the synthetic
capsule renderer (the documented upgrade path #2 in RGBD_TRAINING.md), and the
bbox source: MediaPipe at runtime instead of GT joints offline.

Sources (--source)
------------------
  femtobolt : live Orbbec Femto Bolt via pyorbbecsdk (needs the SDK + hardware)
  folder    : paired rgb_*.png / depth_*.png(uint16 mm) in --path (offline test)
  webcam    : plain RGB webcam, depth channel zeroed (== --rgb_only ablation)
  selftest  : no camera; pushes a synthetic 4-ch tensor to verify model wiring

Examples
--------
  # live camera, GUI overlay
  python infer_rgbd_femtobolt.py --ckpt mobrecon_ckpt/rgbd_1hr/3.pt --source femtobolt

  # verify the model + preprocessing wire up, no hardware needed
  python infer_rgbd_femtobolt.py --ckpt mobrecon_ckpt/rgbd_1hr/3.pt --source selftest

  # replay a recorded folder, save overlays headless
  python infer_rgbd_femtobolt.py --ckpt mobrecon_ckpt/rgbd_1hr/3.pt \
         --source folder --path captures/ --no-gui --save out/femto_eval
"""

import os
import sys
import time
import glob
import argparse

import numpy as np
import cv2
import torch

from datasets.dataset_utils import augmentation
from datasets.depth_synth import normalize_depth
from models.mobrecon_ds import LargeModel_Extra_RGBD
from utils import load_cfg

# Hand skeleton (FreiHAND / MANO 21-keypoint order; matches the model output and
# MediaPipe's landmark order: wrist, thumb, index, middle, ring, pinky).
HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index
    (0, 9), (9, 10), (10, 11), (11, 12),    # middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
]
_FINGER_COLORS = [  # BGR per finger (thumb..pinky)
    (60, 60, 220), (60, 180, 60), (220, 160, 40), (200, 60, 200), (40, 200, 220),
]


# ===========================================================================
# 1. Camera / frame sources  -> each yields (rgb_bgr uint8 HxWx3, depth_m float HxW, K 3x3)
#    depth_m is camera-frame metres, ALIGNED to the colour image (0 = invalid).
# ===========================================================================
class FemtoBoltSource:
    """Live Orbbec Femto Bolt via pyorbbecsdk, depth hardware-aligned to colour.

    Install from https://github.com/orbbec/pyorbbecsdk (Femto Bolt needs Orbbec
    SDK v2). Prebuilt Linux wheels are in the GitHub *Releases* (the PyPI
    `pyorbbecsdk` is a broken macOS-only build):
        pip install <pyorbbecsdk-2.0.15-cp310-cp310-linux_x86_64.whl from Releases>

    The stream setup mirrors the SDK's own `examples/hw_d2c_align.py`: it pairs a
    hardware depth->colour profile with the colour profile via
    `get_d2c_depth_profile_list`, so depth[u,v] lands on colour[u,v] and both
    share the colour intrinsics returned by `get_camera_param().rgb_intrinsic`.
    """

    def __init__(self, color_size=(1280, 720), fps=30):
        try:
            import pyorbbecsdk as ob
        except ImportError as e:
            raise SystemExit(
                "[femtobolt] pyorbbecsdk not found. Install a Linux wheel from "
                "https://github.com/orbbec/pyorbbecsdk/releases, or use "
                "--source folder/webcam/selftest.\n  underlying error: %s" % e
            )
        self._ob = ob
        try:
            self.pipeline = ob.Pipeline()
        except Exception as e:
            raise SystemExit(
                "[femtobolt] no Orbbec device found (%s).\n"
                "  - check the USB3 connection / power,\n"
                "  - install the udev rules for non-root access "
                "(scripts/install_udev_rules.sh in the pyorbbecsdk repo),\n"
                "  - or use --source folder/webcam/selftest." % e)
        config = self._build_d2c_config(color_size, fps)
        self.pipeline.start(config)

        # After HW D2C alignment depth is in the colour frame -> use rgb_intrinsic.
        intr = self.pipeline.get_camera_param().rgb_intrinsic
        self.K = np.array([[intr.fx, 0, intr.cx],
                           [0, intr.fy, intr.cy],
                           [0, 0, 1.0]], dtype=np.float32)
        print(f"[femtobolt] started. colour K=\n{self.K}")

    def _build_d2c_config(self, color_size, fps):
        ob = self._ob
        config = ob.Config()
        prof_list = self.pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)

        # Prefer a colour profile at the requested size; fall back to any colour
        # profile that has a hardware D2C-aligned depth companion.
        def _try(color_profile):
            d2c = self.pipeline.get_d2c_depth_profile_list(color_profile, ob.OBAlignMode.HW_MODE)
            return d2c[0] if len(d2c) else None

        chosen_color, chosen_depth = None, None
        for i in range(len(prof_list)):
            cp = prof_list[i]
            depth_p = _try(cp)
            if depth_p is None:
                continue
            if (cp.get_width(), cp.get_height()) == tuple(color_size):
                chosen_color, chosen_depth = cp, depth_p
                break
            if chosen_color is None:                     # first viable as fallback
                chosen_color, chosen_depth = cp, depth_p
        if chosen_color is None:
            raise SystemExit("[femtobolt] no hardware D2C-alignable colour profile found")

        config.enable_stream(chosen_depth)
        config.enable_stream(chosen_color)
        config.set_align_mode(ob.OBAlignMode.HW_MODE)
        print(f"[femtobolt] colour {chosen_color.get_width()}x{chosen_color.get_height()} "
              f"{chosen_color.get_format()} + HW-D2C depth "
              f"{chosen_depth.get_width()}x{chosen_depth.get_height()}")
        return config

    def _frame_to_bgr(self, frame):
        """Decode any Orbbec colour format to a BGR uint8 image (cf. examples/utils.py)."""
        ob = self._ob
        w, h = frame.get_width(), frame.get_height()
        fmt = frame.get_format()
        data = np.frombuffer(frame.get_data(), dtype=np.uint8)
        if fmt == ob.OBFormat.RGB:
            return cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
        if fmt == ob.OBFormat.BGR:
            return data.reshape(h, w, 3)
        if fmt == ob.OBFormat.MJPG:
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        if fmt == ob.OBFormat.YUYV:
            return cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUYV)
        if fmt == ob.OBFormat.UYVY:
            return cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2BGR_UYVY)
        if fmt == ob.OBFormat.NV12:
            return cv2.cvtColor(data.reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_NV12)
        if fmt == ob.OBFormat.NV21:
            return cv2.cvtColor(data.reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_NV21)
        if fmt == ob.OBFormat.I420:
            return cv2.cvtColor(data.reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_I420)
        raise RuntimeError(f"[femtobolt] unhandled colour format {fmt}")

    def read(self):
        frames = self.pipeline.wait_for_frames(200)
        if frames is None:
            return None
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if color_frame is None or depth_frame is None:
            return None

        bgr = self._frame_to_bgr(color_frame)
        if bgr is None:
            return None
        ch, cw = bgr.shape[:2]

        dw, dh = depth_frame.get_width(), depth_frame.get_height()
        depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(dh, dw)
        # get_depth_scale() converts raw units -> millimetres; /1000 -> metres.
        depth_m = depth_raw.astype(np.float32) * depth_frame.get_depth_scale() / 1000.0
        if (dw, dh) != (cw, ch):                     # safety: match colour grid
            depth_m = cv2.resize(depth_m, (cw, ch), interpolation=cv2.INTER_NEAREST)
        return bgr, depth_m, self.K

    def close(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass


class FolderSource:
    """Replay paired captures: rgb_<id>.png + depth_<id>.png (uint16 mm)."""

    def __init__(self, path, K=None):
        self.rgbs = sorted(glob.glob(os.path.join(path, "rgb_*.png")) +
                           glob.glob(os.path.join(path, "rgb_*.jpg")))
        if not self.rgbs:
            raise SystemExit(f"[folder] no rgb_*.png/jpg in {path}")
        self.i = 0
        # Default intrinsics: Femto Bolt 1280x720 colour (approx). Override with a
        # K.npy in the folder if present.
        kpath = os.path.join(path, "K.npy")
        if K is not None:
            self.K = np.asarray(K, np.float32)
        elif os.path.isfile(kpath):
            self.K = np.load(kpath).astype(np.float32)
        else:
            self.K = np.array([[600.0, 0, 640.0],
                               [0, 600.0, 360.0],
                               [0, 0, 1.0]], np.float32)
        print(f"[folder] {len(self.rgbs)} frames, K=\n{self.K}")

    def read(self):
        if self.i >= len(self.rgbs):
            return None
        rgb_path = self.rgbs[self.i]
        self.i += 1
        bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        dpath = rgb_path.replace("rgb_", "depth_").rsplit(".", 1)[0] + ".png"
        if os.path.isfile(dpath):
            depth_mm = cv2.imread(dpath, cv2.IMREAD_UNCHANGED).astype(np.float32)
            if depth_mm.shape[:2] != bgr.shape[:2]:
                depth_mm = cv2.resize(depth_mm, (bgr.shape[1], bgr.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
            depth_m = depth_mm / 1000.0
        else:
            depth_m = np.zeros(bgr.shape[:2], np.float32)
        return bgr, depth_m, self.K

    def close(self):
        pass


class WebcamSource:
    """Plain RGB webcam; depth zeroed (effectively the --rgb_only ablation)."""

    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise SystemExit(f"[webcam] cannot open camera {index}")
        w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640.0
        h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480.0
        # Rough pinhole guess (fx ~ width). Fine for the RGB-only smoke test.
        self.K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1.0]], np.float32)
        print("[webcam] depth channel will be ZERO (rgb_only behaviour)")

    def read(self):
        ok, bgr = self.cap.read()
        if not ok:
            return None
        return bgr, np.zeros(bgr.shape[:2], np.float32), self.K

    def close(self):
        self.cap.release()


# ===========================================================================
# 2. Hand detector (MediaPipe) -> 21 pixel landmarks + square bbox
# ===========================================================================
_MP_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                 "hand_landmarker/float16/1/hand_landmarker.task")
_MP_MODEL_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "mediapipe",
                               "hand_landmarker.task")


def _ensure_hand_model(path):
    """Return a local path to hand_landmarker.task, downloading it once if needed."""
    if path and os.path.isfile(path):
        return path
    path = path or _MP_MODEL_CACHE
    if os.path.isfile(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[mediapipe] downloading hand_landmarker.task -> {path}")
    try:
        import urllib.request
        urllib.request.urlretrieve(_MP_MODEL_URL, path)
    except Exception as e:
        raise SystemExit(
            f"[mediapipe] could not fetch the hand-landmarker model.\n"
            f"  download it manually from {_MP_MODEL_URL}\n"
            f"  and pass --hand_model <path>.  error: {e}")
    return path


class HandDetector:
    """MediaPipe Tasks HandLandmarker (this mediapipe build ships only the Tasks
    API, not the legacy `mp.solutions`)."""

    def __init__(self, model_path=None, max_hands=1, det_conf=0.5):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        self._mp = mp
        model_path = _ensure_hand_model(model_path)
        opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=det_conf,
            min_tracking_confidence=det_conf)
        self.landmarker = vision.HandLandmarker.create_from_options(opts)

    def detect(self, bgr):
        """Return landmarks_px (21,2) float in image pixels, or None."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        res = self.landmarker.detect(mp_image)
        if not res.hand_landmarks:
            return None
        h, w = bgr.shape[:2]
        lm = res.hand_landmarks[0]
        pts = np.array([[p.x * w, p.y * h] for p in lm], np.float32)  # (21,2)
        return pts

    def close(self):
        self.landmarker.close()


def landmarks_to_bbox(pts, img_hw, margin_frac=0.50):
    """Square bbox [x, y, side, side] around landmarks, clamped to the image.

    Square (longer side) box centred on the hand, expanded by `margin_frac` of the
    side on each edge. The default 0.50 was chosen empirically: it makes the
    hand-to-crop ratio match the model's TRAINING distribution. Training cropped a
    tight GT-joint box (+0-20 px) on 224-px images, i.e. the hand filled most of
    the crop; with MediaPipe landmarks a 0.50 margin reproduces that ratio. A
    margin sweep over 4 HanCo scenes x 2 cams put root-relative MPJPE at 16.5 mm
    @0.20 vs 14.3 mm @0.50-0.55 (flat optimum), ~the GT-bbox eval (13.8 mm). Note
    matching the training *anchoring* (top-left square) instead made it worse,
    because MediaPipe's hand extent != the GT joint extent.
    """
    h, w = img_hw
    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
    side = max(x_max - x_min, y_max - y_min)
    m = margin_frac * side
    side = side + 2 * m
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    x0 = cx - side / 2.0
    y0 = cy - side / 2.0
    return [float(x0), float(y0), float(side), float(side)]


# ===========================================================================
# 3. RGB-D estimator: crop -> 4ch -> model -> root recovery
# ===========================================================================
class RGBDHandPose:
    def __init__(self, ckpt, cfg_path="./configs_rgbd.yaml", device=None,
                 norm_scale=0.1, rgb_only=False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = load_cfg(cfg_path)
        self.norm_scale = norm_scale
        self.rgb_only = rgb_only
        self.model = LargeModel_Extra_RGBD(self.cfg).to(self.device).eval()
        state = torch.load(ckpt, map_location=self.device)
        sd = state.get("model_state_dict", state) if isinstance(state, dict) else state
        self.model.load_state_dict(sd)
        print(f"[model] loaded {ckpt} (epoch {state.get('epoch', '?')}) "
              f"in_chans={self.model.in_chans} device={self.device} rgb_only={rgb_only}")

    def preprocess(self, bgr, depth_m, bbox, K):
        """bbox -> [1,4,256,256] tensor (+ the img2bb_trans used, for debug)."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Reuse the exact training crop path (test mode: scale=1, rot=0).
        aug_img, img2bb_trans, _, _, _, _, _, _ = augmentation(
            rgb.astype(np.float32), bbox, "test", exclude_flip=True,
            rotation=True, cam_param=K)
        rgb_crop = np.clip(aug_img, 0, 255).astype(np.uint8)            # (256,256,3) RGB
        rgb_t = torch.from_numpy(rgb_crop).float().permute(2, 0, 1) / 255.0  # [3,256,256]

        if self.rgb_only:
            depth_norm = np.zeros((256, 256), np.float32)
        else:
            # Warp the *real sensor* depth with the SAME affine as the RGB, then
            # median-centre exactly like training (no synthetic noise at test).
            depth_crop = cv2.warpAffine(depth_m, img2bb_trans, (256, 256),
                                        flags=cv2.INTER_NEAREST)
            depth_norm = normalize_depth(depth_crop, scale=self.norm_scale)
        depth_t = torch.from_numpy(depth_norm).float().unsqueeze(0)     # [1,256,256]

        inp = torch.cat([rgb_t, depth_t], dim=0).unsqueeze(0)           # [1,4,256,256]
        return inp.to(self.device), rgb_crop, depth_norm

    @torch.no_grad()
    def infer(self, inp):
        out = self.model(inp)["keypoints"][0]      # (21,3) root-relative metres
        return out.cpu().numpy()

    @staticmethod
    def recover_root(joints_rel, wrist_px, depth_m, K, win=4):
        """Place the root in camera space using the sensor depth at the wrist.

        Returns absolute joints (21,3) metres, or None if no valid wrist depth.
        Output joints are camera-frame (test mode has no crop rotation, so the
        model output is already in the camera frame, merely root-relative).
        """
        u, v = int(round(wrist_px[0])), int(round(wrist_px[1]))
        h, w = depth_m.shape
        if not (0 <= u < w and 0 <= v < h):
            return None
        patch = depth_m[max(0, v - win):v + win + 1, max(0, u - win):u + win + 1]
        valid = patch[patch > 0]
        if valid.size == 0:
            return None
        z0 = float(np.median(valid))
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        root = np.array([(u - cx) / fx * z0, (v - cy) / fy * z0, z0], np.float32)
        return joints_rel + root[None, :]

    @staticmethod
    def _sample_depth(depth_m, u, v, win=2):
        h, w = depth_m.shape
        u, v = int(round(u)), int(round(v))
        if not (0 <= u < w and 0 <= v < h):
            return 0.0
        p = depth_m[max(0, v - win):v + win + 1, max(0, u - win):u + win + 1]
        p = p[p > 0]
        return float(np.median(p)) if p.size else 0.0

    @staticmethod
    def recover_root_aligned(joints_rel, mp_uv, depth_m, K, surface_offset=0.010,
                             iters=3, win=2):
        """Robust absolute-root placement (much better than recover_root).

        Two scene-agnostic corrections over the single-wrist-pixel method:
          * Z: sample sensor depth at ALL 21 predicted joint pixels (not just the
            wrist) and take the median of (depth - joint_z), then add a fixed
            surface->joint-centre offset (~one capsule radius) since the sensor
            sees the skin, not the joint inside the hand. This removes most of the
            systematic "hand pulled toward the camera" bias.
          * X,Y: align the projected joints to MediaPipe's full 21-landmark hand
            (median over joints) instead of anchoring on the noisy wrist point.

        Args:
            joints_rel: (21,3) root-relative metres (wrist at origin).
            mp_uv: (21,2) MediaPipe pixel landmarks for the same hand.
            depth_m: (H,W) sensor depth in metres (0 = invalid).
            surface_offset: metres added to root_z to convert skin->joint centre.
        Returns:
            (21,3) absolute camera-space joints, or None if no valid depth.
        """
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        z0 = 0.0
        for i in range(len(mp_uv)):                       # seed from any valid landmark
            z0 = RGBDHandPose._sample_depth(depth_m, mp_uv[i, 0], mp_uv[i, 1], win)
            if z0 > 0:
                break
        if z0 <= 0:
            return None
        root = np.array([0.0, 0.0, z0], np.float64)
        for _ in range(iters):
            zc = joints_rel[:, 2] + root[2]
            root[0] = np.median((mp_uv[:, 0] - cx) / fx * zc - joints_rel[:, 0])
            root[1] = np.median((mp_uv[:, 1] - cy) / fy * zc - joints_rel[:, 1])
            uv = project(joints_rel + root, K)
            ds = [RGBDHandPose._sample_depth(depth_m, uv[i, 0], uv[i, 1], win) - joints_rel[i, 2]
                  for i in range(len(uv))
                  if RGBDHandPose._sample_depth(depth_m, uv[i, 0], uv[i, 1], win) > 0]
            if ds:
                root[2] = float(np.median(ds)) + surface_offset
        return (joints_rel + root).astype(np.float32)


# ===========================================================================
# 4. Visualisation
# ===========================================================================
def project(joints_cam, K):
    z = np.clip(joints_cam[:, 2:3], 1e-6, None)
    uv = joints_cam[:, :2] / z * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
    return uv


def draw_skeleton(bgr, uv, bbox=None, color_by_finger=True):
    out = bgr.copy()
    if bbox is not None:
        x, y, s, _ = bbox
        cv2.rectangle(out, (int(x), int(y)), (int(x + s), int(y + s)), (180, 180, 180), 1)
    for fi, (a, b) in enumerate(HAND_BONES):
        c = _FINGER_COLORS[fi // 4] if color_by_finger else (0, 255, 0)
        pa, pb = uv[a].astype(int), uv[b].astype(int)
        cv2.line(out, tuple(pa), tuple(pb), c, 2)
    for p in uv.astype(int):
        cv2.circle(out, tuple(p), 3, (255, 255, 255), -1)
    return out


def depth_vis(depth_norm):
    """[-1,1] depth crop -> BGR colourmap (background black)."""
    d = ((np.clip(depth_norm, -1, 1) + 1) * 127.5).astype(np.uint8)
    vis = cv2.applyColorMap(d, cv2.COLORMAP_JET)
    vis[depth_norm == 0] = 0
    return vis


# ===========================================================================
# 5. Main loop
# ===========================================================================
def build_source(args):
    if args.source == "femtobolt":
        return FemtoBoltSource()
    if args.source == "folder":
        return FolderSource(args.path)
    if args.source == "webcam":
        return WebcamSource(args.cam_index)
    raise ValueError(args.source)


def run_selftest(estimator):
    """No camera: confirm the model + preprocessing produce sane joints."""
    print("[selftest] synthetic hand at ~0.45 m, 1280x720 frame")
    K = np.array([[600, 0, 640], [0, 600, 360], [0, 0, 1.0]], np.float32)
    bgr = np.full((720, 1280, 3), 30, np.uint8)
    # Fake right hand near image centre + a metric depth blob behind it.
    pts = np.array([[640 + dx, 360 + dy] for dx in (-40, 0, 40) for dy in (-60, 0, 60)],
                   np.float32)
    pts = np.repeat(pts, 3, axis=0)[:21]
    depth_m = np.zeros((720, 1280), np.float32)
    cv2.circle(depth_m, (640, 360), 90, 0.45, -1)
    bbox = landmarks_to_bbox(pts, bgr.shape[:2])
    inp, rgb_crop, dn = estimator.preprocess(bgr, depth_m, bbox, K)
    assert inp.shape == (1, 4, 256, 256), inp.shape
    joints = estimator.infer(inp)
    span = np.linalg.norm(joints.max(0) - joints.min(0)) * 1000
    print(f"[selftest] OK  input={tuple(inp.shape)}  output joints={joints.shape}  "
          f"hand-span={span:.0f} mm  depth-coverage={(dn != 0).mean() * 100:.1f}%")
    abs_j = estimator.recover_root(joints, pts[0], depth_m, K)
    print(f"[selftest] root recovery: {'wrist z=%.3f m' % abs_j[0, 2] if abs_j is not None else 'no depth'}")
    print("[selftest] PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cfg", default="./configs_rgbd.yaml")
    ap.add_argument("--source", default="femtobolt",
                    choices=["femtobolt", "folder", "webcam", "selftest"])
    ap.add_argument("--path", default="captures", help="folder source dir")
    ap.add_argument("--cam_index", default=0, type=int, help="webcam index")
    ap.add_argument("--rgb_only", action="store_true", help="zero the depth channel")
    ap.add_argument("--norm_scale", default=0.1, type=float, help="depth norm scale (m->1)")
    ap.add_argument("--margin", default=0.50, type=float, help="bbox margin fraction")
    ap.add_argument("--hand_model", default=None,
                    help="path to MediaPipe hand_landmarker.task (auto-downloaded if absent)")
    ap.add_argument("--no-gui", dest="gui", action="store_false",
                    help="headless (no cv2.imshow)")
    ap.add_argument("--save", default=None, help="dir to dump overlay PNGs")
    args = ap.parse_args()

    estimator = RGBDHandPose(args.ckpt, args.cfg, rgb_only=args.rgb_only,
                             norm_scale=args.norm_scale)

    if args.source == "selftest":
        run_selftest(estimator)
        return

    src = build_source(args)
    detector = HandDetector(model_path=args.hand_model)
    if args.save:
        os.makedirs(args.save, exist_ok=True)

    t_prev, fps, frame_i = time.time(), 0.0, 0
    try:
        while True:
            frame = src.read()
            if frame is None:
                print("[main] stream ended")
                break
            bgr, depth_m, K = frame
            overlay = bgr

            pts = detector.detect(bgr)
            if pts is not None:
                bbox = landmarks_to_bbox(pts, bgr.shape[:2], margin_frac=args.margin)
                inp, rgb_crop, dn = estimator.preprocess(bgr, depth_m, bbox, K)
                joints_rel = estimator.infer(inp)
                abs_j = estimator.recover_root_aligned(joints_rel, pts, depth_m, K)
                if abs_j is not None:
                    uv = project(abs_j, K)
                    overlay = draw_skeleton(bgr, uv, bbox)
                    z = abs_j[0, 2]
                    cv2.putText(overlay, f"wrist z={z*100:5.1f} cm  depth-cov="
                                f"{(dn != 0).mean()*100:4.1f}%", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    overlay = draw_skeleton(bgr, pts, bbox, color_by_finger=False)
                    cv2.putText(overlay, "no valid wrist depth (RGB bbox only)",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                if args.gui:
                    cv2.imshow("depth(crop)", depth_vis(dn))
            else:
                cv2.putText(overlay, "no hand", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now
            cv2.putText(overlay, f"{fps:4.1f} FPS  [{args.source}]", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if args.save:
                cv2.imwrite(os.path.join(args.save, f"{frame_i:05d}.png"), overlay)
            if args.gui:
                cv2.imshow("RGB-D hand pose (Femto Bolt)", overlay)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            frame_i += 1
    finally:
        src.close()
        detector.close()
        if args.gui:
            cv2.destroyAllWindows()
    print(f"[main] processed {frame_i} frames")


if __name__ == "__main__":
    main()
