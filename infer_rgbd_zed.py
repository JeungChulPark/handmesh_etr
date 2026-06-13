# Copyright (c) 2026. Depth-Aware Hand Pose Initiative.
"""
Real-time RGB-D hand-pose estimation from a Stereolabs ZED camera.

ZED counterpart of `infer_rgbd_femtobolt.py`. The ZED is a *passive stereo* depth
camera, which is the right domain for the model trained on DexYCB/HO3D (also
stereo) — see the project note on the ZED deployment sensor. Everything
downstream of capture (MediaPipe bbox, the training crop path, depth median-
centring, the 4-channel model, root recovery, drawing) is REUSED verbatim from
`infer_rgbd_femtobolt.py`; only the camera front-end is swapped for the ZED SDK.

What's different about ZED depth (handled in ZEDSource.read)
-----------------------------------------------------------
  * depth is **metric float metres already** (coordinate_units=METER) — no /1000.
  * invalid depth is **NaN / +-Inf**, NOT 0 like Orbbec. We map those to 0 so the
    pipeline's "0 = invalid" convention (normalize_depth, root recovery) holds.
  * depth is computed in and **aligned to the LEFT image**, so we use the LEFT
    rectified intrinsics and the LEFT view for RGB — depth[u,v] ↔ left[u,v].
  * passive stereo => more holes in low-texture / low-light than RealSense active
    stereo; raise the model's depth-aug dropout at train time if needed, and the
    NEURAL depth mode helps most at inference (default below).

Sources (--source)
------------------
  zed      : live ZED via the ZED SDK Python API (`pyzed`) + hardware
  svo      : replay a recorded .svo/.svo2 file through the same ZED pipeline
  folder   : paired rgb_*.png / depth_*.png(uint16 mm) in --path (offline)
  webcam   : plain RGB webcam, depth channel zeroed (== --rgb_only ablation)
  selftest : no camera; pushes a synthetic 4-ch tensor to verify model wiring

Examples
--------
  python infer_rgbd_zed.py --ckpt mobrecon_ckpt/rgbd_real_0613/best.pt --source zed
  python infer_rgbd_zed.py --ckpt <ckpt> --source svo --svo rec.svo2 --no-gui --save out/
  python infer_rgbd_zed.py --ckpt <ckpt> --source selftest        # no hardware
"""

import os
import time
import argparse

import numpy as np
import cv2

# Reuse the entire Femto Bolt pipeline; only the capture source changes.
from infer_rgbd_femtobolt import (
    RGBDHandPose, HandDetector, landmarks_to_bbox, project, draw_skeleton,
    depth_vis, run_selftest, FolderSource, WebcamSource,
)


# ===========================================================================
# ZED camera source  -> (bgr uint8 HxWx3, depth_m float HxW metres, K 3x3)
#   depth_m is LEFT-frame metres, aligned to the LEFT colour image (0 = invalid).
# ===========================================================================
class ZEDSource:
    """Live ZED (or .svo replay) via the ZED SDK Python API.

    Install: download the ZED SDK (stereolabs.com), then its Python API with
    `python /usr/local/zed/get_python_api.py` (provides the `pyzed` module).
    Needs a CUDA GPU. If unavailable, use --source folder/webcam/selftest.
    """

    def __init__(self, resolution="HD720", fps=30, depth_mode="NEURAL",
                 min_depth=None, max_depth=None, svo=None):
        try:
            import pyzed.sl as sl
        except ImportError as e:
            raise SystemExit(
                "[zed] pyzed not found. Install the ZED SDK and run "
                "`python /usr/local/zed/get_python_api.py`, or use "
                "--source folder/webcam/selftest.\n  underlying error: %s" % e)
        self.sl = sl

        init = sl.InitParameters()
        init.coordinate_units = sl.UNIT.METER            # depth straight in metres
        init.depth_mode = getattr(sl.DEPTH_MODE, depth_mode)
        if min_depth is not None:
            init.depth_minimum_distance = float(min_depth)  # hands are close
        if max_depth is not None:
            init.depth_maximum_distance = float(max_depth)
        if svo:
            init.set_from_svo_file(svo)
            print(f"[zed] replaying SVO: {svo}")
        else:
            init.camera_resolution = getattr(sl.RESOLUTION, resolution)
            init.camera_fps = fps

        self.cam = sl.Camera()
        status = self.cam.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise SystemExit(
                f"[zed] camera open failed: {status}. Check the USB3 cable / power, "
                "the ZED SDK + CUDA install, or use --source folder/webcam/selftest.")

        self.runtime = sl.RuntimeParameters()
        self._img = sl.Mat()
        self._depth = sl.Mat()

        # Depth is in the LEFT rectified frame -> use the LEFT cam intrinsics.
        info = self.cam.get_camera_information()
        try:                                              # ZED SDK >= 4.x
            cp = info.camera_configuration.calibration_parameters.left_cam
        except AttributeError:                            # ZED SDK 3.x
            cp = info.calibration_parameters.left_cam
        self.K = np.array([[cp.fx, 0.0, cp.cx],
                           [0.0, cp.fy, cp.cy],
                           [0.0, 0.0, 1.0]], dtype=np.float32)
        print(f"[zed] opened ({'svo' if svo else resolution}) depth_mode={depth_mode}; "
              f"left K=\n{self.K}")

    def read(self):
        if self.cam.grab(self.runtime) != self.sl.ERROR_CODE.SUCCESS:
            return None  # end of SVO or dropped frame
        self.cam.retrieve_image(self._img, self.sl.VIEW.LEFT)
        self.cam.retrieve_measure(self._depth, self.sl.MEASURE.DEPTH)

        bgra = self._img.get_data()                       # HxWx4 BGRA uint8
        bgr = np.ascontiguousarray(bgra[:, :, :3])
        # ZED depth: metres, with NaN/+-Inf for invalid -> map to 0 (our convention).
        depth_m = np.asarray(self._depth.get_data(), dtype=np.float32)
        depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
        depth_m[depth_m < 0] = 0.0
        return bgr, depth_m, self.K

    def close(self):
        try:
            self.cam.close()
        except Exception:
            pass


# ===========================================================================
# Main loop (mirrors infer_rgbd_femtobolt.main; ZED source + window title)
# ===========================================================================
def build_source(args):
    if args.source == "zed":
        return ZEDSource(resolution=args.zed_resolution, fps=args.zed_fps,
                         depth_mode=args.zed_depth_mode, min_depth=args.zed_min_depth,
                         max_depth=args.zed_max_depth)
    if args.source == "svo":
        if not args.svo:
            raise SystemExit("[zed] --source svo requires --svo <file.svo|.svo2>")
        return ZEDSource(depth_mode=args.zed_depth_mode, min_depth=args.zed_min_depth,
                         max_depth=args.zed_max_depth, svo=args.svo)
    if args.source == "folder":
        return FolderSource(args.path)
    if args.source == "webcam":
        return WebcamSource(args.cam_index)
    raise ValueError(args.source)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cfg", default="./configs_rgbd.yaml")
    ap.add_argument("--source", default="zed",
                    choices=["zed", "svo", "folder", "webcam", "selftest"])
    ap.add_argument("--svo", default=None, help="path to a .svo/.svo2 recording")
    ap.add_argument("--path", default="captures", help="folder source dir")
    ap.add_argument("--cam_index", default=0, type=int, help="webcam index")
    ap.add_argument("--rgb_only", action="store_true", help="zero the depth channel")
    ap.add_argument("--norm_scale", default=0.1, type=float, help="depth norm scale (m->1)")
    ap.add_argument("--margin", default=0.50, type=float, help="bbox margin fraction")
    ap.add_argument("--hand_model", default=None,
                    help="path to MediaPipe hand_landmarker.task (auto-downloaded if absent)")
    # ZED knobs
    ap.add_argument("--zed_resolution", default="HD720",
                    choices=["HD2K", "HD1080", "HD720", "VGA"])
    ap.add_argument("--zed_fps", default=30, type=int)
    ap.add_argument("--zed_depth_mode", default="NEURAL",
                    choices=["NEURAL", "ULTRA", "QUALITY", "PERFORMANCE"],
                    help="NEURAL = best for hands; PERFORMANCE = fastest")
    ap.add_argument("--zed_min_depth", default=0.3, type=float,
                    help="metres; hands are close, so a small min helps (ZED Mini/2 lower)")
    ap.add_argument("--zed_max_depth", default=2.0, type=float, help="metres; cull far scene")
    ap.add_argument("--no-gui", dest="gui", action="store_false", help="headless")
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
                cv2.imshow("RGB-D hand pose (ZED)", overlay)
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
