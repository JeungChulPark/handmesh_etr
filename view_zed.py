# Copyright (c) 2026. Depth-Aware Hand Pose Initiative.
"""
Live ZED camera viewer — RGB + metric depth, with snapshot capture & SVO recording.

A standalone visualiser (no model) to sanity-check the ZED feed before running
`infer_rgbd_zed.py`: confirm the hand is in range, depth coverage on the hand,
and capture frames/clips for offline replay. Reuses `ZEDSource` from
infer_rgbd_zed.py, so capture/intrinsics behaviour is identical to inference.

Window: left = RGB (LEFT view), right = colourised depth (JET, 0/invalid = black),
clamped to [--min_depth, --max_depth] m. HUD shows FPS, valid-depth %, and the
metric depth under the centre cross.

Keys
----
  s : save a snapshot  -> <save>/rgb_<n>.png + <save>/depth_<n>.png (uint16 mm)
      (the exact layout infer_rgbd_zed/femtobolt `--source folder` replays)
  r : toggle SVO recording -> <save>/zed_rec.svo2  (live source only)
  q / ESC : quit

Examples
--------
  python view_zed.py                              # live, GUI
  python view_zed.py --source svo --svo rec.svo2  # replay a recording
  python view_zed.py --save captures              # enable 's' snapshots / 'r' record
  python view_zed.py --no-gui --save captures --max_frames 300   # headless dump
"""

import os
import time
import argparse

import numpy as np
import cv2

from infer_rgbd_zed import ZEDSource


def colourise_depth(depth_m, dmin, dmax):
    """Metric depth (m, 0=invalid) -> BGR JET image, invalid pixels black."""
    valid = depth_m > 0
    d = np.clip((depth_m - dmin) / max(dmax - dmin, 1e-6), 0, 1)
    vis = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_JET)
    vis[~valid] = 0
    return vis


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="zed", choices=["zed", "svo"])
    ap.add_argument("--svo", default=None, help="path to a .svo/.svo2 for --source svo")
    ap.add_argument("--zed_resolution", default="HD720",
                    choices=["HD2K", "HD1080", "HD720", "VGA"])
    ap.add_argument("--zed_fps", default=30, type=int)
    ap.add_argument("--zed_depth_mode", default="NEURAL",
                    choices=["NEURAL", "ULTRA", "QUALITY", "PERFORMANCE"])
    ap.add_argument("--min_depth", default=0.2, type=float, help="m, colourmap near bound")
    ap.add_argument("--max_depth", default=2.0, type=float, help="m, colourmap far bound")
    ap.add_argument("--save", default=None, help="dir for snapshots / SVO recording")
    ap.add_argument("--no-gui", dest="gui", action="store_false", help="headless")
    ap.add_argument("--max_frames", default=0, type=int, help="stop after N frames (0=∞)")
    args = ap.parse_args()

    if args.source == "svo":
        if not args.svo:
            raise SystemExit("[view] --source svo requires --svo <file.svo|.svo2>")
        src = ZEDSource(depth_mode=args.zed_depth_mode, min_depth=args.min_depth,
                        max_depth=args.max_depth, svo=args.svo)
    else:
        src = ZEDSource(resolution=args.zed_resolution, fps=args.zed_fps,
                        depth_mode=args.zed_depth_mode, min_depth=args.min_depth,
                        max_depth=args.max_depth)

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        np.save(os.path.join(args.save, "K.npy"), src.K)  # for folder-replay intrinsics

    recording = False
    t_prev, fps, n_saved, frame_i = time.time(), 0.0, 0, 0
    print("[view] keys: s=snapshot  r=toggle SVO record  q/ESC=quit")
    try:
        while True:
            frame = src.read()
            if frame is None:
                print("[view] stream ended")
                break
            bgr, depth_m, K = frame
            h, w = depth_m.shape
            dvis = colourise_depth(depth_m, args.min_depth, args.max_depth)

            # centre-cross metric readout + coverage
            cy, cx = h // 2, w // 2
            cd = depth_m[cy, cx]
            cov = 100.0 * (depth_m > 0).mean()
            for im in (bgr, dvis):
                cv2.drawMarker(im, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 16, 1)

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now
            hud = f"{fps:4.1f}FPS  depth-cov {cov:4.1f}%  centre {cd*100:5.1f}cm"
            if recording:
                hud += "  [REC]"
            cv2.putText(bgr, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255) if not recording else (0, 0, 255), 2)

            # match heights and concatenate RGB | depth
            if dvis.shape[:2] != bgr.shape[:2]:
                dvis = cv2.resize(dvis, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            combo = np.concatenate([bgr, dvis], axis=1)

            if args.gui:
                cv2.imshow("ZED  |  RGB (left)  |  depth (right)", combo)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("s") and args.save:
                    cv2.imwrite(os.path.join(args.save, f"rgb_{n_saved:05d}.png"), bgr)
                    depth_mm = np.clip(np.round(depth_m * 1000.0), 0, 65535).astype(np.uint16)
                    cv2.imwrite(os.path.join(args.save, f"depth_{n_saved:05d}.png"), depth_mm)
                    print(f"[view] saved snapshot {n_saved} (centre {cd*100:.1f}cm)")
                    n_saved += 1
                if key == ord("r") and args.save and args.source == "zed":
                    recording = _toggle_record(src, recording, args.save)
            frame_i += 1
            if args.max_frames and frame_i >= args.max_frames:
                break
    finally:
        if recording:
            try:
                src.cam.disable_recording()
            except Exception:
                pass
        src.close()
        if args.gui:
            cv2.destroyAllWindows()
    print(f"[view] done. frames={frame_i}  snapshots={n_saved}")


def _toggle_record(src, recording, save_dir):
    """Start/stop native SVO recording on the live ZED camera."""
    sl = src.sl
    if not recording:
        path = os.path.join(save_dir, "zed_rec.svo2")
        params = sl.RecordingParameters(path, sl.SVO_COMPRESSION_MODE.H264)
        status = src.cam.enable_recording(params)
        if status != sl.ERROR_CODE.SUCCESS:
            print(f"[view] recording failed to start: {status}")
            return False
        print(f"[view] ● recording -> {path}")
        return True
    src.cam.disable_recording()
    print("[view] ■ recording stopped")
    return False


if __name__ == "__main__":
    main()
