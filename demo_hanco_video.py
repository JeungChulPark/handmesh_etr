# Copyright (c) 2026. Depth-Aware Hand Pose Initiative.
"""
Make a verification video: run infer_rgbd_femtobolt's pipeline on a HanCo scene
(real RGB + the cached in-distribution depth the model trained on) and composite
GT vs predicted 3D hand pose onto each frame.

No camera needed -- this is the offline stand-in for a live Femto Bolt feed,
streaming a HanCo camera's consecutive frames as if they were a video.

  python demo_hanco_video.py --ckpt mobrecon_ckpt/rgbd_1hr/3.pt --scene 0000 --cam 0
"""
import os, json, glob, argparse
import numpy as np, cv2

from infer_rgbd_femtobolt import (
    RGBDHandPose, HandDetector, landmarks_to_bbox, project, HAND_BONES, _FINGER_COLORS)

HANCO = "/home/jucpark/DeepLearning/Datasets/Hand Dataset/HanCo"


def draw(img, uv, color, thick=2, dot=2):
    for fi, (a, b) in enumerate(HAND_BONES):
        c = color if color else _FINGER_COLORS[fi // 4]
        cv2.line(img, tuple(uv[a].astype(int)), tuple(uv[b].astype(int)), c, thick, cv2.LINE_AA)
    for p in uv.astype(int):
        cv2.circle(img, tuple(p), dot, (255, 255, 255), -1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scene", default="0000")
    ap.add_argument("--cam", default=0, type=int)
    ap.add_argument("--out", default="out/hanco_demo.mp4")
    ap.add_argument("--fps", default=12, type=int)
    ap.add_argument("--scale", default=3, type=int, help="upscale factor for legibility")
    args = ap.parse_args()

    est = RGBDHandPose(args.ckpt)
    det = HandDetector()
    camdir = f"cam{args.cam}"
    frames = sorted(glob.glob(os.path.join(HANCO, "rgb", args.scene, camdir, "*.jpg")))
    print(f"[demo] {len(frames)} frames from scene {args.scene}/{camdir}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    writer, S = None, args.scale
    n_det, n_root, mpjpe2d = 0, 0, []

    for fp in frames:
        fid = os.path.splitext(os.path.basename(fp))[0]
        bgr = cv2.imread(fp)
        dpath = os.path.join(HANCO, "depth", args.scene, camdir, fid + ".png")
        depth_mm = cv2.imread(dpath, cv2.IMREAD_UNCHANGED)
        depth_m = depth_mm.astype(np.float32) / 1000.0 if depth_mm is not None else np.zeros(bgr.shape[:2], np.float32)
        with open(os.path.join(HANCO, "calib", args.scene, fid + ".json")) as f:
            cal = json.load(f)
        K = np.array(cal["K"][args.cam], np.float32)
        M = np.array(cal["M"][args.cam], np.float32)
        with open(os.path.join(HANCO, "xyz", args.scene, fid + ".json")) as f:
            xyz = np.array(json.load(f), np.float32)            # (21,3) world
        gt_cam = (M @ np.concatenate([xyz, np.ones((21, 1))], 1).T).T[:, :3]
        gt_uv = project(gt_cam, K)

        canvas = cv2.resize(bgr, None, fx=S, fy=S, interpolation=cv2.INTER_NEAREST)
        draw(canvas, gt_uv * S, color=(200, 200, 200), thick=1, dot=2)   # GT = grey

        pts = det.detect(bgr)
        if pts is not None:
            n_det += 1
            bbox = landmarks_to_bbox(pts, bgr.shape[:2])
            inp, _, _ = est.preprocess(bgr, depth_m, bbox, K)
            jr = est.infer(inp)
            abs_j = est.recover_root_aligned(jr, pts, depth_m, K)
            if abs_j is not None:
                n_root += 1
                pr_uv = project(abs_j, K)
                draw(canvas, pr_uv * S, color=None, thick=2, dot=3)      # pred = coloured
                mpjpe2d.append(np.linalg.norm(pr_uv - gt_uv, axis=1).mean())

        # depth panel (full-frame colourised, background black)
        dvalid = depth_m > 0
        dn = np.zeros_like(depth_m)
        if dvalid.any():
            lo, hi = np.percentile(depth_m[dvalid], [2, 98])
            dn[dvalid] = np.clip((depth_m[dvalid] - lo) / max(hi - lo, 1e-6), 0, 1)
        dvis = cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_JET)
        dvis[~dvalid] = 0
        dvis = cv2.resize(dvis, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_NEAREST)

        panel = np.hstack([canvas, dvis])
        cv2.putText(panel, "RGB + GT(grey) vs PRED(colour)", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, "sensor depth (synthetic, in-distribution)",
                    (canvas.shape[1] + 8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2, cv2.LINE_AA)
        if writer is None:
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                     args.fps, (panel.shape[1], panel.shape[0]))
        writer.write(panel)

    writer.release()

    # OpenCV writes MPEG-4 Part 2 (mp4v), which browsers / VS Code / web viewers
    # can't play. Re-encode to H.264 (yuv420p, faststart) in place if ffmpeg is
    # available, so the output plays everywhere.
    import shutil, subprocess
    if shutil.which("ffmpeg"):
        tmp = args.out + ".h264.mp4"
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", args.out,
                            "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart", "-crf", "20", tmp])
        if r.returncode == 0:
            os.replace(tmp, args.out)
            print("[demo] re-encoded to H.264 (browser-playable)")
        elif os.path.exists(tmp):
            os.remove(tmp)
    print(f"[demo] wrote {args.out}")
    print(f"[demo] hand detected {n_det}/{len(frames)}, root-recovered {n_root}/{len(frames)}")
    if mpjpe2d:
        print(f"[demo] mean 2D joint error (pred vs GT pixels): {np.mean(mpjpe2d):.1f} px")


if __name__ == "__main__":
    main()
