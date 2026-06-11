"""Pre-render a synthetic metric-depth cache for HanCo.

For every (scene, frame, cam) with both an RGB image and GT joints, this renders
a capsule-mesh depth map (datasets/depth_synth.py) in the camera frame at the
native RGB resolution and writes it as a 16-bit PNG in millimetres, mirroring the
RGB layout:

    <hanco_root>/rgb  /<scene>/cam<c>/<frame>.jpg     (input)
    <hanco_root>/depth/<scene>/cam<c>/<frame>.png     (output, uint16 mm, 0 = bg)

Only the *metric depth at native resolution* is cached; the cheap per-sample
augmentation (crop warp + sensor noise + normalisation) stays on-the-fly in the
dataloader, so one cache serves every augmentation seed and bbox jitter.

Geometry source: the 21 GT joints in `xyz/` (world) transformed into each camera
by the per-frame extrinsics `M[cam]` in `calib/`. (HanCo also ships MANO fits in
`shape/`, but rendering the true mesh needs the license-gated MANO_RIGHT.pkl,
which is absent here; the capsule proxy needs only the joints.)

Run:
    python -m datasets.gen_hanco_depth                       # full set, all cores
    python -m datasets.gen_hanco_depth --limit_scenes 2      # quick smoke test
    python -m datasets.gen_hanco_depth --overwrite           # ignore existing PNGs
"""

import os
import json
import argparse
import multiprocessing as mp

import numpy as np
import cv2

from datasets.depth_synth import render_hand_depth

DEFAULT_ROOT = r"/home/jucpark/DeepLearning/Datasets/Hand Dataset/HanCo"
N_CAMS = 8

# Capsule radii must match the dataloader's depth_cfg defaults so the cached
# metric depth is identical to what on-the-fly rendering would produce.
CAPSULE = dict(base_radius=0.009, palm_radius=0.014, tip_scale=0.55, ring=6)


def _render_one(rgb_path, K, joints_cam, out_path, overwrite):
    if os.path.exists(out_path) and not overwrite:
        return 0  # skipped
    img = cv2.imread(rgb_path)
    if img is None:
        return -1  # rgb missing
    H, W = img.shape[:2]
    depth_m = render_hand_depth(K, H, W, joints=joints_cam, **CAPSULE)
    depth_mm = np.clip(np.round(depth_m * 1000.0), 0, 65535).astype(np.uint16)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, depth_mm)  # 16-bit single channel PNG
    return 1  # written


def _process_scene(task):
    """Render every frame x cam of one scene. Returns (written, skipped, missing)."""
    root, out_dir, scene, overwrite = task
    xyz_dir = os.path.join(root, "xyz", scene)
    calib_dir = os.path.join(root, "calib", scene)
    if not os.path.isdir(xyz_dir):
        return (0, 0, 0)
    written = skipped = missing = 0
    for fn in sorted(os.listdir(xyz_dir)):
        if not fn.endswith(".json"):
            continue
        frame = fn[:-5]
        calib_fp = os.path.join(calib_dir, fn)
        if not os.path.isfile(calib_fp):
            continue
        try:
            xyz = np.asarray(json.load(open(os.path.join(xyz_dir, fn))), np.float64)  # (21,3) world
            calib = json.load(open(calib_fp))
        except Exception:
            continue
        xyz_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1).T  # (4,21)
        for c in range(N_CAMS):
            rgb_path = os.path.join(root, "rgb", scene, f"cam{c}", frame + ".jpg")
            if not os.path.isfile(rgb_path):
                missing += 1
                continue
            K = np.asarray(calib["K"][c], np.float64)
            M = np.asarray(calib["M"][c], np.float64)
            joints_cam = (M @ xyz_h).T[:, :3]
            out_path = os.path.join(out_dir, scene, f"cam{c}", frame + ".png")
            r = _render_one(rgb_path, K, joints_cam, out_path, overwrite)
            if r == 1:
                written += 1
            elif r == 0:
                skipped += 1
            else:
                missing += 1
    return (written, skipped, missing)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hanco_root", default=DEFAULT_ROOT)
    ap.add_argument("--out_dir", default=None, help="default: <hanco_root>/depth")
    ap.add_argument("--num_workers", default=max(1, (os.cpu_count() or 2) - 2), type=int)
    ap.add_argument("--limit_scenes", default=0, type=int, help="0 = all scenes")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = args.hanco_root
    out_dir = args.out_dir or os.path.join(root, "depth")
    os.makedirs(out_dir, exist_ok=True)

    scenes = sorted(s for s in os.listdir(os.path.join(root, "xyz"))
                    if os.path.isdir(os.path.join(root, "xyz", s)))
    if args.limit_scenes > 0:
        scenes = scenes[:args.limit_scenes]
    print(f"[gen_hanco_depth] {len(scenes)} scenes -> {out_dir}  "
          f"(workers={args.num_workers}, overwrite={args.overwrite})")

    tasks = [(root, out_dir, s, args.overwrite) for s in scenes]
    tot_w = tot_s = tot_m = 0
    with mp.Pool(args.num_workers) as pool:
        for i, (w, s, m) in enumerate(pool.imap_unordered(_process_scene, tasks), 1):
            tot_w += w; tot_s += s; tot_m += m
            if i % 20 == 0 or i == len(scenes):
                print(f"[{i}/{len(scenes)} scenes] written={tot_w} skipped={tot_s} missing={tot_m}",
                      flush=True)
    print(f"[done] written={tot_w} skipped={tot_s} missing={tot_m}  out_dir={out_dir}")


if __name__ == "__main__":
    main()
