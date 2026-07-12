"""Cache REAL MediaPipe 2D landmarks on DexYCB frames (full image, original camera).

For the Z-only lifter: the 2D (u,v) fed to the model must match the deploy source
(real MediaPipe), so we run MediaPipe over the actual dataset RGB and cache the 21
pixel landmarks per frame. Runs in the ORIGINAL camera frame (no crop) -- the lifter
unprojects with the full-frame K, so no crop/augmentation bookkeeping is needed.

Cache = one .npz:  paths [N] (color file), uv [N,21,2] px, det [N] (1=detected),
subj [N] (subject id, for a leakage-free train/test split).

    python cache_mediapipe.py --root "$DEXYCB" --stride 8 --out cache/mp_dexycb.npz
    python cache_mediapipe.py --root "$DEXYCB" --limit 30 --out /tmp/mp_test.npz   # smoke
"""
import os
import argparse

import numpy as np
import cv2
from tqdm import tqdm

from infer_rgbd_femtobolt import HandDetector


def _list_frames(dataset, root):
    """Return list of (color_path, group_id). group_id = split key (leakage-free):
    DexYCB subject, HO3D sequence."""
    if dataset == "dexycb":
        from datasets.dexycb import DexYCB_RGBD
        ds = DexYCB_RGBD(root=root, mode="test", with_depth=False,
                         hands="right", flip_left=False)
        out = []
        for s in ds.samples:
            cp = s["color"]
            g = next((p for p in cp.split(os.sep) if "-subject-" in p), "unknown")
            out.append((cp, g))
        return out
    if dataset == "ho3d":
        from datasets.ho3d import HO3D_RGBD
        ds = HO3D_RGBD(root=root, mode="test", with_depth=False)
        # rgb path = .../train/<seq>/rgb/<fid>.jpg -> group = <seq>
        return [(s["rgb"], os.path.basename(os.path.dirname(os.path.dirname(s["rgb"]))))
                for s in ds.samples]
    if dataset == "iphone":
        import glob as _g
        out = []
        for d in str(root).split(","):
            d = d.strip()
            # group = dir path with separators flattened, so FineTuning/rgbd_captures
            # ('FineTuning_...') cannot collide with the top-level rgbd_captures and
            # rgbd_captures_04 stays LAST in sorted order (the holdout convention).
            grp = d.rstrip("/").replace(os.sep, "_")
            for gt in sorted(_g.glob(os.path.join(d, "cap_*_gt.json"))):
                stem = gt[:-len("_gt.json")]
                rgb = next((stem + "_rgb" + e for e in (".png", ".jpg", ".jpeg")
                            if os.path.isfile(stem + "_rgb" + e)), None)
                if rgb:
                    out.append((rgb, grp))
        return out
    raise ValueError(dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dexycb", choices=["dexycb", "ho3d", "iphone"])
    ap.add_argument("--root", required=True, help="dataset root")
    ap.add_argument("--out", default="cache/mp_dexycb.npz")
    ap.add_argument("--stride", default=8, type=int, help="keep every Nth frame")
    ap.add_argument("--limit", default=0, type=int, help="cap total frames (0=all after stride)")
    ap.add_argument("--det_conf", default=0.4, type=float)
    args = ap.parse_args()

    frames = _list_frames(args.dataset, args.root)[::args.stride]
    if args.limit:
        frames = frames[:args.limit]
    print(f"[cache_mp] {args.dataset}: {len(frames)} frames (stride={args.stride})", flush=True)

    det = HandDetector(det_conf=args.det_conf)
    paths, uvs, dets, subjs = [], [], [], []
    n_ok = 0
    for cp, g in tqdm(frames):
        bgr = cv2.imread(cp)
        uv = det.detect(bgr) if bgr is not None else None
        if uv is None:
            uv = np.zeros((21, 2), np.float32); ok = 0
        else:
            ok = 1; n_ok += 1
        paths.append(cp); uvs.append(uv); dets.append(ok); subjs.append(g)
    det.close()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, paths=np.array(paths), uv=np.stack(uvs).astype(np.float32),
                        det=np.array(dets, np.int8), subj=np.array(subjs))
    print(f"[cache_mp] detected {n_ok}/{len(frames)} ({100*n_ok/max(len(frames),1):.1f}%) "
          f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
