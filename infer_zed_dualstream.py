"""Live ZED inference for the DUAL-STREAM RGB-D hand model (ds_lidarsim etc.).

Unlike infer_rgbd_zed.py (early-fusion model + heuristic root recovery), the
dual-stream model predicts the ABSOLUTE joints itself: its TranslationHead
regresses the global root from the depth stream, so there is no wrist-pixel
backprojection at inference. The one extra input it needs is `depth_med` -- the
median of the RAW sensor depth over the hand crop (metres) -- exactly the
absolute-distance signal it was trained on.

Pipeline (per frame):
  ZED -> bgr + depth_m(metres,0=invalid) + K
  MediaPipe bbox -> augmentation(test) -> img2bb_trans
  RGB  warpAffine(LINEAR)/255 -> [3,256,256]
  depth warpAffine(NEAREST)   -> depth_crop(m) -> median => depth_med
                              -> normalize_depth -> [1,256,256]
  model([1,4,256,256], depth_med[1,1]) -> keypoints_abs (21,3) camera-frame metres
  project(K) -> draw

Run:
  python infer_zed_dualstream.py --ckpt mobrecon_ckpt/ds_lidarsim/best.pt --source zed
  python infer_zed_dualstream.py --ckpt ... --source svo --svo rec.svo2 --no-gui --save out/
"""

import os
import time
import argparse

import numpy as np
import cv2
import torch

from datasets.dataset_utils import augmentation
from datasets.depth_synth import normalize_depth
from models.mobrecon_dualstream import MobRecon_DualStream

# Reuse capture sources + detector + drawing from the existing ZED/Femto pipeline.
from infer_rgbd_zed import ZEDSource, build_source
from infer_rgbd_femtobolt import (
    HandDetector, landmarks_to_bbox, project, draw_skeleton, depth_vis,
    FolderSource, WebcamSource, RGBDHandPose,
)


class DualStreamHandPose:
    """Dual-stream estimator: outputs absolute camera-frame joints directly."""

    def __init__(self, ckpt, device=None, rgb_only=False, norm_scale=0.1):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.rgb_only = rgb_only
        self.norm_scale = norm_scale
        c = torch.load(ckpt, map_location="cpu")
        sd = c.get("model_state_dict", c)
        pose_in_chans = int(sd["rgb_backbone.pre_layer.0.0.weight"].shape[1])
        self.model = MobRecon_DualStream(cfg=None, pose_in_chans=pose_in_chans)
        self.model.load_state_dict(sd, strict=True)
        self.model.to(self.device).eval()
        print(f"[dualstream] {ckpt} (pose_in_chans={pose_in_chans}, "
              f"epoch={c.get('epoch','?')}) device={self.device} rgb_only={rgb_only}")

    def preprocess(self, bgr, depth_m, bbox, K):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        aug_img, img2bb_trans, _, _, _, _, _, _ = augmentation(
            rgb.astype(np.float32), bbox, "test", exclude_flip=True,
            rotation=True, cam_param=K)
        rgb_crop = np.clip(aug_img, 0, 255).astype(np.uint8)
        rgb_t = torch.from_numpy(rgb_crop).float().permute(2, 0, 1) / 255.0

        depth_med = 0.0
        if self.rgb_only:
            depth_norm = np.zeros((256, 256), np.float32)
        else:
            depth_crop = cv2.warpAffine(depth_m, img2bb_trans, (256, 256),
                                        flags=cv2.INTER_NEAREST)
            valid = depth_crop > 0
            # depth_med = absolute-distance signal the TranslationHead expects
            # (raw-depth median over the hand crop, exactly as in training).
            depth_med = float(np.median(depth_crop[valid])) if valid.any() else 0.0
            depth_norm = normalize_depth(depth_crop, scale=self.norm_scale)
        depth_t = torch.from_numpy(depth_norm).float().unsqueeze(0)

        inp = torch.cat([rgb_t, depth_t], dim=0).unsqueeze(0)  # [1,4,256,256]
        return inp.to(self.device), rgb_crop, depth_norm, depth_med

    @torch.no_grad()
    def infer(self, inp, depth_med):
        med = torch.tensor([[depth_med]], dtype=torch.float32, device=self.device)
        out = self.model(inp, med)
        rel = out["keypoints"][0].cpu().numpy()        # (21,3) root-relative metres
        abs_j = out["keypoints_abs"][0].cpu().numpy()  # (21,3) camera-frame metres (learned root)
        root = out["root"][0].cpu().numpy()
        return abs_j, root, rel


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="mobrecon_ckpt/ds_lidarsim/best.pt")
    ap.add_argument("--source", default="zed",
                    choices=["zed", "svo", "folder", "webcam"])
    ap.add_argument("--svo", default=None)
    ap.add_argument("--path", default="captures")
    ap.add_argument("--cam_index", default=0, type=int)
    ap.add_argument("--rgb_only", action="store_true")
    ap.add_argument("--geom_root", action="store_true",
                    help="HYBRID: use the dual-stream POSE (rel joints) but recover the root "
                         "GEOMETRICALLY from wrist depth (sensor-agnostic), not the learned head. "
                         "More robust across sensor domains (e.g. ZED passive stereo).")
    ap.add_argument("--norm_scale", default=0.1, type=float)
    ap.add_argument("--margin", default=0.50, type=float)
    ap.add_argument("--hand_model", default=None)
    ap.add_argument("--zed_resolution", default="HD720",
                    choices=["HD2K", "HD1080", "HD720", "VGA"])
    ap.add_argument("--zed_fps", default=30, type=int)
    ap.add_argument("--zed_depth_mode", default="NEURAL",
                    choices=["NEURAL", "ULTRA", "QUALITY", "PERFORMANCE"])
    ap.add_argument("--zed_min_depth", default=0.3, type=float)
    ap.add_argument("--zed_max_depth", default=2.0, type=float)
    ap.add_argument("--no-gui", dest="gui", action="store_false")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    est = DualStreamHandPose(args.ckpt, rgb_only=args.rgb_only, norm_scale=args.norm_scale)
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
                inp, rgb_crop, dn, depth_med = est.preprocess(bgr, depth_m, bbox, K)
                abs_j, root, rel = est.infer(inp, depth_med)
                mode = "learned-root"
                if args.geom_root:
                    # Hybrid: dual-stream pose + sensor-agnostic geometric root.
                    geom = RGBDHandPose.recover_root_aligned(rel, pts, depth_m, K)
                    if geom is not None:
                        abs_j, mode = geom, "geom-root"
                    else:
                        mode = "geom-root(no depth->learned)"
                uv = project(abs_j, K)
                overlay = draw_skeleton(bgr, uv, bbox)
                cv2.putText(overlay, f"[{mode}] z={abs_j[0,2]*100:5.1f}cm  med={depth_med*100:5.1f}cm  "
                            f"cov={(dn != 0).mean()*100:4.1f}%", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if args.gui:
                    cv2.imshow("depth(crop)", depth_vis(dn))
            else:
                cv2.putText(overlay, "no hand", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now
            cv2.putText(overlay, f"{fps:4.1f} FPS  [dualstream/{args.source}]", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if args.save:
                cv2.imwrite(os.path.join(args.save, f"{frame_i:05d}.png"), overlay)
            if args.gui:
                cv2.imshow("dual-stream RGB-D hand pose (ZED)", overlay)
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
