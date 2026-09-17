"""Stream live 3D hand joints to Unity over UDP.

Runs the deploy-best HYBRID lifter (FastViT-SA12 replay_ft by default) on a ZED
camera / SVO recording / saved iPhone captures, and sends one small JSON packet
per frame so a Unity scene can render the hand and drive grab-and-rotate
interaction (see unity/HandGrab/).

Packet (UDP, JSON, one datagram per frame):
    {"t": <sec>, "det": 1, "fps": 21.3, "j": [x0,y0,z0, x1,y1,z1, ... x20,y20,z20]}
    {"t": <sec>, "det": 0}                                    # no hand this frame
`j` is 21 joints * 3 floats, METERS, camera frame (x right, y down, z forward),
absolute (root placed at the sensor wrist depth). Unity does y-flip + smoothing.

Examples
--------
  python scripts/tools/unity_stream_hand.py                                  # live ZED -> 127.0.0.1:9750
  python scripts/tools/unity_stream_hand.py --source replay --replay_dir rgbd_captures_04
  python scripts/tools/unity_stream_hand.py --udp 192.168.0.42:9750 --no-gui
"""

import _bootstrap  # noqa: F401  (repo-root + sibling script imports)

import os
import re
import glob
import json
import time
import socket
import argparse

import numpy as np
import cv2
import torch

from utils import load_cfg
from train_hybrid import HybridLifter
from train_zlifter import build_hand_crop
from infer_rgbd_femtobolt import HandDetector, project, draw_skeleton
from infer_hybrid import sample_depth


@torch.no_grad()
def infer_joints(model, det, bgr, depth_m, K, device):
    """One frame -> (abs_joints[21,3] meters | None, uv | None)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pts = det.detect(bgr)
    if pts is None:
        return None, None
    uv = np.asarray(pts, np.float32)[:, :2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    zs = np.zeros(21, np.float32); zv = np.zeros(21, np.float32)
    for j in range(21):
        zs[j], zv[j] = sample_depth(depth_m, uv[j, 0], uv[j, 1])
    z_ref = float(np.median(zs[zv > 0])) if zv.any() else 0.5
    z_use = np.where(zv > 0, zs, z_ref).astype(np.float32)
    feat = np.stack([uv[:, 0] / bgr.shape[1] * 2 - 1, uv[:, 1] / bgr.shape[0] * 2 - 1,
                     z_use - z_ref, zv], axis=1).astype(np.float32)
    crop = build_hand_crop(rgb, depth_m, uv)
    p, _, _ = model(
        torch.from_numpy(feat).unsqueeze(0).to(device),
        crop.unsqueeze(0).to(device),
        torch.from_numpy(uv).unsqueeze(0).to(device),
        torch.from_numpy(z_use).unsqueeze(0).to(device),
        torch.tensor([[fx, fy, cx, cy]], dtype=torch.float32).to(device))
    p = p[0].cpu().numpy()
    zw = zs[0] if zv[0] > 0 else z_ref
    root = np.array([(uv[0, 0] - cx) / fx * zw, (uv[0, 1] - cy) / fy * zw, zw], np.float32)
    return p + root, uv


class ReplaySource:
    """Loop over saved cap_*.json captures (e.g. rgbd_captures_04) at a fixed fps."""

    def __init__(self, cap_dir, fps=15.0):
        from infer_rgbd_captures import load_capture, rotate_frame, vote_rotation
        self._load, self._rot = load_capture, rotate_frame
        base = re.compile(r"^cap_\d+\.json$")
        self.stems = sorted(s[:-5] for s in glob.glob(os.path.join(cap_dir, "cap_*.json"))
                            if base.match(os.path.basename(s)))
        if not self.stems:
            raise SystemExit(f"[replay] no cap_*.json in {cap_dir}")
        det = HandDetector()
        self.k = vote_rotation(self.stems, det)
        det.close()
        self.i, self.dt, self._t = 0, 1.0 / max(fps, 1e-3), 0.0
        print(f"[replay] {len(self.stems)} frames from {cap_dir} (rot k={self.k})")

    def read(self):
        wait = self._t + self.dt - time.time()
        if wait > 0:
            time.sleep(wait)
        self._t = time.time()
        cap = None
        for _ in range(len(self.stems)):                 # skip unreadable frames
            cap = self._load(self.stems[self.i % len(self.stems)])
            self.i += 1
            if cap is not None:
                break
        return None if cap is None else self._rot(*cap, self.k)

    def close(self):
        pass


def build_source(args):
    if args.source in ("zed", "svo"):
        from infer_rgbd_zed import ZEDSource
        if args.source == "svo" and not args.svo:
            raise SystemExit("[zed] --source svo requires --svo <file.svo|.svo2>")
        return ZEDSource(resolution=args.zed_resolution, fps=args.zed_fps,
                         depth_mode=args.zed_depth_mode, min_depth=args.zed_min_depth,
                         max_depth=args.zed_max_depth,
                         svo=(args.svo if args.source == "svo" else None))
    if args.source == "replay":
        return ReplaySource(args.replay_dir, fps=args.replay_fps)
    raise ValueError(args.source)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="mobrecon_ckpt/hybrid_sa12_replay_ft_bb_uv2d/best.pt")
    ap.add_argument("--cfg", default="./configs/configs_rgbd.yaml")
    ap.add_argument("--backbone", default="fastvit", choices=["densestack", "fastvit"])
    ap.add_argument("--fastvit_variant", default="fastvit_sa12")
    ap.add_argument("--mode", default="backbone", choices=["backbone", "locked"])
    ap.add_argument("--source", default="zed", choices=["zed", "svo", "replay"])
    ap.add_argument("--svo", default=None)
    ap.add_argument("--replay_dir", default="rgbd_captures_04")
    ap.add_argument("--replay_fps", default=15.0, type=float)
    ap.add_argument("--udp", default="127.0.0.1:9750", help="host:port Unity listens on")
    ap.add_argument("--zed_resolution", default="HD720",
                    choices=["HD2K", "HD1080", "HD720", "VGA"])
    ap.add_argument("--zed_fps", default=30, type=int)
    ap.add_argument("--zed_depth_mode", default="NEURAL",
                    choices=["NEURAL", "ULTRA", "QUALITY", "PERFORMANCE"])
    ap.add_argument("--zed_min_depth", default=0.3, type=float)
    ap.add_argument("--zed_max_depth", default=2.0, type=float)
    ap.add_argument("--no-gui", dest="gui", action="store_false")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_cfg(args.cfg)
    model = HybridLifter(cfg, mode=args.mode, backbone=args.backbone,
                         fastvit_variant=args.fastvit_variant).to(device).eval()
    st = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(st.get("model_state_dict", st))
    print(f"[stream] loaded {args.ckpt} (epoch {st.get('epoch','?')}, "
          f"MPJPE {st.get('best_test_mpjpe', float('nan')):.2f}mm)", flush=True)

    host, port = args.udp.rsplit(":", 1)
    addr = (host, int(port))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"[stream] sending to udp://{addr[0]}:{addr[1]}", flush=True)

    det = HandDetector()
    src = build_source(args)
    t_prev, fps, frame_i, n_det = time.time(), 0.0, 0, 0
    try:
        while True:
            frame = src.read()
            if frame is None:
                print("[stream] source ended")
                break
            bgr, depth_m, K = frame
            joints, uv = infer_joints(model, det, bgr, depth_m, K, device)

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now
            if joints is not None:
                n_det += 1
                pkt = {"t": round(now, 4), "det": 1, "fps": round(fps, 1),
                       "j": [round(float(v), 5) for v in joints.reshape(-1)]}
            else:
                pkt = {"t": round(now, 4), "det": 0}
            sock.sendto(json.dumps(pkt).encode(), addr)

            if args.gui:
                overlay = bgr.copy()
                if joints is not None:
                    overlay = draw_skeleton(overlay, project(joints, K))
                    cv2.putText(overlay, f"wrist_z={joints[0,2]*100:5.1f}cm", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(overlay, f"{fps:4.1f} FPS -> udp {addr[0]}:{addr[1]}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.imshow("unity hand stream", overlay)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            if frame_i and frame_i % 60 == 0:
                print(f"[stream] frame {frame_i}  {fps:4.1f} FPS  (hand {n_det}/{frame_i})",
                      flush=True)
            frame_i += 1
    finally:
        src.close()
        det.close()
        sock.close()
        if args.gui:
            cv2.destroyAllWindows()
    print(f"[stream] done: {frame_i} frames, hand in {n_det}", flush=True)


if __name__ == "__main__":
    main()
