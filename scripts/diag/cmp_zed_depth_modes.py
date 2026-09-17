"""NEURAL vs PERFORMANCE depth-mode accuracy comparison for hybrid_B on the ZED.

No GT on a live rig, so we make the comparison FRAME-ALIGNED and use NEURAL (the
accurate mode) as the reference. A ZED SVO stores only the raw stereo; depth is
recomputed at replay time from the chosen depth_mode, so replaying ONE recording
through both modes yields per-frame depth maps for the identical scene.

Steps:
  1. record a short SVO from the live camera (or reuse one via --svo),
  2. replay it twice (NEURAL, then PERFORMANCE), running the full hybrid_B path,
  3. report GT-free proxies:
       * depth fill-rate  (% valid pixels in [min,max])   -- higher = denser
       * PERF vs NEURAL depth agreement (median |dz| where both valid, mm)
       * hand-joint depth coverage (% of 21 joints with a valid sensor sample)
       * hybrid_B output shift: mean per-joint |pose_PERF - pose_NEURAL| (mm),
         i.e. how much the cheaper depth changes the delivered 3D pose.

  python scripts/diag/cmp_zed_depth_modes.py --seconds 12
  python scripts/diag/cmp_zed_depth_modes.py --svo scratch/cmp.svo2   # reuse a recording
"""

import _bootstrap  # noqa: F401  (repo-root + sibling script imports)

import os
import argparse

import numpy as np
import cv2
import torch

from utils import load_cfg
from train_hybrid import HybridLifter
from train_zlifter import build_hand_crop
from infer_rgbd_femtobolt import HandDetector
from infer_rgbd_zed import ZEDSource
from infer_hybrid import sample_depth

SCRATCH = ("/tmp/claude-1000/-home-jucpark-DeepLearning-handmesh-etr/"
           "76a27eb5-861c-4268-abae-87193cc8e5f3/scratchpad")


def record_svo(path, seconds, resolution="HD720", fps=30):
    import pyzed.sl as sl
    init = sl.InitParameters()
    init.camera_resolution = getattr(sl.RESOLUTION, resolution)
    init.camera_fps = fps
    init.coordinate_units = sl.UNIT.METER
    cam = sl.Camera()
    if cam.open(init) != sl.ERROR_CODE.SUCCESS:
        raise SystemExit("[rec] camera open failed")
    rp = sl.RecordingParameters(path, sl.SVO_COMPRESSION_MODE.LOSSLESS)
    if cam.enable_recording(rp) != sl.ERROR_CODE.SUCCESS:
        raise SystemExit("[rec] enable_recording failed")
    rt = sl.RuntimeParameters()
    n = int(seconds * fps)
    print(f"[rec] recording {n} frames (~{seconds}s) to {path} -- keep a hand in view", flush=True)
    got = 0
    for _ in range(n):
        if cam.grab(rt) == sl.ERROR_CODE.SUCCESS:
            got += 1
    cam.disable_recording()
    cam.close()
    print(f"[rec] wrote {got} frames", flush=True)


@torch.no_grad()
def run_pass(svo, mode, model, det, device):
    """Replay the SVO under one depth mode. Returns per-frame dicts."""
    src = ZEDSource(depth_mode=mode, min_depth=0.3, max_depth=2.0, svo=svo)
    lo, hi = 0.3, 2.0
    out = []
    while True:
        fr = src.read()
        if fr is None:
            break
        bgr, depth, K = fr
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        valid = (depth >= lo) & (depth <= hi)
        rec = {"fill": float(valid.mean()), "depth": depth, "valid": valid, "hand": False}
        pts = det.detect(bgr)
        if pts is not None:
            uv = np.asarray(pts, np.float32)[:, :2]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            zs = np.zeros(21, np.float32); zv = np.zeros(21, np.float32)
            for j in range(21):
                zs[j], zv[j] = sample_depth(depth, uv[j, 0], uv[j, 1])
            z_ref = float(np.median(zs[zv > 0])) if zv.any() else 0.5
            z_use = np.where(zv > 0, zs, z_ref).astype(np.float32)
            feat = np.stack([uv[:, 0] / bgr.shape[1] * 2 - 1, uv[:, 1] / bgr.shape[0] * 2 - 1,
                             z_use - z_ref, zv], axis=1).astype(np.float32)
            crop = build_hand_crop(rgb, depth, uv)
            p, _, _ = model(torch.from_numpy(feat).unsqueeze(0).to(device),
                            crop.unsqueeze(0).to(device),
                            torch.from_numpy(uv).unsqueeze(0).to(device),
                            torch.from_numpy(z_use).unsqueeze(0).to(device),
                            torch.tensor([[fx, fy, cx, cy]], dtype=torch.float32).to(device))
            zw = zs[0] if zv[0] > 0 else z_ref
            root = np.array([(uv[0, 0] - cx) / fx * zw, (uv[0, 1] - cy) / fy * zw, zw], np.float32)
            rec.update(hand=True, cov=float((zv > 0).mean()),
                       wrist_z=float(zw), pose=(p[0].cpu().numpy() + root))
        out.append(rec)
    src.close()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--svo", default=None, help="reuse an existing SVO instead of recording")
    ap.add_argument("--seconds", default=12, type=int)
    ap.add_argument("--ckpt", default="mobrecon_ckpt/hybrid_B/best.pt")
    ap.add_argument("--cfg", default="./configs/configs_rgbd.yaml")
    args = ap.parse_args()

    svo = args.svo or os.path.join(SCRATCH, "cmp_depth.svo2")
    if not args.svo:
        record_svo(svo, args.seconds)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_cfg(args.cfg)
    model = HybridLifter(cfg, mode="locked").to(device).eval()
    st = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(st.get("model_state_dict", st))
    det = HandDetector()
    print(f"[cmp] hybrid_B loaded (epoch {st.get('epoch','?')})", flush=True)

    print("[cmp] pass 1/2: NEURAL", flush=True)
    R_n = run_pass(svo, "NEURAL", model, det, device)
    print("[cmp] pass 2/2: PERFORMANCE", flush=True)
    R_p = run_pass(svo, "PERFORMANCE", model, det, device)
    det.close()

    m = min(len(R_n), len(R_p))
    R_n, R_p = R_n[:m], R_p[:m]

    fill_n = np.mean([r["fill"] for r in R_n]) * 100
    fill_p = np.mean([r["fill"] for r in R_p]) * 100
    # depth agreement where BOTH valid (PERF deviation from NEURAL reference), mm
    dz = []
    for a, b in zip(R_n, R_p):
        both = a["valid"] & b["valid"]
        if both.any():
            dz.append(np.median(np.abs(a["depth"][both] - b["depth"][both])) * 1000)
    dz = float(np.mean(dz)) if dz else float("nan")

    hn = [r for r in R_n if r["hand"]]; hp = [r for r in R_p if r["hand"]]
    cov_n = np.mean([r["cov"] for r in hn]) * 100 if hn else float("nan")
    cov_p = np.mean([r["cov"] for r in hp]) * 100 if hp else float("nan")
    # frames where a hand was found in BOTH modes -> compare final pose + wrist z
    pose_d, wrist_d = [], []
    for a, b in zip(R_n, R_p):
        if a["hand"] and b["hand"]:
            pose_d.append(np.linalg.norm(a["pose"] - b["pose"], axis=1).mean() * 1000)
            wrist_d.append(abs(a["wrist_z"] - b["wrist_z"]) * 1000)
    pose_d_mean = float(np.mean(pose_d)) if pose_d else float("nan")
    wrist_d_mean = float(np.mean(wrist_d)) if wrist_d else float("nan")

    print("\n================ NEURAL vs PERFORMANCE (frame-aligned) ================")
    print(f"frames compared           : {m}   (hand in both: {len(pose_d)})")
    print(f"depth fill-rate  NEURAL    : {fill_n:5.1f}%   valid pixels in [0.3,2.0]m")
    print(f"depth fill-rate  PERF      : {fill_p:5.1f}%")
    print(f"PERF vs NEURAL depth diff  : {dz:5.1f} mm   median |dz|, both-valid pixels")
    print(f"hand-joint coverage NEURAL : {cov_n:5.1f}%   of 21 joints with valid sensor depth")
    print(f"hand-joint coverage PERF   : {cov_p:5.1f}%")
    print(f"wrist-z |PERF - NEURAL|    : {wrist_d_mean:5.1f} mm   sensor-anchored root shift")
    print(f"hybrid_B pose |PERF-NEURAL|: {pose_d_mean:5.1f} mm   mean per-joint output shift")
    print("=======================================================================")
    print("NEURAL is the accurate reference; smaller PERF deviations = PERF is 'closer'.")


if __name__ == "__main__":
    main()
