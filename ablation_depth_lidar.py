"""Depth-contribution ablation for the LiDAR-sim RGB-D model (rgbd_lidar_0613).

Same protocol as ablation_depth.py, but evaluates the LiDAR-simulation-trained
checkpoint and adds a deployment-realistic condition: the model's depth channel
fed with LiDAR-coarsened depth (downscale=4, deterministic eval coarsening),
which is what the iPhone Pro LiDAR will actually deliver.

All conditions run on the SAME HanCo held-out split (seed-0 95/5, the trainer's
split), so gaps isolate the depth channel's effect:

  A    rgbd_lidar + clean depth      -> upper bound with full-res synthetic depth
  A_L  rgbd_lidar + LiDAR depth      -> deployment-realistic (coarse iPhone LiDAR)
  C    rgbd_lidar + depth zeroed     -> how much the LiDAR-sim model RELIES on depth
  B    rgb_only    + depth zeroed    -> the RGB-only baseline (trained w/o depth)

  deployment cost      (A_L - A) = what coarse LiDAR depth costs vs clean depth
  depth RELIANCE       (C  - A) = degradation w/o depth  (old rgbd_1hr: ~106mm collapse)
  depth CONTRIBUTION   (B - A_L) = what LiDAR depth buys over RGB-only at deployment

The whole point of LiDAR-sim training was to BREAK the over-reliance the original
ablation exposed -> expect C - A here to be far smaller than the rgbd_1hr collapse.

    python ablation_depth_lidar.py [rgbd_ckpt] [rgb_only_ckpt]
"""

import os
import sys

import torch
from torch.utils.data import DataLoader, random_split, Subset

from datasets.hanco_ty import HanCo_ETRI_jitter
from models.mobrecon_ds import LargeModel_Extra_RGBD
from utils import load_cfg

CK_RGBD = sys.argv[1] if len(sys.argv) > 1 else "mobrecon_ckpt/rgbd_lidar_0613/best.pt"
CK_RGBO = sys.argv[2] if len(sys.argv) > 2 else "mobrecon_ckpt/rgb_only_1hr/3.pt"
N = 2000
LIDAR_DOWNSCALE = 4


def build_loader(depth_cfg):
    """A HanCo held-out loader; identical seed/split/order across calls so the
    clean- and LiDAR-depth loaders yield matched samples (only the D channel
    differs). shuffle=False + per-idx deterministic eval corruption keep it
    reproducible."""
    torch.manual_seed(0)                              # reproduce the trainer's split
    ds = HanCo_ETRI_jitter(limit=5e6, with_depth=True, depth_source="render",
                           depth_cfg=depth_cfg)
    n_train = int(len(ds) * 0.95)
    _, test_set = random_split(ds, [n_train, len(ds) - n_train])
    ds.mode = "test"                                  # deterministic eval corruption
    test_set = Subset(test_set, list(range(min(N, len(test_set)))))
    return DataLoader(test_set, batch_size=64, shuffle=False, num_workers=8)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for ck in (CK_RGBD, CK_RGBO):
        if not os.path.isfile(ck):
            raise SystemExit(f"[ablation] checkpoint not found: {ck}")

    cfg = load_cfg("configs_rgbd.yaml")

    def load(ck):
        m = LargeModel_Extra_RGBD(cfg).to(dev).eval()
        st = torch.load(ck, map_location=dev)
        m.load_state_dict(st.get("model_state_dict", st))
        return m

    m_rgbd, m_rgbo = load(CK_RGBD), load(CK_RGBO)
    print(f"[ablation] rgbd_lidar = {CK_RGBD}")
    print(f"[ablation] rgb_only   = {CK_RGBO}")

    # matched loaders: clean depth vs LiDAR-coarsened depth (same samples/order)
    dl_clean = build_loader(None)
    dl_lidar = build_loader({"lidar_sim": True, "lidar_downscale": LIDAR_DOWNSCALE})
    print(f"[ablation] {len(dl_clean.dataset)} held-out HanCo samples "
          f"(LiDAR downscale={LIDAR_DOWNSCALE})")

    errs = {"A  rgbd + clean depth": [], "A_L rgbd + LiDAR depth": [],
            "C  rgbd - depth zeroed": [], "B  rgb_only (baseline)": []}

    with torch.no_grad():
        for bc, bl in zip(dl_clean, dl_lidar):
            img = bc["image"].float().to(dev)           # clean RGB-D
            imgL = bl["image"].float().to(dev)          # LiDAR-coarsened depth
            k3 = bc["keypoints3D"].float().to(dev)
            img0 = img.clone(); img0[:, 3:4] = 0.0       # depth channel zeroed

            def mpj(out):
                return (torch.sqrt(((out - k3) ** 2).sum(-1)) * 1000.0).cpu()

            errs["A  rgbd + clean depth"].append(mpj(m_rgbd(img)["keypoints"]))
            errs["A_L rgbd + LiDAR depth"].append(mpj(m_rgbd(imgL)["keypoints"]))
            errs["C  rgbd - depth zeroed"].append(mpj(m_rgbd(img0)["keypoints"]))
            errs["B  rgb_only (baseline)"].append(mpj(m_rgbo(img0)["keypoints"]))

    res = {}
    print("\n============== DEPTH ABLATION — LiDAR-sim model (HanCo held-out) ==============")
    print(f"{'condition':24s} {'MPJPE':>9s} {'PCK@20mm':>9s} {'tipMPJPE':>9s} {'wrist':>7s}")
    for k, v in errs.items():
        e = torch.cat(v, 0)
        res[k] = e.mean().item()
        print(f"{k:24s} {e.mean():7.2f}mm {100*(e<20).float().mean():7.1f}% "
              f"{e[:, [4,8,12,16,20]].mean():7.2f}mm {e[:, 0].mean():6.2f}mm")
    A = res["A  rgbd + clean depth"]; AL = res["A_L rgbd + LiDAR depth"]
    C = res["C  rgbd - depth zeroed"]; B = res["B  rgb_only (baseline)"]
    print("------------------------------------------------------------------------------")
    print(f"deployment cost    (A_L - A) = {AL - A:+.2f} mm  (coarse LiDAR depth vs clean)")
    print(f"depth RELIANCE     (C  - A) = {C - A:+.2f} mm  (degradation w/o depth; want SMALL)")
    print(f"depth CONTRIBUTION (B - A_L) = {B - AL:+.2f} mm  (LiDAR depth vs RGB-only)")
    print("==============================================================================")


if __name__ == "__main__":
    main()
