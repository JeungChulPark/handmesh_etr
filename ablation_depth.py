"""Depth-contribution ablation for the early-fusion RGB-D model.

Evaluates three conditions on the SAME HanCo held-out split (seed-0 95/5, the
trainer's split), so the gap isolates what the depth channel buys in the current
*early-fusion* (4-ch channel-concat) architecture, in-domain (synthetic depth):

  A) rgbd_1hr      + real depth      -> what the RGB-D model achieves WITH depth
  B) rgb_only_1hr  + depth zeroed    -> the RGB-only baseline (trained w/o depth)
  C) rgbd_1hr      + depth zeroed    -> how much the RGB-D model RELIES on depth

  depth contribution (net) = B - A     ;   depth reliance = C - A

Note: these checkpoints were trained on HanCo synthetic depth, so this measures
the architecture's in-domain depth use. Real-depth contribution needs ablating
the rgbd_real_0613 model once it converges.

    python ablation_depth.py            # ~a few min, shares the GPU with training
"""

import torch
from torch.utils.data import DataLoader, random_split, Subset

from datasets.hanco_ty import HanCo_ETRI_jitter
from models.mobrecon_ds import LargeModel_Extra_RGBD
from utils import load_cfg

CK_RGBD = "mobrecon_ckpt/rgbd_1hr/3.pt"
CK_RGBO = "mobrecon_ckpt/rgb_only_1hr/3.pt"
N = 2000


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)                         # reproduce the trainer's split
    ds = HanCo_ETRI_jitter(limit=5e6, with_depth=True, depth_source="render")
    n_train = int(len(ds) * 0.95)
    _, test_set = random_split(ds, [n_train, len(ds) - n_train])
    ds.mode = "test"
    test_set = Subset(test_set, list(range(min(N, len(test_set)))))
    dl = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=8)
    print(f"[ablation] {len(test_set)} held-out HanCo samples")

    cfg = load_cfg("configs_rgbd.yaml")

    def load(ck):
        m = LargeModel_Extra_RGBD(cfg).to(dev).eval()
        st = torch.load(ck, map_location=dev)
        m.load_state_dict(st.get("model_state_dict", st))
        return m

    m_rgbd, m_rgbo = load(CK_RGBD), load(CK_RGBO)
    errs = {"A rgbd + depth": [], "B rgb_only (baseline)": [], "C rgbd - depth zeroed": []}

    with torch.no_grad():
        for b in dl:
            img = b["image"].float().to(dev)
            k3 = b["keypoints3D"].float().to(dev)
            img0 = img.clone(); img0[:, 3:4] = 0.0
            def mpj(out):
                return (torch.sqrt(((out - k3) ** 2).sum(-1)) * 1000.0).cpu()
            errs["A rgbd + depth"].append(mpj(m_rgbd(img)["keypoints"]))
            errs["C rgbd - depth zeroed"].append(mpj(m_rgbd(img0)["keypoints"]))
            errs["B rgb_only (baseline)"].append(mpj(m_rgbo(img0)["keypoints"]))

    res = {}
    print("\n================ DEPTH ABLATION (HanCo held-out, synthetic depth) ================")
    print(f"{'condition':24s} {'MPJPE':>9s} {'PCK@20mm':>9s} {'tipMPJPE':>9s} {'wrist':>7s}")
    for k, v in errs.items():
        e = torch.cat(v, 0)
        res[k] = e.mean().item()
        print(f"{k:24s} {e.mean():7.2f}mm {100*(e<20).float().mean():7.1f}% "
              f"{e[:, [4,8,12,16,20]].mean():7.2f}mm {e[:, 0].mean():6.2f}mm")
    A = res["A rgbd + depth"]; B = res["B rgb_only (baseline)"]; C = res["C rgbd - depth zeroed"]
    print("---------------------------------------------------------------------------------")
    print(f"depth CONTRIBUTION (B - A) = {B - A:+.2f} mm  ({100*(B-A)/B:+.1f}% vs RGB-only)")
    print(f"depth RELIANCE     (C - A) = {C - A:+.2f} mm  (rgbd model degradation w/o depth)")
    print("=================================================================================")


if __name__ == "__main__":
    main()
