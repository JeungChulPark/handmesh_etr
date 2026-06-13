"""Integration smoke test for the multi-dataset RGB-D training pipeline.

Builds a small ConcatDataset over the *verified, on-disk* real-depth sets
(DexYCB + HO3D + MSRA), pushes mixed batches through the early-fusion 4-channel
model with the masked loss, and runs a few optimiser steps. This validates the
whole integration end-to-end (mixed-batch collation, joint_valid masking, warm
3->4 stem graft, forward/backward) before committing to a long run.

    python smoke_rgbd.py            # ~1-2 min, GPU
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from datasets.dexycb import DexYCB_RGBD
from datasets.ho3d import HO3D_RGBD
from datasets.tof_depth import MSRA_RGBD
from train_mobrecon_rgbd import (WithJointValid, masked_l2, masked_mpjpe_mm,
                                 cam2pixel_torch)
from models.mobrecon_ds import LargeModel_Extra_RGBD
from utils import load_cfg

HD = "/home/jucpark/DeepLearning/Datasets/Hand Dataset"
ROOTS = {
    "dexycb": f"{HD}/DexYCB_full/data",
    "ho3d":   f"{HD}/HO3D_full",
    "msra":   f"{HD}/cvpr15_MSRAHandGestureDB",
}
LIMIT = 200


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # --- build the mixed real-depth dataset ---
    parts = [
        WithJointValid(DexYCB_RGBD(root=ROOTS["dexycb"], mode="train", with_depth=True, limit=LIMIT)),
        WithJointValid(HO3D_RGBD(root=ROOTS["ho3d"], mode="train", with_depth=True, limit=LIMIT)),
        WithJointValid(MSRA_RGBD(root=ROOTS["msra"], mode="train", with_depth=True, limit=LIMIT)),
    ]
    ds = ConcatDataset(parts)
    dl = DataLoader(ds, batch_size=6, shuffle=True, num_workers=4, drop_last=True)
    print(f"[smoke] ConcatDataset: {[len(p) for p in parts]} -> {len(ds)} frames")

    # --- check mixed-batch collation (the integration risk) ---
    batch = next(iter(dl))
    print("[smoke] batch keys/shapes:", {k: tuple(v.shape) for k, v in batch.items()})
    assert batch["image"].shape[1:] == (4, 256, 256)
    assert batch["joint_valid"].shape[1] == 21
    print(f"[smoke] joint_valid mean = {batch['joint_valid'].mean():.3f} (1.0 = all 21 supervised)")

    # --- model + warm 3->4 stem graft (mirrors train_mobrecon_rgbd) ---
    cfg = load_cfg("configs_rgbd.yaml")
    cfg.MODEL.set_new_allowed(True)
    cfg.MODEL.IN_CHANS = 4
    model = LargeModel_Extra_RGBD(cfg)
    ckpt = torch.load("pretrain/100.pt", map_location="cpu")
    src = ckpt.get("model_state_dict", ckpt)
    tgt = model.state_dict()
    stem = "backbone.pre_layer.0.0.weight"
    adapted = {}
    for k, v in src.items():
        if k == stem and k in tgt and v.shape != tgt[k].shape:
            w = tgt[k].clone(); w[:, :3] = v; w[:, 3:] = 0.0; adapted[k] = w
        elif k in tgt and v.shape == tgt[k].shape:
            adapted[k] = v
    model.load_state_dict(adapted, strict=False)
    print(f"[smoke] warm-start: loaded {len(adapted)}/{len(tgt)} tensors (stem 3->4 grafted)")
    model.to(dev).train()

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    it = iter(dl)
    for step in range(5):
        try:
            b = next(it)
        except StopIteration:
            it = iter(dl); b = next(it)
        img = b["image"].float().to(dev)
        k3 = b["keypoints3D"].float().to(dev)
        xy = b["keypoints2D"].float().to(dev)
        v = b["joint_valid"].float().to(dev)
        out = model(img)
        loss_3d = masked_l2(out["keypoints"], k3, v, dim=3)
        pred_xy = cam2pixel_torch(out["keypoints"] + b["root"][:, None, :].float().to(dev),
                                  b["cam"].float().to(dev)) / 256.0
        loss_2d = masked_l2(pred_xy, xy, v, dim=2)
        loss = loss_3d + loss_2d
        opt.zero_grad(); loss.backward(); opt.step()
        print(f"[smoke] step {step}: loss={loss.item():.4f} "
              f"(3d={loss_3d.item():.4f} 2d={loss_2d.item():.4f}) "
              f"mpjpe={float(masked_mpjpe_mm(out['keypoints'], k3, v)):.1f}mm")

    print("[smoke] OK: mixed-dataset RGB-D training pipeline runs end-to-end.")


if __name__ == "__main__":
    main()
