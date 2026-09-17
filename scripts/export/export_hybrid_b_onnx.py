"""Export model B (hybrid_B, locked mode) for Unity Sentis.

B's ONLY learned component is the RGB-D backbone (LargeModel_Extra_RGBD): a 4-ch
256x256 hand crop -> 21 keypoints [B,21,3]. In locked mode the deployed 3D pose is
assembled DETERMINISTICALLY from that backbone plus the 2D detector + sensor depth:

    dz    = kp[:, 2] - kp[0, 2]        # backbone per-joint RELATIVE depth (wrist=0)
    Zabs  = Z0 + dz                    # Z0 = sensor wrist depth (LiDAR/ZED)
    X     = (u - cx) / fx * Zabs       # u,v = detector 2D landmarks (LOCKED)
    Y     = (v - cy) / fy * Zabs

So Unity only needs the backbone in ONNX; the locked fusion above is a few lines of
C# (see HybridBHandProvider). The backbone's 4-ch crop input is IDENTICAL to the
existing dual-stream `image` tensor (RGB 0..1 + median-centred depth), so Unity's
crop-building path is reused verbatim.

    python scripts/export/export_hybrid_b_onnx.py                       # -> Assets/Models/hybrid_B_backbone.onnx
    python scripts/export/export_hybrid_b_onnx.py --out pretrain/hybrid_B_backbone.onnx
"""

import _bootstrap  # noqa: F401  (repo-root + sibling script imports)

import argparse

import torch
import torch.nn as nn

from utils import load_cfg
from models.mobrecon_ds import LargeModel_Extra_RGBD

UNITY_MODELS = "unity/DepthRefinement/Assets/Models"


class BackboneKP(nn.Module):
    """LargeModel_Extra_RGBD wrapper: 4-ch crop -> keypoints [1,21,3] (single output)."""
    def __init__(self, cfg):
        super().__init__()
        cfg.MODEL.set_new_allowed(True); cfg.MODEL.IN_CHANS = 4
        self.backbone = LargeModel_Extra_RGBD(cfg)

    def forward(self, image):                       # image [1,4,256,256]
        return self.backbone(image)["keypoints"]    # [1,21,3]


def main(args):
    cfg = load_cfg(args.cfg)
    model = BackboneKP(cfg).eval()

    ck = torch.load(args.ckpt, map_location="cpu")
    sd = ck.get("model_state_dict", ck)
    # HybridLifter.backbone == our wrapper's .backbone, so the `backbone.*` subset of
    # the checkpoint maps 1:1 onto this wrapper's keys (NO prefix stripping).
    bb = {k: v for k, v in sd.items() if k.startswith("backbone.")}
    missing, unexpected = model.load_state_dict(bb, strict=False)
    assert not missing and not unexpected, f"key mismatch: missing={missing[:3]} unexpected={unexpected[:3]}"
    print(f"[load] {args.ckpt} epoch={ck.get('epoch','?')}  "
          f"loaded {len(bb)} backbone tensors (exact match)")

    example = torch.randn(1, 4, args.size, args.size)
    torch.onnx.export(model, example, args.out,
                      input_names=["image"], output_names=["keypoints"],
                      opset_version=args.opset, dynamic_axes=None)
    print(f"[onnx] wrote {args.out}  (image[1,4,{args.size},{args.size}] -> keypoints[1,21,3])")

    try:
        import onnx
        onnx.checker.check_model(onnx.load(args.out))
        print("[onnx] checker OK")
    except Exception as e:
        print(f"[onnx] checker warning: {e}")

    with torch.no_grad():
        ref = model(example)
    print(f"[parity] keypoints shape {tuple(ref.shape)}  "
          f"z-range[{ref[0,:,2].min():.3f},{ref[0,:,2].max():.3f}]")

    # cross-check: ONNX runtime output vs the real HybridLifter backbone on the same crop
    try:
        import onnxruntime as ort
        from train_hybrid import HybridLifter
        hl = HybridLifter(cfg, mode="locked").eval()
        hl.load_state_dict(sd)
        with torch.no_grad():
            ref_hl = hl.backbone(example)["keypoints"].numpy()
        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        onx = sess.run(["keypoints"], {"image": example.numpy()})[0]
        err = float(abs(onx - ref_hl).max())
        print(f"[parity] max|ONNX - HybridLifter.backbone| = {err:.2e}  "
              f"({'OK' if err < 1e-3 else 'MISMATCH'})")
    except ImportError as e:
        print(f"[parity] onnxruntime unavailable, skipped runtime check ({e})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="mobrecon_ckpt/hybrid_B/best.pt")
    ap.add_argument("--cfg", default="./configs/configs_rgbd.yaml")
    ap.add_argument("--out", default=f"{UNITY_MODELS}/hybrid_B_backbone.onnx")
    ap.add_argument("--size", default=256, type=int)
    ap.add_argument("--opset", default=17, type=int)
    main(ap.parse_args())
