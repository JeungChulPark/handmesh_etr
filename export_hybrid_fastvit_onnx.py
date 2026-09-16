"""Export the FastViT hybrid lifter (backbone-primary mode) for Unity Sentis.

Unlike hybrid_B (locked mode, backbone-only export), the deploy-best FastViT model
`hybrid_sa12_replay_ft_bb_uv2d` is mode="backbone": the learned MLP head refines the
backbone's 3D with the MediaPipe/Vision 2D + sensor-depth prior. The head is part of
the model, so the WHOLE HybridLifter forward is exported as one graph:

    image [1,4,256,256]  4-ch hand crop (RGB 0..1 + median-centred depth) — same
                         tensor the dual-stream / hybrid_B Unity pipeline builds
    feat  [1,21,4]       per joint (u_norm, v_norm, z_use - z_ref, valid)
                         u_norm = u/W*2-1, v_norm = v/H*2-1 (full image);
                         z_ref = median of valid sampled depths
    uv    [1,21,2]       detector 2D landmarks, full-image pixels
    z_use [1,21]         per-joint sensor depth (metres; invalid -> z_ref)
    K     [1,4]          fx, fy, cx, cy (full-image intrinsics)
      -> keypoints [1,21,3]   root-relative metric 3D (wrist = 0)

Sentis-safety rewrites (numerically identical, verified below):
  * einsum "bjhw,bchw->bjc" -> bmm            (attention pooling)
  * BatchNorm1d (eval)      -> explicit affine (rank-2 BN is patchy in runtimes)
  * timm reparameterize_model fuses the FastViT MobileOne branches into plain convs
    (--no-reparam to skip; expect ~1e-5 drift, reported)

    python export_hybrid_fastvit_onnx.py     # -> Assets/Models/hybrid_fastvit.onnx
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fastvit_backbone import FastViTHandSoftArgmax

UNITY_MODELS = "unity/DepthRefinement/Assets/Models"
N = 21


class HybridFastVitFull(nn.Module):
    """ONNX-safe re-implementation of HybridLifter(mode='backbone', backbone='fastvit').

    Owns the same submodules (state-dict compatible modulo the backbone.->bb. prefix)
    but re-expresses einsum/BatchNorm1d with primitive ops for Sentis.
    """

    def __init__(self, variant="fastvit_sa12", h=512):
        super().__init__()
        self.bb = FastViTHandSoftArgmax(variant=variant, in_chans=4, pretrained=False)
        self.head = nn.Sequential(
            nn.Linear(N * 10, h), nn.BatchNorm1d(h), nn.ReLU(True), nn.Dropout(0.1),
            nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(True),
            nn.Linear(h, N * 3))

    def load_from(self, sd):
        new = {}
        for k, v in sd.items():
            if k.startswith("backbone."):
                new["bb." + k[len("backbone."):]] = v
            elif k.startswith("head."):
                new[k] = v
        self.load_state_dict(new, strict=True)

    @staticmethod
    def _bn(x, bn):     # eval-mode BatchNorm1d as explicit affine
        return (x - bn.running_mean) / torch.sqrt(bn.running_var + bn.eps) * bn.weight + bn.bias

    def _head(self, x):
        h = self.head
        x = torch.relu(self._bn(h[0](x), h[1]))
        x = torch.relu(self._bn(h[4](x), h[5]))
        return h[7](x)

    def _backbone(self, x):
        b = self.bb
        rgb = (x[:, :3] - b.mean) / b.std
        x = torch.cat([rgb, x[:, 3:4]], dim=1)
        hi, deep = b.encoder(x)
        deep = F.interpolate(deep, size=hi.shape[-2:], mode="bilinear", align_corners=False)
        feat = b.fuse(torch.cat([hi, deep], dim=1))
        B, Cf, H, W = feat.shape
        hm = b.hm(feat).view(B, N, H * W)
        prob = torch.softmax(hm, dim=-1)                              # [B,J,HW]
        pm = prob.view(B, N, H, W)
        gx = torch.linspace(-1, 1, W).view(1, 1, 1, W)
        gy = torch.linspace(-1, 1, H).view(1, 1, H, 1)
        u = (pm * gx).sum(dim=(2, 3))
        v = (pm * gy).sum(dim=(2, 3))
        uvn = torch.stack([u, v], dim=-1)                             # [B,J,2]
        fj = torch.bmm(prob, feat.view(B, Cf, H * W).transpose(1, 2))  # == the einsum
        return b.mlp(torch.cat([fj, uvn], dim=-1)).view(B, N, 3)

    def forward(self, image, feat, uv, z_use, K):
        p_cnn = self._backbone(image)
        fx, fy, cx, cy = K[:, 0:1], K[:, 1:2], K[:, 2:3], K[:, 3:4]
        Xg = (uv[..., 0] - cx) / fx * z_use
        Yg = (uv[..., 1] - cy) / fy * z_use
        p_geom = torch.stack([Xg, Yg, z_use], dim=-1)
        p_geom = p_geom - p_geom[:, 0:1]
        x = torch.cat([p_cnn, p_geom, feat], dim=-1)                  # [B,21,10]
        dp = self._head(x.reshape(x.shape[0], -1)).reshape(-1, N, 3)
        p = p_cnn + dp
        return p - p[:, 0:1]


def example_inputs(size, seed=0):
    g = torch.Generator().manual_seed(seed)
    image = torch.rand(1, 4, size, size, generator=g)
    image[:, 3] = image[:, 3] * 2 - 1                                 # depth channel ~[-1,1]
    uv = torch.rand(1, N, 2, generator=g) * torch.tensor([1440.0, 1920.0]) \
        + torch.tensor([200.0, 300.0])
    z_use = 0.45 + 0.15 * torch.rand(1, N, generator=g)
    z_ref = z_use.median(dim=1, keepdim=True).values
    valid = torch.ones(1, N)
    feat = torch.stack([uv[..., 0] / 1920 * 2 - 1, uv[..., 1] / 2560 * 2 - 1,
                        (z_use - z_ref), valid], dim=-1)
    K = torch.tensor([[1450.0, 1450.0, 960.0, 1280.0]])
    return image, feat, uv, z_use, K


def main(args):
    ck = torch.load(args.ckpt, map_location="cpu")
    sd = ck.get("model_state_dict", ck)

    model = HybridFastVitFull(variant=args.fastvit_variant).eval()
    model.load_from(sd)
    print(f"[load] {args.ckpt} epoch={ck.get('epoch','?')}  "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    ex = example_inputs(args.size)

    # reference: the real training-time HybridLifter on the same inputs
    with torch.no_grad():
        ref_wrap = model(*ex)
    try:
        from utils import load_cfg
        from train_hybrid import HybridLifter
        cfg = load_cfg(args.cfg)
        hl = HybridLifter(cfg, mode="backbone", backbone="fastvit",
                          fastvit_variant=args.fastvit_variant).eval()
        hl.load_state_dict(sd)
        image, feat, uv, z_use, K = ex
        with torch.no_grad():
            ref_hl = hl(feat, image, uv, z_use, K)[0]   # HybridLifter arg order: (feat, crop, ...)
        err = float((ref_wrap - ref_hl).abs().max())
        print(f"[parity] max|wrapper - HybridLifter| = {err:.2e}  "
              f"({'OK' if err < 1e-5 else 'MISMATCH'})")
        assert err < 1e-4, "wrapper does not reproduce HybridLifter"
    except ImportError as e:
        print(f"[parity] HybridLifter reference skipped ({e})")

    if not args.no_reparam:
        from timm.utils.model import reparameterize_model
        model.bb.encoder = reparameterize_model(model.bb.encoder)
        model.eval()
        with torch.no_grad():
            ref_rep = model(*ex)
        err = float((ref_rep - ref_wrap).abs().max())
        print(f"[reparam] fused MobileOne branches; max drift = {err:.2e} m "
              f"({err*1000:.4f} mm)")
        ref_wrap = ref_rep

    torch.onnx.export(model, ex, args.out,
                      input_names=["image", "feat", "uv", "z_use", "K"],
                      output_names=["keypoints"],
                      opset_version=args.opset, dynamic_axes=None)
    print(f"[onnx] wrote {args.out}")

    import onnx
    m = onnx.load(args.out)
    onnx.checker.check_model(m)
    ops = sorted({n.op_type for n in m.graph.node})
    print(f"[onnx] checker OK; {len(m.graph.node)} nodes, op types: {', '.join(ops)}")

    import onnxruntime as ort
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    onx = sess.run(["keypoints"], {k: v.numpy() for k, v in
                                   zip(["image", "feat", "uv", "z_use", "K"], ex)})[0]
    err = float(np.abs(onx - ref_wrap.numpy()).max())
    print(f"[parity] max|ONNX - torch| = {err:.2e} m ({err*1000:.4f} mm)  "
          f"({'OK' if err < 1e-4 else 'MISMATCH'})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="mobrecon_ckpt/hybrid_sa12_replay_ft_bb_uv2d/best.pt")
    ap.add_argument("--cfg", default="./configs_rgbd.yaml")
    ap.add_argument("--fastvit_variant", default="fastvit_sa12")
    ap.add_argument("--out", default=f"{UNITY_MODELS}/hybrid_fastvit.onnx")
    ap.add_argument("--size", default=256, type=int)
    ap.add_argument("--opset", default=17, type=int)
    ap.add_argument("--no-reparam", action="store_true",
                    help="skip timm reparameterize_model (bigger/slower graph)")
    main(ap.parse_args())
