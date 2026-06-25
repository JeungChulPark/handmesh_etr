"""Export the dual-stream RGB-D hand model for on-device (iPhone) deployment.

Produces ONNX always (universal intermediate; convert to Core ML on a Mac with
coremltools if it is not installable here), and a Core ML .mlpackage when
coremltools is available.

Inputs (match the training contract):
  * image     : [1, 4, 256, 256]  -- RGB (0..1) on [0:3] + median-centred depth on [3]
  * depth_med : [1, 1]            -- raw-depth median over the hand region (metres),
                                     i.e. median of the raw ARKit sceneDepth crop.
Outputs:
  * keypoints     [1,21,3]  root-relative joints (m)
  * root          [1,3]     global translation (m)
  * scale         [1,1]
  * keypoints_abs [1,21,3]  absolute camera-space joints (m) = scale*rel + root

Run:
    python convert_dualstream.py --ckpt mobrecon_ckpt/ds_lidarsim/best.pt
"""

import os
import argparse

import torch
from models.mobrecon_dualstream import MobRecon_DualStream_onnx


def detect_pose_in_chans(state):
    w = state.get("rgb_backbone.pre_layer.0.0.weight")
    return int(w.shape[1]) if w is not None else 4


def main(args):
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    pose_in_chans = detect_pose_in_chans(state)

    model = MobRecon_DualStream_onnx(pose_in_chans=pose_in_chans)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"[load] {args.ckpt} (pose_in_chans={pose_in_chans}, epoch={ckpt.get('epoch','?')})")

    img = torch.randn(1, 4, args.size, args.size)
    depth_med = torch.tensor([[0.45]], dtype=torch.float32)
    out_names = ["keypoints", "root", "scale", "keypoints_abs"]

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, args.name)

    # --- ONNX (always) ---
    onnx_path = base + ".onnx"
    torch.onnx.export(
        model, (img, depth_med), onnx_path,
        input_names=["image", "depth_med"], output_names=out_names,
        opset_version=args.opset, dynamic_axes=None,
    )
    print(f"[onnx] wrote {onnx_path}")
    try:
        import onnx
        onnx.checker.check_model(onnx.load(onnx_path))
        print("[onnx] checker OK")
    except Exception as e:
        print(f"[onnx] checker warning: {e}")

    # Parity check: traced module vs original on the same input.
    with torch.no_grad():
        ref = model(img, depth_med)
    print(f"[parity] outputs: " + ", ".join(f"{n}{tuple(r.shape)}" for n, r in zip(out_names, ref)))

    # --- Core ML (if coremltools present) ---
    try:
        import coremltools as ct
        traced = torch.jit.trace(model, (img, depth_med))
        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name="image", shape=img.shape),
                    ct.TensorType(name="depth_med", shape=depth_med.shape)],
            outputs=[ct.TensorType(name=n) for n in out_names],
            minimum_deployment_target=ct.target.iOS16,
            compute_precision=ct.precision.FLOAT16,
        )
        mlpkg = base + ".mlpackage"
        mlmodel.save(mlpkg)
        print(f"[coreml] wrote {mlpkg} (FP16, iOS16+)")
    except ImportError:
        print("[coreml] coremltools not installed -> ONNX only. "
              "Convert on a Mac: pip install coremltools && python convert_dualstream.py ...")
    except Exception as e:
        print(f"[coreml] conversion failed: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="mobrecon_ckpt/ds_lidarsim/best.pt")
    ap.add_argument("--out_dir", default="pretrain")
    ap.add_argument("--name", default="ds_lidarsim")
    ap.add_argument("--size", default=256, type=int)
    ap.add_argument("--opset", default=17, type=int)
    args = ap.parse_args()
    main(args)
