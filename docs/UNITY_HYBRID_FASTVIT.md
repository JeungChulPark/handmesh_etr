# Running the FastViT hybrid lifter in the Unity app

Goal: run the **deploy-best** model (`hybrid_sa12_replay_ft_bb_uv2d`, FastViT-SA12
hybrid lifter, **19.53 mm combined / 16.29 mm iPhone**) on-device in
`unity/DepthRefinement` via Sentis. Date: 2026-07-16.

Supersedes model B (`hybrid_B`, 27.01 mm — [UNITY_HYBRID_B.md](UNITY_HYBRID_B.md))
as the on-device model; B stays in the project untouched for A/B comparison.

## Why this is a different export than B

B is `locked` mode: only the backbone is learned, so B's ONNX is backbone-only and
the fusion is C#. The FastViT model is **backbone-primary** (`mode="backbone"`):

    p_cnn = FastViT-SA12(4-ch crop)                    # backbone 3D
    p_geom = backproject(detector uv, sensor z, K)     # geometric prior, root-relative
    Δp    = MLP head([p_cnn, p_geom, feat])            # learned refinement
    p     = (p_cnn + Δp) root-relativised

The head is part of the model, so `export_hybrid_fastvit_onnx.py` exports the WHOLE
`HybridLifter` forward as one graph. Sentis-safety rewrites (einsum→bmm,
BatchNorm1d→explicit affine) are bit-exact vs the training model (0.00e+00), timm
reparameterisation drift 0.0004 mm, ONNX-runtime parity 0.0013 mm.

## ONNX contract — `Assets/Models/hybrid_fastvit.onnx` (46 MB, opset 17)

| Input | Shape | Meaning |
|---|---|---|
| `image` | `[1,4,256,256]` | RGB 0..1 + median-centred depth — the EXACT tensor `DualStreamHandProvider` already builds |
| `feat`  | `[1,21,4]` | per joint `(u01*2-1, v01*2-1, z_use - z_ref, valid)`; `z_ref` = median of valid sampled depths |
| `uv`    | `[1,21,2]` | detector landmarks in full-image **pixels** |
| `z_use` | `[1,21]` | per-joint sensor depth (metres; invalid → `z_ref`) |
| `K`     | `[1,4]` | `fx, fy, cx, cy` full-image intrinsics |

Output: `keypoints [1,21,3]` — **root-relative** metric 3D (wrist = 0).
`HybridFastVitHandProvider` anchors it at the wrist landmark back-projected to the
sensor wrist depth (same anchoring as `infer_hybrid.py`).

## What was added (all additive, non-breaking)

| Path | Change |
|---|---|
| `export_hybrid_fastvit_onnx.py` | **new** — full-model export + 3-way parity check |
| `Assets/Models/hybrid_fastvit.onnx` | **new** — verified export of the deploy-best ckpt (epoch 31) |
| `Assets/Runtime/Recon/HybridFastVitHandProvider.cs` | **new** — 5-input Sentis provider; reuses the crop-only `DualStreamHandProvider` hooks + `HandBboxBridge` landmarks (same wiring as B) |
| `HandPoseModeController.cs` | `_fastvit` field — disabled in Server mode like the other on-device providers |
| `HandSphereDriver.cs` | `_fastvitProvider` source, priority **server > fastvit > B > dual-stream** |
| `SimpleHandGrab.cs` | `_fastvitProvider` pose gate (same priority) |
| `Grab/HandGrabController.cs` | `_fastvit` source + OnPose subscription, priority server > fastvit > B > dual |

## Scene wiring (editor)

1. Add a `HybridFastVitHandProvider` component; assign `hybrid_fastvit.onnx`.
2. Point `_crop` at the crop-only `DualStreamHandProvider` (ModelAsset left empty)
   and `_landmarks` at the `HandBboxBridge` — identical to the B wiring
   ([UNITY_HYBRID_B.md](UNITY_HYBRID_B.md) hooks A/B must already be in place).
3. To make it the active on-device model: disable the `HybridBHandProvider`
   component in the scene (the mode controller restores scene-authored states, and
   `HandSphereDriver` prefers fastvit over B anyway while both are enabled — but
   don't pay for two GPU inferences).
4. `HandSphereDriver._fastvitProvider` / `SimpleHandGrab._fastvitProvider` — assign,
   or leave empty and keep driving from B.

## Caveats

* **Landmark order** — trained on MediaPipe 21-point order; the Apple-Vision →
  MediaPipe remap caveat from B applies verbatim (the head consumes uv directly).
* **Per-joint depth quality** — unlike B (wrist-only), this model consumes ALL 21
  sampled depths (`z_use`, and `feat[:,2]`). `TrySampleDepthMetres` is a nearest
  sample; the Python path samples a 7×7 median of valid pixels (`sample_depth`,
  `infer_hybrid.py`). ARKit LiDAR is dense/smooth so nearest is usually fine, but if
  fingertips flicker, port the median window into `TrySampleDepthMetres` first.
* **Invalid depth** — joints with no depth get `z_use = z_ref` and `valid = 0`,
  matching training; the provider needs ≥1 valid sample per frame to emit a pose.
* **Model size / latency** — 11.6M params vs B's 5.0M. FastViT is reparameterised
  (pure conv + one attention stage) and Core-ML-friendly, but verify frame time on
  device; if too slow, keep B as the fallback (both coexist in the scene).
* Cannot be compiled/tested outside Unity here — the C# is written against the
  project's confirmed Sentis 2.x idioms (same API surface as the working
  `HybridBHandProvider`); compile + wire in the editor.

## Reproduce the export

```bash
python export_hybrid_fastvit_onnx.py \
    --ckpt mobrecon_ckpt/hybrid_sa12_replay_ft_bb_uv2d/best.pt \
    --out unity/DepthRefinement/Assets/Models/hybrid_fastvit.onnx
# [parity] max|wrapper - HybridLifter| = 0.00e+00  (OK)
# [reparam] fused MobileOne branches; max drift = 4.15e-07 m (0.0004 mm)
# [parity] max|ONNX - torch| = 1.31e-06 m (0.0013 mm)  (OK)
```
