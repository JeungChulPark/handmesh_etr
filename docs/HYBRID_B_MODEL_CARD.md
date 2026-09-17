# Model Card — `hybrid_B` (locked 2D + backbone Z)

Hand-pose lifter used for live ZED inference and exported to the Unity app.
Checkpoint: `mobrecon_ckpt/hybrid_B/best.pt` (epoch 21). Date: 2026-07-04.

## 1. Summary

`hybrid_B` is a `HybridLifter` in **`locked` mode**: the 2D joint positions are
**locked to an external detector (MediaPipe / Apple Vision)** and only the depth
(Z) is produced by a learned RGB-D backbone. The delivered 3D pose is assembled
deterministically from the backbone's relative depth plus the sensor wrist depth.

| | |
|---|---|
| Class | `HybridLifter(mode="locked")` (`scripts/train/train_hybrid.py`) |
| Learned net | `LargeModel_Extra_RGBD` backbone (`models/mobrecon_ds.py`) |
| Params | ≈5.4M total (backbone **5.00M** + unused head 0.41M) |
| Input | crop `[1,4,256,256]` + detector `uv[21,2]` + `z_use[21]` + `K[4]` |
| Output | 21 joints, 3D, root-relative (metres) |
| Combined MPJPE | **27.01 mm** (DexYCB 28.2 / HO3D 26.4 / iPhone 17.1) |
| Warm-start | `rgbd_real_0613` (4-ch RGB-D backbone) |

## 2. Two-layer structure

```
input: crop[1,4,256,256] + uv[21,2] + z_use[21] + K[4]
   │
   ├── (neural) backbone LargeModel_Extra_RGBD  →  p_cnn[21,3]   (only Z is used)
   │
   └── (deterministic fusion, no learned params)
         Z0   = z_use[0]                       # sensor wrist depth (LiDAR/ZED)
         dz_j = p_cnn[j].z − p_cnn[0].z        # backbone per-joint relative depth
         Z_j  = Z0 + dz_j
         X_j  = (u_j − cx)/fx · Z_j            # u,v = detector 2D landmarks (LOCKED)
         Y_j  = (v_j − cy)/fy · Z_j
   ▼
output: p[21,3] root-relative → anchored at wrist depth for display
```

* The **only trained part is the backbone** (5.00M). The fusion is pure geometry.
* The `head` (0.41M) belongs to the `backbone` mode and is **unused in `locked`**,
  so only the backbone is exported to Unity ONNX.

## 3. Backbone — `DenseStack_Backbone_like_prev` (MobRecon family)

A DenseStack (hourglass) 2.5D regressor: 4-channel RGB-D crop → 21×3 joints.

```
crop[1,4,256,256]
 │ pre_layer   : conv s2 (4→32) + mobile_unit(32→64)   → [1,64,128,128]  (RGB-D 4-ch stem)
 │ reorg+thrink: space-to-depth conv s2 → channel squeeze → [1,64,64,64]
 │ dense_stack1: DenseStack hourglass                   (1.52M — largest block)
 │ +stack1_remap, concat, thrink2
 │ dense_stack2: DenseStack2 hourglass → spatial map + bottleneck   (1.52M)
 ├─▶ [branch A · 2.5D]  reduce(→21 maps) → uv_reg MLP(1024→128→64→3) → uv_linear → 128-d/joint
 └─▶ [branch B · latent] mid_proj(→1024) → de_layer_conv(→512) → Conv1d chain
                          (cat_conv_final, final_conv) → latent_linear → 512-d/joint
        branch A + branch B
         │ prev_linear  : residual MLP ×2 (inter-joint refinement)
         │ final_linear1: 128→64→3
         ▼
   keypoints[1,21,3]   (root-relative 3D, wrist = 0)
```

### Parameter distribution (backbone 5.00M)

| Block | Params | Role |
|---|---|---|
| `dense_stack1` | 1.52M | 1st hourglass encoder (largest) |
| `dense_stack2` | 1.52M | 2nd hourglass (spatial map + bottleneck) |
| `mid_proj` | 0.52M | bottleneck → latent 1024 |
| `de_layer_conv` | 0.52M | latent → 512 decoder entry |
| `cat_conv_final` + `final_conv` | 0.51M | per-joint Conv1d refinement |
| `reorg` | 0.15M | space-to-depth downsample |
| `uv_reg` | 0.14M | 2.5D coordinate regression branch |
| `prev_linear` | 0.07M | inter-joint residual correction |
| `pre_layer` | 0.004M | **4-ch RGB-D stem (the key difference)** |
| others (reduce / uv_linear / final_linear1 …) | ~0.05M | |

### Difference from the RGB original

Identical to `LargeModel_Extra` (RGB) except the stem's first conv input channels:
```python
conv_layer(in_chans, 32, 3, 2, 1)   # in_chans = 4  (RGB 3 + depth 1)
```
→ **early-fusion 4-channel input** (RGB + normalised metric depth, ~[−1,1],
centred by `datasets/depth_synth.py`).

## 4. Training

| | |
|---|---|
| Base | warm-start from `rgbd_real_0613/best.pt` (5.05M, backbone-only) |
| Datasets | `dexycb + ho3d + iphone` (real GT + iPhone pseudo-GT) |
| Schedule | 40 epochs, best = **epoch 21**; AdamW, backbone lr 1e-4 / head lr 1e-3, cosine |
| Loss | 3D L2 + bone-length + aux(`p_cnn` vs GT) + uv-reg. In `locked`, X,Y are fixed, so the backbone effectively **specialises in relative depth**. |
| Holdout | DexYCB 2 subj / HO3D last 1/5 seq / iPhone last capture dir |
| Result | combined **27.01 mm** (DexYCB 28.19 / HO3D 26.36 / iPhone 17.15) |

Comparison: the pure Z-lifter baseline was 37.43 mm; the sibling `hybrid_A`
(backbone-primary + reprojection loss) **diverged** (1669 mm).

## 5. Inference paths

* **Backbone only runs** — crop → `backbone` → keypoints; only Z (relative depth)
  is used, the rest is geometric fusion.
* **ZED live:** `scripts/infer/infer_zed_hybrid.py` (MediaPipe uv + ZED depth). NEURAL depth
  recommended — PERFORMANCE moved the root ~90 mm / pose ~104 mm
  (`docs/ZED_HYBRID_B_DEPLOY.md`).
* **Unity:** backbone exported by `scripts/export/export_hybrid_b_onnx.py` →
  `hybrid_B_backbone.onnx` (`image[1,4,256,256] → keypoints[1,21,3]`, opset 17,
  parity 5.6e-8 vs `HybridLifter.backbone`). Fusion runs in `HybridBHandProvider.cs`
  using Apple Vision's 21 landmarks + LiDAR depth + intrinsics
  (`docs/UNITY_HYBRID_B.md`).

## 6. I/O contract

Full model (`HybridLifter.forward`):

| Input | Shape | Meaning |
|---|---|---|
| `feat` | `[B,21,4]` | normalised uv + centred depth + validity |
| `crop` | `[B,4,256,256]` | RGB (0..1) + median-centred depth |
| `uv` | `[B,21,2]` | detector 2D landmarks (pixels) — **locks X,Y** |
| `z_use` | `[B,21]` | per-joint sensor depth (metres) |
| `K` | `[B,4]` | fx, fy, cx, cy |

Output: `p[B,21,3]` root-relative joints (+ `p_cnn`, `p_geom` auxiliaries).

Exported ONNX (backbone only): `image[1,4,256,256] → keypoints[1,21,3]`.

## 7. Files

| Path | Role |
|---|---|
| `mobrecon_ckpt/hybrid_B/best.pt` | trained checkpoint (ep21, 27.01 mm) |
| `scripts/train/train_hybrid.py` | `HybridLifter` (locked/backbone modes) |
| `models/mobrecon_ds.py` | `LargeModel_Extra_RGBD` backbone wrapper |
| `models/densestack.py` | `DenseStack_Backbone_like_prev` |
| `scripts/export/export_hybrid_b_onnx.py` | backbone → ONNX (parity-checked) |
| `scripts/infer/infer_zed_hybrid.py` | live ZED inference |
| `unity/.../Assets/Models/hybrid_B_backbone.onnx` | Unity model asset |
| `unity/.../Assets/Runtime/Recon/HybridBHandProvider.cs` | Unity locked-fusion provider |

Related: [ZED deploy](ZED_HYBRID_B_DEPLOY.md) · [Unity integration](UNITY_HYBRID_B.md)
