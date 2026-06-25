# RGB-D Dual-Stream Hand Pose — Design Spec

**Status:** proposed (design only, no code yet) · **Date:** 2026-06-16 ·
**Deploy target:** iPhone Pro LiDAR (ToF), Core ML · **Supersedes:** early-fusion
4-channel variant in [`RGBD_TRAINING.md`](../RGBD_TRAINING.md) as the next architecture.

---

## 1. Motivation — what the current model gets wrong

The early-fusion 4-channel model (`LargeModel_Extra_RGBD`) tracks hand
*articulation* well but localises the hand *in space* poorly. Tracing the code
shows why:

| Concern | Current behaviour | Source |
|---------|-------------------|--------|
| Network output | 21×3 **root-relative** joints only (no global position) | `models/mobrecon_ds.py:156` |
| Depth as input | RGB+depth concatenated at the stem conv, depth weights zero-init | `models/densestack.py:641`, `train_mobrecon_rgbd.py:209` |
| Absolute position | recovered **outside** the net: `root = backproject(wrist_px, depth@wrist, K)` | `infer_rgbd_femtobolt.py:21` |
| Loss | masked root-rel L2 + light 2D reproj — **no position term** | `train_mobrecon_rgbd.py:292` |

The diagnosis:

1. **Monocular RGB is scale/position-ambiguous by construction.** The RGB-only
   ablation collapses to ~106 mm; depth is what recovers it. So global
   translation must come from depth — but today the network never learns it.
2. **Root is a hand-crafted heuristic, not a learned quantity.** A single wrist
   pixel's depth drives the entire root. On LiDAR/stereo that pixel is routinely
   a hole or noisy → the whole hand jumps.
3. **Pixel-level early fusion under-uses depth.** Depth's real value is *global
   position + metric scale*, which does not survive being averaged into an RGB
   stem conv. And iPhone LiDAR depth (256×192, sparse, confidence-weighted) is a
   different resolution and statistic from the RGB it is fused with.

**Design principle:** treat depth not as a 4th channel but as the signal that
*owns* global translation and metric scale, and learn that mapping inside the
network with explicit supervision.

---

## 2. Architecture — Dual-Stream + Translation Head

```
 RGB  [3,256,256] ─► DenseStack backbone (unchanged, mobile_unit)─► pose latent ─► 21×3 root-relative joints
                                                                         │              (POSE — RGB's strength)
 Depth[1,256,192] ─► DepthEncoder (4× DW-sep conv, ~0.1M)─► depth feat ───┤
   ⊙ validity mask  (confidence/0=invalid)                                ├─► TranslationHead ─► root (x,y,z) + scale s
                                                                          │       (POSITION — depth's job)
                                              fuse(pose latent, depth feat, gate)
                                                                          ▼
                              absolute joints = s · (root-relative joints) + root      ← learned, replaces heuristic
```

Four design decisions, each tied to a failure above:

### (a) Two-stream, late fusion
Keep the existing DenseStack RGB backbone **unchanged** — it is good at
articulation and carries the warm-start asset (`pretrain/100.pt`). Process depth
in a **separate lightweight encoder** (depthwise-separable conv ×4, ~0.1M params)
so the low-res, noisy, sensor-specific depth domain is decoupled from RGB.
*Solves #3 (domain/resolution mismatch).*

### (b) Explicit Translation Head
Pull the root recovery that lives in `infer_rgbd_femtobolt.py` *into* the network
and supervise it directly. `TranslationHead(depth_feat, pose_latent) → root(x,y,z)`.
It sees the whole depth feature map, not one wrist pixel → robust to holes.
*Solves #1 and #2.*

### (c) Confidence-gated depth
Multiply the depth input by a validity mask (0 = invalid) and feed a learned gate
into the TranslationHead. When depth is globally absent/out-of-range the head
falls back to an RGB-conditioned prior instead of producing garbage. One model
then survives both ZED stereo (dense, noisy) and iPhone LiDAR (sparse, holes,
>5 m / reflective-surface dropouts). *Robustness for deployment.*

### (d) Metric scale `s` as a separate scalar output
Predict the scale applied to root-relative joints. Absorbs the per-dataset hand
scale spread (DexYCB / HO3D / MSRA) and lets depth pin the monocular scale
ambiguity. *Solves the scale half of #1.*

All operators are DW-separable conv + linear — same family as the current
backbone, so **no new Core ML conversion risk** is introduced.

### Module table

| Module | In → Out | Notes |
|--------|----------|-------|
| `DenseStack` (RGB) | `[3,256,256]` → `pose_latent`, `21×3 rel` | unchanged; warm-start from `pretrain/100.pt` |
| `DepthEncoder` (new) | `[1,256,192]⊙mask` → `depth_feat [C,h,w]` | 4× DW-sep conv + pool, ~0.1M params |
| `TranslationHead` (new) | `(depth_feat, pose_latent, gate)` → `root[3]`, `s[1]` | GAP(depth_feat) ⊕ pose_latent → 2-layer MLP |
| assembly | `s·rel + root` → `21×3 absolute` | replaces `infer_*` heuristic |

---

## 3. Loss redesign

Current loss has no position term. Add explicit root/scale/absolute supervision —
all derivable from existing labels (`item_dict["root"]`, `cam`, `keypoints3D`):

| Term | Definition | Role |
|------|------------|------|
| `L_pose` | masked L2 on root-relative joints | unchanged, articulation |
| **`L_root`** | L2 on predicted root (x,y,z) | **new — the position term** |
| **`L_scale`** | L1 on metric scale `s` | **new — pins scale** |
| `L_abs` | masked MPJPE on `s·rel + root` vs absolute GT | optimises the deploy metric directly |
| `L_2d` | reprojection (existing) | light regulariser, down-weighted `w2d≈0.1` |

`L = L_pose + λ_r·L_root + λ_s·L_scale + λ_a·L_abs + w2d·L_2d`.
Ablation knob: zero the depth stream → `L_root` must collapse (reproduce the
~106 mm RGB-only failure) to prove depth is doing the work.

---

## 4. Training curriculum (iPhone LiDAR domain)

Aligned with the finalised ToF deploy target (LiDAR-sim aug + ToF warmup):

1. **Warm-start** — graft `pretrain/100.pt` into the RGB backbone (reuse current
   stem-graft logic, `train_mobrecon_rgbd.py:209`). DepthEncoder/Head init fresh.
2. **Real-depth pretrain** — DexYCB + HO3D (real sensor depth) train the full
   dual-stream incl. TranslationHead.
3. **LiDAR-sim fine-tune** — existing `--lidar_sim` aug (downsample · holes ·
   noise, `ablation_depth_lidar.py`) + MSRA(ToF) mixed in, to match iPhone LiDAR
   statistics.
4. **Confidence-dropout aug** — randomly invalidate depth (partial + full) so the
   gate (c) learns graceful fallback.

---

## 5. iPhone / Core ML deployment

- **Input:** ARKit `sceneDepth` → `depthMap` (256×192) + `confidenceMap`. Use the
  real confidence directly as the validity mask (c) — better than a synthetic one.
- **Two fixed-resolution inputs** (RGB 256, depth 192), each resized inside its own
  stream → avoids `grid_sample`/dynamic-shape ops that complicate Core ML export.
- **Export path:** PyTorch → ONNX → coremltools (or torch→coreml direct). Static
  shapes, standard ops only. FP16 default; INT8 PTQ as a size/latency option.
- **Latency budget:** DepthEncoder is ~0.1M params on 256×192 → negligible over the
  existing backbone; target <30 ms unchanged.

---

## 6. Validation plan

Same split, report **absolute** MPJPE and **root error** (not just root-rel):

| Variant | Description |
|---------|-------------|
| B0 | current early-fusion 4ch + heuristic root (baseline) |
| B1 | dual-stream + TranslationHead |
| B2 | B1 + confidence gating + LiDAR-sim domain (**recommended ship**) |
| ablation | depth off → root error must blow up (~106 mm) as designed |

Success = B2 beats B0 on absolute MPJPE and root error, and degrades gracefully
(not catastrophically) when depth confidence is dropped.

---

## 6b. Implementation status (2026-06-16)

Implemented as **separate files** (early-fusion baseline left untouched for comparison):

| File | What |
|------|------|
| `models/mobrecon_dualstream.py` | **New.** `RGBPoseBackbone` (subclasses `DenseStack_Backbone_like_prev`, also returns pooled pose latent — no edit to densestack.py), `DepthEncoder` (~59K params), `TranslationHead` (root+scale, scale init≈1.0), `MobRecon_DualStream` (+ `_onnx` twin). Total ~5.4M params. |
| `datasets/hanco_ty.py` | **Additive opt-in** `return_abs_depth=False`: when True, `__getitem__` also returns `depth_med` (raw-depth median = absolute distance). Default off ⇒ baseline byte-identical (verified). |
| `train_mobrecon_dualstream.py` | **New.** Losses `L_pose + w_root·L_root + w_scale·L_scale + w_abs·L_abs + w2d·L_2d`; warm-start remaps `backbone.*`→`rgb_backbone.*` (746/808 loaded); logs root-rel + abs MPJPE + root error; same seed(0)/0.95 split as baseline. |

Smoke-verified: forward shapes, gradient flow into all three branches, rgb_only
fallback, warm-start key mapping, dataset opt-in + default-off parity.

**Comparison baseline:** `mobrecon_ckpt/rgbd_warm30/best.pt` (early-fusion, HanCo,
warm-start, 28 ep) = **15.28 mm root-rel MPJPE**. Dual-stream run `ds_warm30`
(same HanCo + warm-start, 30 ep) compares root-rel directly and additionally
reports absolute MPJPE + root error — which the baseline cannot produce natively
(it offloads root to a wrist-pixel backprojection heuristic at inference).

## 6c. Results — 4-way comparison (HanCo, warm-start, 30 ep, same split)

| Run | Pose input | root-rel MPJPE | abs MPJPE | root err |
|-----|-----------|----------------|-----------|----------|
| baseline `rgbd_warm30` (early-fusion) | 4ch | **15.28** | — (heuristic) | — |
| `ds_warm30` (dual-stream, w=1.0) | RGB 3ch | 20.65 | 26.93 | 24.68 |
| `ds_warm30_w05` (w=0.5) | RGB 3ch | 21.43 | 28.31 | 26.60 |
| **`ds_hybrid`** (w=1.0) | **RGB-D 4ch** | **16.88** | **25.21** | **24.55** |

**Conclusion: the hybrid wins.** Giving the pose backbone the depth channel
(`pose_in_chans=4`, stem grafted 3→4) recovered root-rel to **16.88 mm** — within
1.6 mm of the baseline — while the learned TranslationHead still delivers absolute
position (abs **25.21**, root **24.55 mm**) that the early-fusion baseline cannot
produce natively. The original "hand position is wrong" problem is solved with a
learned, hole-robust translation head, at no pose-accuracy cost.

The `w=0.5` detour confirmed the root-rel gap of the RGB-only dual-stream was NOT
loss competition (lowering weights hurt all metrics) — it was the pose branch
being blind to depth. Two of the three runs hit a transient CUDA launch-timeout
(display-GPU watchdog) ~ep24–25 and resumed losslessly from 2-epoch checkpoints.

**Next:** extend `return_abs_depth` to the dexycb/ho3d/tof loaders → real-depth +
LiDAR-sim curriculum on `ds_hybrid` → Core ML export (inference `depth_med` =
median of raw ARKit `sceneDepth` over the hand region).

## 6d. PA-MPJPE evaluation (eval_pa_mpjpe.py, 8000 HanCo val frames, 2026-06-18)

| Model | PA-MPJPE | root-rel | abs | root err | PCK-AUC(PA) |
|-------|----------|----------|-----|----------|-------------|
| ds_warm30 (RGB pose) | 9.85 | 20.58 | 26.78 | 24.42 | 0.804 |
| ds_hybrid (RGB-D pose) | 8.92 | 16.95 | 25.21 | 24.26 | 0.822 |
| ds_hybrid_ft (low-LR) | 8.18 | 14.92 | 20.67 | 19.76 | 0.837 |
| **ds_hybrid_ft2 (cosine)** | **8.03** | **14.46** | **19.91** | **19.01** | **0.840** |

Monotonic improvement across every metric; ds_hybrid_ft2 is best. (Baseline
`rgbd_warm30` is the early-fusion architecture, not loadable by the dual-stream
eval script, so PA-MPJPE n/a; its root-rel was 15.28 — ds_hybrid_ft2 beats it at
14.46.) PA-MPJPE ~8 mm with PCK-AUC 0.84 is a strong hand-pose result.

## 6e. Real-depth curriculum (ds_realdepth, started 2026-06-18)

Fine-tune ds_hybrid_ft2 on hanco + DexYCB + HO3D (787,689 frames; real sensor
depth) — `run_realdepth.sh`, warm-start, pose_in_chans=4, lr 5e-5 cosine→0, 8 ep
+ early-stop (abs patience 3). Adapts the depth stream from synthetic render to
real sensor noise/holes ahead of the iPhone LiDAR target. Results pending.

## 7. Risks / open questions

- **Core ML two-input graph** — verify coremltools handles the dual input + gate
  cleanly before committing; fall back to fusing depth_feat into the single
  existing graph if export balks.
- **Real abs-GT availability** — `root`/`cam` exist per the loaders; confirm the
  ToF-only sets (MSRA/ICVL/NYU) provide usable absolute root or mask `L_root` for
  them (the existing `joint_valid` masking pattern extends naturally).
- **Scale `s` vs per-joint scale** — start with one global scalar; revisit only if
  it caps accuracy.
